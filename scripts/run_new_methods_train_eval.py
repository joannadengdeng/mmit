"""New methods train → eval: DoRA, RandLoRA, MoLE, ReFT.

Same setup as previous experiments: 1000 train, 200 eval/benchmark.
Runs each method sequentially, outputs comparison table at the end.

Usage:
    !cd /content/mmit && git pull && \
     PYTHONPATH=src python scripts/run_new_methods_train_eval.py

Options:
    --methods dora,randlora    # run specific methods only
    --lavender                 # add Lavender attention alignment loss
    --train-samples 1000       # number of training samples
    --eval-samples 200         # per-benchmark eval samples
"""
import sys, os, time, math, gc, json, traceback, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

P = "\033[92m✓\033[0m"
F = "\033[91m✗\033[0m"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
TRAIN_DATASET = "HuggingFaceH4/llava-instruct-mix-vsft"
OUTPUT_ROOT = "/content/mmit_output"
FORWARD_KEYS = {
    "input_ids", "attention_mask", "labels",
    "pixel_values", "image_sizes", "image_grid_thw",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        return False
    def fileno(self):
        return self.streams[0].fileno()


# ── Model loading ──

def load_model(quantize_4bit=False):
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if quantize_4bit:
        model = AutoVLM.from_pretrained(
            MODEL_ID,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            ),
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoVLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
    mode = "4-bit" if quantize_4bit else "bf16"
    print(f"   Model loaded ({mode}), GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    return model, processor


# ── Data loading ──

def load_and_preprocess(processor, max_samples):
    from mmit.data.adapters.hf_datasets import HFDatasetsAdapter
    from mmit.training.preprocessors import ChatTemplatePreprocessor

    adapter = HFDatasetsAdapter(dataset_name=TRAIN_DATASET, split="train")
    samples = []
    for i, s in enumerate(adapter):
        samples.append(s)
        if max_samples and i >= max_samples - 1:
            break

    preproc = ChatTemplatePreprocessor()
    processed = []
    for i, s in enumerate(samples):
        try:
            processed.append(preproc.tokenize(s, processor, max_length=2048))
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            print(f"   Preprocessed {i+1}/{len(samples)}...")
    print(f"   {P} {len(processed)} train samples ready")
    return processed, preproc


# ── Method setup ──

def setup_method(method_name, model, processor, use_lavender=False):
    """Setup method + loss function. Returns (model, method, loss_fn, info)."""
    from mmit.training.losses import CrossEntropyLoss, CEPlusOrthoLoss

    if method_name == "dora":
        from mmit.training.methods.dora import DoRAMethod
        m = DoRAMethod()
        c = {**m.default_config(), "lora_r": 8, "lora_alpha": 16,
             "lora_dropout": 0.05, "freeze_patterns": ["vision_tower"]}
        model, info = m.prepare_model(model, processor, c)
        loss_fn = CrossEntropyLoss()

    elif method_name == "randlora":
        from mmit.training.methods.randlora import RandLoRAMethod
        m = RandLoRAMethod()
        c = {**m.default_config(), "lora_r": 8, "lora_alpha": 16,
             "lora_dropout": 0.05, "freeze_patterns": ["vision_tower"]}
        model, info = m.prepare_model(model, processor, c)
        loss_fn = CrossEntropyLoss()

    elif method_name == "mole":
        from mmit.training.methods.mole import MoLEMethod
        m = MoLEMethod()
        c = {**m.default_config(), "num_experts": 4, "lora_r": 8, "lora_alpha": 16,
             "freeze_patterns": ["vision_tower", "vision_model"]}
        model, info = m.prepare_model(model, processor, c)
        # MoLE has its own compute_loss with balance loss
        class _MoLELoss:
            def compute(self, model, batch, outputs):
                return m.compute_loss(model, batch, outputs)
            def get_trainable_params(self):
                return []
        loss_fn = _MoLELoss()

    elif method_name == "reft":
        from mmit.training.methods.reft import ReFTMethod
        m = ReFTMethod()
        c = {**m.default_config(), "intervention_rank": 4, "positions": "f4+l5",
             "share_weights": False, "steer_visual_only": False, "dropout": 0.0}
        model, info = m.prepare_model(model, processor, c)
        # ReFT uses its own compute_loss for ortho loss
        class _ReFTLoss:
            def compute(self, model, batch, outputs):
                return m.compute_loss(model, batch, outputs)
            def get_trainable_params(self):
                return []
        loss_fn = _ReFTLoss()

    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Wrap with Lavender if requested
    if use_lavender:
        from mmit.training.losses.lavender_loss import LavenderLoss
        base_loss = loss_fn
        lavender = LavenderLoss(sd_xattn_loss_scale=10.0, sd_attn_dir="")
        lavender.on_prepare(model, {})
        info += "\n+ Lavender attention alignment loss"
        # Lavender falls back to CE-only if no SD attention data available
        loss_fn = lavender

    return model, m, loss_fn, info


# ── Training ──

def train_method(model, method, loss_fn, processed, preproc, lr,
                 num_epochs=1, batch_size=1, grad_accum=8):
    model.gradient_checkpointing_enable()
    model.train()
    device = next(model.parameters()).device

    params = method.get_trainable_params(model)
    trainable_count = sum(p.numel() for pg in params for p in pg["params"])

    # Add Lavender aligner params if available
    if hasattr(loss_fn, 'get_trainable_params'):
        lav_params = loss_fn.get_trainable_params()
        if lav_params:
            params.append({"params": lav_params, "lr": lr})

    for pg in params:
        pg.setdefault("lr", lr)
    optimizer = AdamW(params, weight_decay=0.0)

    loader = DataLoader(processed, batch_size=batch_size, shuffle=True,
                        collate_fn=preproc.collate, drop_last=True)
    steps_per_epoch = len(loader) // grad_accum
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.03)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    log_interval = max(1, total_steps // 10)

    print(f"   Steps: {total_steps}, Params: {trainable_count:,}, LR: {lr}")
    global_step, total_loss = 0, 0.0
    t0 = time.time()

    for epoch in range(num_epochs):
        for step, batch in enumerate(loader):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items() if k in FORWARD_KEYS}
            batch_gpu["labels"] = method.preprocess_labels(
                batch_gpu["input_ids"], batch_gpu["labels"],
            )

            outputs = model(**batch_gpu)
            loss, metrics = loss_fn.compute(model, batch_gpu, outputs)

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in params for p in pg["params"] if p.grad is not None], 1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                total_loss += loss.item() * grad_accum

                if global_step % log_interval == 0 or global_step == 1:
                    avg = total_loss / global_step
                    extra = " ".join(f"{k}={v:.4f}" for k, v in metrics.items()) if metrics else ""
                    elapsed = time.time() - t0
                    eta = elapsed / global_step * (total_steps - global_step)
                    print(f"   [{global_step}/{total_steps}] loss={loss.item()*grad_accum:.4f} avg={avg:.4f} {extra} eta={eta:.0f}s")

    avg_loss = total_loss / max(1, global_step)
    train_time = time.time() - t0
    print(f"   {P} {global_step} steps, avg_loss={avg_loss:.4f}, params={trainable_count:,}, time={train_time:.0f}s")
    return avg_loss, train_time, trainable_count


# ── Evaluation ──

def evaluate(model, processor, max_samples):
    from mmit.eval.methods.local_method import LocalMethod
    from mmit.data.types import EvalSample
    from datasets import load_dataset

    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    eval_method = LocalMethod(model, processor)
    results = {}

    for bench_name, hf_id, split in [
        ("VQAv2", "lmms-lab/VQAv2", "validation"),
        ("POPE", "lmms-lab/POPE", "test"),
        ("TextVQA", "lmms-lab/textvqa", "validation"),
    ]:
        print(f"   --- {bench_name} ---")
        try:
            ds = load_dataset(hf_id, split=split, streaming=True)
            eval_samples = []
            for j, row in enumerate(ds):
                if max_samples and j >= max_samples:
                    break
                eval_samples.append(row)

            correct, total = 0, 0
            eval_t0 = time.time()

            for j, s in enumerate(eval_samples):
                question = s.get("question", "Describe this image.")
                image = s.get("image")
                answers = s.get("answers", s.get("answer", ""))
                if isinstance(answers, list):
                    if answers and isinstance(answers[0], dict):
                        gt = answers[0].get("answer", str(answers[0]))
                    else:
                        gt = str(answers[0]) if answers else ""
                else:
                    gt = str(answers)

                if bench_name == "POPE":
                    prompt = question + " Please answer yes or no."
                else:
                    prompt = question + " Answer the question using a single word or phrase."

                metadata = {"_pil_image": image} if image else {}
                es = EvalSample(
                    id=str(j), image_path="<in_memory>" if image else "",
                    question=prompt, ground_truth=gt, metadata=metadata,
                )

                prepared = eval_method.prepare_eval_input(es)
                pred = eval_method.generate(prepared, max_new_tokens=32)

                pred_c = pred.strip().lower().rstrip(".")
                gt_c = gt.strip().lower().rstrip(".")
                if pred_c == gt_c or gt_c in pred_c.split():
                    correct += 1
                total += 1

                if (j + 1) % 100 == 0:
                    elapsed = time.time() - eval_t0
                    eta = elapsed / (j + 1) * (len(eval_samples) - j - 1)
                    print(f"   {bench_name}: {j+1}/{len(eval_samples)} ({correct}/{total}, eta={eta:.0f}s)")

            acc = correct / max(1, total) * 100
            eval_time = time.time() - eval_t0
            results[bench_name] = {"accuracy": round(acc, 2), "correct": correct, "total": total}
            print(f"   {P} {bench_name}: {acc:.1f}% ({correct}/{total}) in {eval_time:.0f}s")

        except Exception as e:
            print(f"   {F} {bench_name}: {e}")
            traceback.print_exc()

    return results


# ── Method configs ──

METHOD_CONFIGS = {
    "dora":     {"quantize": True,  "lr": 2e-4},
    "randlora": {"quantize": True,  "lr": 2e-4},
    "mole":     {"quantize": False, "lr": 2e-4},
    "reft":     {"quantize": False, "lr": 1e-3},
}

ALL_METHODS = list(METHOD_CONFIGS.keys())


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Train + eval new PEFT methods")
    parser.add_argument("--methods", type=str, default=",".join(ALL_METHODS),
                        help=f"Comma-separated methods to run (default: all)")
    parser.add_argument("--lavender", action="store_true",
                        help="Add Lavender attention alignment loss")
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    methods_to_run = [m.strip() for m in args.methods.split(",")]
    for m in methods_to_run:
        if m not in METHOD_CONFIGS:
            print(f"{F} Unknown method: {m}. Available: {ALL_METHODS}")
            sys.exit(1)

    t0 = time.time()
    lav_tag = " + Lavender" if args.lavender else ""
    print(f"New Methods Train → Eval{lav_tag}")
    print(f"Model: {MODEL_ID}")
    print(f"Methods: {methods_to_run}")
    print(f"Train: {args.train_samples} | Eval: {args.eval_samples}/benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    import mmit

    # Load data once
    print("Loading data...")
    tmp_model, processor = load_model(quantize_4bit=True)
    processed, preproc = load_and_preprocess(processor, max_samples=args.train_samples)
    del tmp_model
    gc.collect()
    torch.cuda.empty_cache()

    all_results = {}

    for method_name in methods_to_run:
        mcfg = METHOD_CONFIGS[method_name]
        print(f"\n{'='*60}")
        print(f"  {method_name.upper()}")
        print(f"{'='*60}")

        try:
            # Fresh model
            gc.collect()
            torch.cuda.empty_cache()
            model, processor = load_model(quantize_4bit=mcfg["quantize"])

            # Setup
            model, method, loss_fn, info = setup_method(
                method_name, model, processor, use_lavender=args.lavender,
            )
            print(f"   {info}")

            # Train
            print("   Training...")
            avg_loss, train_time, param_count = train_method(
                model, method, loss_fn, processed, preproc, mcfg["lr"],
                num_epochs=args.epochs, batch_size=args.batch_size,
                grad_accum=args.grad_accum,
            )

            # Save checkpoint
            out_dir = os.path.join(OUTPUT_ROOT, f"{method_name}_experiment", "final")
            os.makedirs(out_dir, exist_ok=True)
            method.save_checkpoint(model, processor, out_dir, {
                "base_model": MODEL_ID, "method": method_name,
                "num_samples": len(processed), "avg_loss": round(avg_loss, 6),
            })

            # Eval
            print("   Evaluating...")
            eval_results = evaluate(model, processor, max_samples=args.eval_samples)

            all_results[method_name] = {
                "avg_loss": round(avg_loss, 4),
                "train_time_s": round(train_time),
                "params": param_count,
                "eval": eval_results,
            }
            del model
            print(f"   {P} {method_name} done")

        except Exception as e:
            print(f"   {F} {method_name} FAILED: {e}")
            traceback.print_exc()
            all_results[method_name] = {"error": str(e)}
            gc.collect()
            torch.cuda.empty_cache()

    # ── Summary ──
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  RESULTS ({total_time/60:.0f} min)")
    print(f"{'='*70}")
    print(f"{'Method':<10} {'Params':>10} {'Loss':>8} {'VQAv2':>8} {'POPE':>8} {'TextVQA':>8} {'Train':>8}")
    print(f"{'-'*70}")

    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<10} FAILED: {res['error'][:40]}")
            continue
        params = f"{res['params']/1e6:.1f}M"
        loss = f"{res['avg_loss']:.4f}"
        vqa = res['eval'].get('VQAv2', {}).get('accuracy', '-')
        pope = res['eval'].get('POPE', {}).get('accuracy', '-')
        tvqa = res['eval'].get('TextVQA', {}).get('accuracy', '-')
        tt = f"{res['train_time_s']}s"
        print(f"{name:<10} {params:>10} {loss:>8} {vqa:>8} {pope:>8} {tvqa:>8} {tt:>8}")

    print(f"\n  Previous results (for comparison):")
    print(f"  QLoRA, LoRA, Freeze, L2T, MoReS — see /content/drive/MyDrive/mmit_results/")
    print(f"\nTotal time: {total_time/60:.0f} min")

    # Save results
    results_path = os.path.join(OUTPUT_ROOT, "new_methods_results.json")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "model": MODEL_ID, "methods": all_results,
            "lavender": args.lavender, "total_time_s": total_time,
        }, f, indent=2)
    print(f"Saved to {results_path}")

    # Backup to Drive
    try:
        import shutil
        drive_path = "/content/drive/MyDrive/mmit_results/new_methods_results.json"
        if os.path.exists("/content/drive/MyDrive"):
            os.makedirs(os.path.dirname(drive_path), exist_ok=True)
            shutil.copy2(results_path, drive_path)
            print(f"Backed up to {drive_path}")
    except Exception:
        pass


if __name__ == "__main__":
    log_path = "/content/drive/MyDrive/mmit_results/new_methods_train_eval.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    try:
        main()
    except Exception as e:
        print(f"\n{F} FATAL: {e}")
        traceback.print_exc()
    finally:
        log_file.close()
