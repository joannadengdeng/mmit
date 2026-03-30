"""Universal train → eval script for any single method.

Usage:
    # Quick test (1K train, 200 eval)
    python scripts/run_method.py qlora

    # Full training (all data, full eval)
    python scripts/run_method.py qlora --full

    # Custom size
    python scripts/run_method.py mores --train-samples 5000 --eval-samples 500

Supported methods: qlora, lora, dora, freeze, l2t, mores
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
FORWARD_KEYS = {"input_ids", "attention_mask", "labels", "pixel_values", "image_sizes", "image_grid_thw"}

# Dataset options
DATASETS = {
    "llava_mix": {
        "name": "HuggingFaceH4/llava-instruct-mix-vsft",
        "split": "train",
        "description": "LLaVA instruct mix (~259K, pre-combined)",
    },
}

METHOD_CONFIGS = {
    "qlora":  {"quantize": True,  "lr": 2e-4},
    "lora":   {"quantize": False, "lr": 2e-4},
    "dora":   {"quantize": False, "lr": 2e-4},
    "freeze": {"quantize": False, "lr": 2e-4},
    "l2t":    {"quantize": True,  "lr": 2e-4},
    "mores":  {"quantize": False, "lr": 1e-3},
}

# Eval benchmarks with proper scoring
EVAL_BENCHMARKS = [
    ("VQAv2",   "lmms-lab/VQAv2",  "validation"),
    ("POPE",    "lmms-lab/POPE",    "test"),
    ("TextVQA", "lmms-lab/textvqa", "validation"),
]


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

def load_model(quantize_4bit):
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

def load_train_data(processor, max_samples=0, dataset_key="llava_mix"):
    """Load and preprocess training data."""
    from mmit.data.adapters.hf_datasets import HFDatasetsAdapter
    from mmit.training.preprocessors import ChatTemplatePreprocessor

    ds_info = DATASETS[dataset_key]
    adapter = HFDatasetsAdapter(dataset_name=ds_info["name"], split=ds_info["split"])

    samples = []
    for i, s in enumerate(adapter):
        samples.append(s)
        if max_samples and i >= max_samples - 1:
            break
        if (i + 1) % 10000 == 0:
            print(f"   Loaded {i+1} samples...")

    print(f"   {len(samples)} raw samples loaded")

    preproc = ChatTemplatePreprocessor()
    processed, errors = [], 0
    for i, s in enumerate(samples):
        try:
            processed.append(preproc.tokenize(s, processor, max_length=2048))
        except:
            errors += 1
        if (i + 1) % 5000 == 0:
            print(f"   Preprocessed {i+1}/{len(samples)}...")

    print(f"   {P} {len(processed)} samples ready (errors: {errors})")
    return processed, preproc


# ── Method setup ──

def setup_method(method_name, model, processor):
    from mmit.training.losses import CrossEntropyLoss, CEPlusOrthoLoss

    if method_name == "qlora":
        from mmit.training.methods.lora import QLoRAMethod
        m = QLoRAMethod()
        c = {**m.default_config(), "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
        model, info = m.prepare_model(model, processor, c)
        return model, m, CrossEntropyLoss(), info

    elif method_name == "lora":
        from mmit.training.methods.lora import LoRAMethod
        m = LoRAMethod()
        c = {**m.default_config(), "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
        model, info = m.prepare_model(model, processor, c)
        return model, m, CrossEntropyLoss(), info

    elif method_name == "dora":
        from mmit.training.methods.lora import DoRAMethod
        m = DoRAMethod()
        c = {**m.default_config(), "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
        model, info = m.prepare_model(model, processor, c)
        return model, m, CrossEntropyLoss(), info

    elif method_name == "freeze":
        from mmit.training.methods.freeze import FreezeTuningMethod
        m = FreezeTuningMethod()
        c = {**m.default_config(), "train_modules": ["Projector"]}
        model, info = m.prepare_model(model, processor, c)
        return model, m, CrossEntropyLoss(), info

    elif method_name == "l2t":
        from mmit.training.methods.l2t import L2TMethod
        m = L2TMethod()
        c = {**m.default_config(), "base_method": "qlora", "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
        model, info = m.prepare_model(model, processor, c)
        return model, m, CrossEntropyLoss(), info

    elif method_name == "mores":
        from mmit.training.methods.mores import MoReSMethod
        m = MoReSMethod()
        c = {**m.default_config(), "intervention_rank": 1, "positions": "f4+l5", "steer_ratio": 0.01, "dropout": 0.0}
        model, info = m.prepare_model(model, processor, c)
        return model, m, CEPlusOrthoLoss(ortho_weight=0.01), info

    else:
        raise ValueError(f"Unknown method: {method_name}")


# ── Training ──

def train(model, method, loss_fn, processed, preproc, lr, num_epochs=1, batch_size=1, grad_accum=8):
    model.gradient_checkpointing_enable()
    model.train()
    device = next(model.parameters()).device

    params = method.get_trainable_params(model)
    trainable_count = sum(p.numel() for pg in params for p in pg["params"])
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
    log_interval = max(1, total_steps // 20)

    print(f"   Steps: {total_steps}, Params: {trainable_count:,}, LR: {lr}")
    global_step, total_loss = 0, 0.0
    t0 = time.time()

    for epoch in range(num_epochs):
        for step, batch in enumerate(loader):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items() if k in FORWARD_KEYS}
            batch_gpu["labels"] = method.preprocess_labels(batch_gpu["input_ids"], batch_gpu["labels"])

            outputs = model(**batch_gpu)
            loss, metrics = loss_fn.compute(model, batch_gpu, outputs)

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in params for p in pg["params"] if p.grad is not None], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                total_loss += loss.item() * grad_accum

                if global_step % log_interval == 0 or global_step == 1:
                    avg = total_loss / global_step
                    elapsed = time.time() - t0
                    eta = elapsed / global_step * (total_steps - global_step)
                    extra = " ".join(f"{k}={v:.4f}" for k, v in metrics.items()) if metrics else ""
                    print(f"   [{global_step}/{total_steps}] loss={loss.item()*grad_accum:.4f} avg={avg:.4f} {extra} eta={eta:.0f}s")

    avg_loss = total_loss / max(1, global_step)
    train_time = time.time() - t0
    print(f"\n   {P} {global_step} steps, avg_loss={avg_loss:.4f}, time={train_time:.0f}s ({train_time/60:.1f}min)")
    return avg_loss, train_time, trainable_count


# ── Evaluation ──

def vqa_accuracy(pred, gt_answers):
    """VQA accuracy: min(#humans who said pred / 3, 1). Simplified version."""
    pred_clean = pred.strip().lower().rstrip(".,!?")
    if isinstance(gt_answers, list):
        # Multiple reference answers
        match_count = 0
        for ans in gt_answers:
            ans_text = ans.get("answer", str(ans)) if isinstance(ans, dict) else str(ans)
            if pred_clean == ans_text.strip().lower().rstrip(".,!?"):
                match_count += 1
        return min(match_count / 3.0, 1.0)
    else:
        gt_clean = str(gt_answers).strip().lower().rstrip(".,!?")
        return 1.0 if pred_clean == gt_clean else 0.0


def evaluate(model, processor, max_samples=0):
    """Evaluate on VQAv2, POPE, TextVQA with proper scoring."""
    from mmit.eval.methods.local_method import LocalMethod
    from mmit.data.types import EvalSample
    from datasets import load_dataset

    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except:
        pass

    eval_method = LocalMethod(model, processor)
    results = {}

    for bench_name, hf_id, split in EVAL_BENCHMARKS:
        print(f"\n   --- {bench_name} ---")
        try:
            ds = load_dataset(hf_id, split=split, streaming=True)
            eval_samples = []
            for j, row in enumerate(ds):
                if max_samples and j >= max_samples:
                    break
                eval_samples.append(row)

            total_score = 0.0
            total = 0
            eval_t0 = time.time()

            for j, s in enumerate(eval_samples):
                question = s.get("question", "Describe this image.")
                image = s.get("image")
                answers = s.get("answers", s.get("answer", ""))

                # Format prompt
                if bench_name == "POPE":
                    prompt = question + " Please answer yes or no."
                else:
                    prompt = question + " Answer the question using a single word or phrase."

                metadata = {"_pil_image": image} if image else {}
                es = EvalSample(
                    id=str(j), image_path="<in_memory>" if image else "",
                    question=prompt, ground_truth="", metadata=metadata,
                )

                prepared = eval_method.prepare_eval_input(es)
                pred = eval_method.generate(prepared, max_new_tokens=32)

                # Score
                score = vqa_accuracy(pred, answers)
                total_score += score
                total += 1

                if (j + 1) % 100 == 0:
                    acc = total_score / total * 100
                    elapsed = time.time() - eval_t0
                    eta = elapsed / (j + 1) * (len(eval_samples) - j - 1)
                    print(f"   {bench_name}: {j+1}/{len(eval_samples)} acc={acc:.1f}% eta={eta:.0f}s")

            acc = total_score / max(1, total) * 100
            eval_time = time.time() - eval_t0
            results[bench_name] = {
                "accuracy": round(acc, 2),
                "total": total,
                "time_s": round(eval_time),
            }
            print(f"   {P} {bench_name}: {acc:.1f}% ({total} samples, {eval_time:.0f}s)")

        except Exception as e:
            print(f"   {F} {bench_name}: {e}")
            traceback.print_exc()

    return results


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="mmit: train + eval a single method")
    parser.add_argument("method", choices=list(METHOD_CONFIGS.keys()), help="Training method")
    parser.add_argument("--full", action="store_true", help="Full training (all data, full eval)")
    parser.add_argument("--train-samples", type=int, default=0, help="Max train samples (0=all)")
    parser.add_argument("--eval-samples", type=int, default=0, help="Max eval samples per benchmark (0=all)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--output", type=str, default="", help="Output directory override")
    args = parser.parse_args()

    method_name = args.method
    mcfg = METHOD_CONFIGS[method_name]

    # Defaults based on --full flag
    if args.full:
        num_train = args.train_samples or 0  # 0 = all
        num_eval = args.eval_samples or 0    # 0 = all
    else:
        num_train = args.train_samples or 1000
        num_eval = args.eval_samples or 200

    output_dir = args.output or f"/content/mmit_output/{method_name}_experiment"

    t0 = time.time()
    print(f"{method_name.upper()} Train → Eval")
    print(f"Model: {MODEL_ID}")
    print(f"Train: {num_train or 'ALL'} samples, {args.epochs} epoch(s)")
    print(f"Eval: {num_eval or 'ALL'} samples/benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    import mmit

    # 1. Load model
    print("1. Loading model...")
    model, processor = load_model(mcfg["quantize"])
    print()

    # 2. Load data
    print(f"2. Loading train data...")
    processed, preproc = load_train_data(processor, max_samples=num_train)
    print()

    # 3. Setup method
    print(f"3. Setting up {method_name}...")
    model, method, loss_fn, info = setup_method(method_name, model, processor)
    print(f"   {info}\n")

    # 4. Train
    print("4. Training...")
    avg_loss, train_time, param_count = train(
        model, method, loss_fn, processed, preproc,
        lr=mcfg["lr"], num_epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
    )

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, "final")
    os.makedirs(ckpt_path, exist_ok=True)
    method.save_checkpoint(model, processor, ckpt_path, {
        "base_model": MODEL_ID, "method": method_name,
        "num_samples": len(processed), "num_epochs": args.epochs,
        "avg_loss": round(avg_loss, 6), "train_time_s": round(train_time, 1),
    })
    print(f"   {P} Checkpoint: {ckpt_path}\n")

    # 5. Eval
    print("5. Evaluating...")
    eval_results = evaluate(model, processor, max_samples=num_eval)

    # 6. Summary
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  {method_name.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"  Params: {param_count:,}")
    print(f"  Train: {len(processed)} samples, avg_loss={avg_loss:.4f}, time={train_time:.0f}s")
    print(f"  Eval:")
    for b, r in eval_results.items():
        print(f"    {b}: {r['accuracy']}% ({r['total']} samples)")
    print(f"  Total: {total_time/60:.1f} min")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": MODEL_ID, "method": method_name,
            "trainable_params": param_count,
            "train_samples": len(processed), "train_epochs": args.epochs,
            "avg_loss": avg_loss, "train_time_s": train_time,
            "eval": eval_results, "total_time_s": total_time,
        }, f, indent=2)
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    log_dir = "/content/drive/MyDrive/mmit_results"
    method_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    log_path = f"{log_dir}/{method_name}_train_eval.txt"
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    try:
        main()
    except Exception as e:
        print(f"\n{F} FAILED: {e}")
        traceback.print_exc()
    finally:
        log_file.close()
