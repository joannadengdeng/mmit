"""Train + Eval all PEFT methods on same data for comparison.

Methods: QLoRA, MoReS, Freeze (projector), L2T, LoRA, DoRA
All use same 1000 train samples, evaluate on VQAv2/POPE/TextVQA (200 each).

Usage:
    !cd /content/mmit && git fetch origin && git reset --hard origin/master && \
     PYTHONPATH=src python scripts/run_all_methods_eval.py
"""
import sys, os, time, math, gc, json, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

P = "\033[92m✓\033[0m"
F = "\033[91m✗\033[0m"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
TRAIN_DATASET = "HuggingFaceH4/llava-instruct-mix-vsft"
NUM_TRAIN = 100
NUM_EVAL = 50
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 8
OUTPUT_ROOT = "/content/mmit_output/comparison"
FORWARD_KEYS = {"input_ids", "attention_mask", "labels", "pixel_values", "image_sizes", "image_grid_thw"}


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


# ── Shared data loading ──

def load_data():
    """Load and preprocess train data + eval benchmarks (once, reused across methods)."""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Train data
    print("Loading train data...")
    from mmit.data.adapters.hf_datasets import HFDatasetsAdapter
    from mmit.training.preprocessors import ChatTemplatePreprocessor

    adapter = HFDatasetsAdapter(dataset_name=TRAIN_DATASET, split="train")
    samples = []
    for i, s in enumerate(adapter):
        samples.append(s)
        if i >= NUM_TRAIN - 1: break

    preproc = ChatTemplatePreprocessor()
    processed = []
    for i, s in enumerate(samples):
        try:
            processed.append(preproc.tokenize(s, processor, max_length=2048))
        except:
            pass
        if (i + 1) % 500 == 0:
            print(f"  Preprocessed {i+1}/{len(samples)}...")
    print(f"{P} {len(processed)} train samples ready")

    # Eval data (load once)
    print("Loading eval benchmarks...")
    eval_benchmarks = _load_eval_benchmarks()
    print(f"{P} {len(eval_benchmarks)} benchmarks ready\n")

    return processor, processed, preproc, eval_benchmarks


def _load_eval_benchmarks():
    """Load VQAv2, POPE, TextVQA from HF."""
    from datasets import load_dataset
    from mmit.data.types import EvalSample

    benchmarks = {}
    for name, hf_id, split in [
        ("VQAv2", "lmms-lab/VQAv2", "validation"),
        ("POPE", "lmms-lab/POPE", "test"),
        ("TextVQA", "lmms-lab/textvqa", "validation"),
    ]:
        try:
            ds = load_dataset(hf_id, split=split, streaming=True)
            samples = []
            for i, row in enumerate(ds):
                if i >= NUM_EVAL: break
                samples.append(row)
            if samples:
                benchmarks[name] = samples
                print(f"  {name}: {len(samples)} samples")
        except Exception as e:
            print(f"  {name}: failed ({e})")
    return benchmarks


# ── Training ──

def load_model(quantize_4bit=True):
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
    return model, processor


def train_method(model, method, processed, preproc, loss_fn, lr=2e-4):
    """Train for 1 epoch, return avg_loss and time."""
    model.gradient_checkpointing_enable()
    model.train()
    device = next(model.parameters()).device

    params = method.get_trainable_params(model)
    trainable_count = sum(p.numel() for pg in params for p in pg["params"])
    for pg in params:
        pg.setdefault("lr", lr)
    optimizer = AdamW(params, weight_decay=0.0)

    loader = DataLoader(processed, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=preproc.collate, drop_last=True)
    steps_per_epoch = len(loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * 0.03)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    log_interval = max(1, total_steps // 10)

    global_step, total_loss = 0, 0.0
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(loader):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items() if k in FORWARD_KEYS}
            batch_gpu["labels"] = method.preprocess_labels(batch_gpu["input_ids"], batch_gpu["labels"])

            outputs = model(**batch_gpu)
            loss, metrics = loss_fn.compute(model, batch_gpu, outputs)
            loss = loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in params for p in pg["params"] if p.grad is not None], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                total_loss += loss.item() * GRAD_ACCUM

                if global_step % log_interval == 0 or global_step == 1:
                    avg = total_loss / global_step
                    print(f"    [{global_step}/{total_steps}] loss={loss.item()*GRAD_ACCUM:.4f} avg={avg:.4f}")

    avg_loss = total_loss / max(1, global_step)
    train_time = time.time() - t0
    print(f"    {P} {global_step} steps, avg_loss={avg_loss:.4f}, params={trainable_count:,}, time={train_time:.0f}s")
    return avg_loss, train_time, trainable_count


# ── Evaluation ──

def evaluate_model(model, processor, eval_benchmarks):
    """Run eval on all benchmarks, return results dict."""
    from mmit.eval.methods.local_method import LocalMethod
    from mmit.data.types import EvalSample

    method = LocalMethod(model, processor)
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except:
        pass

    device = next(model.parameters()).device
    results = {}

    for bench_name, samples in eval_benchmarks.items():
        t0 = time.time()
        correct, total = 0, 0

        for i, s in enumerate(samples):
            question = s.get("question", "Describe this image.")
            image = s.get("image")
            answers = s.get("answers", s.get("answer", ""))

            # Ground truth
            if isinstance(answers, list):
                gt = answers[0].get("answer", str(answers[0])) if answers and isinstance(answers[0], dict) else str(answers[0]) if answers else ""
            else:
                gt = str(answers)

            # Prompt
            if bench_name == "POPE":
                prompt = question + " Please answer yes or no."
            else:
                prompt = question + " Answer the question using a single word or phrase."

            # Build eval sample
            metadata = {"_pil_image": image} if image else {}
            es = EvalSample(
                id=str(i), image_path="<in_memory>" if image else "",
                question=prompt, ground_truth=gt, metadata=metadata,
            )
            prepared = method.prepare_eval_input(es)
            pred = method.generate(prepared, max_new_tokens=32)

            # Score
            pred_clean = pred.strip().lower().rstrip(".")
            gt_clean = gt.strip().lower().rstrip(".")
            if pred_clean == gt_clean or gt_clean in pred_clean.split():
                correct += 1
            total += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(samples) - i - 1)
                print(f"    {bench_name}: {i+1}/{len(samples)} ({correct}/{total} correct, eta={eta:.0f}s)")

        acc = correct / max(1, total) * 100
        eval_time = time.time() - t0
        results[bench_name] = {"accuracy": round(acc, 2), "correct": correct, "total": total, "time_s": round(eval_time)}
        print(f"    {P} {bench_name}: {acc:.1f}% ({correct}/{total}) in {eval_time:.0f}s")

    return results


# ── Method configs ──

METHODS = [
    {
        "name": "QLoRA",
        "quantize": True,
        "setup": lambda model, processor: _setup_qlora(model, processor),
        "lr": 2e-4,
    },
    {
        "name": "MoReS",
        "quantize": False,  # needs bf16
        "setup": lambda model, processor: _setup_mores(model, processor),
        "lr": 1e-3,
    },
    {
        "name": "Freeze",
        "quantize": False,  # needs bf16
        "setup": lambda model, processor: _setup_freeze(model, processor),
        "lr": 2e-4,
    },
    {
        "name": "L2T",
        "quantize": True,
        "setup": lambda model, processor: _setup_l2t(model, processor),
        "lr": 2e-4,
    },
    {
        "name": "LoRA",
        "quantize": False,
        "setup": lambda model, processor: _setup_lora(model, processor),
        "lr": 2e-4,
    },
    {
        "name": "DoRA",
        "quantize": False,
        "setup": lambda model, processor: _setup_dora(model, processor),
        "lr": 2e-4,
    },
]


def _setup_qlora(model, processor):
    from mmit.training.methods.lora import QLoRAMethod
    from mmit.training.losses import CrossEntropyLoss
    method = QLoRAMethod()
    config = {**method.default_config(), "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
    model, info = method.prepare_model(model, processor, config)
    return model, method, CrossEntropyLoss(), info


def _setup_mores(model, processor):
    from mmit.training.methods.mores import MoReSMethod
    from mmit.training.losses import CEPlusOrthoLoss
    method = MoReSMethod()
    config = {**method.default_config(), "intervention_rank": 1, "positions": "f4+l5", "steer_ratio": 0.01, "dropout": 0.0}
    model, info = method.prepare_model(model, processor, config)
    return model, method, CEPlusOrthoLoss(ortho_weight=0.01), info


def _setup_freeze(model, processor):
    from mmit.training.methods.freeze import FreezeTuningMethod
    from mmit.training.losses import CrossEntropyLoss
    method = FreezeTuningMethod()
    config = {**method.default_config(), "train_modules": ["Projector"]}
    model, info = method.prepare_model(model, processor, config)
    return model, method, CrossEntropyLoss(), info


def _setup_l2t(model, processor):
    from mmit.training.methods.l2t import L2TMethod
    from mmit.training.losses import CrossEntropyLoss
    method = L2TMethod()
    config = {**method.default_config(), "base_method": "qlora", "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
    model, info = method.prepare_model(model, processor, config)
    return model, method, CrossEntropyLoss(), info


def _setup_lora(model, processor):
    from mmit.training.methods.lora import LoRAMethod
    from mmit.training.losses import CrossEntropyLoss
    method = LoRAMethod()
    config = {**method.default_config(), "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
    model, info = method.prepare_model(model, processor, config)
    return model, method, CrossEntropyLoss(), info


def _setup_dora(model, processor):
    from mmit.training.methods.lora import DoRAMethod
    from mmit.training.losses import CrossEntropyLoss
    method = DoRAMethod()
    config = {**method.default_config(), "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
    model, info = method.prepare_model(model, processor, config)
    return model, method, CrossEntropyLoss(), info


# ── Main ──

def main():
    t0 = time.time()
    print(f"All Methods Comparison")
    print(f"Model: {MODEL_ID}")
    print(f"Train: {NUM_TRAIN} samples | Eval: {NUM_EVAL} samples/benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    import mmit

    # Load data once
    processor, processed, preproc, eval_benchmarks = load_data()

    all_results = {}

    for i, mcfg in enumerate(METHODS):
        name = mcfg["name"]
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(METHODS)}] {name}")
        print(f"{'='*60}")

        try:
            # Load fresh model
            gc.collect()
            torch.cuda.empty_cache()
            model, _ = load_model(quantize_4bit=mcfg["quantize"])
            mode = "4-bit" if mcfg["quantize"] else "bf16"
            print(f"  Model loaded ({mode}), GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

            # Setup method
            model, method, loss_fn, info = mcfg["setup"](model, processor)
            print(f"  {info}")

            # Train
            print(f"  Training...")
            avg_loss, train_time, param_count = train_method(
                model, method, processed, preproc, loss_fn, lr=mcfg["lr"],
            )

            # Eval
            print(f"  Evaluating...")
            eval_results = evaluate_model(model, processor, eval_benchmarks)

            all_results[name] = {
                "avg_loss": round(avg_loss, 4),
                "train_time_s": round(train_time),
                "trainable_params": param_count,
                "eval": eval_results,
            }

            del model
            print(f"  {P} {name} done\n")

        except Exception as e:
            print(f"  {F} {name} FAILED: {e}")
            traceback.print_exc()
            all_results[name] = {"error": str(e)}
            gc.collect()
            torch.cuda.empty_cache()

    # ── Summary table ──
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  COMPARISON RESULTS ({total_time/60:.0f} min)")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'Params':>10} {'Loss':>8} {'VQAv2':>8} {'POPE':>8} {'TextVQA':>8} {'Train':>8}")
    print(f"{'-'*70}")

    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<12} {'FAILED':>10}")
            continue
        params = f"{res['trainable_params']/1e6:.1f}M"
        loss = f"{res['avg_loss']:.4f}"
        vqa = res['eval'].get('VQAv2', {}).get('accuracy', '-')
        pope = res['eval'].get('POPE', {}).get('accuracy', '-')
        tvqa = res['eval'].get('TextVQA', {}).get('accuracy', '-')
        ttime = f"{res['train_time_s']}s"
        print(f"{name:<12} {params:>10} {loss:>8} {vqa:>8} {pope:>8} {tvqa:>8} {ttime:>8}")

    print(f"\nTotal time: {total_time/60:.0f} min")

    # Save
    results_path = os.path.join(OUTPUT_ROOT, "comparison_results.json")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"model": MODEL_ID, "train_samples": NUM_TRAIN, "eval_samples": NUM_EVAL, "methods": all_results}, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    log_path = "/content/drive/MyDrive/mmit_results/all_methods_comparison.txt"
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
