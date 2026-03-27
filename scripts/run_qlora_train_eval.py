"""QLoRA: full train → eval pipeline.

Train on llava-instruct-mix, evaluate on VQAv2, POPE, TextVQA.

Usage:
    !cd /content/mmit && git fetch origin && git reset --hard origin/master && \
     PYTHONPATH=src python scripts/run_qlora_train_eval.py
"""
import sys, os, time, math, gc, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

P = "\033[92m✓\033[0m"
F = "\033[91m✗\033[0m"

# ── Config ──
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
TRAIN_DATASET = "HuggingFaceH4/llava-instruct-mix-vsft"
NUM_TRAIN_SAMPLES = 1000          # set to 0 for full dataset
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
OUTPUT_DIR = "/content/mmit_output/qlora_experiment"
EVAL_MAX_SAMPLES = 200            # samples per eval benchmark (0=full)
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


def train(model, processor):
    """Train with QLoRA and return the trained model."""
    from mmit.data.adapters.hf_datasets import HFDatasetsAdapter
    from mmit.training.preprocessors import ChatTemplatePreprocessor
    from mmit.training.methods.lora import QLoRAMethod
    from mmit.training.losses import CrossEntropyLoss

    # Load data
    print(f"\n{'='*60}")
    print(f"  TRAINING: QLoRA on {TRAIN_DATASET}")
    print(f"{'='*60}")

    print(f"Loading data (max {NUM_TRAIN_SAMPLES or 'all'} samples)...")
    adapter = HFDatasetsAdapter(dataset_name=TRAIN_DATASET, split="train")
    samples = []
    for i, s in enumerate(adapter):
        samples.append(s)
        if NUM_TRAIN_SAMPLES and i >= NUM_TRAIN_SAMPLES - 1:
            break

    preproc = ChatTemplatePreprocessor()
    processed, errors = [], 0
    for i, s in enumerate(samples):
        try:
            processed.append(preproc.tokenize(s, processor, max_length=2048))
        except:
            errors += 1
        if (i + 1) % 200 == 0:
            print(f"  Preprocessed {i+1}/{len(samples)}...")
    print(f"{P} {len(processed)} samples ready (errors: {errors})")

    # QLoRA
    method = QLoRAMethod()
    config = {
        **method.default_config(),
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.05,
        "freeze_patterns": ["vision_tower"],
    }
    model, info = method.prepare_model(model, processor, config)
    print(info)

    # Training loop
    loss_fn = CrossEntropyLoss()
    model.gradient_checkpointing_enable()
    model.train()
    device = next(model.parameters()).device

    params = method.get_trainable_params(model)
    for pg in params:
        pg.setdefault("lr", LR)
    optimizer = AdamW(params, weight_decay=0.0)

    loader = DataLoader(
        processed, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=preproc.collate, drop_last=True,
    )
    steps_per_epoch = len(loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * 0.03)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    log_interval = max(1, total_steps // 20)

    print(f"\nTraining: {len(processed)} samples, {total_steps} steps, bs={BATCH_SIZE}x{GRAD_ACCUM}")
    global_step, total_loss = 0, 0.0
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(loader):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items() if k in FORWARD_KEYS}
            outputs = model(**batch_gpu)
            loss, _ = loss_fn.compute(model, batch_gpu, outputs)
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
                step_loss = loss.item() * GRAD_ACCUM
                total_loss += step_loss

                if global_step % log_interval == 0 or global_step == 1:
                    elapsed = time.time() - t0
                    eta = elapsed / global_step * (total_steps - global_step)
                    print(f"  [{global_step}/{total_steps}] loss={step_loss:.4f} avg={total_loss/global_step:.4f} lr={scheduler.get_last_lr()[0]:.2e} eta={eta:.0f}s")

    avg_loss = total_loss / max(1, global_step)
    train_time = time.time() - t0
    print(f"\n{P} Training: {global_step} steps, avg_loss={avg_loss:.4f}, time={train_time:.0f}s")

    # Save
    ckpt_path = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(ckpt_path, exist_ok=True)
    method.save_checkpoint(model, processor, ckpt_path, {
        "base_model": MODEL_ID,
        "train_dataset": TRAIN_DATASET,
        "num_samples": len(processed),
        "num_steps": global_step,
        "avg_loss": round(avg_loss, 6),
        "lora_r": LORA_R,
        "training_time_s": round(train_time, 1),
    })
    print(f"{P} Checkpoint saved to {ckpt_path}")

    return model, avg_loss, global_step, train_time


def evaluate(model, processor):
    """Evaluate on VQAv2, POPE, TextVQA using local model inference."""
    from mmit.eval.methods.local_method import LocalMethod
    from mmit.eval.engine import EvalEngine
    from mmit.eval.benchmarks.vqav2 import VQAv2Benchmark
    from mmit.eval.benchmarks.pope import POPEBenchmark
    from mmit.eval.benchmarks.textvqa import TextVQABenchmark

    print(f"\n{'='*60}")
    print(f"  EVALUATION")
    print(f"{'='*60}")

    method = LocalMethod(model, processor)
    engine = EvalEngine(max_new_tokens=128, temperature=0.0)
    all_results = {}

    benchmarks = [
        ("VQAv2", VQAv2Benchmark),
        ("POPE", POPEBenchmark),
        ("TextVQA", TextVQABenchmark),
    ]

    for bench_name, bench_cls in benchmarks:
        print(f"\n  --- {bench_name} ---")
        try:
            # Try to load benchmark from HuggingFace datasets
            bench = _load_benchmark_from_hf(bench_name, bench_cls)
            if bench is None:
                print(f"  Skipped (benchmark data not available)")
                continue

            out_path = os.path.join(OUTPUT_DIR, f"eval_{bench_name.lower()}.jsonl")
            metrics = engine.run(method, bench, output_file=out_path, show_progress=True)
            all_results[bench_name] = metrics
            print(f"  {P} {bench_name}: {metrics}")

        except Exception as e:
            print(f"  {F} {bench_name} failed: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def _load_benchmark_from_hf(name, bench_cls):
    """Try to load benchmark data. Return None if unavailable."""
    try:
        from datasets import load_dataset
        import tempfile

        if name == "VQAv2":
            # Use lmms-lab/VQAv2 from HuggingFace
            ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
            samples = []
            for i, row in enumerate(ds):
                if EVAL_MAX_SAMPLES and i >= EVAL_MAX_SAMPLES:
                    break
                samples.append(row)

            if not samples:
                return None

            # Write to temp JSONL for VQAv2Benchmark
            tmp_q = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
            tmp_a = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)

            annotations = []
            for s in samples:
                q_entry = {
                    "question_id": s.get("question_id", s.get("id", 0)),
                    "image": "",
                    "question": s.get("question", ""),
                }
                # Store image in metadata
                q_entry["_pil_image"] = True
                tmp_q.write(json.dumps(q_entry) + "\n")

                answers = s.get("answers", [])
                if isinstance(answers, list) and answers:
                    if isinstance(answers[0], dict):
                        ans_list = [a.get("answer", "") for a in answers]
                    else:
                        ans_list = [str(a) for a in answers]
                else:
                    ans_list = [str(answers)]

                annotations.append({
                    "question_id": q_entry["question_id"],
                    "answers": [{"answer": a} for a in ans_list],
                })

            tmp_q.close()
            json.dump({"annotations": annotations}, open(tmp_a.name, "w"))
            tmp_a.close()

            # This benchmark needs image paths, but HF VQAv2 has embedded images.
            # We need a custom approach. Skip for now and use a simpler eval.
            print(f"  VQAv2 from HF: {len(samples)} samples (using simple accuracy)")
            return _SimpleVQABenchmark(samples, name="VQAv2")

        elif name == "POPE":
            # POPE is typically a local file, try HF
            ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
            samples = []
            for i, row in enumerate(ds):
                if EVAL_MAX_SAMPLES and i >= EVAL_MAX_SAMPLES:
                    break
                samples.append(row)
            if not samples:
                return None
            print(f"  POPE from HF: {len(samples)} samples")
            return _SimpleVQABenchmark(samples, name="POPE")

        elif name == "TextVQA":
            ds = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
            samples = []
            for i, row in enumerate(ds):
                if EVAL_MAX_SAMPLES and i >= EVAL_MAX_SAMPLES:
                    break
                samples.append(row)
            if not samples:
                return None
            print(f"  TextVQA from HF: {len(samples)} samples")
            return _SimpleVQABenchmark(samples, name="TextVQA")

    except Exception as e:
        print(f"  Could not load {name}: {e}")
        return None


class _SimpleVQABenchmark:
    """Simple benchmark wrapper for HF datasets with embedded images."""

    def __init__(self, samples, name=""):
        self.samples = samples
        self.name = name

    def iter_questions(self):
        from mmit.data.types import EvalSample
        for i, s in enumerate(self.samples):
            question = s.get("question", "Describe this image.")
            image = s.get("image")
            answers = s.get("answers", s.get("answer", ""))

            # Normalize answers
            if isinstance(answers, list):
                if answers and isinstance(answers[0], dict):
                    gt = answers[0].get("answer", "")
                else:
                    gt = str(answers[0]) if answers else ""
            else:
                gt = str(answers)

            # Format prompt based on benchmark type
            if self.name == "POPE":
                prompt = question + " Please answer yes or no."
            elif self.name in ("VQAv2", "TextVQA"):
                prompt = question + " Answer the question using a single word or phrase."
            else:
                prompt = question

            metadata = {}
            if image is not None:
                metadata["_pil_image"] = image

            yield EvalSample(
                id=str(s.get("question_id", s.get("id", i))),
                image_path="<in_memory>" if image else "",
                question=prompt,
                ground_truth=gt,
                metadata=metadata,
            )

    def build_prompt(self, sample):
        return sample.question

    def score(self, predictions):
        correct = 0
        total = len(predictions)
        for p in predictions:
            pred = p["prediction"].strip().lower()
            gt = p["ground_truth"].strip().lower()

            # Simple exact match (with some normalization)
            pred_clean = pred.rstrip(".").strip()
            gt_clean = gt.rstrip(".").strip()

            if pred_clean == gt_clean:
                correct += 1
            elif gt_clean in pred_clean.split():
                correct += 1

        acc = correct / max(1, total) * 100
        return {"accuracy": round(acc, 2), "correct": correct, "total": total}


def main():
    t0 = time.time()
    print(f"QLoRA Train → Eval Pipeline")
    print(f"Model: {MODEL_ID}")
    print(f"Train: {TRAIN_DATASET} ({NUM_TRAIN_SAMPLES or 'full'} samples)")
    print(f"Eval: VQAv2, POPE, TextVQA ({EVAL_MAX_SAMPLES or 'full'} samples each)")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    import mmit

    # Load model
    print("Loading model (4-bit)...")
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoVLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        ),
        device_map="auto", trust_remote_code=True,
    )
    print(f"{P} Model loaded, GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # Train
    model, avg_loss, steps, train_time = train(model, processor)

    # Eval
    model.eval()
    # Disable gradient checkpointing for inference
    try:
        model.gradient_checkpointing_disable()
    except:
        pass

    eval_results = evaluate(model, processor)

    # Final summary
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Model: {MODEL_ID}")
    print(f"  Method: QLoRA (r={LORA_R}, alpha={LORA_ALPHA})")
    print(f"  Train: {NUM_TRAIN_SAMPLES or 'full'} samples, {steps} steps, avg_loss={avg_loss:.4f}")
    print(f"  Train time: {train_time:.0f}s")
    print(f"  Eval results:")
    for bench, metrics in eval_results.items():
        print(f"    {bench}: {metrics}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Checkpoint: {OUTPUT_DIR}/final")
    print(f"{'='*60}")

    # Save results JSON
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "method": "qlora",
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "train_dataset": TRAIN_DATASET,
            "train_samples": NUM_TRAIN_SAMPLES,
            "train_steps": steps,
            "avg_loss": avg_loss,
            "train_time_s": train_time,
            "eval": eval_results,
            "total_time_s": total_time,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    log_path = "/content/drive/MyDrive/mmit_results/qlora_train_eval_output.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    try:
        main()
    except Exception as e:
        print(f"\n{F} FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()
