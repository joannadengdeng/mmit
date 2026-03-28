"""Universal train → eval script for any single method.

Usage:
    python scripts/run_method.py freeze
    python scripts/run_method.py lora
    python scripts/run_method.py dora

Supported methods: qlora, lora, dora, freeze, l2t, mores
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
NUM_TRAIN = 1000
NUM_EVAL = 200
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 8
OUTPUT_ROOT = "/content/mmit_output"
FORWARD_KEYS = {"input_ids", "attention_mask", "labels", "pixel_values", "image_sizes", "image_grid_thw"}

METHOD_CONFIGS = {
    "qlora": {"quantize": True, "lr": 2e-4},
    "lora": {"quantize": False, "lr": 2e-4},
    "dora": {"quantize": False, "lr": 2e-4},
    "freeze": {"quantize": False, "lr": 2e-4},
    "l2t": {"quantize": True, "lr": 2e-4},
    "mores": {"quantize": False, "lr": 1e-3},
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


def main(method_name):
    t0 = time.time()
    mcfg = METHOD_CONFIGS[method_name]

    print(f"{method_name.upper()} Train → Eval")
    print(f"Model: {MODEL_ID}")
    print(f"Train: {NUM_TRAIN} | Eval: {NUM_EVAL}/benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    import mmit

    # 1. Load model
    print("1. Loading model...")
    model, processor = load_model(mcfg["quantize"])
    print()

    # 2. Load data
    print(f"2. Loading {NUM_TRAIN} samples...")
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
            print(f"   {i+1}/{len(samples)}...")
    print(f"   {P} {len(processed)} samples\n")

    # 3. Setup method
    print(f"3. Setting up {method_name}...")
    model, method, loss_fn, info = setup_method(method_name, model, processor)
    print(f"   {info}\n")

    # 4. Train
    print("4. Training...")
    model.gradient_checkpointing_enable()
    model.train()
    device = next(model.parameters()).device
    lr = mcfg["lr"]

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
    log_interval = max(1, total_steps // 20)

    print(f"   Steps: {total_steps}, Params: {trainable_count:,}, LR: {lr}")
    global_step, total_loss = 0, 0.0
    train_start = time.time()

    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(loader):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items() if k in FORWARD_KEYS}
            batch_gpu["labels"] = method.preprocess_labels(batch_gpu["input_ids"], batch_gpu["labels"])

            outputs = model(**batch_gpu)
            loss, metrics = loss_fn.compute(model, batch_gpu, outputs)

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

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
                    elapsed = time.time() - train_start
                    eta = elapsed / global_step * (total_steps - global_step)
                    extra = " ".join(f"{k}={v:.4f}" for k, v in metrics.items()) if metrics else ""
                    print(f"   [{global_step}/{total_steps}] loss={loss.item()*GRAD_ACCUM:.4f} avg={avg:.4f} {extra} eta={eta:.0f}s")

    avg_loss = total_loss / max(1, global_step)
    train_time = time.time() - train_start
    print(f"\n   {P} {global_step} steps, avg_loss={avg_loss:.4f}, time={train_time:.0f}s")

    # Save
    out_dir = os.path.join(OUTPUT_ROOT, f"{method_name}_experiment", "final")
    os.makedirs(out_dir, exist_ok=True)
    method.save_checkpoint(model, processor, out_dir, {
        "base_model": MODEL_ID, "method": method_name,
        "num_samples": len(processed), "avg_loss": round(avg_loss, 6),
    })
    print(f"   {P} Checkpoint: {out_dir}\n")

    # 5. Eval
    print("5. Evaluating...")
    from mmit.eval.methods.local_method import LocalMethod
    from mmit.data.types import EvalSample
    from datasets import load_dataset

    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except:
        pass

    eval_method = LocalMethod(model, processor)
    eval_results = {}

    for bench_name, hf_id, split in [
        ("VQAv2", "lmms-lab/VQAv2", "validation"),
        ("POPE", "lmms-lab/POPE", "test"),
        ("TextVQA", "lmms-lab/textvqa", "validation"),
    ]:
        print(f"\n   --- {bench_name} ---")
        try:
            ds = load_dataset(hf_id, split=split, streaming=True)
            eval_samples = []
            for j, row in enumerate(ds):
                if j >= NUM_EVAL: break
                eval_samples.append(row)

            correct, total = 0, 0
            eval_t0 = time.time()

            for j, s in enumerate(eval_samples):
                question = s.get("question", "Describe this image.")
                image = s.get("image")
                answers = s.get("answers", s.get("answer", ""))
                if isinstance(answers, list):
                    gt = answers[0].get("answer", str(answers[0])) if answers and isinstance(answers[0], dict) else str(answers[0]) if answers else ""
                else:
                    gt = str(answers)

                prompt = question + (" Please answer yes or no." if bench_name == "POPE" else " Answer the question using a single word or phrase.")
                metadata = {"_pil_image": image} if image else {}
                es = EvalSample(id=str(j), image_path="<in_memory>" if image else "", question=prompt, ground_truth=gt, metadata=metadata)

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
            eval_results[bench_name] = {"accuracy": round(acc, 2), "correct": correct, "total": total}
            print(f"   {P} {bench_name}: {acc:.1f}% ({correct}/{total})")

        except Exception as e:
            print(f"   {F} {bench_name}: {e}")

    # Summary
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  {method_name.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"  Params: {trainable_count:,}")
    print(f"  Train: {len(processed)} samples, {global_step} steps, avg_loss={avg_loss:.4f}")
    print(f"  Eval:")
    for b, r in eval_results.items():
        print(f"    {b}: {r['accuracy']}%")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"{'='*60}")

    results_path = os.path.join(OUTPUT_ROOT, f"{method_name}_experiment", "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": MODEL_ID, "method": method_name,
            "trainable_params": trainable_count, "train_samples": len(processed),
            "train_steps": global_step, "avg_loss": avg_loss, "train_time_s": train_time,
            "eval": eval_results, "total_time_s": total_time,
        }, f, indent=2)
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <method>")
        print(f"Methods: {', '.join(METHOD_CONFIGS.keys())}")
        sys.exit(1)

    method_name = sys.argv[1].lower()
    if method_name not in METHOD_CONFIGS:
        print(f"Unknown method: {method_name}. Choose from: {', '.join(METHOD_CONFIGS.keys())}")
        sys.exit(1)

    log_path = f"/content/drive/MyDrive/mmit_results/{method_name}_train_eval.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    try:
        main(method_name)
    except Exception as e:
        print(f"\n{F} FAILED: {e}")
        traceback.print_exc()
    finally:
        log_file.close()
