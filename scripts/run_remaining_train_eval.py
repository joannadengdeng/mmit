"""Freeze + LoRA + DoRA: train → eval back to back.

Runs 3 methods sequentially, same data, outputs comparison at the end.

Usage:
    !rm -rf /content/mmit && git clone https://github.com/joannadengdeng/mmit.git /content/mmit && \
     pip install -q torch torchvision transformers peft accelerate bitsandbytes datasets pillow pyyaml && \
     cd /content/mmit && PYTHONPATH=src python scripts/run_remaining_train_eval.py
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


def load_and_preprocess(processor):
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
            print(f"   Preprocessed {i+1}/{len(samples)}...")
    print(f"   {P} {len(processed)} train samples ready")
    return processed, preproc


def train(model, method, loss_fn, processed, preproc, lr):
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
                    print(f"   [{global_step}/{total_steps}] loss={loss.item()*GRAD_ACCUM:.4f} avg={avg:.4f}")

    avg_loss = total_loss / max(1, global_step)
    train_time = time.time() - t0
    print(f"   {P} {global_step} steps, avg_loss={avg_loss:.4f}, params={trainable_count:,}, time={train_time:.0f}s")
    return avg_loss, train_time, trainable_count


def evaluate(model, processor):
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
            eval_time = time.time() - eval_t0
            results[bench_name] = {"accuracy": round(acc, 2), "correct": correct, "total": total}
            print(f"   {P} {bench_name}: {acc:.1f}% ({correct}/{total}) in {eval_time:.0f}s")

        except Exception as e:
            print(f"   {F} {bench_name}: {e}")

    return results


# ── Method configs ──

METHODS = [
    {
        "name": "Freeze",
        "quantize": False,
        "lr": 2e-4,
        "setup": lambda model, proc: _setup_freeze(model, proc),
    },
    {
        "name": "LoRA",
        "quantize": False,
        "lr": 2e-4,
        "setup": lambda model, proc: _setup_lora(model, proc),
    },
    {
        "name": "DoRA",
        "quantize": False,
        "lr": 2e-4,
        "setup": lambda model, proc: _setup_dora(model, proc),
    },
]


def _setup_freeze(model, processor):
    from mmit.training.methods.freeze import FreezeTuningMethod
    from mmit.training.losses import CrossEntropyLoss
    method = FreezeTuningMethod()
    config = {**method.default_config(), "train_modules": ["Projector"]}
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
    from mmit.training.methods.dora import DoRAMethod
    from mmit.training.losses import CrossEntropyLoss
    method = DoRAMethod()
    config = {**method.default_config(), "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
    model, info = method.prepare_model(model, processor, config)
    return model, method, CrossEntropyLoss(), info


def main():
    t0 = time.time()
    print(f"Remaining Methods: Freeze + LoRA + DoRA")
    print(f"Model: {MODEL_ID}")
    print(f"Train: {NUM_TRAIN} | Eval: {NUM_EVAL}/benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    import mmit

    # Load data once
    print("Loading data...")
    tmp_model, processor = load_model(quantize_4bit=True)
    processed, preproc = load_and_preprocess(processor)
    del tmp_model
    gc.collect()
    torch.cuda.empty_cache()

    # Load eval benchmarks info
    all_results = {}

    for mcfg in METHODS:
        name = mcfg["name"]
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        try:
            # Fresh model
            gc.collect()
            torch.cuda.empty_cache()
            model, processor = load_model(quantize_4bit=mcfg["quantize"])

            # Setup
            model, method, loss_fn, info = mcfg["setup"](model, processor)
            print(f"   {info}")

            # Train
            print("   Training...")
            avg_loss, train_time, param_count = train(model, method, loss_fn, processed, preproc, mcfg["lr"])

            # Save
            out_dir = os.path.join(OUTPUT_ROOT, f"{name.lower()}_experiment", "final")
            os.makedirs(out_dir, exist_ok=True)
            method.save_checkpoint(model, processor, out_dir, {
                "base_model": MODEL_ID, "method": name.lower(),
                "num_samples": len(processed), "avg_loss": round(avg_loss, 6),
            })

            # Eval
            print("   Evaluating...")
            eval_results = evaluate(model, processor)

            all_results[name] = {
                "avg_loss": round(avg_loss, 4),
                "train_time_s": round(train_time),
                "params": param_count,
                "eval": eval_results,
            }
            del model
            print(f"   {P} {name} done")

        except Exception as e:
            print(f"   {F} {name} FAILED: {e}")
            traceback.print_exc()
            all_results[name] = {"error": str(e)}
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
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

    # Include previous results for full table
    print(f"\n  (Previously completed: QLoRA, L2T, MoReS)")
    print(f"\nTotal time: {total_time/60:.0f} min")

    results_path = os.path.join(OUTPUT_ROOT, "remaining_results.json")
    with open(results_path, "w") as f:
        json.dump({"model": MODEL_ID, "methods": all_results}, f, indent=2)
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    log_path = "/content/drive/MyDrive/mmit_results/remaining_train_eval.txt"
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
