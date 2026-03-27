"""Test remaining methods: LoRA, DoRA, FullFT, LoRA-in-LoRA.

Usage (in Colab):
    !cd /content/mmit && git fetch origin && git reset --hard origin/master && \
     PYTHONPATH=src python scripts/run_remaining_methods.py
"""
import sys, os, time, traceback, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

P = "\033[92m✓\033[0m"
F = "\033[91m✗\033[0m"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MAX_SAMPLES = 20


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


FORWARD_KEYS = {"input_ids", "attention_mask", "labels", "pixel_values", "image_sizes", "image_grid_thw"}


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
    mode = "4-bit" if quantize_4bit else "bf16"
    print(f"   Model loaded ({mode}), GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    return model, processor


def load_data(processor):
    from mmit.data.adapters.hf_datasets import HFDatasetsAdapter
    from mmit.training.preprocessors import ChatTemplatePreprocessor

    adapter = HFDatasetsAdapter(dataset_name="HuggingFaceH4/llava-instruct-mix-vsft", split="train")
    samples = []
    for i, s in enumerate(adapter):
        samples.append(s)
        if i >= MAX_SAMPLES - 1: break

    preproc = ChatTemplatePreprocessor()
    processed = []
    for s in samples:
        try:
            processed.append(preproc.tokenize(s, processor, max_length=2048))
        except:
            pass
    print(f"   {len(processed)} samples preprocessed")
    return processed, preproc


def train_5_steps(model, method, processed, preproc, loss_fn, lr=2e-4):
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    model.gradient_checkpointing_enable()
    model.train()

    params = method.get_trainable_params(model)
    trainable_count = sum(p.numel() for pg in params for p in pg["params"])
    for pg in params:
        pg.setdefault("lr", lr)
    optimizer = AdamW(params)
    loader = DataLoader(processed, batch_size=1, shuffle=True, collate_fn=preproc.collate, drop_last=True)

    losses = []
    for step, batch in enumerate(loader):
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items() if k in FORWARD_KEYS}
        batch_gpu["labels"] = method.preprocess_labels(batch_gpu["input_ids"], batch_gpu["labels"])

        outputs = model(**batch_gpu)
        loss, metrics = loss_fn.compute(model, batch_gpu, outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for pg in params for p in pg["params"] if p.grad is not None], 1.0
        )
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        extra = " ".join(f"{k}={v:.4f}" for k, v in metrics.items()) if metrics else ""
        print(f"   Step {step+1}: loss={loss.item():.4f} {extra}")
        if step >= 4: break

    print(f"   Params: {trainable_count:,} | Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    return losses


def fresh_model_and_cleanup(quantize_4bit=True):
    gc.collect()
    torch.cuda.empty_cache()
    return load_model(quantize_4bit)


def test_lora(processed, preproc):
    print(f"\n{'='*60}")
    print(f"  TEST 1: LoRA (bf16)")
    print(f"{'='*60}")
    model, processor = fresh_model_and_cleanup(quantize_4bit=False)
    from mmit.training.methods.lora import LoRAMethod
    from mmit.training.losses import CrossEntropyLoss

    method = LoRAMethod()
    config = {**method.default_config(), "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
    model, info = method.prepare_model(model, processor, config)
    print(info)

    losses = train_5_steps(model, method, processed, preproc, CrossEntropyLoss())
    del model
    print(f"{P} LoRA passed\n")
    return True


def test_dora(processed, preproc):
    print(f"\n{'='*60}")
    print(f"  TEST 2: DoRA (bf16)")
    print(f"{'='*60}")
    model, processor = fresh_model_and_cleanup(quantize_4bit=False)
    from mmit.training.methods.lora import DoRAMethod
    from mmit.training.losses import CrossEntropyLoss

    method = DoRAMethod()
    config = {**method.default_config(), "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
    model, info = method.prepare_model(model, processor, config)
    print(info)

    losses = train_5_steps(model, method, processed, preproc, CrossEntropyLoss())
    del model
    print(f"{P} DoRA passed\n")
    return True


def test_full_ft(processed, preproc):
    print(f"\n{'='*60}")
    print(f"  TEST 3: FullFT (bf16, freeze vision)")
    print(f"{'='*60}")
    model, processor = fresh_model_and_cleanup(quantize_4bit=False)
    from mmit.training.methods.full_ft import FullFTMethod
    from mmit.training.losses import CrossEntropyLoss

    method = FullFTMethod()
    config = {**method.default_config(), "freeze_patterns": ["vision_tower"]}
    model, info = method.prepare_model(model, processor, config)
    print(info)

    # FullFT has tons of params — use very low lr
    losses = train_5_steps(model, method, processed, preproc, CrossEntropyLoss(), lr=2e-5)
    del model
    print(f"{P} FullFT passed\n")
    return True


def test_lora_in_lora(processed, preproc):
    print(f"\n{'='*60}")
    print(f"  TEST 4: LoRA-in-LoRA (4-bit, 2-stage)")
    print(f"{'='*60}")

    # Stage 1: standard QLoRA
    print("   --- Stage 1: QLoRA base ---")
    model, processor = fresh_model_and_cleanup(quantize_4bit=True)
    from mmit.training.methods.lora import QLoRAMethod
    from mmit.training.losses import CrossEntropyLoss

    method1 = QLoRAMethod()
    config1 = {**method1.default_config(), "lora_r": 8, "freeze_patterns": ["vision_tower"]}
    model, info = method1.prepare_model(model, processor, config1)
    print(f"   {info}")

    train_5_steps(model, method1, processed, preproc, CrossEntropyLoss())

    # Save stage 1 checkpoint
    s1_path = "/tmp/mmit_lilora_s1"
    method1.save_checkpoint(model, processor, s1_path, {"base_model": MODEL_ID, "stage": "s1"})
    print(f"   Stage 1 saved to {s1_path}")
    del model

    # Stage 2: LoRA-in-LoRA on top
    print("\n   --- Stage 2: LoRA-in-LoRA ---")
    model, processor = fresh_model_and_cleanup(quantize_4bit=True)
    from mmit.training.methods.lora_in_lora import LoRAInLoRAMethod

    method2 = LoRAInLoRAMethod()
    config2 = {
        **method2.default_config(),
        "outer_checkpoint": s1_path,
        "inner_lora_r": 4,
        "inner_lora_alpha": 8,
        "freeze_patterns": ["vision_tower"],
    }
    model, info = method2.prepare_model(model, processor, config2)
    print(f"   {info}")

    losses = train_5_steps(model, method2, processed, preproc, CrossEntropyLoss())
    del model
    print(f"{P} LoRA-in-LoRA passed\n")
    return True


def main():
    t0 = time.time()
    print(f"Testing remaining methods on {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    import mmit

    # Load data once (using 4-bit model just for processor)
    print("Loading data...")
    model, processor = load_model(quantize_4bit=True)
    processed, preproc = load_data(processor)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    results = {}

    for name, test_fn in [
        ("LoRA", test_lora),
        ("DoRA", test_dora),
        ("LoRA-in-LoRA", test_lora_in_lora),
    ]:
        try:
            results[name] = test_fn(processed, preproc)
        except Exception as e:
            print(f"{F} {name} FAILED: {e}")
            traceback.print_exc()
            results[name] = False
            # Cleanup on failure
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")
    for name, passed in results.items():
        print(f"  {P if passed else F} {name}")
    print(f"\n  {sum(results.values())}/{len(results)} passed")


if __name__ == "__main__":
    log_path = "/content/drive/MyDrive/mmit_results/remaining_methods_output.txt"
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
