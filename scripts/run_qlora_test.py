"""Self-contained QLoRA test — no Drive job queue needed.

Usage (in Colab):
    !PYTHONPATH=/content/mmit/src python /content/mmit/scripts/run_qlora_test.py
"""
import sys, os, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

P = "\033[92m✓\033[0m"
F = "\033[91m✗\033[0m"

def main():
    t0 = time.time()
    MODEL_ID = "llava-hf/llava-1.5-7b-hf"

    # 1. Import
    print("1. Import mmit...")
    import mmit
    print(f"   {P} {mmit.registry}\n")

    # 2. Load model
    print("2. Loading model (4-bit)...")
    import torch
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
    print(f"   {P} Model loaded, GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB\n")

    # 3. Load data
    print("3. Loading dataset (20 samples)...")
    from mmit.data.adapters.hf_datasets import HFDatasetsAdapter
    adapter = HFDatasetsAdapter(dataset_name="HuggingFaceH4/llava-instruct-mix-vsft", split="train")
    samples = []
    for i, s in enumerate(adapter):
        samples.append(s)
        if i >= 19: break
    print(f"   {P} {len(samples)} samples loaded\n")

    # 4. Preprocess
    print("4. Preprocessing...")
    from mmit.training.preprocessors import ChatTemplatePreprocessor
    preproc = ChatTemplatePreprocessor()
    processed, errors = [], 0
    for i, s in enumerate(samples):
        try:
            processed.append(preproc.tokenize(s, processor, max_length=2048))
        except Exception as e:
            errors += 1
            if errors <= 2: print(f"   Skip {i}: {e}")
    print(f"   {P} {len(processed)}/{len(samples)} preprocessed (errors: {errors})\n")

    # 5. QLoRA
    print("5. Injecting QLoRA...")
    from mmit.training.methods.lora import QLoRAMethod
    method = QLoRAMethod()
    config = {**method.default_config(), "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
    model, info = method.prepare_model(model, processor, config)
    print(f"   {info}\n")

    # 6. Training step
    print("6. Training step...")
    from mmit.training.losses import CrossEntropyLoss
    from torch.utils.data import DataLoader
    loss_fn = CrossEntropyLoss()
    batch = preproc.collate(processed[:2])
    device = next(model.parameters()).device

    # Only pass keys the model accepts
    FORWARD_KEYS = {"input_ids", "attention_mask", "labels", "pixel_values", "image_sizes", "image_grid_thw"}
    def to_device(batch):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items() if k in FORWARD_KEYS}

    batch_gpu = to_device(batch)
    print(f"   Batch keys: {list(batch_gpu.keys())}")
    print(f"   input_ids: {batch_gpu['input_ids'].shape}")
    if 'pixel_values' in batch_gpu:
        print(f"   pixel_values: {batch_gpu['pixel_values'].shape}")

    model.train()
    outputs = model(**batch_gpu)
    loss, _ = loss_fn.compute(model, batch_gpu, outputs)
    loss.backward()
    print(f"   {P} Loss: {loss.item():.4f}\n")

    # 7. Full training loop (5 steps)
    print("7. Training loop (5 steps)...")
    from torch.optim import AdamW
    params = method.get_trainable_params(model)
    for pg in params: pg.setdefault("lr", 2e-4)
    optimizer = AdamW(params)
    loader = DataLoader(processed, batch_size=2, shuffle=True, collate_fn=preproc.collate, drop_last=True)

    model.train()
    for step, batch in enumerate(loader):
        batch_gpu = to_device(batch)
        outputs = model(**batch_gpu)
        loss, _ = loss_fn.compute(model, batch_gpu, outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"   Step {step+1}: loss={loss.item():.4f}")
        if step >= 4: break
    print(f"   {P} Training OK\n")

    # 8. Save
    print("8. Saving checkpoint...")
    out_dir = "/content/mmit_output/qlora_test/final"
    os.makedirs(out_dir, exist_ok=True)
    method.save_checkpoint(model, processor, out_dir, {"base_model": MODEL_ID, "test": True})
    files = os.listdir(out_dir)
    print(f"   {P} Saved to {out_dir}: {files}\n")

    print(f"{'='*50}")
    print(f"ALL TESTS PASSED in {time.time()-t0:.0f}s")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Tee stdout to Drive so local machine can poll results
    import io

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

    log_path = "/content/drive/MyDrive/mmit_results/qlora_test_output.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
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
        print(f"\nOutput saved to {log_path}", file=sys.__stdout__)
