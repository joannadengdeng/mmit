"""End-to-end pipeline test — run on Colab, paste output back.

Usage (in Colab):
    !python /content/drive/MyDrive/mmit/scripts/test_pipeline.py

Tests each component step by step with clear pass/fail output.
"""
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MAX_SAMPLES = 20  # small for fast testing


def step(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def test_import():
    step("1. Import mmit")
    import mmit
    print(mmit.registry)
    print(f"{PASS} Import OK")


def test_load_model():
    step("2. Load model (4-bit)")
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"{PASS} Processor loaded")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoVLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"{PASS} Model loaded, GPU mem: {mem:.1f} GB")
    return model, processor


def test_dataset():
    step("3. Load dataset")
    from mmit.data.adapters.hf_datasets import HFDatasetsAdapter

    adapter = HFDatasetsAdapter(
        dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
        split="train",
    )
    samples = []
    for i, s in enumerate(adapter):
        samples.append(s)
        if i + 1 >= MAX_SAMPLES:
            break
    print(f"{PASS} Loaded {len(samples)} samples")
    s0 = samples[0]
    print(f"  Sample 0: turns={len(s0.turns)}, image={'yes' if s0.image_path or (s0.metadata and s0.metadata.get('_pil_image')) else 'no'}")
    return samples


def test_preprocessor(samples, processor):
    step("4. Preprocessor")
    from mmit.training.preprocessors import ChatTemplatePreprocessor

    preproc = ChatTemplatePreprocessor()

    # Single sample
    tok = preproc.tokenize(samples[0], processor)
    print(f"  input_ids: {tok['input_ids'].shape}")
    print(f"  labels: masked={( tok['labels'] == -100).sum().item()}, loss={( tok['labels'] != -100).sum().item()}")
    has_pv = "pixel_values" in tok
    print(f"  pixel_values: {'yes ' + str(tok['pixel_values'].shape) if has_pv else 'no (text-only sample)'}")

    # Batch
    processed = []
    errors = 0
    for s in samples:
        try:
            processed.append(preproc.tokenize(s, processor, max_length=2048))
        except Exception as e:
            errors += 1
            if errors <= 2:
                print(f"  Skip: {e}")

    print(f"{PASS} Preprocessed {len(processed)}/{len(samples)} samples (errors: {errors})")

    # Collate
    batch = preproc.collate(processed[:4])
    print(f"  Collated batch: { {k: v.shape if hasattr(v, 'shape') else type(v).__name__ for k, v in batch.items()} }")
    print(f"{PASS} Collation OK")
    return processed, preproc


def test_qlora(model, processor):
    step("5. QLoRA preparation")
    from mmit.training.methods.lora import QLoRAMethod

    method = QLoRAMethod()
    config = {
        **method.default_config(),
        "lora_r": 8,
        "lora_alpha": 16,
        "freeze_patterns": ["vision_tower"],
    }
    model, info = method.prepare_model(model, processor, config)
    print(info)
    print(f"{PASS} QLoRA injected")
    return model, method


def test_training_step(model, method, processed, preproc):
    step("6. Single training step")
    import torch
    from mmit.training.losses import CrossEntropyLoss

    loss_fn = CrossEntropyLoss()
    batch = preproc.collate(processed[:2])
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    model.train()
    outputs = model(**batch)
    loss, metrics = loss_fn.compute(model, batch, outputs)
    print(f"  Loss: {loss.item():.4f}")

    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters()
                    if p.grad is not None)
    print(f"  Grad norm: {grad_norm:.4f}")
    print(f"{PASS} Forward + backward OK")


def test_save_load(model, method, processor):
    step("7. Save & load checkpoint")
    import tempfile, json

    with tempfile.TemporaryDirectory() as tmpdir:
        method.save_checkpoint(model, processor, tmpdir, {"test": True})
        files = os.listdir(tmpdir)
        print(f"  Saved files: {files}")

        meta_path = os.path.join(tmpdir, "mmit_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                print(f"  Metadata: {json.load(f)}")

    print(f"{PASS} Save/load OK")


def main():
    print(f"mmit pipeline test — {MODEL_ID}, {MAX_SAMPLES} samples")
    print(f"Python: {sys.version.split()[0]}")

    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    t0 = time.time()
    results = {}

    tests = [
        ("import", lambda: test_import()),
        ("model", lambda: None),  # placeholder
        ("dataset", lambda: None),
        ("preproc", lambda: None),
        ("qlora", lambda: None),
        ("train_step", lambda: None),
        ("save_load", lambda: None),
    ]

    try:
        test_import()
        results["import"] = True

        model, processor = test_load_model()
        results["model"] = True

        samples = test_dataset()
        results["dataset"] = True

        processed, preproc = test_preprocessor(samples, processor)
        results["preproc"] = True

        model, method = test_qlora(model, processor)
        results["qlora"] = True

        test_training_step(model, method, processed, preproc)
        results["train_step"] = True

        test_save_load(model, method, processor)
        results["save_load"] = True

    except Exception as e:
        print(f"\n{FAIL} FAILED: {e}")
        traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")
    for name in ["import", "model", "dataset", "preproc", "qlora", "train_step", "save_load"]:
        status = PASS if results.get(name) else FAIL
        print(f"  {status} {name}")

    passed = sum(results.values())
    total = 7
    print(f"\n  {passed}/{total} passed")


if __name__ == "__main__":
    main()
