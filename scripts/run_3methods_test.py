"""Test 3 compute-efficient methods: MoReS, Freeze, L2T.

Usage (in Colab):
    !cd /content/mmit && git pull && PYTHONPATH=src python scripts/run_3methods_test.py
"""
import sys, os, time, traceback, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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


def load_base():
    """Load model + processor + preprocessed data (shared across all tests)."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    print("Loading processor + model (4-bit)...")
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

    print("Loading + preprocessing dataset...")
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
    print(f"{P} {len(processed)} samples ready\n")
    return model, processor, processed, preproc


def train_5_steps(model, method, processed, preproc, loss_fn, method_name, lr=2e-4):
    """Run 5 training steps and report loss."""
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    FORWARD_KEYS = {"input_ids", "attention_mask", "labels", "pixel_values", "image_sizes", "image_grid_thw"}
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

        # Apply label preprocessing (L2T modifies labels here)
        batch_gpu["labels"] = method.preprocess_labels(
            batch_gpu["input_ids"], batch_gpu["labels"],
        )

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

    print(f"   Trainable params: {trainable_count:,}")
    print(f"   Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    return losses


def reload_model():
    """Reload fresh model (needed between methods that modify model differently)."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()
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
    return model, processor


def test_mores(model, processor, processed, preproc):
    """Test MoReS: representation steering, ~500x fewer params than LoRA."""
    print(f"\n{'='*60}")
    print(f"  TEST 1: MoReS (representation steering)")
    print(f"{'='*60}")
    from mmit.training.methods.mores import MoReSMethod
    from mmit.training.losses import CEPlusOrthoLoss

    method = MoReSMethod()
    config = {
        **method.default_config(),
        "intervention_rank": 1,
        "positions": "f4+l5",
        "steer_ratio": 0.01,
        "dropout": 0.0,
    }
    model, info = method.prepare_model(model, processor, config)
    print(info)

    loss_fn = CEPlusOrthoLoss(ortho_weight=0.01)
    losses = train_5_steps(model, method, processed, preproc, loss_fn, "MoReS", lr=1e-3)
    print(f"{P} MoReS passed\n")
    return True


def test_freeze(model, processor, processed, preproc):
    """Test Freeze: only train the projector (~8M params)."""
    print(f"\n{'='*60}")
    print(f"  TEST 2: Freeze (projector only)")
    print(f"{'='*60}")
    from mmit.training.methods.freeze import FreezeTuningMethod
    from mmit.training.losses import CrossEntropyLoss

    method = FreezeTuningMethod()
    config = {
        **method.default_config(),
        "train_modules": ["Projector"],
    }
    model, info = method.prepare_model(model, processor, config)
    print(info)

    loss_fn = CrossEntropyLoss()
    losses = train_5_steps(model, method, processed, preproc, loss_fn, "Freeze")
    print(f"{P} Freeze passed\n")
    return True


def test_l2t(model, processor, processed, preproc):
    """Test L2T: instruction-aware loss masking wrapping QLoRA."""
    print(f"\n{'='*60}")
    print(f"  TEST 3: L2T (instruction-aware loss, wrapping QLoRA)")
    print(f"{'='*60}")
    from mmit.training.methods.l2t import L2TMethod
    from mmit.training.losses import CrossEntropyLoss

    method = L2TMethod()
    config = {
        **method.default_config(),
        "base_method": "qlora",
        "template_top_k": 20,
        "lora_r": 8,
        "lora_alpha": 16,
        "freeze_patterns": ["vision_tower"],
    }
    model, info = method.prepare_model(model, processor, config)
    print(info)

    loss_fn = CrossEntropyLoss()
    losses = train_5_steps(model, method, processed, preproc, loss_fn, "L2T")
    print(f"{P} L2T passed\n")
    return True


def main():
    import torch
    t0 = time.time()
    print(f"Testing 3 methods on {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    import mmit
    model, processor, processed, preproc = load_base()

    results = {}

    # Test 1: MoReS
    try:
        results["MoReS"] = test_mores(model, processor, processed, preproc)
    except Exception as e:
        print(f"{F} MoReS FAILED: {e}")
        traceback.print_exc()
        results["MoReS"] = False

    # Reload model for next test (MoReS adds hooks that would interfere)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model, processor = reload_model()

    # Test 2: Freeze
    try:
        results["Freeze"] = test_freeze(model, processor, processed, preproc)
    except Exception as e:
        print(f"{F} Freeze FAILED: {e}")
        traceback.print_exc()
        results["Freeze"] = False

    # Reload for L2T
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model, processor = reload_model()

    # Test 3: L2T
    try:
        results["L2T"] = test_l2t(model, processor, processed, preproc)
    except Exception as e:
        print(f"{F} L2T FAILED: {e}")
        traceback.print_exc()
        results["L2T"] = False

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = P if passed else F
        print(f"  {status} {name}")
    print(f"\n  {sum(results.values())}/{len(results)} passed")


if __name__ == "__main__":
    log_path = "/content/drive/MyDrive/mmit_results/3methods_output.txt"
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
