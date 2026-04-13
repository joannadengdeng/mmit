"""Test 4 new methods: DoRA, RandLoRA, LLaVA-MoLE, ReFT.

Usage (in Colab):
    !cd /content/mmit && git pull && PYTHONPATH=src python scripts/run_new_methods_test.py
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


def load_data(processor):
    """Load & preprocess a small dataset for smoke testing."""
    from mmit.data.adapters.hf_datasets import HFDatasetsAdapter
    from mmit.training.preprocessors import ChatTemplatePreprocessor

    adapter = HFDatasetsAdapter(
        dataset_name="HuggingFaceH4/llava-instruct-mix-vsft", split="train",
    )
    samples = []
    for i, s in enumerate(adapter):
        samples.append(s)
        if i >= MAX_SAMPLES - 1:
            break

    preproc = ChatTemplatePreprocessor()
    processed = []
    for s in samples:
        try:
            processed.append(preproc.tokenize(s, processor, max_length=2048))
        except Exception:
            pass
    print(f"{P} {len(processed)} samples ready\n")
    return processed, preproc


def load_model(quantize_4bit):
    """Load base model."""
    import torch
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


def train_5_steps(model, method, processed, preproc, loss_fn, lr=2e-4):
    """Run 5 training steps and report loss."""
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    FORWARD_KEYS = {
        "input_ids", "attention_mask", "labels",
        "pixel_values", "image_sizes", "image_grid_thw",
    }
    device = next(model.parameters()).device

    model.gradient_checkpointing_enable()
    model.train()

    params = method.get_trainable_params(model)
    trainable_count = sum(p.numel() for pg in params for p in pg["params"])
    for pg in params:
        pg.setdefault("lr", lr)
    optimizer = AdamW(params)

    loader = DataLoader(
        processed, batch_size=1, shuffle=True,
        collate_fn=preproc.collate, drop_last=True,
    )

    losses = []
    for step, batch in enumerate(loader):
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items() if k in FORWARD_KEYS
        }
        batch_gpu["labels"] = method.preprocess_labels(
            batch_gpu["input_ids"], batch_gpu["labels"],
        )

        outputs = model(**batch_gpu)
        loss, metrics = loss_fn.compute(model, batch_gpu, outputs)

        if torch.isnan(loss):
            print(f"   Step {step+1}: NaN loss — skipping")
            optimizer.zero_grad()
            if step >= 6:
                break
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for pg in params for p in pg["params"] if p.grad is not None], 1.0,
        )
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        extra = " ".join(f"{k}={v:.4f}" for k, v in metrics.items()) if metrics else ""
        print(f"   Step {step+1}: loss={loss.item():.4f} {extra}")
        if len(losses) >= 5:
            break

    print(f"   Trainable params: {trainable_count:,}")
    if losses:
        print(f"   Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    return losses


def reload_model(quantize_4bit=True):
    """Reload fresh model (needed between methods that modify model differently)."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    model, processor = load_model(quantize_4bit)
    return model, processor


# ── Individual method tests ──────────────────────────────────────────

def test_dora(model, processor, processed, preproc):
    """Test DoRA: weight-decomposed LoRA (drop-in LoRA replacement)."""
    print(f"\n{'='*60}")
    print(f"  TEST 1: DoRA (weight-decomposed LoRA)")
    print(f"{'='*60}")
    from mmit.training.methods.dora import DoRAMethod
    from mmit.training.losses import CrossEntropyLoss

    method = DoRAMethod()
    config = {
        **method.default_config(),
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "freeze_patterns": ["vision_tower"],
    }
    model, info = method.prepare_model(model, processor, config)
    print(info)

    loss_fn = CrossEntropyLoss()
    losses = train_5_steps(model, method, processed, preproc, loss_fn, lr=2e-4)
    assert len(losses) >= 3, f"Only {len(losses)} steps completed"
    print(f"{P} DoRA passed\n")
    return True


def test_randlora(model, processor, processed, preproc):
    """Test RandLoRA: full-rank LoRA via random bases."""
    print(f"\n{'='*60}")
    print(f"  TEST 2: RandLoRA (full-rank via random bases)")
    print(f"{'='*60}")
    from mmit.training.methods.randlora import RandLoRAMethod
    from mmit.training.losses import CrossEntropyLoss

    method = RandLoRAMethod()
    config = {
        **method.default_config(),
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "freeze_patterns": ["vision_tower"],
    }
    model, info = method.prepare_model(model, processor, config)
    print(info)

    loss_fn = CrossEntropyLoss()
    losses = train_5_steps(model, method, processed, preproc, loss_fn, lr=2e-4)
    assert len(losses) >= 3, f"Only {len(losses)} steps completed"
    print(f"{P} RandLoRA passed\n")
    return True


def test_mole(model, processor, processed, preproc):
    """Test LLaVA-MoLE: sparse mixture of LoRA experts."""
    print(f"\n{'='*60}")
    print(f"  TEST 3: LLaVA-MoLE (sparse MoE LoRA)")
    print(f"{'='*60}")
    from mmit.training.methods.mole import MoLEMethod
    from mmit.training.losses import CrossEntropyLoss

    method = MoLEMethod()
    config = {
        **method.default_config(),
        "num_experts": 4,
        "lora_r": 8,
        "lora_alpha": 16,
        "balance_loss_coeff": 0.01,
        "freeze_patterns": ["vision_tower", "vision_model"],
    }
    model, info = method.prepare_model(model, processor, config)
    print(info)

    loss_fn = CrossEntropyLoss()
    # MoLE uses its own compute_loss for balance loss; wrap it
    class MoLELoss:
        def compute(self, model, batch, outputs):
            return method.compute_loss(model, batch, outputs)
    losses = train_5_steps(model, method, processed, preproc, MoLELoss(), lr=2e-4)
    assert len(losses) >= 3, f"Only {len(losses)} steps completed"
    print(f"{P} LLaVA-MoLE passed\n")
    return True


def test_reft(model, processor, processed, preproc):
    """Test ReFT/LoReFT: representation finetuning."""
    print(f"\n{'='*60}")
    print(f"  TEST 4: ReFT / LoReFT (representation finetuning)")
    print(f"{'='*60}")
    from mmit.training.methods.reft import ReFTMethod
    from mmit.training.losses import CrossEntropyLoss

    method = ReFTMethod()
    config = {
        **method.default_config(),
        "intervention_rank": 4,
        "positions": "f4+l5",
        "share_weights": False,
        "steer_visual_only": False,
        "dropout": 0.0,
    }
    model, info = method.prepare_model(model, processor, config)
    print(info)

    # ReFT uses its own compute_loss for ortho loss
    class ReFTLoss:
        def compute(self, model, batch, outputs):
            return method.compute_loss(model, batch, outputs)
    losses = train_5_steps(model, method, processed, preproc, ReFTLoss(), lr=1e-3)
    assert len(losses) >= 3, f"Only {len(losses)} steps completed"
    print(f"{P} ReFT passed\n")
    return True


# ── Checkpoint save/load test ────────────────────────────────────────

def test_save_load():
    """Verify save_checkpoint + load_for_inference for DoRA (representative)."""
    print(f"\n{'='*60}")
    print(f"  TEST 5: Checkpoint save/load (DoRA)")
    print(f"{'='*60}")
    import torch, tempfile
    from mmit.training.methods.dora import DoRAMethod

    model, processor = load_model(quantize_4bit=True)

    method = DoRAMethod()
    config = {**method.default_config(), "lora_r": 8, "lora_alpha": 16, "freeze_patterns": ["vision_tower"]}
    model, _ = method.prepare_model(model, processor, config)

    with tempfile.TemporaryDirectory() as tmp:
        method.save_checkpoint(model, processor, tmp, {"base_model": MODEL_ID})
        print(f"   Saved to {tmp}")

        # Check files exist
        files = os.listdir(tmp)
        assert "mmit_meta.json" in files, f"Missing mmit_meta.json, got: {files}"
        print(f"   Files: {files}")

        # Load back
        method2 = DoRAMethod()
        model2, proc2, info2 = method2.load_for_inference(tmp, MODEL_ID)
        print(f"   Loaded: {info2}")
        assert model2 is not None

    del model, model2
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{P} Save/load passed\n")
    return True


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import torch
    t0 = time.time()
    print(f"Testing 4 new methods on {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    import mmit
    print(f"Registered methods: {mmit.registry.list('training_method')}\n")

    # Load data once (using a 4-bit model for preprocessing)
    print("Loading data...")
    model_tmp, processor = load_model(quantize_4bit=True)
    processed, preproc = load_data(processor)
    del model_tmp
    gc.collect()
    torch.cuda.empty_cache()

    results = {}

    # ── Test 1: DoRA (uses 4-bit like QLoRA) ──
    print("   Loading model (4-bit) for DoRA...")
    model, processor = reload_model(quantize_4bit=True)
    try:
        results["DoRA"] = test_dora(model, processor, processed, preproc)
    except Exception as e:
        print(f"{F} DoRA FAILED: {e}")
        traceback.print_exc()
        results["DoRA"] = False

    # ── Test 2: RandLoRA (uses 4-bit) ──
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("   Loading model (4-bit) for RandLoRA...")
    model, processor = reload_model(quantize_4bit=True)
    try:
        results["RandLoRA"] = test_randlora(model, processor, processed, preproc)
    except Exception as e:
        print(f"{F} RandLoRA FAILED: {e}")
        traceback.print_exc()
        results["RandLoRA"] = False

    # ── Test 3: LLaVA-MoLE (uses bf16 — MoLE replaces Linear layers directly) ──
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("   Loading model (bf16) for MoLE...")
    model, processor = reload_model(quantize_4bit=False)
    try:
        results["MoLE"] = test_mole(model, processor, processed, preproc)
    except Exception as e:
        print(f"{F} MoLE FAILED: {e}")
        traceback.print_exc()
        results["MoLE"] = False

    # ── Test 4: ReFT (uses bf16 — interventions on hidden states) ──
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("   Loading model (bf16) for ReFT...")
    model, processor = reload_model(quantize_4bit=False)
    try:
        results["ReFT"] = test_reft(model, processor, processed, preproc)
    except Exception as e:
        print(f"{F} ReFT FAILED: {e}")
        traceback.print_exc()
        results["ReFT"] = False

    # ── Test 5: Save/load ──
    del model
    gc.collect()
    torch.cuda.empty_cache()
    try:
        results["Save/Load"] = test_save_load()
    except Exception as e:
        print(f"{F} Save/Load FAILED: {e}")
        traceback.print_exc()
        results["Save/Load"] = False

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = P if passed else F
        print(f"  {status} {name}")
    print(f"\n  {sum(results.values())}/{len(results)} passed")


if __name__ == "__main__":
    log_path = "/content/drive/MyDrive/mmit_results/new_methods_output.txt"
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
