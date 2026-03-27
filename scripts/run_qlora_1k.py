"""QLoRA training on 1000 samples with eval.

Usage:
    !cd /content/mmit && git fetch origin && git reset --hard origin/master && \
     PYTHONPATH=src python scripts/run_qlora_1k.py
"""
import sys, os, time, math, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

P = "\033[92m✓\033[0m"
F = "\033[91m✗\033[0m"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
NUM_SAMPLES = 1000
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4
OUTPUT_DIR = "/content/mmit_output/qlora_1k"
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


def main():
    t0 = time.time()
    print(f"QLoRA training: {NUM_SAMPLES} samples, {NUM_EPOCHS} epoch")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Effective batch size: {BATCH_SIZE}x{GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}\n")

    import mmit

    # 1. Load model
    print("1. Loading model (4-bit)...")
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
    print(f"   {P} GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB\n")

    # 2. Load + preprocess data
    print(f"2. Loading {NUM_SAMPLES} samples...")
    from mmit.data.adapters.hf_datasets import HFDatasetsAdapter
    from mmit.training.preprocessors import ChatTemplatePreprocessor

    adapter = HFDatasetsAdapter(dataset_name="HuggingFaceH4/llava-instruct-mix-vsft", split="train")
    samples = []
    for i, s in enumerate(adapter):
        samples.append(s)
        if i >= NUM_SAMPLES - 1:
            break
    print(f"   Loaded {len(samples)} samples")

    preproc = ChatTemplatePreprocessor()
    processed, errors = [], 0
    for i, s in enumerate(samples):
        try:
            processed.append(preproc.tokenize(s, processor, max_length=2048))
        except:
            errors += 1
        if (i + 1) % 200 == 0:
            print(f"   Preprocessed {i+1}/{len(samples)}...")
    print(f"   {P} {len(processed)} samples ready (errors: {errors})\n")

    # 3. Prepare QLoRA
    print("3. Injecting QLoRA...")
    from mmit.training.methods.lora import QLoRAMethod
    method = QLoRAMethod()
    config = {
        **method.default_config(),
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "freeze_patterns": ["vision_tower"],
    }
    model, info = method.prepare_model(model, processor, config)
    print(f"   {info}\n")

    # 4. Training
    print("4. Training...")
    from mmit.training.losses import CrossEntropyLoss
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

    print(f"   Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")

    global_step = 0
    total_loss = 0.0
    log_interval = max(1, total_steps // 20)  # ~20 log lines
    train_start = time.time()

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
                    elapsed = time.time() - train_start
                    eta = elapsed / global_step * (total_steps - global_step)
                    avg = total_loss / global_step
                    print(
                        f"   [{global_step}/{total_steps}] "
                        f"loss={step_loss:.4f} avg={avg:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"eta={eta:.0f}s"
                    )

    avg_loss = total_loss / max(1, global_step)
    train_time = time.time() - train_start
    print(f"\n   {P} Training done: {global_step} steps, avg_loss={avg_loss:.4f}, time={train_time:.0f}s\n")

    # 5. Save
    print("5. Saving checkpoint...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_path = os.path.join(OUTPUT_DIR, "final")
    method.save_checkpoint(model, processor, final_path, {
        "base_model": MODEL_ID,
        "num_samples": len(processed),
        "num_steps": global_step,
        "final_avg_loss": round(avg_loss, 6),
        "training_time_s": round(train_time, 1),
    })
    print(f"   {P} Saved to {final_path}\n")

    # 6. Inference test
    print("6. Inference test...")
    model.eval()
    from PIL import Image
    import requests
    from io import BytesIO

    test_cases = [
        ("https://llava-vl.github.io/static/images/view.jpg",
         "What do you see in this image?"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
         "What animal is this? Answer in one word."),
    ]

    for url, question in test_cases:
        try:
            image = Image.open(BytesIO(requests.get(url, timeout=10).content)).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"   Q: {question}")
            print(f"   A: {response}\n")
        except Exception as e:
            print(f"   Inference error: {e}\n")

    total_time = time.time() - t0
    print(f"{'='*60}")
    print(f"DONE in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Samples: {len(processed)}, Steps: {global_step}")
    print(f"  Avg loss: {avg_loss:.4f}")
    print(f"  Checkpoint: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    log_path = "/content/drive/MyDrive/mmit_results/qlora_1k_output.txt"
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
