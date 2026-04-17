"""
MoReS Training & Evaluation on Colab
=====================================
复制到 Colab notebook 里，每个 "# %%" 是一个 cell。
或者直接在 Colab 终端里按步骤跑。
"""

# %% [markdown]
# # MoReS Training on Colab
# Bi et al., "LLaVA Steering", ACL 2025
#
# 训练 Qwen2.5-VL-3B + MoReS intervention
# 参数量: ~0.16M (vs LoRA ~6M)
# 显存: ~8GB (T4 16GB 足够)

# ══════════════════════════════════════════
# Step 1: 安装依赖
# ══════════════════════════════════════════
# %%
!pip install -q torch torchvision transformers>=4.37 peft accelerate datasets
!pip install -q bitsandbytes pyyaml pillow

# 安装 mmit（从 Google Drive）
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/mmit/src')

# 验证安装
import mmit
print("mmit loaded, available methods:", mmit.registry.list("training_method"))
# 应该输出: ['qlora', 'lora', 'dora', 'full_ft', 'freeze', 'l2t', 'mores']

# ══════════════════════════════════════════
# Step 2: 训练
# ══════════════════════════════════════════
# %%
import json, subprocess, sys

# 训练配置（和 colab_mores.yaml 一样，但直接用 JSON 跑更方便）
config = {
    "method": {
        "model_path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "",
    },
    "training": {
        "ft_method": "mores",
        "num_epochs": 3,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "save_steps": 200,
        "output_dir": "/content/drive/MyDrive/mmit_output/mores_qwen2vl_3b",
        "params": {
            "intervention_rank": 1,       # 论文最优
            "positions": "f4+l5",
            "dropout": 0.05,
            "share_weights": True,
            "steer_visual_only": True,
            "steer_ratio": 0.01,
        },
    },
    "data": {
        "adapter": "hf_datasets",
        "data_path": "HuggingFaceH4/llava-instruct-mix-vsft",
        "split": "train",
        "image_root": "",
        "max_samples": 500,   # ← 先跑 500 条验证，确认 OK 后改成 0
    },
}

config_json = json.dumps(config)
print("Starting training...")
print(f"Output: {config['training']['output_dir']}")
print(f"Samples: {config['data']['max_samples']}")
print()

# 运行训练
proc = subprocess.Popen(
    [sys.executable, "-m", "mmit.training", "--config-json", config_json],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)

for line in iter(proc.stdout.readline, ""):
    line = line.strip()
    if not line:
        continue
    try:
        event = json.loads(line)
        etype = event.get("type", "")
        data = event.get("data", {})
        if etype == "metric":
            step = data.get("step", 0)
            total = data.get("total", 0)
            loss = data.get("loss", 0)
            avg = data.get("avg_loss", 0)
            lr = data.get("lr", 0)
            eta = data.get("eta", 0)
            m, s = divmod(int(eta), 60)
            print(f"  Step {step}/{total} | Loss: {loss:.4f} | Avg: {avg:.4f} | LR: {lr:.2e} | ETA: {m}m{s:02d}s")
        elif etype == "log":
            print(f"  [{data.get('level','INFO')}] {data.get('message','')}")
        elif etype == "status":
            status = data.get("status", "")
            result = data.get("result", "")
            print(f"  [STATUS] {status}" + (f" — {result}" if result else ""))
        elif etype == "error":
            print(f"  [ERROR] {data.get('message','')}")
    except json.JSONDecodeError:
        print(line)

proc.wait()
print(f"\nDone! Exit code: {proc.returncode}")

# ══════════════════════════════════════════
# Step 3: 验证 checkpoint 存在
# ══════════════════════════════════════════
# %%
import os

ckpt_dir = "/content/drive/MyDrive/mmit_output/mores_qwen2vl_3b"
print(f"Checkpoint 目录: {ckpt_dir}")
if os.path.exists(ckpt_dir):
    for f in sorted(os.listdir(ckpt_dir)):
        size = os.path.getsize(os.path.join(ckpt_dir, f))
        print(f"  {f:40s}  {size/1024:.1f} KB")
else:
    print("  !! 目录不存在，训练可能失败了")

# 应该看到:
#   mmit_meta.json          ~1 KB
#   mores_interventions.pt  ~几十 KB（参数极小）
#   projector_state.pt      ~几 MB（projector 权重）
#   tokenizer 相关文件...

# ══════════════════════════════════════════
# Step 4: 加载模型 + 推理
# ══════════════════════════════════════════
# %%
import torch
from mmit.training.methods.mores import MoReSMethod

ckpt_dir = "/content/drive/MyDrive/mmit_output/mores_qwen2vl_3b"
base_model = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Loading base model + MoReS intervention...")
method = MoReSMethod()
model, processor, info = method.load_for_inference(ckpt_dir, base_model)
print(f"Loaded: {info['model_id']}")

# ══════════════════════════════════════════
# Step 5: 跑一张图测试
# ══════════════════════════════════════════
# %%
from PIL import Image
import requests

# 下载一张测试图
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

question = "What do you see in this image?"

# 构造 chat 格式
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)

# 移到 GPU
device = next(model.parameters()).device
moved = {}
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        v = v.to(device)
        if k == "pixel_values":
            v = v.to(torch.float16)
        moved[k] = v
    else:
        moved[k] = v

# 生成
with torch.no_grad():
    output_ids = model.generate(**moved, max_new_tokens=256, do_sample=False)

# 解码
input_len = moved["input_ids"].shape[1]
generated = output_ids[0][input_len:]
answer = processor.decode(generated, skip_special_tokens=True).strip()

print(f"Q: {question}")
print(f"A: {answer}")

# ══════════════════════════════════════════
# Step 6 (可选): 批量评测
# ══════════════════════════════════════════
# %%
# 如果你有评测数据集，可以用 mmit 的评测框架：
#
# from mmit import HFMethod
# method = HFMethod(model=model, processor=processor)
# result = method.inference("What is this?", image)
# print(result)
#
# 或者在 Gradio dashboard 里评测：
# python demos/local_viewer.py
# （需要在有 GPU 的机器上跑）

print("=" * 50)
print("恭喜！MoReS 训练 + 推理 全流程完成")
print("=" * 50)
print(f"Checkpoint: {ckpt_dir}")
print(f"训练参数量: ~0.16M（vs LoRA ~6M，少了 37 倍）")
print(f"下次使用只需要 Step 4 + Step 5（加载 + 推理）")
