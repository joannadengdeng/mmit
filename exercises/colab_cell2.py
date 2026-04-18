import math
import json
import os
import gc
import tempfile
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

print("=" * 80)
print("  Freeze Tuning 真实模型实验 — Qwen2-VL-2B on Colab T4")
print("=" * 80)


# ══════════════════════════════════════════════════
# Step 1: 加载模型
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 1: 加载 Qwen2-VL-2B 模型")
print("━" * 80)

model_id = "Qwen/Qwen2-VL-2B-Instruct"

print(f"\n加载模型: {model_id}")
print("（首次运行需要下载 ~4GB，请耐心等待...）")

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"\n✓ 模型加载完成！")
print(f"  模型类型: {model.__class__.__name__}")
print(f"  设备: {next(model.parameters()).device}")
print(f"  精度: {next(model.parameters()).dtype}")

# 打印模型顶层结构
print(f"\n模型顶层结构 (只看第一级 _modules):")
for name, module in model.named_children():
    num_params = sum(p.numel() for p in module.parameters())
    print(f"  {name:40s} {module.__class__.__name__:30s} params={num_params:>12,}")

# 总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n  总参数量: {total_params:,} ({total_params/1e9:.2f}B)")


# ══════════════════════════════════════════════════
# Step 2: 查看 LLM 层结构
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 2: 查看 LLM 的 Transformer 层")
print("━" * 80)

# 找到 LLM 的 decoder layers
llm_layers = None
search_paths = [
    "model.model.layers",            # Qwen2-VL
    "model.layers",                  # 一些模型
    "language_model.model.layers",   # LLaVA
    "language_model.layers",         # 一些模型
    "transformer.h",                 # GPT 风格
]
for attr_path in search_paths:
    obj = model
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        if isinstance(obj, nn.ModuleList):
            llm_layers = obj
            print(f"\n找到 LLM 层: model.{attr_path}")
            break
    except AttributeError:
        continue

# Fallback: 搜索所有 named_modules，找最大的 ModuleList（跳过 visual）
if llm_layers is None:
    print("\n已知路径都没找到，尝试自动搜索...")
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 4:
            if any(skip in name.lower() for skip in ("visual", "vision", "vit", "encoder", "patch")):
                continue
            llm_layers = module
            print(f"  找到: model.{name} ({len(module)} 层)")
            break

if llm_layers is None:
    print("❌ 找不到 LLM 层！打印模型结构帮助调试:")
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList):
            print(f"  ModuleList: {name} ({len(module)} 个子模块)")
else:
    print(f"  共 {len(llm_layers)} 层 Transformer")
    print(f"\n  第 0 层的结构:")
    layer0 = llm_layers[0]
    for name, module in layer0.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"    {name:30s} {module.__class__.__name__:25s} params={num_params:>10,}")

    # 每层参数量
    layer_params = sum(p.numel() for p in layer0.parameters())
    print(f"\n  每层参数量: {layer_params:,}")
    print(f"  最后 4 层参数量: {layer_params * 4:,} ({layer_params * 4 / total_params * 100:.1f}%)")


# ══════════════════════════════════════════════════
# Step 3: 冻结全部参数
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 3: 冻结全部参数")
print("━" * 80)

# 冻结前
trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n冻结前可训练参数: {trainable_before:,}")

# 冻结
for p in model.parameters():
    if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
        p.requires_grad = False

trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"冻结后可训练参数: {trainable_after:,}")


# ══════════════════════════════════════════════════
# Step 4: 解冻最后 4 层
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 4: 解冻最后 4 层")
print("━" * 80)

num_to_unfreeze = 4

if llm_layers is None:
    print("❌ 无法继续：找不到 LLM 层。请检查模型结构。")
    import sys; sys.exit(1)

total_layers = len(llm_layers)

print(f"\n模型共 {total_layers} 层，解冻 layers[{total_layers - num_to_unfreeze}:] (第 {total_layers - num_to_unfreeze} 到 {total_layers - 1} 层)")

for layer in llm_layers[-num_to_unfreeze:]:
    for p in layer.parameters():
        if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
            p.requires_grad = True

trainable_final = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_final = total_params - trainable_final

print(f"\n可训练: {trainable_final:,} ({trainable_final/total_params*100:.2f}%)")
print(f"冻结:   {frozen_final:,} ({frozen_final/total_params*100:.2f}%)")

# 打印每个参数的状态（只看代表性的）
print(f"\n参数状态抽样（每层取一个）:")
for name, p in model.named_parameters():
    # 只打印每层的第一个参数
    if "self_attn.q_proj.weight" in name or "embed" in name.lower() or "lm_head" in name:
        status = "✓ 可训练" if p.requires_grad else "✗ 冻结"
        print(f"  {name:55s} {status}  shape={str(p.shape):20s}")


# ══════════════════════════════════════════════════
# Step 5: 创建 optimizer
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 5: 创建 optimizer")
print("━" * 80)

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"\n传给 optimizer 的参数: {len(trainable_params)} 个 tensor")
print(f"可训练值总数: {sum(p.numel() for p in trainable_params):,}")

optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
print(f"optimizer 类型: AdamW, lr=1e-4")


# ══════════════════════════════════════════════════
# Step 6: 准备训练数据
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 6: 准备训练数据")
print("━" * 80)

# 构造一个简单的文本训练样本（不用图片，简化实验）
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
]

# 用 processor 构造 input
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
inputs = processor(text=[text], return_tensors="pt", padding=True)

# 移到和模型一样的设备
device = next(model.parameters()).device
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

print(f"\n输入文本: {text[:100]}...")
print(f"input_ids shape: {input_ids.shape}")
print(f"input_ids[0, :20]: {input_ids[0, :20].tolist()}")
print(f"设备: {input_ids.device}")

# 构造 labels（自回归：labels = input_ids，指令部分设 -100）
labels = input_ids.clone()
# 简单处理：前一半设为 -100（模拟指令部分不算 loss）
half = labels.shape[1] // 2
labels[:, :half] = -100
print(f"\nlabels shape: {labels.shape}")
print(f"labels[0, :20]: {labels[0, :20].tolist()}")
print(f"  前 {half} 个 token 的 label=-100（不算 loss）")
print(f"  后 {labels.shape[1] - half} 个 token 的 label=真实 token ID（算 loss）")


# ══════════════════════════════════════════════════
# Step 7: 前向传播 + loss
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 7: 前向传播 + 计算 loss")
print("━" * 80)

model.train()
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

print(f"\noutputs 的类型: {type(outputs).__name__}")
print(f"outputs 包含的 key: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'N/A'}")
print(f"outputs.loss = {outputs.loss.item():.4f}")
print(f"outputs.logits shape: {outputs.logits.shape}")
print(f"  (batch={outputs.logits.shape[0]}, seq_len={outputs.logits.shape[1]}, vocab_size={outputs.logits.shape[2]})")

vocab_size = outputs.logits.shape[-1]
random_loss = -math.log(1.0 / vocab_size)
print(f"\n随机猜测的 loss: -log(1/{vocab_size}) = {random_loss:.4f}")
print(f"当前 loss: {outputs.loss.item():.4f}")
if outputs.loss.item() < random_loss * 1.5:
    print("  模型已经有一些语言能力（预训练过的）")
else:
    print("  loss 偏高")


# ══════════════════════════════════════════════════
# Step 8: 反向传播
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 8: 反向传播 (loss.backward())")
print("━" * 80)

outputs.loss.backward()

# 统计梯度
params_with_grad = 0
params_without_grad = 0
for name, p in model.named_parameters():
    if p.grad is not None:
        params_with_grad += 1
    else:
        params_without_grad += 1

print(f"\n有梯度的参数: {params_with_grad} 个 tensor")
print(f"无梯度的参数: {params_without_grad} 个 tensor")

# 打印有梯度的参数的梯度范数（抽样）
print(f"\n有梯度参数的 grad_norm（抽样）:")
count = 0
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_norm = p.grad.norm().item()
        print(f"  {name:55s} grad_norm={grad_norm:.6f}")
        count += 1
        if count >= 10:
            print(f"  ... (省略剩余 {params_with_grad - 10} 个)")
            break


# ══════════════════════════════════════════════════
# Step 9: 梯度裁剪 + 参数更新
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 9: 梯度裁剪 + optimizer.step()")
print("━" * 80)

# 记录更新前的值
old_values = {}
for name, p in model.named_parameters():
    if p.requires_grad:
        old_values[name] = p.data.clone()

# 梯度裁剪
total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
print(f"\n裁剪前总梯度范数: {total_norm:.4f}")
print(f"max_norm=1.0, {'需要裁剪' if total_norm > 1.0 else '不需要裁剪'}")

# 更新
optimizer.step()
optimizer.zero_grad()

# 检查参数变化
changed_count = 0
unchanged_count = 0
print(f"\n参数更新检查（抽样）:")
count = 0
for name, p in model.named_parameters():
    if name in old_values:
        changed = not torch.equal(p.data, old_values[name])
        if changed:
            diff = (p.data - old_values[name]).abs().mean().item()
            changed_count += 1
            if count < 5:
                print(f"  {name:55s} ✓ 变了  平均变化={diff:.8f}")
                count += 1
        else:
            unchanged_count += 1
    else:
        # 冻结的参数不在 old_values 里
        pass

print(f"\n总计: {changed_count} 个参数被更新, {unchanged_count} 个可训练但未变化")


# ══════════════════════════════════════════════════
# Step 10: 验证 — loss 是否降低
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 10: 验证 — loss 是否降低")
print("━" * 80)

with torch.no_grad():
    outputs2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

loss_before = outputs.loss.item()
loss_after = outputs2.loss.item()
print(f"\n更新前 loss: {loss_before:.4f}")
print(f"更新后 loss: {loss_after:.4f}")
print(f"变化: {loss_after - loss_before:+.4f}")
if loss_after < loss_before:
    print("✓ loss 降低了！Freeze Tuning 在起作用。")
else:
    print("? loss 没降低（一步更新效果可能很小）")


# ══════════════════════════════════════════════════
# Step 11: 多跑几步看趋势
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 11: 多跑 10 步，观察 loss 趋势")
print("━" * 80)

model.train()
losses = [loss_before, loss_after]

for step in range(10):
    outputs_step = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss_val = outputs_step.loss.item()
    losses.append(loss_val)

    outputs_step.loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    print(f"  Step {step + 3:2d}: loss = {loss_val:.4f}  {'↓' if loss_val < losses[-2] else '↑'}")

print(f"\nLoss 趋势: {losses[0]:.4f} → {losses[-1]:.4f} (变化 {losses[-1] - losses[0]:+.4f})")


# ══════════════════════════════════════════════════
# Step 12: 保存 checkpoint
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 12: 保存 checkpoint（只保存可训练参数）")
print("━" * 80)

save_dir = "/tmp/freeze_checkpoint"
os.makedirs(save_dir, exist_ok=True)

# 只保存可训练参数
trained_names = {n for n, p in model.named_parameters() if p.requires_grad}
trainable_state = {k: v.cpu() for k, v in model.state_dict().items() if k in trained_names}

save_path = os.path.join(save_dir, "freeze_tuned.pt")
torch.save(trainable_state, save_path)
file_size = os.path.getsize(save_path)

print(f"\n保存了 {len(trainable_state)} / {len(model.state_dict())} 个参数")
print(f"文件大小: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
print(f"  如果保存全部参数: ~{total_params * 2 / 1024 / 1024:.0f} MB (bfloat16)")

print(f"\n保存的 key（前 10 个）:")
for i, k in enumerate(sorted(trainable_state.keys())):
    print(f"  {k}")
    if i >= 9:
        print(f"  ... 共 {len(trainable_state)} 个")
        break

# 保存 metadata
metadata = {
    "ft_method": "freeze",
    "base_model_id": model_id,
    "num_unfrozen_layers": num_to_unfreeze,
    "trainable_params": trainable_final,
    "total_params": total_params,
    "trained_param_names": sorted(trained_names),
}
with open(os.path.join(save_dir, "mmit_meta.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nmetadata 保存到: {save_dir}/mmit_meta.json")


# ══════════════════════════════════════════════════
# Step 13: 加载到新模型验证
# ══════════════════════════════════════════════════

print("\n" + "━" * 80)
print("  Step 13: 加载到新模型验证")
print("━" * 80)

# 释放旧模型内存
del model
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n加载新的基础模型...")
model_new = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 加载前的 loss
model_new.eval()
with torch.no_grad():
    out_before = model_new(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
print(f"加载 checkpoint 前 loss: {out_before.loss.item():.4f} (基础模型)")

# 加载 checkpoint
state = torch.load(save_path, map_location="cpu", weights_only=True)
result = model_new.load_state_dict(state, strict=False)
print(f"\nload_state_dict 结果:")
print(f"  missing_keys: {len(result.missing_keys)} 个 (这些是冻结的，没保存，正常)")
print(f"  unexpected_keys: {len(result.unexpected_keys)} 个 (应该是 0)")

# 加载后的 loss
with torch.no_grad():
    out_after = model_new(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
print(f"\n加载 checkpoint 后 loss: {out_after.loss.item():.4f}")
print(f"变化: {out_after.loss.item() - out_before.loss.item():+.4f}")
if out_after.loss.item() < out_before.loss.item():
    print("✓ 加载后 loss 更低 = checkpoint 正确加载了训练好的参数！")


# ══════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  总结")
print("=" * 80)
print(f"模型: {model_id}")
print(f"总参数: {total_params:,} ({total_params/1e9:.2f}B)")
print(f"冻结参数: {frozen_final:,} ({frozen_final/total_params*100:.1f}%)")
print(f"可训练参数: {trainable_final:,} ({trainable_final/total_params*100:.1f}%)")
print(f"训练策略: 冻结全部 → 解冻最后 {num_to_unfreeze} 层")
print(f"训练 12 步: loss {losses[0]:.4f} → {losses[-1]:.4f}")
print(f"Checkpoint: {file_size/1024/1024:.1f} MB (全量保存需 ~{total_params*2/1024/1024:.0f} MB)")
print()
print("对应 freeze.py 的代码:")
print("  _prepare_model_impl() → Step 3-4 (冻结 + 解冻)")
print("  compute_loss()        → Step 7 (CE loss)")
print("  get_trainable_params()→ Step 5 (传给 optimizer)")
print("  save_checkpoint()     → Step 12 (只保存可训练参数)")
print("  load_for_inference()  → Step 13 (加载基础模型 + 覆盖训练过的参数)")