"""
Freeze Tuning 完整流程实验
==========================
用一个 tiny 模型（不需要 GPU）跑完 Freeze 的每一步。
在每个关键节点打印内部状态，让你看到代码到底做了什么。

运行方式：
  cd mmit
  python exercises/test_freeze_walkthrough.py
"""

import math
import json
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("  Freeze Tuning 完整流程实验 — 从创建模型到保存加载")
print("=" * 70)


# ══════════════════════════════════════════════════
# Step 1: 创建一个 tiny 模型（模拟 LLaVA 结构）
# ══════════════════════════════════════════════════

print("\n" + "─" * 70)
print("Step 1: 创建 tiny 模型")
print("─" * 70)


class TinyTransformerLayer(nn.Module):
    """模拟一层 Transformer（极简版）"""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class TinyVLM(nn.Module):
    """模拟一个 VLM（4 层 Transformer + embedding + lm_head）"""
    def __init__(self, vocab_size=100, dim=16, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        # 注意：用 nn.ModuleList（不是 Python list），这样才会被 nn.Module 追踪
        self.layers = nn.ModuleList([
            TinyTransformerLayer(dim) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )
        return logits, loss


model = TinyVLM(vocab_size=100, dim=16, num_layers=4)

# 打印模型结构
print("\nprint(model):")
print(model)

# 打印参数
print("\n所有参数:")
total_params = 0
for name, p in model.named_parameters():
    total_params += p.numel()
    print(f"  {name:30s} shape={str(p.shape):15s} numel={p.numel():6d} grad={p.requires_grad}")
print(f"\n  总参数量: {total_params:,}")


# ══════════════════════════════════════════════════
# Step 2: 模拟 freeze.py 的 _prepare_model_impl
# ══════════════════════════════════════════════════

print("\n" + "─" * 70)
print("Step 2: 冻结全部参数")
print("─" * 70)

# Line 68-70: 冻结全部
for p in model.parameters():
    if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
        p.requires_grad = False

trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n冻结后可训练参数: {trainable_after_freeze}")
print("所有参数的 requires_grad:")
for name, p in model.named_parameters():
    print(f"  {name:30s} grad={p.requires_grad}")


# ══════════════════════════════════════════════════
# Step 3: 解冻最后 2 层
# ══════════════════════════════════════════════════

print("\n" + "─" * 70)
print("Step 3: 解冻最后 2 层 (layers[-2:])")
print("─" * 70)

num_layers_to_train = 2
all_layers = list(model.layers)  # model.layers 是 ModuleList

print(f"\n模型共 {len(all_layers)} 层:")
for i, layer in enumerate(all_layers):
    print(f"  Layer {i}: {layer.__class__.__name__}")

print(f"\n解冻 layers[{len(all_layers) - num_layers_to_train}:] = layers[{len(all_layers) - num_layers_to_train}] 和 layers[{len(all_layers) - 1}]")

# Line 84-85: 解冻最后 N 层
for layer in all_layers[-num_layers_to_train:]:
    for p in layer.parameters():
        if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
            p.requires_grad = True

print("\n解冻后所有参数的状态:")
trainable = 0
frozen = 0
for name, p in model.named_parameters():
    if p.requires_grad:
        trainable += p.numel()
        marker = "✓ 可训练"
    else:
        frozen += p.numel()
        marker = "✗ 冻结"
    print(f"  {name:30s} {marker}")

print(f"\n可训练: {trainable:,} / {trainable + frozen:,} ({100 * trainable / (trainable + frozen):.1f}%)")
print(f"冻结:   {frozen:,}")


# ══════════════════════════════════════════════════
# Step 4: 创建 optimizer（只接收可训练参数）
# ══════════════════════════════════════════════════

print("\n" + "─" * 70)
print("Step 4: 创建 optimizer")
print("─" * 70)

# Line 113-114: get_trainable_params
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"\n传给 optimizer 的参数数量: {len(trainable_params)}")
print(f"传给 optimizer 的参数总值数: {sum(p.numel() for p in trainable_params):,}")

optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
print(f"optimizer.param_groups[0] 里有 {len(optimizer.param_groups[0]['params'])} 个参数")


# ══════════════════════════════════════════════════
# Step 5: 一次前向传播 + loss
# ══════════════════════════════════════════════════

print("\n" + "─" * 70)
print("Step 5: 前向传播 + 计算 loss")
print("─" * 70)

# 造假数据
torch.manual_seed(42)
input_ids = torch.randint(0, 100, (2, 10))  # batch=2, seq=10
labels = input_ids.clone()  # 自回归：labels = input_ids
labels[:, :3] = -100  # 前 3 个 token 是指令，不算 loss

print(f"\ninput_ids shape: {input_ids.shape}")
print(f"input_ids[0]: {input_ids[0].tolist()}")
print(f"labels[0]:    {labels[0].tolist()}")
print(f"  (-100 的位置不算 loss)")

# 前向传播
model.train()
logits, loss = model(input_ids, labels=labels)
print(f"\nlogits shape: {logits.shape}")
print(f"loss = {loss.item():.4f}")
print(f"  (随机模型的 loss ≈ -log(1/100) = {-math.log(1 / 100):.4f}，因为 vocab=100)")


# ══════════════════════════════════════════════════
# Step 6: 反向传播
# ══════════════════════════════════════════════════

print("\n" + "─" * 70)
print("Step 6: 反向传播 (loss.backward())")
print("─" * 70)

loss.backward()

print("\n每个参数的梯度状态:")
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_norm = p.grad.norm().item()
        print(f"  {name:30s} grad_norm={grad_norm:.6f}  ← 有梯度（可训练）")
    else:
        print(f"  {name:30s} grad=None           ← 无梯度（冻结的）")


# ══════════════════════════════════════════════════
# Step 7: 梯度裁剪 + optimizer.step()
# ══════════════════════════════════════════════════

print("\n" + "─" * 70)
print("Step 7: 梯度裁剪 + 参数更新")
print("─" * 70)

# 记录更新前的值
old_values = {}
for name, p in model.named_parameters():
    old_values[name] = p.data.clone()

# 梯度裁剪
total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
print(f"\n裁剪前梯度总范数: {total_norm:.6f}")

# 更新
optimizer.step()
optimizer.zero_grad()

# 检查哪些参数变了
print("\n参数是否被更新:")
for name, p in model.named_parameters():
    changed = not torch.equal(p.data, old_values[name])
    if changed:
        diff = (p.data - old_values[name]).abs().mean().item()
        print(f"  {name:30s} ✓ 变了！平均变化量={diff:.6f}")
    else:
        print(f"  {name:30s} ✗ 没变（冻结的，符合预期）")


# ══════════════════════════════════════════════════
# Step 8: 再跑一次前向，看 loss 是否降低
# ══════════════════════════════════════════════════

print("\n" + "─" * 70)
print("Step 8: 验证 — 更新后 loss 是否降低")
print("─" * 70)

logits2, loss2 = model(input_ids, labels=labels)
print(f"\n更新前 loss = {loss.item():.4f}")
print(f"更新后 loss = {loss2.item():.4f}")
print(f"变化 = {loss2.item() - loss.item():.4f}")
if loss2.item() < loss.item():
    print("✓ loss 降低了！训练在起作用。")
else:
    print("? loss 没降低（一步可能不够，或者学习率太小）")


# ══════════════════════════════════════════════════
# Step 9: 保存 checkpoint（只保存可训练参数）
# ══════════════════════════════════════════════════

print("\n" + "─" * 70)
print("Step 9: 保存 checkpoint (模拟 save_checkpoint)")
print("─" * 70)

save_dir = tempfile.mkdtemp()
print(f"\n保存目录: {save_dir}")

# Line 119-120: 只保存可训练参数
trained_names = {n for n, p in model.named_parameters() if p.requires_grad}
trainable_state = {k: v for k, v in model.state_dict().items() if k in trained_names}

print(f"\nmodel.state_dict() 共有 {len(model.state_dict())} 个 key")
print(f"其中可训练的有 {len(trainable_state)} 个 key:")
for k, v in trainable_state.items():
    print(f"  {k:30s} shape={str(v.shape)}")

save_path = os.path.join(save_dir, "freeze_tuned.pt")
torch.save(trainable_state, save_path)
file_size = os.path.getsize(save_path)
print(f"\n保存文件大小: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
print(f"  (如果保存全部参数会是 ~{sum(p.numel() * 4 for p in model.parameters()):,} bytes)")

# 保存 metadata
metadata = {"ft_method": "freeze", "trained_param_names": sorted(trained_names)}
with open(os.path.join(save_dir, "mmit_meta.json"), "w") as f:
    json.dump(metadata, f, indent=2)


# ══════════════════════════════════════════════════
# Step 10: 加载 checkpoint 到新模型
# ══════════════════════════════════════════════════

print("\n" + "─" * 70)
print("Step 10: 加载到全新模型 (模拟 load_for_inference)")
print("─" * 70)

# 创建全新的模型（随机权重）
model2 = TinyVLM(vocab_size=100, dim=16, num_layers=4)

# 对比加载前的一个可训练参数
param_name = list(trained_names)[0]
print(f"\n比较参数 '{param_name}':")
print(f"  原模型(训练后): {model.state_dict()[param_name].flatten()[:5].tolist()}")
print(f"  新模型(加载前): {model2.state_dict()[param_name].flatten()[:5].tolist()}")

# Line 140-144: 加载
state = torch.load(save_path, map_location="cpu", weights_only=True)
model2.load_state_dict(state, strict=False)  # strict=False: 只填匹配的 key

print(f"  新模型(加载后): {model2.state_dict()[param_name].flatten()[:5].tolist()}")

# 验证加载后的参数和原模型一致
match = True
for name in trained_names:
    if not torch.equal(model.state_dict()[name], model2.state_dict()[name]):
        print(f"  ✗ {name} 不匹配！")
        match = False
if match:
    print(f"\n✓ 所有 {len(trained_names)} 个可训练参数成功加载，和原模型完全一致！")

# 验证未训练的参数是新模型的随机值（不是原模型的）
untrained_name = "embed.weight"
orig = model.state_dict()[untrained_name].flatten()[:3]
new = model2.state_dict()[untrained_name].flatten()[:3]
print(f"\n未训练的参数 '{untrained_name}':")
print(f"  原模型: {orig.tolist()}")
print(f"  新模型: {new.tolist()} ← 不同！因为这个参数没被保存/加载")
print(f"  (实际使用时，新模型会从 base_model_id 加载基础权重，这里只是 demo)")


# ══════════════════════════════════════════════════
# Step 11: 用加载后的模型推理
# ══════════════════════════════════════════════════

print("\n" + "─" * 70)
print("Step 11: 推理")
print("─" * 70)

model2.eval()
with torch.no_grad():
    logits3, _ = model2(input_ids)
    pred = logits3[0, -1].argmax().item()
    print(f"\n模型预测最后一个位置的 token: {pred}")
    print(f"  (随机模型的预测没有意义，这只是展示推理流程)")

# 清理
import shutil
shutil.rmtree(save_dir)


# ══════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  总结：Freeze Tuning 完整流程")
print("=" * 70)
print("""
Step 1:  创建模型           → TinyVLM (4 层, dim=16)
Step 2:  冻结全部参数        → 所有 requires_grad = False
Step 3:  解冻最后 2 层       → layers[2], layers[3] 的 requires_grad = True
Step 4:  创建 optimizer      → 只接收可训练参数
Step 5:  前向传播            → input → embed → 4 层 → lm_head → logits → CE loss
Step 6:  反向传播            → loss.backward()，冻结参数无梯度，解冻参数有梯度
Step 7:  梯度裁剪 + 更新     → clip_grad_norm_ + optimizer.step()，只有解冻参数被更新
Step 8:  验证                → 更新后 loss 降低 ✓
Step 9:  保存                → 只保存可训练参数 (state_dict 过滤)
Step 10: 加载                → load_state_dict(strict=False) 只填匹配的 key
Step 11: 推理                → model.eval() + torch.no_grad()

每一步对应 freeze.py 的代码:
  Step 2-3: _prepare_model_impl()
  Step 4:   get_trainable_params()
  Step 5:   model(input_ids) — 不在 freeze.py 里，在 stage_runner.py 里
  Step 6:   loss.backward() — 同上
  Step 7:   clip + step — 同上
  Step 8:   验证 — 同上
  Step 9:   save_checkpoint()
  Step 10:  load_for_inference()
""")
