#!/usr/bin/env python3
"""
PyTorch 基础反复练习题集 (40 题)
================================
设计目的：通过反复练习建立 PyTorch 肌肉记忆。
每道题都是自包含、可运行、自动评分的。

使用方法：
  1. 把每个函数中的 raise NotImplementedError 替换成你的实现
  2. 运行 python pytorch_drills.py 查看得分
  3. 反复练习直到全部通过！
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  Group 1: Tensor 基础 (8 题)
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────
# 练习 1: create_tensor (★)
# ───────────────────────────────────────────────────────────────
def create_tensor():
    """
    创建一个包含 [1, 2, 3, 4, 5] 的 float 类型 tensor 并返回。

    返回:
      - t: shape 为 (5,)，dtype 为 torch.float32 的 tensor

    提示: 使用 torch.tensor(..., dtype=torch.float32)
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_1():
    t = create_tensor()
    assert isinstance(t, torch.Tensor), "返回值必须是 torch.Tensor"
    assert t.dtype == torch.float32, f"dtype 应为 float32，得到 {t.dtype}"
    assert t.shape == (5,), f"shape 应为 (5,)，得到 {t.shape}"
    assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])), "值不正确"
    print("  \u2713 Ex 1: create_tensor — 创建 float tensor")


# ───────────────────────────────────────────────────────────────
# 练习 2: tensor_shape (★)
# ───────────────────────────────────────────────────────────────
def tensor_shape(x):
    """
    给定一个 3 维 tensor x，返回它的 (batch, seq_len, dim) 三个值。

    参数:
      - x: shape 为 (B, S, D) 的 tensor
    返回:
      - batch: int
      - seq_len: int
      - dim: int

    提示: 可以用 x.shape 或 x.size()
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_2():
    x = torch.randn(4, 16, 64)
    b, s, d = tensor_shape(x)
    assert b == 4, f"batch 应为 4，得到 {b}"
    assert s == 16, f"seq_len 应为 16，得到 {s}"
    assert d == 64, f"dim 应为 64，得到 {d}"
    print("  \u2713 Ex 2: tensor_shape — 获取 tensor 维度")


# ───────────────────────────────────────────────────────────────
# 练习 3: reshape_tensor (★)
# ───────────────────────────────────────────────────────────────
def reshape_tensor(x):
    """
    把 shape 为 (2, 12) 的 tensor 变形为 (2, 3, 4)。

    参数:
      - x: shape 为 (2, 12) 的 tensor
    返回:
      - y: shape 为 (2, 3, 4) 的 tensor

    提示: 使用 x.reshape() 或 x.view()
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_3():
    x = torch.arange(24).float().reshape(2, 12)
    y = reshape_tensor(x)
    assert y.shape == (2, 3, 4), f"shape 应为 (2, 3, 4)，得到 {y.shape}"
    # 检查数据一致性
    assert torch.equal(y.reshape(2, 12), x), "数据不一致"
    print("  \u2713 Ex 3: reshape_tensor — tensor 变形")


# ───────────────────────────────────────────────────────────────
# 练习 4: transpose_last_two (★)
# ───────────────────────────────────────────────────────────────
def transpose_last_two(x):
    """
    把 (B, S, D) tensor 的最后两维转置成 (B, D, S)。

    参数:
      - x: shape 为 (B, S, D) 的 tensor
    返回:
      - y: shape 为 (B, D, S) 的 tensor

    提示: 使用 x.transpose(-1, -2) 或 x.permute(0, 2, 1)
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_4():
    x = torch.randn(3, 5, 7)
    y = transpose_last_two(x)
    assert y.shape == (3, 7, 5), f"shape 应为 (3, 7, 5)，得到 {y.shape}"
    assert torch.equal(y, x.transpose(-1, -2)), "转置结果不正确"
    print("  \u2713 Ex 4: transpose_last_two — 最后两维转置")


# ───────────────────────────────────────────────────────────────
# 练习 5: matmul_exercise (★)
# ───────────────────────────────────────────────────────────────
def matmul_exercise(a, b):
    """
    计算两个矩阵的乘积。a 的 shape 为 (2, 3)，b 的 shape 为 (3, 4)。

    参数:
      - a: shape (2, 3) 的 tensor
      - b: shape (3, 4) 的 tensor
    返回:
      - c: shape (2, 4) 的 tensor，即 a @ b

    提示: 使用 torch.matmul() 或 @ 运算符
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_5():
    torch.manual_seed(42)
    a = torch.randn(2, 3)
    b = torch.randn(3, 4)
    c = matmul_exercise(a, b)
    assert c.shape == (2, 4), f"shape 应为 (2, 4)，得到 {c.shape}"
    assert torch.allclose(c, a @ b), "矩阵乘积结果不正确"
    print("  \u2713 Ex 5: matmul_exercise — 矩阵乘法")


# ───────────────────────────────────────────────────────────────
# 练习 6: split_heads (★★)
# ───────────────────────────────────────────────────────────────
def split_heads(x, num_heads):
    """
    把 (B, S, H*D) 的 tensor 拆成 (B, H, S, D)。
    用于多头注意力的 head 拆分。

    参数:
      - x: shape 为 (B, S, H*D) 的 tensor
      - num_heads: int，头的数量 H
    返回:
      - y: shape 为 (B, H, S, D) 的 tensor

    提示: 先 reshape 成 (B, S, H, D)，再 transpose/permute
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_6():
    B, S, H, D = 2, 10, 4, 8
    x = torch.randn(B, S, H * D)
    y = split_heads(x, H)
    assert y.shape == (B, H, S, D), f"shape 应为 {(B, H, S, D)}，得到 {y.shape}"
    # 验证数据一致性：第一个 batch、第一个位置的前 D 个元素应该是 head 0 的数据
    assert torch.allclose(y[0, 0, 0, :], x[0, 0, :D]), "数据排列不正确"
    print("  \u2713 Ex 6: split_heads — 拆分注意力头")


# ───────────────────────────────────────────────────────────────
# 练习 7: merge_heads (★★)
# ───────────────────────────────────────────────────────────────
def merge_heads(x):
    """
    把 (B, H, S, D) 的 tensor 合并成 (B, S, H*D)。
    这是 split_heads 的逆操作。

    参数:
      - x: shape 为 (B, H, S, D) 的 tensor
    返回:
      - y: shape 为 (B, S, H*D) 的 tensor

    提示: 先 transpose/permute 成 (B, S, H, D)，再 reshape/contiguous().view()
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_7():
    B, H, S, D = 2, 4, 10, 8
    x = torch.randn(B, H, S, D)
    y = merge_heads(x)
    assert y.shape == (B, S, H * D), f"shape 应为 {(B, S, H * D)}，得到 {y.shape}"
    # 验证可逆性
    x2 = split_heads(y, H)
    assert torch.allclose(x, x2), "merge_heads 应该是 split_heads 的逆操作"
    print("  \u2713 Ex 7: merge_heads — 合并注意力头")


# ───────────────────────────────────────────────────────────────
# 练习 8: move_to_device (★)
# ───────────────────────────────────────────────────────────────
def move_to_device(x, device, dtype):
    """
    把 tensor 移到指定的 device 和 dtype。

    参数:
      - x: 任意 tensor
      - device: 目标设备，如 'cpu'
      - dtype: 目标类型，如 torch.float16
    返回:
      - y: 移动后的 tensor

    提示: 使用 x.to(device=..., dtype=...)
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_8():
    x = torch.randn(3, 3)
    y = move_to_device(x, "cpu", torch.float16)
    assert y.device.type == "cpu", f"device 应为 cpu，得到 {y.device}"
    assert y.dtype == torch.float16, f"dtype 应为 float16，得到 {y.dtype}"
    print("  \u2713 Ex 8: move_to_device — 移动 tensor 到指定设备和类型")


# ═══════════════════════════════════════════════════════════════
#  Group 2: 激活函数 & 基础运算 (6 题)
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────
# 练习 9: my_relu (★)
# ───────────────────────────────────────────────────────────────
def my_relu(x):
    """
    手写 ReLU 激活函数：max(0, x)。

    参数:
      - x: 任意 shape 的 tensor
    返回:
      - y: 同 shape 的 tensor，所有负值变为 0

    提示: 可以用 torch.clamp(x, min=0) 或 x * (x > 0)
    注意: 不能直接调用 torch.relu 或 F.relu！
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_9():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    y = my_relu(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 3.0])
    assert torch.allclose(y, expected), f"结果不正确: {y}"
    print("  \u2713 Ex 9: my_relu — 手写 ReLU")


# ───────────────────────────────────────────────────────────────
# 练习 10: my_softmax (★)
# ───────────────────────────────────────────────────────────────
def my_softmax(x, dim=-1):
    """
    手写数值稳定的 softmax。

    步骤:
      1. 减去 max 值（数值稳定性）
      2. 取 exp
      3. 除以 sum 归一化

    参数:
      - x: 任意 shape 的 tensor
      - dim: 在哪个维度做 softmax
    返回:
      - y: softmax 结果，同 shape

    提示: x_stable = x - x.max(dim=dim, keepdim=True).values
    注意: 不能直接调用 F.softmax！
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_10():
    torch.manual_seed(42)
    x = torch.randn(2, 5)
    y = my_softmax(x, dim=-1)
    expected = F.softmax(x, dim=-1)
    assert torch.allclose(y, expected, atol=1e-6), "结果与 F.softmax 不一致"
    # 验证归一化
    assert torch.allclose(y.sum(dim=-1), torch.ones(2), atol=1e-6), "每行之和应为 1"
    print("  \u2713 Ex 10: my_softmax — 手写数值稳定 softmax")


# ───────────────────────────────────────────────────────────────
# 练习 11: my_gelu (★★)
# ───────────────────────────────────────────────────────────────
def my_gelu(x):
    """
    手写 GELU 近似公式：
      GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    参数:
      - x: 任意 shape 的 tensor
    返回:
      - y: GELU 激活后的结果

    提示: sqrt(2/pi) ≈ 0.7978845608
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_11():
    torch.manual_seed(42)
    x = torch.randn(3, 4)
    y = my_gelu(x)
    # PyTorch 的 GELU tanh 近似
    expected = F.gelu(x, approximate="tanh")
    assert torch.allclose(y, expected, atol=1e-5), "结果与 F.gelu(approximate='tanh') 不一致"
    print("  \u2713 Ex 11: my_gelu — 手写 GELU 近似")


# ───────────────────────────────────────────────────────────────
# 练习 12: my_silu (★)
# ───────────────────────────────────────────────────────────────
def my_silu(x):
    """
    手写 SiLU (Swish) 激活函数：x * sigmoid(x)。

    参数:
      - x: 任意 shape 的 tensor
    返回:
      - y: SiLU 激活后的结果

    提示: sigmoid(x) = torch.sigmoid(x) 或 1 / (1 + exp(-x))
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_12():
    torch.manual_seed(42)
    x = torch.randn(3, 4)
    y = my_silu(x)
    expected = F.silu(x)
    assert torch.allclose(y, expected, atol=1e-6), "结果与 F.silu 不一致"
    print("  \u2713 Ex 12: my_silu — 手写 SiLU")


# ───────────────────────────────────────────────────────────────
# 练习 13: swiglu_forward (★★)
# ───────────────────────────────────────────────────────────────
def swiglu_forward(gate_out, up_out):
    """
    计算 SwiGLU 激活：silu(gate_out) * up_out。
    这是现代 LLM (如 LLaMA) FFN 层的核心运算。

    参数:
      - gate_out: shape (B, S, D) 的 tensor，gate 投影的输出
      - up_out: shape (B, S, D) 的 tensor，up 投影的输出
    返回:
      - y: shape (B, S, D) 的 tensor

    提示: silu(x) = x * sigmoid(x)
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_13():
    torch.manual_seed(42)
    gate = torch.randn(2, 5, 8)
    up = torch.randn(2, 5, 8)
    y = swiglu_forward(gate, up)
    expected = F.silu(gate) * up
    assert y.shape == (2, 5, 8), f"shape 不正确: {y.shape}"
    assert torch.allclose(y, expected, atol=1e-6), "SwiGLU 结果不正确"
    print("  \u2713 Ex 13: swiglu_forward — SwiGLU 激活")


# ───────────────────────────────────────────────────────────────
# 练习 14: top_k_by_norm (★★)
# ───────────────────────────────────────────────────────────────
def top_k_by_norm(x, k):
    """
    给定 (seq, dim) 的 tensor，找出 L2 norm 最大的 k 个位置的索引。

    参数:
      - x: shape (S, D) 的 tensor
      - k: int，要选出的数量
    返回:
      - indices: shape (k,) 的 LongTensor，按 norm 从大到小排列

    提示: 先用 torch.norm(x, dim=-1) 计算每行的 L2 norm，再用 .topk(k)
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_14():
    torch.manual_seed(42)
    x = torch.randn(10, 8)
    k = 3
    indices = top_k_by_norm(x, k)
    assert indices.shape == (k,), f"shape 应为 ({k},)，得到 {indices.shape}"
    # 验证：这些确实是 norm 最大的 k 个
    norms = torch.norm(x, dim=-1)
    expected_indices = norms.topk(k).indices
    assert torch.equal(indices, expected_indices), "选出的索引不正确"
    print("  \u2713 Ex 14: top_k_by_norm — 找 L2 norm 最大的 k 个位置")


# ═══════════════════════════════════════════════════════════════
#  Group 3: Loss 函数 (6 题)
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────
# 练习 15: my_cross_entropy (★)
# ───────────────────────────────────────────────────────────────
def my_cross_entropy(logits, targets):
    """
    手写交叉熵 loss（数值稳定版本）。

    步骤:
      1. 计算 log_softmax：log_probs = logits - log(sum(exp(logits)))
         数值稳定: 先减去 max
      2. 取出正确类别的 log_prob
      3. 取负平均

    参数:
      - logits: shape (N, C) 的 tensor，N 个样本，C 个类别
      - targets: shape (N,) 的 LongTensor，每个样本的正确类别
    返回:
      - loss: 标量 tensor

    提示: 用 torch.log_softmax 或手动实现 log-sum-exp
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_15():
    torch.manual_seed(42)
    logits = torch.randn(4, 10)
    targets = torch.tensor([3, 7, 1, 0])
    loss = my_cross_entropy(logits, targets)
    expected = F.cross_entropy(logits, targets)
    assert torch.allclose(loss, expected, atol=1e-5), f"loss 不正确: {loss.item()} vs {expected.item()}"
    print("  \u2713 Ex 15: my_cross_entropy — 手写交叉熵 loss")


# ───────────────────────────────────────────────────────────────
# 练习 16: ce_with_ignore (★★)
# ───────────────────────────────────────────────────────────────
def ce_with_ignore(logits, targets, ignore_index=-100):
    """
    手写带 ignore_index 的交叉熵：target 等于 ignore_index 的位置不计算 loss。

    参数:
      - logits: shape (N, C) 的 tensor
      - targets: shape (N,) 的 LongTensor，可能包含 ignore_index
      - ignore_index: 要忽略的标签值，默认 -100
    返回:
      - loss: 标量 tensor（只对有效位置求平均）

    提示: 先做 mask = (targets != ignore_index)，然后只在有效位置计算
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_16():
    torch.manual_seed(42)
    logits = torch.randn(6, 5)
    targets = torch.tensor([2, -100, 1, 4, -100, 0])
    loss = ce_with_ignore(logits, targets, ignore_index=-100)
    expected = F.cross_entropy(logits, targets, ignore_index=-100)
    assert torch.allclose(loss, expected, atol=1e-5), f"loss 不正确: {loss.item()} vs {expected.item()}"
    print("  \u2713 Ex 16: ce_with_ignore — 带 ignore_index 的交叉熵")


# ───────────────────────────────────────────────────────────────
# 练习 17: focal_loss (★★)
# ───────────────────────────────────────────────────────────────
def focal_loss(logits, targets, gamma=2.0):
    """
    实现 Focal Loss：-(1-p_t)^gamma * log(p_t)
    其中 p_t 是正确类别的预测概率。

    Focal Loss 让模型更关注难分类的样本（p_t 小的样本权重更大）。

    参数:
      - logits: shape (N, C) 的 tensor
      - targets: shape (N,) 的 LongTensor
      - gamma: float，聚焦参数，默认 2.0
    返回:
      - loss: 标量 tensor（所有样本的平均）

    提示:
      1. 先算 log_probs = log_softmax(logits)
      2. 取出 p_t = exp(log_probs[targets])
      3. focal_weight = (1 - p_t) ** gamma
      4. loss = -focal_weight * log_probs[targets]
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_17():
    torch.manual_seed(42)
    logits = torch.randn(4, 5)
    targets = torch.tensor([0, 2, 1, 4])
    loss = focal_loss(logits, targets, gamma=2.0)
    # 手动验证
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    expected = (-((1 - pt) ** 2) * log_pt).mean()
    assert torch.allclose(loss, expected, atol=1e-5), f"focal loss 不正确: {loss.item()} vs {expected.item()}"
    print("  \u2713 Ex 17: focal_loss — Focal Loss")


# ───────────────────────────────────────────────────────────────
# 练习 18: label_smoothing_loss (★★)
# ───────────────────────────────────────────────────────────────
def label_smoothing_loss(logits, targets, epsilon=0.1):
    """
    实现 Label Smoothing 交叉熵。

    不使用 one-hot 硬标签，而是：
      - 正确类别的概率 = 1 - epsilon
      - 其他类别平分剩余概率 = epsilon / C

    等价公式：
      loss = (1 - epsilon) * CE(logits, targets) + epsilon * (-log_probs.mean(dim=-1)).mean()

    参数:
      - logits: shape (N, C) 的 tensor
      - targets: shape (N,) 的 LongTensor
      - epsilon: float，平滑系数，默认 0.1
    返回:
      - loss: 标量 tensor

    提示: 可以分成两项计算：正确类别项 + 均匀分布项
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_18():
    torch.manual_seed(42)
    logits = torch.randn(4, 10)
    targets = torch.tensor([3, 7, 1, 0])
    loss = label_smoothing_loss(logits, targets, epsilon=0.1)
    expected = F.cross_entropy(logits, targets, label_smoothing=0.1)
    assert torch.allclose(loss, expected, atol=1e-4), f"loss 不正确: {loss.item()} vs {expected.item()}"
    print("  \u2713 Ex 18: label_smoothing_loss — Label Smoothing 交叉熵")


# ───────────────────────────────────────────────────────────────
# 练习 19: contrastive_loss (★★)
# ───────────────────────────────────────────────────────────────
def contrastive_loss(queries, keys, temperature=0.07):
    """
    实现 InfoNCE 对比学习 loss。

    给定 queries (B, D) 和 keys (B, D)，对角线位置是正样本对。

    步骤:
      1. L2 归一化 queries 和 keys
      2. 计算相似度矩阵 sim = queries @ keys^T / temperature
      3. labels = [0, 1, 2, ..., B-1]（对角线是正样本）
      4. loss = CE(sim, labels)

    参数:
      - queries: shape (B, D) 的 tensor
      - keys: shape (B, D) 的 tensor
      - temperature: float，温度系数
    返回:
      - loss: 标量 tensor

    提示: 用 F.normalize(x, dim=-1) 做 L2 归一化
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_19():
    torch.manual_seed(42)
    B, D = 8, 32
    queries = torch.randn(B, D)
    keys = torch.randn(B, D)
    loss = contrastive_loss(queries, keys, temperature=0.07)
    # 手动验证
    q_norm = F.normalize(queries, dim=-1)
    k_norm = F.normalize(keys, dim=-1)
    sim = q_norm @ k_norm.T / 0.07
    labels = torch.arange(B)
    expected = F.cross_entropy(sim, labels)
    assert torch.allclose(loss, expected, atol=1e-5), f"loss 不正确: {loss.item()} vs {expected.item()}"
    print("  \u2713 Ex 19: contrastive_loss — InfoNCE 对比学习 loss")


# ───────────────────────────────────────────────────────────────
# 练习 20: moe_load_balance_loss (★★★)
# ───────────────────────────────────────────────────────────────
def moe_load_balance_loss(router_logits, top_k=2):
    """
    实现 MoE (Mixture of Experts) 负载均衡 loss。

    目标：鼓励 token 均匀分配到各个 expert，避免负载不均。

    步骤:
      1. 计算 routing 概率: probs = softmax(router_logits, dim=-1)  # (T, E)
      2. 选出 top_k expert: indices = topk(router_logits, k=top_k).indices  # (T, top_k)
      3. 计算每个 expert 的实际分配频率 f_i:
         - 统计每个 expert 被选中多少次，除以 (T * top_k)
      4. 计算每个 expert 的平均 routing 概率 p_i:
         - p_i = probs[:, i].mean()  对每个 expert 在所有 token 上的平均
      5. loss = E * sum(f_i * p_i)  其中 E 是 expert 数量

    参数:
      - router_logits: shape (T, E) 的 tensor，T 个 token，E 个 expert
      - top_k: int，每个 token 选几个 expert
    返回:
      - loss: 标量 tensor

    提示: 用 torch.zeros(E).scatter_add_ 或 one_hot 统计频率
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_20():
    torch.manual_seed(42)
    T, E = 32, 8
    router_logits = torch.randn(T, E)
    loss = moe_load_balance_loss(router_logits, top_k=2)
    # 手动验证
    probs = F.softmax(router_logits, dim=-1)
    indices = router_logits.topk(2, dim=-1).indices
    mask = torch.zeros(T, E)
    mask.scatter_(1, indices, 1.0)
    f = mask.mean(dim=0)
    p = probs.mean(dim=0)
    expected = E * (f * p).sum()
    assert torch.allclose(loss, expected, atol=1e-5), f"loss 不正确: {loss.item()} vs {expected.item()}"
    assert loss.item() > 0, "loss 应该是正数"
    print("  \u2713 Ex 20: moe_load_balance_loss — MoE 负载均衡 loss")


# ═══════════════════════════════════════════════════════════════
#  Group 4: nn.Module 构建 (6 题)
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────
# 练习 21: SimpleLinear (★)
# ───────────────────────────────────────────────────────────────
class SimpleLinear(nn.Module):
    """
    手写线性层 nn.Module。

    等效于 nn.Linear(in_features, out_features)。
    forward: y = x @ W^T + b

    参数:
      - in_features: int，输入维度
      - out_features: int，输出维度

    要求:
      - 用 nn.Parameter 存储 weight (out_features, in_features) 和 bias (out_features,)
      - weight 初始化为随机正态分布
      - bias 初始化为零

    提示: self.weight = nn.Parameter(torch.randn(out_features, in_features))
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        # YOUR CODE HERE
        raise NotImplementedError("请实现 __init__")

    def forward(self, x):
        # YOUR CODE HERE
        raise NotImplementedError("请实现 forward")


def _check_21():
    torch.manual_seed(42)
    layer = SimpleLinear(8, 4)
    assert hasattr(layer, "weight"), "需要定义 weight 参数"
    assert hasattr(layer, "bias"), "需要定义 bias 参数"
    assert layer.weight.shape == (4, 8), f"weight shape 应为 (4, 8)，得到 {layer.weight.shape}"
    assert layer.bias.shape == (4,), f"bias shape 应为 (4,)，得到 {layer.bias.shape}"
    x = torch.randn(2, 8)
    y = layer(x)
    assert y.shape == (2, 4), f"输出 shape 应为 (2, 4)，得到 {y.shape}"
    expected = x @ layer.weight.T + layer.bias
    assert torch.allclose(y, expected, atol=1e-6), "forward 计算不正确"
    print("  \u2713 Ex 21: SimpleLinear — 手写线性层")


# ───────────────────────────────────────────────────────────────
# 练习 22: MyLayerNorm (★★)
# ───────────────────────────────────────────────────────────────
class MyLayerNorm(nn.Module):
    """
    手写 Layer Normalization。

    步骤:
      1. 计算最后一维的 mean 和 var
      2. 归一化: x_norm = (x - mean) / sqrt(var + eps)
      3. 缩放和偏移: y = gamma * x_norm + beta

    参数:
      - dim: int，归一化的维度大小（最后一维）
      - eps: float，防止除零，默认 1e-5

    要求:
      - gamma (weight): nn.Parameter，shape (dim,)，初始化为 1
      - beta (bias): nn.Parameter，shape (dim,)，初始化为 0

    提示: mean 和 var 都在 dim=-1 上计算，keepdim=True
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        # YOUR CODE HERE
        raise NotImplementedError("请实现 __init__")

    def forward(self, x):
        # YOUR CODE HERE
        raise NotImplementedError("请实现 forward")


def _check_22():
    torch.manual_seed(42)
    dim = 64
    my_ln = MyLayerNorm(dim)
    ref_ln = nn.LayerNorm(dim)
    # 同步参数
    with torch.no_grad():
        my_ln.weight.copy_(ref_ln.weight)
        my_ln.bias.copy_(ref_ln.bias)
    x = torch.randn(2, 10, dim)
    y = my_ln(x)
    expected = ref_ln(x)
    assert y.shape == expected.shape, f"shape 不匹配: {y.shape} vs {expected.shape}"
    assert torch.allclose(y, expected, atol=1e-5), "LayerNorm 结果与官方实现不一致"
    print("  \u2713 Ex 22: MyLayerNorm — 手写 LayerNorm")


# ───────────────────────────────────────────────────────────────
# 练习 23: MyRMSNorm (★★)
# ───────────────────────────────────────────────────────────────
class MyRMSNorm(nn.Module):
    """
    手写 RMSNorm (Root Mean Square Layer Normalization)。

    公式: y = x / sqrt(mean(x^2) + eps) * weight

    与 LayerNorm 的区别：不减均值，只做尺度归一化。

    参数:
      - dim: int，归一化的维度大小
      - eps: float，默认 1e-6

    要求:
      - weight: nn.Parameter，shape (dim,)，初始化为 1

    提示: rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        # YOUR CODE HERE
        raise NotImplementedError("请实现 __init__")

    def forward(self, x):
        # YOUR CODE HERE
        raise NotImplementedError("请实现 forward")


def _check_23():
    torch.manual_seed(42)
    dim = 64
    rms = MyRMSNorm(dim)
    x = torch.randn(2, 10, dim)
    y = rms(x)
    assert y.shape == x.shape, f"shape 不匹配: {y.shape} vs {x.shape}"
    # 手动计算验证
    rms_val = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = x / rms_val * rms.weight
    assert torch.allclose(y, expected, atol=1e-5), "RMSNorm 结果不正确"
    print("  \u2713 Ex 23: MyRMSNorm — 手写 RMSNorm")


# ───────────────────────────────────────────────────────────────
# 练习 24: MyDropout (★★)
# ───────────────────────────────────────────────────────────────
class MyDropout(nn.Module):
    """
    手写 Dropout 模块。

    训练模式 (self.training == True):
      1. 以概率 p 随机将元素置零
      2. 存活的元素缩放 1/(1-p)（保持期望不变）
    评估模式:
      直接返回输入（不做任何修改）

    参数:
      - p: float，dropout 概率，默认 0.1

    提示: mask = torch.bernoulli(torch.full_like(x, 1-p))
          output = x * mask / (1 - p)
    """

    def __init__(self, p=0.1):
        super().__init__()
        # YOUR CODE HERE
        raise NotImplementedError("请实现 __init__")

    def forward(self, x):
        # YOUR CODE HERE
        raise NotImplementedError("请实现 forward")


def _check_24():
    torch.manual_seed(42)
    drop = MyDropout(p=0.5)
    x = torch.ones(1000)
    # 评估模式
    drop.eval()
    y_eval = drop(x)
    assert torch.equal(y_eval, x), "eval 模式下应原样返回"
    # 训练模式
    drop.train()
    y_train = drop(x)
    # 检查大约一半为零
    zero_ratio = (y_train == 0).float().mean().item()
    assert 0.3 < zero_ratio < 0.7, f"大约一半应为零，实际零比例 {zero_ratio:.2f}"
    # 检查非零值的缩放
    nonzero_vals = y_train[y_train > 0]
    assert torch.allclose(nonzero_vals, torch.full_like(nonzero_vals, 2.0), atol=0.01), \
        "存活元素应被缩放为 1/(1-p) = 2.0"
    print("  \u2713 Ex 24: MyDropout — 手写 Dropout")


# ───────────────────────────────────────────────────────────────
# 练习 25: MyEmbedding (★★)
# ───────────────────────────────────────────────────────────────
class MyEmbedding(nn.Module):
    """
    手写 Embedding 层。

    用 nn.Parameter 存储一个 (vocab_size, embed_dim) 的权重矩阵。
    forward 时，根据输入的整数索引查表返回对应的 embedding 向量。

    参数:
      - vocab_size: int，词表大小
      - embed_dim: int，embedding 维度

    要求:
      - weight: nn.Parameter，shape (vocab_size, embed_dim)，正态分布初始化

    提示: forward 就是 self.weight[input_ids]，即简单的 indexing
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # YOUR CODE HERE
        raise NotImplementedError("请实现 __init__")

    def forward(self, input_ids):
        # YOUR CODE HERE
        raise NotImplementedError("请实现 forward")


def _check_25():
    torch.manual_seed(42)
    emb = MyEmbedding(100, 32)
    ids = torch.tensor([0, 5, 99, 42])
    y = emb(ids)
    assert y.shape == (4, 32), f"shape 应为 (4, 32)，得到 {y.shape}"
    assert torch.equal(y[0], emb.weight[0]), "索引 0 的 embedding 不正确"
    assert torch.equal(y[2], emb.weight[99]), "索引 99 的 embedding 不正确"
    print("  \u2713 Ex 25: MyEmbedding — 手写 Embedding")


# ───────────────────────────────────────────────────────────────
# 练习 26: LoRALinear (★★★)
# ───────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """
    手写 LoRA (Low-Rank Adaptation) 线性层。

    LoRA 原理:
      - 冻结原始权重 W
      - 添加低秩分解 A, B
      - forward: y = W(x) + scale * x @ A^T @ B^T

    参数:
      - base_linear: nn.Linear，原始的（冻结的）线性层
      - rank: int，LoRA 的秩
      - scale: float，缩放系数，默认 1.0

    要求:
      - 冻结 base_linear 的参数 (requires_grad = False)
      - A: nn.Parameter，shape (rank, in_features)，正态分布初始化
      - B: nn.Parameter，shape (out_features, rank)，零初始化
      - forward: base_linear(x) + scale * (x @ A^T @ B^T)

    提示:
      - B 初始化为零确保训练开始时 LoRA 不改变原始行为
      - A 用较小的标准差初始化，如 0.01
    """

    def __init__(self, base_linear, rank, scale=1.0):
        super().__init__()
        # YOUR CODE HERE
        raise NotImplementedError("请实现 __init__")

    def forward(self, x):
        # YOUR CODE HERE
        raise NotImplementedError("请实现 forward")


def _check_26():
    torch.manual_seed(42)
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, rank=4, scale=1.0)
    # 检查 base 被冻结
    for p in base.parameters():
        assert not p.requires_grad, "base_linear 的参数应被冻结"
    # 检查 A, B 形状
    assert lora.A.shape == (4, 16), f"A shape 应为 (4, 16)，得到 {lora.A.shape}"
    assert lora.B.shape == (8, 4), f"B shape 应为 (8, 4)，得到 {lora.B.shape}"
    # B 初始化为零时，LoRA 输出应等于 base 输出
    with torch.no_grad():
        lora.B.zero_()
    x = torch.randn(2, 16)
    y_lora = lora(x)
    y_base = base(x)
    assert torch.allclose(y_lora, y_base, atol=1e-6), "B=0 时 LoRA 应与 base 输出一致"
    # 设置 B 非零后应有差异
    with torch.no_grad():
        lora.B.fill_(0.1)
    y_lora2 = lora(x)
    assert not torch.allclose(y_lora2, y_base, atol=1e-3), "B 非零时应与 base 不同"
    print("  \u2713 Ex 26: LoRALinear — 手写 LoRA")


# ═══════════════════════════════════════════════════════════════
#  Group 5: Attention 机制 (6 题)
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────
# 练习 27: scaled_dot_product_attention (★★)
# ───────────────────────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V):
    """
    实现缩放点积注意力:
      Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    参数:
      - Q: shape (B, S, D) 的 tensor，查询
      - K: shape (B, S, D) 的 tensor，键
      - V: shape (B, S, D) 的 tensor，值
    返回:
      - output: shape (B, S, D) 的 tensor
      - weights: shape (B, S, S) 的 tensor，注意力权重

    提示:
      - d_k = Q.shape[-1]
      - scores = Q @ K.transpose(-1, -2) / sqrt(d_k)
      - weights = softmax(scores, dim=-1)
      - output = weights @ V
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_27():
    torch.manual_seed(42)
    B, S, D = 2, 5, 8
    Q = torch.randn(B, S, D)
    K = torch.randn(B, S, D)
    V = torch.randn(B, S, D)
    output, weights = scaled_dot_product_attention(Q, K, V)
    assert output.shape == (B, S, D), f"output shape 不正确: {output.shape}"
    assert weights.shape == (B, S, S), f"weights shape 不正确: {weights.shape}"
    # 权重应该归一化
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, S), atol=1e-5), "注意力权重每行之和应为 1"
    # 验证计算
    d_k = D
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
    expected_w = F.softmax(scores, dim=-1)
    expected_o = expected_w @ V
    assert torch.allclose(output, expected_o, atol=1e-5), "注意力输出不正确"
    print("  \u2713 Ex 27: scaled_dot_product_attention — 缩放点积注意力")


# ───────────────────────────────────────────────────────────────
# 练习 28: causal_mask (★★)
# ───────────────────────────────────────────────────────────────
def causal_mask(seq_len):
    """
    生成 (S, S) 的因果掩码（用于自回归模型）。

    上三角为 True（需要屏蔽的位置），对角线和下三角为 False。
    即：位置 i 只能看到 <= i 的位置。

    参数:
      - seq_len: int，序列长度
    返回:
      - mask: shape (S, S) 的 bool tensor，True 表示需要屏蔽

    提示: torch.triu(torch.ones(S, S), diagonal=1).bool()
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_28():
    mask = causal_mask(4)
    assert mask.shape == (4, 4), f"shape 应为 (4, 4)，得到 {mask.shape}"
    assert mask.dtype == torch.bool, f"dtype 应为 bool，得到 {mask.dtype}"
    expected = torch.tensor([
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ])
    assert torch.equal(mask, expected), f"因果掩码不正确:\n{mask}"
    print("  \u2713 Ex 28: causal_mask — 因果掩码")


# ───────────────────────────────────────────────────────────────
# 练习 29: masked_attention (★★)
# ───────────────────────────────────────────────────────────────
def masked_attention(Q, K, V, mask):
    """
    在注意力中应用因果掩码。

    步骤:
      1. 计算 scores = Q @ K^T / sqrt(d_k)
      2. 在 mask 为 True 的位置填充 -inf
      3. softmax
      4. 乘以 V

    参数:
      - Q: shape (B, S, D)
      - K: shape (B, S, D)
      - V: shape (B, S, D)
      - mask: shape (S, S) 的 bool tensor，True 表示需要屏蔽
    返回:
      - output: shape (B, S, D)

    提示: scores.masked_fill_(mask.unsqueeze(0), float('-inf'))
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_29():
    torch.manual_seed(42)
    B, S, D = 2, 4, 8
    Q = torch.randn(B, S, D)
    K = torch.randn(B, S, D)
    V = torch.randn(B, S, D)
    mask = causal_mask(S)
    output = masked_attention(Q, K, V, mask)
    assert output.shape == (B, S, D), f"shape 不正确: {output.shape}"
    # 第一个位置应该只关注自己（因为其他位置被 mask 了）
    # 验证因果性：修改未来位置的 V 不应影响当前位置的输出
    V2 = V.clone()
    V2[:, -1, :] = 999.0  # 修改最后一个位置
    output2 = masked_attention(Q, K, V2, mask)
    # 前 S-1 个位置的输出应该不变
    assert torch.allclose(output[:, :-1, :], output2[:, :-1, :], atol=1e-5), \
        "因果掩码没有正确阻止未来信息泄露"
    print("  \u2713 Ex 29: masked_attention — 带因果掩码的注意力")


# ───────────────────────────────────────────────────────────────
# 练习 30: multi_head_attention (★★★)
# ───────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    完整的多头注意力模块。

    步骤:
      1. 投影: Q, K, V = W_q(x), W_k(x), W_v(x)
      2. 拆分多头: (B, S, D) -> (B, H, S, D//H)
      3. 计算 scaled dot product attention
      4. 合并多头: (B, H, S, D//H) -> (B, S, D)
      5. 输出投影: output = W_o(merged)

    参数:
      - dim: int，模型维度 D
      - num_heads: int，注意力头数 H

    要求:
      - W_q, W_k, W_v, W_o: 四个 nn.Linear(dim, dim)
      - 假设 dim 能被 num_heads 整除
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        # YOUR CODE HERE
        raise NotImplementedError("请实现 __init__")

    def forward(self, x):
        """
        参数:
          - x: shape (B, S, D)
        返回:
          - output: shape (B, S, D)
        """
        # YOUR CODE HERE
        raise NotImplementedError("请实现 forward")


def _check_30():
    torch.manual_seed(42)
    B, S, D, H = 2, 10, 64, 8
    mha = MultiHeadAttention(dim=D, num_heads=H)
    x = torch.randn(B, S, D)
    y = mha(x)
    assert y.shape == (B, S, D), f"输出 shape 应为 {(B, S, D)}，得到 {y.shape}"
    # 检查参数数量（4 个线性层，每个 D*D + D）
    total_params = sum(p.numel() for p in mha.parameters())
    expected_params = 4 * (D * D + D)
    assert total_params == expected_params, f"参数数量不正确: {total_params} vs {expected_params}"
    print("  \u2713 Ex 30: MultiHeadAttention — 完整多头注意力")


# ───────────────────────────────────────────────────────────────
# 练习 31: rope_frequencies (★★)
# ───────────────────────────────────────────────────────────────
def rope_frequencies(dim, seq_len, base=10000.0):
    """
    计算 RoPE (Rotary Position Embedding) 的频率。

    公式: freq_i = 1 / (base ^ (2i / dim))  for i = 0, 1, ..., dim//2 - 1

    然后对每个位置 pos = 0, 1, ..., seq_len-1:
      angles[pos, i] = pos * freq_i

    参数:
      - dim: int，向量维度（必须是偶数）
      - seq_len: int，序列长度
      - base: float，基底频率，默认 10000
    返回:
      - freqs: shape (seq_len, dim//2) 的 tensor，每个位置的角度

    提示:
      - inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
      - positions = torch.arange(seq_len).float()
      - freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_31():
    freqs = rope_frequencies(dim=8, seq_len=4, base=10000.0)
    assert freqs.shape == (4, 4), f"shape 应为 (4, 4)，得到 {freqs.shape}"
    # 位置 0 的角度应全为 0
    assert torch.allclose(freqs[0], torch.zeros(4)), "位置 0 的角度应全为 0"
    # 验证频率递减
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
    expected = torch.arange(4).float().unsqueeze(1) * inv_freq.unsqueeze(0)
    assert torch.allclose(freqs, expected, atol=1e-5), "频率计算不正确"
    print("  \u2713 Ex 31: rope_frequencies — RoPE 频率计算")


# ───────────────────────────────────────────────────────────────
# 练习 32: apply_rope (★★★)
# ───────────────────────────────────────────────────────────────
def apply_rope(x, freqs):
    """
    对 tensor 应用 RoPE 旋转。

    RoPE 把相邻的两个维度看作一个复数，进行旋转:
      [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]

    步骤:
      1. 把 x 的最后一维拆成 (..., dim//2, 2)
      2. 分别取出奇偶位: x_even = x[..., 0], x_odd = x[..., 1]
      3. cos_f = cos(freqs), sin_f = sin(freqs)
      4. 旋转:
         y_even = x_even * cos_f - x_odd * sin_f
         y_odd  = x_even * sin_f + x_odd * cos_f
      5. 交织合并回原来的维度

    参数:
      - x: shape (B, S, D) 的 tensor，D 必须是偶数
      - freqs: shape (S, D//2) 的 tensor，角度
    返回:
      - y: shape (B, S, D) 的 tensor，旋转后的结果

    提示: 用 torch.stack([y_even, y_odd], dim=-1).flatten(-2)
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_32():
    torch.manual_seed(42)
    B, S, D = 2, 4, 8
    x = torch.randn(B, S, D)
    freqs = rope_frequencies(D, S)
    y = apply_rope(x, freqs)
    assert y.shape == (B, S, D), f"shape 不正确: {y.shape}"
    # 位置 0 的角度为 0，旋转后应不变
    assert torch.allclose(y[:, 0, :], x[:, 0, :], atol=1e-5), "位置 0 应该不变（角度为 0）"
    # 旋转应该保持 L2 范数不变（旋转是正交变换）
    x_norm = torch.norm(x, dim=-1)
    y_norm = torch.norm(y, dim=-1)
    assert torch.allclose(x_norm, y_norm, atol=1e-4), "RoPE 旋转应保持 L2 范数不变"
    print("  \u2713 Ex 32: apply_rope — 应用 RoPE 旋转")


# ═══════════════════════════════════════════════════════════════
#  Group 6: 训练循环 (4 题)
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────
# 练习 33: training_step (★★)
# ───────────────────────────────────────────────────────────────
def training_step(model, optimizer, x, targets, max_grad_norm=1.0):
    """
    实现一步完整的训练过程。

    步骤:
      1. 前向传播: logits = model(x)
      2. 计算 loss: loss = CE(logits, targets)
      3. 反向传播: loss.backward()
      4. 梯度裁剪: clip_grad_norm_(model.parameters(), max_grad_norm)
      5. 更新参数: optimizer.step()
      6. 清零梯度: optimizer.zero_grad()

    参数:
      - model: nn.Module，模型
      - optimizer: torch.optim.Optimizer，优化器
      - x: 输入 tensor
      - targets: 目标标签
      - max_grad_norm: float，梯度裁剪阈值
    返回:
      - loss_value: float，这一步的 loss 值

    提示: 用 torch.nn.utils.clip_grad_norm_ 做梯度裁剪
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_33():
    torch.manual_seed(42)
    model = nn.Linear(8, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(2, 8)
    targets = torch.tensor([1, 3])
    # 记录初始权重
    w_before = model.weight.clone()
    loss_val = training_step(model, optimizer, x, targets)
    # 检查 loss 是合理的正数
    assert isinstance(loss_val, float), "返回值应为 float"
    assert loss_val > 0, "loss 应为正数"
    # 检查权重被更新了
    assert not torch.equal(model.weight, w_before), "权重应该被更新"
    # 检查梯度已清零
    assert model.weight.grad is None or torch.allclose(
        model.weight.grad, torch.zeros_like(model.weight.grad)
    ), "梯度应被清零"
    print("  \u2713 Ex 33: training_step — 单步训练")


# ───────────────────────────────────────────────────────────────
# 练习 34: gradient_accumulation_step (★★)
# ───────────────────────────────────────────────────────────────
def gradient_accumulation_step(model, optimizer, micro_batches, targets_list, accum_steps):
    """
    实现梯度累积训练。

    当显存不够用时，可以把大 batch 拆成多个 micro batch，
    累积梯度后再一次性更新。

    步骤:
      1. optimizer.zero_grad()
      2. 对每个 micro_batch:
         a. logits = model(micro_batch)
         b. loss = CE(logits, targets) / accum_steps  # 注意要除以累积步数！
         c. loss.backward()  # 梯度会累积
      3. optimizer.step()

    参数:
      - model: nn.Module
      - optimizer: Optimizer
      - micro_batches: list of tensor，多个小 batch 的输入
      - targets_list: list of tensor，对应的标签
      - accum_steps: int，累积步数（应等于 len(micro_batches)）
    返回:
      - total_loss: float，所有 micro batch 的平均 loss

    提示: 关键是 loss / accum_steps，这样累积后的梯度等效于大 batch 的梯度
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_34():
    torch.manual_seed(42)
    model = nn.Linear(8, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # 创建 4 个 micro batch
    micro_batches = [torch.randn(2, 8) for _ in range(4)]
    targets_list = [torch.randint(0, 4, (2,)) for _ in range(4)]
    w_before = model.weight.clone()
    total_loss = gradient_accumulation_step(model, optimizer, micro_batches, targets_list, accum_steps=4)
    assert isinstance(total_loss, float), "返回值应为 float"
    assert total_loss > 0, "loss 应为正数"
    assert not torch.equal(model.weight, w_before), "权重应该被更新"
    print("  \u2713 Ex 34: gradient_accumulation_step — 梯度累积")


# ───────────────────────────────────────────────────────────────
# 练习 35: cosine_warmup_schedule (★★)
# ───────────────────────────────────────────────────────────────
def cosine_warmup_schedule(step, warmup_steps, total_steps, min_lr_ratio=0.1):
    """
    实现 cosine warmup 学习率调度。

    两个阶段:
      1. Warmup (step < warmup_steps):
         lr_mult = step / warmup_steps （线性增长）
      2. Cosine decay (step >= warmup_steps):
         progress = (step - warmup_steps) / (total_steps - warmup_steps)
         lr_mult = min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + cos(pi * progress))

    参数:
      - step: int，当前步数
      - warmup_steps: int，warmup 步数
      - total_steps: int，总步数
      - min_lr_ratio: float，最终 lr 与初始 lr 的比值
    返回:
      - lr_mult: float，学习率乘数 (0 到 1 之间)

    提示: 用 math.cos 和 math.pi
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_35():
    # 测试 warmup 阶段
    assert cosine_warmup_schedule(0, 100, 1000) == 0.0, "step=0 时 lr 应为 0"
    assert abs(cosine_warmup_schedule(50, 100, 1000) - 0.5) < 1e-6, "step=50 时 lr 应为 0.5"
    assert abs(cosine_warmup_schedule(100, 100, 1000) - 1.0) < 1e-6, "step=100 时 lr 应为 1.0"
    # 测试 cosine decay 阶段
    lr_end = cosine_warmup_schedule(1000, 100, 1000, min_lr_ratio=0.1)
    assert abs(lr_end - 0.1) < 1e-6, f"最终 lr 应为 min_lr_ratio=0.1，得到 {lr_end}"
    # 单调递减
    lrs = [cosine_warmup_schedule(s, 100, 1000) for s in range(100, 1001)]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-6, "cosine decay 阶段 lr 应单调递减"
    print("  \u2713 Ex 35: cosine_warmup_schedule — Cosine warmup 调度")


# ───────────────────────────────────────────────────────────────
# 练习 36: freeze_and_count (★★)
# ───────────────────────────────────────────────────────────────
def freeze_and_count(model, trainable_names):
    """
    冻结模型所有参数，然后解冻指定名称的参数，返回可训练参数数量。

    步骤:
      1. 冻结所有参数: param.requires_grad = False
      2. 遍历 model.named_parameters()，如果名称包含 trainable_names 中的任意一个，
         则解冻 (requires_grad = True)
      3. 统计并返回可训练参数总数

    参数:
      - model: nn.Module
      - trainable_names: list of str，要解冻的参数名称（子字符串匹配）
    返回:
      - count: int，可训练参数的总数量

    提示: 用 any(name in param_name for name in trainable_names) 做子字符串匹配
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_36():
    model = nn.Sequential(
        nn.Linear(8, 16),   # weight: 128, bias: 16 => 144
        nn.ReLU(),
        nn.Linear(16, 4),   # weight: 64, bias: 4 => 68
    )
    count = freeze_and_count(model, trainable_names=["1."])  # 解冻第二个 Linear (索引为 2)
    # "1." 匹配 nn.ReLU，没有参数
    # 所有参数都应该被冻结
    # 重新测试：解冻 "2."
    count = freeze_and_count(model, trainable_names=["2."])
    assert count == 68, f"可训练参数数量应为 68，得到 {count}"
    # 验证第一层确实被冻结
    assert not model[0].weight.requires_grad, "第一层应被冻结"
    assert model[2].weight.requires_grad, "指定层应被解冻"
    print("  \u2713 Ex 36: freeze_and_count — 冻结和解冻参数")


# ═══════════════════════════════════════════════════════════════
#  Group 7: 模型保存 & Hook (4 题)
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────
# 练习 37: save_trainable_only (★★)
# ───────────────────────────────────────────────────────────────
def save_trainable_only(model):
    """
    只保存 requires_grad=True 的参数到字典。

    在 LoRA/PEFT 场景中，只需要保存可训练的低秩矩阵，不需要保存冻结的原始权重。

    参数:
      - model: nn.Module
    返回:
      - state_dict: dict，只包含可训练参数的 {name: tensor} 字典

    提示: 遍历 model.named_parameters()，检查 p.requires_grad
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_37():
    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.Linear(4, 2),
    )
    # 冻结第一层
    for p in model[0].parameters():
        p.requires_grad = False
    sd = save_trainable_only(model)
    assert isinstance(sd, dict), "返回值应为字典"
    assert len(sd) == 2, f"应只有 2 个可训练参数（第二层的 weight 和 bias），得到 {len(sd)}"
    assert "1.weight" in sd, "应包含 '1.weight'"
    assert "1.bias" in sd, "应包含 '1.bias'"
    assert "0.weight" not in sd, "不应包含冻结的 '0.weight'"
    print("  \u2713 Ex 37: save_trainable_only — 只保存可训练参数")


# ───────────────────────────────────────────────────────────────
# 练习 38: load_partial (★★)
# ───────────────────────────────────────────────────────────────
def load_partial(model, partial_state_dict):
    """
    用 strict=False 加载部分参数到模型。

    当 state_dict 只包含部分参数时（如 LoRA 权重），
    需要用 strict=False 来避免报错。

    参数:
      - model: nn.Module
      - partial_state_dict: dict，部分参数字典
    返回:
      - missing: list of str，模型中有但 state_dict 中缺少的参数名
      - unexpected: list of str，state_dict 中有但模型中没有的参数名

    提示: result = model.load_state_dict(partial_state_dict, strict=False)
          返回 result.missing_keys 和 result.unexpected_keys
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_38():
    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.Linear(4, 2),
    )
    # 只加载第二层
    partial_sd = {"1.weight": torch.randn(2, 4), "1.bias": torch.zeros(2)}
    missing, unexpected = load_partial(model, partial_sd)
    assert "0.weight" in missing, "'0.weight' 应在 missing 列表中"
    assert "0.bias" in missing, "'0.bias' 应在 missing 列表中"
    assert len(unexpected) == 0, f"不应有 unexpected keys，得到 {unexpected}"
    # 验证第二层确实被更新了
    assert torch.allclose(model[1].bias, torch.zeros(2)), "第二层 bias 应被更新为 0"
    print("  \u2713 Ex 38: load_partial — 部分加载参数")


# ───────────────────────────────────────────────────────────────
# 练习 39: register_intervention_hook (★★★)
# ───────────────────────────────────────────────────────────────
def register_intervention_hook(model, layer_name, delta):
    """
    注册一个 forward hook，在指定层的输出上加一个 delta 向量。

    这模拟了 Representation Engineering / MoReS 中的干预操作：
    在模型某一层的隐藏状态上施加方向性偏移。

    参数:
      - model: nn.Module
      - layer_name: str，目标层的名称（支持 model.named_modules() 中的名称）
      - delta: tensor，要加的偏移向量
    返回:
      - hook_handle: RemovableHook handle（用于后续移除 hook）

    提示:
      - 用 dict(model.named_modules())[layer_name] 获取目标层
      - hook 函数签名: hook(module, input, output) -> modified_output
      - 返回 output + delta
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_39():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    x = torch.randn(1, 8)
    # 没有 hook 的输出
    y_original = model(x).clone()
    # 在第一个 Linear 层后加 delta
    delta = torch.ones(1, 4) * 0.5
    handle = register_intervention_hook(model, "0", delta)
    y_hooked = model(x)
    # 输出应该不同
    assert not torch.allclose(y_original, y_hooked, atol=1e-3), "加了 hook 后输出应该不同"
    # 移除 hook
    handle.remove()
    y_after_remove = model(x)
    assert torch.allclose(y_original, y_after_remove, atol=1e-6), "移除 hook 后输出应恢复"
    print("  \u2713 Ex 39: register_intervention_hook — 注册干预 hook")


# ───────────────────────────────────────────────────────────────
# 练习 40: orthogonal_intervention (★★★)
# ───────────────────────────────────────────────────────────────
def orthogonal_intervention(h, R, W):
    """
    实现 MoReS 正交干预:
      h' = h + R^T @ (W @ h_flat - R @ h_flat)

    这里:
      - h 是原始隐藏状态 (B, D)
      - R 是正交旋转矩阵 (D, D)
      - W 是干预权重矩阵 (D, D)
      - h_flat = h  (已经是 2D)

    干预含义：
      1. Rh: 把 h 旋转到干预子空间
      2. Wh: 在干预子空间中修改 h
      3. Wh - Rh: 干预差值
      4. R^T(Wh - Rh): 旋转回原空间
      5. h + R^T(Wh - Rh): 加回原始状态

    同时返回正交惩罚项: ||R^T @ R - I||_F^2

    参数:
      - h: shape (B, D) 的 tensor，隐藏状态
      - R: shape (D, D) 的 tensor，旋转矩阵
      - W: shape (D, D) 的 tensor，干预矩阵
    返回:
      - h_prime: shape (B, D) 的 tensor，干预后的隐藏状态
      - ortho_penalty: 标量 tensor，正交惩罚

    提示:
      - h_prime = h + (W @ h.T - R @ h.T).T  然后左乘 R^T
      - 更简洁: h_prime = h + ((W - R) @ h.T).T  再通过 R^T
      - ortho_penalty = ||(R^T @ R) - I||_F^2 = ((R^T@R - I)**2).sum()
    """
    # YOUR CODE HERE
    raise NotImplementedError("请实现这个函数")


def _check_40():
    torch.manual_seed(42)
    B, D = 4, 16
    h = torch.randn(B, D)
    # 创建一个近正交矩阵
    Q, _ = torch.linalg.qr(torch.randn(D, D))
    R = Q  # 正交矩阵
    W = torch.randn(D, D) * 0.1 + R  # 接近 R 的矩阵
    h_prime, ortho_penalty = orthogonal_intervention(h, R, W)
    assert h_prime.shape == (B, D), f"h_prime shape 不正确: {h_prime.shape}"
    # 验证干预公式
    diff = W @ h.T - R @ h.T  # (D, B)
    expected = h + (R.T @ diff).T   # (B, D)
    assert torch.allclose(h_prime, expected, atol=1e-4), "干预结果不正确"
    # 对于正交矩阵 R，惩罚应接近 0
    assert ortho_penalty.item() < 1e-4, f"正交矩阵的惩罚应接近 0，得到 {ortho_penalty.item()}"
    # 对于非正交矩阵，惩罚应更大
    R_bad = torch.randn(D, D)
    _, penalty_bad = orthogonal_intervention(h, R_bad, W)
    assert penalty_bad.item() > 1.0, "非正交矩阵的惩罚应较大"
    print("  \u2713 Ex 40: orthogonal_intervention — MoReS 正交干预")


# ═══════════════════════════════════════════════════════════════
#  运行所有检查
# ═══════════════════════════════════════════════════════════════

def run_all():
    """运行所有习题检查，打印得分"""
    checks = [
        _check_1, _check_2, _check_3, _check_4, _check_5,
        _check_6, _check_7, _check_8, _check_9, _check_10,
        _check_11, _check_12, _check_13, _check_14, _check_15,
        _check_16, _check_17, _check_18, _check_19, _check_20,
        _check_21, _check_22, _check_23, _check_24, _check_25,
        _check_26, _check_27, _check_28, _check_29, _check_30,
        _check_31, _check_32, _check_33, _check_34, _check_35,
        _check_36, _check_37, _check_38, _check_39, _check_40,
    ]
    passed = 0
    failed = 0
    errors = []
    print("=" * 60)
    print("  PyTorch 基础练习 — 自动评分")
    print("=" * 60)
    for i, check in enumerate(checks, 1):
        try:
            check()
            passed += 1
        except NotImplementedError:
            print(f"  \u2717 Ex {i}: 尚未实现")
            failed += 1
        except AssertionError as e:
            print(f"  \u2717 Ex {i}: 断言失败 — {e}")
            failed += 1
            errors.append((i, str(e)))
        except Exception as e:
            print(f"  \u2717 Ex {i}: 运行错误 — {type(e).__name__}: {e}")
            failed += 1
            errors.append((i, str(e)))
    print()
    print("=" * 60)
    print(f"  得分: {passed}/{passed + failed}")
    if passed == 40:
        print("  全部通过！你已经掌握了 PyTorch 基础！")
    elif passed >= 30:
        print("  很好！再接再厉！")
    elif passed >= 20:
        print("  不错的开始，继续加油！")
    else:
        print("  继续练习，熟能生巧！")
    print("=" * 60)
    return passed, failed


if __name__ == "__main__":
    run_all()
