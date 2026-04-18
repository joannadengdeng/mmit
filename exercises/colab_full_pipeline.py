"""
mmit 完整训练流水线 —— 从零开始在 Colab 上运行
=================================================
不依赖 mmit 包，全部内联实现，每一步都有详细中文打印。
13 个阶段覆盖：数据构造 → 预处理 → tokenization → label masking →
collate → 模型准备 → 优化器 → 训练 → 验证 → 保存 → 加载 → 推理
"""

import gc
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

IGNORE_INDEX = -100
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# ──────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────

def separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def subsep(title: str):
    print(f"\n--- {title} ---")


# ──────────────────────────────────────────────────────────────────────
# 数据类型（对应 mmit/data/types.py）
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str       # "human" | "assistant"
    content: str


@dataclass
class CanonicalSample:
    id: str
    image_path: str
    turns: List[Turn]
    metadata: Dict[str, Any] = field(default_factory=dict)
    instruction: str = ""

    @property
    def first_question(self) -> str:
        for t in self.turns:
            if t.role == "human":
                return t.content
        return ""

    @property
    def first_answer(self) -> str:
        for t in self.turns:
            if t.role == "assistant":
                return t.content
        return ""


# ======================================================================
# Phase 1: 加载模型
# ======================================================================

def phase1_load_model():
    separator("Phase 1: 加载模型")

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    print(f"[1] 正在从 HuggingFace 加载模型: {MODEL_ID}")
    print("    mmit 中对应代码: StageRunner._load_model()")
    print("    使用 AutoModelForImageTextToText + AutoProcessor")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoVLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )

    print(f"\n[1] 模型加载完成!")
    print(f"    模型类型: {type(model).__name__}")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print(f"    设备: {device}")
    print(f"    数据类型: {dtype}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"    总参数量: {total_params:,} ({total_params/1e9:.2f}B)")

    subsep("模型顶层 _modules 结构")
    for name, child in model.named_children():
        child_params = sum(p.numel() for p in child.parameters())
        print(f"    {name}: {type(child).__name__}  参数量={child_params:,}")

    subsep("模型 _parameters（顶层直接参数）")
    direct_params = list(model._parameters.items())
    if direct_params:
        for pname, p in direct_params:
            print(f"    {pname}: shape={p.shape}, dtype={p.dtype}")
    else:
        print("    （顶层无直接参数，所有参数都在子模块中）")

    return model, processor


# ======================================================================
# Phase 2: 构造训练数据
# ======================================================================

def phase2_create_samples():
    separator("Phase 2: 构造训练数据 (CanonicalSample)")

    print("[2] 在真实的 mmit 中，训练数据来自:")
    print("    - HuggingFace datasets (通过 HFDatasetsAdapter)")
    print("    - 本地 JSON 文件 (通过 JsonAdapter)")
    print("    每个样本会被转换为统一的 CanonicalSample 格式")
    print("    CanonicalSample 定义在 mmit/data/types.py")
    print()

    samples = [
        CanonicalSample(
            id="sample_001",
            image_path="",
            turns=[
                Turn(role="human", content="What is 2+2?"),
                Turn(role="assistant", content="2+2 equals 4."),
            ],
        ),
        CanonicalSample(
            id="sample_002",
            image_path="",
            turns=[
                Turn(role="human", content="What is the capital of France?"),
                Turn(role="assistant", content="The capital of France is Paris."),
            ],
        ),
        CanonicalSample(
            id="sample_003",
            image_path="",
            turns=[
                Turn(role="human", content="Explain gravity briefly."),
                Turn(role="assistant", content="Gravity is a force that attracts objects with mass toward each other."),
            ],
        ),
    ]

    for s in samples:
        print(f"  样本 {s.id}:")
        print(f"    image_path: '{s.image_path}' (空=纯文本)")
        print(f"    turns:")
        for t in s.turns:
            print(f"      [{t.role}] {t.content}")
        print(f"    first_question: {s.first_question}")
        print(f"    first_answer:   {s.first_answer}")
        print()

    print(f"[2] 共创建 {len(samples)} 个 CanonicalSample")
    return samples


# ======================================================================
# Phase 3: apply_chat_template
# ======================================================================

def phase3_apply_chat_template(samples, processor):
    separator("Phase 3: 数据预处理 -- apply_chat_template")

    print("[3] 对应 mmit 代码: ChatTemplatePreprocessor.tokenize() 中的")
    print("    _build_messages() + processor.apply_chat_template()")
    print("    目的: 把 CanonicalSample 转成模型期望的对话格式文本")
    print()

    full_texts = []
    prompt_texts = []
    all_messages = []

    for s in samples:
        has_image = bool(s.image_path)
        # _build_messages: 与 mmit/training/preprocessors/chat_template.py 一致
        messages = []
        for turn in s.turns:
            role = "user" if turn.role == "human" else "assistant"
            if role == "user" and has_image and not messages:
                content = [{"type": "image"}, {"type": "text", "text": turn.content}]
            else:
                content = [{"type": "text", "text": turn.content}]
            messages.append({"role": role, "content": content})
        all_messages.append(messages)

        # 完整对话文本
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        full_texts.append(full_text)

        # prompt-only 文本（排除最后一个 assistant 回复，用于 label masking）
        last_asst_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                last_asst_idx = i
                break
        prompt_msgs = messages[:last_asst_idx] if last_asst_idx >= 0 else messages
        if prompt_msgs:
            prompt_text = processor.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text = ""
        prompt_texts.append(prompt_text)

        subsep(f"样本 {s.id}")
        print(f"  HF messages 格式:")
        for m in messages:
            print(f"    role={m['role']}, content={m['content']}")
        print(f"\n  完整文本 (full_text):")
        print(f"    {repr(full_text)}")
        print(f"\n  仅 prompt 文本 (prompt_text):")
        print(f"    {repr(prompt_text)}")

    return full_texts, prompt_texts, all_messages


# ======================================================================
# Phase 4: Tokenization
# ======================================================================

def phase4_tokenize(full_texts, processor):
    separator("Phase 4: Tokenization -- 从文本到 token ID")

    print("[4] 对应 mmit 代码: ChatTemplatePreprocessor.tokenize() 中的")
    print("    processor(text=full_text, ..., return_tensors='pt')")
    print("    目的: 把文本转成模型能处理的 token ID 序列")
    print()

    tokenized_list = []

    for i, text in enumerate(full_texts):
        inputs = processor(
            text=text, images=None,
            return_tensors="pt", truncation=True, max_length=2048,
        )
        input_ids = inputs["input_ids"].squeeze(0)
        tokenized_list.append(inputs)

        subsep(f"样本 {i+1} tokenization 结果")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  token 数量: {input_ids.shape[0]}")
        print(f"\n  前 20 个 token ID 及其对应文本:")
        for j in range(min(20, input_ids.shape[0])):
            tid = input_ids[j].item()
            decoded = processor.tokenizer.decode([tid])
            print(f"    位置 {j:3d}: token ID {tid:6d} -> {repr(decoded)}")

        if input_ids.shape[0] > 20:
            print(f"    ... 后续还有 {input_ids.shape[0] - 20} 个 token")

    return tokenized_list


# ======================================================================
# Phase 5: Label Masking
# ======================================================================

def phase5_label_masking(full_texts, prompt_texts, processor):
    separator("Phase 5: Label Masking -- 哪些 token 算 loss")

    print("[5] 对应 mmit 代码: ChatTemplatePreprocessor.tokenize() 中的")
    print("    两步差分法 (two-pass diff):")
    print("    1. tokenize 完整对话 -> input_ids")
    print("    2. tokenize 仅 prompt -> prompt_len")
    print("    3. labels = input_ids.clone(); labels[:prompt_len] = -100")
    print("    被设为 -100 的位置不会参与 loss 计算 (PyTorch CE 的约定)")
    print()

    processed_samples = []

    for i, (full_text, prompt_text) in enumerate(zip(full_texts, prompt_texts)):
        full_inputs = processor(
            text=full_text, images=None,
            return_tensors="pt", truncation=True, max_length=2048,
        )
        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs.get(
            "attention_mask", torch.ones_like(input_ids)
        )
        if attention_mask.dim() > 1:
            attention_mask = attention_mask.squeeze(0)

        labels = input_ids.clone()

        if prompt_text:
            prompt_inputs = processor(
                text=prompt_text, images=None,
                return_tensors="pt", truncation=True, max_length=2048,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
        else:
            prompt_len = 0

        labels[:prompt_len] = IGNORE_INDEX

        subsep(f"样本 {i+1} label masking")
        print(f"  总 token 数: {input_ids.shape[0]}")
        print(f"  prompt 长度 (被 mask): {prompt_len}")
        print(f"  参与 loss 的 token 数: {(labels != IGNORE_INDEX).sum().item()}")
        print()
        print(f"  逐位置对比 (input_ids vs labels):")
        print(f"  {'位置':>4s}  {'input_id':>8s}  {'label':>8s}  {'文本':>10s}  {'参与loss':>6s}")
        print(f"  {'----':>4s}  {'--------':>8s}  {'-----':>8s}  {'----':>10s}  {'------':>6s}")
        for j in range(input_ids.shape[0]):
            tid = input_ids[j].item()
            lid = labels[j].item()
            decoded = processor.tokenizer.decode([tid])
            in_loss = "是" if lid != IGNORE_INDEX else "否(masked)"
            print(f"  {j:4d}  {tid:8d}  {lid:8d}  {repr(decoded):>10s}  {in_loss}")

        processed_samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        })

    return processed_samples


# ======================================================================
# Phase 6: Collate
# ======================================================================

def phase6_collate(processed_samples):
    separator("Phase 6: Collate -- 把多个样本拼成一个 batch")

    print("[6] 对应 mmit 代码: ChatTemplatePreprocessor.collate()")
    print("    目的: 不同长度的样本需要 padding 到相同长度才能组成 batch tensor")
    print("    padding 策略: input_ids 用 0 填充, labels 用 -100 填充, attention_mask 用 0 填充")
    print()

    max_len = max(s["input_ids"].size(0) for s in processed_samples)
    batch_size = len(processed_samples)

    print(f"  batch_size = {batch_size}")
    print(f"  各样本长度: {[s['input_ids'].size(0) for s in processed_samples]}")
    print(f"  padding 到统一长度: {max_len}")

    batch_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    batch_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    batch_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, s in enumerate(processed_samples):
        seq_len = s["input_ids"].size(0)
        batch_ids[i, :seq_len] = s["input_ids"]
        batch_labels[i, :seq_len] = s["labels"]
        batch_mask[i, :seq_len] = s["attention_mask"]

    batch = {
        "input_ids": batch_ids,
        "labels": batch_labels,
        "attention_mask": batch_mask,
    }

    subsep("batch 张量信息")
    for key, tensor in batch.items():
        print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")

    subsep("padding 示例 (样本 1 末尾)")
    s0_len = processed_samples[0]["input_ids"].size(0)
    if s0_len < max_len:
        print(f"  样本 1 原始长度={s0_len}, padding 到 {max_len}")
        print(f"  input_ids 末尾 5 位: {batch_ids[0, -5:].tolist()} (0=padding)")
        print(f"  labels 末尾 5 位:    {batch_labels[0, -5:].tolist()} (-100=不算loss)")
        print(f"  attention_mask 末尾: {batch_mask[0, -5:].tolist()} (0=忽略)")
    else:
        print(f"  样本 1 正好是最长的 ({s0_len})，无需 padding")

    subsep("每个样本参与 loss 的 token 数")
    for i in range(batch_size):
        n_loss = (batch_labels[i] != IGNORE_INDEX).sum().item()
        print(f"  样本 {i+1}: {n_loss} 个 token 参与 loss")

    return batch


# ======================================================================
# Phase 7: Freeze Tuning
# ======================================================================

def phase7_prepare_model(model):
    separator("Phase 7: 准备模型 -- Freeze Tuning")

    print("[7] 对应 mmit 代码: FreezeTuningMethod._prepare_model_impl()")
    print("    策略: 先冻结所有参数，再解冻最后 N 层 LLM layers")
    print("    目的: 只训练少量参数，大幅减少显存和计算量")
    print()

    # 第一步: 冻结全部参数
    for p in model.parameters():
        if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
            p.requires_grad = False

    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    print(f"  冻结后: {frozen_count}/{total_count} 个参数张量的 requires_grad=False")

    # 第二步: 找到 LLM transformer layers
    search_paths = [
        "model.language_model.layers",
        "model.model.layers",
        "model.layers",
        "language_model.model.layers",
        "language_model.layers",
    ]
    llm_layers = None
    found_path = None
    for attr_path in search_paths:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            llm_layers = list(obj)
            found_path = attr_path
            break
        except AttributeError:
            continue

    if llm_layers is None:
        print("  警告: 未找到 LLM layers，尝试备用方法...")
        # 备用: 使用 freeze.py 中的 _find_transformer_layers
        for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                llm_layers = list(obj)
                found_path = attr
                break
            except AttributeError:
                continue

    NUM_UNFREEZE = 4
    print(f"  找到 LLM layers 路径: {found_path}")
    print(f"  总共 {len(llm_layers)} 层 transformer layers")
    print(f"  将解冻最后 {NUM_UNFREEZE} 层")

    # 第三步: 解冻最后 N 层
    for layer in llm_layers[-NUM_UNFREEZE:]:
        for p in layer.parameters():
            if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    total_tensors = sum(1 for p in model.parameters())

    subsep("参数状态汇总")
    print(f"  可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"  可训练张量: {trainable_tensors} / {total_tensors}")
    print(f"  冻结参数:   {total - trainable:,} ({100*(total-trainable)/total:.2f}%)")
    print(f"  节省显存:   只需为 {100*trainable/total:.2f}% 的参数存储梯度和优化器状态")

    subsep("解冻的层 (最后4层)")
    for idx in range(len(llm_layers) - NUM_UNFREEZE, len(llm_layers)):
        layer = llm_layers[idx]
        layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        print(f"  Layer {idx}: {layer_params:,} 可训练参数")

    return model, llm_layers


# ======================================================================
# Phase 8: Optimizer + Scheduler
# ======================================================================

def phase8_optimizer_scheduler(model):
    separator("Phase 8: 创建 Optimizer + LR Scheduler")

    print("[8] 对应 mmit 代码: StageRunner.run_stage() 中的")
    print("    optimizer = AdamW(param_groups, ...)")
    print("    scheduler = _cosine_schedule(optimizer, warmup_steps, total_steps)")
    print()

    LR = 2e-5
    WEIGHT_DECAY = 0.0
    TOTAL_STEPS = 5
    WARMUP_RATIO = 0.03
    warmup_steps = max(1, int(TOTAL_STEPS * WARMUP_RATIO))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  可训练参数张量数: {len(trainable_params)}")
    print(f"  学习率: {LR}")
    print(f"  weight_decay: {WEIGHT_DECAY}")

    optimizer = AdamW([{"params": trainable_params, "lr": LR}],
                      weight_decay=WEIGHT_DECAY)

    print(f"\n  Optimizer 类型: {type(optimizer).__name__}")
    print(f"  参数组数: {len(optimizer.param_groups)}")
    print(f"  参数组 0: {len(optimizer.param_groups[0]['params'])} 个张量, lr={optimizer.param_groups[0]['lr']}")

    # cosine schedule with linear warmup (与 stage_runner.py 完全一致)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, TOTAL_STEPS - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    subsep("学习率调度")
    print(f"  total_steps: {TOTAL_STEPS}")
    print(f"  warmup_steps: {warmup_steps} (warmup_ratio={WARMUP_RATIO})")
    print(f"  调度类型: cosine with linear warmup")
    print(f"  初始 lr: {scheduler.get_last_lr()[0]:.6e}")
    print(f"\n  各步骤的 lr 预览:")
    for s in range(TOTAL_STEPS + 1):
        lr_val = lr_lambda(s) * LR
        bar = "#" * int(lr_val / LR * 40)
        print(f"    step {s}: lr={lr_val:.6e}  {bar}")

    return optimizer, scheduler, warmup_steps, TOTAL_STEPS


# ======================================================================
# Phase 9: 训练循环
# ======================================================================

def phase9_training_loop(model, batch, optimizer, scheduler):
    separator("Phase 9: 训练循环 -- 5 步详细打印")

    print("[9] 对应 mmit 代码: StageRunner.run_stage() 中的训练循环")
    print("    每步: forward -> loss -> backward -> clip_grad -> optimizer.step -> scheduler.step")
    print()

    MAX_GRAD_NORM = 1.0
    GRAD_ACCUM = 1
    NUM_STEPS = 5

    device = next(model.parameters()).device
    # 将 batch 移到 device
    batch_gpu = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_gpu[k] = v.to(device)
        else:
            batch_gpu[k] = v

    model.train()
    loss_history = []

    # 找 3 个代表性参数用于打印梯度
    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    repr_params = []
    if len(named_trainable) >= 3:
        repr_params = [
            named_trainable[0],
            named_trainable[len(named_trainable) // 2],
            named_trainable[-1],
        ]
    else:
        repr_params = named_trainable[:3]

    # 记录一个参数的初始值用于对比
    track_name, track_param = repr_params[0]
    initial_values = track_param.data.flatten()[:5].clone().float()

    for step in range(NUM_STEPS):
        subsep(f"训练步骤 {step + 1}/{NUM_STEPS}")

        # 输入信息
        n_masked = (batch_gpu["labels"] == IGNORE_INDEX).sum().item()
        n_total = batch_gpu["labels"].numel()
        print(f"  输入: input_ids shape={batch_gpu['input_ids'].shape}")
        print(f"  labels 中 -100 的数量: {n_masked}/{n_total} "
              f"({100*n_masked/n_total:.1f}% 被 mask)")

        # Forward pass
        t0 = time.time()
        outputs = model(**batch_gpu)
        loss = outputs.loss
        fwd_time = time.time() - t0
        print(f"  Forward pass: {fwd_time:.3f}s")
        print(f"  Loss = {loss.item():.6f}")

        loss_value = loss.item()
        loss_history.append(loss_value)

        scaled_loss = loss / GRAD_ACCUM
        print(f"  scaled_loss = loss / {GRAD_ACCUM} = {scaled_loss.item():.6f}")

        # Backward pass
        t0 = time.time()
        scaled_loss.backward()
        bwd_time = time.time() - t0
        print(f"  Backward pass: {bwd_time:.3f}s")

        # 打印梯度范数
        print(f"  代表性参数梯度范数:")
        for pname, p in repr_params:
            if p.grad is not None:
                grad_norm = p.grad.data.norm(2).item()
                short_name = pname.split(".")[-3] + "." + pname.split(".")[-2] + "." + pname.split(".")[-1] if len(pname.split(".")) >= 3 else pname
                print(f"    {short_name}: grad_norm = {grad_norm:.6f}")

        # Gradient clipping
        all_params = [p for p in model.parameters() if p.requires_grad]
        total_norm_before = torch.nn.utils.clip_grad_norm_(all_params, float("inf")).item()
        # 重新 backward 会出错，所以我们只是计算 norm 后再真正 clip
        # 实际上 clip_grad_norm_ 会修改梯度，所以先打印 before，再 clip
        # 由于上面已经 clip 了一次 (inf), 梯度没变，所以再 clip 一次
        total_norm_after = torch.nn.utils.clip_grad_norm_(all_params, MAX_GRAD_NORM).item()
        print(f"  梯度裁剪: 总梯度范数 = {total_norm_before:.6f}, "
              f"裁剪阈值 = {MAX_GRAD_NORM}, 裁剪后 = {min(total_norm_before, MAX_GRAD_NORM):.6f}")

        # 记录更新前的参数值
        before_values = track_param.data.flatten()[:5].clone().float()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # 记录更新后的参数值
        after_values = track_param.data.flatten()[:5].clone().float()

        print(f"  参数更新 ({track_name} 前5个值):")
        print(f"    更新前: {before_values.tolist()}")
        print(f"    更新后: {after_values.tolist()}")
        diff = (after_values - before_values).abs()
        print(f"    变化量: {diff.tolist()}")

        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"  学习率更新: lr = {current_lr:.6e}")

    subsep("5 步训练总结")
    print(f"  Loss 变化趋势:")
    for i, l in enumerate(loss_history):
        bar = "#" * int(l * 5)
        print(f"    step {i+1}: loss = {l:.6f}  {bar}")
    if loss_history[-1] < loss_history[0]:
        print(f"  Loss 下降了! {loss_history[0]:.6f} -> {loss_history[-1]:.6f} "
              f"(下降 {loss_history[0]-loss_history[-1]:.6f})")
    else:
        print(f"  Loss 变化: {loss_history[0]:.6f} -> {loss_history[-1]:.6f}")

    # 对比初始参数
    final_values = track_param.data.flatten()[:5].clone().float()
    total_change = (final_values - initial_values).abs().mean().item()
    print(f"  参数累计变化 (平均绝对变化): {total_change:.8f}")

    return loss_history


# ======================================================================
# Phase 10: 验证
# ======================================================================

def phase10_validation(model, batch):
    separator("Phase 10: 验证 -- loss 是否降低")

    print("[10] 用训练过的模型在同样数据上算 loss（无梯度）")
    print("     对应: 训练后的 eval 阶段")
    print()

    device = next(model.parameters()).device
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**batch_gpu)
        val_loss = outputs.loss.item()

    print(f"  验证 loss = {val_loss:.6f}")
    return val_loss


# ======================================================================
# Phase 11: 保存 Checkpoint
# ======================================================================

def phase11_save_checkpoint(model, processor):
    separator("Phase 11: 保存 Checkpoint")

    print("[11] 对应 mmit 代码: FreezeTuningMethod.save_checkpoint()")
    print("     只保存 requires_grad=True 的参数 (不保存冻结的参数)")
    print("     这样 checkpoint 文件非常小")
    print()

    save_dir = tempfile.mkdtemp(prefix="mmit_ckpt_")
    print(f"  保存目录: {save_dir}")

    # 找到可训练的参数名
    trained_names = {n for n, p in model.named_parameters() if p.requires_grad}
    print(f"  可训练参数名数量: {len(trained_names)}")

    # 保存可训练参数
    trainable_state = {k: v for k, v in model.state_dict().items() if k in trained_names}
    ckpt_path = os.path.join(save_dir, "freeze_tuned.pt")
    torch.save(trainable_state, ckpt_path)

    ckpt_size = os.path.getsize(ckpt_path)
    print(f"  checkpoint 文件: {ckpt_path}")
    print(f"  checkpoint 大小: {ckpt_size / 1024 / 1024:.2f} MB")

    # 计算完整模型大小
    full_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  完整模型大小(估算): {full_size / 1024 / 1024:.2f} MB")
    print(f"  压缩比: checkpoint 仅占完整模型的 {100*ckpt_size/full_size:.2f}%")

    # 保存 metadata
    metadata = {
        "base_model": MODEL_ID,
        "ft_method": "freeze",
        "stage": "demo",
        "trained_param_names": sorted(trained_names),
    }
    meta_path = os.path.join(save_dir, "mmit_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    subsep("metadata 内容 (mmit_meta.json)")
    print(f"  base_model: {metadata['base_model']}")
    print(f"  ft_method: {metadata['ft_method']}")
    print(f"  trained_param_names 数量: {len(metadata['trained_param_names'])}")
    print(f"  前 5 个参数名:")
    for name in metadata["trained_param_names"][:5]:
        print(f"    {name}")
    if len(metadata["trained_param_names"]) > 5:
        print(f"    ... 还有 {len(metadata['trained_param_names'])-5} 个")

    # 保存 processor
    processor.save_pretrained(save_dir)
    print(f"\n  processor 也已保存到: {save_dir}")

    return save_dir


# ======================================================================
# Phase 12: 加载 Checkpoint
# ======================================================================

def phase12_load_checkpoint(save_dir, batch, trained_loss):
    separator("Phase 12: 加载 Checkpoint 到新模型")

    print("[12] 对应 mmit 代码: FreezeTuningMethod.load_for_inference()")
    print("     1. 加载新的 base model")
    print("     2. load_state_dict(strict=False) 加载训练过的权重")
    print("     3. 验证 loss 与训练后模型一致")
    print()

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    print("  正在加载新的 base model...")
    new_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    new_model = AutoVLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )

    # 加载 checkpoint
    ckpt_path = os.path.join(save_dir, "freeze_tuned.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"  加载 checkpoint: {ckpt_path}")
    print(f"  checkpoint 中参数数: {len(state)}")

    result = new_model.load_state_dict(state, strict=False)
    print(f"\n  load_state_dict 结果 (strict=False):")
    print(f"    missing_keys 数量: {len(result.missing_keys)} (这些是冻结的参数，符合预期)")
    print(f"    unexpected_keys 数量: {len(result.unexpected_keys)} (应该是 0)")

    if result.unexpected_keys:
        print(f"    unexpected_keys: {result.unexpected_keys[:5]}")

    # 验证 loss
    device = next(new_model.parameters()).device
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

    new_model.eval()
    with torch.no_grad():
        outputs = new_model(**batch_gpu)
        new_loss = outputs.loss.item()

    print(f"\n  验证:")
    print(f"    训练后模型的 loss: {trained_loss:.6f}")
    print(f"    加载 checkpoint 后的 loss: {new_loss:.6f}")
    print(f"    差异: {abs(new_loss - trained_loss):.8f}")
    if abs(new_loss - trained_loss) < 0.01:
        print(f"    checkpoint 加载成功! 两个模型的 loss 基本一致")
    else:
        print(f"    注意: loss 有差异，可能是精度或 device mapping 差异")

    return new_model, new_processor


# ======================================================================
# Phase 13: 推理生成
# ======================================================================

def phase13_inference(model, processor):
    separator("Phase 13: 推理生成")

    print("[13] 使用训练后的模型生成回答")
    print("     流程: 构建 prompt -> tokenize -> model.generate() -> decode")
    print()

    question = "What is 3+3?"
    print(f"  问题: {question}")

    # 构建 messages
    messages = [
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    print(f"\n  格式化 prompt:")
    print(f"    {repr(prompt_text)}")

    # Tokenize
    inputs = processor(
        text=prompt_text, images=None,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    print(f"\n  input_ids shape: {inputs['input_ids'].shape}")
    print(f"  input token 数: {inputs['input_ids'].shape[1]}")

    # Generate
    model.eval()
    print(f"\n  开始生成...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    # Decode
    # 只取新生成的 token
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, input_len:]
    generated_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"  生成的 token 数: {new_tokens.shape[0]}")
    print(f"  生成的文本: {generated_text}")

    # 展示生成过程中的 token
    subsep("逐 token 查看生成结果")
    for i, tid in enumerate(new_tokens):
        decoded = processor.tokenizer.decode([tid.item()])
        print(f"    token {i}: id={tid.item():6d} -> {repr(decoded)}")

    full_output = processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
    subsep("完整输出 (含特殊 token)")
    print(f"  {repr(full_output)}")


# ======================================================================
# Final Summary
# ======================================================================

def final_summary():
    separator("最终总结: 各阶段与 mmit 代码的对应关系")

    print()
    table = [
        ("Phase 1",  "加载模型",            "StageRunner._load_model()"),
        ("Phase 2",  "构造 CanonicalSample", "data/types.py: CanonicalSample, Turn"),
        ("Phase 3",  "apply_chat_template",  "chat_template.py: _build_messages() + processor.apply_chat_template()"),
        ("Phase 4",  "Tokenization",         "ChatTemplatePreprocessor.tokenize() 中 processor(text=...)"),
        ("Phase 5",  "Label Masking",        "ChatTemplatePreprocessor.tokenize() 中两步差分法"),
        ("Phase 6",  "Collate",              "ChatTemplatePreprocessor.collate()"),
        ("Phase 7",  "Freeze Tuning",        "FreezeTuningMethod._prepare_model_impl()"),
        ("Phase 8",  "Optimizer+Scheduler",  "StageRunner.run_stage() 中 AdamW + _cosine_schedule()"),
        ("Phase 9",  "训练循环",             "StageRunner.run_stage() 中 for epoch/step 循环"),
        ("Phase 10", "验证",                 "训练后 eval (模型 forward 无梯度)"),
        ("Phase 11", "保存 Checkpoint",      "FreezeTuningMethod.save_checkpoint()"),
        ("Phase 12", "加载 Checkpoint",      "FreezeTuningMethod.load_for_inference()"),
        ("Phase 13", "推理生成",             "model.generate() (HuggingFace 标准 API)"),
    ]

    print(f"  {'阶段':<10s}  {'内容':<22s}  {'mmit 对应代码'}")
    print(f"  {'─'*10}  {'─'*22}  {'─'*55}")
    for phase, desc, code in table:
        print(f"  {phase:<10s}  {desc:<22s}  {code}")

    print()
    print("  关键设计模式:")
    print("    1. CanonicalSample 是统一数据格式 -- 所有数据源都转换为这个格式")
    print("    2. ChatTemplatePreprocessor 用 HF 的 apply_chat_template 处理 -- 适配所有模型")
    print("    3. 两步差分法做 label masking -- tokenize 完整对话 vs tokenize 仅 prompt")
    print("    4. TrainingMethod 是策略模式 -- freeze/qlora/lora 等都实现相同接口")
    print("    5. StageRunner 是编排器 -- 协调数据、预处理、方法、损失函数、优化器")
    print("    6. Checkpoint 只保存可训练参数 -- 节省存储空间")
    print()
    print("  完整流水线运行结束!")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("  mmit 完整训练流水线 -- 13 阶段全流程演示")
    print("  模型: Qwen2-VL-2B-Instruct")
    print("  方法: Freeze Tuning (冻结大部分参数，只训练最后4层)")
    print("=" * 70)

    # Phase 1
    model, processor = phase1_load_model()

    # Phase 2
    samples = phase2_create_samples()

    # Phase 3
    full_texts, prompt_texts, all_messages = phase3_apply_chat_template(samples, processor)

    # Phase 4
    tokenized_list = phase4_tokenize(full_texts, processor)

    # Phase 5
    processed_samples = phase5_label_masking(full_texts, prompt_texts, processor)

    # Phase 6
    batch = phase6_collate(processed_samples)

    # Phase 7
    model, llm_layers = phase7_prepare_model(model)

    # Phase 8
    optimizer, scheduler, warmup_steps, total_steps = phase8_optimizer_scheduler(model)

    # 记录训练前 loss
    device = next(model.parameters()).device
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        pre_loss = model(**batch_gpu).loss.item()
    print(f"\n  训练前 loss: {pre_loss:.6f}")

    # Phase 9
    loss_history = phase9_training_loop(model, batch, optimizer, scheduler)

    # Phase 10
    val_loss = phase10_validation(model, batch)
    print(f"  训练前 loss: {pre_loss:.6f}")
    print(f"  训练后 loss: {val_loss:.6f}")
    if val_loss < pre_loss:
        print(f"  loss 下降了 {pre_loss - val_loss:.6f}，训练有效!")
    else:
        print(f"  loss 未下降（可能需要更多步骤或调参）")

    # Phase 11
    save_dir = phase11_save_checkpoint(model, processor)

    # Phase 12: 释放旧模型，加载新的
    trained_loss = val_loss
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n  已释放旧模型显存")

    new_model, new_processor = phase12_load_checkpoint(save_dir, batch, trained_loss)

    # Phase 13
    phase13_inference(new_model, new_processor)

    # Final summary
    final_summary()


if __name__ == "__main__":
    main()
