# mmit 完整流水线实验解读

基于 Qwen2-VL-2B + Freeze Tuning 的 Colab 实验输出，逐阶段解读。

---

## Phase 1: 加载模型

```
模型类型: Qwen2VLForConditionalGeneration
总参数量: 2,208,985,600 (2.21B)

顶层 _modules:
  model:   Qwen2VLModel    2,208,985,600 参数
  lm_head: Linear            233,373,696 参数
```

**为什么 model 和 lm_head 的参数量加起来不等于总数？**
因为 lm_head 的 weight 和 model 内部的 embed_tokens 是**共享的**（weight tying）。233M 被计算了两次，但实际只有一份。

**为什么顶层没有直接的 _parameters？**
所有参数都在子模块里（model 和 lm_head）。顶层 Module 只是一个"容器"。

---

## Phase 2: 构造训练数据

```python
CanonicalSample(
    id="sample_001",
    image_path="",           # 空=纯文本，不需要图片
    turns=[
        Turn(role="human", content="What is 2+2?"),
        Turn(role="assistant", content="2+2 equals 4."),
    ]
)
```

**这个格式的意义：** mmit 把所有数据源（HuggingFace 数据集、JSON 文件、自定义格式）都统一转成 CanonicalSample。不管数据从哪来，后面的处理流程都一样。这就是"规范化"（canonical）的含义。

**真实场景中数据从哪来：**
- `HFDatasetsAdapter` → 从 HuggingFace Hub 下载数据集
- `JsonAdapter` → 从本地 JSON 文件加载
- `DataMixer` → 把多个数据源按权重混合

---

## Phase 3: apply_chat_template

```
完整文本:
  '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
   <|im_start|>user\nWhat is 2+2?<|im_end|>\n
   <|im_start|>assistant\n2+2 equals 4.<|im_end|>\n'

仅 prompt 文本:
  '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
   <|im_start|>user\nWhat is 2+2?<|im_end|>\n
   <|im_start|>assistant\n'
```

**两个文本有什么区别？**
- `full_text` = 完整对话（含回答）→ 用来生成 input_ids
- `prompt_text` = 只有问题（不含回答）→ 用来确定"哪些 token 是指令部分，不算 loss"

**`<|im_start|>` 和 `<|im_end|>` 是什么？**
Qwen 模型的特殊标记。`im_start` = "对话开始"，`im_end` = "对话结束"。每个模型有自己的格式（LLaMA 用 `[INST]`），`apply_chat_template` 自动处理这些差异。

---

## Phase 4: Tokenization

```
位置   0: token ID 151644 -> '<|im_start|>'   ← 特殊标记
位置  14: token ID   3838 -> 'What'           ← 普通英文词
位置  17: token ID     17 -> '2'              ← 数字
位置  18: token ID     10 -> '+'              ← 符号
```

**关键观察：**
- 词表大小 = 151,936（非常大，包含中英文和特殊标记）
- 一个英文单词通常对应一个 token（"What" → 3838）
- 标点符号和数字各一个 token（"+" → 10，"2" → 17）
- 样本 1 和 2 都是 35 个 token，样本 3 是 39 个 token（因为答案更长）

---

## Phase 5: Label Masking — 最关键的一步

```
位置  input_id     label          文本  参与Loss
  0    151644      -100  '<|im_start|>'  否(masked)  ← system prompt
 ...
 14      3838      -100      'What'      否(masked)  ← 用户问题
 ...
 25       198      -100        '\n'      否(masked)  ← assistant 开头
 26        17        17         '2'      是          ← 回答的第一个 token！
 27        10        10         '+'      是
 28        17        17         '2'      是
 29     16819     16819   ' equals'      是
 30       220       220         ' '      是
 31        19        19         '4'      是
 32        13        13         '.'      是
 33    151645    151645  '<|im_end|>'    是          ← 回答的结束标记也算 loss
 34       198       198        '\n'      是
```

**这告诉我们什么？**

1. **前 26 个 token**（system prompt + 用户问题 + "assistant\n"）的 label 全是 -100
   - 这些是"指令"，模型不需要学会"预测"它们
   - `F.cross_entropy(ignore_index=-100)` 会跳过这些位置

2. **后 9 个 token**（"2+2 equals 4.<|im_end|>\n"）的 label = input_id
   - 这些是"回答"，模型要学会预测它们
   - 每个位置的 loss = -log(模型给正确 token 的概率)

3. **两步差分法的原理：**
   - tokenize("完整对话") → 35 个 token
   - tokenize("仅 prompt") → 26 个 token
   - labels[:26] = -100（前 26 个不算 loss）
   - 简单但有效！

4. **`<|im_end|>` 也参与 loss**
   - 模型必须学会在回答结束时生成结束标记
   - 否则推理时不知道何时停止

---

## Phase 6: Collate

```
各样本长度: [35, 35, 39]
padding 到统一长度: 39

样本 1 末尾: input_ids = [198, 0, 0, 0, 0]    labels = [198, -100, -100, -100, -100]
```

**为什么需要 padding？**
GPU 做矩阵运算需要 batch 内所有序列长度相同。短的序列用 0 填充（padding），但：
- `attention_mask = 0` 告诉 attention 机制"忽略这些位置"
- `labels = -100` 告诉 loss "不算这些位置"
所以 padding 不影响计算结果。

**71.8% 被 mask：** 117 个 label 中只有 33 个参与 loss。大部分 token 是指令/padding。

---

## Phase 7: Freeze Tuning

```
28 层 LLM → 冻结前 24 层 + 解冻后 4 层
可训练: 187,191,296 (8.47%)
冻结:   2,021,794,304 (91.53%)
```

**每层有什么？（46.8M 参数/层）**
```
self_attn (q/k/v/o_proj):  5.5M   ← attention 的 4 个线性层
mlp (gate/up/down_proj):  41.3M   ← SwiGLU MLP 的 3 个线性层
layernorm × 2:             3K     ← 很小
```

MLP 占了每层 88% 的参数！这是因为 Qwen2-VL-2B 的 MLP 维度远大于 attention 维度。

---

## Phase 8: Optimizer + LR Schedule

```
lr 预览:
  step 0: lr=0.000000e+00   ← warmup 阶段（从 0 开始）
  step 1: lr=2.000000e-05   ← warmup 结束，达到峰值
  step 2: lr=1.707107e-05   ← cosine 衰减开始
  step 3: lr=1.000000e-05
  step 4: lr=2.928932e-06
  step 5: lr=0.000000e+00   ← 衰减到 0
```

**warmup 为什么重要？**
Step 0 的 lr = 0（不更新），Step 1 才开始。如果一开始就用大 lr，随机权重 + 大更新 = 不稳定。warmup 让模型"热身"。

**为什么 step 1 的 loss 和 step 2 一样 (0.634233)？**
因为 step 0 的 lr = 0，参数没更新！step 1 用的还是原始参数，所以 loss 不变。step 2 才是第一次看到更新后参数的 loss。

---

## Phase 9: 训练循环

```
step 1: loss = 0.634233   ← 初始（预训练模型已经不错了）
step 2: loss = 0.634233   ← step 0 lr=0 没更新，所以不变
step 3: loss = 0.356222   ← 第一次看到更新效果！降了 44%
step 4: loss = 0.172803   ← 继续降
step 5: loss = 0.136325   ← 5 步后降了 78%
```

**关键观察：**

1. **初始 loss = 0.634**，远低于随机猜测的 11.93
   - 预训练模型已经知道英语，只需要微调
   - 如果是未训练的模型，初始 loss ≈ 12

2. **梯度裁剪每步都生效**
   ```
   step 1: 总梯度范数 = 10.81, 裁剪到 1.0
   step 5: 总梯度范数 = 5.34,  裁剪到 1.0
   ```
   梯度范数在下降（模型越来越接近目标，梯度越来越小），但还是超过 1.0 需要裁剪。

3. **参数变化非常小**
   ```
   变化量: [3.05e-05, 0.0, 1.91e-05, ...]
   ```
   每个值只变了 0.00003 左右。但 187M 个参数同时变化，累积效果很大。

4. **lr 衰减的效果**
   step 4 (lr=2.9e-06) 的变化量比 step 2 (lr=2e-05) 小得多。lr 在控制"步子大小"。

---

## Phase 10: 验证

```
训练前 loss: 0.634233
训练后 loss: 0.130254 (下降 79.5%)
```

模型对这 3 条训练数据的预测能力大幅提升。但注意这是**训练集上的 loss**——如果拿新数据测试（泛化能力），改善会小得多。

---

## Phase 11: 保存

```
保存了 48 / 730 个参数张量
checkpoint: 357 MB vs 全量 4213 MB (节省 91.5%)
```

**48 个张量是什么？** 最后 4 层，每层有 12 个参数张量（q/k/v/o_proj 的 weight 和 bias + gate/up/down_proj 的 weight + 2 个 layernorm 的 weight）。4 × 12 = 48。

**metadata 记录了什么：**
- base_model：加载时需要知道用哪个基础模型
- ft_method：加载时需要知道用什么方法
- trained_param_names：加载时需要知道覆盖哪些参数

---

## Phase 12: 加载验证

```
missing_keys: 682    ← 冻结的参数没保存，所以"缺失"
unexpected_keys: 0   ← 没有多余的 key
加载后 loss: 0.130254 = 训练后 loss（完全一致！）
```

**`strict=False` 的作用：** 正常 `load_state_dict` 要求 key 完全匹配。加了 `strict=False`，682 个"缺失"的 key 不报错——它们保持基础模型的原始值（这正是我们要的）。

---

## Phase 13: 推理

```
问题: What is 3+3?
回答: 3+3 equals 6.
```

**生成过程：**
```
token 0: '3'         ← 模型看到 "What is 3+3?" 后生成
token 1: '+'         ← 看到 "3" 后生成
token 2: '3'         ← 看到 "3+" 后生成
token 3: ' equals'   ← 看到 "3+3" 后生成
token 4: ' '
token 5: '6'         ← 会算了！
token 6: '.'
token 7: '<|im_end|>' ← 知道要停止了
```

**模型学到了什么？** 训练数据只有 "2+2 equals 4"，但模型对 "3+3" 给出了正确答案。这不是因为它学会了加法——而是因为预训练模型本来就懂数学，微调只是加强了 "X+X equals Y." 这个回答格式。

---

## 总结：mmit 的完整流水线

```
原始数据 (JSON/HF)
    │
    ▼ DatasetAdapter
CanonicalSample (统一格式)
    │
    ▼ ChatTemplatePreprocessor
apply_chat_template → full_text / prompt_text
    │
    ▼ tokenize
input_ids + labels (两步差分法 mask)
    │
    ▼ collate
batch tensor (padding 到统一长度)
    │
    ▼ FreezeTuningMethod._prepare_model_impl
冻结 91.5% 参数 + 解冻最后 4 层
    │
    ▼ StageRunner.run_stage 训练循环
forward → loss → backward → clip → step → schedule
    │ × N steps
    ▼
save_checkpoint (只存 8.5% 参数)
    │
    ▼ load_for_inference
基础模型 + checkpoint → 可推理的模型
    │
    ▼ model.generate()
输出回答
```
