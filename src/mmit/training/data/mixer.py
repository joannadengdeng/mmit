"""DataMixer — combines multiple data sources into a single training stream.

Stage 3 的核心模块。训练 VLM 时往往需要混合多个数据集（比如 LLaVA-1.5
同时用了 VQA、Caption、OCR 等 10 个数据集），这个模块负责把它们合成一个列表。

两种混合策略：
  - ConcatMixer: 简单拼接 [数据集A全部, 数据集B全部, ...]
  - WeightedInterleaveMixer: 按权重交错采样 [A, A, B, A, B, A, ...]

每个 DataSource 还可以带一个 instruction_suffix，比如 "Answer briefly."，
会被追加到该数据源每个样本的最后一个 human turn 后面。这样不同数据集可以
用不同的回答格式要求。
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, List, Optional

from mmit.data.types import CanonicalSample, Turn


@dataclass
class DataSource:
    """一个数据源的混合配置。

    在 YAML 里的对应写法：
        data_sources:
          - adapter: "hf_datasets"
            dataset: "liuhaotian/LLaVA-Instruct-150K"
            weight: 1.0                    # 采样权重（ConcatMixer 忽略此字段）
            instruction_suffix: ""          # 追加到 human turn 末尾
            max_samples: 0                  # 0 = 用全部

    stage_runner.py 会把 YAML 里的每个 data_source 条目构建成一个 DataSource 对象。
    """
    adapter: Any                     # DatasetAdapter 实例（HFDatasetsAdapter 或 LLaVAJSONAdapter）
    weight: float = 1.0              # 采样概率权重（只有 WeightedInterleaveMixer 用到）
    instruction_suffix: str = ""     # 追加到最后一个 human turn 的文本
    max_samples: int = 0             # 最多取多少条样本，0 = 全部


def _apply_suffix(sample: CanonicalSample, suffix: str) -> CanonicalSample:
    """把 instruction_suffix 追加到样本的最后一个 human turn 后面。

    为什么是"最后一个" human turn？因为多轮对话中，suffix 应该加在最终的提问后面，
    而不是中间的某个 turn。

    例子：
        suffix = "Answer briefly."

        原始 turns:
          Turn("human", "What is this?")
          Turn("assistant", "It's a cat.")
          Turn("human", "What color?")       ← 最后一个 human turn

        修改后:
          Turn("human", "What is this?")
          Turn("assistant", "It's a cat.")
          Turn("human", "What color? Answer briefly.")  ← suffix 追加在这里

    注意：返回的是一个新的 CanonicalSample，不修改原对象（immutable 设计）。
    """
    if not suffix:
        return sample
    # 从后往前找第一个 human turn
    new_turns = list(sample.turns)
    for i in range(len(new_turns) - 1, -1, -1):
        if new_turns[i].role == "human":
            # 找到了，在 content 末尾追加 suffix
            new_turns[i] = Turn(
                role="human",
                content=new_turns[i].content.rstrip() + " " + suffix,
            )
            break
    # 返回新对象，保留原 sample 的所有其他字段
    return CanonicalSample(
        id=sample.id,
        image_path=sample.image_path,
        turns=new_turns,
        metadata=sample.metadata,
        instruction=sample.instruction,
    )


class DataMixer(ABC):
    """数据混合策略的基类。

    所有 mixer 都注册在 Registry 的 "mixer" slot 里：
        registry.register("mixer", "concat", ConcatMixer)
        registry.register("mixer", "weighted_interleave", WeightedInterleaveMixer)

    在 stage_runner.py 里被调用：
        mixer = registry.build("mixer", stage.mixer)  # "concat" 或 "weighted_interleave"
        samples = mixer.mix(sources)                   # sources 是 List[DataSource]
    """

    @abstractmethod
    def mix(self, sources: List[DataSource]) -> List[CanonicalSample]:
        """把多个数据源合成一个样本列表。"""

    @abstractmethod
    def __len__(self) -> int:
        ...


class ConcatMixer(DataMixer):
    """最简单的混合策略：按顺序拼接所有数据源。

    结果：[源A的全部样本, 源B的全部样本, 源C的全部样本, ...]

    适用场景：数据源之间没有数量差异太大的情况，或者你不在意采样比例。
    DataLoader 的 shuffle=True 会在训练时打乱顺序。
    """

    def __init__(self, **kwargs):
        # **kwargs 是为了兼容 registry.build() 可能传入的额外参数
        self._length = 0

    def mix(self, sources: List[DataSource]) -> List[CanonicalSample]:
        result = []
        for src in sources:
            # 1. 从 adapter 里遍历所有样本（adapter 实现了 __iter__）
            samples = list(src.adapter)

            # 2. 如果设了 max_samples，只取前 N 条
            if src.max_samples > 0:
                samples = samples[:src.max_samples]

            # 3. 给每个样本追加该数据源的 instruction_suffix
            for s in samples:
                result.append(_apply_suffix(s, src.instruction_suffix))

        self._length = len(result)
        return result

    def __len__(self) -> int:
        return self._length


class WeightedInterleaveMixer(DataMixer):
    """按权重交错采样多个数据源。

    每一步按概率选一个数据源，从中取下一个样本。

    例子：
        源A (weight=1.0, 100条): LLaVA 通用对话
        源B (weight=0.5, 50条):  OCR 文字识别

        概率: A = 1.0/(1.0+0.5) = 67%, B = 0.5/(1.0+0.5) = 33%

        结果: [A₁, A₂, B₁, A₃, A₄, B₂, A₅, B₃, A₆, ...]
                ↑    ↑    ↑    ↑    ↑    ↑
               67%的概率选A  33%的概率选B

    用途：LLaVA-1.5 训练时混合 10 个数据集，每个数据集有不同的权重和
    instruction_suffix，用这个 mixer 保证训练过程中各数据集均匀出现。

    seed=42 保证结果可复现。同样的数据+同样的 seed = 同样的混合顺序。
    """

    def __init__(self, seed: int = 42, **kwargs):
        self._seed = seed
        self._length = 0

    def mix(self, sources: List[DataSource]) -> List[CanonicalSample]:
        # ── Step 1: 把每个数据源的样本全部加载到内存 ──
        pools = []
        for src in sources:
            samples = list(src.adapter)
            if src.max_samples > 0:
                samples = samples[:src.max_samples]
            # 现在就给每个样本加 suffix（不是训练时加，是混合时就加好）
            samples = [_apply_suffix(s, src.instruction_suffix) for s in samples]
            pools.append(samples)

        if not pools:
            return []

        # ── Step 2: 计算每个数据源的采样概率 ──
        weights = [src.weight for src in sources]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        # 例: weights=[1.0, 0.5] → probs=[0.667, 0.333]

        result = []
        indices = [0] * len(pools)  # 每个源当前读到第几个样本
        rng = random.Random(self._seed)  # 固定种子 → 可复现

        # ── Step 3: 逐个采样，直到所有源都用完 ──
        total_samples = sum(len(p) for p in pools)
        for _ in range(total_samples):
            # 按概率随机选一个数据源
            src_idx = rng.choices(range(len(pools)), weights=probs, k=1)[0]

            if indices[src_idx] < len(pools[src_idx]):
                # 这个源还有剩余 → 取下一个样本
                result.append(pools[src_idx][indices[src_idx]])
                indices[src_idx] += 1
            else:
                # 这个源已经用完了 → 找其他还有剩余的源
                for j in range(len(pools)):
                    if indices[j] < len(pools[j]):
                        result.append(pools[j][indices[j]])
                        indices[j] += 1
                        break

            # 所有源都用完了 → 结束
            if all(indices[j] >= len(pools[j]) for j in range(len(pools))):
                break

        self._length = len(result)
        return result

    def __len__(self) -> int:
        return self._length
