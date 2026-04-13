"""DataFilter — pre-training data quality filtering.

Stage 3 的第二步。Mixer 混合完之后，Filter 对每个样本做质量检查，
不合格的直接丢掉。

在 stage_runner.py 里的调用：
    if stage.filter_config:
        filter_obj = registry.build("filter", filter_config["type"], **params)
        samples = [s for s in samples if filter_obj.filter(s)]

每个 Filter 就是一个函数：输入 CanonicalSample，输出 True（保留）或 False（丢弃）。

YAML 配置示例：
    filter_config:
      type: "composite"
      params:
        filters:
          - {type: "text_length", min_length: 10, max_length: 5000}
        logic: "and"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from mmit.data.types import CanonicalSample


class DataFilter(ABC):
    """数据过滤器基类。

    所有 filter 注册在 Registry 的 "filter" slot：
        registry.register("filter", "composite", CompositeFilter)
        registry.register("filter", "text_length", TextLengthFilter)

    自定义 filter 只需要实现 filter() 方法：
        class MyFilter(DataFilter):
            def filter(self, sample):
                return sample.image_path != ""  # 丢掉没有图片的样本
    """

    @abstractmethod
    def filter(self, sample: CanonicalSample) -> bool:
        """检查一个样本是否合格。True = 保留，False = 丢弃。"""

    def score(self, sample: CanonicalSample) -> float:
        """返回 0~1 的质量分数。默认实现：通过=1.0，不通过=0.0。
        可以被子类覆盖实现更细粒度的评分。"""
        return 1.0 if self.filter(sample) else 0.0


class CompositeFilter(DataFilter):
    """组合过滤器：把多个 filter 用 AND/OR 逻辑链起来。

    例子：
        # 同时满足：文本长度 >= 10 AND 文本长度 <= 5000
        composite = CompositeFilter(
            filters=[
                TextLengthFilter(min_length=10),
                TextLengthFilter(max_length=5000),
            ],
            logic="and"  # 两个都要通过
        )

        # 满足任一：文本包含 "describe" OR 文本包含 "explain"
        composite = CompositeFilter(filters=[...], logic="or")

    logic="and"：所有子 filter 都返回 True 才保留（更严格）
    logic="or"：任意一个子 filter 返回 True 就保留（更宽松）
    """

    def __init__(self, filters: List[DataFilter], logic: str = "and", **kwargs):
        self.filters = filters
        self.logic = logic.lower()

    def filter(self, sample: CanonicalSample) -> bool:
        if self.logic == "and":
            # 所有 filter 都要通过。all() 遇到第一个 False 就短路返回。
            return all(f.filter(sample) for f in self.filters)
        elif self.logic == "or":
            # 任意一个通过即可。any() 遇到第一个 True 就短路返回。
            return any(f.filter(sample) for f in self.filters)
        raise ValueError(f"Unknown logic: {self.logic}")


class TextLengthFilter(DataFilter):
    """按文本总长度过滤样本。

    计算方式：把所有 turn 的 content 长度加起来（字符数，不是 token 数）。

    用途：
      - min_length=1: 过滤掉空样本（content 全是空字符串的）
      - max_length=10000: 过滤掉异常超长的样本（可能是数据错误）
      - 两者结合使用确保样本在合理范围内

    例子：
        filter = TextLengthFilter(min_length=10, max_length=5000)
        sample.turns = [Turn("human", "Hi"), Turn("assistant", "Hello")]
        total_len = len("Hi") + len("Hello") = 7
        filter.filter(sample)  # → False (7 < 10，太短了)
    """

    def __init__(self, min_length: int = 1, max_length: int = 10000, **kwargs):
        self.min_length = min_length
        self.max_length = max_length

    def filter(self, sample: CanonicalSample) -> bool:
        # 所有 turn 的 content 字符数之和
        total_len = sum(len(t.content) for t in sample.turns)
        return self.min_length <= total_len <= self.max_length
