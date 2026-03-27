"""DatasetAdapter ABC — implement this to plug in any data format."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from mmit.data.types import CanonicalSample


class DatasetAdapter(ABC):
    """Yields :class:`CanonicalSample` objects from any on-disk format.

    Example
    -------
    >>> class MyCSVDataset(DatasetAdapter):
    ...     def __init__(self, csv_path, image_root):
    ...         self._rows = load_csv(csv_path)
    ...         self._root = image_root
    ...     def __iter__(self):
    ...         for r in self._rows:
    ...             yield CanonicalSample(id=r["id"], image_path=r["img"],
    ...                                   turns=[Turn("human", r["q"])])
    ...     def __len__(self):
    ...         return len(self._rows)
    """

    @abstractmethod
    def __iter__(self) -> Iterator[CanonicalSample]: ...

    @abstractmethod
    def __len__(self) -> int: ...
