from __future__ import annotations

from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    @abstractmethod
    def process(self, input_dir: str, output_dir: str):
        """Read raw dataset, output unified format (data.jsonl, statistics.yaml, images/, metadata.yaml)."""
        ...
