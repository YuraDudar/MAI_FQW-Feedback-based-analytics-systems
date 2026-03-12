from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from ozon_parser_util.core.models import OzonReview


class BaseExporter(ABC):
    @abstractmethod
    def export(self, reviews: list[OzonReview], output_path: Path) -> None: ...

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)