"""Abstract base class for all WB review exporters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from wb_parser_util.core.models import Review


class BaseExporter(ABC):
    @abstractmethod
    def export(self, reviews: list[Review], output_path: Path) -> None: ...

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)