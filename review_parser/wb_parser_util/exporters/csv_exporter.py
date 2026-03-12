"""
CSV exporter for WB reviews.

- Column headers: English snake_case (field names from the Review dataclass).
- Missing values: written as the literal string 'nan' (pandas-compatible).
- Encoding: UTF-8 with BOM so Excel opens the file correctly by default.
"""
from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any

from wb_parser_util.core.models import Review
from wb_parser_util.exporters.base import BaseExporter

logger = logging.getLogger(__name__)


class CsvExporter(BaseExporter):
    def export(self, reviews: list[Review], output_path: Path) -> None:
        self._ensure_parent(output_path)

        if not reviews:
            logger.warning("CsvExporter: no reviews — file not created")
            return

        with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(Review.field_names())
            for review in reviews:
                writer.writerow(
                    [_to_csv(v) for v in review.to_dict().values()]
                )

        logger.info("CSV saved  → %s  (%d rows)", output_path, len(reviews))


def _to_csv(value: Any) -> str:
    """NaN / None → literal 'nan'; list/dict → JSON string; else as-is."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    if isinstance(value, (list, dict)):
        import json
        return json.dumps(value, ensure_ascii=False)
    return value