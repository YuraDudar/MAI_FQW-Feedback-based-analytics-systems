"""CSV exporter for Ozon reviews — UTF-8 BOM for Excel compatibility."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from ozon_parser_util.core.models import OzonReview
from ozon_parser_util.exporters.base import BaseExporter

logger = logging.getLogger(__name__)


class CsvExporter(BaseExporter):
    def export(self, reviews: list[OzonReview], output_path: Path) -> None:
        self._ensure_parent(output_path)

        if not reviews:
            logger.warning("CsvExporter: no reviews — file not created")
            return

        with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(OzonReview.column_names())
            for review in reviews:
                writer.writerow(list(review.to_dict().values()))

        logger.info("CSV saved  → %s  (%d rows)", output_path, len(reviews))