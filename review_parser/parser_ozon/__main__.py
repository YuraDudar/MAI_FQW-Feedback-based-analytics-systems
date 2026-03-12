"""
Entry point for the Ozon review parser (Selenium-based).

    python parser_ozon <SKU[,SKU2,...]> [--format excel csv] [--count N]
                       [--stars N] [--output-dir DIR] [--order ORDER]
                       [--headless] [--verbose]
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ozon_parser_util.cli.args import build_parser, parse_skus
from ozon_parser_util.config import FILENAME_TEMPLATE, OUTPUT_DIR
from ozon_parser_util.core.extractor import OzonReviewExtractor
from ozon_parser_util.core.fetcher import OzonBrowser
from ozon_parser_util.exporters.csv_exporter import CsvExporter
from ozon_parser_util.exporters.excel_exporter import ExcelExporter
from ozon_parser_util.utils.logger import setup_logging

logger = logging.getLogger(__name__)

_EXPORTER_MAP = {
    "excel": (ExcelExporter, "xlsx"),
    "csv":   (CsvExporter,   "csv"),
}


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    setup_logging(verbose=args.verbose)

    skus = parse_skus(args.sku)
    if not skus:
        parser.error("Не указан ни один SKU.")

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    browser   = OzonBrowser(headless=args.headless)
    extractor = OzonReviewExtractor(browser)

    all_reviews = []

    try:
        for sku in skus:
            logger.info("=" * 60)
            logger.info("Обработка SKU: %s", sku)
            reviews = extractor.extract(
                product_id=sku,
                limit=args.count,
                stars=args.stars,
                sort=args.order,
            )
            if not reviews:
                logger.warning("SKU %s: отзывы не найдены", sku)
            else:
                logger.info("SKU %s: собрано %d отзывов", sku, len(reviews))
            all_reviews.extend(reviews)
    finally:
        browser.close()

    if not all_reviews:
        logger.warning("Нет отзывов для выгрузки. Проверьте SKU.")
        sys.exit(0)

    logger.info("=" * 60)
    logger.info("Итого отзывов: %d", len(all_reviews))

    sku_part  = _build_sku_label(skus)
    base_name = FILENAME_TEMPLATE.format(sku_part=sku_part, timestamp=timestamp)

    for fmt in args.formats:
        exporter_cls, ext = _EXPORTER_MAP[fmt]
        output_path = output_dir / f"{base_name}.{ext}"
        exporter_cls().export(all_reviews, output_path)

    logger.info("Готово!")


def _build_sku_label(skus: list[str]) -> str:
    if len(skus) <= 3:
        return "_".join(skus)
    return "_".join(skus[:3]) + f"_and_{len(skus) - 3}_more"


if __name__ == "__main__":
    main()