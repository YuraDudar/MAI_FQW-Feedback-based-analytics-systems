"""CLI definition for parser_ozon."""
from __future__ import annotations
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parser_ozon",
        description="Парсер отзывов покупателей с Ozon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )

    parser.add_argument(
        "sku",
        metavar="SKU",
        help=(
            "ID товара(ов) через запятую без пробелов. "
            "Например: 1716667850  или  1716667850,987654321"
        ),
    )

    parser.add_argument(
        "--format", "-f",
        nargs="+",
        choices=["excel", "csv"],
        default=["excel", "csv"],
        metavar="FORMAT",
        dest="formats",
        help="Форматы выгрузки: excel / csv. По умолчанию: оба.",
    )

    parser.add_argument(
        "--count", "-c",
        type=_positive_int,
        default=None,
        metavar="N",
        dest="count",
        help="Максимальное количество отзывов на один SKU. По умолчанию: все.",
    )

    parser.add_argument(
        "--stars", "-s",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        metavar="N",
        dest="stars",
        help="Скачивать только отзывы с N звёздами (1–5). По умолчанию: все.",
    )

    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        metavar="DIR",
        dest="output_dir",
        help="Папка для сохранения файлов. По умолчанию: ./results",
    )

    parser.add_argument(
        "--order",
        choices=["dateDesc", "dateAsc", "rating"],
        default="dateDesc",
        dest="order",
        help="Порядок сортировки. По умолчанию: dateDesc (сначала новые).",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Запускать браузер в скрытом режиме (без окна). По умолчанию: показывать.",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="DEBUG-уровень логирования.",
    )

    return parser


def parse_skus(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _positive_int(value: str) -> int:
    num = int(value)
    if num <= 0:
        raise argparse.ArgumentTypeError(f"Должно быть > 0, получено: {value}")
    return num


_EPILOG = """
Примеры использования:
  python parser_ozon 1594745546
  python parser_ozon 1594745546 --format csv --count 100
  python parser_ozon 1594745546 --stars 5 --format excel --output-dir reports
  python parser_ozon 1594745546,987654321 --format csv excel
  python parser_ozon 1594745546 --headless --verbose
"""