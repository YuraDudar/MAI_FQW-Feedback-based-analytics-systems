"""
Command-line interface definition for parser_wb.

Usage:
    python parser_wb <sku[,sku2,...]> [OPTIONS]

Examples:
    python parser_wb 12345678
    python parser_wb 12345678,87654321 --format csv excel
    python parser_wb 12345678 --count 200 --stars 5
    python parser_wb 12345678 --format excel --output-dir ~/reports --order dateAsc
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parser_wb",
        description="Парсер отзывов покупателей с Wildberries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )

    parser.add_argument(
        "sku",
        metavar="SKU",
        help=(
            "Артикул(ы) товара через запятую без пробелов. "
            "Например: 12345678  или  12345678,87654321,11223344"
        ),
    )

    parser.add_argument(
        "--format", "-f",
        nargs="+",
        choices=["excel", "csv"],
        default=["excel", "csv"],
        metavar="FORMAT",
        dest="formats",
        help=(
            "Форматы выгрузки: excel / csv (можно указать оба через пробел). "
            "По умолчанию: excel csv"
        ),
    )

    parser.add_argument(
        "--count", "-c",
        type=_positive_int,
        default=None,
        metavar="N",
        dest="count",
        help=(
            "Максимальное количество скачиваемых отзывов на один SKU. "
            "По умолчанию: все доступные отзывы."
        ),
    )

    parser.add_argument(
        "--stars", "-s",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        metavar="N",
        dest="stars",
        help="Скачивать только отзывы с N звёздами (1–5). По умолчанию: все оценки.",
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
        help="Порядок сортировки отзывов. По умолчанию: dateDesc (сначала новые).",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Выводить отладочные сообщения (DEBUG-уровень логирования).",
    )

    return parser


def parse_skus(raw: str) -> list[str]:
    """Split comma-separated SKU string into a clean list."""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _positive_int(value: str) -> int:
    num = int(value)
    if num <= 0:
        raise argparse.ArgumentTypeError(
            f"Должно быть положительным числом, получено: {value}"
        )
    return num


_EPILOG = """
Примеры использования:
  # Скачать все отзывы по одному артикулу (Excel + CSV)
  python parser_wb 12345678

  # Только 100 отзывов в CSV
  python parser_wb 12345678 --format csv --count 100

  # Только пятизвёздочные, Excel, в папку reports/
  python parser_wb 12345678 --stars 5 --format excel --output-dir reports

  # Несколько артикулов сразу
  python parser_wb 12345678,87654321,11223344 --format csv excel

  # С подробным логированием
  python parser_wb 12345678 --verbose
"""