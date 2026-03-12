"""
Domain model for a single WB customer review.

Naming convention:
  - Python field names: English snake_case  → used as CSV / DB column names
  - COLUMN_RU: Russian display names        → used as Excel column headers
  - DATA_DICT: full field descriptions      → rendered as Data Dictionary in Excel
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from math import nan as NAN
from typing import Any


@dataclass
class Review:
    # ── 1. Метаданные парсинга ────────────────────────────────────────────────
    parsed_at:   Any = NAN   # дата и время запуска парсера
    input_sku:   Any = NAN   # SKU, поданный пользователем на вход

    # ── 2. Идентификация отзыва ───────────────────────────────────────────────
    review_id:      Any = NAN
    nm_id:          Any = NAN
    wb_user_id:     Any = NAN
    global_user_id: Any = NAN

    # ── 3. Автор ─────────────────────────────────────────────────────────────
    reviewer_name:       Any = NAN
    reviewer_country:    Any = NAN
    reviewer_has_avatar: Any = NAN

    # ── 4. Контент отзыва ─────────────────────────────────────────────────────
    rating:        Any = NAN
    advantages:    Any = NAN
    disadvantages: Any = NAN
    comment:       Any = NAN
    variant_color: Any = NAN
    size:          Any = NAN
    tags:          Any = NAN   # bables → строка через "; "

    # ── 5. Даты и статус ─────────────────────────────────────────────────────
    created_date:    Any = NAN
    updated_date:    Any = NAN
    status_id:       Any = NAN
    purchase_status: Any = NAN

    # ── 6. Ответ продавца ────────────────────────────────────────────────────
    seller_response:       Any = NAN
    seller_response_state: Any = NAN

    # ── 7. Соответствия ──────────────────────────────────────────────────────
    matching_size:        Any = NAN
    matching_photo:       Any = NAN
    matching_description: Any = NAN

    # ── 8. Голоса и ранжирование ─────────────────────────────────────────────
    votes_plus:  Any = NAN
    votes_minus: Any = NAN
    rank:        Any = NAN
    helpfulness: Any = NAN

    # ── 9. Медиа ─────────────────────────────────────────────────────────────
    has_video: Any = NAN

    # ── 10. Исключение из рейтинга ───────────────────────────────────────────
    excluded_from_rating: Any = NAN
    excluded_reasons:     Any = NAN

    # ── 11. Причины оценки ───────────────────────────────────────────────────
    good_reasons: Any = NAN
    bad_reasons:  Any = NAN

    # ── Interface ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """English snake_case keys — for CSV / DB."""
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    @classmethod
    def field_names(cls) -> list[str]:
        """Ordered list of English snake_case field names."""
        return [f.name for f in dc_fields(cls)]

    @classmethod
    def column_names_ru(cls) -> list[str]:
        """Russian display names in field order — for Excel headers."""
        return [COLUMN_RU[name] for name in cls.field_names()]


# ── Russian display names (Excel headers) ────────────────────────────────────

COLUMN_RU: dict[str, str] = {
    "parsed_at":              "Дата и время парсинга",
    "input_sku":              "Входной SKU",
    "review_id":              "ID отзыва",
    "nm_id":                  "SKU варианта товара",
    "wb_user_id":             "ID пользователя WB",
    "global_user_id":         "Глобальный ID пользователя",
    "reviewer_name":          "Имя покупателя",
    "reviewer_country":       "Страна",
    "reviewer_has_avatar":    "Есть аватар",
    "rating":                 "Оценка (1–5 ★)",
    "advantages":             "Достоинства",
    "disadvantages":          "Недостатки",
    "comment":                "Комментарий",
    "variant_color":          "Вариант / цвет товара",
    "size":                   "Размер",
    "tags":                   "Теги-плюсы",
    "created_date":           "Дата отзыва",
    "updated_date":           "Дата обновления",
    "status_id":              "Код статуса покупки",
    "purchase_status":        "Статус покупки",
    "seller_response":        "Ответ продавца",
    "seller_response_state":  "Источник ответа",
    "matching_size":          "Соответствие размера",
    "matching_photo":         "Соответствие фото",
    "matching_description":   "Соответствие описанию",
    "votes_plus":             "Голосов «полезно»",
    "votes_minus":            "Голосов «не полезно»",
    "rank":                   "Рейтинг отзыва",
    "helpfulness":            "Полезность",
    "has_video":              "Есть видео",
    "excluded_from_rating":   "Исключён из рейтинга",
    "excluded_reasons":       "Причины исключения",
    "good_reasons":           "Коды положительных причин",
    "bad_reasons":            "Коды отрицательных причин",
}


# ── Data Dictionary (field_name, ru_name, type, group, description) ──────────

DATA_DICT: list[tuple[str, str, str, str, str]] = [
    ("parsed_at",              "Дата и время парсинга",       "datetime",  "Метаданные",          "Временная метка запуска парсера"),
    ("input_sku",              "Входной SKU",                 "str",       "Метаданные",          "nmId, введённый пользователем в команде запуска"),
    ("review_id",              "ID отзыва",                   "str",       "Идентификация",       "Уникальный идентификатор отзыва в системе WB"),
    ("nm_id",                  "SKU варианта товара",         "int",       "Идентификация",       "nmId конкретного варианта (цвет, объём, размер)"),
    ("wb_user_id",             "ID пользователя WB",          "int",       "Идентификация",       "Числовой идентификатор покупателя в WB"),
    ("global_user_id",         "Глобальный ID пользователя",  "str",       "Идентификация",       "Глобальный UUID пользователя"),
    ("reviewer_name",          "Имя покупателя",              "str",       "Автор",               "Отображаемое имя из профиля WB"),
    ("reviewer_country",       "Страна",                      "str",       "Автор",               "Страна покупателя (код ISO, например 'ru')"),
    ("reviewer_has_avatar",    "Есть аватар",                 "bool",      "Автор",               "True — пользователь загрузил фото профиля"),
    ("rating",                 "Оценка (1–5 ★)",              "int",       "Контент отзыва",      "Рейтинг товара от 1 до 5 звёзд"),
    ("advantages",             "Достоинства",                 "str",       "Контент отзыва",      "Поле «Достоинства» из формы отзыва"),
    ("disadvantages",          "Недостатки",                  "str",       "Контент отзыва",      "Поле «Недостатки» из формы отзыва"),
    ("comment",                "Комментарий",                 "str",       "Контент отзыва",      "Основной текст отзыва (поле «Комментарий»)"),
    ("variant_color",          "Вариант / цвет товара",       "str",       "Контент отзыва",      "Выбранный вариант товара: цвет, объём и т.д."),
    ("size",                   "Размер",                      "str",       "Контент отзыва",      "Выбранный размер товара"),
    ("tags",                   "Теги-плюсы",                  "str",       "Контент отзыва",      "Теги-плюсы, отмеченные покупателем (через '; ')"),
    ("created_date",           "Дата отзыва",                 "datetime",  "Даты и статус",       "Дата и время публикации отзыва (UTC)"),
    ("updated_date",           "Дата обновления",             "datetime",  "Даты и статус",       "Дата последнего редактирования отзыва (UTC)"),
    ("status_id",              "Код статуса покупки",         "int",       "Даты и статус",       "8=Вернули · 14=Отказались · 16=Выкупили · 120=Претензия"),
    ("purchase_status",        "Статус покупки",              "str",       "Даты и статус",       "Статус покупки в виде текстовой метки"),
    ("seller_response",        "Ответ продавца",              "str",       "Ответ продавца",      "Текст официального ответа продавца на отзыв"),
    ("seller_response_state",  "Источник ответа",             "str",       "Ответ продавца",      "'wbRu' — ответ от имени WB, иначе — сам продавец"),
    ("matching_size",          "Соответствие размера",        "str",       "Соответствия",        "Комментарий покупателя о соответствии размера"),
    ("matching_photo",         "Соответствие фото",           "str",       "Соответствия",        "Комментарий о соответствии фото реальному товару"),
    ("matching_description",   "Соответствие описанию",       "str",       "Соответствия",        "Комментарий о соответствии описанию на сайте"),
    ("votes_plus",             "Голосов «полезно»",           "int",       "Голоса и рейтинг",    "Число пользователей, нашедших отзыв полезным"),
    ("votes_minus",            "Голосов «не полезно»",        "int",       "Голоса и рейтинг",    "Число пользователей, нашедших отзыв бесполезным"),
    ("rank",                   "Рейтинг отзыва",              "float",     "Голоса и рейтинг",    "Внутренний скор WB: чем выше — тем «полезнее» отзыв"),
    ("helpfulness",            "Полезность",                  "str/null",  "Голоса и рейтинг",    "Сводка голосов полезности: 'plus: N, minus: M'"),
    ("has_video",              "Есть видео",                  "str",       "Медиа",               "'Да' — к отзыву прикреплено видео, 'Нет' — отсутствует"),
    ("excluded_from_rating",   "Исключён из рейтинга",        "bool",      "Исключение",          "True — отзыв не учитывается в итоговом рейтинге товара"),
    ("excluded_reasons",       "Причины исключения",          "str",       "Исключение",          "Коды причин исключения из рейтинга (через '; ')"),
    ("good_reasons",           "Коды положительных причин",   "str",       "Причины оценки",      "Числовые коды позитивных факторов отзыва (через '; ')"),
    ("bad_reasons",            "Коды отрицательных причин",   "str",       "Причины оценки",      "Числовые коды негативных факторов отзыва (через '; ')"),
]