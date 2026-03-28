from __future__ import annotations

import json


SYSTEM_PROMPT = """Ты опытный русскоязычный аналитик клиентских отзывов.
Твоя задача: придумывать короткие, точные и различимые названия тем (топиков).

Правила:
1) Пиши на русском языке.
2) Название 2-5 слов, без точки в конце.
3) Название должно отражать реальную суть отзывов, а не общий шум.
4) Избегай повторов с уже использованными названиями.
5) Не используй слишком общие слова вроде "Проблемы", "Разное", "Прочее".
6) Не используй эмодзи и кавычки.
7) Верни только JSON: {"title":"...","reason":"кратко"}.
"""


def build_user_prompt(
    pool_name: str,
    topic_id: int,
    keywords: list[str],
    used_titles: list[str],
    sample_reviews: list[str],
) -> str:
    payload = {
        "pool_name": pool_name,
        "topic_id": topic_id,
        "keywords": keywords,
        "used_titles": used_titles,
        "sample_reviews": sample_reviews,
        "task": "Придумай одно короткое название для этого топика.",
    }
    return (
        "Контекст топика в формате JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Верни строго JSON вида: {\"title\":\"...\",\"reason\":\"...\"}"
    )


def build_batch_user_prompt(pool_name: str, topic_payloads: list[dict]) -> str:
    payload = {
        "pool_name": pool_name,
        "topics": topic_payloads,
        "task": (
            "Для каждого topic_id придумай короткий заголовок. "
            "Заголовки должны быть различимы между собой."
        ),
        "output_format": {
            "titles": [
                {"topic_id": 0, "title": "Пример", "reason": "Короткое объяснение"},
            ]
        },
    }
    return (
        "Контекст группы топиков в формате JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Верни строго JSON вида: "
        "{\"titles\":[{\"topic_id\":...,\"title\":\"...\",\"reason\":\"...\"}]}"
    )
