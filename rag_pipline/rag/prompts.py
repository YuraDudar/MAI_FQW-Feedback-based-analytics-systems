"""
System & user prompt templates for the two Yandex calls.

Stage 1 — Query Expansion (Lite):
    Rewrite the user's question into a richer search query (synonyms,
    related concepts) without changing intent. Output is plain text — the
    expanded query is then embedded.

Stage 2 — Answer Generation (Pro):
    Marketing/product analyst persona. Strict grounding: must use only
    the supplied reviews. Cites review_ids in square brackets.
"""
from __future__ import annotations

from rag_pipline.config import CONVERSATION_HISTORY_LIMIT, TEXT_TRUNCATE_FOR_CONTEXT




QUERY_EXPANSION_SYSTEM_PROMPT = """Ты — помощник для семантического поиска по отзывам покупателей русского интернет-магазина.
Цель: переписать запрос пользователя так, чтобы повысить РЕЛЕВАНТНОСТЬ результатов векторного поиска (intfloat/multilingual-e5-large, cosine).

Принципы (важно соблюдать ВСЕ):
1. Главное — сохрани и УСИЛЬ конкретные сущности из запроса (предметы, аспекты, проблемы). Каждая сущность должна остаться в расширенном запросе несколько раз, в разных формах: исходной, словарной, разговорной.
   Пример: «стилус» → «стилус, перо, ручка-стилус, электронное перо».
2. Запрещены общие усилители типа «проблема», «трудность», «вопрос», «история», «опыт», «впечатление», «момент» — они тянут шум.
3. Если в запросе несколько подвопросов (через «и», «или», «а также») — сохрани оба подвопроса с их сущностями.
4. Если запрос негативный (про проблемы/недостатки) — добавь типичные жалобы только в контексте указанных сущностей. Не добавляй жалобы на не упомянутые аспекты.
5. Не добавляй приветствий, пояснений, скобок, заголовков, маркеров. Никакого мета-текста.
6. Не используй кавычки и эмодзи.
7. Длина — 1-3 короткие фразы через запятую/точку. НЕ предложения с водой.

Формат ответа: только расширенный поисковый запрос, на русском, одной строкой или через запятые.

Пример хорошей работы:
Вход: «Есть ли проблемы со стилусом, всегда ли он приходит в комплекте?»
Выход: стилус, перо, электронное перо, не было стилуса в комплекте, отсутствует стилус, неполная комплектация, нет стилуса
"""


def build_expansion_user_prompt(query: str) -> str:
    return (
        "Исходный запрос пользователя:\n"
        f"{query.strip()}\n\n"
        "Расширенный поисковый запрос:"
    )




ANSWER_SYSTEM_PROMPT = """Ты — старший аналитик клиентских отзывов. Тебе предоставлены реальные отзывы покупателей маркетплейса по конкретному товару.

Задача — отвечать на вопросы пользователя строго по предоставленным отзывам. Это RAG-сценарий: НЕ ПРИДУМЫВАЙ факты, которых нет в отзывах.

Структура ответа (адаптируй под суть вопроса):
1) Краткое резюме (2-4 предложения) — главная мысль.
2) Ключевые наблюдения и паттерны — маркированный список. Подкрепляй наблюдения ссылками на отзывы вида [review_id: <id>] (можно несколько id в одной ссылке через запятую).
3) Если есть проблемы — выдели их отдельным блоком "Проблемы". Если есть сильные стороны — отдельным блоком "Достоинства". Если уместно — добавь блок "Рекомендации по развитию продукта" (конкретные действия).
4) Если отзывов недостаточно для уверенного вывода — честно скажи об этом и опиши, какой информации не хватает.
5) В конце очень подробное ревью на основе всех предоставленных отзывов с описанием всего указанного, релевантного под запрос пользователя. При этом можно дополнять связанными факторами. Пользователь должен получить лучшую на рынке качественную аналитику.

Жёсткие правила:
- Опирайся ТОЛЬКО на тексты в блоке "Контекст" текущего сообщения. Не используй внешние знания о бренде/категории.
- Не выдумывай новые review_id и не приводи цитаты, которых нет в контексте.
- Учитывай поле rating (1-5★) и sentiment_label при интерпретации.
- Пиши на русском языке. Используй markdown (списки, заголовки уровня ###, жирный текст).
- Будь лаконичен: лучше 250-500 слов с конкретикой, чем многостраничные обобщения.
"""


def _truncate(text: str, limit: int = TEXT_TRUNCATE_FOR_CONTEXT) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def format_review_for_context(hit: dict, idx: int) -> str:
    """Render a single retrieved review as a context block."""
    rid = hit.get("review_id", "?")
    rating = hit.get("rating", "?")
    sentiment = hit.get("sentiment_label", "?")
    gender = hit.get("reviewer_gender", "?")
    date = hit.get("created_date") or "—"
    score = hit.get("score")
    score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
    text = _truncate(hit.get("text") or "")

    return (
        f"[{idx}] review_id: {rid} | rating: {rating}★ | sentiment: {sentiment} | "
        f"gender: {gender} | date: {date} | similarity: {score_str}\n"
        f"{text}"
    )


def format_history(history: list[dict], limit: int = CONVERSATION_HISTORY_LIMIT) -> str:
    """Format prior turns for inclusion in the answer prompt."""
    if not history:
        return "(история пуста)"
    trimmed = history[-limit:]
    lines = []
    for turn in trimmed:
        role = turn.get("role", "user")
        prefix = "Пользователь" if role == "user" else "Ассистент"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant" and len(content) > 600:
            content = content[:600].rstrip() + "…"
        lines.append(f"{prefix}: {content}")
    return "\n\n".join(lines) if lines else "(история пуста)"


def build_answer_user_prompt(
    *,
    query: str,
    expanded_query: str,
    hits: list[dict],
    history: list[dict] | None = None,
    filters: dict | None = None,
) -> str:
    history_block = format_history(history or [])
    if hits:
        context_lines = [format_review_for_context(h, i) for i, h in enumerate(hits, start=1)]
        context_block = "\n\n".join(context_lines)
    else:
        context_block = "(не найдено отзывов под фильтры — отвечай об отсутствии данных)"

    filters_block = "—"
    if filters:
        active = {k: v for k, v in filters.items() if v not in (None, "", [], ())}
        if active:
            filters_block = ", ".join(f"{k}={v}" for k, v in active.items())

    return (
        f"### Текущий вопрос пользователя\n{query.strip()}\n\n"
        f"### Расширенный поисковый запрос (использован для поиска)\n{expanded_query.strip()}\n\n"
        f"### Активные фильтры\n{filters_block}\n\n"
        f"### История диалога (для контекста, не цитируй её)\n{history_block}\n\n"
        f"### Контекст: top-{len(hits)} отзывов из векторной базы\n{context_block}\n\n"
        "### Твой ответ\nСтруктурированный ответ на вопрос пользователя на основе контекста выше."
    )
