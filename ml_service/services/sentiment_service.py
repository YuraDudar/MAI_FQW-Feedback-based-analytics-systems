import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from models.db_models import ReviewNLP, ReviewSentiment
from core.model_registry import model_registry

import sys
sys.path.insert(0, "/app")
from infrastructure.config import MIN_TEXT_CHARS

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}

_FEMALE_ENDINGS = ("а", "я", "ова", "ева", "ина", "ая", "ья")
_MALE_ENDINGS = ("ов", "ев", "ин", "ский", "цкий", "ий", "ый")


class SentimentService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def process_reviews(self, reviews: list[dict], product_id: int):
        if not reviews:
            return

        nlp_rows = []
        sentiment_inputs = []

        for r in reviews:
            merged = _merge_text(r)
            if not merged or len(merged) < MIN_TEXT_CHARS:
                continue
            nlp_rows.append({
                "review_id": r["review_id"],
                "product_id": product_id,
                "merged_text": merged,
                "tokens_count": len(merged.split()),
                "is_informative": True,
            })
            sentiment_inputs.append((r["review_id"], r.get("reviewer_name", ""), merged))

        for row in nlp_rows:
            stmt = insert(ReviewNLP).values(**row).on_conflict_do_nothing(index_elements=["review_id"])
            await self.db.execute(stmt)
        await self.db.flush()

        if not sentiment_inputs or model_registry.sentiment_model is None:
            return

        texts = [x[2] for x in sentiment_inputs]
        batch_size = 32

        all_labels = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                preds = model_registry.sentiment_model(batch, truncation=True)
                all_labels.extend(preds)
            except Exception as exc:
                logger.error("Ошибка сентимент-модели: %s", exc)
                all_labels.extend([{"label": "neutral", "score": 0.5}] * len(batch))

        for (review_id, reviewer_name, _), pred in zip(sentiment_inputs, all_labels):
            label = _LABEL_MAP.get(pred["label"], "neutral")
            score = pred.get("score", 0.5)
            gender = _detect_gender(reviewer_name, model_registry.morph)

            stmt = insert(ReviewSentiment).values(
                review_id=review_id,
                product_id=product_id,
                sentiment_label=label,
                sentiment_score=score,
                reviewer_gender=gender,
            ).on_conflict_do_update(
                index_elements=["review_id"],
                set_={"sentiment_label": label, "sentiment_score": score, "reviewer_gender": gender},
            )
            await self.db.execute(stmt)

        await self.db.flush()
        logger.info("Обработан сентимент для %d отзывов", len(sentiment_inputs))


def _merge_text(review: dict) -> str:
    parts = []
    if review.get("advantages"):
        parts.append(f"Достоинства: {review['advantages']}")
    if review.get("disadvantages"):
        parts.append(f"Недостатки: {review['disadvantages']}")
    if review.get("comment"):
        parts.append(f"Комментарий: {review['comment']}")
    return " ".join(parts).strip()


def _detect_gender(name: str | None, morph) -> str:
    if not name:
        return "unknown"
    try:
        if morph:
            parts = name.strip().split()
            if parts:
                parsed = morph.parse(parts[0])[0]
                tags = str(parsed.tag)
                if "femn" in tags:
                    return "female"
                if "masc" in tags:
                    return "male"
        name_lower = name.lower()
        for ending in _FEMALE_ENDINGS:
            if name_lower.endswith(ending):
                return "female"
        for ending in _MALE_ENDINGS:
            if name_lower.endswith(ending):
                return "male"
    except Exception:
        pass
    return "unknown"
