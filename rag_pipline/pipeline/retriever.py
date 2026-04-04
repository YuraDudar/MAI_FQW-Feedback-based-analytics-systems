"""
Filter construction + hit formatting helpers for the retrieval stage.

All filters are HNSW-level (Qdrant `query_filter`), not post-filtering — this
preserves recall and matches the spec requirement for indexed payload fields.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _normalize_iso(date_value) -> str | None:
    if date_value is None:
        return None
    if isinstance(date_value, str):
        return date_value
    try:
        return date_value.isoformat()
    except Exception:
        return None


def build_qdrant_filter(filters: dict | None):
    """
    Convert a UI filters dict to a Qdrant `Filter` object.

    Filters dict shape (all keys optional):
        {
            "rating_min":     int,                       # 1..5
            "rating_max":     int,                       # 1..5
            "date_from":      str | datetime,            # ISO8601
            "date_to":        str | datetime,            # ISO8601
            "sentiment_labels": ["positive", ...],
            "genders":        ["male", "female"],
        }
    """
    if not filters:
        return None

    from qdrant_client.models import (
        DatetimeRange,
        FieldCondition,
        Filter,
        MatchAny,
        Range,
    )

    must: list[Any] = []

    rmin = filters.get("rating_min")
    rmax = filters.get("rating_max")
    if rmin is not None or rmax is not None:
        must.append(FieldCondition(key="rating", range=Range(gte=rmin, lte=rmax)))

    date_from = _normalize_iso(filters.get("date_from"))
    date_to = _normalize_iso(filters.get("date_to"))
    if date_from or date_to:
        must.append(FieldCondition(
            key="created_date",
            range=DatetimeRange(gte=date_from, lte=date_to),
        ))

    sentiments = filters.get("sentiment_labels")
    if sentiments:
        must.append(FieldCondition(
            key="sentiment_label",
            match=MatchAny(any=list(sentiments)),
        ))

    genders = filters.get("genders")
    if genders:
        must.append(FieldCondition(
            key="reviewer_gender",
            match=MatchAny(any=list(genders)),
        ))

    if not must:
        return None
    return Filter(must=must)


def format_hit(hit) -> dict:
    """Convert a Qdrant ScoredPoint into a flat dict for the UI / LLM context."""
    payload = hit.payload or {}
    return {
        "score": float(getattr(hit, "score", 0.0)),
        "review_id": payload.get("review_id"),
        "rating": payload.get("rating"),
        "sentiment_label": payload.get("sentiment_label"),
        "reviewer_gender": payload.get("reviewer_gender"),
        "created_date": payload.get("created_date"),
        "reviewer_name": payload.get("reviewer_name"),
        "text": payload.get("text") or payload.get("comment") or "",
        "advantages": payload.get("advantages") or "",
        "disadvantages": payload.get("disadvantages") or "",
        "comment": payload.get("comment") or "",
        "product_id": payload.get("product_id"),
        "variant_id": payload.get("variant_id"),
    }
