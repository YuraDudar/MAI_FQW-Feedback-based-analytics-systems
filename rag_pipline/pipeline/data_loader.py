"""
CSV ingestion + per-review record preparation for the RAG indexing pipeline.

Differs from klasteristion_pipline/pipeline/data_loader.py:
* Keeps reviewer_name and created_date (needed for payload)
* Produces a flat list of `Record` dicts ready for indexer
* Smart concatenation with structural prefixes (mirrors klasteristion behaviour)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from rag_pipline.config import (
    CSV_DIR,
    CREATED_DATE_FIELD,
    MIN_TEXT_CHARS,
    PRODUCT_ID_FIELD,
    RATING_FIELD,
    REVIEW_ID_FIELD,
    REVIEWER_NAME_FIELD,
    TEXT_FIELDS,
    VARIANT_ID_FIELD,
)

log = logging.getLogger(__name__)

_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")
_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "]+",
    flags=re.UNICODE,
)


@dataclass
class Record:
    """One review prepared for indexing."""
    review_id: str
    product_id: int            
    variant_id: int            
    rating: int                
    created_date: str | None   
    reviewer_name: str         
    advantages: str
    disadvantages: str
    comment: str
    combined_text: str         

    sentiment_label: str | None = None
    reviewer_gender: str | None = None

    def to_payload(self) -> dict:
        """Strip text fields if you want a smaller payload — kept full for UX."""
        payload = {
            "review_id": self.review_id,
            "product_id": int(self.product_id) if self.product_id is not None else None,
            "variant_id": int(self.variant_id) if self.variant_id is not None else None,
            "rating": int(self.rating) if self.rating is not None else None,
            "created_date": self.created_date,
            "reviewer_name": self.reviewer_name or None,
            "reviewer_gender": self.reviewer_gender or "unknown",
            "sentiment_label": self.sentiment_label or "neutral",
            "text": self.combined_text,
            "advantages": self.advantages,
            "disadvantages": self.disadvantages,
            "comment": self.comment,
        }
        return payload



def _clean_text(text: str) -> str:
    if not text or text in ("nan", "None", "none"):
        return ""
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMOJI.sub(" ", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def build_combined_text(advantages: str, disadvantages: str, comment: str) -> str:
    """Smart concatenation with structural prefixes (preserves sentiment cues)."""
    parts: list[str] = []
    if advantages:
        parts.append(f"Достоинства: {advantages}")
    if disadvantages:
        parts.append(f"Недостатки: {disadvantages}")
    if comment:
        parts.append(f"Комментарий: {comment}")
    return ". ".join(parts)


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _normalize_date(value) -> str | None:
    """Return ISO8601 string suitable for Qdrant DatetimeRange, or None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=False)
        if ts is pd.NaT or pd.isna(ts):
            return None

        return ts.isoformat()
    except Exception:
        return None



def list_csv_files(csv_dir: Path = CSV_DIR) -> list[Path]:
    """Return CSV files in csv_dir sorted by mtime (newest first)."""
    if not csv_dir.exists():
        return []
    return sorted(csv_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_reviews(
    csv_path: str | Path,
    *,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Load CSV with all string types, dedup by review_id."""
    csv_path = Path(csv_path)
    log.info("Loading reviews from %s", csv_path.name)
    df = pd.read_csv(csv_path, dtype=str)
    if REVIEW_ID_FIELD in df.columns and deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset=[REVIEW_ID_FIELD], keep="first")
        log.info("Dedup: %d → %d rows", before, len(df))

    for col in TEXT_FIELDS + [REVIEWER_NAME_FIELD]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def list_products(df: pd.DataFrame) -> list[dict]:
    """Distinct product ids present in the CSV (sorted by review count)."""
    if PRODUCT_ID_FIELD not in df.columns:
        return []
    grouped = (
        df.groupby(PRODUCT_ID_FIELD)
        .size()
        .reset_index(name="reviews")
        .sort_values("reviews", ascending=False)
    )
    return [
        {"product_id": _safe_int(row[PRODUCT_ID_FIELD]), "reviews": int(row["reviews"])}
        for _, row in grouped.iterrows()
        if _safe_int(row[PRODUCT_ID_FIELD]) > 0
    ]


def prepare_records(
    df: pd.DataFrame,
    *,
    product_id: int | None = None,
    min_chars: int = MIN_TEXT_CHARS,
) -> list[Record]:
    """
    Convert DataFrame rows to Record objects ready for the pipeline.

    If *product_id* is given, only rows whose `input_sku` matches are kept.
    """
    if product_id is not None and PRODUCT_ID_FIELD in df.columns:
        df = df[df[PRODUCT_ID_FIELD].apply(_safe_int) == int(product_id)]

    records: list[Record] = []
    for idx, row in df.iterrows():
        adv = _clean_text(str(row.get("advantages", "")))
        dis = _clean_text(str(row.get("disadvantages", "")))
        com = _clean_text(str(row.get("comment", "")))
        combined = build_combined_text(adv, dis, com)
        if len(combined) < min_chars:
            continue

        rid = str(row.get(REVIEW_ID_FIELD, idx))
        records.append(
            Record(
                review_id=rid,
                product_id=_safe_int(row.get(PRODUCT_ID_FIELD), 0),
                variant_id=_safe_int(row.get(VARIANT_ID_FIELD), 0),
                rating=_safe_int(row.get(RATING_FIELD), 0),
                created_date=_normalize_date(row.get(CREATED_DATE_FIELD)),
                reviewer_name=str(row.get(REVIEWER_NAME_FIELD, "") or ""),
                advantages=adv,
                disadvantages=dis,
                comment=com,
                combined_text=combined,
            )
        )

    log.info("Prepared %d records (min_chars=%d)", len(records), min_chars)
    return records


def records_summary(records: Iterable[Record]) -> dict:
    """Quick stats — useful for the indexing tab."""
    records = list(records)
    if not records:
        return {"total": 0}
    ratings = [r.rating for r in records if r.rating]
    lengths = [len(r.combined_text) for r in records]
    return {
        "total": len(records),
        "avg_text_len": sum(lengths) / len(lengths),
        "min_text_len": min(lengths),
        "max_text_len": max(lengths),
        "rating_dist": {s: sum(1 for r in ratings if r == s) for s in range(1, 6)},
        "with_date": sum(1 for r in records if r.created_date),
        "with_name": sum(1 for r in records if r.reviewer_name and r.reviewer_name.lower() not in ("", "nan", "none")),
    }
