"""
Text Preprocessing: cleaning, smart concatenation, heuristic sentiment split.

Key design decisions
--------------------
* NO stemming / lemmatisation — transformer embeddings work on raw tokens.
* Structural prefixes ("Достоинства:", "Недостатки:", "Комментарий:") are
  prepended so the Self-Attention mechanism can leverage document structure.
* Heuristic split isolates negative and positive semantic signals *before*
  embedding, preventing mixed-sentiment clusters.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from config import (
    MIN_TEXT_CHARS,
    NEGATIVE_RATING_THRESHOLD,
    TEXT_FIELDS,
    TAGS_FIELD,
    RATING_FIELD,
    REVIEW_ID_FIELD,
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
class PoolEntry:
    """Single text unit inside a positive or negative pool."""
    text: str
    review_id: str
    rating: int
    source_fields: str
    original_idx: int


@dataclass
class SplitResult:
    negative_pool: list[PoolEntry] = field(default_factory=list)
    positive_pool: list[PoolEntry] = field(default_factory=list)
    skipped_empty: int = 0



def clean_text(text: str) -> str:
    """Light regex cleaning preserving semantics for transformers."""
    if not text:
        return ""
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMOJI.sub(" ", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def smart_concatenate(row: pd.Series) -> str:
    """
    Build a single document from filled text fields with structural prefixes.

    Format: "Достоинства: {adv}. Недостатки: {dis}. Комментарий: {com}."
    Tags are appended without prefix when other fields are empty.
    """
    parts: list[str] = []

    adv = clean_text(str(row.get("advantages", "")))
    dis = clean_text(str(row.get("disadvantages", "")))
    com = clean_text(str(row.get("comment", "")))
    tags = clean_text(str(row.get(TAGS_FIELD, "")))

    if adv:
        parts.append(f"Достоинства: {adv}")
    if dis:
        parts.append(f"Недостатки: {dis}")
    if com:
        parts.append(f"Комментарий: {com}")

    if not parts and tags:
        parts.append(tags)

    return ". ".join(parts)


def preprocess(df: pd.DataFrame, min_chars: int = MIN_TEXT_CHARS) -> pd.DataFrame:
    """
    Add ``combined_text`` column and filter short / empty reviews.

    Returns a copy — original DataFrame is not mutated.
    """
    df = df.copy()

    for col in TEXT_FIELDS + [TAGS_FIELD]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    df["combined_text"] = df.apply(smart_concatenate, axis=1)

    n_before = len(df)
    df = df[df["combined_text"].str.len() >= min_chars].reset_index(drop=True)
    n_after = len(df)
    log.info("Preprocess: %d → %d reviews (dropped %d short/empty)", n_before, n_after, n_before - n_after)

    return df


def heuristic_split(df: pd.DataFrame) -> SplitResult:
    """
    Split preprocessed reviews into Negative and Positive pools.

    Strategy
    --------
    * **Negative pool**
      - Full concatenated text of every review with ``rating ≤ 3``.
      - Isolated ``disadvantages`` field from reviews with ``rating ≥ 4``
        (captures negative aspects mentioned in otherwise positive reviews).

    * **Positive pool**
      - ``advantages`` + ``comment`` from reviews with ``rating ≥ 4``.
    """
    result = SplitResult()
    threshold = NEGATIVE_RATING_THRESHOLD

    for idx, row in df.iterrows():
        rid = str(row.get(REVIEW_ID_FIELD, idx))
        rating = int(row.get(RATING_FIELD, 0))
        combined = row.get("combined_text", "")
        adv = clean_text(str(row.get("advantages", "")))
        dis = clean_text(str(row.get("disadvantages", "")))
        com = clean_text(str(row.get("comment", "")))

        if rating <= threshold:
            if len(combined) >= MIN_TEXT_CHARS:
                result.negative_pool.append(PoolEntry(
                    text=combined,
                    review_id=rid,
                    rating=rating,
                    source_fields="full",
                    original_idx=int(idx),
                ))
            else:
                result.skipped_empty += 1
        else:
            pos_parts = []
            if adv:
                pos_parts.append(f"Достоинства: {adv}")
            if com:
                pos_parts.append(f"Комментарий: {com}")
            pos_text = ". ".join(pos_parts)

            if len(pos_text) >= MIN_TEXT_CHARS:
                result.positive_pool.append(PoolEntry(
                    text=pos_text,
                    review_id=rid,
                    rating=rating,
                    source_fields="advantages+comment",
                    original_idx=int(idx),
                ))

            if dis and len(dis) >= MIN_TEXT_CHARS:
                result.negative_pool.append(PoolEntry(
                    text=f"Недостатки: {dis}",
                    review_id=rid,
                    rating=rating,
                    source_fields="disadvantages_only",
                    original_idx=int(idx),
                ))

    _log_split(result)
    return result


def pool_texts(pool: list[PoolEntry]) -> list[str]:
    """Extract plain text list from a pool (for embedding)."""
    return [e.text for e in pool]


def pool_to_dataframe(pool: list[PoolEntry]) -> pd.DataFrame:
    """Convert pool entries to a DataFrame for inspection."""
    return pd.DataFrame([
        {"text": e.text, "review_id": e.review_id, "rating": e.rating,
         "source_fields": e.source_fields, "original_idx": e.original_idx}
        for e in pool
    ])


def _log_split(result: SplitResult) -> None:
    msg = (
        f"\n{'=' * 60}\n"
        f"  HEURISTIC SPLIT RESULT\n"
        f"{'=' * 60}\n"
        f"  Negative pool : {len(result.negative_pool):,} entries\n"
        f"  Positive pool : {len(result.positive_pool):,} entries\n"
        f"  Skipped empty : {result.skipped_empty:,}\n"
        f"{'=' * 60}\n"
    )
    log.info(msg)
