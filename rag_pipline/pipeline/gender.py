"""
Heuristic gender detection from Russian first names via pymorphy3.

Strategy
--------
1. Take the first whitespace-separated token of the reviewer's display name.
2. Strip non-Cyrillic characters (handles emoji, latin tail-additions).
3. Use pymorphy3 to obtain morphological parses; prefer parses tagged as
   `Name` (proper personal names) — they carry the most reliable gender info.
4. Read `tag.gender` ('masc' → male, 'femn' → female).
5. Return 'unknown' for empty / ambiguous / non-Russian inputs.

The library has no GPU dependency — safe to use in a streaming loop.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

log = logging.getLogger(__name__)

_RE_NON_CYR = re.compile(r"[^Ѐ-ӿЁё\-]+")
_EMPTY_TOKENS = {"", "nan", "none", "null", "anonymous", "аноним"}


class GenderDetector:
    """Thin wrapper around pymorphy3 with deterministic caching."""

    def __init__(self):
        import pymorphy3
        self._morph = pymorphy3.MorphAnalyzer()

    def detect(self, raw_name: str) -> str:
        """Return one of {'male', 'female', 'unknown'}."""
        if not raw_name:
            return "unknown"
        cleaned = str(raw_name).strip().lower()
        if cleaned in _EMPTY_TOKENS:
            return "unknown"

        first_token = cleaned.split()[0] if cleaned.split() else ""
        first_token = _RE_NON_CYR.sub("", first_token)
        if not first_token or len(first_token) < 2:
            return "unknown"

        return self._gender_for_token(first_token)

    @lru_cache(maxsize=4096)
    def _gender_for_token(self, token: str) -> str:
        try:
            parses = self._morph.parse(token)
        except Exception as exc:
            log.debug("pymorphy parse failed for %r: %s", token, exc)
            return "unknown"
        if not parses:
            return "unknown"

        name_parses = [p for p in parses if "Name" in p.tag]
        candidates = name_parses or parses[:2]

        for p in candidates:
            g = getattr(p.tag, "gender", None)
            if g == "masc":
                return "male"
            if g == "femn":
                return "female"
        return "unknown"


    def detect_many(self, names: list[str], progress_callback=None) -> list[str]:
        out: list[str] = []
        total = len(names)
        for i, n in enumerate(names, start=1):
            out.append(self.detect(n))
            if progress_callback and i % 100 == 0:
                try:
                    progress_callback(i, total)
                except Exception:
                    pass
        if progress_callback:
            try:
                progress_callback(total, total)
            except Exception:
                pass
        return out


def gender_distribution(genders: list[str]) -> dict[str, int]:
    out = {"male": 0, "female": 0, "unknown": 0}
    for g in genders:
        out[g if g in out else "unknown"] += 1
    return out
