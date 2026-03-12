"""
High-level review extractor for Wildberries.

Flow per SKU:
  1. Resolve nmId → imtId via basket CDN.
  2. Paginate feedbacks API with duplicate detection.
  3. Map raw API payload → typed Review dataclass (all 34 fields).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from math import nan as NAN
from typing import Any

from wb_parser_util.config import WB_PAGE_SIZE
from wb_parser_util.core.fetcher import WBFetcher
from wb_parser_util.core.models import Review

logger = logging.getLogger(__name__)

_STATUS_ID_MAP: dict[int, str] = {
    8:   "Вернули",
    14:  "Отказались",
    16:  "Выкупили",
    120: "Претензия",
}


class ReviewExtractor:
    def __init__(self, fetcher: WBFetcher) -> None:
        self._fetcher = fetcher

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        product_id: str,
        limit: int | None = None,
        stars: int | None = None,
        order: str = "dateDesc",
    ) -> list[Review]:
        imt_id = self._fetcher.fetch_imt_id(product_id)
        fetch_id = imt_id if imt_id else product_id

        if not imt_id:
            logger.warning(
                "SKU %s | imtId resolution failed — using nmId as fallback", product_id
            )

        parsed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return self._paginate(
            nm_id=product_id,
            fetch_id=fetch_id,
            input_sku=product_id,
            parsed_at=parsed_at,
            limit=limit,
            stars=stars,
            order=order,
        )

    # ── Private: pagination ───────────────────────────────────────────────────

    def _paginate(
        self,
        nm_id: str,
        fetch_id: str,
        input_sku: str,
        parsed_at: str,
        limit: int | None,
        stars: int | None,
        order: str,
    ) -> list[Review]:
        collected: list[Review] = []
        seen_ids: set[str] = set()
        skip = 0
        wb_total: int | None = None

        logger.info(
            "SKU %s | starting extraction (fetch_id=%s, limit=%s, stars=%s)",
            nm_id, fetch_id, limit, stars,
        )

        while True:
            data = self._fetcher.fetch_reviews_page(
                imt_id=fetch_id, nm_id=nm_id,
                take=WB_PAGE_SIZE, skip=skip, order=order,
            )

            if data is None:
                logger.warning("SKU %s | fetch None at skip=%d — stopping", nm_id, skip)
                break

            if wb_total is None:
                wb_total = int(
                    data.get("feedbackCount") or data.get("totalFeedbackCount") or 0
                )
                logger.info("SKU %s | total reviews on WB: %d", nm_id, wb_total)

            raw_list: list[dict[str, Any]] = data.get("feedbacks") or []
            if not raw_list:
                logger.info("SKU %s | empty page at skip=%d", nm_id, skip)
                break

            page_ids = {str(r.get("id", "")) for r in raw_list}
            new_ids  = page_ids - seen_ids
            if not new_ids:
                logger.warning(
                    "SKU %s | only duplicates — WB API cap reached. "
                    "Collected %d of %d total.",
                    nm_id, len(collected), wb_total or "?",
                )
                break
            seen_ids |= page_ids

            for raw in raw_list:
                if str(raw.get("id", "")) not in new_ids:
                    continue
                review = self._parse_review(raw, input_sku, parsed_at)
                if stars is not None and review.rating != stars:
                    continue
                collected.append(review)
                if limit is not None and len(collected) >= limit:
                    logger.info("SKU %s | reached limit of %d", nm_id, limit)
                    return collected

            skip += len(raw_list)
            logger.info(
                "SKU %s | fetched %d so far (skip=%d / wb_total=%d)",
                nm_id, len(collected), skip, wb_total or 0,
            )

            if len(raw_list) < WB_PAGE_SIZE:
                break

        if wb_total and len(collected) < wb_total:
            logger.warning(
                "SKU %s | collected %d / %d — WB public API cap ~1 000 reviews.",
                nm_id, len(collected), wb_total,
            )

        logger.info("SKU %s | extraction complete — %d reviews", nm_id, len(collected))
        return collected

    # ── Private: parsing ─────────────────────────────────────────────────────

    def _parse_review(
        self, raw: dict[str, Any], input_sku: str, parsed_at: str
    ) -> Review:
        user:     dict = raw.get("wbUserDetails") or {}
        answer:   dict = raw.get("answer") or {}
        votes:    dict = raw.get("votes") or {}
        excluded: dict = raw.get("excludedFromRating") or {}
        reasons:  dict = raw.get("reasons") or {}

        # statusId → text
        status_id  = raw.get("statusId")
        status_txt = _STATUS_ID_MAP.get(int(status_id), NAN) if status_id else NAN

        # bables → joined string
        bables = raw.get("bables") or []
        tags   = "; ".join(str(b) for b in bables) if bables else NAN

        # video → "Да" / "Нет"
        has_video = "Да" if raw.get("video") else "Нет"

        # excluded reasons
        exc_reasons = excluded.get("reasons") or []
        exc_reasons_str = "; ".join(str(r) for r in exc_reasons) if exc_reasons else NAN

        # good / bad reason codes
        good_r = reasons.get("good") or []
        bad_r  = reasons.get("bad")  or []
        good_reasons_str = "; ".join(str(x) for x in good_r) if good_r else NAN
        bad_reasons_str  = "; ".join(str(x) for x in bad_r)  if bad_r  else NAN

        return Review(
            parsed_at  = parsed_at,
            input_sku  = input_sku,
            review_id      = _s(raw.get("id")),
            nm_id          = _s(raw.get("nmId")),
            wb_user_id     = _s(raw.get("wbUserId")),
            global_user_id = _s(raw.get("globalUserId")),
            reviewer_name       = _s(user.get("name")),
            reviewer_country    = _s(user.get("country")),
            reviewer_has_avatar = _b(user.get("hasPhoto")),
            rating        = _s(raw.get("productValuation")),
            advantages    = _s(raw.get("pros")),
            disadvantages = _s(raw.get("cons")),
            comment       = _s(raw.get("text")),
            variant_color = _s(raw.get("color")),
            size          = _s(raw.get("size")),
            tags          = tags,
            created_date    = _parse_date(raw.get("createdDate", "")),
            updated_date    = _parse_date(raw.get("updatedDate", "")),
            status_id       = _s(status_id),
            purchase_status = status_txt,
            seller_response       = _s(answer.get("text")),
            seller_response_state = _s(answer.get("state")),
            matching_size        = _s(raw.get("matchingSize")),
            matching_photo       = _s(raw.get("matchingPhoto")),
            matching_description = _s(raw.get("matchingDescription")),
            votes_plus  = _s(votes.get("pluses")),
            votes_minus = _s(votes.get("minuses")),
            rank        = _s(raw.get("rank")),
            helpfulness = _helpfulness(raw.get("feedbackHelpfulness")),
            has_video = has_video,
            excluded_from_rating = _b(excluded.get("isExcluded")),
            excluded_reasons     = exc_reasons_str,
            good_reasons = good_reasons_str,
            bad_reasons  = bad_reasons_str,
        )


# ── Field-level helpers ───────────────────────────────────────────────────────

def _s(value: Any) -> Any:
    """Return NAN for None and empty string; keep False/0 as-is."""
    if value is None:
        return NAN
    if isinstance(value, str) and value.strip() == "":
        return NAN
    return value


def _b(value: Any) -> Any:
    """Safely return bool or NAN (False is a valid value, not missing)."""
    if value is None:
        return NAN
    return bool(value)


def _helpfulness(value: Any) -> Any:
    """
    feedbackHelpfulness is a list of vote objects.
    Summarise as 'plus: N, minus: M', or NAN if absent.
    """
    if not value:
        return NAN
    if isinstance(value, list):
        plus  = sum(1 for v in value if v.get("helpfulness") == "plus")
        minus = sum(1 for v in value if v.get("helpfulness") == "minus")
        return f"plus: {plus}, minus: {minus}"
    return str(value)


def _parse_date(raw_date: str) -> Any:
    """Normalise ISO-8601 string to 'YYYY-MM-DD HH:MM' (UTC), or NAN."""
    if not raw_date:
        return NAN
    try:
        dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return raw_date if raw_date.strip() else NAN