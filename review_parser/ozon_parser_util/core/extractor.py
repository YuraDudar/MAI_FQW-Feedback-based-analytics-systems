"""
High-level review extractor for Ozon (Selenium + dual-tier parsing).

Tier 1 — widgetStates JSON (preferred):
  The Ozon SSR page embeds the same JSON structure used by its internal API.
  If the fetcher successfully extracts widgetStates from the page's <script>
  tags, reviews are parsed from that clean JSON.

Tier 2 — BeautifulSoup HTML (fallback):
  When JS extraction fails (CSR page / no embedded state), the extractor
  falls back to parsing the rendered HTML. Uses data-widget attributes as
  stable anchors; field extraction relies on tag semantics and text patterns
  rather than dynamic hashed CSS class names.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup, Tag

from ozon_parser_util.config import OZON_PAGE_SIZE, OZON_REVIEW_WIDGET_PREFIXES
from ozon_parser_util.core.fetcher import OzonBrowser
from ozon_parser_util.core.models import OzonReview

logger = logging.getLogger(__name__)

_RU_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
_RU_MONTH_RE = "|".join(_RU_MONTHS.keys())

_SELLER_MARKERS = re.compile(
    r"Ответ\s+(продавца|магазина|официальный)|Официальный\s+ответ", re.I
)


class OzonReviewExtractor:
    def __init__(self, browser: OzonBrowser) -> None:
        self._browser = browser

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        product_id: str,
        limit: int | None = None,
        stars: int | None = None,
        sort: str = "dateDesc",
    ) -> list[OzonReview]:
        collected: list[OzonReview] = []
        page = 1
        total_pages: int | None = None

        logger.info(
            "product %s | starting extraction (limit=%s, stars=%s, sort=%s)",
            product_id, limit, stars, sort,
        )

        while True:
            data = self._browser.get_page_data(product_id, page, sort)
            if data is None:
                logger.warning("product %s | page %d returned None", product_id, page)
                break

            if total_pages is None:
                total_pages = self._browser.get_max_page()
                logger.info("product %s | total pages: %d", product_id, total_pages)

            if "widget_states" in data:
                logger.info("product %s | page %d — parsing via Tier-1 (widgetStates)", product_id, page)
                raw_reviews = _parse_from_widget_states(data["widget_states"], product_id)
                if not raw_reviews:
                    # widget_states present but no reviews found → fall back to HTML
                    logger.info("product %s | Tier-1 returned 0 reviews — retrying with Tier-2 HTML", product_id)
                    raw_reviews = _parse_from_html(data.get("html", ""), product_id)
            else:
                logger.info("product %s | page %d — parsing via Tier-2 (HTML)", product_id, page)
                raw_reviews = _parse_from_html(data.get("html", ""), product_id)

            if not raw_reviews:
                logger.info("product %s | no reviews on page %d — stopping", product_id, page)
                break

            for review in raw_reviews:
                if stars is not None and review.rating != stars:
                    continue
                collected.append(review)
                if limit is not None and len(collected) >= limit:
                    logger.info("product %s | reached limit %d", product_id, limit)
                    return collected

            logger.info(
                "product %s | page %d done — collected %d so far",
                product_id, page, len(collected),
            )

            if page >= total_pages:
                break
            if len(raw_reviews) < OZON_PAGE_SIZE:
                break
            page += 1

        logger.info("product %s | extraction complete — %d reviews", product_id, len(collected))
        return collected


# ── Tier 1: widgetStates JSON parsing ─────────────────────────────────────────

def _parse_from_widget_states(
    widget_states: dict[str, Any], product_id: str
) -> list[OzonReview]:
    widget_data = _find_review_widget(widget_states)
    if widget_data is None:
        logger.debug(
            "No review widget in widgetStates. Keys: %s",
            list(widget_states.keys())[:10],
        )
        return []

    raw_list: list[dict] = widget_data.get("reviews") or widget_data.get("items") or []
    if not raw_list:
        return []

    total = (
        widget_data.get("feedbackCount")
        or widget_data.get("totalReviewCount")
        or (widget_data.get("paging") or {}).get("total")
        or 0
    )
    logger.info("product %s | total reviews (WS): %d", product_id, total)

    return [_review_from_json(raw, product_id) for raw in raw_list]


def _find_review_widget(states: dict) -> dict | None:
    for key, value in states.items():
        if not any(key.startswith(p) for p in OZON_REVIEW_WIDGET_PREFIXES):
            if "review" not in key.lower():
                continue
        decoded = _decode_widget(value)
        if decoded and (decoded.get("reviews") or decoded.get("items")):
            return decoded
    return None


def _decode_widget(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            d = json.loads(value)
            return d if isinstance(d, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _review_from_json(raw: dict, product_id: str) -> OzonReview:
    author: dict = raw.get("author") or {}
    first = (author.get("firstName") or "").strip()
    last  = (author.get("lastName")  or "").strip()
    title = (author.get("title")     or "").strip()
    reviewer = title or (f"{first} {last}".strip() if last else first) or "Аноним"

    rating = int(raw.get("rating") or raw.get("score") or 0)

    date_str = raw.get("publishedAt") or raw.get("date") or raw.get("createdAt") or ""
    review_date = _parse_iso_date(date_str)

    content: dict = raw.get("content") or raw
    char = str(content.get("itemCharacteristics") or raw.get("itemCharacteristics") or "").strip()
    aspects: list[dict] = (
        content.get("aspects") or raw.get("aspects")
        or content.get("itemAspects") or raw.get("itemAspects") or []
    )
    aspect_str = ", ".join(
        f"{a.get('name', '')}: {a.get('value', '')}"
        for a in aspects if a.get("name")
    )
    body = str(
        content.get("text") or raw.get("text")
        or content.get("body") or raw.get("body") or ""
    ).strip()
    comment = "\n".join(p for p in [char, aspect_str, body] if p)

    comments: list[dict] = raw.get("comments") or raw.get("responses") or []
    sellers_response = ""
    for c in comments:
        auth: dict = c.get("author") or {}
        if auth.get("isCompany") or c.get("isCompany") or auth.get("type") == "company":
            sellers_response = str(
                (c.get("content") or c).get("text") or c.get("text") or c.get("body") or ""
            ).strip()
            break
    if not sellers_response and comments:
        first_c = comments[0]
        sellers_response = str(
            (first_c.get("content") or first_c).get("text") or first_c.get("text") or ""
        ).strip()

    return OzonReview(
        product_id=product_id,
        reviewer=reviewer,
        rating=rating,
        review_date=review_date,
        comment=comment,
        sellers_response=sellers_response,
    )


# ── Tier 2: BeautifulSoup HTML parsing ────────────────────────────────────────

def _parse_from_html(html: str, product_id: str) -> list[OzonReview]:
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")

    container: Tag | None = None
    for widget_name in ["webListReviews", "webReviewTabs"]:
        container = soup.find(attrs={"data-widget": widget_name})
        if container:
            logger.debug("HTML parsing: found container data-widget=%s", widget_name)
            break

    if container is None:
        available = [el.get("data-widget") for el in soup.find_all(attrs={"data-widget": True})]
        logger.warning(
            "HTML parsing: review container not found. Available widgets: %s",
            available[:15],
        )
        return []

    items = _find_review_items(container)
    logger.debug("HTML parsing: found %d candidate review items", len(items))

    reviews = []
    for item in items:
        review = _parse_html_item(item, product_id)
        if review:
            reviews.append(review)
    return reviews


def _find_review_items(container: Tag) -> list[Tag]:
    articles = container.find_all("article")
    if articles:
        return articles

    candidates = []
    for child in container.find_all("div", recursive=False):
        text = child.get_text()
        has_date = bool(
            child.find("time")
            or re.search(rf"\d{{1,2}}\s+(?:{_RU_MONTH_RE})\s+\d{{4}}", text, re.I)
        )
        if has_date and len(text) > 30:
            candidates.append(child)

    if candidates:
        return candidates

    return [
        c for c in container.find_all("div", recursive=True)
        if c.find("time") and len(c.get_text(strip=True)) > 20
    ]


def _parse_html_item(item: Tag, product_id: str) -> OzonReview | None:
    try:
        reviewer    = _html_reviewer(item)
        rating      = _html_rating(item)
        review_date = _html_date(item)
        comment     = _html_comment(item)
        sellers_r   = _html_seller_response(item)

        if not comment and not rating:
            return None

        return OzonReview(
            product_id=product_id,
            reviewer=reviewer,
            rating=rating,
            review_date=review_date,
            comment=comment,
            sellers_response=sellers_r,
        )
    except Exception as exc:
        logger.debug("HTML item parse error: %s", exc)
        return None


def _html_reviewer(item: Tag) -> str:
    date_el = item.find("time")
    for tag in item.find_all(["span", "div", "b", "strong", "p"]):
        if date_el and tag is date_el:
            break
        text = tag.get_text(strip=True)
        if text and 2 < len(text) < 60 and not re.match(r"^\d", text):
            if not re.search(r"Ответ|ответ|рейтинг|★|звезд", text, re.I):
                return text
    return "Аноним"


def _html_rating(item: Tag) -> int:
    for el in item.find_all(attrs={"aria-label": True}):
        label = el.get("aria-label", "")
        m = re.search(r"(\d)\s*(из\s*5|звезд|рейтинг)", label, re.I)
        if m:
            return min(5, max(1, int(m.group(1))))
        m2 = re.match(r"^(\d)$", label.strip())
        if m2:
            return min(5, max(1, int(m2.group(1))))

    all_stars = item.find_all(class_=re.compile(r"star|Star", re.I))
    if all_stars:
        filled = [
            s for s in all_stars
            if re.search(r"fill|active|solid", " ".join(s.get("class", [])), re.I)
        ]
        if filled:
            return min(5, len(filled))
    return 0


def _html_date(item: Tag) -> str:
    time_el = item.find("time")
    if time_el:
        dt_attr = time_el.get("datetime", "")
        if dt_attr:
            return _parse_iso_date(dt_attr)
        return _parse_ru_date(time_el.get_text(strip=True))

    text = item.get_text()
    m = re.search(rf"(\d{{1,2}})\s+({_RU_MONTH_RE})\s+(\d{{4}})", text, re.I)
    if m:
        return _parse_ru_date(m.group(0))
    return ""


def _html_comment(item: Tag) -> str:
    clone = BeautifulSoup(str(item), "lxml").body or BeautifulSoup(str(item), "lxml")

    for el in clone.find_all(string=_SELLER_MARKERS):
        parent = el.find_parent()
        if parent:
            for sibling in list(parent.find_next_siblings()):
                sibling.decompose()
            parent.decompose()

    lines = [
        ln.strip()
        for ln in clone.get_text(separator="\n").splitlines()
        if ln.strip()
    ]

    start = 0
    for i, line in enumerate(lines[:6]):
        if re.search(rf"\d{{1,2}}\s+(?:{_RU_MONTH_RE})\s+\d{{4}}", line, re.I):
            start = i + 1
            break
        if i == 0 and len(line) < 60:
            start = 1

    return "\n".join(lines[start:]).strip()


def _html_seller_response(item: Tag) -> str:
    text = item.get_text(separator="\n")
    m = _SELLER_MARKERS.search(text)
    if m:
        after = text[m.end():].strip()
        next_marker = _SELLER_MARKERS.search(after)
        if next_marker:
            after = after[:next_marker.start()].strip()
        return after[:500].strip()
    return ""


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_iso_date(s: str) -> str:
    if not s:
        return ""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:26], fmt).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return s


def _parse_ru_date(s: str) -> str:
    if not s:
        return ""
    m = re.match(rf"(\d{{1,2}})\s+({_RU_MONTH_RE})\s+(\d{{4}})", s.strip(), re.I)
    if not m:
        return s
    day, month_ru, year = m.group(1), m.group(2).lower(), m.group(3)
    month = _RU_MONTHS.get(month_ru)
    if month:
        try:
            return datetime(int(year), month, int(day)).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
    return s