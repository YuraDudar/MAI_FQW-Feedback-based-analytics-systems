"""
Central configuration for the Ozon review parser.
"""
from pathlib import Path

# ── Ozon Entrypoint API ───────────────────────────────────────────────────────
OZON_ENTRYPOINT_URL: str = "https://www.ozon.ru/api/entrypoint-api.bx/page/json/v2"
OZON_REVIEWS_PATH: str = "/product/{product_id}/reviews/"

# Known widget key prefixes that contain the review list in widgetStates
OZON_REVIEW_WIDGET_PREFIXES: tuple[str, ...] = (
    "webListReviews-",
    "reviewList-",
    "review-list-",
)

OZON_PAGE_SIZE: int = 12

# ── Sort orders ───────────────────────────────────────────────────────────────
OZON_SORT_MAP: dict[str, str] = {
    "dateDesc": "date_desc",
    "dateAsc":  "date_asc",
    "rating":   "score_desc",
}
OZON_DEFAULT_SORT: str = "dateDesc"

# ── Anti-ban / Politeness ─────────────────────────────────────────────────────
REQUEST_DELAY_MIN: float = 1.5
REQUEST_DELAY_MAX: float = 4.0
REQUEST_TIMEOUT: int   = 30
MAX_RETRIES: int       = 3
RETRY_DELAY_BASE: float = 5.0

# ── Selenium settings ─────────────────────────────────────────────────────────
SELENIUM_PAGE_TIMEOUT: int    = 25    # seconds to wait for page load
SELENIUM_SCROLL_STEP: int     = 400   # pixels per scroll step
SELENIUM_SCROLL_DELAY: float  = 0.07  # seconds between scroll steps

# ── User-agent rotation pool ──────────────────────────────────────────────────
USER_AGENTS: list[str] = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) "
        "Gecko/20100101 Firefox/123.0"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
    ),
]

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = Path("results")
FILENAME_TEMPLATE: str = "ozon_reviews_{sku_part}_{timestamp}"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str       = "INFO"
LOG_FORMAT: str      = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"