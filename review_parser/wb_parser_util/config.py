"""
Central configuration for the Wildberries review parser.
All tunable parameters live here — no magic numbers scattered across the code.
"""
from pathlib import Path

# ── WB Feedbacks API ──────────────────────────────────────────────────────────
# WB exposes the feedbacks endpoint on two mirrored hosts; we rotate between them.
WB_FEEDBACKS_HOSTS: list[str] = [
    "https://feedbacks1.wb.ru",
    "https://feedbacks2.wb.ru",
]
WB_FEEDBACKS_PATH: str = "/feedbacks/v1/{imt_id}"  # {sku} → nm_id (артикул)
WB_PAGE_SIZE: int = 30                           # max items per API request

# Supported sort orders accepted by the API
WB_SORT_ORDERS: list[str] = ["dateDesc", "dateAsc", "rating"]

# ── WB Basket (static content) API — for nmId → imtId resolution ─────────────
# WB hosts static product JSON on basket-NN.wbbasket.ru CDN servers.
# The server number is determined by the nmId's "vol" prefix (nmId // 100000).
WB_BASKET_DOMAINS: list[str] = ["wbbasket.ru", "wb.ru"]
WB_BASKET_CARD_PATH: str = "/vol{vol}/part{part}/{nm_id}/info/ru/card.json"

# vol → basket-number thresholds (keep ordered, last entry is the catch-all)
WB_BASKET_THRESHOLDS: list[tuple[int, str]] = [
    (143,  "01"), (287,  "02"), (431,  "03"), (719,  "04"),
    (1007, "05"), (1061, "06"), (1115, "07"), (1169, "08"),
    (1313, "09"), (1601, "10"), (1655, "11"), (1919, "12"),
    (2045, "13"), (2189, "14"), (2405, "15"), (2621, "16"),
    (2837, "17"), (3053, "18"), (3269, "19"), (3485, "20"),
    (3701, "21"), (3917, "22"), (4133, "23"), (4349, "24"),
    (4565, "25"), (4781, "26"), (4997, "27"), (5213, "28"),
    (5429, "29"), (5645, "30"),
]
WB_BASKET_FALLBACK: str = "31"  # for products newer than the last threshold

# ── Anti-ban / Politeness ─────────────────────────────────────────────────────
REQUEST_DELAY_MIN: float = 1.0    # minimum pause between HTTP requests (s)
REQUEST_DELAY_MAX: float = 3.5    # maximum pause between HTTP requests (s)
REQUEST_TIMEOUT: int   = 30       # socket / read timeout per request (s)
MAX_RETRIES: int       = 3        # per-URL retry attempts before giving up
RETRY_DELAY_BASE: float = 5.0    # base seconds for exponential back-off
                                  # actual wait = RETRY_DELAY_BASE * attempt

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
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.3 Safari/605.1.15"
    ),
]

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = Path("results")
# Filename template: wb_reviews_<skus>_<timestamp>.<ext>
FILENAME_TEMPLATE: str = "wb_reviews_{sku_part}_{timestamp}"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str      = "INFO"
LOG_FORMAT: str     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"