"""
Selenium-based browser manager for Ozon review pages.

Uses undetected-chromedriver (uc) instead of plain Selenium Chrome.

Why uc instead of selenium.webdriver.Chrome:
  Plain Selenium leaves dozens of automation markers that Ozon detects.
  undetected-chromedriver patches the ChromeDriver binary at compile-level,
  removing all automation indicators unconditionally.

Page extraction strategy (two tiers):
  1. Execute JS to find the embedded widgetStates JSON blob. Fast, clean.
  2. Fall back to raw page HTML for BeautifulSoup parsing if JS fails.
"""
from __future__ import annotations

import json
import logging
import platform
import random
import re
import subprocess
import time
from typing import Any

import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ozon_parser_util.config import (
    OZON_REVIEWS_PATH,
    OZON_SORT_MAP,
    SELENIUM_PAGE_TIMEOUT,
    SELENIUM_SCROLL_DELAY,
    SELENIUM_SCROLL_STEP,
    USER_AGENTS,
)

logger = logging.getLogger(__name__)

_REVIEW_WIDGETS = ["webListReviews", "webReviewTabs", "webReviewProductScore"]
_REVIEW_WIDGET_CSS = ", ".join(f'[data-widget="{w}"]' for w in _REVIEW_WIDGETS)


class OzonBrowser:
    """
    Manages an undetected Chrome instance for scraping Ozon review pages.
    Create once per session and reuse across multiple SKUs.
    """

    def __init__(self, headless: bool = False) -> None:
        self._driver = _create_driver(headless)
        self._slug_cache: dict[str, str] = {}
        logger.info("Chrome (undetected) launched  headless=%s", headless)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_page_data(
        self, product_id: str, page: int = 1, sort: str = "dateDesc"
    ) -> dict[str, Any] | None:
        """
        Load a reviews page and return either:
          {"widget_states": {...}}  — JS extraction succeeded (clean JSON), or
          {"html": "..."}           — HTML fallback for BeautifulSoup.
        Returns None on unrecoverable error.
        """
        slug = self._resolve_slug(product_id)
        url  = _build_reviews_url(slug, page, sort)

        logger.info("product %s | loading page %d → %s", product_id, page, url)
        try:
            self._driver.get(url)
            self._wait_for_reviews()
            time.sleep(random.uniform(1.5, 2.5))
            self._scroll_to_load()
        except TimeoutException:
            logger.warning(
                "product %s | page %d: timeout waiting for reviews widget",
                product_id, page,
            )
        except WebDriverException as exc:
            logger.error("product %s | browser error: %s", product_id, exc)
            return None

        widget_states = self._extract_widget_states()
        if widget_states:
            logger.debug("product %s | JS widgetStates extracted OK", product_id)
            return {"widget_states": widget_states}

        logger.debug("product %s | falling back to HTML parsing", product_id)
        return {"html": self._driver.page_source}

    def get_max_page(self) -> int:
        """Return the highest page number in the pagination widget."""
        try:
            els = self._driver.find_elements(
                By.CSS_SELECTOR,
                '[data-widget="paginator"] a, [class*="paginat"] a, nav a',
            )
            numbers = [int(el.text.strip()) for el in els if el.text.strip().isdigit()]
            return max(numbers) if numbers else 1
        except Exception:
            return 1

    def close(self) -> None:
        try:
            self._driver.quit()
        except Exception:
            pass

    # ── Private ───────────────────────────────────────────────────────────────

    def _resolve_slug(self, product_id: str) -> str:
        """Follow the redirect for a numeric product ID to get the canonical slug."""
        if product_id in self._slug_cache:
            return self._slug_cache[product_id]

        try:
            self._driver.get(f"https://www.ozon.ru/product/{product_id}/")
            time.sleep(random.uniform(1.5, 2.5))
            final_url = self._driver.current_url
            match = re.search(r"/product/([A-Za-z0-9][A-Za-z0-9_-]+-\d+)/", final_url)
            slug = match.group(1) if match else product_id
        except WebDriverException:
            slug = product_id

        logger.info("product %s → slug: %s", product_id, slug)
        self._slug_cache[product_id] = slug
        return slug

    def _wait_for_reviews(self) -> None:
        WebDriverWait(self._driver, SELENIUM_PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, _REVIEW_WIDGET_CSS))
        )

    def _scroll_to_load(self) -> None:
        """Gradually scroll to trigger lazy-loaded review content."""
        try:
            total = self._driver.execute_script("return document.body.scrollHeight")
            for pos in range(0, total, SELENIUM_SCROLL_STEP):
                self._driver.execute_script(f"window.scrollTo(0, {pos});")
                time.sleep(SELENIUM_SCROLL_DELAY)
            time.sleep(0.5)
        except Exception:
            pass

    def _extract_widget_states(self) -> dict | None:
        """
        Search <script> tags for the embedded widgetStates JSON.
        Ozon SSR injects the initial page state as a JS blob on the page.
        """
        try:
            result: str | None = self._driver.execute_script("""
                var scripts = Array.from(document.querySelectorAll('script'));
                for (var s of scripts) {
                    var t = s.textContent;
                    if (!t.includes('widgetStates')) continue;
                    var m = t.match(/window\\.\\w+\\s*=\\s*(\\{[\\s\\S]+\\})/);
                    if (m) { try { return m[1]; } catch(e) {} }
                    var m2 = t.match(/(\\{"widgetStates":[\\s\\S]+\\})/);
                    if (m2) { try { return m2[1]; } catch(e) {} }
                }
                return null;
            """)
            if not result:
                return None
            data = json.loads(result)
            states = data.get("widgetStates") or data
            if any(k for k in states if "review" in k.lower() or "Review" in k):
                return states
            return None
        except Exception as exc:
            logger.debug("JS widget extraction failed: %s", exc)
            return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_reviews_url(slug: str, page: int, sort: str) -> str:
    path = OZON_REVIEWS_PATH.format(product_id=slug)
    ozon_sort = OZON_SORT_MAP.get(sort, "date_desc")
    url = f"https://www.ozon.ru{path}?sort={ozon_sort}"
    if page > 1:
        url += f"&page={page}"
    return url


def _detect_chrome_version() -> int | None:
    """
    Detect the major version of the installed Chrome browser.

    undetected-chromedriver downloads the ChromeDriver that matches this
    version. Without it, uc might grab the latest ChromeDriver which won't
    work if Chrome itself is one version behind.
    """
    system = platform.system()

    if system == "Windows":
        reg_paths = [
            r"HKCU\Software\Google\Chrome\BLBeacon",
            r"HKLM\SOFTWARE\Google\Chrome\BLBeacon",
            r"HKLM\SOFTWARE\Wow6432Node\Google\Chrome\BLBeacon",
        ]
        for path in reg_paths:
            try:
                out = subprocess.check_output(
                    ["reg", "query", path, "/v", "version"],
                    stderr=subprocess.DEVNULL, timeout=5,
                ).decode(errors="ignore")
                m = re.search(r"(\d+)\.\d+\.\d+", out)
                if m:
                    return int(m.group(1))
            except Exception:
                continue

        # Fallback: PowerShell
        for chrome_path in [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]:
            try:
                out = subprocess.check_output(
                    ["powershell", "-command",
                     f'(Get-Item "{chrome_path}").VersionInfo.FileVersion'],
                    stderr=subprocess.DEVNULL, timeout=5,
                ).decode(errors="ignore").strip()
                m = re.search(r"(\d+)\.", out)
                if m:
                    return int(m.group(1))
            except Exception:
                continue

    elif system == "Darwin":
        try:
            out = subprocess.check_output(
                ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 "--version"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode(errors="ignore")
            m = re.search(r"Chrome/(\d+)", out)
            if m:
                return int(m.group(1))
        except Exception:
            pass

    else:  # Linux
        for cmd in (
            ["google-chrome", "--version"],
            ["google-chrome-stable", "--version"],
            ["chromium-browser", "--version"],
            ["chromium", "--version"],
        ):
            try:
                out = subprocess.check_output(
                    cmd, stderr=subprocess.DEVNULL, timeout=5,
                ).decode(errors="ignore")
                m = re.search(r"(\d+)\.", out)
                if m:
                    return int(m.group(1))
            except Exception:
                continue

    return None


def _create_driver(headless: bool) -> uc.Chrome:
    """
    Create an undetected Chrome instance with the correct ChromeDriver version.

    uc patches the ChromeDriver binary to remove all automation fingerprints.
    version_main must match the installed Chrome major version — otherwise uc
    downloads the latest ChromeDriver which may not support the current Chrome.
    """
    chrome_ver = _detect_chrome_version()
    if chrome_ver:
        logger.info(
            "Detected Chrome version: %d — downloading matching ChromeDriver", chrome_ver
        )
    else:
        logger.warning(
            "Could not detect Chrome version automatically. "
            "If startup fails, set version_main manually in fetcher.py."
        )

    options = uc.ChromeOptions()
    options.add_argument("--lang=ru-RU")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")

    driver = uc.Chrome(
        options=options,
        headless=headless,
        use_subprocess=True,
        version_main=chrome_ver,  # None → uc auto-detects (may pick wrong version)
    )
    return driver