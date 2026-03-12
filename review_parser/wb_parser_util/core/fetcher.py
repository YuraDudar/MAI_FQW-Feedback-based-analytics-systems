"""
Low-level HTTP client for the Wildberries feedbacks API.

Handles:
- User-agent & host rotation
- Random request delays (politeness / anti-ban)
- Automatic retries with exponential back-off
- Graceful error detection (429, 5xx, network timeouts, JSON decode errors)
- nmId → imtId resolution via basket static-content CDN
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests
from requests.exceptions import (
    ConnectionError as ReqConnectionError,
    JSONDecodeError as ReqJSONDecodeError,
    ReadTimeout,
    RequestException,
)

from wb_parser_util.config import (
    MAX_RETRIES,
    REQUEST_DELAY_MAX,
    REQUEST_DELAY_MIN,
    REQUEST_TIMEOUT,
    RETRY_DELAY_BASE,
    USER_AGENTS,
    WB_BASKET_CARD_PATH,
    WB_BASKET_DOMAINS,
    WB_BASKET_FALLBACK,
    WB_BASKET_THRESHOLDS,
    WB_FEEDBACKS_HOSTS,
    WB_FEEDBACKS_PATH,
)

logger = logging.getLogger(__name__)

# NOTE: Accept-Encoding is intentionally omitted — requests/urllib3 manages
# gzip negotiation and decompression automatically; setting it manually
# prevents auto-decompression and breaks resp.json().
_BASE_HEADERS: dict[str, str] = {
    "Accept": "*/*",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Origin": "https://www.wildberries.ru",
    "Referer": "https://www.wildberries.ru/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
}


class WBFetcher:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._rotate_user_agent()

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_imt_id(self, nm_id: str) -> str | None:
        """
        Resolve imtId (product-group ID) from nmId (SKU артикул).

        WB's feedbacks API requires imtId. This queries the static basket CDN
        (no PoW protection) to read card.json which contains `imt_id`.
        """
        nm   = int(nm_id)
        vol  = nm // 100000
        part = nm // 1000
        basket = self._basket_number(vol)

        for domain in WB_BASKET_DOMAINS:
            path = WB_BASKET_CARD_PATH.format(vol=vol, part=part, nm_id=nm_id)
            url  = f"https://basket-{basket}.{domain}{path}"
            logger.debug("Resolving imtId: GET %s", url)
            try:
                resp = self._session.get(url, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    imt_id = data.get("imt_id")
                    if imt_id:
                        logger.info(
                            "nmId %s → imtId %s  (basket-%s.%s)",
                            nm_id, imt_id, basket, domain,
                        )
                        return str(imt_id)
                    logger.warning(
                        "basket-%s.%s: card.json has no imt_id field", basket, domain
                    )
                elif resp.status_code == 404:
                    logger.debug("basket-%s.%s: 404 — trying next domain", basket, domain)
                else:
                    logger.debug(
                        "basket-%s.%s: HTTP %d", basket, domain, resp.status_code
                    )
            except (ReqConnectionError, ReadTimeout) as exc:
                logger.debug("basket-%s.%s unreachable: %s", basket, domain, exc)
            except (ReqJSONDecodeError, ValueError) as exc:
                logger.warning(
                    "basket-%s.%s: invalid JSON: %s", basket, domain, exc
                )

        logger.warning(
            "nmId %s | could not resolve imtId via basket-%s — "
            "will attempt feedbacks API with nmId directly",
            nm_id, basket,
        )
        return None

    def fetch_reviews_page(
        self,
        imt_id: str,
        nm_id: str,
        take: int = 30,
        skip: int = 0,
        order: str = "dateDesc",
    ) -> dict[str, Any] | None:
        """
        Fetch a single page of reviews.

        :param imt_id: product-group ID used by the feedbacks API
        :param nm_id:  original SKU (used only for log messages)
        """
        hosts = WB_FEEDBACKS_HOSTS.copy()
        random.shuffle(hosts)
        host_cycle = (hosts[i % len(hosts)] for i in range(MAX_RETRIES))

        params: dict[str, Any] = {"take": take, "skip": skip, "order": order}

        for attempt in range(1, MAX_RETRIES + 1):
            host = next(host_cycle)
            url  = f"{host}{WB_FEEDBACKS_PATH.format(imt_id=imt_id)}"
            self._polite_delay()
            try:
                resp = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            except ReadTimeout:
                logger.warning(
                    "SKU %s | timeout on attempt %d/%d (%s)",
                    nm_id, attempt, MAX_RETRIES, host,
                )
                self._backoff(attempt)
                continue
            except ReqConnectionError as exc:
                logger.warning(
                    "SKU %s | connection error on attempt %d/%d (%s): %s",
                    nm_id, attempt, MAX_RETRIES, host, exc,
                )
                self._backoff(attempt)
                continue
            except RequestException as exc:
                logger.error("SKU %s | unexpected request error: %s", nm_id, exc)
                return None

            if resp.status_code == 200:
                return self._parse_json(resp, nm_id)

            if resp.status_code == 404:
                logger.error("SKU %s | product not found (404) at %s", nm_id, url)
                return None

            if resp.status_code == 429:
                retry_after = float(
                    resp.headers.get("Retry-After", RETRY_DELAY_BASE * attempt)
                )
                logger.warning(
                    "SKU %s | rate-limited (429) — cooling down %.1fs (attempt %d/%d)",
                    nm_id, retry_after, attempt, MAX_RETRIES,
                )
                time.sleep(retry_after)
                self._rotate_user_agent()
                continue

            if resp.status_code >= 500:
                logger.warning(
                    "SKU %s | server error %d on attempt %d/%d",
                    nm_id, resp.status_code, attempt, MAX_RETRIES,
                )
                self._backoff(attempt)
                continue

            logger.warning(
                "SKU %s | unexpected status %d — aborting", nm_id, resp.status_code
            )
            return None

        logger.error("SKU %s | all %d retries exhausted", nm_id, MAX_RETRIES)
        return None

    def close(self) -> None:
        self._session.close()

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _basket_number(vol: int) -> str:
        for threshold, basket in WB_BASKET_THRESHOLDS:
            if vol <= threshold:
                return basket
        return WB_BASKET_FALLBACK

    def _rotate_user_agent(self) -> None:
        headers = _BASE_HEADERS.copy()
        headers["User-Agent"] = random.choice(USER_AGENTS)
        self._session.headers.update(headers)

    def _polite_delay(self) -> None:
        delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
        logger.debug("Sleeping %.2fs", delay)
        time.sleep(delay)

    def _backoff(self, attempt: int) -> None:
        wait = RETRY_DELAY_BASE * attempt
        logger.debug("Back-off %.1fs", wait)
        time.sleep(wait)

    @staticmethod
    def _parse_json(resp: requests.Response, nm_id: str) -> dict[str, Any] | None:
        try:
            return resp.json()
        except (ReqJSONDecodeError, ValueError) as exc:
            logger.error("SKU %s | invalid JSON in response: %s", nm_id, exc)
            return None