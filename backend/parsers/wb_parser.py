import logging
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, "/app")

from review_parser.wb_parser_util.core.fetcher import WBFetcher
from review_parser.wb_parser_util.core.extractor import ReviewExtractor

logger = logging.getLogger(__name__)


class WBParser:
    def __init__(self):
        self._fetcher = WBFetcher()
        self._extractor = ReviewExtractor()

    def parse_reviews(
        self,
        nm_id: str,
        product_id: int,
        job_id: int,
        max_reviews: int | None = None,
    ) -> list[dict[str, Any]]:
        imt_id = self._fetcher.fetch_imt_id(nm_id)
        if not imt_id:
            logger.warning("Не удалось получить imtId для nmId=%s, используем nmId напрямую", nm_id)
            imt_id = nm_id

        all_reviews: list[dict[str, Any]] = []
        skip = 0
        take = 30

        while True:
            page = self._fetcher.fetch_reviews_page(
                imt_id=imt_id,
                nm_id=nm_id,
                take=take,
                skip=skip,
            )
            if not page:
                break

            feedbacks = page.get("feedbacks") or []
            if not feedbacks:
                break

            for raw in feedbacks:
                review = self._extractor.extract(raw)
                if review:
                    mapped = self._map_to_db(review, nm_id, product_id, job_id)
                    all_reviews.append(mapped)

            skip += len(feedbacks)
            if len(feedbacks) < take:
                break
            if max_reviews and len(all_reviews) >= max_reviews:
                all_reviews = all_reviews[:max_reviews]
                break

        logger.info("Получено %d отзывов для nmId=%s", len(all_reviews), nm_id)
        return all_reviews

    def _map_to_db(self, review: Any, nm_id: str, product_id: int, job_id: int) -> dict[str, Any]:
        r = review if isinstance(review, dict) else review.__dict__
        return {
            "review_id": str(r.get("id") or r.get("review_id", "")),
            "product_id": product_id,
            "parsing_job_id": job_id,
            "input_sku": nm_id,
            "parsed_at": datetime.now(timezone.utc).isoformat(),
            "platform": "wildberries",
            "nm_id": int(nm_id) if nm_id else None,
            "wb_user_id": r.get("wbUserId"),
            "global_user_id": str(r.get("globalUserId") or ""),
            "reviewer_name": r.get("userName") or r.get("reviewer_name"),
            "reviewer_country": r.get("countryId"),
            "reviewer_has_avatar": r.get("hasAvatar", False),
            "rating": r.get("productValuation") or r.get("rating"),
            "advantages": r.get("pros") or r.get("advantages"),
            "disadvantages": r.get("cons") or r.get("disadvantages"),
            "comment": r.get("text") or r.get("comment"),
            "variant_color": r.get("color") or r.get("variant_color"),
            "size": r.get("size"),
            "tags": r.get("tags"),
            "created_date": r.get("createdDate") or r.get("created_date"),
            "updated_date": r.get("updatedDate") or r.get("updated_date"),
            "status_id": r.get("status"),
            "purchase_status": r.get("buyerStatus"),
            "seller_response": r.get("answer"),
            "seller_response_state": r.get("answerState"),
            "votes_plus": r.get("votesPlus", 0),
            "votes_minus": r.get("votesMinus", 0),
            "rank": r.get("rank"),
            "has_video": r.get("hasVideo", False),
            "excluded_from_rating": r.get("isRejected", False),
        }

    def close(self):
        self._fetcher.close()
