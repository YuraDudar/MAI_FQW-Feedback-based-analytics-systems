"""Domain model for a single Ozon customer review."""
from dataclasses import dataclass, fields as dc_fields


@dataclass
class OzonReview:
    """Represents one customer review scraped from Ozon."""

    product_id: str       # Ozon product ID / SKU entered by the user
    reviewer: str         # display name of the reviewer
    rating: int           # 1–5 stars
    review_date: str      # normalised date string
    comment: str          # full review text: characteristics + aspects + body
    sellers_response: str # first reply from the seller (may be empty)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    @classmethod
    def column_names(cls) -> list[str]:
        return [
            "SKU товара",
            "Пользователь",
            "Рейтинг",
            "Дата отзыва",
            "Отзыв",
            "Ответ продавца",
        ]