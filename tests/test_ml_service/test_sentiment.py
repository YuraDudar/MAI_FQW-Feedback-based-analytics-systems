import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession


def _make_mock_registry(labels: list[str]):
    registry = MagicMock()
    registry.morph = None
    preds = [{"label": lbl, "score": 0.9} for lbl in labels]
    registry.sentiment_model = MagicMock(return_value=preds)
    return registry


@pytest.mark.asyncio
async def test_merge_text():
    from ml_service.services.sentiment_service import _merge_text
    review = {
        "advantages": "Хорошее качество",
        "disadvantages": "Дорого",
        "comment": "Всё понравилось",
    }
    merged = _merge_text(review)
    assert "Достоинства: Хорошее качество" in merged
    assert "Недостатки: Дорого" in merged
    assert "Комментарий: Всё понравилось" in merged


def test_merge_text_empty():
    from ml_service.services.sentiment_service import _merge_text
    review = {"advantages": None, "disadvantages": None, "comment": None}
    assert _merge_text(review) == ""


def test_merge_text_partial():
    from ml_service.services.sentiment_service import _merge_text
    review = {"advantages": "Хорошо", "disadvantages": None, "comment": "OK"}
    merged = _merge_text(review)
    assert "Недостатки" not in merged
    assert "Хорошо" in merged


@pytest.mark.parametrize("name,expected", [
    ("Мария Иванова", "female"),
    ("Александр Петров", "male"),
    ("", "unknown"),
    (None, "unknown"),
])
def test_detect_gender_basic(name, expected):
    from ml_service.services.sentiment_service import _detect_gender
    result = _detect_gender(name, None)
    assert result == expected


def test_label_map():
    from ml_service.services.sentiment_service import _LABEL_MAP
    assert _LABEL_MAP["POSITIVE"] == "positive"
    assert _LABEL_MAP["NEGATIVE"] == "negative"
    assert _LABEL_MAP["NEUTRAL"] == "neutral"
    assert _LABEL_MAP.get("UNKNOWN", "neutral") == "neutral"


@pytest.mark.asyncio
async def test_process_reviews_short_text_skipped(db_session: AsyncSession):
    from unittest.mock import patch
    import sys

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.flush = AsyncMock()

    reviews = [
        {"review_id": "r1", "advantages": "OK", "disadvantages": None, "comment": None, "reviewer_name": "Иван"},
    ]

    with patch("ml_service.services.sentiment_service.model_registry") as mock_reg:
        mock_reg.sentiment_model = None
        mock_reg.morph = None
        from ml_service.services.sentiment_service import SentimentService
        svc = SentimentService(mock_db)
        await svc.process_reviews(reviews, product_id=1)

    mock_db.execute.assert_not_called()
