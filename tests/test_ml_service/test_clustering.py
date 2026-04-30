import pytest
from unittest.mock import MagicMock, patch, AsyncMock


def test_get_target_topics():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from infrastructure.config import get_target_topics
    assert get_target_topics(10) == 3
    assert get_target_topics(50) == 3
    assert get_target_topics(51) == 5
    assert get_target_topics(200) == 5
    assert get_target_topics(201) == 8
    assert get_target_topics(600) == 8
    assert get_target_topics(601) == 12
    assert get_target_topics(10000) == 12


def test_max_topics_constant():
    from infrastructure.config import MAX_TOPICS
    assert MAX_TOPICS == 15


def test_min_text_chars_constant():
    from infrastructure.config import MIN_TEXT_CHARS
    assert MIN_TEXT_CHARS == 15


def _make_review(review_id: str, rating: int, advantages: str = "", disadvantages: str = "", comment: str = "") -> dict:
    return {
        "review_id": review_id,
        "rating": rating,
        "advantages": advantages,
        "disadvantages": disadvantages,
        "comment": comment,
        "reviewer_name": "Тест",
        "created_date": "2024-01-01",
        "excluded_from_rating": False,
    }


def test_negative_pool_logic():
    reviews = [
        _make_review("r1", 1, disadvantages="Плохое качество"),
        _make_review("r2", 2, disadvantages="Быстро сломалось"),
        _make_review("r3", 3, comment="Ничего особенного"),
        _make_review("r4", 4, advantages="Хорошо"),
        _make_review("r5", 5, advantages="Отлично"),
    ]
    neg_reviews = [r for r in reviews if r["rating"] <= 3]
    pos_reviews = [r for r in reviews if r["rating"] >= 4]
    assert len(neg_reviews) == 3
    assert len(pos_reviews) == 2


def test_cyrillic_tokenizer():
    import re
    pattern = re.compile(r"[Ѐ-ӿ]{2,}")
    text = "Хорошее качество, отличный размер the good"
    tokens = pattern.findall(text)
    assert "Хорошее" in tokens
    assert "качество" in tokens
    assert "the" not in tokens
    assert "good" not in tokens


@pytest.mark.asyncio
async def test_clustering_service_save_empty_clusters():
    from unittest.mock import AsyncMock
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.flush = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.refresh = AsyncMock()

    with patch("ml_service.services.clustering_service.ClusteringService._upsert_qdrant", new=AsyncMock()):
        from ml_service.services.clustering_service import ClusteringService
        svc = ClusteringService(mock_db)
        result = {
            "positive": {"clusters": [], "review_ids": []},
            "negative": {"clusters": [], "review_ids": []},
            "pos_count": 0,
            "neg_count": 0,
        }
        await svc._save_results(product_id=1, job_id=1, result=result)


def test_all_stop_words_not_empty():
    from infrastructure.config import ALL_STOP_WORDS, RUSSIAN_STOP_WORDS, MARKETPLACE_STOP_WORDS
    assert len(ALL_STOP_WORDS) > 0
    assert len(RUSSIAN_STOP_WORDS) > 0
    assert len(MARKETPLACE_STOP_WORDS) > 0
    assert "wildberries" in MARKETPLACE_STOP_WORDS
    assert "ozon" in MARKETPLACE_STOP_WORDS


def test_embedding_models_config():
    from infrastructure.config import EMBEDDING_MODELS
    assert "bge-m3" in EMBEDDING_MODELS
    assert "e5-large" in EMBEDDING_MODELS
    assert EMBEDDING_MODELS["e5-large"]["dim"] == 1024
    assert EMBEDDING_MODELS["bge-m3"]["dim"] == 1024
    assert EMBEDDING_MODELS["e5-large"]["prefix"] == "passage: "
