import pytest
from unittest.mock import MagicMock, patch, AsyncMock


def test_map_filters_empty():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ml_service"))
    from ml_service.services.rag_service import _map_filters
    assert _map_filters(None) is None
    assert _map_filters({}) is None


def test_map_filters_full():
    from ml_service.services.rag_service import _map_filters
    filters = {
        "date_from": "2024-01-01",
        "date_to": "2024-12-31",
        "stars_min": 3,
        "stars_max": 5,
        "sentiment": "positive",
        "gender": "female",
    }
    mapped = _map_filters(filters)
    assert mapped is not None
    assert mapped["created_date_from"] == "2024-01-01"
    assert mapped["created_date_to"] == "2024-12-31"
    assert mapped["rating_min"] == 3
    assert mapped["rating_max"] == 5
    assert mapped["sentiment_label"] == "positive"
    assert mapped["reviewer_gender"] == "female"


def test_map_filters_partial():
    from ml_service.services.rag_service import _map_filters
    filters = {"stars_min": 4}
    mapped = _map_filters(filters)
    assert mapped is not None
    assert "rating_min" in mapped
    assert "created_date_from" not in mapped


@pytest.mark.asyncio
async def test_rag_service_query_calls_orchestrator():
    mock_orchestrator = MagicMock()
    mock_result = MagicMock()
    mock_result.answer = "Тестовый ответ"
    mock_result.source_review_ids = ["r1", "r2"]
    mock_result.expanded_query = "расширенный запрос"
    mock_result.timings = {"total_sec": 1.5}
    mock_orchestrator.run = MagicMock(return_value=mock_result)

    from ml_service.services.rag_service import RAGService
    svc = RAGService()
    svc._orchestrator = mock_orchestrator

    result = await svc.query(
        query="Как размер товара?",
        product_id=42,
        top_k=10,
        filters=None,
    )
    assert result["answer"] == "Тестовый ответ"
    assert result["sources"] == ["r1", "r2"]
    assert result["expanded_query"] == "расширенный запрос"
    mock_orchestrator.run.assert_called_once()


@pytest.mark.asyncio
async def test_rag_service_query_with_filters():
    mock_orchestrator = MagicMock()
    mock_result = MagicMock()
    mock_result.answer = "Ответ"
    mock_result.source_review_ids = []
    mock_result.expanded_query = "query"
    mock_result.timings = {}
    mock_orchestrator.run = MagicMock(return_value=mock_result)

    from ml_service.services.rag_service import RAGService
    svc = RAGService()
    svc._orchestrator = mock_orchestrator

    filters = {"stars_min": 4, "sentiment": "positive"}
    await svc.query("Вопрос", product_id=1, top_k=5, filters=filters)

    call_kwargs = mock_orchestrator.run.call_args
    assert call_kwargs is not None
