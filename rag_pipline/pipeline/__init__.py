"""Offline preprocessing + indexing pipeline."""

from .data_loader import load_reviews, prepare_records, build_combined_text  # noqa: F401
from .sentiment import SentimentAnalyzer  # noqa: F401
from .gender import GenderDetector  # noqa: F401
from .embedder import E5Embedder  # noqa: F401
from .indexer import QdrantStore, make_point_id  # noqa: F401
from .retriever import build_qdrant_filter, format_hit  # noqa: F401
