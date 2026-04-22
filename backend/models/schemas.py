from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8)


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    user_id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class ProductCreate(BaseModel):
    name: str = Field(min_length=1, max_length=500)
    source_product_id: str = Field(min_length=1, max_length=255)
    platform: str = Field(pattern="^(wildberries|ozon)$")


class ProductResponse(BaseModel):
    product_id: int
    name: str
    source_product_id: str
    platform: str
    user_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class JobCreate(BaseModel):
    product_id: int
    job_type: str = Field(pattern="^(parsing|clustering|auto_reply)$")
    parameters: dict[str, Any] | None = None


class JobResponse(BaseModel):
    job_id: int
    product_id: int
    user_id: int
    job_type: str
    status: str
    parameters: dict | None
    results_summary: dict | None
    error_message: str | None
    start_time: datetime | None
    end_time: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ReviewResponse(BaseModel):
    review_id: str
    product_id: int
    rating: int | None
    advantages: str | None
    disadvantages: str | None
    comment: str | None
    reviewer_name: str | None
    created_date: datetime
    platform: str

    model_config = {"from_attributes": True}


class ReviewsListResponse(BaseModel):
    items: list[ReviewResponse]
    total: int
    page: int
    page_size: int


class RAGQueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    product_id: int
    top_k: int = Field(default=10, ge=5, le=40)
    filters: RAGFilters | None = None


class RAGFilters(BaseModel):
    date_from: datetime | None = None
    date_to: datetime | None = None
    stars_min: int | None = Field(default=None, ge=1, le=5)
    stars_max: int | None = Field(default=None, ge=1, le=5)
    sentiment: str | None = Field(default=None, pattern="^(positive|negative|neutral)$")
    gender: str | None = Field(default=None, pattern="^(male|female|unknown)$")


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    expanded_query: str
    timings: dict[str, float]


class ConversationCreate(BaseModel):
    product_id: int | None = None


class ConversationResponse(BaseModel):
    conversation_id: int
    product_id: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    content: str = Field(min_length=1)
    filters: RAGFilters | None = None
    top_k: int = Field(default=10, ge=5, le=40)


class MessageResponse(BaseModel):
    message_id: int
    conversation_id: int
    role: str
    content: str
    rag_review_ids: list[str] | None
    filters_applied: dict | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DashboardResponse(BaseModel):
    product_id: int
    total_reviews: int
    avg_rating: float | None
    positive_count: int
    negative_count: int
    neutral_count: int
    clusters_count: int
    last_analysis: datetime | None
    health_score: float | None
    top_problems: list[dict] | None
    top_positives: list[dict] | None


class ParseJobRequest(BaseModel):
    product_id: int
    max_reviews: int | None = Field(default=None, ge=1, le=10000)


class ClusterJobRequest(BaseModel):
    product_id: int


class AutoReplyJobRequest(BaseModel):
    review_ids: list[str] = Field(min_length=1)
    product_id: int


class AdminKafkaStats(BaseModel):
    topics: list[dict]
    consumer_groups: list[dict]


class AdminJobStats(BaseModel):
    pending: int
    running: int
    completed: int
    failed: int
    total: int


class CSVExportRequest(BaseModel):
    product_id: int
    include_sentiment: bool = True
    include_clusters: bool = True
    date_from: datetime | None = None
    date_to: datetime | None = None
