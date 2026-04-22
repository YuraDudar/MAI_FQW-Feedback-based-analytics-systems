import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, Column, DateTime, Enum,
    Float, ForeignKey, Index, Integer, SmallInteger, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    analyst = "analyst"


class JobType(str, enum.Enum):
    parsing = "parsing"
    clustering = "clustering"
    auto_reply = "auto_reply"


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class PlatformType(str, enum.Enum):
    wildberries = "wildberries"
    ozon = "ozon"


class User(Base):
    __tablename__ = "users"

    user_id = Column(BigInteger, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.analyst)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    products = relationship("Product", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("AnalysisJob", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("LLMConversation", back_populates="user", cascade="all, delete-orphan")


class DataSource(Base):
    __tablename__ = "data_sources"

    source_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    platform = Column(Enum(PlatformType), nullable=False)
    site_url = Column(String(255), nullable=False)

    products = relationship("Product", back_populates="source")


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        UniqueConstraint("source_id", "source_product_id"),
    )

    product_id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    source_product_id = Column(String(255), nullable=False)
    source_id = Column(Integer, ForeignKey("data_sources.source_id"), nullable=False)
    user_id = Column(BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="products")
    source = relationship("DataSource", back_populates="products")
    jobs = relationship("AnalysisJob", back_populates="product", cascade="all, delete-orphan")
    reviews = relationship("RawReview", back_populates="product", cascade="all, delete-orphan")
    conversations = relationship("LLMConversation", back_populates="product")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    job_id = Column(BigInteger, primary_key=True, autoincrement=True)
    product_id = Column(BigInteger, ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    job_type = Column(Enum(JobType), nullable=False)
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.pending)
    parameters = Column(JSONB)
    results_summary = Column(JSONB)
    error_message = Column(Text)
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    product = relationship("Product", back_populates="jobs")
    user = relationship("User", back_populates="jobs")
    reviews = relationship("RawReview", back_populates="parsing_job")


class RawReview(Base):
    __tablename__ = "raw_reviews"

    review_id = Column(String(255), primary_key=True)
    product_id = Column(BigInteger, ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False)
    parsing_job_id = Column(BigInteger, ForeignKey("analysis_jobs.job_id"), nullable=True)
    input_sku = Column(String(255))
    parsed_at = Column(DateTime(timezone=True))
    platform = Column(Enum(PlatformType), nullable=False)
    nm_id = Column(BigInteger)
    wb_user_id = Column(BigInteger)
    global_user_id = Column(String(255))
    reviewer_name = Column(String(255))
    reviewer_country = Column(String(10))
    reviewer_has_avatar = Column(Boolean)
    rating = Column(SmallInteger, CheckConstraint("rating BETWEEN 1 AND 5"))
    advantages = Column(Text)
    disadvantages = Column(Text)
    comment = Column(Text)
    variant_color = Column(String(255))
    size = Column(String(100))
    tags = Column(Text)
    created_date = Column(DateTime(timezone=True), nullable=False)
    updated_date = Column(DateTime(timezone=True))
    status_id = Column(SmallInteger)
    purchase_status = Column(String(100))
    seller_response = Column(Text)
    seller_response_state = Column(String(50))
    matching_size = Column(String(255))
    matching_photo = Column(String(255))
    matching_description = Column(String(255))
    votes_plus = Column(Integer, default=0)
    votes_minus = Column(Integer, default=0)
    rank = Column(Float)
    helpfulness = Column(String(100))
    has_video = Column(Boolean, default=False)
    excluded_from_rating = Column(Boolean, default=False)
    excluded_reasons = Column(Text)
    good_reasons = Column(Text)
    bad_reasons = Column(Text)

    product = relationship("Product", back_populates="reviews")
    parsing_job = relationship("AnalysisJob", back_populates="reviews")

    __table_args__ = (
        Index("idx_raw_reviews_product_review", "product_id", "review_id", unique=True),
        Index("idx_raw_reviews_created_date", "created_date"),
        Index("idx_raw_reviews_product_id", "product_id"),
        Index("idx_raw_reviews_rating", "rating"),
    )


class LLMConversation(Base):
    __tablename__ = "llm_conversations"

    conversation_id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    product_id = Column(BigInteger, ForeignKey("products.product_id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="conversations")
    product = relationship("Product", back_populates="conversations")
    messages = relationship("LLMMessage", back_populates="conversation", cascade="all, delete-orphan")


class LLMMessage(Base):
    __tablename__ = "llm_messages"

    message_id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(BigInteger, ForeignKey("llm_conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), CheckConstraint("role IN ('user', 'assistant')"), nullable=False)
    content = Column(Text, nullable=False)
    rag_review_ids = Column(JSONB)
    filters_applied = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("LLMConversation", back_populates="messages")
