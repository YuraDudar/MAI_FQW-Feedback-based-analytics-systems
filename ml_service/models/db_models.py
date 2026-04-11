import enum

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, Column, Date, DateTime,
    Enum, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database import Base


class SentimentLabel(str, enum.Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class ReviewNLP(Base):
    __tablename__ = "review_nlp"

    review_id = Column(String(255), primary_key=True)
    product_id = Column(BigInteger, nullable=False)
    merged_text = Column(Text)
    tokens_count = Column(Integer)
    is_informative = Column(Boolean, nullable=False, default=True)
    processed_at = Column(DateTime(timezone=True), server_default=func.now())

    sentiment = relationship("ReviewSentiment", back_populates="nlp", uselist=False)

    __table_args__ = (
        Index("idx_review_nlp_product", "product_id"),
    )


class ReviewSentiment(Base):
    __tablename__ = "review_sentiment"

    review_id = Column(String(255), ForeignKey("review_nlp.review_id", ondelete="CASCADE"), primary_key=True)
    product_id = Column(BigInteger, nullable=False)
    sentiment_label = Column(Enum(SentimentLabel), nullable=False)
    sentiment_score = Column(Float, CheckConstraint("sentiment_score BETWEEN 0.0 AND 1.0"))
    reviewer_gender = Column(String(10), CheckConstraint("reviewer_gender IN ('male', 'female', 'unknown')"))
    processed_at = Column(DateTime(timezone=True), server_default=func.now())

    nlp = relationship("ReviewNLP", back_populates="sentiment")

    __table_args__ = (
        Index("idx_review_sentiment_product", "product_id"),
        Index("idx_review_sentiment_label", "sentiment_label"),
        Index("idx_review_sentiment_gender", "reviewer_gender"),
    )


class Cluster(Base):
    __tablename__ = "clusters"

    cluster_id = Column(BigInteger, primary_key=True, autoincrement=True)
    clustering_job_id = Column(BigInteger)
    product_id = Column(BigInteger, nullable=False)
    sentiment_category = Column(String(20), CheckConstraint("sentiment_category IN ('positive', 'negative')"), nullable=False)
    bertopic_topic_id = Column(Integer)
    llm_label = Column(String(255))
    keywords = Column(JSONB)
    review_count = Column(Integer, default=0)
    avg_rating = Column(Float)
    avg_sentiment = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    mappings = relationship("ReviewClusterMapping", back_populates="cluster", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_clusters_product", "product_id"),
        Index("idx_clusters_job", "clustering_job_id"),
        Index("idx_clusters_sentiment", "sentiment_category"),
    )


class ReviewClusterMapping(Base):
    __tablename__ = "review_cluster_mapping"
    __table_args__ = (
        UniqueConstraint("review_id", "cluster_id"),
        Index("idx_rcm_cluster", "cluster_id"),
        Index("idx_rcm_product", "product_id"),
        Index("idx_rcm_review", "review_id"),
    )

    mapping_id = Column(BigInteger, primary_key=True, autoincrement=True)
    review_id = Column(String(255), nullable=False)
    cluster_id = Column(BigInteger, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False)
    product_id = Column(BigInteger, nullable=False)
    probability = Column(Float, CheckConstraint("probability BETWEEN 0.0 AND 1.0"))
    is_outlier = Column(Boolean, default=False)

    cluster = relationship("Cluster", back_populates="mappings")


class ProductDailyInsights(Base):
    __tablename__ = "product_daily_insights"
    __table_args__ = (
        UniqueConstraint("product_id", "analysis_date"),
        Index("idx_pdi_product", "product_id"),
        Index("idx_pdi_date", "analysis_date"),
    )

    insight_id = Column(BigInteger, primary_key=True, autoincrement=True)
    product_id = Column(BigInteger, nullable=False)
    analysis_date = Column(Date, nullable=False)
    health_score = Column(Float, CheckConstraint("health_score BETWEEN 0.0 AND 10.0"))
    spam_rate = Column(Float, CheckConstraint("spam_rate BETWEEN 0.0 AND 1.0"))
    avg_rating = Column(Float)
    total_reviews = Column(Integer)
    positive_count = Column(Integer)
    negative_count = Column(Integer)
    llm_summary = Column(Text)
    top_problems = Column(JSONB)
    top_positives = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AutoReplyLog(Base):
    __tablename__ = "auto_reply_log"
    __table_args__ = (
        Index("idx_auto_reply_product", "product_id"),
        Index("idx_auto_reply_status", "status"),
    )

    reply_id = Column(BigInteger, primary_key=True, autoincrement=True)
    review_id = Column(String(255), unique=True, nullable=False)
    product_id = Column(BigInteger, nullable=False)
    job_id = Column(BigInteger)
    generated_text = Column(Text, nullable=False)
    status = Column(String(20), CheckConstraint("status IN ('generated', 'sent', 'failed')"), nullable=False, default="generated")
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
