CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE sentiment_label AS ENUM ('positive', 'negative', 'neutral');

CREATE TABLE review_nlp (
    review_id       VARCHAR(255)    PRIMARY KEY,
    product_id      BIGINT          NOT NULL,
    merged_text     TEXT,
    tokens_count    INTEGER,
    is_informative  BOOLEAN         NOT NULL DEFAULT TRUE,
    processed_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_review_nlp_product ON review_nlp(product_id);
CREATE INDEX idx_review_nlp_informative ON review_nlp(is_informative);

CREATE TABLE review_sentiment (
    review_id           VARCHAR(255)    PRIMARY KEY REFERENCES review_nlp(review_id) ON DELETE CASCADE,
    product_id          BIGINT          NOT NULL,
    sentiment_label     sentiment_label NOT NULL,
    sentiment_score     FLOAT           CHECK (sentiment_score BETWEEN 0.0 AND 1.0),
    reviewer_gender     VARCHAR(10)     CHECK (reviewer_gender IN ('male', 'female', 'unknown')),
    processed_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_review_sentiment_product ON review_sentiment(product_id);
CREATE INDEX idx_review_sentiment_label ON review_sentiment(sentiment_label);
CREATE INDEX idx_review_sentiment_gender ON review_sentiment(reviewer_gender);

CREATE TABLE clusters (
    cluster_id          BIGSERIAL       PRIMARY KEY,
    clustering_job_id   BIGINT,
    product_id          BIGINT          NOT NULL,
    sentiment_category  VARCHAR(20)     NOT NULL CHECK (sentiment_category IN ('positive', 'negative')),
    bertopic_topic_id   INTEGER,
    llm_label           VARCHAR(255),
    keywords            JSONB,
    review_count        INTEGER         DEFAULT 0,
    avg_rating          REAL,
    avg_sentiment       REAL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_clusters_product ON clusters(product_id);
CREATE INDEX idx_clusters_job ON clusters(clustering_job_id);
CREATE INDEX idx_clusters_sentiment ON clusters(sentiment_category);

CREATE TABLE review_cluster_mapping (
    mapping_id      BIGSERIAL       PRIMARY KEY,
    review_id       VARCHAR(255)    NOT NULL,
    cluster_id      BIGINT          NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
    product_id      BIGINT          NOT NULL,
    probability     FLOAT           CHECK (probability BETWEEN 0.0 AND 1.0),
    is_outlier      BOOLEAN         DEFAULT FALSE,
    UNIQUE(review_id, cluster_id)
);

CREATE INDEX idx_rcm_cluster ON review_cluster_mapping(cluster_id);
CREATE INDEX idx_rcm_product ON review_cluster_mapping(product_id);
CREATE INDEX idx_rcm_review ON review_cluster_mapping(review_id);
CREATE INDEX idx_rcm_outlier ON review_cluster_mapping(is_outlier);

CREATE TABLE product_daily_insights (
    insight_id      BIGSERIAL   PRIMARY KEY,
    product_id      BIGINT      NOT NULL,
    analysis_date   DATE        NOT NULL,
    health_score    FLOAT       CHECK (health_score BETWEEN 0.0 AND 10.0),
    spam_rate       FLOAT       CHECK (spam_rate BETWEEN 0.0 AND 1.0),
    avg_rating      FLOAT,
    total_reviews   INTEGER,
    positive_count  INTEGER,
    negative_count  INTEGER,
    llm_summary     TEXT,
    top_problems    JSONB,
    top_positives   JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(product_id, analysis_date)
);

CREATE INDEX idx_pdi_product ON product_daily_insights(product_id);
CREATE INDEX idx_pdi_date ON product_daily_insights(analysis_date DESC);

CREATE TABLE auto_reply_log (
    reply_id        BIGSERIAL       PRIMARY KEY,
    review_id       VARCHAR(255)    UNIQUE NOT NULL,
    product_id      BIGINT          NOT NULL,
    job_id          BIGINT,
    generated_text  TEXT            NOT NULL,
    status          VARCHAR(20)     NOT NULL CHECK (status IN ('generated', 'sent', 'failed')) DEFAULT 'generated',
    error_message   TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_auto_reply_product ON auto_reply_log(product_id);
CREATE INDEX idx_auto_reply_status ON auto_reply_log(status);
CREATE INDEX idx_auto_reply_created ON auto_reply_log(created_at DESC);
