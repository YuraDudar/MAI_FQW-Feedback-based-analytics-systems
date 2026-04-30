CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

CREATE TYPE user_role AS ENUM ('admin', 'analyst');
CREATE TYPE job_type AS ENUM ('parsing', 'clustering', 'auto_reply');
CREATE TYPE job_status AS ENUM ('pending', 'running', 'completed', 'failed');
CREATE TYPE platform_type AS ENUM ('wildberries', 'ozon');

CREATE TABLE users (
    user_id     BIGSERIAL       PRIMARY KEY,
    username    VARCHAR(100)    UNIQUE NOT NULL,
    email       VARCHAR(255)    UNIQUE NOT NULL,
    password_hash VARCHAR(255)  NOT NULL,
    role        user_role       NOT NULL DEFAULT 'analyst',
    is_active   BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE data_sources (
    source_id   SERIAL          PRIMARY KEY,
    name        VARCHAR(100)    UNIQUE NOT NULL,
    platform    platform_type   NOT NULL,
    site_url    VARCHAR(255)    NOT NULL
);

INSERT INTO data_sources (name, platform, site_url) VALUES
    ('Wildberries', 'wildberries', 'https://www.wildberries.ru'),
    ('Ozon',        'ozon',        'https://www.ozon.ru');

CREATE TABLE products (
    product_id          BIGSERIAL       PRIMARY KEY,
    name                TEXT            NOT NULL,
    source_product_id   VARCHAR(255)    NOT NULL,
    source_id           INT             NOT NULL REFERENCES data_sources(source_id),
    user_id             BIGINT          NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(source_id, source_product_id)
);

CREATE INDEX idx_products_user_id ON products(user_id);
CREATE INDEX idx_products_source ON products(source_id, source_product_id);

CREATE TABLE analysis_jobs (
    job_id          BIGSERIAL       PRIMARY KEY,
    product_id      BIGINT          NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    user_id         BIGINT          NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    job_type        job_type        NOT NULL,
    status          job_status      NOT NULL DEFAULT 'pending',
    parameters      JSONB,
    results_summary JSONB,
    error_message   TEXT,
    start_time      TIMESTAMPTZ,
    end_time        TIMESTAMPTZ,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_analysis_jobs_product ON analysis_jobs(product_id);
CREATE INDEX idx_analysis_jobs_user ON analysis_jobs(user_id);
CREATE INDEX idx_analysis_jobs_status ON analysis_jobs(status);
CREATE INDEX idx_analysis_jobs_created ON analysis_jobs(created_at DESC);

CREATE TABLE raw_reviews (
    review_id               VARCHAR(255)    PRIMARY KEY,
    product_id              BIGINT          NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    parsing_job_id          BIGINT          REFERENCES analysis_jobs(job_id),
    input_sku               VARCHAR(255),
    parsed_at               TIMESTAMPTZ,
    platform                platform_type   NOT NULL,
    nm_id                   BIGINT,
    wb_user_id              BIGINT,
    global_user_id          VARCHAR(255),
    reviewer_name           VARCHAR(255),
    reviewer_country        VARCHAR(10),
    reviewer_has_avatar     BOOLEAN,
    rating                  SMALLINT        CHECK (rating BETWEEN 1 AND 5),
    advantages              TEXT,
    disadvantages           TEXT,
    comment                 TEXT,
    variant_color           VARCHAR(255),
    size                    VARCHAR(100),
    tags                    TEXT,
    created_date            TIMESTAMPTZ     NOT NULL,
    updated_date            TIMESTAMPTZ,
    status_id               SMALLINT,
    purchase_status         VARCHAR(100),
    seller_response         TEXT,
    seller_response_state   VARCHAR(50),
    matching_size           VARCHAR(255),
    matching_photo          VARCHAR(255),
    matching_description    VARCHAR(255),
    votes_plus              INTEGER         DEFAULT 0,
    votes_minus             INTEGER         DEFAULT 0,
    rank                    FLOAT,
    helpfulness             VARCHAR(100),
    has_video               BOOLEAN         DEFAULT FALSE,
    excluded_from_rating    BOOLEAN         DEFAULT FALSE,
    excluded_reasons        TEXT,
    good_reasons            TEXT,
    bad_reasons             TEXT
);

CREATE UNIQUE INDEX idx_raw_reviews_product_review ON raw_reviews(product_id, review_id);
CREATE INDEX idx_raw_reviews_created_date ON raw_reviews(created_date DESC);
CREATE INDEX idx_raw_reviews_product_id ON raw_reviews(product_id);
CREATE INDEX idx_raw_reviews_parsing_job ON raw_reviews(parsing_job_id);
CREATE INDEX idx_raw_reviews_rating ON raw_reviews(rating);

CREATE TABLE llm_conversations (
    conversation_id BIGSERIAL       PRIMARY KEY,
    user_id         BIGINT          NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    product_id      BIGINT          REFERENCES products(product_id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_llm_conversations_user ON llm_conversations(user_id);
CREATE INDEX idx_llm_conversations_product ON llm_conversations(product_id);

CREATE TABLE llm_messages (
    message_id          BIGSERIAL   PRIMARY KEY,
    conversation_id     BIGINT      NOT NULL REFERENCES llm_conversations(conversation_id) ON DELETE CASCADE,
    role                VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content             TEXT        NOT NULL,
    rag_review_ids      JSONB,
    filters_applied     JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_llm_messages_conversation ON llm_messages(conversation_id);
CREATE INDEX idx_llm_messages_created ON llm_messages(created_at DESC);

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
