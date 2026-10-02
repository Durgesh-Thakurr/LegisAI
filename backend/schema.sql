CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id         BIGSERIAL PRIMARY KEY,
    content    TEXT NOT NULL,
    source_url TEXT NOT NULL,
    title      TEXT NOT NULL,
    category   TEXT,
    section    TEXT,
    embedding  vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section TEXT;

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Read-only role for the API; use its connection string in backend/app/.env.
-- CREATE ROLE legisai_reader LOGIN PASSWORD 'change-me';
-- GRANT USAGE ON SCHEMA public TO legisai_reader;
-- GRANT SELECT ON chunks TO legisai_reader; 