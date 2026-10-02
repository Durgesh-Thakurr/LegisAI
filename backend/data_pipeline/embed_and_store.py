import json
import os
import sys

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common import BASE_DIR, CHUNKS_FILE

load_dotenv(BASE_DIR / ".env")

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16
EMBEDDING_DIM = 1024

INSERT_SQL = """
    INSERT INTO chunks (content, source_url, title, category, section, embedding)
    VALUES %s
"""
INSERT_TEMPLATE = "(%s, %s, %s, %s, %s, %s::vector)"


def vector_literal(embedding):
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return [chunk for chunk in json.load(f) if chunk.get("content", "").strip()]


def embed(texts):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        embeddings.extend(model.encode(texts[start:start + BATCH_SIZE], normalize_embeddings=True))
    return embeddings


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        sys.exit("DATABASE_URL is not set")

    chunks = load_chunks()
    if not chunks:
        sys.exit("chunks.json has no usable chunks; refusing to wipe the table")

    embeddings = embed([chunk["content"] for chunk in chunks])
    if len(embeddings[0]) != EMBEDDING_DIM:
        sys.exit(f"Embedding dimension {len(embeddings[0])} does not match schema ({EMBEDDING_DIM})")

    rows = [
        (
            chunk["content"],
            chunk["source_url"],
            chunk["title"],
            chunk.get("category", "unknown"),
            chunk.get("section"),
            vector_literal(embedding),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    conn = psycopg2.connect(database_url, connect_timeout=15)
    try:
        with conn, conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
            execute_values(cur, INSERT_SQL, rows, template=INSERT_TEMPLATE, page_size=100)
    finally:
        conn.close()

    print(f"Stored {len(rows)} chunks")


if __name__ == "__main__":
    main() 