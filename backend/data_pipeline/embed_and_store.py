import json
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

CHUNKS_FILE = Path("chunks.json")
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_connection():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set -- check your .env file")
    return psycopg2.connect(url)


def vector_literal(embedding):
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {MODEL_NAME} (first run downloads ~2GB)")
    model = SentenceTransformer(MODEL_NAME)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks")
    existing = cur.fetchone()[0]
    if existing > 0:
        print(f"Table already has {existing} rows. Clearing before re-inserting.")
        cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
        conn.commit()

    insert_query = """
        INSERT INTO chunks (content, source_url, title, category, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
    """

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding + inserting"):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True)

        rows = [
            (c["content"], c["source_url"], c["title"], c["category"], vector_literal(emb))
            for c, emb in zip(batch, embeddings)
        ]
        cur.executemany(insert_query, rows)
        conn.commit()

    cur.execute("SELECT COUNT(*) FROM chunks")
    total = cur.fetchone()[0]
    print(f"\nDone. {total} rows in chunks table.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main() 