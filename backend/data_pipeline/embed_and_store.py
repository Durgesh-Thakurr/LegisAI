import argparse
import hashlib
import json
import os
import sys

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_batch, execute_values
from sentence_transformers import SentenceTransformer

from common import BASE_DIR, CHUNKS_FILE

load_dotenv(BASE_DIR / ".env")

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16
EMBEDDING_DIM = 1024
MAX_DELETE_FRACTION = 0.5
REQUIRED_COLUMNS = {"content", "source_url", "title", "category", "section", "embedding"}

INSERT_SQL = """
    INSERT INTO chunks (content, source_url, title, category, section, embedding)
    VALUES %s
"""
INSERT_TEMPLATE = "(%s, %s, %s, %s, %s, %s::vector)"
UPDATE_SQL = "UPDATE chunks SET source_url = %s, title = %s, category = %s, section = %s WHERE id = %s"


def vector_literal(embedding):
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def metadata_of(chunk):
    return (chunk["source_url"], chunk["title"], chunk.get("category") or "unknown", chunk.get("section"))


def connect(database_url):
    return psycopg2.connect(database_url, connect_timeout=15)


def check_schema(database_url):
    conn = connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'chunks'"
            )
            existing = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    if not existing:
        sys.exit("Table 'chunks' does not exist. Run backend/schema.sql in the Neon SQL editor first.")
    missing = REQUIRED_COLUMNS - existing
    if missing:
        sys.exit(f"Table 'chunks' is missing columns: {sorted(missing)}. Run backend/schema.sql first.")


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        raw = [chunk for chunk in json.load(f) if chunk.get("content", "").strip()]
    unique = {}
    for chunk in raw:
        unique.setdefault(content_hash(chunk["content"]), chunk)
    return list(unique.values())


def fetch_existing(cur):
    cur.execute("SELECT id, content, source_url, title, category, section FROM chunks ORDER BY id")
    by_hash, duplicate_ids = {}, []
    for row_id, content, source_url, title, category, section in cur.fetchall():
        digest = content_hash(content)
        if digest in by_hash:
            duplicate_ids.append(row_id)
        else:
            by_hash[digest] = (row_id, (source_url, title, category or "unknown", section))
    return by_hash, duplicate_ids


def read_existing(database_url):
    conn = connect(database_url)
    try:
        with conn.cursor() as cur:
            return fetch_existing(cur)
    finally:
        conn.close()


def plan_sync(chunks, existing, duplicate_ids):
    wanted, to_embed, to_update, kept_ids = set(), [], [], set()
    for chunk in chunks:
        digest = content_hash(chunk["content"])
        wanted.add(digest)
        current = existing.get(digest)
        if current is None:
            to_embed.append(chunk)
            continue
        kept_ids.add(current[0])
        if current[1] != metadata_of(chunk):
            to_update.append((current[0], chunk))
    to_delete = duplicate_ids + [row_id for digest, (row_id, _) in existing.items() if digest not in wanted]
    return to_embed, to_update, to_delete, kept_ids


def embed(texts):
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)


def build_rows(chunks, embeddings):
    return [
        (chunk["content"], *metadata_of(chunk), vector_literal(embedding))
        for chunk, embedding in zip(chunks, embeddings)
    ]


def apply_changes(database_url, rows, to_update, to_delete, kept_ids, full):
    conn = connect(database_url)
    try:
        with conn, conn.cursor() as cur:
            cur.execute("LOCK TABLE chunks IN SHARE ROW EXCLUSIVE MODE")
            if full:
                cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
            else:
                cur.execute("SELECT id FROM chunks")
                vanished = kept_ids - {row[0] for row in cur.fetchall()}
                if vanished:
                    sys.exit("The chunks table changed while embedding. Nothing was written; run the sync again.")
                if to_delete:
                    cur.execute("DELETE FROM chunks WHERE id = ANY(%s)", (to_delete,))
                if to_update:
                    execute_batch(cur, UPDATE_SQL, [(*metadata_of(chunk), row_id) for row_id, chunk in to_update])
            if rows:
                execute_values(cur, INSERT_SQL, rows, template=INSERT_TEMPLATE, page_size=100)
    finally:
        conn.close()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Sync chunks.json into the chunks table, embedding only what changed")
    parser.add_argument("--full", action="store_true", help="re-embed everything (needed after changing the model)")
    parser.add_argument("--allow-large-delete", action="store_true", help="allow removing more than half of the existing rows")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        sys.exit("DATABASE_URL is not set")

    check_schema(database_url)

    chunks = load_chunks()
    if not chunks:
        sys.exit("chunks.json has no usable chunks; refusing to change the table")

    if args.full:
        to_embed, to_update, to_delete, kept_ids = chunks, [], [], set()
    else:
        existing, duplicate_ids = read_existing(database_url)
        to_embed, to_update, to_delete, kept_ids = plan_sync(chunks, existing, duplicate_ids)
        total_existing = len(existing) + len(duplicate_ids)
        if total_existing and len(to_delete) > MAX_DELETE_FRACTION * total_existing and not args.allow_large_delete:
            sys.exit(
                f"This would delete {len(to_delete)} of {total_existing} existing chunks. "
                "If chunks.json is incomplete (a source failed to download?), fix that first; "
                "otherwise rerun with --allow-large-delete."
            )

    embeddings = embed([chunk["content"] for chunk in to_embed]) if to_embed else []
    if to_embed and len(embeddings[0]) != EMBEDDING_DIM:
        sys.exit(f"Embedding dimension {len(embeddings[0])} does not match schema ({EMBEDDING_DIM})")

    apply_changes(database_url, build_rows(to_embed, embeddings), to_update, to_delete, kept_ids, args.full)

    print(
        f"Embedded {len(to_embed)}, kept {len(kept_ids)} unchanged, updated metadata on {len(to_update)}, "
        f"deleted {len(to_delete)}. Table holds {len(chunks)} chunks."
    )


if __name__ == "__main__":
    main() 