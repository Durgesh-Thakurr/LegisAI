import os

import google.generativeai as genai
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI
from langdetect import detect
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
SIMILARITY_THRESHOLD = 0.45
TOP_K = 5

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-3.6-flash")

embed_model = SentenceTransformer("BAAI/bge-m3")

app = FastAPI(title="LegisAI API")

SYSTEM_PROMPT = """You are a legal information assistant. Answer ONLY using the CONTEXT below.
If the answer is not fully contained in the CONTEXT, say "I don't have verified information on this -- please consult the official source or a legal professional."
Do not use outside knowledge. Do not infer beyond what the CONTEXT states.
Quote section numbers exactly as they appear in CONTEXT.
Respond in {language}.
Always end with: "This is informational guidance based on official sources, not a substitute for professional legal advice."

CONTEXT:
{context}

QUESTION:
{question}
"""


class ChatRequest(BaseModel):
    session_id: str
    message: str


class Source(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: str


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def vector_literal(embedding):
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def detect_language(text: str) -> str:
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    return "Hindi" if lang == "hi" else "English"


def retrieve_chunks(query: str):
    query_embedding = embed_model.encode([query], normalize_embeddings=True)[0]
    query_vector = vector_literal(query_embedding)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content, source_url, title,
               1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vector, query_vector, TOP_K),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {"content": r[0], "source_url": r[1], "title": r[2], "similarity": r[3]}
        for r in rows
    ]


def build_context(chunks):
    return "\n\n---\n\n".join(
        f"[{c['title']}]\n{c['content']}" for c in chunks
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chunks = retrieve_chunks(req.message)

    if not chunks or chunks[0]["similarity"] < SIMILARITY_THRESHOLD:
        return ChatResponse(
            answer="I don't have verified information on this -- please consult the official source or a legal professional.",
            sources=[],
            confidence="low",
        )

    language = detect_language(req.message)
    context = build_context(chunks)
    prompt = SYSTEM_PROMPT.format(language=language, context=context, question=req.message)

    response = gemini_model.generate_content(prompt)
    answer = response.text

    seen_urls = set()
    sources = []
    for c in chunks:
        if c["source_url"] not in seen_urls:
            sources.append(Source(title=c["title"], url=c["source_url"]))
            seen_urls.add(c["source_url"])

    confidence = "high" if chunks[0]["similarity"] >= 0.6 else "low"

    return ChatResponse(answer=answer, sources=sources, confidence=confidence) 