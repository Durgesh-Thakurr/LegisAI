import os
import re

import google.generativeai as genai
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
SIMILARITY_THRESHOLD = 0.35
TOP_K = 8

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-3.6-flash")
groq_client = Groq(api_key=GROQ_API_KEY)

embed_model = SentenceTransformer("BAAI/bge-m3")

app = FastAPI(title="LegisAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GREETING_PATTERN = re.compile(
    r"^\s*(hi|hii+|hello+|hey+|namaste|namaskar|yo|kaise ho|kya haal|how are you)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
HINGLISH_HINTS = re.compile(r"\b(kaise|kya|ho|hai|haal|kr|kro|mujhe|mera|meri|hua|sath)\b", re.IGNORECASE)

GREETING_RESPONSES = {
    "english": "Hey there! I'm LegisAI -- think of me as your friendly guide through Indian cyber law. Got scammed, confused about a law, or just want to know your rights online? Tell me what happened and I'll dig through official sources to help.",
    "hindi": "Namaste! Main LegisAI hoon -- Bharatiya cyber kanoon mein aapki madad ke liye yahan hoon. OTP fraud, UPI scam, ya koi aur cyber problem? Bataiye kya hua, main official sources se aapki madad karunga.",
    "hinglish": "Hey! Main LegisAI hoon -- Indian cyber law mein tumhari madad karne ke liye. Scam hua, confusion hai kisi law ko lekar, ya bas apne rights jaanne hain? Bata do kya hua, official sources se poori madad karta hoon.",
}


def pick_greeting_response(message: str) -> str:
    if DEVANAGARI_PATTERN.search(message):
        return GREETING_RESPONSES["hindi"]
    if HINGLISH_HINTS.search(message):
        return GREETING_RESPONSES["hinglish"]
    return GREETING_RESPONSES["english"]


def detect_response_language(text: str) -> str:
    if DEVANAGARI_PATTERN.search(text):
        return "Hindi (Devanagari script)"
    if HINGLISH_HINTS.search(text):
        return "Hinglish (Hindi words written in Roman/English script, mixed with English)"
    return "English"


def is_non_english(text: str) -> bool:
    return bool(DEVANAGARI_PATTERN.search(text) or HINGLISH_HINTS.search(text))


TRANSLATE_PROMPT = """Translate the following question into clear, simple English for a search query.
Only output the translated question, nothing else -- no explanation, no quotes.

QUESTION:
{question}
"""

SYSTEM_PROMPT = """You are a legal information assistant. Answer using ONLY the CONTEXT below -- never use outside knowledge or infer beyond what CONTEXT states.

If the CONTEXT contains ANY relevant information -- even if partial or incomplete -- synthesize the best possible answer from it, and clearly note what specific detail is missing at the end (e.g. "The context does not specify X").
Only say "I don't have verified information on this -- please consult the official source or a legal professional" if the CONTEXT has NO relevant information at all on the topic.
Quote section numbers exactly as they appear in CONTEXT.
You MUST respond in {language}. This is mandatory.
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
    model_used: str


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def vector_literal(embedding):
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def call_llm_with_fallback(prompt: str):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip(), "gemini"
    except Exception as gemini_error:
        print(f"Gemini failed, falling back to Groq: {gemini_error}")
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip(), "groq"
        except Exception as groq_error:
            print(f"Groq also failed: {groq_error}")
            raise


def translate_to_english(question: str) -> str:
    prompt = TRANSLATE_PROMPT.format(question=question)
    try:
        translated, _ = call_llm_with_fallback(prompt)
        print(f"Translated query: {translated}")
        return translated
    except Exception as e:
        print(f"Translation failed, using original query: {e}")
        return question


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
    if GREETING_PATTERN.match(req.message):
        return ChatResponse(
            answer=pick_greeting_response(req.message),
            sources=[],
            confidence="high",
            model_used="none",
        )

    language = detect_response_language(req.message)

    chunks = retrieve_chunks(req.message)
    top_similarity = chunks[0]["similarity"] if chunks else 0
    print(f"Top similarity (native query): {top_similarity}")

    if top_similarity < SIMILARITY_THRESHOLD and is_non_english(req.message):
        translated_query = translate_to_english(req.message)
        retry_chunks = retrieve_chunks(translated_query)
        retry_similarity = retry_chunks[0]["similarity"] if retry_chunks else 0
        print(f"Top similarity (translated query): {retry_similarity}")
        if retry_similarity > top_similarity:
            chunks = retry_chunks
            top_similarity = retry_similarity

    if not chunks or top_similarity < SIMILARITY_THRESHOLD:
        return ChatResponse(
            answer="I don't have verified information on this -- please consult the official source or a legal professional.",
            sources=[],
            confidence="low",
            model_used="none",
        )

    context = build_context(chunks)
    prompt = SYSTEM_PROMPT.format(context=context, question=req.message, language=language)

    answer, model_used = call_llm_with_fallback(prompt)

    seen_urls = set()
    sources = []
    for c in chunks:
        if c["source_url"] not in seen_urls:
            sources.append(Source(title=c["title"], url=c["source_url"]))
            seen_urls.add(c["source_url"])

    confidence = "high" if top_similarity >= 0.6 else "low"

    return ChatResponse(answer=answer, sources=sources, confidence=confidence, model_used=model_used) 