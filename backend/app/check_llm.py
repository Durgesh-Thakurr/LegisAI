import os
import time
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

PROMPT = "Reply with the single word OK."
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def timed(label, call):
    started = time.monotonic()
    try:
        print(f"[OK]   {label}: {call()} ({time.monotonic() - started:.1f}s)")
    except Exception as exc:
        print(f"[FAIL] {label}: {type(exc).__name__} after {time.monotonic() - started:.1f}s: {str(exc)[:220]}")


def check_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[SKIP] Gemini: GEMINI_API_KEY is not set")
        return
    import google.generativeai as genai

    genai.configure(api_key=api_key, transport="rest")

    def list_models():
        names = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        return ", ".join(n.removeprefix("models/") for n in names if "flash" in n)[:300] or "none with 'flash' in the name"

    def generate():
        model = genai.GenerativeModel(GEMINI_MODEL)
        return model.generate_content(PROMPT, request_options={"timeout": 20, "retry": None}).text.strip()

    timed("Gemini models available to this key", list_models)
    timed(f"Gemini answer from '{GEMINI_MODEL}'", generate)


def check_groq():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[SKIP] Groq: GROQ_API_KEY is not set")
        return
    from groq import Groq

    client = Groq(api_key=api_key, timeout=15, max_retries=0)

    def generate():
        reply = client.chat.completions.create(model=GROQ_MODEL, messages=[{"role": "user", "content": PROMPT}])
        return (reply.choices[0].message.content or "").strip()

    timed(f"Groq answer from '{GROQ_MODEL}'", generate)


if __name__ == "__main__":
    check_gemini()
    check_groq()