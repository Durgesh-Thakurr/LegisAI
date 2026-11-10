import os
import time
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

PROMPT = "In two sentences, explain what a SIM swap fraud is."
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
RATE_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
)


def timed(label, call):
    started = time.monotonic()
    try:
        print(f"[OK]   {label}: {call()} ({time.monotonic() - started:.1f}s)")
    except Exception as exc:
        print(f"[FAIL] {label}: {type(exc).__name__} after {time.monotonic() - started:.1f}s: {str(exc)[:200]}")


def check_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[SKIP] Gemini: GEMINI_API_KEY is not set")
        return
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)),
    )

    def list_models():
        names = [m.name.removeprefix("models/") for m in client.models.list() if "flash" in m.name]
        return ", ".join(names)[:300] or "none with 'flash' in the name"

    def generate(thinking):
        config = types.GenerateContentConfig(max_output_tokens=512, thinking_config=thinking)
        return client.models.generate_content(model=GEMINI_MODEL, contents=PROMPT, config=config).text.strip()[:60]

    timed("Gemini flash models available to this key", list_models)
    variants = {
        "no thinking setting": None,
        "thinking_level MINIMAL": types.ThinkingConfig(thinking_level="MINIMAL"),
        "thinking_level LOW": types.ThinkingConfig(thinking_level="LOW"),
        "thinking_budget 0": types.ThinkingConfig(thinking_budget=0),
    }
    for label, thinking in variants.items():
        timed(f"Gemini '{GEMINI_MODEL}', {label}", lambda thinking=thinking: generate(thinking))


def check_groq():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[SKIP] Groq: GROQ_API_KEY is not set")
        return
    from groq import Groq

    client = Groq(api_key=api_key, timeout=20, max_retries=0)
    extra = {"extra_body": {"reasoning_effort": "low"}} if "gpt-oss" in GROQ_MODEL else {}

    def generate():
        raw = client.chat.completions.with_raw_response.create(
            model=GROQ_MODEL, messages=[{"role": "user", "content": PROMPT}], max_completion_tokens=400, **extra
        )
        limits = ", ".join(f"{name.removeprefix('x-ratelimit-')}={raw.headers.get(name)}" for name in RATE_HEADERS)
        print(f"       rate limits: {limits}")
        return (raw.parse().choices[0].message.content or "").strip()[:60]

    timed(f"Groq answer from '{GROQ_MODEL}'", generate)


if __name__ == "__main__":
    check_gemini()
    check_groq()