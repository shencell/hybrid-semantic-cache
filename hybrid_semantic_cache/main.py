"""Self-contained FastAPI demo of the hybrid semantic cache.

Run it with the ``demo`` extras installed::

    pip install "hybrid-semantic-cache[demo]"
    uvicorn hybrid_semantic_cache.main:app --reload

POST a JSON body ``{"message": "..."}`` to ``/chat``. On a cache miss the demo
falls back to Google Gemini when ``GEMINI_API_KEY`` is set; without a key it
returns a stub answer so the caching behaviour itself is still observable.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from .cache import HybridSemanticCache
from .normalizer import normalize_text

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid Semantic Cache - Demo",
    description="Normalise -> semantic cache lookup -> LLM fallback on miss.",
    version="0.2.0",
)

# One process-wide cache. Tune the threshold to your embedding model.
cache = HybridSemanticCache(threshold=0.80, max_records=1000)


class ChatRequest(BaseModel):
    message: str


def _llm_fallback(prompt: str) -> str:
    """Answer with Gemini when configured, otherwise return a stub."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return f"[no GEMINI_API_KEY set - stub answer for: {prompt}]"

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    return model.generate_content(prompt).text


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    """Normalise the prompt, try the semantic cache, fall back to the LLM."""
    started = time.perf_counter()

    clean = normalize_text(request.message)
    hit = cache.search(clean)

    if hit is not None:
        return {
            "answer": hit.response,
            "source": "cache",
            "similarity": hit.score,
            "normalized_query": clean,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    answer = _llm_fallback(clean)
    cache.add(clean, answer)
    return {
        "answer": answer,
        "source": "llm",
        "similarity": None,
        "normalized_query": clean,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.get("/")
async def root() -> dict:
    return {"status": "online", "stats": cache.stats(), "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
