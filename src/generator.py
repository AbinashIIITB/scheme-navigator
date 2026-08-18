"""
generator.py — Grounded RAG prompt assembly, abstain guard, and LLM generation.

Pipeline:
    1. retrieve(query, k)  → top-k relevant chunks with similarity scores.
    2. Confidence check    → if highest score < threshold, ABSTAIN (no LLM call).
    3. Prompt assembly     → inject retrieved clauses with [Source: ..., Page ...] tags.
    4. Gemini call         → generate grounded answer citing the source clauses.
    5. Fallback loop       → tries fallback models if the primary model errors out.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from google import genai

from src.config import GOOGLE_API_KEY, LLM_MODEL_NAME, TOP_K, SIM_THRESHOLD
from src.vectorstore import retrieve

# ── Model fallback chain ───────────────────────────────────────────────────────
# If the primary model hits a rate limit or error, the generator will try
# the next one in this list. dict.fromkeys() preserves order and removes
# duplicates (important if LLM_MODEL_NAME matches one of the hardcoded names).
FALLBACK_MODELS = list(dict.fromkeys([
    LLM_MODEL_NAME,                # Configured model (default: gemini-3.6-flash)
    "gemini-3.6-flash",            # Latest stable Flash model
    "gemini-3.5-flash",            # Stable Flash fallback
    "gemini-3-flash-preview",      # Fast preview flash
    "gemini-flash-lite-latest",    # Lightweight fallback
    "gemini-flash-latest",         # Latest flash alias
]))

# ── Singleton Gemini client ───────────────────────────────────────────────────
# Initialised once on first use so we don't re-authenticate on every call.
_genai_client = None


def get_genai_client() -> genai.Client:
    """
    Return an authenticated Google GenAI client (created once, reused after).

    Reads GOOGLE_API_KEY from config (which checks Streamlit secrets → .env →
    environment variable, in that order).
    """
    global _genai_client
    if _genai_client is None:
        api_key = GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Add it to your .env file or Streamlit secrets."
            )
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Assemble a grounded RAG prompt from retrieved source clauses.

    Each clause is labelled with its document name, page, and similarity score
    so the LLM knows exactly where to cite from.

    Args:
        query:  The user's question.
        chunks: Retrieved chunk dicts from retrieve().

    Returns:
        A formatted string ready to send to the Gemini model.
    """
    context_blocks = []
    for idx, c in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source Clause {idx} | Document: {c.get('source', 'Unknown')} "
            f"| Page: {c.get('page', '?')} | Score: {c.get('score', 0.0):.3f}]\n"
            f"{c.get('text', '').strip()}"
        )

    context_str = "\n\n".join(context_blocks)

    return f"""You are the official Sarkari Scheme Navigator assistant. Your mission is to provide accurate, truthful, and helpful information about Indian Government schemes based STRICTLY on the official source documents provided below.

RULES:
1. Ground every factual answer directly in the source clauses provided below.
2. For every fact, eligibility criterion, grant amount, or rule you state, explicitly cite the source file name and page number (e.g. [Source: example.pdf, Page 1]).
3. If the context does not contain enough information to answer the question with certainty, state clearly that the provided scheme documents do not contain sufficient details to answer. Do NOT fabricate or assume details.
4. Keep the answer structured, clear, and easy to understand for citizens.

--- CONTEXT CLAUSES ---
{context_str}
-----------------------

Question: {query}

Answer:"""


def generate_llm_response(prompt: str) -> str:
    """
    Send a prompt to Google Gemini and return the response text.

    Automatically tries each model in FALLBACK_MODELS until one succeeds,
    so the app keeps working even during rate-limits or model deprecations.

    Args:
        prompt: The fully assembled RAG prompt.

    Returns:
        The model's response as a plain string.

    Raises:
        RuntimeError: If every model in FALLBACK_MODELS fails.
    """
    client = get_genai_client()
    last_err = None

    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_err = e
            continue  # Try next model

    raise RuntimeError(
        f"All models in FALLBACK_MODELS failed. Last error: {last_err}"
    )


def answer(
    query: str,
    k: int = TOP_K,
    threshold: float = SIM_THRESHOLD,
) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline: retrieve → check confidence → generate → cite.

    Args:
        query:     The user's question (plain text).
        k:         Number of chunks to retrieve from ChromaDB.
        threshold: Minimum cosine similarity to proceed with generation.
                   If the best chunk scores below this, the system abstains
                   rather than risk hallucinating an answer.

    Returns:
        {
            'query':            Original question,
            'answer':           Generated text (or abstention message),
            'sources':          List of cited source dicts,
            'abstained':        True if confidence was too low,
            'confidence':       Best chunk similarity score (float),
            'retrieved_chunks': Raw retrieval output (for debugging),
        }
    """
    query = query.strip()
    if not query:
        return {
            "query": query,
            "answer": "Please enter a valid question.",
            "sources": [],
            "abstained": True,
            "confidence": 0.0,
            "retrieved_chunks": [],
        }

    # ── Step 1: Retrieve the most relevant chunks ─────────────────────────────
    retrieved_chunks = retrieve(query, k=k)

    # ── Step 2: Confidence gate — abstain if best score is too low ────────────
    confidence = max((c["score"] for c in retrieved_chunks), default=0.0)

    if not retrieved_chunks or confidence < threshold:
        return {
            "query": query,
            "answer": (
                f"I cannot find sufficient relevant information in the government scheme "
                f"documents to answer this question reliably "
                f"(retrieval confidence {confidence:.2f} < threshold {threshold:.2f})."
            ),
            "sources": [],
            "abstained": True,
            "confidence": round(confidence, 4),
            "retrieved_chunks": retrieved_chunks,
        }

    # ── Step 3: Build grounded prompt and call Gemini ─────────────────────────
    prompt = build_prompt(query, retrieved_chunks)
    try:
        response_text = generate_llm_response(prompt)
    except Exception as e:
        response_text = f"Error generating answer from language model: {e}"

    # ── Step 4: Deduplicate sources for the citation panel ────────────────────
    seen_keys: set = set()
    sources = []
    for c in retrieved_chunks:
        key = (c["source"], c["page"])
        if key not in seen_keys:
            seen_keys.add(key)
            text = c["text"]
            sources.append({
                "source":  c["source"],
                "page":    c["page"],
                "score":   c["score"],
                "excerpt": text[:180] + "..." if len(text) > 180 else text,
            })

    return {
        "query":            query,
        "answer":           response_text,
        "sources":          sources,
        "abstained":        False,
        "confidence":       round(confidence, 4),
        "retrieved_chunks": retrieved_chunks,
    }


# ── Quick dev smoke-test: python src/generator.py ────────────────────────────
if __name__ == "__main__":
    in_scope = "How much prize money can the top 10 winners of the Arunachal Pradesh Entrepreneurship Challenge receive?"
    print(f"\nQuery (in-scope): {in_scope}")
    res = answer(in_scope)
    print("Answer:\n", res["answer"])
    print("Sources:", res["sources"])
    print("Confidence:", res["confidence"], "| Abstained:", res["abstained"])

    print("\n" + "=" * 60)
    out_scope = "What is the capital of France?"
    print(f"Query (out-of-scope): {out_scope}")
    res2 = answer(out_scope)
    print("Answer:\n", res2["answer"])
    print("Confidence:", res2["confidence"], "| Abstained:", res2["abstained"])
