import os
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import errors

from src.config import (
    GOOGLE_API_KEY,
    LLM_MODEL_NAME,
    TOP_K,
    SIM_THRESHOLD,
)
from src.vectorstore import retrieve

# List of preferred model names in case of deprecation/rate-limit failover
FALLBACK_MODELS = [
    LLM_MODEL_NAME,
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-pro",
]

_genai_client = None


def get_genai_client() -> genai.Client:
    """
    Returns an authenticated Google GenAI client instance.
    """
    global _genai_client
    if _genai_client is None:
        api_key = GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in environment or .env file.")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Constructs a grounded RAG prompt strictly requiring source clause citations.
    """
    context_blocks = []
    for idx, c in enumerate(chunks, start=1):
        source = c.get("source", "Unknown")
        page = c.get("page", "?")
        score = c.get("score", 0.0)
        text = c.get("text", "").strip()
        context_blocks.append(
            f"[Source Clause {idx} | Document: {source} | Page: {page} | Score: {score:.3f}]\n{text}"
        )

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are the official Sarkari Scheme Navigator assistant. Your mission is to provide accurate, truthful, and helpful information about Indian Government schemes based STRICTLY on the official source documents provided below.

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
    return prompt


def generate_llm_response(prompt: str) -> str:
    """
    Invokes the Google Gemini model with automatic fallback.
    """
    client = get_genai_client()
    last_err = None

    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to generate response across all models. Last error: {last_err}")


def answer(
    query: str,
    k: int = TOP_K,
    threshold: float = SIM_THRESHOLD
) -> Dict[str, Any]:
    """
    End-to-end RAG answer pipeline with retrieval confidence guardrails.

    Args:
        query: User question
        k: Number of chunks to retrieve
        threshold: Minimum similarity threshold (0.0 to 1.0). If confidence < threshold, abstains.

    Returns:
        Dict with answer, sources, confidence, and abstained status.
    """
    query = query.strip()
    if not query:
        return {
            "query": query,
            "answer": "Please ask a valid question.",
            "sources": [],
            "abstained": True,
            "confidence": 0.0,
            "retrieved_chunks": []
        }

    # Step 1: Retrieve candidate chunks
    retrieved_chunks = retrieve(query, k=k)

    # Step 2: Check confidence threshold
    confidence = max([c["score"] for c in retrieved_chunks]) if retrieved_chunks else 0.0

    if not retrieved_chunks or confidence < threshold:
        return {
            "query": query,
            "answer": (
                f"I cannot find sufficient relevant information in the government scheme documents "
                f"to answer this question reliably (retrieval confidence {confidence:.2f} is below "
                f"the confidence threshold of {threshold:.2f})."
            ),
            "sources": [],
            "abstained": True,
            "confidence": round(confidence, 4),
            "retrieved_chunks": retrieved_chunks
        }

    # Step 3: Format grounded prompt & generate answer
    prompt = build_prompt(query, retrieved_chunks)
    try:
        response_text = generate_llm_response(prompt)
    except Exception as e:
        response_text = f"Error generating answer from language model: {str(e)}"

    # Step 4: Extract deduplicated sources list
    seen_sources = set()
    sources = []
    for c in retrieved_chunks:
        key = (c["source"], c["page"])
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "source": c["source"],
                "page": c["page"],
                "score": c["score"],
                "excerpt": c["text"][:180] + "..." if len(c["text"]) > 180 else c["text"]
            })

    return {
        "query": query,
        "answer": response_text,
        "sources": sources,
        "abstained": False,
        "confidence": round(confidence, 4),
        "retrieved_chunks": retrieved_chunks
    }


if __name__ == "__main__":
    test_q = "How much prize money can the top 10 winners of the Arunachal Pradesh Entrepreneurship Challenge receive?"
    print(f"\nQuery: {test_q}")
    res = answer(test_q)
    print("\nAnswer:\n", res["answer"])
    print("\nSources:", res["sources"])
    print("\nConfidence:", res["confidence"], "| Abstained:", res["abstained"])

    print("\n" + "=" * 60)
    out_of_scope_q = "What is the capital of France?"
    print(f"Out-of-Scope Query: {out_of_scope_q}")
    res_out = answer(out_of_scope_q)
    print("\nAnswer:\n", res_out["answer"])
    print("\nConfidence:", res_out["confidence"], "| Abstained:", res_out["abstained"])