import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import TOP_K, SIM_THRESHOLD, LLM_MODEL_NAME, COLLECTION_NAME, CHROMA_DIR, RAW_DIR
from src.generator import answer
from src.vectorstore import get_chroma_client, get_embedder, populate_db
from src.ingest import load_documents
from src.chunker import chunk_documents

app = FastAPI(
    title="Sarkari Scheme Navigator API",
    description="REST API for Grounded RAG over Indian Government Scheme documents with source citation and confidence thresholds.",
    version="1.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., description="Query about government schemes", example="What is the maximum loan amount under Mahila Samridhi Yojana?")
    k: Optional[int] = Field(default=TOP_K, ge=1, le=20, description="Number of context chunks to retrieve")
    threshold: Optional[float] = Field(default=SIM_THRESHOLD, ge=0.0, le=1.0, description="Minimum cosine similarity threshold before abstaining")


class SourceItem(BaseModel):
    source: str
    page: int
    score: float
    excerpt: Optional[str] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    abstained: bool
    confidence: float


class HealthResponse(BaseModel):
    status: str
    collection_name: str
    indexed_chunks: int
    model: str
    chroma_dir: str


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint verifying vector index status.
    """
    chunk_count = 0
    try:
        client = get_chroma_client()
        embed_fn = get_embedder()
        col = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
        chunk_count = col.count()
    except Exception:
        chunk_count = 0

    return HealthResponse(
        status="ok",
        collection_name=COLLECTION_NAME,
        indexed_chunks=chunk_count,
        model=LLM_MODEL_NAME,
        chroma_dir=str(CHROMA_DIR)
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(req: AskRequest):
    """
    Ask a question about government schemes.
    Retrieves grounded context, generates citation-backed answers, or abstains if confidence is below threshold.
    """
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer(
        query=q,
        k=req.k or TOP_K,
        threshold=req.threshold if req.threshold is not None else SIM_THRESHOLD
    )

    return AskResponse(
        query=result["query"],
        answer=result["answer"],
        sources=[
            SourceItem(
                source=s["source"],
                page=s["page"],
                score=s["score"],
                excerpt=s.get("excerpt")
            )
            for s in result.get("sources", [])
        ],
        abstained=result["abstained"],
        confidence=result["confidence"]
    )


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    raw_dir: str


@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
def ingest():
    """
    (Re)build the ChromaDB vector index from all PDFs in data/raw.
    Safe to call multiple times — drops and rebuilds the collection.
    """
    try:
        docs = load_documents(raw_dir=RAW_DIR)
        chunks = chunk_documents(docs)
        populate_db(chunks, persist_dir=CHROMA_DIR)
        return IngestResponse(
            status="ok",
            documents_loaded=len({d["source"] for d in docs}),
            chunks_created=len(chunks),
            raw_dir=str(RAW_DIR)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)