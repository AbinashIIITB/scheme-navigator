import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import chromadb
from chromadb.utils import embedding_functions

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    TOP_K,
)

# Module-level cache for embedding function and Chroma clients
_embedder = None
_clients: Dict[str, chromadb.PersistentClient] = {}


def get_embedder():
    """
    Returns the SentenceTransformer embedding function for all-MiniLM-L6-v2.
    """
    global _embedder
    if _embedder is None:
        _embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
    return _embedder


def get_chroma_client(persist_dir: Union[str, Path] = CHROMA_DIR) -> chromadb.PersistentClient:
    """
    Returns a cached PersistentClient for the given directory.
    """
    path_str = str(Path(persist_dir).resolve())
    if path_str not in _clients:
        _clients[path_str] = chromadb.PersistentClient(path=path_str)
    return _clients[path_str]


def populate_db(
    chunks: List[Dict[str, Any]],
    persist_dir: Union[str, Path] = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME
) -> None:
    """
    Populate or rebuild the Chroma vector database with chunk embeddings.
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = get_chroma_client(persist_dir)
    embed_fn = get_embedder()

    # Try deleting existing collection to avoid duplicate/stale embeddings
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    if not chunks:
        print("No chunks provided to populate_db.")
        return

    print(f"Indexing {len(chunks)} chunks into ChromaDB at {persist_path}...")

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [c["id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [{"source": c["source"], "page": int(c["page"])} for c in batch]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    print(f"Successfully indexed {len(chunks)} chunks into collection '{collection_name}'.")


def retrieve(
    query: str,
    k: int = TOP_K,
    persist_dir: Union[str, Path] = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant chunks for a given query with similarity scores.

    Returns:
        List of dicts: [
            {
                'id': ...,
                'source': ...,
                'page': ...,
                'text': ...,
                'score': float,      # Cosine similarity (1 - cosine_distance)
                'distance': float   # Raw cosine distance
            },
            ...
        ]
    """
    client = get_chroma_client(persist_dir)
    embed_fn = get_embedder()

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embed_fn
        )
    except Exception as e:
        print(f"Collection '{collection_name}' not found: {e}")
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count() if collection.count() > 0 else k)
    )

    retrieved: List[Dict[str, Any]] = []

    if not results or not results["documents"] or not results["documents"][0]:
        return retrieved

    docs = results["documents"][0]
    metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
    ids = results["ids"][0] if results["ids"] else [""] * len(docs)
    distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

    for doc_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
        # In cosine distance: distance is in [0, 2]. Similarity = 1 - distance.
        sim_score = max(0.0, min(1.0, 1.0 - dist))
        retrieved.append({
            "id": doc_id,
            "source": meta.get("source", ""),
            "page": meta.get("page", 1),
            "text": doc_text,
            "score": round(sim_score, 4),
            "distance": round(dist, 4)
        })

    # Sort descending by score
    retrieved.sort(key=lambda x: x["score"], reverse=True)
    return retrieved


if __name__ == "__main__":
    from src.ingest import load_documents
    from src.chunker import chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    populate_db(chunks)

    # Test retrieval
    test_query = "What is the maximum loan amount under Mahila Samridhi Yojana?"
    results = retrieve(test_query, k=5)
    print(f"\nRetrieval test for: '{test_query}'")
    for r in results:
        print(f"- [Score: {r['score']:.4f}] Source: {r['source']} (Page {r['page']})")