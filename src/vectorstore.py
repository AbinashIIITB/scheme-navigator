"""
vectorstore.py — ChromaDB vector index: build it once, query it fast.

ChromaDB stores our text chunks as dense vector embeddings (numbers that
capture meaning). When a user asks a question, we embed the question the
same way and find the chunks whose vectors are closest — that's retrieval.
"""
from pathlib import Path
from typing import List, Dict, Any, Union

import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME, TOP_K


# ── Module-level singletons ───────────────────────────────────────────────────
# We create these once and reuse them across calls to avoid repeatedly loading
# the 80 MB embedding model or re-opening the database.
_embedder = None
_clients: Dict[str, chromadb.PersistentClient] = {}


def get_embedder():
    """
    Return the shared SentenceTransformer embedding function.

    First call downloads/loads the model (~80 MB); subsequent calls are instant
    because the result is cached in the module-level `_embedder` variable.
    """
    global _embedder
    if _embedder is None:
        _embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
    return _embedder


def get_chroma_client(persist_dir: Union[str, Path] = CHROMA_DIR) -> chromadb.PersistentClient:
    """
    Return a cached ChromaDB client for the given directory.

    A PersistentClient saves the index to disk so embeddings survive restarts —
    you only pay the indexing cost once.
    """
    path_str = str(Path(persist_dir).resolve())
    if path_str not in _clients:
        _clients[path_str] = chromadb.PersistentClient(path=path_str)
    return _clients[path_str]


def populate_db(
    chunks: List[Dict[str, Any]],
    persist_dir: Union[str, Path] = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    Build (or rebuild) the ChromaDB collection from a list of text chunks.

    Safe to call multiple times — it always drops and recreates the collection
    so there are no duplicate or stale embeddings.

    Args:
        chunks:          Output of chunk_documents() — list of chunk dicts.
        persist_dir:     Where to store the ChromaDB files on disk.
        collection_name: Name for the collection inside ChromaDB.
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = get_chroma_client(persist_dir)
    embed_fn = get_embedder()

    # Drop existing collection to avoid duplicate / stale data
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection didn't exist yet — that's fine

    # hnsw:space = "cosine" means similarity = 1 − cosine_distance
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if not chunks:
        print("No chunks provided — nothing to index.")
        return

    print(f"Indexing {len(chunks)} chunks into ChromaDB at {persist_path}...")

    # Add in batches of 100 to avoid memory spikes during embedding
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "page": int(c["page"])} for c in batch],
        )

    print(f"✓ Indexed {len(chunks)} chunks into collection '{collection_name}'.")


def retrieve(
    query: str,
    k: int = TOP_K,
    persist_dir: Union[str, Path] = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """
    Find the top-k most relevant chunks for a user query.

    How it works:
        1. Embed the query with the same model used at index time.
        2. ChromaDB computes cosine distance to every stored chunk vector.
        3. Return the k closest chunks with their similarity scores.

    Similarity score = 1 − cosine_distance, so:
        1.0 = perfect match,   0.0 = completely unrelated.

    Args:
        query:           The user's question (plain text).
        k:               How many chunks to return.
        persist_dir:     ChromaDB directory (must already exist).
        collection_name: Collection to query.

    Returns:
        List of chunk dicts sorted by score descending:
            {
                'id':       Chunk ID,
                'source':   PDF filename,
                'page':     Page number,
                'text':     Chunk text,
                'score':    Cosine similarity  (float in [0, 1]),
                'distance': Raw cosine distance (float in [0, 2]),
            }
    """
    client = get_chroma_client(persist_dir)
    embed_fn = get_embedder()

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embed_fn,
        )
    except Exception as e:
        print(f"Collection '{collection_name}' not found: {e}")
        return []

    # Don't request more results than we actually have
    n_results = min(k, collection.count()) if collection.count() > 0 else k

    results = collection.query(query_texts=[query], n_results=n_results)

    if not results or not results["documents"] or not results["documents"][0]:
        return []

    docs      = results["documents"][0]
    metas     = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
    ids       = results["ids"][0]       if results["ids"]       else [""] * len(docs)
    distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

    retrieved = []
    for doc_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
        # Cosine distance ∈ [0, 2]; convert to similarity ∈ [0, 1]
        sim_score = max(0.0, min(1.0, 1.0 - dist))
        retrieved.append({
            "id":       doc_id,
            "source":   meta.get("source", ""),
            "page":     meta.get("page", 1),
            "text":     doc_text,
            "score":    round(sim_score, 4),
            "distance": round(dist, 4),
        })

    # Highest similarity first
    retrieved.sort(key=lambda x: x["score"], reverse=True)
    return retrieved