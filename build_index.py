"""
Build / Rebuild the ChromaDB Vector Index from PDF documents in data/raw.
"""
import time
from src.ingest import load_documents
from src.chunker import chunk_documents
from src.vectorstore import populate_db
from src.config import RAW_DIR, CHROMA_DIR


def main():
    print("=" * 60)
    print("Sarkari Scheme Navigator — Vector Index Builder")
    print("=" * 60)

    start_time = time.time()

    print(f"\n1. Ingesting PDFs from: {RAW_DIR}")
    docs = load_documents(raw_dir=RAW_DIR)
    print(f"   -> Loaded {len(docs)} pages.")

    print(f"\n2. Chunking documents (500 chars, 50 overlap)...")
    chunks = chunk_documents(docs)
    print(f"   -> Created {len(chunks)} text chunks.")

    print(f"\n3. Generating embeddings & populating ChromaDB at {CHROMA_DIR}...")
    populate_db(chunks, persist_dir=CHROMA_DIR)

    elapsed = time.time() - start_time
    print(f"\n✓ Index build complete in {elapsed:.2f} seconds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
