# 🏛️ Sarkari Scheme Navigator

> **Production-grade Retrieval-Augmented Generation (RAG) System over Indian Government Scheme Guidelines with Clause-level Grounding, Anti-Hallucination Thresholding, FastAPI REST API, and Streamlit Chat Interface.**

---

## 📌 Project Overview & Highlights

* **Corpus**: Indexed **30 official Indian government scheme policy documents** (PDF format).
* **Chunking Strategy**: Chunked at **500 characters (~100 tokens) with 50-character sliding overlap**, preserving page-level and document-level metadata.
* **Embeddings & Vector Store**: Dense semantic indexing using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings) in persistent **ChromaDB**.
* **Evaluation Benchmark**: Evaluated on a **40-question hand-written evaluation dataset**, achieving **100% Recall@5** (`1.0000`) and **0.9875 MRR@5**.
* **Source Clause Grounding**: Every answer is strictly grounded in cited clauses referencing `[Source: <filename>, Page <page>]`.
* **Abstain Guardrails**: Built-in similarity confidence thresholding ($S < 0.35$) that abstains rather than hallucinating when queries are out-of-scope or unverified.
* **Serving Layer**: High-performance asynchronous **FastAPI REST endpoints** (`/health`, `/ask`) and an interactive **Streamlit chat application**.

---

## 🏗️ Architecture

```
                               ┌─────────────────────────────┐
                               │  30 Government Scheme PDFs   │
                               └──────────────┬──────────────┘
                                              │ (pdfplumber)
                                              ▼
                               ┌─────────────────────────────┐
                               │ Page Extraction (46 Pages)  │
                               └──────────────┬──────────────┘
                                              │ (RecursiveTextSplitter)
                                              ▼
                               ┌─────────────────────────────┐
                               │   793 Semantic Chunks       │
                               └──────────────┬──────────────┘
                                              │ (all-MiniLM-L6-v2)
                                              ▼
                               ┌─────────────────────────────┐
                               │ ChromaDB Vector Database    │
                               │ (Persistent Cosine Index)   │
                               └──────────────┬──────────────┘
                                              │
              User Query                      │ Similarity Search (Top-k)
                 │                            ▼
                 ├────────────────► Candidate Chunks (k=5)
                 │                            │
                 │                    Confidence Check
                 │                     (Score < 0.35?)
                 │                     /            \
                 │             YES (Abstain)       NO (Grounded Prompt)
                 │                   │                      │
                 │         "Confidence low..."              ▼
                 │                                  Google Gemini LLM
                 │                                          │
                 ▼                                          ▼
     ┌───────────────────────┐                    ┌───────────────────┐
     │  Streamlit / FastAPI  │ ◄──────────────────┤ Grounded Response │
     │  Interactive Client   │                    │ + Source Citation │
     └───────────────────────┘                    └───────────────────┘
```

---

## 📊 Evaluation Results

Evaluated against `eval/questions.json` (40 hand-curated questions covering eligibility, subsidies, loan caps, and terms across schemes):

| Metric | Measured Value |
|---|---|
| **Total Evaluation Questions** | **40** |
| **Hits @ 5** | **40 / 40** |
| **Recall @ 5** | **100.00%** |
| **MRR @ 5 (Mean Reciprocal Rank)** | **0.9875** |
| **Rank 1 Accuracy** | **39 / 40 (97.5%)** |

Run evaluation at any time:
```bash
python eval/evaluate.py
```

---

## 📂 Project Structure

```
scheme-navigator/
├── data/
│   ├── raw/                 # 30 Government scheme PDF documents
│   └── chroma_db/           # Persistent ChromaDB vector index
├── eval/
│   ├── questions.json       # 40 hand-written benchmark Q&A pairs
│   └── evaluate.py          # Recall@5 and MRR evaluation suite
├── src/
│   ├── __init__.py
│   ├── config.py            # Central configurations, model names, thresholds
│   ├── ingest.py            # PDF text extraction with pdfplumber
│   ├── chunker.py           # Recursive text splitting & metadata tagging
│   ├── vectorstore.py       # ChromaDB indexing and similarity retriever
│   └── generator.py         # Grounded prompt builder, abstain guard & LLM caller
├── api.py                   # FastAPI REST API (/health, /ask)
├── app.py                   # Streamlit web UI with source explorer
├── build_index.py           # One-click data ingestion & vector DB builder
├── repl.py                  # CLI test script
├── requirements.txt         # Pinned python dependencies
└── README.md                # Documentation
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites & Virtual Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/scheme-navigator.git
cd scheme-navigator

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
LLM_MODEL=gemini-3-flash-preview
```

### 3. Build the Vector Index

Ingest all 30 PDF documents, split them into chunks, and index embeddings:

```bash
python build_index.py
```

### 4. Run Evaluation

```bash
python eval/evaluate.py
```

### 5. Launch the FastAPI Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

* **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Endpoint**: `GET http://localhost:8000/health`
* **Ask Endpoint**: `POST http://localhost:8000/ask`

#### Example API Request:
```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is the maximum loan amount provided to women entrepreneurs under Mahila Samridhi Yojana?"
     }'
```

### 6. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the chat interface, adjust retrieval hyperparameters dynamically, and explore grounded clause citations.
