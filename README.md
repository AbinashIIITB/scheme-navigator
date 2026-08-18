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
* **Serving Layer**: High-performance asynchronous **FastAPI REST endpoints** (`/health`, `/ask`, `/ingest`) and an interactive **Streamlit chat application**.

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

Two evaluation suites are provided:

### Suite 1 — Retrieval Quality (`eval/questions.json`)
40 hand-curated questions covering eligibility, subsidies, loan caps, and terms across schemes:

| Metric | Measured Value |
|---|---|
| **Total Evaluation Questions** | **40** |
| **Hits @ 5** | **40 / 40** |
| **Recall @ 5** | **100.00%** |
| **MRR @ 5 (Mean Reciprocal Rank)** | **0.9875** |
| **Rank 1 Accuracy** | **39 / 40 (97.5%)** |

### Suite 2 — Abstain Guard (`eval/abstain_questions.json`)
10 deliberately out-of-scope questions (recipes, sports, science, etc.) that the system must **refuse to answer** instead of hallucinating:

| Metric | Expected Value |
|---|---|
| **Total Out-of-Scope Questions** | **10** |
| **Abstain Rate** | **100%** |
| **Hallucinated Answers** | **0** |

Run both suites together:
```bash
python eval/evaluate.py
```

---

## 📂 Project Structure

```
scheme-navigator/
├── data/
│   ├── raw/                       # 30 Government scheme PDF documents
│   └── chroma_db/                 # Persistent ChromaDB vector index
├── eval/
│   ├── questions.json             # 40 hand-written benchmark Q&A pairs (retrieval)
│   ├── abstain_questions.json     # 10 out-of-scope questions (abstain guard)
│   └── evaluate.py                # Recall@5, MRR & abstain rate evaluation suite
├── src/
│   ├── __init__.py
│   ├── config.py                  # Central configuration (models, paths, thresholds)
│   ├── ingest.py                  # PDF text extraction with pdfplumber
│   ├── chunker.py                 # Recursive text splitting & metadata tagging
│   ├── vectorstore.py             # ChromaDB indexing and similarity retriever
│   └── generator.py               # Grounded prompt builder, abstain guard & LLM caller
├── api.py                         # FastAPI REST API (/health, /ask, /ingest)
├── app.py                         # Streamlit web UI with source explorer
├── build_index.py                 # One-command data ingestion & vector DB builder
├── repl.py                        # Interactive CLI for testing without a UI
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites & Virtual Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/scheme-navigator.git
cd scheme-navigator

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your Google API Key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
LLM_MODEL=gemini-2.5-flash
```

> **Get a free API key**: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 3. Download the Dataset & Build the Vector Index

```bash
# Download 30 government scheme PDFs from HuggingFace Hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('shrijayan/gov_myscheme', repo_type='dataset',
                  local_dir='data/raw', allow_patterns='*.pdf')
"

# Build the ChromaDB vector index (runs once, ~30–60 seconds)
python build_index.py
```

### 4. Run the Evaluation Suite

```bash
python eval/evaluate.py
```

Expected output: **100% Recall@5, MRR = 0.9875**

### 5. Try the Interactive CLI (Fastest)

No browser needed — ask questions directly in the terminal:

```bash
python repl.py
```

### 6. Launch the FastAPI Server

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

### 7. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the chat interface, adjust retrieval hyperparameters dynamically, and explore grounded clause citations.
