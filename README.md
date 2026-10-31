# RawRAG 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/Model-Gemini%202.5%20Flash-orange)](#)

**RawRAG** is a production-grade Retrieval-Augmented Generation (RAG) pipeline built from scratch under the **[RawCompute]** organization. 

**Our Philosophy:** Zero bloated orchestration wrappers (No LangChain, No LlamaIndex) and zero managed vector databases (No Pinecone, No ChromaDB). We build the infrastructure from first principles using pure Python, linear algebra via NumPy, and standard SQLite.

## Architecture Overview

RawRAG is designed to process massive PDFs, chunk them intelligently, embed them into high-dimensional space, and retrieve exact context via manual cosine similarity calculations.

1. **Ingestion & Chunking:** Extracts text from PDFs using `PyPDF2` and slices it using a custom sliding-window algorithm to preserve semantic boundaries.
2. **Embeddings:** Pings the `gemini-embedding-001` endpoint via the modern `google-genai` SDK to convert text chunks into 3072-dimensional floating-point vectors.
3. **Vector Mathematics:** Maintains a global NumPy matrix (`np.vstack`) and executes manual dot-product cosine similarity searches (`np.argsort`) to find the top-k matches instantly.
4. **LLM Synthesis:** Injects retrieved chunk text into a strict prompt template and generates grounded answers using `gemini-2.5-flash`.

## Repository Structure

```text
RawRAG/
├── core/
│   ├── chunker.py         # PDF extraction and sliding-window logic
│   ├── gemini_client.py   # Model endpoints (google-genai SDK)
│   └── vector_math.py     # NumPy matrix and cosine similarity engine
├── db/
│   └── database.py        # SQLite mapping (Vector Index <-> Text Chunk)
├── .env                   # Ignored via .gitignore 
├── pyproject.toml         # Dependency mapping for uv
└── main.py                # FastAPI server and endpoints [WIP]
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RawCompute/RawRAG.git](https://github.com/RawCompute/RawRAG.git)
   cd RawRAG
   ```

2. **Configure the environment (using uv):**
   ```bash
   # Initialize a fast uv virtual environment
   uv venv

   # Activate the environment
   # Linux/macOS:
   source .venv/bin/activate
   # Windows:
   # .venv\Scripts\activate

   # Sync dependencies directly
   uv sync
   
   # Note: If you are strictly using the legacy requirements file instead of pyproject.toml, run:
   # uv pip install -r requirements.txt
   ```

3. **Configure the API Key:**
   Create a `.env` file in the root directory and add your Google AI Studio key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
   *Note: `.env` is included in `.gitignore` to prevent credential leaks.*

## Development Roadmap

- [x] **Sliding-Window Chunker:** Completed and tested on complex PDFs.
- [x] **Gemini API Integration:** Updated to the latest `google-genai` SDK, configured for 3072-dimensional vectors.
- [x] **Vector Math Engine:** Custom NumPy `VectorDatabase` class built.
- [x] **SQLite Database (`db/database.py`):** Map NumPy row indices to raw text strings.
- [ ] **FastAPI Endpoints (`main.py`):** Hook up `/ingest` and `/query` routes.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
