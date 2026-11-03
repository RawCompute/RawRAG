# RawRAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/Model-Gemini%20Flash-orange)](#)

**RawRAG** is a production-grade Retrieval-Augmented Generation (RAG) pipeline built entirely from scratch under the **RawCompute** organization. 

**Design Philosophy:** Zero bloated orchestration wrappers (no LangChain, no LlamaIndex) and zero managed vector databases (no Pinecone, no ChromaDB). The infrastructure is built from first principles using pure Python, linear algebra via NumPy, and standard SQLite.

## Technical Architecture

RawRAG is designed to process complex documents, embed them into high-dimensional space, and retrieve context via manual mathematical operations.

1. **Ingestion & Chunking:** 
   Extracts text from PDFs using `PyPDF2`, applying automated local OCR for scanned pages. Text is processed through a custom sliding-window algorithm to preserve semantic boundaries and maintain context overlap between chunks.
2. **Embedding Pipeline:** 
   Utilizes the `google-genai` SDK to interface with `gemini-embedding-001`. The pipeline converts text chunks into 3072-dimensional floating-point vectors, featuring custom exponential backoff and timeout handlers to bypass network latency and rate limits during batch processing.
3. **Vector Mathematics & Storage:** 
   Maintains a global NumPy matrix (`np.vstack`) in memory. Document retrieval is executed via manual dot-product cosine similarity searches (`np.argsort`). Standard SQLite (`db/database.py`) maps the NumPy row indices to the raw text strings and handles document deduplication via SHA-256 hashing.
4. **LLM Synthesis:** 
   Injects the top-k retrieved chunks into a strict prompt template, generating grounded responses via low-latency Gemini Flash models.
5. **Client Interfaces:** 
   Exposes a high-performance asynchronous REST API via FastAPI (`/ingest`, `/query`) and provides a decoupled, interactive web UI using Streamlit.

## Repository Structure

```text
RawRAG/
├── core/
│   ├── chunker.py         # PDF extraction and sliding-window logic
│   ├── gemini_client.py   # Model endpoints, batching, and retry logic
│   └── vector_math.py     # NumPy matrix operations and similarity engine
├── db/
│   └── database.py        # SQLite schema mapping (Vector Index <-> Text Chunk)
├── .env                   # Environment variables (API Keys)
├── pyproject.toml         # Dependency configuration for uv
├── main.py                # FastAPI application and REST endpoints
└── app.py                 # Streamlit frontend UI
```

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/RawCompute/RawRAG.git
cd RawRAG
```

2. **Configure the environment (using uv):**
```bash
# Initialize a fast uv virtual environment
uv venv

# Activate the environment
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\activate

# Sync dependencies directly
uv sync
```

3. **Configure the API Key:**
Create a `.env` file in the root directory and add your Google AI Studio key:
```env
GEMINI_API_KEY=your_api_key_here
```
*Note: `.env` is explicitly ignored via `.gitignore` to prevent credential leaks.*

## Running the Application

RawRAG operates with a decoupled backend and frontend. You will need two concurrent terminal sessions.

**1. Start the FastAPI Backend:**
```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload --timeout-keep-alive 5
```
*API documentation and manual testing available at `http://127.0.0.1:8000/docs`.*

**2. Start the Streamlit Frontend:**
Open a new terminal window, ensure the virtual environment is active, and launch the UI:
```bash
uv run streamlit run app.py
```
*The interactive web interface will launch at `http://localhost:8501`.*

## License

This project is licensed under the MIT License. See the LICENSE file for details.