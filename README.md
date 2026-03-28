# RAG Chatbot — Spotify Play Store Reviews

[![RAG chatbot with OpenAI](https://github.com/hasantavision/Spotify_Chatbot/blob/master/assets/yt1.png)](https://www.youtube.com/watch?v=fL4f9hgiwPw "RAG chatbot with OpenAI")
[![RAG chatbot with Llama2](https://github.com/hasantavision/Spotify_Chatbot/blob/master/assets/yt2.png)](https://www.youtube.com/watch?v=D-mgXB96r1o "RAG chatbot with Llama2")

A Streamlit chatbot that lets management query Spotify app reviews using RAG (Retrieval-Augmented Generation). Supports both OpenAI (GPT-4o mini) and a local Llama 2 model.

## Architecture

```
User question
    │
    ├─ OpenAI path ──► SelfQueryRetriever (extracts date/rating filters)
    │                        + BM25 keyword search
    │                  ──► EnsembleRetriever (RRF fusion)
    │
    └─ Llama 2 path ─► MMR vector search (diverse, low-redundancy)
                             + BM25 keyword search
                       ──► EnsembleRetriever (RRF fusion)
                           │
                    Context + conversation history
                           │
                         LLM answer
                           │
              (optional) DeepEval metrics
```

### Key RAG techniques used

| Technique | Purpose |
|---|---|
| **BAAI/bge-base-en-v1.5** embeddings | Top MTEB benchmark; `normalize_embeddings=True` for cosine similarity |
| **Review-aware chunking** | Each review is one atomic chunk — never split mid-sentence |
| **Hybrid search** (BM25 + vector) | BM25 catches exact keywords; vector catches paraphrases; fused via RRF |
| **MMR retrieval** | Reduces redundant documents in the retrieved context |
| **SelfQueryRetriever** (OpenAI) | LLM auto-extracts structured filters from natural language: "1-star reviews from 2023" |
| **Conversation history** | Last 3 Q&A turns injected into every prompt for follow-up question support |
| **DeepEval** (AnswerRelevancy + Faithfulness) | Scores whether the answer is on-topic and grounded in retrieved context |

**Time / rating-aware query examples** (OpenAI only):
- "What were users complaining about in 2023?"
- "Summarise 1-star reviews about audio quality"
- "What improved between version 8.6 and 8.8?"

## Setup

### Requirements

- Python 3.10 – 3.11
- CUDA environment (for GPU-accelerated Llama 2; CPU-only also works, but slower)

### Install

```bash
git clone https://github.com/hasantavision/Spotify_Chatbot.git
cd Spotify_Chatbot
pip install -r requirements.txt
```

**Llama 2 with CUDA** (optional — only needed if you want GPU acceleration):
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall
```

### Data preparation

```bash
cd data
python data_processing.py
```

This will:
1. Download `SPOTIFY_REVIEWS.csv` from Google Drive
2. Build one `Document` per review with structured metadata (`rating`, `date`, `app_version`)
3. Embed using `BAAI/bge-base-en-v1.5` and persist to `data/chroma_db/`

Runtime depends on dataset size and hardware (CPU: ~1–2 hours; GPU: much faster).

### Llama 2 model download (optional)

Pre-download the model to avoid a delay on first app start:

```bash
cd llms
python get_llama.py
```

The model (`llama-2-7b-chat.Q4_0.gguf`, ~4 GB) is downloaded from HuggingFace automatically if absent when the app starts.

### Run

```bash
cd app
streamlit run app.py
```

## Usage

| Sidebar option | Description |
|---|---|
| **OpenAI API Key** | Required for OpenAI model and evaluation metrics |
| **Use OpenAI** | Toggle between GPT-4o mini and local Llama 2 |
| **Use Eval metrics** | Score each answer with AnswerRelevancy + Faithfulness (requires OpenAI key) |

Without an OpenAI key the app runs fully offline using Llama 2.
