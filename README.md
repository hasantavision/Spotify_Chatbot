# RAG Chatbot Streamlit App with Spotify Data

[![RAG chatbot with OpenAI](https://github.com/hasantavision/Spotify_Chatbot/blob/master/assets/yt1.png)](https://www.youtube.com/watch?v=fL4f9hgiwPw "RAG chatbot with OpenAI")
[![RAG chatbot with Llama2](https://github.com/hasantavision/Spotify_Chatbot/blob/master/assets/yt2.png)](https://www.youtube.com/watch?v=D-mgXB96r1o "RAG chatbot with Llama2")

A Retrieval-Augmented Generation (RAG) chatbot that lets you query Spotify app reviews from the Google Play Store using OpenAI (GPT-4o-mini) or a local Llama 2 model.

---

## Architecture

```
User Question (Streamlit UI)
        │
        ▼
┌──────────────────────────────────────┐
│  Stage 1 — Retrieval (Bi-Encoder)    │
│  BAAI/bge-base-en-v1.5 + MMR        │
│  k=15 candidates, fetch_k=30        │
└──────────────┬───────────────────────┘
               │  (OpenAI only)
               ▼
┌──────────────────────────────────────┐
│  MultiQueryRetriever                 │
│  LLM rephrases query → merge results │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Stage 2 — Re-ranking (Cross-Encoder)│
│  cross-encoder/ms-marco-MiniLM-L-6-v2│
│  Scores all 15 pairs → top 6        │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  LLM Generation                      │
│  OpenAI GPT-4o-mini  or  Llama 2    │
└──────────────┬───────────────────────┘
               │
               ▼
        Answer + DeepEval Metrics
```

### Why two stages?

| Stage | Model type | Speed | Accuracy | Role |
|---|---|---|---|---|
| Bi-encoder (BAAI/bge) | Independent embeddings | Fast — precomputed | Moderate | Wide recall from millions of docs |
| Cross-encoder (MiniLM) | Joint query+doc encoding | Slower — computed at query time | High | Precise relevance scoring of the candidate pool |

The bi-encoder casts a wide net quickly; the cross-encoder then re-scores every `(query, candidate)` pair jointly, surfacing the truly relevant documents before they reach the LLM. This pattern consistently improves NDCG@10 by 30–48% over single-stage retrieval.

---

## Re-ranking: current model and alternatives

The current implementation uses **`cross-encoder/ms-marco-MiniLM-L-6-v2`** — a lightweight English-only model that runs on CPU and is ideal for prototyping.

### Upgrade options (2025–2026)

| Model | License | Multilingual | Notes |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Open | No | Fast, CPU-friendly, good baseline |
| `BAAI/bge-reranker-v2-m3` *(current)* | Apache 2.0 | Yes (100+) | State-of-the-art open-source; matches Cohere on GPU |
| `Cohere Rerank 4 Pro` | Proprietary API | Yes (100+) | Highest ELO (1627); best for production/finance |
| `ZeroEntropy zerank-2` | Non-commercial | Yes (100+) | Calibrated scores; no threshold tuning needed |
| `Jina Reranker v2` | Apache 2.0 | Yes | Strong multilingual + agentic RAG support |

**To revert to the lighter CPU-friendly model**, change one line in `app/rag_functionality.py`:

```python
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

---

## How to use this repo

### Clone the repo
```bash
git clone https://github.com/hasantavision/Spotify_Chatbot.git
```

### Install requirements

Make sure you have Python 3.9–3.11 and a CUDA-ready environment.

**PyTorch installation**

Refer to the [PyTorch official installation guide](https://pytorch.org/).
```bash
pip install torch torchvision torchaudio
```

**Install from requirements.txt**
```bash
cd Spotify_Chatbot
pip install -r requirements.txt
```

**llama-cpp-python installation**

Required only if you want to run the local Llama 2 model. Needs CUDA.
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

### Data preparation
```bash
cd data
python data_processing.py
```
Downloads `SPOTIFY_REVIEWS.csv`, removes unused columns, saves `SPOTIFY_REVIEWS_CLEANED.csv`, chunks reviews, embeds them with `BAAI/bge-base-en-v1.5`, and persists to ChromaDB. This can take a while depending on your hardware.

### Llama 2 model download
To avoid long startup times, download the model before running the app:
```bash
cd llms
python get_llama.py
```

### Run the app
```bash
cd app
streamlit run app.py
```

---

## Key components

| File | Purpose |
|---|---|
| `app/app.py` | Streamlit UI, session state, evaluation display |
| `app/rag_functionality.py` | Retrieval, re-ranking, prompt formatting, LLM invocation, DeepEval metrics |
| `data/data_processing.py` | CSV loading, chunking, embedding, ChromaDB indexing |
| `llms/get_llama.py` | Downloads Llama 2 GGUF model from HuggingFace |

---

## Sources

- [Ultimate Guide to Choosing the Best Reranking Model (ZeroEntropy)](https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025)
- [Best Reranker Models for RAG: Open-Source vs API Comparison 2026](https://docs.bswen.com/blog/2026-02-25-best-reranker-models/)
- [Enhancing RAG Pipelines with Re-Ranking (NVIDIA Technical Blog)](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/)
- [Rerankers and Two-Stage Retrieval (Pinecone)](https://www.pinecone.io/learn/series/rag/rerankers/)
- [Top 7 Rerankers for RAG (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2025/06/top-rerankers-for-rag/)
- [The Evolution of Reranking Models in Information Retrieval (arXiv)](https://arxiv.org/pdf/2512.16236)
