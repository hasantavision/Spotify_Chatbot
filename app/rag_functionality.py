import logging
import os
import shutil
from typing import Any, List, Optional, Tuple
from urllib.request import urlretrieve

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embeddings & Vector Store
# ---------------------------------------------------------------------------

# BAAI/bge-base-en-v1.5 is a top-ranked model on the MTEB benchmark.
# normalize_embeddings=True ensures unit vectors for cosine similarity.
# Must match the model used in data_processing.py.
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

vector_db = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings,
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Chat prompt for OpenAI (supports structured system/human turns)
_SYSTEM_PROMPT = """You are an expert analyst of Spotify app reviews from the Google Play Store. \
Your role is to help top management understand user feedback based solely on the provided review data.

Guidelines:
- Answer only based on the context provided below; do not make up information.
- If the context is insufficient, say so clearly.
- Provide structured, evidence-backed answers with key themes and patterns.
- Politely decline questions unrelated to Spotify reviews.
{history_section}
Context from reviews:
{context}"""

QA_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# Plain-text prompt for LlamaCpp (completion-style LLM)
_LLAMA_TEMPLATE = """\
You are analyzing Spotify app reviews from the Google Play Store.
Answer management questions based only on the reviews provided.
If you don't know the answer or the question is unrelated, say so politely.
{history_section}
Context:
{context}

Question: {question}
Answer:"""

QA_LLAMA_PROMPT = PromptTemplate(
    template=_LLAMA_TEMPLATE,
    input_variables=["context", "question", "history_section"],
)

# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

_llm_openai: Optional[BaseChatModel] = None
_openai_initialized: bool = False


def init_openai() -> Optional[BaseChatModel]:
    """Lazy-initialise ChatOpenAI. Returns None on failure."""
    global _llm_openai, _openai_initialized
    try:
        from langchain_openai import ChatOpenAI

        _llm_openai = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",  # cost-effective and more capable than gpt-4-1106-preview
        )
        _openai_initialized = True
        logger.info("OpenAI LLM initialised.")
        return _llm_openai
    except Exception as exc:
        logger.error("Failed to initialise OpenAI: %s", exc)
        _openai_initialized = False
        return None


# Download Llama 2 model if absent
_LLAMA_MODEL_PATH = "llms/llama-2-7b-chat.Q4_0.gguf"
if not os.path.isfile(_LLAMA_MODEL_PATH):
    logger.info("Downloading Llama 2 model…")
    _url = (
        "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF"
        "/resolve/main/llama-2-7b-chat.Q4_0.gguf?download=true"
    )
    _tmp = "llama-2-7b-chat.Q4_0.gguf"
    urlretrieve(_url, _tmp)
    shutil.move(_tmp, _LLAMA_MODEL_PATH)

llm_llama = LlamaCpp(
    model_path=_LLAMA_MODEL_PATH,
    n_gpu_layers=33,
    n_batch=512,
    temperature=0.0,
    top_p=1,
    n_ctx=6000,
    verbose=False,
)

# ---------------------------------------------------------------------------
# Retriever helpers
# ---------------------------------------------------------------------------


def _build_retriever(llm=None):
    """
    MMR (Maximal Marginal Relevance) retrieval balances relevance and diversity,
    reducing the redundancy that pure similarity search produces.

    fetch_k=20 gives MMR a wide candidate pool to select the final k=6 docs from.
    lambda_mult=0.7 biases slightly toward relevance over diversity.

    When an OpenAI LLM is supplied, MultiQueryRetriever wraps the base retriever:
    it generates several rephrased queries and merges the results, significantly
    improving recall for ambiguous or multi-faceted questions.
    """
    base = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
    )
    if llm is not None and isinstance(llm, BaseChatModel):
        return MultiQueryRetriever.from_llm(retriever=base, llm=llm)
    return base


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _format_history(chat_history: List[Tuple[str, str]]) -> str:
    if not chat_history:
        return ""
    lines = ["Previous conversation:"]
    for human, ai in chat_history[-3:]:  # cap at last 3 turns to stay within context
        lines.append(f"User: {human}\nAssistant: {ai}")
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Core RAG function
# ---------------------------------------------------------------------------


def rag_func(
    question: str,
    use_openai: bool,
    chat_history: Optional[List[Tuple[str, str]]] = None,
) -> dict[str, Any]:
    """
    Execute a RAG query.

    Args:
        question:     The user's question.
        use_openai:   Use ChatOpenAI when True, LlamaCpp when False.
        chat_history: Optional list of (user_msg, assistant_msg) pairs.

    Returns:
        {"answer": str, "context": List[Document]}
    """
    global _llm_openai

    if use_openai:
        if not _openai_initialized:
            _llm_openai = init_openai()
        llm = _llm_openai
    else:
        llm = llm_llama

    retriever = _build_retriever(llm if use_openai else None)
    docs = retriever.invoke(question)
    context_str = _format_docs(docs)
    history_section = _format_history(chat_history or [])

    logger.info("Retrieved %d document chunks.", len(docs))

    if use_openai and isinstance(llm, BaseChatModel):
        messages = QA_CHAT_PROMPT.format_messages(
            context=context_str,
            question=question,
            history_section=history_section,
        )
        response = llm.invoke(messages)
        answer = response.content
    else:
        prompt_str = QA_LLAMA_PROMPT.format(
            context=context_str,
            question=question,
            history_section=history_section,
        )
        answer = llm.invoke(prompt_str)

    return {"answer": answer, "context": docs}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_eval_metrics = [
    AnswerRelevancyMetric(minimum_score=0.5, model="gpt-4o-mini", include_reason=True),
    FaithfulnessMetric(minimum_score=0.5, model="gpt-4o-mini", include_reason=True),
]


def eval_func(
    question: str, answer: str, retrieval_context: List[str]
) -> List[Tuple[str, float, str]]:
    """
    Evaluate the RAG output with multiple DeepEval metrics.

    Returns:
        List of (metric_name, score, reason) tuples.
    """
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieval_context,
    )
    results = []
    for metric in _eval_metrics:
        try:
            metric.measure(test_case)
            results.append((type(metric).__name__, metric.score, metric.reason))
        except Exception as exc:
            logger.error("Metric %s failed: %s", type(metric).__name__, exc)
    return results
