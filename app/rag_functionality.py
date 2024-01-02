import os.path
import shutil
from typing import List, Any

from deepeval.test_case import LLMTestCase
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores.chroma import Chroma
from urllib.request import urlretrieve
from langchain_core.prompts import PromptTemplate
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric
from deepeval.models.gpt_model import ChatOpenAI

metric = AnswerRelevancyMetric(minimum_score=0.5,
                               model="gpt-4-1106-preview",
                               include_reason=True
                               )

embeddings = HuggingFaceEmbeddings()

vector_db = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)

template = """You are analyzing spotify app reviews from google play store to be used to answer top management's questions based on the reviews' summary.
If you don't know the answer or the question is not related to this task, answer it politely, don't try to make up an answer. 
answer only the points asked.
    {context}
    Question: {question}
    Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

openai_initialized = False
llm_openai = None


def init_openai():
    global openai_initialized
    try:
        openai = ChatOpenAI(
            temperature=0,
            model="gpt-4-1106-preview"
        )
        openai_initialized = True
        return openai
    except Exception:
        openai_initialized = False


model_path = "llms/llama-2-7b-chat.Q4_0.gguf"
if not os.path.isfile(model_path):
    url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf?download=true"
    filename = "llama-2-7b-chat.Q4_0.gguf"
    urlretrieve(url, filename)
    shutil.move(filename, model_path)

llm_llama = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=33,
    n_batch=512,
    temperature=0.0,
    top_p=1,
    n_ctx=6000
)


def load_chain(llm):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_db.as_retriever(
            search_kwargs={"k": 10}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain


def rag_func(question: str, use_openai: bool) -> dict[str, Any]:
    global llm_openai
    if use_openai:
        if not openai_initialized:
            llm_openai = init_openai()
        qa_chain = load_chain(llm_openai)
    else:
        qa_chain = load_chain(llm_llama)
    result = qa_chain({"query": question})
    print([doc.page_content for doc in result['source_documents']])
    return result


def eval_func(question: str, result, retrieval_context: List[str]):
    test_case = LLMTestCase(
        input=question,
        actual_output=result,
        retrieval_context=retrieval_context
    )
    metric.measure(test_case)
    return metric.score, metric.reason
