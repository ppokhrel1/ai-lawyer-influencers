# vectordb_setup.py
import os

from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import GCSDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, BitsAndBytesConfig, AutoModelForCausalLM

# reduce in-memory cache usage
set_llm_cache(InMemoryCache())

# ── CONFIGURATION ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "/app/models/minilm")
LLM_MODEL       = os.getenv("LLM_MODEL", "/app/models/flan-t5-base")
SHADOW_LLM_MODEL = os.getenv("LLM_MODEL", "/app/models/flan-t5-base")


CHROMA_DB_USER = os.getenv("CHROMA_DB_USER", "chroma_master")
CHROMA_DB_PASS = os.getenv("CHROMA_DB_PASS", "")
CHROMA_DB_CONN = os.getenv("CHROMA_DB_CONN", "")
CHROMA_DB_NAME = os.getenv("CHROMA_DB_NAME", "chromadb_db")


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "/app/models/minilm")
LLM_MODEL      = os.getenv("LLM_MODEL", "/app/models/flan-t5-base")
CHROMA_DIR      = os.getenv("CHROMA_DIR", "chroma_db")

# ── CONFIG ──────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "/app/models/minilm")
CHROMA_DIR         = os.getenv("CHROMA_DIR", "/tmp/chroma")    # e.g. /tmp on Cloud Run
DOCUMENT_PATH      = os.getenv("DOCUMENT_PATH", "documents")

GCS_BUCKET      = os.getenv("GCS_BUCKET_NAME", "ai-lawyers-influencers-vector-store")
GCS_DOCUMENTS   = f"{GCS_BUCKET}/docs/"


IS_LOCAL        = os.getenv("IS_LOCAL", "false").lower() == "true"
SYSTEM_PROMPT         = os.getenv("SYSTEM_PROMPT", "You are a helpful legal assistant.")


# ── EMBEDDINGS ──────────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ── LOAD & SPLIT ─────────────────────────────────────────────────────────────────
if IS_LOCAL:
    loader = DirectoryLoader(DOCUMENT_PATH, glob="**/*.txt", loader_cls=TextLoader)
else:
    loader = GCSDirectoryLoader('ai-lawyers-influencers', GCS_BUCKET)

docs   = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)\
             .split_documents(docs or [Document(page_content="seed document")])

# ── VECTORSTORE ─────────────────────────────────────────────────────────────────
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)
# cold-start: only first run
if not os.path.exists(os.path.join(CHROMA_DIR, "index")):
    vectordb.add_documents(chunks)
    vectordb.persist()

# ── LLM LOADING ─────────────────────────────────────────────────────────────────

def _load_llm(model_name: str, temp: float) -> HuggingFacePipeline:
    # Load the tokenizer and model for text generation
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map=None,  # <- no auto device map
        low_cpu_mem_usage=False,  # <- optional, just to be explicit
        torch_dtype="auto"  # <- will use float32
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=temp,
        do_sample=True
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm, tokenizer


llm, tokenizer        = _load_llm(LLM_MODEL, float(os.getenv("LLM_TEMPERATURE", "0.3")))
shadow_llm, tokenizer = _load_llm(SHADOW_LLM_MODEL, float(os.getenv("SHADOW_LLM_TEMPERATURE", "0.7")))

# ── PROMPT TEMPLATE ─────────────────────────────────────────────────────────────
template = """<|system|>
You are a helpful assistant that answers questions based on the provided context.


Use the provided context to answer the question as accurately as possible.
If the context does not contain the answer, use your own knowledge to generate the best possible response.

<|user|>
Context: {context}
Question: {question}

<|assistant|>"""



PROMPT_TEMPLATE = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)


