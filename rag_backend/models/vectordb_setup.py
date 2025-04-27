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

# reduce in-memory cache usage
set_llm_cache(InMemoryCache())

# ── CONFIGURATION ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "/app/models/minilm")
LLM_MODEL       = os.getenv("LLM_MODEL", "/app/models/flan-t5-base")
CHROMA_DIR      = os.getenv("CHROMA_DIR", "/tmp/chroma")      # local path for Chroma files
DOCUMENT_PATH   = os.getenv("DOCUMENT_PATH", "documents")    # local txt folder
GCS_BUCKET      = os.getenv("GCS_BUCKET_NAME", "")
IS_LOCAL        = os.getenv("IS_LOCAL", "false").lower() == "true"

# ── EMBEDDINGS ──────────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ── DOCUMENT LOADING & SPLITTING ───────────────────────────────────────────────
if IS_LOCAL:
    loader = DirectoryLoader(DOCUMENT_PATH, glob="**/*.txt", loader_cls=TextLoader)
else:
    loader = GCSDirectoryLoader(
        bucket_name=GCS_BUCKET,
        prefix="docs/"  # adjust if needed
    )

docs   = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)\
             .split_documents(docs)

# ensure at least one dummy chunk
if not chunks:
    chunks = [Document(page_content="seed document")]

# ── VECTOR STORE ────────────────────────────────────────────────────────────────
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)

# cold-start: only on first run
if not os.path.exists(os.path.join(CHROMA_DIR, "index")):
    vectordb.add_documents(chunks)
    vectordb.persist()

# ── LLM INITIALIZATION ─────────────────────────────────────────────────────────
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=False)
    model     = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    hf_pipe   = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "512")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3"))
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
except Exception as e:
    raise RuntimeError(f"Failed to initialize LLM: {e}")
