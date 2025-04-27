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
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

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
    loader = GCSDirectoryLoader(bucket_name=GCS_BUCKET, prefix="docs/")

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
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Create the text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_length=512,  # Adjust as needed
        temperature=temp,
    )

    # Wrap the pipeline in HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=generator)
    return llm, tok


llm, tokenizer        = _load_llm(LLM_MODEL, float(os.getenv("LLM_TEMPERATURE", "0.3")))
shadow_llm, tokenizer = _load_llm(SHADOW_LLM_MODEL, float(os.getenv("SHADOW_LLM_TEMPERATURE", "0.7")))

# ── PROMPT TEMPLATE ─────────────────────────────────────────────────────────────
template = """<|system|>
You are a helpful assistant that answers questions based on the provided context.
If the context does not contain the answer, truthfully and concisely say "The answer is not available in the provided context."
You should only use information from the context and answer in your own words. Do not repeat the question verbatim.
<|user|>
Context: {context}
Question: {question}
<|assistant|>"""


PROMPT_TEMPLATE = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
