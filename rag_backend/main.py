from fastapi import FastAPI, HTTPException
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.schema import Document
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
import requests
import os
from pydantic import BaseModel
import numpy as np
from langchain_core.documents import Document
import chromadb

from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())  # Reduce ChromaDB memory usage

# Add request model
class AddUrlRequest(BaseModel):
    url: str

from pdf_handling import *
#app = FastAPI()

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

EMBEDDING_MODEL = "/app/models/minilm"
LLM_MODEL = "/app/models/flan-t5-base"

CHROMA_DIR = "chroma_db"
DOCUMENT_DIR = "documents"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize ChromaDB
if not os.path.exists(CHROMA_DIR):
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    loader = DirectoryLoader(DOCUMENT_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    if not texts: 
        texts = ["your", "text", "data"] 
        texts = [Document(page_content=text) for text in texts]
    settings = chromadb.Settings(allow_reset=True)  # Removed chunk_size parameter
    vectordb = Chroma(embedding_function=embeddings, client_settings=settings)
    vectordb.persist()  # Updated from vectordb to vectorstore
else:
    vectordb = Chroma(persist_directory=None, embedding_function=embeddings)

# Initialize LLM
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3
    )
    llm = HuggingFacePipeline(pipeline=pipe)
except Exception as e:
    raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
    chain_type="stuff"
)

class QuestionRequest(BaseModel):
    question: str



def get_user_vectordb(user: User):
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=user.username  # Create per-user collections
    )

@app.post("/ask")
async def ask_question(
    request: QuestionRequest,
    user: Optional[User] = Depends(get_current_user)
):
    try:
        if user:
            # User-specific retriever
            user_db = Chroma(
                collection_name=user.username,
                embedding_function=embeddings
            )
            retriever = user_db.as_retriever()
        else:
            # Public retriever
            retriever = vectordb.as_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        result = qa_chain.run(request.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_url_test")
async def add_url_test(request: AddUrlRequest):
    try:
        url = request.url  # Get URL from request body
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)

        doc = Document(page_content=text, metadata={"source": url})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents([doc])

        vectordb.add_documents(split_docs)
        vectordb.persist()

        return {"message": f"Added {len(split_docs)} document chunks from {url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

