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
from langchain.vectorstores import PGVector
import gcsfs
from langchain_community.document_loaders import DirectoryLoader, GCSDirectoryLoader, TextLoader
from pdf_handling import *
from metrics.check_metrics import *
from models.vectordb_setup import *
import time
import datetime

import random
import uuid

app.include_router(monitoring_router, prefix="/monitoring", tags=["monitoring"])

from urllib.parse import urlparse
import httpx  # Better async alternative to requests
 

set_llm_cache(InMemoryCache())  # Reduce ChromaDB memory usage

# Add request model
class AddUrlRequest(BaseModel):
    url: str

#app = FastAPI()

# Configuration
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_MODEL = "google/flan-t5-base"



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
        # Choose retriever
        if user:
            user_db   = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
                collection_name=user.username
            )
            retriever = user_db.as_retriever()
        else:
            retriever = vectordb.as_retriever()

        # A/B routing
        start = time.time()
        if random.random() < 0.2:
            chosen_llm, model_type = shadow_llm, "shadow"
        else:
            chosen_llm, model_type = llm,        "production"

        qa = RetrievalQA.from_chain_type(
            llm=chosen_llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT_TEMPLATE}
        )
        answer = qa.run(request.question)
        latency = time.time() - start

        # *** Log into metrics collection ***
        metrics_collection.add(
            documents=[request.question],
            metadatas=[{
                "model": model_type,
                "response_time": latency,
                "timestamp": time.time(),
                "user": user.username if user else 'guest'
            }],
            ids=[str(uuid.uuid4())]
        )
        #metrics_collection.persist()

        # Save conversation memory
        if user:
            convo_doc = Document(
                page_content=f"Q: {request.question}\nA: {answer}",
                metadata={"source": "conversation"}
            )
            retriever.add_documents([convo_doc])
            vectordb.persist()

        return {"answer": answer}

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

