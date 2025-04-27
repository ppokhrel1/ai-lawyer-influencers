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
from auth_handling import *

from urllib.parse import urlparse
import httpx  # Better async alternative to requests
 
# main.py
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from bs4 import BeautifulSoup
import requests

from models.vectordb_setup import vectordb, llm, embeddings  # our setup file
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


app = FastAPI()

# ── REQUEST MODELS ─────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class AddUrlRequest(BaseModel):
    url: str

# ── /ask ENDPOINT ───────────────────────────────────────────────────────────────
@app.post("/ask")
async def ask_question(
    request: QuestionRequest,
    user: Optional[User] = Depends(get_current_user)
):
    try:
        # pick public vs. user‐private retriever
        if user:
            user_vectordb = vectordb.with_collection(user.username)
            retriever = user_vectordb.as_retriever()
        else:
            retriever = vectordb.as_retriever()

        # build a fresh chain per request
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        answer = qa_chain.run(request.question)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── /add_url_test ENDPOINT ─────────────────────────────────────────────────────
@app.post("/add_url_test")
async def add_url_test(request: AddUrlRequest):
    try:
        url = request.url
        resp = requests.get(url)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        doc = Document(page_content=text, metadata={"source": url})
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents([doc])

        vectordb.add_documents(split_docs)
        vectordb.persist()

        return {"message": f"Added {len(split_docs)} chunks from {url}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

