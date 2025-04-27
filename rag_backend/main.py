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
import chromadb

from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.vectorstores import PGVector
import gcsfs
from langchain_community.document_loaders import DirectoryLoader, GCSDirectoryLoader, TextLoader
from pdf_handling import *
from metrics.check_metrics import *
from models.vectordb_setup import *
from metrics.feedback import feedback_router  # Import the new router
from metrics.drift_analysis import drift_router
from helpers.helpers import *

import time
import random
import uuid

previous_answers = []
previous_tokens = []

metrics_client = vectordb._client
metrics_collection = metrics_client.get_or_create_collection("metrics")


app.include_router(monitoring_router, prefix="/monitoring", tags=["monitoring"])
app.include_router(feedback_router, prefix="/feedback", tags=["feedback"]) # Include the feedback router
app.include_router(drift_router, prefix="/drift", tags=["drift_analysis"])

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
            user_db = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
                collection_name=user.username
            )
            retriever = user_db.as_retriever()
        else:
            retriever = vectordb.as_retriever()


        # Fetch relevant docs
        docs = retriever.get_relevant_documents(request.question)

        # Combine the contents of the documents into a single context string
        context = "\n\n".join([d.page_content for d in docs])



        # A/B routing logic (choose between production and shadow LLM)
        start = time.time()
        if random.random() < 0.2:
            chosen_llm, model_type = shadow_llm, "shadow"
        else:
            chosen_llm, model_type = llm, "production"

        # Pass the context and question to the QA chain
        qa = RetrievalQA.from_chain_type(
            llm=chosen_llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt": PROMPT_TEMPLATE,
                "document_variable_name": "context" # Key to inject documents into the prompt

            }
        )

        # Invoke the QA pipeline
        result = await qa.ainvoke({
            "context": context,  # The context gathered from relevant docs
            "query": request.question  # The user query
        }) # Use "query" as the default input key


        print(result)
        # Extract the answer from the result
        answer = result["result"]
        answer = answer.split('<|assistant|>')[-1].strip()

        tokenizer = llm.pipeline.tokenizer
        current_tokens = len(tokenizer.encode(answer))  # Ensure tokenizer works with strings
        # Measure the latency for the QA process
        latency = time.time() - start

        drift_detected = check_model_drift(answer, previous_answers)
        token_drift_detected = check_token_drift(current_tokens, previous_tokens)


        # Check token length and other metrics for model drift
        metadatas = metrics_collection.get().get("metadatas", [])
        previous_answers = [entry["answer_length"] for entry in metadatas] if metadatas else []
        previous_tokens = [entry["token_length"] for entry in metadatas] if metadatas else []
        # Drift detection

        question_id = str(uuid.uuid4()) # Generate a unique ID for the question

        # *** Log into metrics collection ***
        metrics_collection.add(
            documents=[request.question],
            metadatas=[{
                "model": model_type,
                "response_time": latency,
                "timestamp": time.time(),
                "answer_length": len(str(answer)),  # Ensure answer is treated as a string
                "token_length": current_tokens,
                "drift": bool(drift_detected),
                "token_drift": bool(token_drift_detected),
                "question_id": question_id,
                "prod_answer": answer if model_type == "production" else "",
                "shadow_answer": answer if model_type == "shadow" else "",
                "retrieved_documents_prod": [doc.page_content for doc in docs] if model_type == "production" else [],
                "retrieved_documents_shadow": [doc.page_content for doc in docs] if model_type == "shadow" else [],
            }],
            ids=[question_id]
        )

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

