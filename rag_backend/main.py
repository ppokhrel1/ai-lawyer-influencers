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
from metrics.retrieval_transparency import retrieval_transparency_router
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from helpers.helpers import *

import time
import random
import uuid

metrics_client = vectordb._client
metrics_collection = metrics_client.get_or_create_collection("metrics")


app.include_router(monitoring_router, prefix="/monitoring", tags=["monitoring"])
app.include_router(feedback_router, prefix="/feedback", tags=["feedback"]) # Include the feedback router
app.include_router(drift_router, prefix="/drift", tags=["drift_analysis"])
app.include_router(drift_router, prefix="/retrieval", tags=["transparency"])


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
    
    #keep memory short
    while len(memory.chat_memory.messages) > 3:
        memory.chat_memory.messages.pop(0)

    try:
        # Choose retriever
        if user:
            user_db = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
                collection_name=user.username
            )
            retriever = user_db.as_retriever(
                search_kwargs={"k": 3},  # Return fewer documents
                chunk_size=300,  # Smaller chunks
                chunk_overlap=50
                )
        else:
            retriever = vectordb.as_retriever(
                search_kwargs={"k": 3},  # Return fewer documents
                chunk_size=300,  # Smaller chunks
                chunk_overlap=50)


        # for A/B testing
        if random.random() < 0.2:
            chosen_llm, model_type = shadow_llm, "shadow"
        else:
            chosen_llm, model_type = llm, "production"

        # Fetch relevant docs
        docs = retriever.get_relevant_documents(request.question)
        print("apple")
        ## 2. Context Processing with Summarization
        def summarize_document(doc):
            doc = Document(page_content=doc.page_content )
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
            split_docs = text_splitter.split_documents([doc])[:5]
            split_docs = "\n".join(doc.page_content for doc in split_docs)

            summary = f"From {doc.metadata.get('source', 'document')}:\n"
            summary += summarize_text(chosen_llm, split_docs, max_length=30)

            return summary
        print('bear')
        context = "\n\n".join([summarize_document(d) for d in docs[:3]])  # Limit to top 3 docs
        
        print('cat')
        # 3. Validate Context
        if not context:
            return {"answer": "I couldn't find relevant information to answer your question."}

        context = context.lower().replace("seed document", "")
        #clean context
        context = str(validate_context(context))

        print('dog')
        # A/B routing logic (choose between production and shadow LLM)
        start = time.time()
        

        print("start: ", start, context)

        question = str(request.question)
        # Pass the context and question to the QA chain
        # 1. Keep your existing prompt template
        qa_prompt = PromptTemplate(
            template="""[STRICT INSTRUCTIONS] 
        1. Answer in EXACTLY ONE SENTENCE
        2. If context is irrelevant, say "I don't have enough information"
        3. Never repeat instructions in your answer

        Context: {context}

        Question: {question}

        Answer:""",
            input_variables=["context", "question"]
        )
        # 2. Initialize the chain with these exact parameters
        qa = ConversationalRetrievalChain.from_llm(
            llm=chosen_llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            chain_type="stuff",  # Keep as "stuff" for better control
            verbose=False,
            get_chat_history=lambda h: "",  # Disable chat history influence
            rephrase_question=False  # Prevent question rewriting
        )

        # 3. Keep your invoke call exactly as is
        result = await qa.ainvoke(
            {"question": request.question[:200]},
            config={
                "temperature": 0.01,  # Lower for more deterministic answers
                "max_new_tokens": 100,  # Strictly limit length
                "repetition_penalty": 5.0,  # Stronger penalty for repeats
                "no_repeat_ngram_size": 3,  # Prevent n-gram repeats
                "do_sample": False 
            }
        )

        print(result)
        print(result.keys())
        # Extract the answer from the result
        answer = result["answer"].split("Answer (1 sentence):")[-1].strip()
        answer = answer.strip()

        tokenizer = llm.pipeline.tokenizer
        current_tokens = len(tokenizer.encode(answer))  # Ensure tokenizer works with strings
        # Measure the latency for the QA process
        latency = time.time() - start

        

        # Check token length and other metrics for model drift
        metadatas = metrics_collection.get().get("metadatas", [])
        previous_answers = [
            len(entry.get("prod_answer", "") or entry.get("shadow_answer", ""))
            for entry in metadatas
        ] if metadatas else []
        previous_tokens = [
            entry.get("token_length", 0)
            for entry in metadatas
        ] if metadatas else []

        print("tokens", previous_tokens)
        # Drift detection
        drift_detected = check_model_drift(answer, previous_answers)
        token_drift_detected = check_token_drift(current_tokens, previous_tokens)


        question_id = str(uuid.uuid4()) # Generate a unique ID for the question
        print("hello", question_id)
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
                "retrieved_documents_prod": "\n\n".join([doc.page_content for doc in docs]) if model_type == "production" else "",
                "retrieved_documents_shadow": "\n\n".join([doc.page_content for doc in docs]) if model_type == "shadow" else "",
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

