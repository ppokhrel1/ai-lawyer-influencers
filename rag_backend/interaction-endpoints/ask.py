from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime
import uuid

# In Chroma client settings
CHROMA_SETTINGS = ClientSettings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="chroma",
    anonymized_telemetry=False,
    collection_ttl=86400  # 24h expiration
)

# Memory & A/B Testing Models
class ConversationMemory(BaseModel):
    session_id: str
    history: List[Dict[str, str]] = []
    timestamp: datetime = datetime.utcnow()

class ABTestConfig(BaseModel):
    enabled: bool = True
    variant_a_prompt: str = "legal-expert"
    variant_b_prompt: str = "legal-simple"
    split_ratio: float = 0.5  # 50/50 split

# Initialize A/B Testing
ab_test_config = ABTestConfig()

@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(
    request: Request,
    question_data: QuestionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user: Optional[User] = Depends(get_current_user)
):
    try:
        # Get or create conversation memory
        session_id = request.cookies.get("session_id", str(uuid.uuid4()))
        memory_collection = Chroma(
            collection_name="conversation_memory",
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )
        
        # Retrieve previous messages
        memory = memory_collection.get(where={"session_id": session_id})
        if not memory:
            memory = ConversationMemory(session_id=session_id)
        
        # A/B Testing - Select prompt variant
        test_variant = "a" if hash(session_id) % 100 < ab_test_config.split_ratio * 100 else "b"
        prompt_template = (
            ab_test_config.variant_a_prompt if test_variant == "a"
            else ab_test_config.variant_b_prompt
        )
        
        # Build context-aware question
        chat_history = "\n".join(
            [f"Q: {q['question']}\nA: {q['answer']}" 
             for q in memory.history[-3:]]  # Last 3 exchanges
        )
        augmented_question = f"""
        Conversation history:
        {chat_history}
        
        New question: {question_data.question}
        """
        
        # Initialize retriever (user-specific or public)
        retriever = (
            Chroma(
                collection_name=f"user_{user.username}_{test_variant}" if user else f"public_{test_variant}",
                embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS
            ).as_retriever(search_kwargs={"k": 5})
            if ab_test_config.enabled
            else vectordb.as_retriever()
        )
        
        # Build QA chain with memory
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt_template,
                "memory": ConversationBufferMemory(
                    memory_key="chat_history",
                    input_key="question"
                )
            }
        )
        
        # Execute query
        result = qa_chain.invoke({
            "query": augmented_question,
            "chat_history": chat_history
        })
        
        # Store interaction in memory
        memory.history.append({
            "question": question_data.question,
            "answer": result["result"],
            "timestamp": datetime.utcnow().isoformat(),
            "variant": test_variant
        })
        
        # Persist memory
        memory_collection.upsert(
            ids=[session_id],
            documents=[memory.json()],
            metadatas=[{"type": "conversation_memory"}]
        )
        
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result.get("source_documents", [])],
            "session_id": session_id,
            "variant": test_variant,
            "memory_length": len(memory.history)
        }

    except Exception as e:
        logger.error(f"QA Error: {str(e)}", extra={
            "session_id": session_id,
            "question": question_data.question
        })
        raise HTTPException(status_code=500, detail="Legal query service unavailable")