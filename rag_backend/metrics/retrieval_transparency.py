# retrieval_transparency.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, List, Dict, Any
from auth_handling import get_current_user, User
from models.vectordb_setup import vectordb
from nltk import word_tokenize, sent_tokenize

retrieval_transparency_router = APIRouter()

# Get the metrics collection
metrics_client = vectordb._client
metrics_collection = metrics_client.get_or_create_collection("metrics")

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Basic keyword highlighting using HTML bold tags."""
    tokens = word_tokenize(text)
    highlighted_tokens = [f"<b>{token}</b>" if token.lower() in keywords else token for token in tokens]
    return " ".join(highlighted_tokens)

@retrieval_transparency_router.get("/retrieval_context/{question_id}")
async def get_retrieval_context(question_id: str, current_user: Optional[User] = Depends(get_current_user)):
    """
    Returns the retrieved documents, their metadata, and (optionally) highlights keywords.
    """
    try:
        results = metrics_collection.get(ids=[question_id], include=["metadatas", "documents"])
        if not results or not results["metadatas"]:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question ID not found in metrics")

        metadata = results["metadatas"][0]
        retrieved_info = metadata.get("retrieved_documents_info")
        if not retrieved_info:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Retrieved document information not found for this question")

        question_keywords = [word.lower() for word in word_tokenize(results['documents'][0])] if results['documents'] else []

        enhanced_retrieved_info = []
        for doc_info in retrieved_info:
            highlighted_content = highlight_keywords(doc_info["content"], question_keywords)
            enhanced_retrieved_info.append({
                "content": highlighted_content,
                "metadata": doc_info["metadata"],
                "relevance_score": doc_info.get("relevance_score")
            })

        return {"question_id": question_id, "retrieved_documents": enhanced_retrieved_info}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))