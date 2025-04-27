# drift_analysis.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, List, Dict
from auth_handling import get_current_user, User
from models.vectordb_setup import vectordb
from difflib import unified_diff

drift_router = APIRouter()

ANSWER_LENGTH_THRESHOLD = 50  # Example threshold for answer length drift
TOKEN_COUNT_THRESHOLD = 20   # Example threshold for token count drift
DRIFT_WINDOW_SIZE = 10       # Number of previous answers/tokens to consider for drift


# Get the metrics collection
metrics_client = vectordb._client
metrics_collection = metrics_client.get_or_create_collection("metrics")

def compare_answers_detailed(prod_answer: str, shadow_answer: str) -> Dict:
    """
    Compares two answers and highlights differences using difflib.
    """
    diff = list(unified_diff(prod_answer.splitlines(keepends=True),
                             shadow_answer.splitlines(keepends=True)))
    diff_str = ''.join(diff)
    return {"identical": not diff, "diff": diff_str}

@drift_router.get("/detailed_drift_analysis/{question_id}")
async def analyze_detailed_drift(question_id: str, current_user: Optional[User] = Depends(get_current_user)):
    """
    Analyzes drift in detail for a specific question, including retrieved documents
    and a more detailed answer comparison.
    """
    try:
        results = metrics_collection.get(ids=[question_id], include=["metadatas", "documents"])
        if not results or not results["metadatas"]:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question ID not found in metrics")

        metadata = results["metadatas"][0]
        question = results.get("documents", ["N/A"])[0]
        retrieved_documents_prod = metadata.get("retrieved_documents_prod", [])
        retrieved_documents_shadow = metadata.get("retrieved_documents_shadow", [])
        prod_answer = metadata.get("prod_answer", "N/A")
        shadow_answer = metadata.get("shadow_answer", "N/A")

        answer_comparison = compare_answers_detailed(prod_answer, shadow_answer)

        return {
            "question_id": question_id,
            "question": question,
            "production_model": metadata.get("model_used") == "production",
            "shadow_model": metadata.get("model_used") == "shadow",
            "production_answer": prod_answer,
            "shadow_answer": shadow_answer,
            "answer_comparison": answer_comparison,
            "retrieved_documents": {
                "production": retrieved_documents_prod,
                "shadow": retrieved_documents_shadow,
            },
            "metrics": {
                "response_time": metadata.get("response_time"),
                "drift": metadata.get("drift"),
                "token_drift": metadata.get("token_drift"),
            },
            "timestamp": metadata.get("timestamp")
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@drift_router.get("/drift_explorer_with_context")
async def explore_drift_with_context(from_timestamp: Optional[float] = None, to_timestamp: Optional[float] = None, current_user: Optional[User] = Depends(get_current_user)):
    """
    Explores drift over a period, including retrieved documents and detailed answer comparison.
    """
    try:
        query_params = {}
        if from_timestamp:
            query_params["timestamp"] = {"$gte": from_timestamp}
        if to_timestamp:
            if "timestamp" in query_params:
                query_params["timestamp"]["$lte"] = to_timestamp
            else:
                query_params["timestamp"] = {"$lte": to_timestamp}

        results = metrics_collection.get(where=query_params, include=["metadatas", "documents"])
        drift_data = []

        if results and results["metadatas"]:
            for metadata in results["metadatas"]:
                question = results.get("documents", ["N/A"])[0]
                retrieved_documents_prod = metadata.get("retrieved_documents_prod", [])
                retrieved_documents_shadow = metadata.get("retrieved_documents_shadow", [])
                prod_answer = metadata.get("prod_answer", "N/A")
                shadow_answer = metadata.get("shadow_answer", "N/A")

                answer_comparison = compare_answers_detailed(prod_answer, shadow_answer)

                drift_data.append({
                    "question_id": metadata.get("ids", ["unknown"])[0],
                    "question": question,
                    "production_answer": prod_answer,
                    "shadow_answer": shadow_answer,
                    "answer_comparison": answer_comparison,
                    "retrieved_documents": {
                        "production": retrieved_documents_prod,
                        "shadow": retrieved_documents_shadow,
                    },
                    "metrics": {
                        "response_time": metadata.get("response_time"),
                        "drift": metadata.get("drift"),
                        "token_drift": metadata.get("token_drift"),
                    },
                    "timestamp": metadata.get("timestamp")
                })
        return drift_data
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
