# feedback.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List
from auth_handling import get_current_user, User
from models.vectordb_setup import vectordb

feedback_router = APIRouter()

# Define a Pydantic model for the enhanced feedback request
class EnhancedFeedbackRequest(BaseModel):
    question_id: str = Field(..., description="Unique ID of the question asked")
    rating: Optional[int] = Field(None, description="Optional rating on a scale (e.g., 1-5)")
    feedback_text: Optional[str] = Field(None, description="Optional free-text feedback")
    categories: Optional[List[str]] = Field(None, description="Optional categories for the feedback")
    severity: Optional[str] = Field(None, description="Optional severity level (e.g., Minor, Major, Critical)")
    highlighted_text: Optional[str] = Field(None, description="Optional highlighted text from the answer")

# Get the metrics collection
metrics_client = vectordb._client
metrics_collection = metrics_client.get_or_create_collection("metrics")

@feedback_router.post("/submit_enhanced_feedback", status_code=status.HTTP_201_CREATED)
async def submit_enhanced_feedback(feedback_request: EnhancedFeedbackRequest, current_user: Optional[User] = Depends(get_current_user)):
    """
    Submits enhanced user feedback for a given question.
    """
    try:
        question_id = feedback_request.question_id

        # Retrieve the original metric entry
        results = metrics_collection.get(ids=[question_id])
        if not results or not results["metadatas"]:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question ID not found in metrics")

        metadata = results["metadatas"][0]

        # Add the feedback details to the metadata
        if feedback_request.rating is not None:
            metadata["rating"] = feedback_request.rating
        if feedback_request.feedback_text:
            metadata["feedback_text"] = feedback_request.feedback_text
        if feedback_request.categories:
            metadata["categories"] = feedback_request.categories
        if feedback_request.severity:
            metadata["severity"] = feedback_request.severity
        if feedback_request.highlighted_text:
            metadata["highlighted_text"] = feedback_request.highlighted_text

        metrics_collection.update(
            ids=[question_id],
            metadatas=[metadata]
        )

        return {"message": "Enhanced feedback submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@feedback_router.get("/feedback_summary")
async def get_feedback_summary(current_user: Optional[User] = Depends(get_current_user)):
    """
    Returns a summary of the feedback received (now includes more details).
    """
    try:
        results = metrics_collection.get(include=["metadatas"])
        feedback_summary = {"total_feedback": 0, "ratings": [], "categories": defaultdict(int), "severities": defaultdict(int)}

        if results and results["metadatas"]:
            for metadata in results["metadatas"]:
                feedback_summary["total_feedback"] += 1
                if "rating" in metadata and metadata["rating"] is not None:
                    feedback_summary["ratings"].append(metadata["rating"])
                if "categories" in metadata and metadata["categories"]:
                    for category in metadata["categories"]:
                        feedback_summary["categories"][category] += 1
                if "severity" in metadata and metadata["severity"]:
                    feedback_summary["severities"][metadata["severity"]] += 1

        summary_stats = {}
        if feedback_summary["ratings"]:
            summary_stats["average_rating"] = mean(feedback_summary["ratings"])
        summary_stats["category_counts"] = dict(feedback_summary["categories"])
        summary_stats["severity_counts"] = dict(feedback_summary["severities"])
        summary_stats["total_feedback"] = feedback_summary["total_feedback"]

        return summary_stats
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


