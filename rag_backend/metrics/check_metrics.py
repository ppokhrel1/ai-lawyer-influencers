# monitoring.py
import os
from typing import Optional
from statistics import mean

from fastapi import APIRouter, Depends, HTTPException
from chromadb import Client
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from collections import defaultdict
from datetime import timedelta
from statistics import mean

import datetime

from models.vectordb_setup import CHROMA_DIR, embeddings, vectordb
from auth_handling import get_current_user, User

monitoring_router = APIRouter()

from models.vectordb_setup import vectordb

# instead of: metrics_client = Client(Settings(persist_directory=CHROMA_DIR))
metrics_client = vectordb._client
metrics_collection = metrics_client.get_or_create_collection("metrics")


@monitoring_router.get("/metrics")
async def get_metrics():
    """
    Returns A/B test counts and average response times.
    """
    try:
        docs    = metrics_collection.get()
        entries = docs["metadatas"] or []
        prod    = [e["response_time"] for e in entries if e.get("model") == "production"]
        shadow  = [e["response_time"] for e in entries if e.get("model") == "shadow"]
        return {
            "production": {
                "count":               len(prod),
                "avg_response_time":   mean(prod)   if prod   else None,
            },
            "shadow": {
                "count":               len(shadow),
                "avg_response_time":   mean(shadow) if shadow else None,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/rag_stats")
async def get_rag_stats(user=Depends(get_current_user)):
    if user:
        user_col = vectordb._client.get_or_create_collection(user.username)
    else:
        user_col = vectordb._collection
    return {
        "collection_name": user_col.name,
        "document_count":  user_col.count()
    }


@monitoring_router.get("/daily_metrics")
async def get_daily_metrics():
    try:
        # Group metrics by day
        daily_metrics = defaultdict(list)
        docs = metrics_collection.get()
        entries = docs["metadatas"] or []
        
        for entry in entries:
            timestamp = entry.get("timestamp")
            if timestamp:
                date_time_obj = datetime.datetime.fromtimestamp(timestamp)

                # Convert to a string in the desired format (e.g., YYYY-MM-DD)
                date_string = date_time_obj.strftime('%Y-%m-%d')
                date = date_string.split(' ')[0]  # Extract the date (YYYY-MM-DD)
                response_time = entry.get("response_time")
                daily_metrics[date].append(response_time)

        # Aggregate the metrics for each day
        daily_stats = {
            date: {
                "count": len(times),
                "avg_response_time": mean(times) if times else 'guest'
            }
            for date, times in daily_metrics.items()
        }

        return daily_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/weekly_metrics")
async def get_weekly_metrics():
    try:
        # Group metrics by week (use the ISO week date format)
        weekly_metrics = defaultdict(list)
        docs = metrics_collection.get()
        entries = docs["metadatas"] or []

        for entry in entries:
            timestamp = entry.get("timestamp")
            if timestamp:
                date_time_obj = datetime.datetime.fromtimestamp(timestamp)

                # Convert to a string in the desired format (e.g., YYYY-MM-DD)
                date_string = date_time_obj.strftime('%Y-%m-%d')
                date = date_string.split(' ')[0]  # Extract the date (YYYY-MM-DD)
                week = datetime.datetime.strptime(date, '%Y-%m-%d').isocalendar()[1]  # Get ISO week number
                response_time = entry.get("response_time")
                weekly_metrics[week].append(response_time)

        # Aggregate the metrics for each week
        weekly_stats = {
            week: {
                "count": len(times),
                "avg_response_time": mean(times) if times else None
            }
            for week, times in weekly_metrics.items()
        }

        return weekly_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


