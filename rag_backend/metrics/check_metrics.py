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
from statistics import mean, median
import numpy as np

import datetime

from models.vectordb_setup import CHROMA_DIR, embeddings, vectordb
from auth_handling import get_current_user, User

monitoring_router = APIRouter()

from models.vectordb_setup import vectordb

# instead of: metrics_client = Client(Settings(persist_directory=CHROMA_DIR))
metrics_client = vectordb._client
metrics_collection = metrics_client.get_or_create_collection("metrics")



# Function to compute basic statistics
def compute_stats(times):
    if not times:
        return {"count": 0, "avg": None, "median": None, "p95": None}
    return {
        "count": len(times),
        "avg_response_time": mean(times),
        "median_response_time": median(times),
        "p95_response_time": np.percentile(times, 95)
    }

@monitoring_router.get("/metrics")
async def get_metrics():
    """
    Returns A/B test counts, average/median/p95 response times, and model drift metrics.
    """
    try:
        docs = metrics_collection.get()
        entries = docs["metadatas"] or []

        prod = [e["response_time"] for e in entries if e.get("model") == "production"]
        shadow = [e["response_time"] for e in entries if e.get("model") == "shadow"]

        # Collect drift and token drift metrics
        drift_counts = defaultdict(int)
        token_drift_counts = defaultdict(int)
        for entry in entries:
            if entry.get("drift"):
                drift_counts[entry["drift"]] += 1
            if entry.get("token_drift"):
                token_drift_counts[entry["token_drift"]] += 1

        # Prepare stats
        production_stats = compute_stats(prod)
        shadow_stats = compute_stats(shadow)

        return {
            "production": production_stats,
            "shadow": shadow_stats,
            "drift_counts": dict(drift_counts),
            "token_drift_counts": dict(token_drift_counts),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/rag_stats")
async def get_rag_stats(user=Depends(get_current_user)):
    """
    Returns document counts from RAG (retrieval-augmented generation) model per user.
    """
    if user:
        user_col = vectordb._client.get_or_create_collection(user.username)
    else:
        user_col = vectordb._collection
    return {
        "collection_name": user_col.name,
        "document_count": user_col.count()
    }


@monitoring_router.get("/daily_metrics")
async def get_daily_metrics():
    """
    Returns daily metrics, including response time, drift counts, and token drift.
    """
    try:
        daily_metrics = defaultdict(list)
        drift_metrics = defaultdict(int)
        token_drift_metrics = defaultdict(int)

        docs = metrics_collection.get()
        entries = docs["metadatas"] or []

        for entry in entries:
            timestamp = entry.get("timestamp")
            if timestamp:
                date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                response_time = entry.get("response_time")
                drift = entry.get("drift")
                token_drift = entry.get("token_drift")

                daily_metrics[date].append(response_time)

                if drift is not None:
                    drift_metrics[date] += drift
                if token_drift is not None:
                    token_drift_metrics[date] += token_drift

        daily_stats = {
            date: {
                "count": len(times),
                "avg_response_time": mean(times) if times else None,
                "median_response_time": median(times) if times else None,
                "p95_response_time": np.percentile(times, 95) if times else None,
                "drift_count": drift_metrics[date],
                "token_drift_count": token_drift_metrics[date],
            }
            for date, times in daily_metrics.items()
        }

        return daily_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@monitoring_router.get("/weekly_metrics")
async def get_weekly_metrics():
    """
    Returns weekly metrics, including response time, drift counts, and token drift.
    """
    try:
        weekly_metrics = defaultdict(list)
        drift_metrics = defaultdict(int)
        token_drift_metrics = defaultdict(int)

        docs = metrics_collection.get()
        entries = docs["metadatas"] or []

        for entry in entries:
            timestamp = entry.get("timestamp")
            if timestamp:
                date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                week = datetime.datetime.strptime(date, '%Y-%m-%d').isocalendar()[1]
                response_time = entry.get("response_time")
                drift = entry.get("drift")
                token_drift = entry.get("token_drift")

                weekly_metrics[week].append(response_time)

                if drift is not None:
                    drift_metrics[week] += drift
                if token_drift is not None:
                    token_drift_metrics[week] += token_drift

        weekly_stats = {
            week: {
                "count": len(times),
                "avg_response_time": mean(times) if times else None,
                "median_response_time": median(times) if times else None,
                "p95_response_time": np.percentile(times, 95) if times else None,
                "drift_count": drift_metrics[week],
                "token_drift_count": token_drift_metrics[week],
            }
            for week, times in weekly_metrics.items()
        }

        return weekly_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

