# AI Model Monitoring and Feedback API

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deployed at](https://img.shields.io/badge/Deployed_at-https://legal--qa--frontend--754457156890.us--central1--run--app-blue)](https://legal-qa-frontend-754457156890.us-central1.run.app)

This API is used to assess the health of QA models, gathering user insights, and spotting any performance dips. Built with FastAPI and ChromaDB, it gives us the observability we need to ensure our models are performing as expected and meeting user needs.

## Endpoints at a Glance

* **`POST /submit_enhanced_feedback`**: Collects detailed user feedback (ratings, text, categories, severity, highlighted text) on legal query responses.
* **`GET /feedback_summary`**: Provides a summary of all user feedback, including ratings and categories.
* **`GET /retrieval_context/{question_id}`**: Shows the retrieved legal documents used for a specific query, highlighting relevant keywords.
* **`GET /detailed_drift_analysis/{question_id}`**: Compares production and shadow model answers and retrieved documents for a specific legal question, highlighting differences.
* **`GET /drift_explorer_with_context`**: Allows exploring drift over time, showing answer comparisons and retrieved documents for legal queries.
* **`GET /metrics`**: Returns overall performance metrics: A/B test split, response times, and drift/token drift counts.
* **`GET /rag_stats`**: Shows document counts in the RAG knowledge base (user-specific if authenticated).
* **`GET /daily_metrics`**: Provides daily aggregated performance metrics and drift counts.

## Tech Stack

Key technologies powering this API:

* **FastAPI**: For building the API efficiently.
* **Pydantic**: For data validation.
* **ChromaDB**: Our vector database for storing monitoring data.
* **Langchain**: For vector database interactions and potentially embeddings.
* **NLTK**: For basic text processing (keyword highlighting).
* **`difflib`**: For comparing text differences in drift analysis.
* **Uvicorn**: To run the FastAPI app.
* **NumPy & Statistics**: For calculating performance metrics.
* **`python-jose` & `passlib`**: For secure authentication (JWT).

## Getting Started

1.  **Clone the repo** (if you have it).
2.  **Install dependencies**: `pip install -r requirements.txt` (make sure this file exists with all the necessary packages).
3.  **Check ChromaDB setup**: Ensure `models/vectordb_setup.py` has the correct ChromaDB directory configured.
4.  **Run the app**: `uvicorn main:app --reload` (adjust `main` and `app` if needed).

## Authentication

Endpoints are secured with JWT. You'll need to implement user login to get tokens for accessing protected routes.

## How to Use

Interact with the API using standard HTTP methods. See the endpoint descriptions above for details on requests and responses. The API is live at `https://legal-qa-frontend-754457156890.us-central1.run.app`.

## Configuration

Configuration like ChromaDB path and drift detection thresholds can be found in the respective Python files (`models/vectordb_setup.py`, `drift_analysis.py`).

## Contributing

Feel free to contribute following standard Git practices.

## License

[MIT License](https://opensource.org/licenses/MIT)