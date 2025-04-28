# AI Model Monitoring and Feedback API

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deployed at](https://img.shields.io/badge/Deployed_at-https://legal--qa--frontend--754457156890.us--central1--run--app-blue)](https://legal-qa-frontend-754457156890.us-central1.run.app)

This API serves as a critical backend component for monitoring the performance, gathering user feedback, and analyzing potential drift in AI models, specifically within a legal question-answering (QA) context. Leveraging a robust architecture built on FastAPI and ChromaDB, it provides valuable insights into model behavior and user satisfaction, enabling proactive maintenance and iterative improvement.

## Overview of Endpoints

This API offers a suite of endpoints designed for comprehensive AI model observability:

### Feedback Management (`feedback.py`)

* **`POST /submit_enhanced_feedback`**: Enables users to provide detailed feedback on model responses to legal queries. This includes an optional numerical rating, free-text comments elaborating on the quality or relevance of the answer, categorization of the feedback (e.g., accuracy, clarity, completeness), a severity level (e.g., Minor, Major, Critical), and the ability to highlight specific text within the answer that the feedback pertains to. This granular feedback mechanism is crucial for identifying specific areas of improvement.
* **`GET /feedback_summary`**: Aggregates and summarizes all collected user feedback. This endpoint returns the total number of feedback submissions, the average rating, a breakdown of feedback by category, and the distribution of feedback severity levels. This provides a high-level overview of user satisfaction and common issues.

### Retrieval Transparency (`retrieval_transparency.py`)

* **`GET /retrieval_context/{question_id}`**: Offers insights into the information retrieval process underpinning the legal QA model. For a given legal question ID, it returns the specific documents retrieved from the knowledge base that were used to formulate the answer. The content of these documents is presented with keywords from the original question highlighted, along with the metadata associated with each document (e.g., source, publication date, relevance score). This transparency is vital for debugging incorrect answers and understanding the model's reasoning process.

### Drift Analysis (`drift_analysis.py`)

* **`GET /detailed_drift_analysis/{question_id}`**: Facilitates in-depth analysis of model drift for a specific legal query. By comparing the responses of a production model against a shadow model (potentially a newer version or a baseline), this endpoint highlights textual differences at a line-by-line level using `difflib`. It also provides the retrieved documents for both models, allowing for the assessment of drift in the information retrieval stage. Key performance indicators (KPIs) like response time and binary drift/token drift flags are also included, offering a comprehensive view of model stability.
* **`GET /drift_explorer_with_context`**: Enables the exploration of model drift trends over time. By specifying optional start and end timestamps, users can retrieve a series of drift events, each providing the original legal question, the production and shadow model answers with detailed comparisons, the retrieved documents for each model, and associated metrics. This temporal analysis is essential for identifying gradual degradation in model performance or the impact of data/model updates.

### Monitoring (`monitoring.py`)

* **`GET /metrics`**: Provides a real-time snapshot of key model performance indicators. This includes the volume of requests handled by the production and shadow models (for A/B testing analysis), statistical measures of response times for both models (average, median, 95th percentile latency), and aggregated counts of detected model drift and token drift across all evaluated queries. This endpoint is crucial for operational monitoring and alerting.
* **`GET /rag_stats`**: Offers insights into the usage of the Retrieval-Augmented Generation (RAG) component within the legal QA system. For authenticated users, it returns the document count within their personalized knowledge base. If no user is authenticated, it provides the document count of the general legal knowledge base. This helps in understanding the scale and utilization of the underlying knowledge resources.
* **`GET /daily_metrics`**: Presents a time-series view of critical performance metrics aggregated on a daily basis. This includes the daily average, median, and 95th percentile response times, as well as the daily counts of detected model drift and token drift. This allows for the identification of performance trends and potential anomalies over time.

## Technologies Used

This API leverages a modern and robust technology stack suitable for building scalable and maintainable AI monitoring solutions:

* **FastAPI**: A high-performance, asynchronous Python web framework that ensures efficient handling of API requests and responses, crucial for real-time monitoring dashboards and feedback processing.
* **Pydantic**: Used for data validation and serialization, ensuring data integrity across API interactions and providing clear contracts for request and response bodies.
* **ChromaDB**: An open-source, AI-native vector database that efficiently stores and indexes embeddings of monitoring metrics and feedback data. Its vector similarity search capabilities are likely used for analyzing trends in feedback or identifying similar instances of model drift.
* **Langchain**: Provides the necessary abstractions and integrations for working with vector databases (like ChromaDB) and potentially for generating embeddings of the legal queries and model responses, which could be used for more advanced drift detection or semantic analysis of feedback.
* **NLTK (Natural Language Toolkit)**: Employed for fundamental natural language processing tasks such as tokenization, which is used in highlighting relevant keywords in the retrieved documents, enhancing the interpretability of the retrieval context.
* **`difflib`**: A Python standard library module used for comparing sequences, specifically employed here to perform detailed line-by-line comparisons of the production and shadow model answers, providing a clear visualization of textual drift.
* **`python-multipart`**: While not directly evident in the described endpoints, this dependency of FastAPI suggests the API is capable of handling multipart form data, which could be relevant for future extensions like uploading feedback with attachments.
* **`uvicorn`**: An ASGI (Asynchronous Server Gateway Interface) server that efficiently runs the FastAPI application, enabling high concurrency and performance required for handling monitoring data streams.
* **NumPy and Statistics**: Essential libraries for numerical computations and statistical analysis, used here to calculate key performance indicators like average, median, and percentile response times, providing quantitative insights into model performance.
* **`python-jose` (JSON Object Signing and Encryption) and `passlib`**: These libraries are critical for implementing secure authentication and authorization using JWT. `python-jose` handles the creation and verification of JWTs, while `passlib` provides secure password hashing mechanisms, ensuring that the API endpoints are protected and only accessible to authorized users.

## Deployment

This API is currently deployed and accessible at `https://legal-qa-frontend-754457156890.us-central1.run.app`. This indicates a cloud-based deployment, likely leveraging Google Cloud Run, a fully managed compute platform that allows running stateless containers. This choice suggests a focus on scalability and ease of management.

## Setup and Installation

To set up and run this API locally for development or testing purposes, follow these steps:

1.  **Clone the Repository**: Obtain the source code of the API.
2.  **Install Dependencies**: Navigate to the project directory in your terminal and install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure that the `requirements.txt` file in the project root lists all the dependencies mentioned above, including `fastapi`, `uvicorn`, `chromadb`, `langchain`, `nltk`, `python-jose[cryptography]`, `passlib[bcrypt]`, `python-multipart`, `numpy`, and potentially other related libraries.)*
3.  **Configure ChromaDB**: Review the `models/vectordb_setup.py` file to understand how ChromaDB is initialized and where the data is persisted (`CHROMA_DIR`). Ensure that the specified directory exists and the application has the necessary read/write permissions.
4.  **Run the Application**: Execute the FastAPI application using Uvicorn:
    ```bash
    uvicorn main:app --reload
    ```
    *(Replace `main` with the name of your main application file and `app` with the name of your FastAPI application instance if they are different.)*

## Authentication

This API employs JWT-based authentication to secure its endpoints. The `auth_handling.py` module likely contains the logic for user authentication (login) and the `get_current_user` dependency used to verify the validity of JWT tokens provided in the `Authorization` header of API requests. You will need to implement the user registration and login functionalities (likely through separate API endpoints not detailed here) to obtain valid JWT tokens for accessing protected resources.

## Usage

Once the API is running, you can interact with its endpoints using standard HTTP clients. For example:

* **Using `curl`**:
    ```bash
    # Submit feedback (example)
    curl -X POST -H "Content-Type: application/json" -d '{"question_id": "123", "rating": 4, "feedback_text": "The answer was helpful but could be more concise."}' [https://legal-qa-frontend-754457156890.us-central1.run.app/submit_enhanced_feedback](https://legal-qa-frontend-754457156890.us-central1.run.app/submit_enhanced_feedback)

    # Get feedback summary (requires authentication)
    curl -X GET -H "Authorization: Bearer <YOUR_JWT_TOKEN>" [https://legal-qa-frontend-754457156890.us-central1.run.app/feedback_summary](https://legal-qa-frontend-754457156890.us-central1.run.app/feedback_summary)

    # Get retrieval context (example)
    curl [https://legal-qa-frontend-754457156890.us-central1.run.app/retrieval_context/456](https://legal-qa-frontend-754457156890.us-central1.run.app/retrieval_context/456)
    ```
* **Using Python's `requests` library**:
    ```python
    import requests
    import json

    # Submit feedback (example)
    feedback_data = {"question_id": "789", "rating": 5, "feedback_text": "Excellent and accurate answer."}
    response = requests.post("[https://legal-qa-frontend-754457156890.us-central1.run.app/submit_enhanced_feedback](https://legal-qa-frontend-754457156890.us-central1.run.app/submit_enhanced_feedback)", json=feedback_data)
    print(response.json())

    # Get feedback summary (requires authentication)
    headers = {"Authorization": "Bearer <YOUR_JWT_TOKEN>"}
    response = requests.get("[https://legal-qa-frontend-754457156890.us-central1.run.app/feedback_summary](https://legal-qa-frontend-754457156890.us-central1.run.app/feedback_summary)", headers=headers)
    print(response.json())
    ```

Refer to the "Overview of Endpoints" section for specific details on the request and response formats for each endpoint.

## Configuration

The behavior of this API can be configured through various parameters within the Python source files:

* **ChromaDB Storage**: The location where ChromaDB persists its data is defined in `models/vectordb_setup.py` (`CHROMA_DIR`).
* **Drift Detection Thresholds**: The `drift_analysis.py` file likely contains thresholds (e.g., `ANSWER_LENGTH_THRESHOLD`, `TOKEN_COUNT_THRESHOLD`) that determine when a significant drift in model responses is flagged. These values can be adjusted based on the specific requirements of the legal QA domain.
* **API Security**: Authentication and authorization settings, including JWT secret keys and password hashing algorithms, are configured within `auth_handling.py`.

Review these files to understand and customize the API's behavior.

## Contributing

We welcome contributions to enhance this AI model monitoring and feedback API. Please follow standard Git contribution practices:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear and concise messages.
4.  Push your changes to your fork.
5.  Submit a pull request detailing your changes.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
