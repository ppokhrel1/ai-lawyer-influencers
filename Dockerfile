FROM python:3.10-slim  

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils git \  
    && rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt .

RUN pip install -r ./requirements.txt

RUN apt-get update && apt-get install -y \
    tesseract-ocr


# Create models directory
RUN mkdir -p /app/models

# Download and organize models properly
# Download and organize models properly
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4'); \
model.save('/app/models/minilm')"

RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
model = AutoModelForCausalLM.from_pretrained('distilgpt2'); \
tokenizer = AutoTokenizer.from_pretrained('distilgpt2'); \
model.save_pretrained('/app/models/flan-t5-base'); \
tokenizer.save_pretrained('/app/models/flan-t5-base')"


RUN ls -l /app/models/flan-t5-base && \
    echo "Tokenizer config:" && \
    cat /app/models/flan-t5-base/tokenizer_config.json

# Set environment variables for offline usage
ENV TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    TORCH_HOME=/app/models

RUN ls -l /app/models



# Verify models are accessible
COPY . .

EXPOSE 8080

WORKDIR rag_backend/

# Run with single worker and request timeout
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "30"]

