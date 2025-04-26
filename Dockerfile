FROM python:3.10-slim  

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \  
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    # Core web server
    fastapi==0.109.2 \
    uvicorn==0.27.0 \
    
    # File processing
    pdfplumber==0.11.0 \
    pillow==10.2.0 \
    python-multipart==0.0.9 \
    
    # Database
    aiosqlite==0.20.0 \
    sqlalchemy==2.0.25 \
    databases==0.9.0 \
    
    # Security
    passlib[bcrypt]==1.7.4 \
    python-jose[cryptography]==3.3.0 \
    
    # LangChain ecosystem (latest stable)
    langchain==0.1.13 \
    langchain-community==0.0.34 \
    langchain-core==0.1.45 \
    langchain-text-splitters==0.0.1 \
    
    # Embeddings
    sentence-transformers==2.6.1 \
    pytesseract \  
    bs4 \ 
    # LLM dependencies
    transformers==4.38.2 \
    torch==2.2.1+cpu \
    sentencepiece \
    chromadb[postgres] \
    asyncpg  \
    pg8000 \
    gcsfs  \
    pgvector \
    psycopg2-binary \
    alembic \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Create models directory
RUN mkdir -p /app/models

# Download and organize models properly
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens'); \
model.save('/app/models/minilm')"

RUN python -c "\
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
model = AutoModelForSeq2SeqLM.from_pretrained('t5-small'); \
tokenizer = AutoTokenizer.from_pretrained('t5-small'); \
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

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('/app/models/minilm'); \
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
model = AutoModelForSeq2SeqLM.from_pretrained('/app/models/flan-t5-base'); \
tokenizer = AutoTokenizer.from_pretrained('/app/models/flan-t5-base', use_fast=False); \
from transformers import pipeline; \
pipeline('text2text-generation', model=model, tokenizer=tokenizer)"

# Verify models are accessible
COPY . .

EXPOSE 8080

WORKDIR rag_backend/

# Run with single worker and request timeout
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "30"]

