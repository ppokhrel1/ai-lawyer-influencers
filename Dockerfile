FROM python:3.9
# Set working directory and copy files
WORKDIR /app
COPY rag_backend/ ./rag_backend/

RUN   pip install fastapi uvicorn pdfplumber pytesseract pillow      python-multipart langchain chromadb transformers      beautifulsoup4 requests sentence-transformers aiosqlite passlib[bcrypt] python-jose[cryptography] sqlalchemy databases langchain-community

# Explicitly expose the port
EXPOSE 8000

WORKDIR /app/rag_backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

