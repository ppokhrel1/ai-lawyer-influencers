FROM python:3.9
WORKDIR rag_backend/
COPY . .
RUN   pip install fastapi uvicorn pdfplumber pytesseract pillow      python-multipart langchain chromadb transformers      beautifulsoup4 requests sentence-transformers aiosqlite passlib[bcrypt] python-jose[cryptography] sqlalchemy databases langchain-community

# Explicitly expose the port
EXPOSE $PORT

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]

