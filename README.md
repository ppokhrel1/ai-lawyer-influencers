# 📄 Document QA System with FastAPI

This FastAPI application enables:
- 💾 Uploading and extracting text from PDFs and images
- 🌐 Ingesting documents from URLs
- 🔐 User authentication with JWT
- 🧠 Document vectorization and storage (ChromaDB + HuggingFace embeddings)
- 💬 Question-answering over uploaded documents using `FLAN-T5`

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install fastapi uvicorn pdfplumber pytesseract pillow \
            python-multipart langchain chromadb transformers \
            beautifulsoup4 requests sentence-transformers \
            passlib[bcrypt] jose[cryptography] sqlalchemy databases
```

### 2. Run the API

```bash
uvicorn main:app --reload
```

---

## 🔐 Authentication Endpoints

### ✅ Register

- **Method:** POST  
- **Route:** `/register`  
- **Auth:** ❌

```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass"}'
```

**Response:**
```json
{ "msg": "User created successfully" }
```

---

### 🔑 Login & Get JWT Token

- **Method:** POST  
- **Route:** `/token`  
- **Auth:** ❌

```bash
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=testpass"
```

**Response:**
```json
{ "access_token": "xxxxx.yyyyy.zzzzz", "token_type": "bearer" }
```

---

## 📂 Document Handling Endpoints

### 📤 Upload Documents

- **Method:** POST  
- **Route:** `/upload_documents`  
- **Auth:** ✅

```bash
curl -X POST http://localhost:8000/upload_documents \
  -H "Authorization: Bearer <your_token>" \
  -F "files=@example.pdf"
```

**Response:**
```json
{
  "message": "Document(s) processed and stored successfully.",
  "files": ["example.pdf"]
}
```

---

### 🌐 Ingest Content from a URL

- **Method:** POST  
- **Route:** `/add_url`  
- **Auth:** ❌

```bash
curl -X POST http://localhost:8000/add_url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

**Response:**
```json
{ "message": "Content from URL processed and stored successfully." }
```

---

### 👁️ Preview a Document

- **Method:** GET  
- **Route:** `/preview/{filename}`  
- **Auth:** ❌

```bash
curl http://localhost:8000/preview/example.pdf
```

**Response:**
```json
{
  "filename": "example.pdf",
  "content": "Extracted text content goes here..."
}
```

---

### 🗑️ Delete a Document

- **Method:** DELETE  
- **Route:** `/documents/{filename}`  
- **Auth:** ❌

```bash
curl -X DELETE http://localhost:8000/documents/example.pdf
```

**Response:**
```json
{ "message": "Document deleted successfully." }
```

---

## 💬 Question-Answering Endpoint

### ❓ Ask a Question

- **Method:** POST  
- **Route:** `/ask`  
- **Auth:** ❌

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main idea of the document?"}'
```

**Response:**
```json
{
  "question": "What is the main idea of the document?",
  "answer": "The document discusses principles of modern data science...",
  "source_documents": ["example.pdf"]
}
```

---

## 📘 API Summary Table

| Endpoint                | Method | Auth | Description                          |
|-------------------------|--------|------|--------------------------------------|
| `/register`             | POST   | ❌   | Register a new user                  |
| `/token`                | POST   | ❌   | Authenticate and get JWT             |
| `/upload_documents`     | POST   | ✅   | Upload and process PDF/image files   |
| `/add_url`              | POST   | ❌   | Ingest and store content from a URL  |
| `/preview/{filename}`   | GET    | ❌   | View extracted text from a file      |
| `/documents/{filename}` | DELETE | ❌   | Delete a document                    |
| `/ask`                  | POST   | ❌   | Ask a question on uploaded content   |

---

## 🤖 Tech Stack

| Component     | Tool                     |
|---------------|--------------------------|
| API Framework | FastAPI                  |
| OCR           | Tesseract (pytesseract)  |
| Embeddings    | HuggingFace Transformers |
| Vector Store  | ChromaDB                 |
| Auth          | OAuth2 + JWT             |
| LLM           | FLAN-T5 via LangChain    |

---

## 🗃️ Project Structure

```bash
.
├── main.py                # FastAPI application
├── pdf_handling.py        # PDF/image to text logic
├── auth_handling.py       # Auth routes and token logic
├── chroma_db/             # Vector DB persistent store
├── documents/             # Saved text from uploads
└── README.md
```

---

## ✍️ Author

Made with ❤️ by [Pujan Pokhrel]  
📧 pujan@pokhrel.org

---

## 📜 License

MIT License
