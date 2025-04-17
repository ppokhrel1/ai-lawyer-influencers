from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional

# Security and OAuth imports
from jose import JWTError, jwt
from passlib.context import CryptContext
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware

# LangChain and Transformers imports
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA

# Load environment variables
dotenv_path = os.getenv('DOTENV_PATH', '.env')
load_dotenv(dotenv_path)

# FastAPI setup
app = FastAPI(title="NDA Manager RAG API")

# Session middleware for OAuth
app.add_middleware(SessionMiddleware, secret_key=os.getenv('SECRET_KEY', 'supersecretkey'))

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv('CORS_ORIGINS', '*')],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    response = await call_next(request)
    process_time = (datetime.utcnow() - start_time).total_seconds()
    print(f"{request.method} {request.url.path} completed in {process_time}s")
    return response

# ------------------
# Auth & JWT setup
# ------------------
SECRET_KEY = os.getenv('SECRET_KEY', 'supersecretkey')
ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Utility JWT functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub: str = payload.get("sub")
        if sub is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return sub  # returns string like "instagram_{user_id}"

# ---------------------------------
# OAuth client configuration
# ---------------------------------
oauth = OAuth()
oauth.register(
    name='instagram',
    client_id=os.getenv('INSTAGRAM_CLIENT_ID'),
    client_secret=os.getenv('INSTAGRAM_CLIENT_SECRET'),
    authorize_url='https://api.instagram.com/oauth/authorize',
    access_token_url='https://api.instagram.com/oauth/access_token',
    client_kwargs={'scope': 'user_profile'},
)

oauth.register(
    name='twitter',
    client_id=os.getenv('TWITTER_CLIENT_ID'),
    client_secret=os.getenv('TWITTER_CLIENT_SECRET'),
    request_token_url='https://api.twitter.com/oauth/request_token',
    authorize_url='https://api.twitter.com/oauth/authenticate',
    access_token_url='https://api.twitter.com/oauth/access_token',
)

oauth.register(
    name='tiktok',
    client_id=os.getenv('TIKTOK_CLIENT_ID'),
    client_secret=os.getenv('TIKTOK_CLIENT_SECRET'),
    authorize_url='https://open-api.tiktok.com/platform/oauth/connect',
    access_token_url='https://open-api.tiktok.com/oauth/access_token',
    client_kwargs={'scope': 'user.info.basic'},
)

# -----------------------------
# Custom LLM & RAG Setup
# -----------------------------
generator = pipeline(
    "text-generation", 
    model=os.getenv('HF_LLM_MODEL', 'bigscience/bloom-560m'),
    trust_remote_code=False,
    device_map='auto',
    max_length=512
)
llm = HuggingFacePipeline(pipeline=generator)
embed_model = HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBED_MODEL', 'all-MiniLM-L6-v2'))
vectordb = Chroma(persist_directory=os.getenv('CHROMA_DB_PATH', 'chroma_db'), embedding_function=embed_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

# -------------------
# Request Models
# -------------------
class GenerateRequest(BaseModel):
    party_a: str
    party_b: str
    start_date: str
    end_date: str
    clauses: list[str]

class QueryRequest(BaseModel):
    question: str

# ------------------------------------------------
# Social Login & Callback Endpoints
# ------------------------------------------------
@app.get("/login/{provider}")
async def login_social(request: Request, provider: str):
    if provider not in oauth.clients:
        raise HTTPException(status_code=404, detail="Unknown provider")
    redirect_uri = request.url_for('auth_callback', provider=provider)
    return await oauth.create_client(provider).authorize_redirect(request, redirect_uri)

@app.get("/auth/{provider}/callback")
async def auth_callback(request: Request, provider: str):
    try:
        token = await oauth.create_client(provider).authorize_access_token(request)
    except OAuthError:
        raise HTTPException(status_code=400, detail="OAuth authentication failed")
    # Fetch user info based on provider
    if provider == 'instagram':
        resp = await oauth.instagram.get('https://graph.instagram.com/me?fields=id,username', token=token)
        profile = resp.json()
        sub = f"instagram_{profile['id']}"
    elif provider == 'twitter':
        resp = await oauth.twitter.get('https://api.twitter.com/1.1/account/verify_credentials.json', token=token)
        profile = resp.json()
        sub = f"twitter_{profile['id_str']}"
    else:  # tiktok
        resp = await oauth.tiktok.get('https://open-api.tiktok.com/oauth/userinfo/', token=token)
        profile = resp.json().get('data', {})
        sub = f"tiktok_{profile.get('open_id')}"
    access_token = create_access_token(data={"sub": sub})
    return {"access_token": access_token, "token_type": "bearer"}

# ------------------------------------------------
# Protected RAG & Contract Endpoints
# ------------------------------------------------
@app.post("/upload_contract")
async def upload_contract(
    file: UploadFile = File(...), current_user: str = Depends(get_current_user)
):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files supported")
    text = (await file.read()).decode('utf-8')
    vectordb.add_documents([Document(page_content=text)])
    vectordb.persist()
    return {"message": "Document indexed for RAG.", "user": current_user}

@app.post("/generate_contract")
def generate_contract(
    req: GenerateRequest, current_user: str = Depends(get_current_user)
):
    template = (
        f"NON-DISCLOSURE AGREEMENT\nThis Agreement is made between {req.party_a} and {req.party_b}, "
        f"effective from {req.start_date} to {req.end_date}.\n"
    )
    for c in req.clauses:
        template += f"\n- {c.capitalize()} clause placeholder."
    return {"contract": template, "user": current_user}

@app.post("/extract_clauses")
async def extract_clauses(
    data: QueryRequest, current_user: str = Depends(get_current_user)
):
    prompt = f"Extract all contract clauses and return titles and text: {data.question}"
    return {"clauses": qa_chain.run(prompt), "user": current_user}

@app.post("/summarize")
async def summarize(
    data: QueryRequest, current_user: str = Depends(get_current_user)
):
    prompt = f"Summarize the following contract: {data.question}"
    return {"summary": qa_chain.run(prompt), "user": current_user}

@app.post("/analyze_risk")
async def analyze_risk(
    data: QueryRequest, current_user: str = Depends(get_current_user)
):
    prompt = (
        f"Analyze this contract for potential risks and compliance issues. "
        f"List issues with severity and suggestions: {data.question}"
    )
    return {"risk_report": qa_chain.run(prompt), "user": current_user}

@app.post("/query")
async def query_contract(
    data: QueryRequest, current_user: str = Depends(get_current_user)
):
    return {"answer": qa_chain.run(data.question), "user": current_user}

# To run: uvicorn main:app --reload

