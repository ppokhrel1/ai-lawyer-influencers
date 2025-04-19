from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials, OAuth2AuthorizationCodeBearer
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import databases
import sqlalchemy

# Configuration
DATABASE_URL = "sqlite:///./test.db"
SECRET_KEY = "your-secret-key-keep-it-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer(auto_error=False)

# Database setup
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("username", sqlalchemy.String, unique=True),
    sqlalchemy.Column("email", sqlalchemy.String, unique=True),
    sqlalchemy.Column("hashed_password", sqlalchemy.String),
    sqlalchemy.Column("disabled", sqlalchemy.Boolean, default=False),
)

class User(BaseModel):
    username: str
    hashed_password: str
    disabled: bool = False

class UserInDB(User):
    id: int

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Models
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    hashed_password: str
    disabled: bool

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Database connection
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# Security functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_user(username: str):
    query = users.select().where(users.c.username == username)
    return await database.fetch_one(query)

async def authenticate_user(username: str, password: str):
    user = await get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = await get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Routes
@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    # Check if username or email already exists
    existing_user = await database.fetch_one(
        users.select().where(
            (users.c.username == user.username) | 
            (users.c.email == user.email)
        )
    )
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username or email already registered"
        )
    
    hashed_password = get_password_hash(user.password)
    query = users.insert().values(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    await database.execute(query)
    return {"message": "User created successfully"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserBase)
async def read_users_me(current_user: UserBase = Depends(get_current_user)):
    return current_user


from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from typing import Optional, Union

# Security schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
security = HTTPBearer(auto_error=False)

async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Union[User, None]:
    """
    Unified authentication handler that supports:
    - Both header types (Bearer token and Authorization header)
    - Optional authentication
    - Detailed error tracking
    - Secure token validation
    
    Returns:
        User object if valid credentials provided
        None if no credentials or invalid credentials (with error in request.state)
    """
    # Get token from either source
    auth_token = None
    if credentials:
        auth_token = credentials.credentials
    elif token:
        auth_token = token
    
    # No token provided
    if not auth_token:
        return None
    
    # Token validation
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify token structure
        if auth_token.count(".") != 2:
            raise JWTError("Invalid token structure")
            
        payload = jwt.decode(
            auth_token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"verify_exp": True}
        )
        
        if username := payload.get("sub"):
            if user := await get_user(username=username):
                return user
    
    except jwt.ExpiredSignatureError:
        request.state.auth_error = "Token expired"
    except JWTError as e:
        request.state.auth_error = f"Invalid token: {str(e)}"
    except Exception as e:
        request.state.auth_error = f"Authentication error: {str(e)}"
    
    return None
    
# Install dependencies
# pip install passlib python-jose[cryptography] databases sqlalchemy