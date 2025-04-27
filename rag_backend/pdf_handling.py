from fastapi import UploadFile, File
from typing import List
import pytesseract
from PIL import Image
import pdfplumber
import io
import os
from auth_handling import *


# Add to dependencies
#pip install pdfplumber pytesseract pillow python-multipart

# Configure Tesseract path (if needed)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux/Mac
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise ValueError(f"PDF processing error: {str(e)}")
    return text

def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

@app.post("/upload_documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user)
):
    processed_chunks = 0
    errors = []
    
    for file in files:
        try:
            contents = await file.read()
            
            if file.content_type == "application/pdf":
                text = extract_text_from_pdf(contents)
            elif file.content_type.startswith("image/"):
                text = extract_text_from_image(contents)
            else:
                errors.append(f"Unsupported format: {file.filename}")
                continue
            
            # Create document and split
            doc = Document(
                page_content=text,
                metadata={"source": file.filename, "type": file.content_type}
            )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=100
            )
            split_docs = text_splitter.split_documents([doc])
            
            # Add to vector store
            vectordb.add_documents(split_docs)
            processed_chunks += len(split_docs)
            
        except Exception as e:
            errors.append(f"Failed to process {file.filename}: {str(e)}")
        finally:
            await file.close()
    
    vectordb.persist()
    
    return {
        "message": f"Processed {processed_chunks} chunks from {len(files)-len(errors)} files",
        "errors": errors
    }

@app.get("/preview/{doc_id}")
async def preview_document(
    doc_id: str,
    current_user: User = Depends(get_current_user)
):
    doc = vectordb.get(doc_id)
    return {
        "content": doc.page_content[:500] + "...",
        "metadata": doc.metadata
    }

@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: User = Depends(get_current_user)
):
    vectordb.delete([doc_id])
    return {"message": "Document deleted"}



class AddUrlRequest(BaseModel):
    url: str

@app.post("/add_url")
async def add_url(request: AddUrlRequest):
    try:
        url = request.url
        
        # Validate URL format
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise HTTPException(status_code=400, detail="Invalid URL format")

        # Restrict to legal domains (optional whitelist)
        legal_domains = [
            "cornell.edu", "case.law", "courtlistener.com", 
            "govinfo.gov", "eur-lex.europa.eu"
        ]
        if not any(domain in parsed_url.netloc for domain in legal_domains):
            raise HTTPException(
                status_code=403, 
                detail="URL domain not in allowed legal sources"
            )

        # Fetch content (async with timeout)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Extract clean text
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements (scripts, styles, etc.)
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()
                
            text = soup.get_text(separator='\n', strip=True)

        # Split document
        doc = Document(
            page_content=text,
            metadata={
                "source": url,
                "domain": parsed_url.netloc,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Better for legal texts
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.split_documents([doc])

        # Store in vector DB
        vectordb.add_documents(split_docs)
        vectordb.persist()

        return {
            "message": f"Added {len(split_docs)} chunks from {url}",
            "metadata": doc.metadata
        }

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Source website returned error: {str(e)}"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to URL: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )


