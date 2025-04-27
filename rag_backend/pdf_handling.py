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


# Update existing add_url endpoint to handle PDF URLs
@app.post("/add_url")
async def add_url(
    url: str,
    current_user: User = Depends(get_current_user)
):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        
        if 'pdf' in content_type.lower() or url.lower().endswith('.pdf'):
            text = extract_text_from_pdf(response.content)
        elif any(img_type in content_type.lower() for img_type in ['image/png', 'image/jpeg', 'image/tiff']):
            text = extract_text_from_image(response.content)
        else:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
        
        doc = Document(page_content=text, metadata={"source": url})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents([doc])
        vectordb.add_documents(split_docs)
        vectordb.persist()
        
        return {"message": f"Added {len(split_docs)} chunks from {url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
