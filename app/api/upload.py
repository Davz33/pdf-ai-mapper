"""
Upload API endpoints for file handling.
"""
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any

from ..models.schemas import UploadResponse
from ..services.document_service import DocumentService
from logger import setup_logger

router = APIRouter()
logger = setup_logger("upload-api")


@router.post("/upload/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and process a PDF or image file."""
    logger.info(f"Upload request received for file: {file.filename if file and file.filename else 'unknown'}")
    
    if not file or not file.filename:
        logger.error("No file provided in the request")
        raise HTTPException(status_code=400, detail="No file provided")
    
    doc_id = str(uuid.uuid4())
    logger.info(f"Assigned ID: {doc_id} to file: {file.filename}")
    
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        logger.warning(f"Unsupported file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF and image files are supported")
    
    try:
        file_path = os.path.join("uploads", file.filename)
        
        # Save file in chunks to avoid memory issues
        with open(file_path, "wb") as f:
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        
        logger.info(f"File saved successfully to {file_path}")
        
        # Start background processing
        document_service = DocumentService()
        document_service.start_background_processing(file_path, file.filename, doc_id)
        
        return {
            "status": "success", 
            "message": "File uploaded successfully and processing started (categorization will happen automatically, duplicates will be detected)", 
            "document_id": doc_id,
            "categories": ["Processing"]
        }
            
    except Exception as e:
        logger.error(f"Error handling file upload {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")