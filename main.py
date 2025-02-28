import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import traceback
import time
import uuid
import threading

from document_processor import DocumentProcessor
from search_engine import SearchEngine
from logger import setup_logger

# Set up logger
logger = setup_logger("pdf-ai-mapper")

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

app = FastAPI(title="Document AI Mapper")

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware to log requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(time.time())
    logger.info(f"Request {request_id} started: {request.method} {request.url}")
    
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed: {response.status_code} ({process_time:.2f}s)")
        return response
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} error: ({process_time:.2f}s)")
        raise

# Initialize document processor and search engine
document_processor = DocumentProcessor()
search_engine = SearchEngine()

class SearchQuery(BaseModel):
    query: str
    categories: Optional[List[str]] = None

# Function to process document in background
def process_document_background(file_path, file_name):
    try:
        logger.info(f"Background processing started for file: {file_name}")
        doc_id, categories = document_processor.process(file_path)
        logger.info(f"Background processing completed for file: {file_name}, doc_id: {doc_id}, categories: {categories}")
    except Exception as e:
        logger.error(f"Error in background processing for file {file_name}: {str(e)}")
        logger.error(traceback.format_exc())

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a PDF or image file"""
    # Log the start of upload processing
    logger.info(f"Upload request received for file: {file.filename if file and file.filename else 'unknown'}")
    
    # Check if file is provided
    if not file or not file.filename:
        logger.error("No file provided in the request")
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Generate a unique ID
    doc_id = str(uuid.uuid4())
    logger.info(f"Assigned ID: {doc_id} to file: {file.filename}")
    
    # Check file type
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        logger.warning(f"Unsupported file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF and image files are supported")
    
    try:
        # Save uploaded file
        file_path = os.path.join("uploads", file.filename)
        
        # Read file content with a small buffer to avoid memory issues
        with open(file_path, "wb") as f:
            # Read and write in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        
        logger.info(f"File saved successfully to {file_path}")
        
        # Start processing in a background thread and return immediately
        thread = threading.Thread(
            target=process_document_background,
            args=(file_path, file.filename)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Background processing started for {file.filename}, returning response")
        
        # Return success response immediately
        return {
            "status": "success", 
            "message": "File uploaded successfully and processing started", 
            "document_id": doc_id,
            "categories": ["Processing"]
        }
            
    except Exception as e:
        logger.error(f"Error handling file upload {file.filename}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/search/")
async def search_documents(search_query: SearchQuery):
    """Search through processed documents"""
    try:
        logger.info(f"Search query: {search_query.query}, categories: {search_query.categories}")
        
        results = search_engine.search(
            search_query.query, 
            categories=search_query.categories
        )
        
        logger.info(f"Search completed, found {len(results)} results")
        return {"results": results}
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/categories/")
async def get_categories():
    """Get all available document categories"""
    try:
        logger.info("Retrieving categories")
        categories = document_processor.get_categories()
        logger.info(f"Retrieved categories: {categories}")
        
        # If categories is empty, add a default category
        if not categories:
            logger.info("No categories found, returning default")
            return {"categories": ["Uncategorized"]}
            
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error retrieving categories: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving categories: {str(e)}")

# Add a status endpoint to check processing status
@app.get("/status/")
async def get_status():
    """Get the status of document processing"""
    try:
        # Count documents
        doc_count = len(document_processor.document_index["documents"])
        cat_count = len(document_processor.document_index["categories"])
        
        return {
            "status": "healthy",
            "documents_processed": doc_count,
            "categories_count": cat_count,
            "categories": document_processor.document_index["categories"]
        }
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return {"status": "error", "message": str(e)}

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting PDF AI Mapper application")
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)