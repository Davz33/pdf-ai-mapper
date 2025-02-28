import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import traceback
import time

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
    
    # Record request body for debugging if needed
    try:
        body = await request.body()
        if body:
            logger.debug(f"Request body: {body.decode()}")
    except Exception:
        pass
    
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

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a PDF or image file"""
    # Check file type
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        logger.warning(f"Unsupported file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF and image files are supported")
    
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Save uploaded file
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        logger.info(f"File saved to {file_path}, starting processing")
        
        # Process the document
        doc_id, categories = document_processor.process(file_path)
        
        logger.info(f"File processed successfully: {doc_id}, categories: {categories}")
        
        return {
            "status": "success", 
            "message": "File processed successfully", 
            "document_id": doc_id,
            "categories": categories
        }
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

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
        logger.info(f"Retrieved {len(categories)} categories")
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error retrieving categories: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving categories: {str(e)}")

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting PDF AI Mapper application")
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)