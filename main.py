import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import traceback
import time
import uuid
import threading
import json
import pickle
from sklearn.cluster import KMeans

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

# Function to recategorize all documents
def recategorize_all_documents():
    try:
        logger.info("Auto-recategorizing all documents after new upload")
        
        # Get all documents
        documents = document_processor.document_index["documents"]
        doc_count = len(documents)
        
        if doc_count == 0:
            logger.info("No documents to recategorize")
            return
        
        # Check if we have enough documents to regenerate categories
        if doc_count >= 5:
            # Get all preprocessed texts
            all_texts = [doc["preprocessed_text"] for doc in documents.values()]
            
            # Re-fit the vectorizer and model to ensure fresh categorization
            try:
                logger.info("Re-fitting vectorizer and model with all documents")
                text_vectors = document_processor.vectorizer.fit_transform(all_texts)
                document_processor.model.fit(text_vectors)
                
                # Generate new category names
                document_processor._generate_category_names()
                logger.info(f"Regenerated categories: {document_processor.document_index['categories']}")
                
                # Save the updated model and vectorizer
                with open(document_processor.model_file, 'wb') as f:
                    pickle.dump(document_processor.model, f)
                with open(document_processor.vectorizer_file, 'wb') as f:
                    pickle.dump(document_processor.vectorizer, f)
            except Exception as e:
                logger.error(f"Error re-fitting model: {str(e)}")
                logger.error(traceback.format_exc())
            
        # Process each document
        updated_count = 0
        for doc_id, doc in list(documents.items()):
            try:
                # Get preprocessed text
                preprocessed_text = doc["preprocessed_text"]
                
                # Recategorize
                categories = document_processor._categorize_text(preprocessed_text)
                
                # Update document
                documents[doc_id]["categories"] = categories
                updated_count += 1
                
                logger.info(f"Recategorized document {doc_id}: {categories}")
            except Exception as e:
                logger.error(f"Error recategorizing document {doc_id}: {str(e)}")
        
        # Save updated index
        with open(document_processor.index_file, 'w') as f:
            json.dump(document_processor.document_index, f)
        
        logger.info(f"Auto-recategorization completed: {updated_count} of {doc_count} documents updated")
        logger.info(f"Current categories: {document_processor.document_index['categories']}")
    except Exception as e:
        logger.error(f"Error during auto-recategorization: {str(e)}")
        logger.error(traceback.format_exc())

# Function to process document in background
def process_document_background(file_path, file_name):
    try:
        logger.info(f"Background processing started for file: {file_name}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
            
        # Check file size
        file_size = os.path.getsize(file_path)
        logger.info(f"Processing file of size: {file_size} bytes")
        
        # Process the document
        doc_id, categories = document_processor.process(file_path)
        
        # Check the results
        if categories and categories[0].startswith("Error:"):
            logger.error(f"Error processing document: {categories[0]}")
        else:
            logger.info(f"Background processing completed successfully for file: {file_name}")
            logger.info(f"Document ID: {doc_id}, Categories: {categories}")
            
        # Automatically recategorize all documents
        recategorize_all_documents()
            
        # Update status endpoint data
        try:
            # Check if we have categories
            if document_processor.document_index["categories"]:
                logger.info(f"Current categories: {document_processor.document_index['categories']}")
            else:
                logger.warning("No categories found after processing")
                
            # Check document count
            doc_count = len(document_processor.document_index["documents"])
            logger.info(f"Total documents processed: {doc_count}")
        except Exception as e:
            logger.error(f"Error updating status data: {str(e)}")
            logger.error(traceback.format_exc())
            
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
            "message": "File uploaded successfully and processing started (categorization will happen automatically)", 
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

@app.post("/recategorize/")
async def recategorize():
    """Manually trigger recategorization of all documents"""
    try:
        logger.info("Starting manual recategorization of all documents")
        recategorize_all_documents()
        return {"status": "success", "message": "All documents recategorized"}
    except Exception as e:
        logger.error(f"Error during manual recategorization: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during recategorization: {str(e)}")

@app.post("/recategorize-with-clusters/")
async def recategorize_with_clusters(clusters: int = Query(8, ge=2, le=20)):
    """Manually trigger recategorization with a specific number of clusters"""
    try:
        logger.info(f"Starting manual recategorization with {clusters} clusters")
        
        # Get all documents
        documents = document_processor.document_index["documents"]
        doc_count = len(documents)
        
        if doc_count < 5:
            logger.warning(f"Not enough documents for clustering ({doc_count}/5)")
            return {"status": "warning", "message": f"Not enough documents for clustering ({doc_count}/5)"}
        
        # Ensure we don't try to create more clusters than documents
        if clusters > doc_count:
            adjusted_clusters = doc_count
            logger.warning(f"Requested {clusters} clusters but only have {doc_count} documents. Adjusting to {adjusted_clusters} clusters.")
        else:
            adjusted_clusters = clusters
        
        # Get all preprocessed texts
        all_texts = [doc["preprocessed_text"] for doc in documents.values()]
        
        # Create a new model with the specified number of clusters
        try:
            logger.info(f"Creating new model with {adjusted_clusters} clusters")
            document_processor.model = KMeans(n_clusters=adjusted_clusters, random_state=42)
            
            # Fit the vectorizer and model
            text_vectors = document_processor.vectorizer.fit_transform(all_texts)
            document_processor.model.fit(text_vectors)
            
            # Generate new category names
            document_processor._generate_category_names()
            logger.info(f"Regenerated categories with {adjusted_clusters} clusters: {document_processor.document_index['categories']}")
            
            # Save the updated model and vectorizer
            with open(document_processor.model_file, 'wb') as f:
                pickle.dump(document_processor.model, f)
            with open(document_processor.vectorizer_file, 'wb') as f:
                pickle.dump(document_processor.vectorizer, f)
                
            # Recategorize all documents
            updated_count = 0
            for doc_id, doc in list(documents.items()):
                try:
                    # Get preprocessed text
                    preprocessed_text = doc["preprocessed_text"]
                    
                    # Recategorize
                    categories = document_processor._categorize_text(preprocessed_text)
                    
                    # Update document
                    documents[doc_id]["categories"] = categories
                    updated_count += 1
                    
                    logger.info(f"Recategorized document {doc_id}: {categories}")
                except Exception as e:
                    logger.error(f"Error recategorizing document {doc_id}: {str(e)}")
            
            # Save updated index
            with open(document_processor.index_file, 'w') as f:
                json.dump(document_processor.document_index, f)
            
            message = f"All documents recategorized with {adjusted_clusters} clusters"
            if adjusted_clusters != clusters:
                message += f" (adjusted from {clusters} due to document count)"
                
            logger.info(f"Recategorization with {adjusted_clusters} clusters completed: {updated_count} of {doc_count} documents updated")
            
            return {
                "status": "success", 
                "message": message,
                "categories": document_processor.document_index["categories"]
            }
            
        except Exception as e:
            logger.error(f"Error creating model with {adjusted_clusters} clusters: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error creating model: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error during recategorization with clusters: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during recategorization: {str(e)}")

# Add a status endpoint to check processing status
@app.get("/status/")
async def get_status():
    """
    Get the processing status of all documents.
    """
    try:
        logger.info("Retrieving document status information")
        documents = []
        
        # Get all documents from the index
        for doc_id, doc_info in document_processor.document_index["documents"].items():
            status = "processed" if doc_info.get("processed", False) else "processing"
            
            # Check if there was an error during processing
            if doc_info.get("error", False):
                status = "error"
                
            documents.append({
                "document_id": doc_id,
                "filename": doc_info.get("filename", "Unknown"),
                "status": status,
                "categories": doc_info.get("categories", ["Processing"])
            })
            
        logger.info(f"Retrieved status for {len(documents)} documents")
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error retrieving document status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document status: {str(e)}")

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting PDF AI Mapper application")
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)