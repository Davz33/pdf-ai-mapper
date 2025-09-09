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
import datetime

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
    category_types: Optional[List[str]] = None  # New field for filtering by category type
    keywords: Optional[List[str]] = None  # New field for filtering by keywords

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
                
                # Ensure structured categories are generated
                if 'structured_categories' not in document_processor.document_index or not document_processor.document_index['structured_categories']:
                    logger.info("Generating structured categories")
                    document_processor.generate_structured_categories()
                
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
        
        document_processor._mark_for_save()
        
        # Flush any pending saves after recategorization
        document_processor.flush_pending_saves()
        
        logger.info(f"Auto-recategorization completed: {updated_count} of {doc_count} documents updated")
        logger.info(f"Current categories: {document_processor.document_index['categories']}")
        
        # Log structured categories
        if 'structured_categories' in document_processor.document_index:
            logger.info(f"Current structured categories: {len(document_processor.document_index['structured_categories'])} categories")
    except Exception as e:
        logger.error(f"Error during auto-recategorization: {str(e)}")
        logger.error(traceback.format_exc())

# Function to process document in background
def process_document_background(file_path, file_name, doc_id):
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
        processed_doc_id, categories = document_processor.process(file_path)
        
        # Check if this is a duplicate document (IDs will be different)
        is_duplicate = processed_doc_id != doc_id
        if is_duplicate:
            logger.info(f"Duplicate document detected. Original ID: {processed_doc_id}, Upload ID: {doc_id}")
        
        # Check the results
        if categories and categories[0].startswith("Error:"):
            logger.error(f"Error processing document: {categories[0]}")
        else:
            logger.info(f"Background processing completed successfully for file: {file_name}")
            logger.info(f"Document ID: {processed_doc_id}, Categories: {categories}")
            
        # Automatically recategorize all documents
        recategorize_all_documents()
        
        # Flush any pending saves after processing
        document_processor.flush_pending_saves()
            
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
            args=(file_path, file.filename, doc_id)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Background processing started for {file.filename}, returning response")
        
        # Return success response immediately
        return {
            "status": "success", 
            "message": "File uploaded successfully and processing started (categorization will happen automatically, duplicates will be detected)", 
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
        logger.info(f"Search query: {search_query.query}, categories: {search_query.categories}, " +
                   f"category_types: {search_query.category_types}, keywords: {search_query.keywords}")
        
        # If category_types or keywords are provided, convert them to category filters
        categories = search_query.categories or []
        
        # If structured category filters are provided, find matching categories
        if search_query.category_types or search_query.keywords:
            # Get structured categories if they exist
            structured_categories = document_processor.document_index.get("structured_categories", [])
            
            if structured_categories:
                for cat in structured_categories:
                    # Filter by category type
                    if search_query.category_types and cat["type"] in search_query.category_types:
                        categories.append(cat["display_name"])
                        continue
                        
                    # Filter by keywords
                    if search_query.keywords and any(kw in cat["keywords"] for kw in search_query.keywords):
                        categories.append(cat["display_name"])
        
        # Remove duplicates
        categories = list(set(categories))
        
        results = search_engine.search(
            search_query.query, 
            categories=categories if categories else None
        )
        
        # Get available category types and keywords for filtering
        available_filters = {
            "category_types": [],
            "keywords": []
        }
        
        structured_categories = document_processor.document_index.get("structured_categories", [])
        if structured_categories:
            # Extract unique category types
            available_filters["category_types"] = sorted(list(set(cat["type"] for cat in structured_categories)))
            
            # Extract unique keywords
            all_keywords = []
            for cat in structured_categories:
                all_keywords.extend(cat["keywords"])
            available_filters["keywords"] = sorted(list(set(all_keywords)))
        
        logger.info(f"Search completed, found {len(results)} results")
        return {
            "results": results,
            "available_filters": available_filters
        }
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/categories/")
async def get_categories():
    """Get all available document categories"""
    try:
        logger.info("Retrieving categories")
        
        # Get structured categories if they exist
        structured_categories = document_processor.document_index.get("structured_categories", [])
        
        # If structured categories don't exist, generate them
        if not structured_categories:
            logger.info("No structured categories found, generating them")
            structured_categories = document_processor.generate_structured_categories()
        
        logger.info(f"Retrieved {len(structured_categories)} structured categories")
        
        # If categories is empty, add a default category
        if not structured_categories:
            logger.info("No categories found, returning default")
            return {
                "structured_categories": [
                    {
                        "id": "cat-001",
                        "type": "Uncategorized",
                        "keywords": [],
                        "display_name": "Uncategorized",
                        "created_at": datetime.datetime.now().isoformat()
                    }
                ]
            }
            
        return {"structured_categories": structured_categories}
    except Exception as e:
        logger.error(f"Error retrieving categories: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving categories: {str(e)}")

@app.post("/recategorize/")
async def recategorize():
    """Manually trigger recategorization of all documents"""
    try:
        logger.info("Manual recategorization requested")
        
        # First, clean up any duplicates
        duplicates_removed = document_processor.clean_up_duplicates()
        logger.info(f"Removed {duplicates_removed} duplicate documents")
        
        # Then recategorize all documents
        recategorize_all_documents()
        
        # Ensure structured categories exist
        structured_categories = document_processor.document_index.get("structured_categories", [])
        if not structured_categories:
            structured_categories = document_processor.generate_structured_categories()
        
        return {
            "status": "success", 
            "message": f"Recategorized {len(document_processor.document_index['documents'])} of {len(document_processor.document_index['documents'])} documents",
            "structured_categories": structured_categories
        }
    except Exception as e:
        logger.error(f"Error in recategorization: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in recategorization: {str(e)}")

@app.post("/recategorize-with-clusters/")
async def recategorize_with_clusters(clusters: int = Query(8, ge=2, le=20)):
    """Manually trigger recategorization with a custom number of clusters"""
    try:
        logger.info(f"Manual recategorization with {clusters} clusters requested")
        
        # First, clean up any duplicates
        duplicates_removed = document_processor.clean_up_duplicates()
        logger.info(f"Removed {duplicates_removed} duplicate documents")
        
        # Get all documents
        documents = document_processor.document_index["documents"]
        doc_count = len(documents)
        
        if doc_count == 0:
            logger.warning("No documents to recategorize")
            return {
                "status": "warning",
                "message": "No documents to recategorize",
                "structured_categories": []
            }
        
        # Check if we have enough documents for the requested number of clusters
        adjusted_clusters = clusters
        adjustment_message = ""
        
        if doc_count < 5:
            logger.warning(f"Not enough documents for clustering ({doc_count}/5)")
            # Generate structured categories if they don't exist
            structured_categories = document_processor.document_index.get("structured_categories", [])
            if not structured_categories:
                structured_categories = document_processor.generate_structured_categories()
            return {
                "status": "warning",
                "message": f"Not enough documents for clustering (need at least 5, have {doc_count})",
                "structured_categories": structured_categories
            }
        
        if doc_count < clusters:
            adjusted_clusters = doc_count
            adjustment_message = f" (adjusted from {clusters} due to document count)"
            logger.info(f"Adjusted clusters from {clusters} to {adjusted_clusters} due to document count")
        
        # Get all preprocessed texts
        all_texts = [doc["preprocessed_text"] for doc in documents.values()]
        
        # Create a new model with the specified number of clusters
        document_processor.model = KMeans(n_clusters=adjusted_clusters, random_state=42)
        
        # Re-fit the vectorizer and model
        try:
            logger.info(f"Fitting vectorizer and model with {adjusted_clusters} clusters")
            text_vectors = document_processor.vectorizer.fit_transform(all_texts)
            document_processor.model.fit(text_vectors)
            
            # Generate new category names
            document_processor._generate_category_names()
            logger.info(f"Generated {len(document_processor.document_index['categories'])} categories")
            
            # Save the updated model and vectorizer
            with open(document_processor.model_file, 'wb') as f:
                pickle.dump(document_processor.model, f)
            with open(document_processor.vectorizer_file, 'wb') as f:
                pickle.dump(document_processor.vectorizer, f)
                
            # Recategorize all documents
            recategorize_all_documents()
            
            # Ensure structured categories exist
            structured_categories = document_processor.document_index.get("structured_categories", [])
            if not structured_categories:
                structured_categories = document_processor.generate_structured_categories()
            
            return {
                "status": "success",
                "message": f"All documents recategorized with {adjusted_clusters} clusters{adjustment_message} (removed {duplicates_removed} duplicates)",
                "structured_categories": structured_categories
            }
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error fitting model: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in recategorization: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in recategorization: {str(e)}")

# Add a status endpoint to check processing status
@app.get("/status/")
async def get_status():
    """Get the processing status of all documents"""
    try:
        logger.info("Retrieving document status")
        
        documents = document_processor.document_index["documents"]
        
        # Create a response with document status
        response = {
            "status": "success",
            "document_count": len(documents),
            "documents": []
        }
        
        # Add structured categories if they exist
        if "structured_categories" in document_processor.document_index:
            response["structured_categories"] = document_processor.document_index["structured_categories"]
        
        # Add each document's status
        for doc_id, doc_info in documents.items():
            response["documents"].append({
                "id": doc_id,
                "filename": doc_info.get("filename", "Unknown"),
                "status": "processed" if "categories" in doc_info and doc_info["categories"] != ["Processing"] else "processing",
                "categories": doc_info.get("categories", ["Processing"])
            })
        
        return response
    except Exception as e:
        logger.error(f"Error retrieving document status: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving document status: {str(e)}")

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/cleanup-duplicates/")
async def cleanup_duplicates():
    """Remove duplicate documents from the index"""
    try:
        logger.info("Duplicate cleanup requested")
        
        # Clean up duplicates
        removed_count = document_processor.clean_up_duplicates()
        
        # Get the current document count
        document_count = len(document_processor.document_index["documents"])
        
        logger.info(f"Removed {removed_count} duplicate documents, {document_count} documents remaining")
        
        return {
            "status": "success",
            "message": f"Removed {removed_count} duplicate documents",
            "document_count": document_count
        }
    except Exception as e:
        logger.error(f"Error cleaning up duplicates: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error cleaning up duplicates: {str(e)}")

@app.post("/generate-structured-categories/")
async def generate_structured_categories():
    """Generate structured categories from existing categories"""
    try:
        logger.info("Structured categories generation requested")
        
        # Generate structured categories
        structured_categories = document_processor.generate_structured_categories()
        
        logger.info(f"Generated {len(structured_categories)} structured categories")
        
        return {
            "status": "success",
            "message": f"Generated {len(structured_categories)} structured categories",
            "structured_categories": structured_categories
        }
    except Exception as e:
        logger.error(f"Error generating structured categories: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating structured categories: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting PDF AI Mapper application")
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
