"""
Categories API endpoints for managing document categories.
"""
import datetime
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any

from ..models.schemas import CategoryResponse, RecategorizeResponse
from ..services.document_service import DocumentService
from ..utils.logger import setup_logger

router = APIRouter()
logger = setup_logger("categories-api")


@router.get("/categories/", response_model=CategoryResponse)
async def get_categories() -> Dict[str, Any]:
    """Get all available document categories."""
    try:
        logger.info("Retrieving categories")
        
        document_service = DocumentService()
        processor = document_service.get_processor()
        
        structured_categories = processor.document_index.get("structured_categories", [])
        
        if not structured_categories:
            logger.info("No structured categories found, generating them")
            structured_categories = processor.generate_structured_categories()
        
        logger.info(f"Retrieved {len(structured_categories)} structured categories")
        
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
        raise HTTPException(status_code=500, detail=f"Error retrieving categories: {str(e)}")


@router.post("/recategorize/", response_model=RecategorizeResponse)
async def recategorize() -> Dict[str, Any]:
    """Manually trigger recategorization of all documents."""
    try:
        logger.info("Manual recategorization requested")
        
        document_service = DocumentService()
        processor = document_service.get_processor()
        
        # Clean up duplicates first
        duplicates_removed = processor.clean_up_duplicates()
        logger.info(f"Removed {duplicates_removed} duplicate documents")
        
        # Recategorize all documents
        document_service.recategorize_all_documents()
        
        # Ensure structured categories exist
        structured_categories = processor.document_index.get("structured_categories", [])
        if not structured_categories:
            structured_categories = processor.generate_structured_categories()
        
        return {
            "status": "success", 
            "message": f"Recategorized {len(processor.document_index['documents'])} documents",
            "structured_categories": structured_categories
        }
    except Exception as e:
        logger.error(f"Error in recategorization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in recategorization: {str(e)}")


@router.post("/recategorize-with-clusters/", response_model=RecategorizeResponse)
async def recategorize_with_clusters(clusters: int = Query(8, ge=2, le=20)) -> Dict[str, Any]:
    """Manually trigger recategorization with a custom number of clusters."""
    try:
        logger.info(f"Manual recategorization with {clusters} clusters requested")
        
        document_service = DocumentService()
        processor = document_service.get_processor()
        
        # Clean up duplicates first
        duplicates_removed = processor.clean_up_duplicates()
        logger.info(f"Removed {duplicates_removed} duplicate documents")
        
        documents = processor.document_index["documents"]
        doc_count = len(documents)
        
        if doc_count == 0:
            logger.warning("No documents to recategorize")
            return {
                "status": "warning",
                "message": "No documents to recategorize",
                "structured_categories": []
            }
        
        # Adjust clusters based on document count
        adjusted_clusters = clusters
        adjustment_message = ""
        
        if doc_count < 5:
            logger.warning(f"Not enough documents for clustering ({doc_count}/5)")
            structured_categories = processor.document_index.get("structured_categories", [])
            if not structured_categories:
                structured_categories = processor.generate_structured_categories()
            return {
                "status": "warning",
                "message": f"Not enough documents for clustering (need at least 5, have {doc_count})",
                "structured_categories": structured_categories
            }
        
        if doc_count < clusters:
            adjusted_clusters = doc_count
            adjustment_message = f" (adjusted from {clusters} due to document count)"
            logger.info(f"Adjusted clusters from {clusters} to {adjusted_clusters} due to document count")
        
        # Create new model with specified clusters
        from sklearn.cluster import KMeans
        processor.model = KMeans(n_clusters=adjusted_clusters, random_state=42)
        
        # Refit model and recategorize
        all_texts = [doc["preprocessed_text"] for doc in documents.values()]
        text_vectors = processor.vectorizer.fit_transform(all_texts)
        processor.model.fit(text_vectors)
        
        processor._generate_category_names()
        logger.info(f"Generated {len(processor.document_index['categories'])} categories")
        
        # Save updated model
        import pickle
        with open(processor.model_file, 'wb') as f:
            pickle.dump(processor.model, f)
        with open(processor.vectorizer_file, 'wb') as f:
            pickle.dump(processor.vectorizer, f)
        
        # Recategorize all documents
        document_service.recategorize_all_documents()
        
        # Ensure structured categories exist
        structured_categories = processor.document_index.get("structured_categories", [])
        if not structured_categories:
            structured_categories = processor.generate_structured_categories()
        
        return {
            "status": "success",
            "message": f"All documents recategorized with {adjusted_clusters} clusters{adjustment_message} (removed {duplicates_removed} duplicates)",
            "structured_categories": structured_categories
        }
            
    except Exception as e:
        logger.error(f"Error in recategorization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in recategorization: {str(e)}")


@router.post("/cleanup-duplicates/")
async def cleanup_duplicates() -> Dict[str, Any]:
    """Remove duplicate documents from the index."""
    try:
        logger.info("Duplicate cleanup requested")
        
        document_service = DocumentService()
        processor = document_service.get_processor()
        
        removed_count = processor.clean_up_duplicates()
        document_count = len(processor.document_index["documents"])
        
        logger.info(f"Removed {removed_count} duplicate documents, {document_count} documents remaining")
        
        return {
            "status": "success",
            "message": f"Removed {removed_count} duplicate documents",
            "document_count": document_count
        }
    except Exception as e:
        logger.error(f"Error cleaning up duplicates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cleaning up duplicates: {str(e)}")


@router.post("/generate-structured-categories/")
async def generate_structured_categories() -> Dict[str, Any]:
    """Generate structured categories from existing categories."""
    try:
        logger.info("Structured categories generation requested")
        
        document_service = DocumentService()
        processor = document_service.get_processor()
        
        structured_categories = processor.generate_structured_categories()
        
        logger.info(f"Generated {len(structured_categories)} structured categories")
        
        return {
            "status": "success",
            "message": f"Generated {len(structured_categories)} structured categories",
            "structured_categories": structured_categories
        }
    except Exception as e:
        logger.error(f"Error generating structured categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating structured categories: {str(e)}")