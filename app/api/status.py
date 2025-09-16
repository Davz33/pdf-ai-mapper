"""
Status API endpoints for system monitoring.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ..models.schemas import StatusResponse, HealthResponse
from ..services.document_service import DocumentService
from ..utils.logger import setup_logger

router = APIRouter()
logger = setup_logger("status-api")


@router.get("/status/", response_model=StatusResponse)
async def get_status() -> Dict[str, Any]:
    """Get the processing status of all documents."""
    try:
        logger.info("Retrieving document status")
        
        document_service = DocumentService()
        processor = document_service.get_processor()
        documents = processor.document_index["documents"]
        
        response = {
            "status": "success",
            "document_count": len(documents),
            "documents": []
        }
        
        # Add structured categories if they exist
        if "structured_categories" in processor.document_index:
            response["structured_categories"] = processor.document_index["structured_categories"]
        
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
        raise HTTPException(status_code=500, detail=f"Error retrieving document status: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}