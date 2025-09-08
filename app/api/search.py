"""
Search API endpoints for document search functionality.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ..models.schemas import SearchQuery, SearchResponse
from ..services.document_service import DocumentService
from search_engine import SearchEngine
from logger import setup_logger

router = APIRouter()
logger = setup_logger("search-api")


@router.post("/search/", response_model=SearchResponse)
async def search_documents(search_query: SearchQuery) -> Dict[str, Any]:
    """Search through processed documents."""
    try:
        logger.info(f"Search query: {search_query.query}, categories: {search_query.categories}, " +
                   f"category_types: {search_query.category_types}, keywords: {search_query.keywords}")
        
        document_service = DocumentService()
        processor = document_service.get_processor()
        search_engine = SearchEngine()
        
        # Convert structured filters to categories
        categories = search_query.categories or []
        
        if search_query.category_types or search_query.keywords:
            structured_categories = processor.document_index.get("structured_categories", [])
            
            if structured_categories:
                for cat in structured_categories:
                    if search_query.category_types and cat["type"] in search_query.category_types:
                        categories.append(cat["display_name"])
                        continue
                        
                    if search_query.keywords and any(kw in cat["keywords"] for kw in search_query.keywords):
                        categories.append(cat["display_name"])
        
        categories = list(set(categories))
        
        results = search_engine.search(
            search_query.query, 
            categories=categories if categories else None
        )
        
        # Get available filters
        available_filters = _get_available_filters(processor)
        
        logger.info(f"Search completed, found {len(results)} results")
        return {
            "results": results,
            "available_filters": available_filters
        }
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


def _get_available_filters(processor) -> Dict[str, list]:
    """Get available category types and keywords for filtering."""
    available_filters = {
        "category_types": [],
        "keywords": []
    }
    
    structured_categories = processor.document_index.get("structured_categories", [])
    if structured_categories:
        available_filters["category_types"] = sorted(list(set(cat["type"] for cat in structured_categories)))
        
        all_keywords = []
        for cat in structured_categories:
            all_keywords.extend(cat["keywords"])
        available_filters["keywords"] = sorted(list(set(all_keywords)))
    
    return available_filters