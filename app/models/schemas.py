"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel
from typing import List, Optional


class SearchQuery(BaseModel):
    """Search query model with optional filters."""
    query: str
    categories: Optional[List[str]] = None
    category_types: Optional[List[str]] = None
    keywords: Optional[List[str]] = None


class UploadResponse(BaseModel):
    """Response model for file uploads."""
    status: str
    message: str
    document_id: str
    categories: List[str]


class SearchResponse(BaseModel):
    """Response model for search results."""
    results: List[dict]
    available_filters: dict


class StatusResponse(BaseModel):
    """Response model for system status."""
    status: str
    document_count: int
    documents: List[dict]
    structured_categories: List[dict] = []


class CategoryResponse(BaseModel):
    """Response model for categories."""
    structured_categories: List[dict]


class RecategorizeResponse(BaseModel):
    """Response model for recategorization."""
    status: str
    message: str
    structured_categories: List[dict] = []


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str