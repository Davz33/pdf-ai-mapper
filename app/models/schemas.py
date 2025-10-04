"""
Pydantic models for API request/response schemas.

2025 best practices applied:
- Strongly-typed response models (avoid plain dicts)
- Use Field default_factory for mutable defaults
- Provide JSON Schema examples via model_config
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class SearchQuery(BaseModel):
    """Search query model with optional filters."""
    query: str
    categories: Optional[List[str]] = None
    category_types: Optional[List[str]] = None
    keywords: Optional[List[str]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "AI research",
                    "categories": ["Report"],
                    "category_types": ["Research"],
                    "keywords": ["machine learning", "deep learning"],
                }
            ]
        }
    )


class StructuredCategory(BaseModel):
    """Structured category definition with metadata."""
    id: str
    type: str
    keywords: List[str] = Field(default_factory=list)
    display_name: str
    created_at: str


class AvailableFilters(BaseModel):
    """Available filter facets for subsequent searches."""
    category_types: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Response model for file uploads."""
    status: str
    message: str
    document_id: str
    categories: List[str]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "success",
                    "message": "File uploaded successfully and processing started (categorization will happen automatically, duplicates will be detected)",
                    "document_id": "a1b2c3d4-e5f6-7890-1234-56789abcdef0",
                    "categories": ["Processing"],
                }
            ]
        }
    )


class SearchResultItem(BaseModel):
    """One search result item."""
    document_id: str
    filename: str
    categories: List[str] = Field(default_factory=list)
    structured_categories: Optional[List[StructuredCategory]] = None
    score: int
    snippet: str


class SearchResponse(BaseModel):
    """Response model for search results."""
    results: List[SearchResultItem] = Field(default_factory=list)
    available_filters: AvailableFilters

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "results": [
                        {
                            "document_id": "a1b2c3",
                            "filename": "paper.pdf",
                            "categories": ["Research: machine learning, optimization"],
                            "structured_categories": [
                                {
                                    "id": "cat-001",
                                    "type": "Research",
                                    "keywords": ["machine learning", "optimization"],
                                    "display_name": "Research: machine learning, optimization",
                                    "created_at": "2025-01-01T00:00:00Z",
                                }
                            ],
                            "score": 987,
                            "snippet": "...matching text snippet with highlighted search terms...",
                        }
                    ],
                    "available_filters": {
                        "category_types": ["Research", "Analysis"],
                        "keywords": ["machine learning", "optimization"],
                    },
                }
            ]
        }
    )


class StatusDocument(BaseModel):
    """Status entry for a processed document."""
    id: str
    filename: str
    status: str
    categories: List[str] = Field(default_factory=list)


class StatusResponse(BaseModel):
    """Response model for system status."""
    status: str
    document_count: int
    documents: List[StatusDocument] = Field(default_factory=list)
    structured_categories: Optional[List[StructuredCategory]] = None


class CategoryResponse(BaseModel):
    """Response model for categories."""
    structured_categories: List[StructuredCategory] = Field(default_factory=list)


class RecategorizeResponse(BaseModel):
    """Response model for recategorization."""
    status: str
    message: str
    structured_categories: List[StructuredCategory] = Field(default_factory=list)


class CleanupDuplicatesResponse(BaseModel):
    """Response model for duplicate cleanup."""
    status: str
    message: str
    document_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str


class ErrorResponse(BaseModel):
    """Standard error response model (for HTTPException bodies)."""
    detail: str