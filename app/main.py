"""
Main FastAPI application with modular structure.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import upload, search, categories, status
from .utils.middleware import log_requests_middleware
from .utils.logger import setup_logger
from .core.config import get_settings, create_directories

# Get settings
settings = get_settings()

# Create necessary directories
create_directories()

# Set up logger
logger = setup_logger("pdf-ai-mapper", log_file=settings.log_file)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    description=(
        "PDF and Image AI Mapper with OCR, automatic categorization, duplicate handling, and search.\n\n"
        "This service provides endpoints to upload and process documents, list categories,\n"
        "perform searches with filters, and inspect processing status."
    ),
    contact={
        "name": "PDF AI Mapper Maintainers",
        "url": "https://github.com/Davvz33/pdf-ai-mapper",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {"name": "upload", "description": "Upload and process PDF or image files."},
        {"name": "search", "description": "Search processed documents with optional filters."},
        {"name": "categories", "description": "Manage and view document categories."},
        {"name": "status", "description": "Health and processing status endpoints."},
    ],
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(log_requests_middleware)

# Include API routers
app.include_router(upload.router, tags=["upload"])
app.include_router(search.router, tags=["search"])
app.include_router(categories.router, tags=["categories"])
app.include_router(status.router, tags=["status"])

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PDF AI Mapper application")
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)