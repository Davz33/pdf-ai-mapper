"""
Main FastAPI application with modular structure.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import upload, search, categories, status
from .utils.middleware import log_requests_middleware
from logger import setup_logger

# Set up logger
logger = setup_logger("pdf-ai-mapper")

# Create FastAPI app
app = FastAPI(title="Document AI Mapper")

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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