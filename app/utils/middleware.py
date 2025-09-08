"""
Custom middleware for the FastAPI application.
"""
import time
import traceback
from fastapi import Request
from logger import setup_logger

logger = setup_logger("middleware")


async def log_requests_middleware(request: Request, call_next):
    """Middleware to log requests and responses."""
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