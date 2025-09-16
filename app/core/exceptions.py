"""
Custom exception classes for PDF AI Mapper.
Provides structured error handling with proper error codes and messages.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(Enum):
    """Error codes for different types of failures."""
    
    # File processing errors
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_PROCESSING_FAILED = "FILE_PROCESSING_FAILED"
    FILE_CORRUPTED = "FILE_CORRUPTED"
    
    # OCR errors
    OCR_FAILED = "OCR_FAILED"
    OCR_TIMEOUT = "OCR_TIMEOUT"
    TESSERACT_NOT_FOUND = "TESSERACT_NOT_FOUND"
    
    # PDF processing errors
    PDF_EXTRACTION_FAILED = "PDF_EXTRACTION_FAILED"
    PDF_PASSWORD_PROTECTED = "PDF_PASSWORD_PROTECTED"
    PDF_CORRUPTED = "PDF_CORRUPTED"
    
    # Categorization errors
    CATEGORIZATION_FAILED = "CATEGORIZATION_FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    MODEL_NOT_TRAINED = "MODEL_NOT_TRAINED"
    
    # Storage errors
    STORAGE_ERROR = "STORAGE_ERROR"
    DUPLICATE_DOCUMENT = "DUPLICATE_DOCUMENT"
    INDEX_CORRUPTED = "INDEX_CORRUPTED"
    
    # API errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    
    # Configuration errors
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"


class PDFAIMapperException(Exception):
    """Base exception class for PDF AI Mapper."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details
        }


class FileProcessingError(PDFAIMapperException):
    """Raised when file processing fails."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        file_path: Optional[str] = None,
        file_size: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if file_path:
            details["file_path"] = file_path
        if file_size:
            details["file_size"] = file_size
        kwargs["details"] = details
        super().__init__(message, error_code, **kwargs)


class OCRError(PDFAIMapperException):
    """Raised when OCR processing fails."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        image_path: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if image_path:
            details["image_path"] = image_path
        if timeout:
            details["timeout"] = timeout
        kwargs["details"] = details
        super().__init__(message, error_code, **kwargs)


class PDFProcessingError(PDFAIMapperException):
    """Raised when PDF processing fails."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        pdf_path: Optional[str] = None,
        page_count: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if pdf_path:
            details["pdf_path"] = pdf_path
        if page_count:
            details["page_count"] = page_count
        kwargs["details"] = details
        super().__init__(message, error_code, **kwargs)


class CategorizationError(PDFAIMapperException):
    """Raised when document categorization fails."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        document_id: Optional[str] = None,
        document_count: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if document_id:
            details["document_id"] = document_id
        if document_count:
            details["document_count"] = document_count
        kwargs["details"] = details
        super().__init__(message, error_code, **kwargs)


class StorageError(PDFAIMapperException):
    """Raised when storage operations fail."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        operation: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        if file_path:
            details["file_path"] = file_path
        kwargs["details"] = details
        super().__init__(message, error_code, **kwargs)


class ValidationError(PDFAIMapperException):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        kwargs["details"] = details
        super().__init__(message, error_code, **kwargs)


class ConfigurationError(PDFAIMapperException):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        setting: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if setting:
            details["setting"] = setting
        kwargs["details"] = details
        super().__init__(message, error_code, **kwargs)


class RateLimitError(PDFAIMapperException):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if limit:
            details["limit"] = limit
        if window:
            details["window"] = window
        kwargs["details"] = details
        super().__init__(message, error_code, **kwargs)
