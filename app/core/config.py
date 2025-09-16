"""
Configuration management for PDF AI Mapper.
Uses Pydantic for validation and environment variable handling.
"""

import os
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = Field(default="PDF AI Mapper")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")
    
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)
    
    # CORS
    cors_origins: str = Field(default="http://localhost:3000")
    cors_allow_credentials: bool = Field(default=True)
    
    # File Processing
    max_file_size: int = Field(default=50 * 1024 * 1024)  # 50MB
    allowed_file_types: str = Field(
        default=".pdf,.png,.jpg,.jpeg,.gif,.bmp,.tiff"
    )
    upload_dir: str = Field(default="uploads")
    processed_dir: str = Field(default="processed_data")
    
    # OCR Settings
    tesseract_timeout: int = Field(default=30)
    pdf_timeout: int = Field(default=120)
    max_pages_for_ocr: int = Field(default=5)
    
    # ML/Categorization
    min_documents_for_categorization: int = Field(default=5)
    lda_components: int = Field(default=8)
    max_features: int = Field(default=1000)
    categorization_threshold: float = Field(default=0.1)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")  # json or text
    log_file: Optional[str] = Field(default=None)
    
    # Database (for future use)
    database_url: Optional[str] = Field(default=None)
    
    # Redis (for future use)
    redis_url: Optional[str] = Field(default=None)
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production")
    access_token_expire_minutes: int = Field(default=30)
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)  # seconds
    
    @field_validator("cors_origins", mode="after")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("allowed_file_types", mode="after")
    @classmethod
    def parse_allowed_file_types(cls, v):
        """Parse allowed file types from string or list."""
        if isinstance(v, str):
            return [file_type.strip() for file_type in v.split(",")]
        return v
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value."""
        allowed_envs = ["development", "staging", "production", "testing"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def create_directories(settings_instance: Optional[Settings] = None):
    """Create necessary directories if they don't exist."""
    settings_to_use = settings_instance or settings
    directories = [
        settings_to_use.upload_dir,
        settings_to_use.processed_dir,
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
