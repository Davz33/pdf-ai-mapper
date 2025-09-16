"""
Pytest configuration and fixtures for PDF AI Mapper tests.
"""

import os
import sys
import tempfile
import shutil
import time
import threading
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.document_processor import DocumentProcessor
from app.core.config import Settings, get_settings
from app.main import app


@pytest.fixture
def test_environment():
    """Fixture for setting up isolated test environment."""
    test_dir = tempfile.mkdtemp(prefix="pdf_mapper_test_")
    original_dir = os.getcwd()
    
    try:
        os.chdir(test_dir)
        yield test_dir
    finally:
        os.chdir(original_dir)
        shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def app_test_environment(test_environment):
    """Fixture for setting up app test environment with copied files."""
    # Copy necessary app files to test environment
    app_files = [
        "app/__init__.py",
        "app/main.py",
        "app/models/schemas.py",
        "app/services/document_service.py",
        "app/utils/logger.py",
        "app/utils/middleware.py",
        "app/core/__init__.py",
        "app/core/config.py",
        "app/core/exceptions.py",
        "app/core/document_processor.py",
        "app/core/text_processing/text_preprocessor.py",
        "app/core/categorization/category_manager.py",
        "app/core/storage/document_storage.py",
        "app/core/text_extraction/extractor_factory.py",
        "app/core/text_extraction/pdf_extractor.py",
        "app/core/text_extraction/image_extractor.py",
        "app/api/__init__.py",
        "app/api/upload.py",
        "app/api/search.py",
        "app/api/categories.py",
        "app/api/status.py",
    ]
    
    project_root = Path(__file__).parent.parent
    
    for file_path in app_files:
        src = project_root / file_path
        if src.exists():
            dst = Path(test_environment) / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    
    yield test_environment


@pytest.fixture
def document_processor(app_test_environment):
    """Fixture for DocumentProcessor in isolated environment."""
    with patch('app.core.document_processor.DocumentProcessor') as mock_processor:
        # Create a real instance but with test directory
        processor = DocumentProcessor(processed_dir=os.path.join(app_test_environment, "processed_data"))
        mock_processor.return_value = processor
        yield processor


@pytest.fixture
def mock_server():
    """Fixture for mocking server responses"""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        
        # Configure mock responses for different endpoints
        def mock_get_response(url, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            if '/status' in url:
                mock_response.json.return_value = {"documents": []}
            elif '/categories' in url:
                mock_response.json.return_value = {"categories": []}
            else:
                mock_response.json.return_value = {"status": "ok"}
            return mock_response
        
        def mock_post_response(url, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            if '/search' in url:
                mock_response.json.return_value = {"results": []}
            else:
                mock_response.json.return_value = {"status": "success"}
            return mock_response
        
        mock_get.side_effect = mock_get_response
        mock_post.side_effect = mock_post_response
        
        yield {
            'mock_get': mock_get,
            'mock_post': mock_post
        }


@pytest.fixture
def test_client():
    """Fixture for FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def test_settings():
    """Fixture for test settings."""
    return Settings(
        environment="testing",
        debug=True,
        processed_dir="test_processed_data",
        upload_dir="test_uploads",
        log_level="DEBUG"
    )


@pytest.fixture
def sample_pdf_file():
    """Fixture for sample PDF file."""
    # Create a simple test PDF content
    test_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test PDF Content) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(test_content)
        f.flush()
        yield f.name
    
    # Cleanup
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


@pytest.fixture
def sample_image_file():
    """Fixture for sample image file."""
    from PIL import Image
    
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='white')
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name, 'PNG')
        f.flush()
        yield f.name
    
    # Cleanup
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Auto-cleanup fixture for test files."""
    yield
    # Cleanup any remaining test files
    test_dirs = ["test_processed_data", "test_uploads", "temp"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)
