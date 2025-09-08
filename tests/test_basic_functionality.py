"""
Basic functionality tests for the document AI mapper.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch
from app.models.schemas import SearchQuery, UploadResponse
from app.services.document_service import DocumentService


class TestBasicFunctionality:
    """Test basic application functionality."""
    
    def test_search_query_model(self):
        """Test that SearchQuery model works correctly."""
        query = SearchQuery(query="test query")
        assert query.query == "test query"
        assert query.categories is None
        assert query.category_types is None
        assert query.keywords is None
    
    def test_search_query_with_filters(self):
        """Test SearchQuery with filters."""
        query = SearchQuery(
            query="AI research",
            categories=["Document"],
            category_types=["Research"],
            keywords=["machine learning"]
        )
        assert query.query == "AI research"
        assert query.categories == ["Document"]
        assert query.category_types == ["Research"]
        assert query.keywords == ["machine learning"]
    
    def test_upload_response_model(self):
        """Test UploadResponse model."""
        response = UploadResponse(
            status="success",
            message="File uploaded",
            document_id="test-id",
            categories=["Processing"]
        )
        assert response.status == "success"
        assert response.message == "File uploaded"
        assert response.document_id == "test-id"
        assert response.categories == ["Processing"]
    
    @patch('app.services.document_service.DocumentProcessor')
    def test_document_service_initialization(self, mock_processor):
        """Test DocumentService initialization."""
        service = DocumentService()
        assert service.processor is not None
        assert service.logger is not None
    
    def test_imports_work(self):
        """Test that all imports work correctly."""
        try:
            from app.main import app
            from app.api import upload, search, categories, status
            from app.models import schemas
            from app.services import document_service
            from app.utils import middleware
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    test = TestBasicFunctionality()
    test.test_search_query_model()
    test.test_search_query_with_filters()
    test.test_upload_response_model()
    test.test_imports_work()
    print("✓ All basic tests passed!")