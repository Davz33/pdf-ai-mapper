#!/usr/bin/env python3
"""
Pytest configuration and fixtures for PDF AI Mapper tests
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor


@pytest.fixture
def test_environment():
    """Fixture for setting up isolated test environment"""
    test_dir = tempfile.mkdtemp(prefix="pdf_mapper_test_")
    original_dir = os.getcwd()
    
    try:
        os.chdir(test_dir)
        yield {
            'test_dir': test_dir,
            'original_dir': original_dir
        }
    finally:
        os.chdir(original_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


@pytest.fixture
def app_test_environment():
    """Fixture for application functionality tests with server setup"""
    test_dir = tempfile.mkdtemp(prefix="pdf_mapper_app_test_")
    original_dir = os.getcwd()
    server_process = None
    
    try:
        os.chdir(test_dir)
        
        # Copy necessary app files
        app_files = [
            "main.py", "document_processor.py", "search_engine.py", "logger.py"
        ]
        
        source_dir = Path(original_dir)
        test_dir_path = Path(test_dir)
        
        for file in app_files:
            src = source_dir / file
            dst = test_dir_path / file
            if src.exists():
                shutil.copy2(src, dst)
        
        # Create necessary directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("processed_data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        yield {
            'test_dir': test_dir,
            'original_dir': original_dir,
            'server_process': server_process,
            'base_url': "http://localhost:8000"
        }
    finally:
        if server_process:
            server_process.terminate()
        os.chdir(original_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


@pytest.fixture
def document_processor(test_environment):
    """Fixture for DocumentProcessor with test configuration"""
    processor = DocumentProcessor()
    processor.processed_dir = os.path.join(test_environment['test_dir'], "processed_data")
    processor.index_file = os.path.join(processor.processed_dir, "document_index.json")
    
    os.makedirs(processor.processed_dir, exist_ok=True)
    
    processor.document_index = {
        "documents": {},
        "categories": ["Uncategorized"]
    }
    
    return processor


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
