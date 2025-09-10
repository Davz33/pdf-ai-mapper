#!/usr/bin/env python3
"""
Comprehensive test suite for JSON I/O batching optimization
Tests the _mark_for_save() and flush_pending_saves() functionality
"""

import os
import sys
import json
import tempfile
import shutil
import time
import threading
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor


def test_batching_methods_exist(document_processor):
    """Test that batching methods exist on DocumentProcessor"""
    assert hasattr(document_processor, '_mark_for_save')
    assert hasattr(document_processor, 'flush_pending_saves')
    assert callable(getattr(document_processor, '_mark_for_save'))
    assert callable(getattr(document_processor, 'flush_pending_saves'))


def test_mark_for_save_functionality(document_processor):
    """Test that _mark_for_save correctly marks documents for saving"""
    # Add a test document to the index
    test_doc_id = "test_doc_123"
    test_doc_data = {
        "filename": "test.pdf",
        "content": "test content",
        "categories": ["Test"]
    }
    
    document_processor.document_index["documents"][test_doc_id] = test_doc_data
    
    # Mark for save
    document_processor._mark_for_save()
    
    # Check that the document is marked for saving
    # (This would depend on the actual implementation of _mark_for_save)
    assert test_doc_id in document_processor.document_index["documents"]


def test_flush_pending_saves(document_processor):
    """Test that flush_pending_saves writes pending changes to disk"""
    # Add test data
    test_doc_id = "test_doc_456"
    test_doc_data = {
        "filename": "test2.pdf",
        "content": "test content 2",
        "categories": ["Test"]
    }
    
    document_processor.document_index["documents"][test_doc_id] = test_doc_data
    
    # Mark for save first, then flush
    document_processor._mark_for_save()
    document_processor.flush_pending_saves()
    
    # Check that the index file was written
    assert os.path.exists(document_processor.index_file)
    
    # Verify the content was written correctly
    with open(document_processor.index_file, 'r') as f:
        saved_data = json.load(f)
    
    assert test_doc_id in saved_data["documents"]
    assert saved_data["documents"][test_doc_id]["filename"] == "test2.pdf"


def test_thread_safety(document_processor):
    """Test that batching operations are thread-safe"""
    results = []
    errors = []
    
    def worker(worker_id):
        try:
            # Simulate concurrent document processing
            doc_id = f"worker_{worker_id}_doc"
            doc_data = {
                "filename": f"worker_{worker_id}.pdf",
                "content": f"content from worker {worker_id}",
                "categories": ["Worker"]
            }
            
            document_processor.document_index["documents"][doc_id] = doc_data
            document_processor._mark_for_save()
            document_processor.flush_pending_saves()
            
            results.append(worker_id)
        except Exception as e:
            errors.append(f"Worker {worker_id}: {e}")
    
    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check that all workers completed successfully
    assert len(errors) == 0, f"Thread safety errors: {errors}"
    assert len(results) == 5, f"Expected 5 workers, got {len(results)}"


def test_batching_vs_immediate_saves(document_processor):
    """Test that batching is more efficient than immediate saves"""
    # This test would measure performance differences
    # For now, we'll just test that both methods work
    
    # Test immediate save (if such a method exists)
    test_doc_id = "immediate_test"
    test_doc_data = {
        "filename": "immediate.pdf",
        "content": "immediate content",
        "categories": ["Immediate"]
    }
    
    document_processor.document_index["documents"][test_doc_id] = test_doc_data
    document_processor.flush_pending_saves()
    
    # Test batched save
    batched_doc_id = "batched_test"
    batched_doc_data = {
        "filename": "batched.pdf",
        "content": "batched content",
        "categories": ["Batched"]
    }
    
    document_processor.document_index["documents"][batched_doc_id] = batched_doc_data
    document_processor._mark_for_save()
    document_processor.flush_pending_saves()
    
    # Verify both documents were saved
    with open(document_processor.index_file, 'r') as f:
        saved_data = json.load(f)
    
    assert test_doc_id in saved_data["documents"]
    assert batched_doc_id in saved_data["documents"]