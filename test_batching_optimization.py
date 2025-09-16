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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor

class TestBatchingOptimization:
    def __init__(self):
        self.test_dir = None
        self.processor = None
        self.original_index_file = None
        
    def setup_test_environment(self):
        """Set up isolated test environment"""
        self.test_dir = tempfile.mkdtemp(prefix="pdf_mapper_test_")
        print(f"Created test directory: {self.test_dir}")
        
        self.processor = DocumentProcessor()
        self.original_index_file = self.processor.index_file
        self.processor.processed_dir = os.path.join(self.test_dir, "processed_data")
        self.processor.index_file = os.path.join(self.processor.processed_dir, "document_index.json")
        
        os.makedirs(self.processor.processed_dir, exist_ok=True)
        
        self.processor.document_index = {
            "documents": {},
            "categories": ["Uncategorized"]
        }
        
        return True
        
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"Cleaned up test directory: {self.test_dir}")
            
    def test_batching_methods_exist(self):
        """Test that batching methods are available"""
        print("\n=== Testing Batching Methods Existence ===")
        
        methods = ['_mark_for_save', 'flush_pending_saves', '_save_document_index_immediate']
        for method in methods:
            if hasattr(self.processor, method):
                print(f"✓ Method {method} exists")
            else:
                print(f"✗ Method {method} missing")
                return False
                
        return True
        
    def test_mark_for_save_functionality(self):
        """Test _mark_for_save sets pending flag correctly"""
        print("\n=== Testing Mark for Save Functionality ===")
        
        if self.processor._pending_save:
            print("✗ Initial pending save state should be False")
            return False
        print("✓ Initial pending save state is False")
        
        self.processor._mark_for_save()
        
        if not self.processor._pending_save:
            print("✗ Pending save flag not set after _mark_for_save()")
            return False
        print("✓ Pending save flag set correctly")
        
        return True
        
    def test_flush_pending_saves(self):
        """Test flush_pending_saves writes to disk and clears flag"""
        print("\n=== Testing Flush Pending Saves ===")
        
        self.processor.document_index["test_data"] = "batch_test"
        
        self.processor._mark_for_save()
        
        if os.path.exists(self.processor.index_file):
            os.remove(self.processor.index_file)
            
        self.processor.flush_pending_saves()
        
        if not os.path.exists(self.processor.index_file):
            print("✗ Index file not created after flush_pending_saves()")
            return False
        print("✓ Index file created after flush")
        
        if self.processor._pending_save:
            print("✗ Pending save flag not cleared after flush")
            return False
        print("✓ Pending save flag cleared")
        
        with open(self.processor.index_file, 'r') as f:
            saved_data = json.load(f)
            
        if saved_data.get("test_data") != "batch_test":
            print("✗ Test data not saved correctly")
            return False
        print("✓ Data saved correctly to file")
        
        return True
        
    def test_thread_safety(self):
        """Test thread safety of batching operations"""
        print("\n=== Testing Thread Safety ===")
        
        results = []
        
        def mark_and_flush():
            try:
                self.processor._mark_for_save()
                time.sleep(0.01)  # Small delay to test concurrency
                self.processor.flush_pending_saves()
                results.append(True)
            except Exception as e:
                print(f"Thread error: {e}")
                results.append(False)
                
        threads = []
        for i in range(5):
            thread = threading.Thread(target=mark_and_flush)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        if len(results) != 5 or not all(results):
            print("✗ Thread safety test failed")
            return False
        print("✓ Thread safety test passed")
        
        return True
        
    def test_batching_vs_immediate_saves(self):
        """Test that batching reduces file I/O operations"""
        print("\n=== Testing Batching vs Immediate Saves ===")
        
        immediate_writes = 0
        original_open = open
        
        def count_writes(*args, **kwargs):
            nonlocal immediate_writes
            if len(args) > 1 and 'w' in args[1] and self.processor.index_file in args[0]:
                immediate_writes += 1
            return original_open(*args, **kwargs)
            
        with patch('builtins.open', side_effect=count_writes):
            for i in range(5):
                self.processor.document_index[f"doc_{i}"] = f"data_{i}"
                self.processor._save_document_index_immediate()
                
        print(f"Immediate saves: {immediate_writes} file writes")
        
        batched_writes = 0
        
        def count_batched_writes(*args, **kwargs):
            nonlocal batched_writes
            if len(args) > 1 and 'w' in args[1] and self.processor.index_file in args[0]:
                batched_writes += 1
            return original_open(*args, **kwargs)
            
        with patch('builtins.open', side_effect=count_batched_writes):
            for i in range(5):
                self.processor.document_index[f"batch_doc_{i}"] = f"batch_data_{i}"
                self.processor._mark_for_save()
            self.processor.flush_pending_saves()
            
        print(f"Batched saves: {batched_writes} file writes")
        
        if immediate_writes <= batched_writes:
            print("✗ Batching did not reduce file writes")
            return False
        print(f"✓ Batching reduced file writes from {immediate_writes} to {batched_writes}")
        
        return True
        
    def run_all_tests(self):
        """Run all batching optimization tests"""
        print("Starting JSON I/O Batching Optimization Tests")
        print("=" * 50)
        
        if not self.setup_test_environment():
            print("✗ Failed to set up test environment")
            return False
            
        try:
            tests = [
                self.test_batching_methods_exist,
                self.test_mark_for_save_functionality,
                self.test_flush_pending_saves,
                self.test_thread_safety,
                self.test_batching_vs_immediate_saves
            ]
            
            passed = 0
            total = len(tests)
            
            for test in tests:
                try:
                    if test():
                        passed += 1
                    else:
                        print(f"✗ Test {test.__name__} failed")
                except Exception as e:
                    print(f"✗ Test {test.__name__} failed with exception: {e}")
                    
            print(f"\n=== Test Results ===")
            print(f"Passed: {passed}/{total}")
            
            if passed == total:
                print("✓ All batching optimization tests passed!")
                return True
            else:
                print("✗ Some tests failed")
                return False
                
        finally:
            self.cleanup_test_environment()

if __name__ == "__main__":
    tester = TestBatchingOptimization()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
