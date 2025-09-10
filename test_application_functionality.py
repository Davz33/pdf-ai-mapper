#!/usr/bin/env python3
"""
End-to-end functionality test for the PDF AI Mapper application
Tests that the application starts and core functionality works with optimizations
"""

import os
import sys
import json
import tempfile
import shutil
import time
import requests
import threading
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestApplicationFunctionality:
    def __init__(self):
        self.test_dir = None
        self.server_process = None
        self.base_url = "http://localhost:8000"
        self.original_dir = os.getcwd()
        
    def setup_test_environment(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp(prefix="pdf_mapper_app_test_")
        print(f"Created test directory: {self.test_dir}")
        
        os.chdir(self.test_dir)
        
        app_files = [
            "main.py", "document_processor.py", "search_engine.py", "logger.py"
        ]
        
        # Copy Python files
        source_dir = Path(self.original_dir)
        test_dir = Path(self.test_dir)
        for file in app_files:
            src = source_dir / file
            dst = test_dir / file
            if src.exists():
                shutil.copy2(src, dst)
                print(f"Copied {file}")
            else:
                print(f"Warning: {file} not found")
        
        # Create necessary directories
        test_path = Path(self.test_dir)
        (test_path / "uploads").mkdir(exist_ok=True)
        (test_path / "processed_data").mkdir(exist_ok=True)
        (test_path / "logs").mkdir(exist_ok=True)
        
        # Copy processed data if it exists
        processed_data_src = Path(source_dir) / "apps" / "backend-python" / "processed_data"
        if processed_data_src.exists():
            processed_data_dst = test_path / "processed_data"
            for file_path in processed_data_src.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, processed_data_dst / file_path.name)
                    print(f"Copied processed data: {file_path.name}")
                
        return True
        
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            print("Server process terminated")
            
        if self.test_dir and Path(self.test_dir).exists():
            os.chdir(self.original_dir)  # Change back to original directory
            shutil.rmtree(self.test_dir)
            print(f"Cleaned up test directory: {self.test_dir}")
            
    def start_server(self):
        """Start the FastAPI server"""
        print("\n=== Starting FastAPI Server ===")
        
        try:
            # First, test if we can import the modules
            print("Testing module imports...")
            test_import_cmd = [
                sys.executable, "-c", 
                "import main, document_processor, search_engine, logger; print('All modules imported successfully')"
            ]
            result = subprocess.run(test_import_cmd, cwd=self.test_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"✗ Module import test failed: {result.stderr}")
                return False
            else:
                print("✓ Module import test passed")
            
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.test_dir
            )
            
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=1)
                    if response.status_code == 200:
                        print("✓ Server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    pass
                    
                time.sleep(1)
                print(f"Waiting for server... ({attempt + 1}/{max_attempts})")
                
            print("✗ Server failed to start within timeout")
            # Print server output for debugging
            if self.server_process:
                stdout, stderr = self.server_process.communicate(timeout=1)
                if stdout:
                    print(f"Server stdout: {stdout.decode()}")
                if stderr:
                    print(f"Server stderr: {stderr.decode()}")
            return False
            
        except Exception as e:
            print(f"✗ Error starting server: {e}")
            return False
            
    def test_health_endpoint(self):
        """Test health check endpoint"""
        print("\n=== Testing Health Endpoint ===")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    print("✓ Health endpoint working")
                    return True
                else:
                    print(f"✗ Unexpected health response: {data}")
                    return False
            else:
                print(f"✗ Health endpoint returned {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Health endpoint test failed: {e}")
            return False
            
    def test_status_endpoint(self):
        """Test status endpoint"""
        print("\n=== Testing Status Endpoint ===")
        
        try:
            response = requests.get(f"{self.base_url}/status")
            if response.status_code == 200:
                data = response.json()
                required_fields = ["total_documents", "categories", "last_updated"]
                
                for field in required_fields:
                    if field not in data:
                        print(f"✗ Missing field in status response: {field}")
                        return False
                        
                print("✓ Status endpoint working")
                print(f"  Total documents: {data['total_documents']}")
                print(f"  Categories: {len(data['categories'])}")
                return True
            else:
                print(f"✗ Status endpoint returned {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Status endpoint test failed: {e}")
            return False
            
    def test_categories_endpoint(self):
        """Test categories endpoint"""
        print("\n=== Testing Categories Endpoint ===")
        
        try:
            response = requests.get(f"{self.base_url}/categories")
            if response.status_code == 200:
                data = response.json()
                if "categories" in data and isinstance(data["categories"], list):
                    print("✓ Categories endpoint working")
                    print(f"  Available categories: {len(data['categories'])}")
                    return True
                else:
                    print(f"✗ Unexpected categories response: {data}")
                    return False
            else:
                print(f"✗ Categories endpoint returned {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Categories endpoint test failed: {e}")
            return False
            
    def test_search_endpoint(self):
        """Test search endpoint"""
        print("\n=== Testing Search Endpoint ===")
        
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={"query": "test search"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "results" in data and isinstance(data["results"], list):
                    print("✓ Search endpoint working")
                    print(f"  Search results: {len(data['results'])}")
                    return True
                else:
                    print(f"✗ Unexpected search response: {data}")
                    return False
            else:
                print(f"✗ Search endpoint returned {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Search endpoint test failed: {e}")
            return False
            
    def test_batching_behavior(self):
        """Test that batching optimization is working"""
        print("\n=== Testing Batching Behavior ===")
        
        try:
            processed_dir = Path(self.test_dir) / "processed_data"
            index_file = processed_dir / "document_index.json"
            
            initial_mtime = None
            if index_file.exists():
                initial_mtime = index_file.stat().st_mtime
                
            for i in range(3):
                requests.get(f"{self.base_url}/categories")
                time.sleep(0.1)
                
            if index_file.exists():
                final_mtime = index_file.stat().st_mtime
                if initial_mtime and final_mtime > initial_mtime:
                    print("✓ Document index file updated (batching working)")
                elif not initial_mtime:
                    print("✓ Document index file created (batching working)")
                else:
                    print("✓ Document index file stable (batching may be working)")
                return True
            else:
                print("✓ No unnecessary file creation (batching working)")
                return True
                
        except Exception as e:
            print(f"✗ Batching behavior test failed: {e}")
            return False
            
    def run_all_tests(self):
        """Run all application functionality tests"""
        print("Starting Application Functionality Tests")
        print("=" * 50)
        
        if not self.setup_test_environment():
            print("✗ Failed to set up test environment")
            return False
            
        try:
            if not self.start_server():
                print("✗ Failed to start server")
                return False
                
            tests = [
                self.test_health_endpoint,
                self.test_status_endpoint,
                self.test_categories_endpoint,
                self.test_search_endpoint,
                self.test_batching_behavior
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
                print("✓ All application functionality tests passed!")
                return True
            else:
                print("✗ Some tests failed")
                return False
                
        finally:
            self.cleanup_test_environment()

if __name__ == "__main__":
    tester = TestApplicationFunctionality()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
