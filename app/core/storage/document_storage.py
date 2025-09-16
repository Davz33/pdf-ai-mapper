"""
Document storage module for PDF AI Mapper.
Handles document index persistence and duplicate detection.
"""

import os
import json
import hashlib
import logging
import threading
import datetime
from typing import Dict, Any, Optional


class DocumentStorage:
    """Manages document storage and persistence."""
    
    def __init__(self, processed_dir: str):
        self.logger = logging.getLogger(__name__)
        self.processed_dir = processed_dir
        self.index_file = os.path.join(processed_dir, "document_index.json")
        self._pending_save = False
        self._save_lock = threading.Lock()
        
        # Create processed_data directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)
    
    def load_document_index(self) -> Dict[str, Any]:
        """Load the document index from file."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    document_index = json.load(f)
                self.logger.info(f"Loaded document index with {len(document_index['documents'])} documents")
                return document_index
            except Exception as e:
                self.logger.error(f"Error loading document index: {e}")
                return self._create_empty_index()
        else:
            self.logger.info("No existing document index found, creating new one")
            return self._create_empty_index()
    
    def _create_empty_index(self) -> Dict[str, Any]:
        """Create an empty document index."""
        return {
            "documents": {},
            "categories": ["Uncategorized"]
        }
    
    def save_document_index(self, document_index: Dict[str, Any]) -> None:
        """Save the document index to file."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(document_index, f)
            self.logger.info("Saved document index")
        except Exception as e:
            self.logger.error(f"Error saving document index: {e}")
    
    def mark_for_save(self) -> None:
        """Mark the document index for batched saving."""
        with self._save_lock:
            self._pending_save = True
    
    def flush_pending_saves(self, document_index: Dict[str, Any]) -> None:
        """Flush any pending saves to disk."""
        with self._save_lock:
            if self._pending_save:
                try:
                    self.save_document_index(document_index)
                    self._pending_save = False
                    self.logger.info("Flushed pending document index save")
                except Exception as e:
                    self.logger.error(f"Error flushing document index save: {e}")
                    raise
    
    def calculate_content_hash(self, file_path: str) -> str:
        """Calculate content hash for duplicate detection."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating content hash: {e}")
            return ""
    
    def check_for_duplicate(self, file_name: str, content_hash: str, document_index: Dict[str, Any]) -> Optional[str]:
        """Check if a document with the same filename or content already exists."""
        self.logger.info(f"Checking for duplicates: filename={file_name}, hash={content_hash}")
        
        # First check for exact filename match
        for doc_id, doc in document_index["documents"].items():
            if doc["filename"] == file_name:
                self.logger.info(f"Found duplicate by filename: {file_name}")
                return doc_id
        
        # Then check for content similarity
        for doc_id, doc in document_index["documents"].items():
            if "content_hash" in doc and doc["content_hash"] == content_hash:
                self.logger.info(f"Found duplicate by content hash: {content_hash}")
                return doc_id
            
        # No duplicate found
        return None
    
    def clean_up_duplicates(self, document_index: Dict[str, Any]) -> int:
        """Clean up duplicate documents in the index."""
        try:
            self.logger.info("Starting duplicate cleanup process")
            
            # Track duplicates by content hash
            content_hashes = {}
            duplicates_found = []
            
            for doc_id, doc in document_index["documents"].items():
                if "content_hash" in doc:
                    content_hash = doc["content_hash"]
                    if content_hash in content_hashes:
                        # This is a duplicate
                        duplicates_found.append(doc_id)
                        self.logger.info(f"Found duplicate document: {doc_id} (duplicate of {content_hashes[content_hash]})")
                    else:
                        content_hashes[content_hash] = doc_id
            
            # Remove duplicates
            for duplicate_id in duplicates_found:
                if duplicate_id in document_index["documents"]:
                    del document_index["documents"][duplicate_id]
                    self.logger.info(f"Removed duplicate document: {duplicate_id}")
            
            # Mark for save
            self.mark_for_save()
            
            self.logger.info(f"Duplicate cleanup completed. Removed {len(duplicates_found)} duplicate documents")
            return len(duplicates_found)
            
        except Exception as e:
            self.logger.error(f"Error during duplicate cleanup: {e}")
            return 0
    
    def save_content_file(self, doc_id: str, content: str) -> Optional[str]:
        """Save document content to a file."""
        try:
            content_file = os.path.join(self.processed_dir, f"{doc_id}_content.txt")
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return content_file
        except Exception as e:
            self.logger.error(f"Error saving content file: {e}")
            return None
    
    def generate_structured_categories(self, document_index: Dict[str, Any]) -> list:
        """Generate structured categories from existing categories."""
        try:
            self.logger.info("Generating structured categories")
            
            # Get existing categories
            existing_categories = document_index.get("categories", [])
            if not existing_categories:
                self.logger.warning("No categories found to structure")
                return []
            
            # Create structured categories
            structured_categories = []
            for i, category in enumerate(existing_categories):
                # Parse category name to extract type and keywords
                if ": " in category:
                    category_type, keywords_str = category.split(": ", 1)
                    keywords = [kw.strip() for kw in keywords_str.split(", ")]
                else:
                    category_type = "Document"
                    keywords = [category]
                
                # Create structured category
                structured_category = {
                    "id": f"cat-{i+1:03d}",
                    "type": category_type,
                    "keywords": keywords,
                    "display_name": category,
                    "created_at": datetime.datetime.now().isoformat()
                }
                
                structured_categories.append(structured_category)
                self.logger.info(f"Created structured category: {structured_category}")
            
            # Store the structured categories
            document_index["structured_categories"] = structured_categories
            self.logger.info(f"Added {len(structured_categories)} structured categories to document index")
            
            self.mark_for_save()
            self.logger.info("Marked document index for save with structured categories")
            
            return structured_categories
        except Exception as e:
            self.logger.error(f"Error generating structured categories: {e}")
            return []