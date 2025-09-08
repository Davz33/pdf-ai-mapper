"""
Document storage and indexing utilities.
"""
import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional


class DocumentStorage:
    """Handles document storage and indexing."""
    
    def __init__(self, index_file: str):
        self.index_file = index_file
        self.logger = logging.getLogger(__name__)
        self.document_index = self._load_or_create_index()
    
    def _load_or_create_index(self) -> Dict[str, Any]:
        """Load existing index or create new one."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    index = json.load(f)
                self.logger.info(f"Loaded document index with {len(index['documents'])} documents")
                return index
            except Exception as e:
                self.logger.error(f"Error loading document index: {e}")
                return self._create_empty_index()
        else:
            self.logger.info("No existing document index found, creating new one")
            index = self._create_empty_index()
            self._save_index(index)
            return index
    
    def _create_empty_index(self) -> Dict[str, Any]:
        """Create empty document index."""
        return {
            "documents": {},
            "categories": ["Uncategorized"]
        }
    
    def save_index(self):
        """Save the document index to file."""
        self._save_index(self.document_index)
    
    def _save_index(self, index: Dict[str, Any]):
        """Save index to file."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(index, f)
            self.logger.info("Saved document index")
        except Exception as e:
            self.logger.error(f"Error saving document index: {e}")
    
    def add_document(self, doc_id: str, document_data: Dict[str, Any]):
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            document_data: Document data to store
        """
        self.document_index["documents"][doc_id] = document_data
        self.logger.info(f"Added document {doc_id} to index")
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document data or None if not found
        """
        return self.document_index["documents"].get(doc_id)
    
    def get_all_documents(self) -> Dict[str, Any]:
        """Get all documents."""
        return self.document_index["documents"]
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove document from index.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if document was removed, False if not found
        """
        if doc_id in self.document_index["documents"]:
            del self.document_index["documents"][doc_id]
            self.logger.info(f"Removed document {doc_id} from index")
            return True
        return False
    
    def update_document(self, doc_id: str, updates: Dict[str, Any]):
        """
        Update document data.
        
        Args:
            doc_id: Document identifier
            updates: Data to update
        """
        if doc_id in self.document_index["documents"]:
            self.document_index["documents"][doc_id].update(updates)
            self.logger.info(f"Updated document {doc_id}")
    
    def calculate_content_hash(self, text: str) -> str:
        """
        Calculate content hash for duplicate detection.
        
        Args:
            text: Text content
            
        Returns:
            MD5 hash of content
        """
        text_for_hash = text[:5000].strip().lower()
        return hashlib.md5(text_for_hash.encode('utf-8')).hexdigest()
    
    def find_duplicate_document(self, file_name: str, text: str) -> Optional[str]:
        """
        Find duplicate document by filename or content.
        
        Args:
            file_name: Name of the file
            text: Text content
            
        Returns:
            Document ID if duplicate found, None otherwise
        """
        if not text:
            return None
        
        content_hash = self.calculate_content_hash(text)
        self.logger.info(f"Content hash for duplicate check: {content_hash}")
        
        # Check for exact filename match
        for doc_id, doc in self.document_index["documents"].items():
            if doc["filename"] == file_name:
                self.logger.info(f"Found duplicate by filename: {file_name}")
                return doc_id
        
        # Check for content similarity
        for doc_id, doc in self.document_index["documents"].items():
            if "content_hash" in doc and doc["content_hash"] == content_hash:
                self.logger.info(f"Found duplicate by content hash: {content_hash}")
                return doc_id
        
        return None
    
    def clean_up_duplicates(self) -> int:
        """
        Remove duplicate documents from the index.
        
        Returns:
            Number of duplicates removed
        """
        self.logger.info("Starting duplicate cleanup process")
        
        # Ensure all documents have content hash
        for doc_id, doc in list(self.document_index["documents"].items()):
            if "content_hash" not in doc and "full_text" in doc:
                doc["content_hash"] = self.calculate_content_hash(doc["full_text"])
                self.logger.info(f"Added content hash for document {doc_id}")
        
        # Track documents by content hash
        docs_by_hash = {}
        duplicates_to_remove = []
        
        # Identify duplicates
        for doc_id, doc in self.document_index["documents"].items():
            if "content_hash" not in doc:
                self.logger.warning(f"Document {doc_id} has no content hash, skipping")
                continue
            
            content_hash = doc["content_hash"]
            
            if content_hash in docs_by_hash:
                duplicates_to_remove.append(doc_id)
                self.logger.info(f"Identified duplicate document: {doc_id} (same as {docs_by_hash[content_hash]})")
            else:
                docs_by_hash[content_hash] = doc_id
        
        # Remove duplicates
        for doc_id in duplicates_to_remove:
            del self.document_index["documents"][doc_id]
            self.logger.info(f"Removed duplicate document: {doc_id}")
        
        # Save updated index
        if duplicates_to_remove:
            self.save_index()
            self.logger.info(f"Saved document index after removing {len(duplicates_to_remove)} duplicates")
        
        return len(duplicates_to_remove)