"""
Main document processor for PDF AI Mapper.
Orchestrates text extraction, processing, categorization, and storage.
"""

import os
import uuid
import logging
import datetime
from typing import Optional

from .text_extraction.extractor_factory import ExtractorFactory
from .text_processing.text_preprocessor import TextPreprocessor
from .categorization.category_manager import CategoryManager
from .storage.document_storage import DocumentStorage


class DocumentProcessor:
    """Main document processor that orchestrates all processing steps."""
    
    def __init__(self, processed_dir: str = "processed_data"):
        self.logger = logging.getLogger(__name__)
        self.processed_dir = processed_dir
        
        # Initialize components
        self.extractor_factory = ExtractorFactory()
        self.text_preprocessor = TextPreprocessor()
        self.storage = DocumentStorage(processed_dir)
        
        # Load document index
        self.document_index = self.storage.load_document_index()
        
        # Initialize category manager
        model_file = os.path.join(processed_dir, "category_model.pkl")
        vectorizer_file = os.path.join(processed_dir, "vectorizer.pkl")
        self.category_manager = CategoryManager(model_file, vectorizer_file)
        
        # Save initial index if it's new
        if not os.path.exists(self.storage.index_file):
            self.storage.save_document_index(self.document_index)
            self.logger.info("Created new document index file")
    
    def process(self, file_path: str) -> Optional[str]:
        """Process a document and add it to the index."""
        try:
            file_name = os.path.basename(file_path)
            self.logger.info(f"Processing document: {file_name}")
            
            # Calculate content hash for duplicate detection
            content_hash = self.storage.calculate_content_hash(file_path)
            
            # Check for duplicates
            existing_doc_id = self.storage.check_for_duplicate(file_name, content_hash, self.document_index)
            if existing_doc_id:
                self.logger.info(f"Document already exists with ID: {existing_doc_id}")
                return existing_doc_id
            
            # Extract text based on file type
            extractor = self.extractor_factory.get_extractor(file_path)
            full_text = extractor.extract_text(file_path)
            
            if not full_text or full_text.startswith("Error:"):
                self.logger.error(f"Failed to extract text from {file_name}")
                return None
            
            # Preprocess text
            preprocessed_text = self.text_preprocessor.preprocess_text(full_text)
            
            # Categorize the document
            categories = self.category_manager.categorize_text(full_text, self.document_index, self.text_preprocessor)
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Save content to file
            content_file = self.storage.save_content_file(doc_id, full_text)
            
            # Create document entry
            document_entry = {
                "id": doc_id,
                "filename": file_name,
                "file_path": file_path,
                "content_file": content_file,
                "full_text": full_text,
                "preprocessed_text": preprocessed_text,
                "categories": categories,
                "content_hash": content_hash,
                "processed_at": datetime.datetime.now().isoformat()
            }
            
            # Add to index
            self.document_index["documents"][doc_id] = document_entry
            
            # Mark for save
            self.storage.mark_for_save()
            
            self.logger.info(f"Successfully processed document: {file_name} (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            return None
    
    def get_categories(self) -> list:
        """Get all available categories."""
        return self.document_index.get("categories", ["Uncategorized"])
    
    def clean_up_duplicates(self) -> int:
        """Clean up duplicate documents in the index."""
        return self.storage.clean_up_duplicates(self.document_index)
    
    def generate_structured_categories(self) -> list:
        """Generate structured categories from existing categories."""
        return self.storage.generate_structured_categories(self.document_index)
    
    def flush_pending_saves(self) -> None:
        """Flush any pending saves to disk."""
        self.storage.flush_pending_saves(self.document_index)
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the index."""
        return len(self.document_index.get("documents", {}))
    
    def get_document_by_id(self, doc_id: str) -> Optional[dict]:
        """Get a document by its ID."""
        return self.document_index.get("documents", {}).get(doc_id)
    
    def get_all_documents(self) -> dict:
        """Get all documents in the index."""
        return self.document_index.get("documents", {})