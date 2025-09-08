"""
Document processing service that handles file uploads and processing.
"""
import os
import uuid
import threading
import traceback
from typing import Tuple, List
from logger import setup_logger

from document_processor import DocumentProcessor


class DocumentService:
    """Service for handling document operations."""
    
    def __init__(self):
        self.logger = setup_logger("document-service")
        self.processor = DocumentProcessor()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("processed_data", exist_ok=True)
    
    def process_document_background(self, file_path: str, file_name: str, doc_id: str):
        """Process a document in the background."""
        try:
            self.logger.info(f"Background processing started for file: {file_name}")
            
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return
                
            file_size = os.path.getsize(file_path)
            self.logger.info(f"Processing file of size: {file_size} bytes")
            
            processed_doc_id, categories = self.processor.process(file_path)
            
            is_duplicate = processed_doc_id != doc_id
            if is_duplicate:
                self.logger.info(f"Duplicate document detected. Original ID: {processed_doc_id}, Upload ID: {doc_id}")
            
            if categories and categories[0].startswith("Error:"):
                self.logger.error(f"Error processing document: {categories[0]}")
            else:
                self.logger.info(f"Background processing completed successfully for file: {file_name}")
                self.logger.info(f"Document ID: {processed_doc_id}, Categories: {categories}")
                
            self.recategorize_all_documents()
            self._update_status_data()
                
        except Exception as e:
            self.logger.error(f"Error in background processing for file {file_name}: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def recategorize_all_documents(self):
        """Recategorize all documents after new upload."""
        try:
            self.logger.info("Auto-recategorizing all documents after new upload")
            
            documents = self.processor.document_index["documents"]
            doc_count = len(documents)
            
            if doc_count == 0:
                self.logger.info("No documents to recategorize")
                return
            
            if doc_count >= 5:
                self._refit_model(documents)
            
            self._update_document_categories(documents)
            self._save_updated_index()
            
            self.logger.info(f"Auto-recategorization completed: {doc_count} documents processed")
            
        except Exception as e:
            self.logger.error(f"Error during auto-recategorization: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _refit_model(self, documents: dict):
        """Refit the vectorizer and model with all documents."""
        try:
            self.logger.info("Re-fitting vectorizer and model with all documents")
            all_texts = [doc["preprocessed_text"] for doc in documents.values()]
            
            text_vectors = self.processor.vectorizer.fit_transform(all_texts)
            self.processor.model.fit(text_vectors)
            
            self.processor._generate_category_names()
            self.logger.info(f"Regenerated categories: {self.processor.document_index['categories']}")
            
            if 'structured_categories' not in self.processor.document_index or not self.processor.document_index['structured_categories']:
                self.logger.info("Generating structured categories")
                self.processor.generate_structured_categories()
            
            self._save_model_files()
            
        except Exception as e:
            self.logger.error(f"Error re-fitting model: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _update_document_categories(self, documents: dict):
        """Update categories for all documents."""
        updated_count = 0
        for doc_id, doc in list(documents.items()):
            try:
                preprocessed_text = doc["preprocessed_text"]
                categories = self.processor._categorize_text(preprocessed_text)
                documents[doc_id]["categories"] = categories
                updated_count += 1
                self.logger.info(f"Recategorized document {doc_id}: {categories}")
            except Exception as e:
                self.logger.error(f"Error recategorizing document {doc_id}: {str(e)}")
        
        self.logger.info(f"Updated {updated_count} documents")
    
    def _save_updated_index(self):
        """Save the updated document index."""
        import json
        with open(self.processor.index_file, 'w') as f:
            json.dump(self.processor.document_index, f)
    
    def _save_model_files(self):
        """Save the model and vectorizer files."""
        import pickle
        with open(self.processor.model_file, 'wb') as f:
            pickle.dump(self.processor.model, f)
        with open(self.processor.vectorizer_file, 'wb') as f:
            pickle.dump(self.processor.vectorizer, f)
    
    def _update_status_data(self):
        """Update status endpoint data."""
        try:
            if self.processor.document_index["categories"]:
                self.logger.info(f"Current categories: {self.processor.document_index['categories']}")
            else:
                self.logger.warning("No categories found after processing")
                
            doc_count = len(self.processor.document_index["documents"])
            self.logger.info(f"Total documents processed: {doc_count}")
        except Exception as e:
            self.logger.error(f"Error updating status data: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def start_background_processing(self, file_path: str, file_name: str, doc_id: str):
        """Start document processing in a background thread."""
        thread = threading.Thread(
            target=self.process_document_background,
            args=(file_path, file_name, doc_id)
        )
        thread.daemon = True
        thread.start()
        self.logger.info(f"Background processing started for {file_name}")
    
    def get_processor(self) -> DocumentProcessor:
        """Get the document processor instance."""
        return self.processor