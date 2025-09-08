"""
Refactored document processor using modular components.
"""
import os
import uuid
import time
import logging
from typing import Tuple, List

from .text_extraction.extractor_factory import ExtractorFactory
from .text_processing.text_preprocessor import TextPreprocessor
from .categorization.category_manager import CategoryManager
from .storage.document_storage import DocumentStorage


class DocumentProcessor:
    """Main document processor using modular components."""
    
    def __init__(self):
        self.processed_dir = "processed_data"
        self.index_file = os.path.join(self.processed_dir, "document_index.json")
        self.model_file = os.path.join(self.processed_dir, "category_model.pkl")
        self.vectorizer_file = os.path.join(self.processed_dir, "vectorizer.pkl")
        
        # Create processed_data directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize components
        self.storage = DocumentStorage(self.index_file)
        self.preprocessor = TextPreprocessor()
        self.category_manager = CategoryManager(self.model_file, self.vectorizer_file)
        
        # For backward compatibility, expose document_index
        self.document_index = self.storage.document_index
    
    def process(self, file_path: str) -> Tuple[str, List[str]]:
        """
        Process a document: extract text, categorize, and index it.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (document_id, categories)
        """
        start_time = time.time()
        logging.info(f"Starting document processing: {file_path}")
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        try:
            # Extract text using appropriate extractor
            if not ExtractorFactory.is_supported_file(file_path):
                raise ValueError(f"Unsupported file type: {file_path}")
            
            extractor = ExtractorFactory.create_extractor(file_path)
            text = extractor.extract_text(file_path)
            
            logging.info(f"Text extraction completed in {time.time() - start_time:.2f}s")
            
            # Skip if no text was extracted
            if not text or (text.startswith("Error:") and len(text) < 100):
                logging.error(f"No text could be extracted from the document: {file_path}")
                raise ValueError("No text could be extracted from the document")
            
            # Check for duplicate documents
            file_name = os.path.basename(file_path)
            existing_doc_id = self.storage.find_duplicate_document(file_name, text)
            
            if existing_doc_id:
                logging.info(f"Duplicate document detected. Using existing document ID: {existing_doc_id}")
                
                # Update the existing document with new path if needed
                self.storage.update_document(existing_doc_id, {"path": file_path})
                self.storage.save_index()
                
                # Return existing document ID and categories
                existing_doc = self.storage.get_document(existing_doc_id)
                return existing_doc_id, existing_doc["categories"]
            
            # Log text sample
            text_sample = text[:500].replace('\n', ' ').strip()
            logging.info(f"Extracted text sample: {text_sample}...")
            
            # Preprocess the text
            preprocessed_text = self.preprocessor.preprocess(text)
            logging.info(f"Text preprocessing completed in {time.time() - start_time:.2f}s")
            
            # Categorize the document
            categories = self._categorize_text(preprocessed_text)
            logging.info(f"Categorization completed in {time.time() - start_time:.2f}s")
            
            # Calculate content hash for future duplicate detection
            content_hash = self.storage.calculate_content_hash(text)
            
            # Store document data
            document_data = {
                "id": doc_id,
                "filename": file_name,
                "path": file_path,
                "categories": categories,
                "preprocessed_text": preprocessed_text,
                "full_text": text,
                "content_hash": content_hash
            }
            
            # Add to document index
            self.storage.add_document(doc_id, document_data)
            self.storage.save_index()
            
            logging.info(f"Document processing completed in {time.time() - start_time:.2f}s: {doc_id}")
            return doc_id, categories
            
        except Exception as e:
            logging.error(f"Error processing document {file_path}: {e}")
            # Return temporary ID and error category
            return doc_id, ["Error: " + str(e)]
    
    def _categorize_text(self, text: str) -> List[str]:
        """
        Determine categories for the document based on content.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of categories
        """
        # Handle error messages
        if text.startswith("Error:"):
            logging.warning(f"Categorizing text with error: {text[:100]}...")
            return ["Error"]
        
        try:
            doc_count = len(self.storage.get_all_documents())
            logging.info(f"Current document count: {doc_count}")
            
            # For the first few documents, create simple categories
            if doc_count < 5:
                logging.info(f"Not enough documents for clustering ({doc_count}/5), creating simple category")
                return self._create_simple_category(text)
            
            # If we have exactly 5 documents, fit the model
            if doc_count == 5:
                logging.info("Reached 5 documents, fitting vectorizer and model")
                self._fit_model_with_all_documents()
            
            # If we have more than 5 documents and model is fitted
            if doc_count >= 5 and hasattr(self.category_manager.model, 'cluster_centers_'):
                return self._predict_category(text)
            else:
                logging.info("Model not fitted yet, using Uncategorized")
                return ["Uncategorized"]
                
        except Exception as e:
            logging.error(f"Error in categorize_text: {e}")
            return ["Error"]
    
    def _create_simple_category(self, text: str) -> List[str]:
        """Create simple category for documents when clustering isn't available."""
        keywords = self.preprocessor.extract_keywords(text, max_keywords=3)
        
        if keywords:
            category_name = f"Topic: {', '.join(keywords)}"
            logging.info(f"Created simple category: {category_name}")
            
            # Add this category if it doesn't exist
            if category_name not in self.document_index["categories"]:
                self.document_index["categories"].append(category_name)
                logging.info(f"Added new category: {category_name}")
            
            return [category_name]
        
        # Fallback to Uncategorized
        if not self.document_index["categories"]:
            self.document_index["categories"] = ["Uncategorized"]
        return ["Uncategorized"]
    
    def _fit_model_with_all_documents(self):
        """Fit the model with all documents."""
        try:
            all_texts = [doc["preprocessed_text"] for doc in self.storage.get_all_documents().values()]
            self.category_manager.fit_model(all_texts)
            self.category_manager.generate_category_names(self.document_index)
            self.storage.save_index()
        except Exception as e:
            logging.error(f"Error fitting model: {e}")
    
    def _predict_category(self, text: str) -> List[str]:
        """Predict category using the fitted model."""
        try:
            cluster = self.category_manager.predict_category(text)
            logging.info(f"Document assigned to cluster {cluster}")
            
            # Return the corresponding category
            if 0 <= cluster < len(self.document_index["categories"]):
                category = self.document_index["categories"][cluster]
                logging.info(f"Assigned category: {category}")
                return [category]
            else:
                logging.warning(f"Cluster {cluster} out of range for categories, using Uncategorized")
                return ["Uncategorized"]
        except Exception as e:
            logging.error(f"Error predicting cluster: {e}")
            return ["Uncategorized"]
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        try:
            return self.document_index["categories"]
        except Exception as e:
            logging.error(f"Error getting categories: {e}")
            return []
    
    def generate_structured_categories(self) -> List[dict]:
        """Generate structured categories from existing categories."""
        try:
            categories = self.document_index.get("categories", [])
            if not categories:
                logging.warning("No categories found to structure")
                return []
            
            structured_categories = []
            for i, category_name in enumerate(categories):
                parts = category_name.split(": ", 1)
                if len(parts) == 2:
                    category_type = parts[0]
                    keywords = [k.strip() for k in parts[1].split(",")]
                    
                    structured_category = {
                        "id": f"cat-{i+1:03d}",
                        "type": category_type,
                        "keywords": keywords,
                        "display_name": category_name,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
                    }
                    structured_categories.append(structured_category)
                    logging.info(f"Created structured category: {structured_category}")
                else:
                    logging.warning(f"Could not parse category: {category_name}")
            
            # Store the structured categories
            self.document_index["structured_categories"] = structured_categories
            logging.info(f"Added {len(structured_categories)} structured categories to document index")
            
            # Save the document index
            self.storage.save_index()
            logging.info("Saved document index with structured categories")
            
            return structured_categories
        except Exception as e:
            logging.error(f"Error generating structured categories: {e}")
            return []
    
    def clean_up_duplicates(self) -> int:
        """Clean up duplicate documents in the index."""
        return self.storage.clean_up_duplicates()