import os
import uuid
import json
import pytesseract
from PIL import Image
import PyPDF2
from pdf2image import convert_from_path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pickle
import re
import time
import threading
import logging

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK data: {e}")

class DocumentProcessor:
    def __init__(self):
        self.processed_dir = "processed_data"
        self.index_file = os.path.join(self.processed_dir, "document_index.json")
        self.model_file = os.path.join(self.processed_dir, "category_model.pkl")
        self.vectorizer_file = os.path.join(self.processed_dir, "vectorizer.pkl")
        
        # Create processed_data directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize or load document index
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.document_index = json.load(f)
                logging.info(f"Loaded document index with {len(self.document_index['documents'])} documents")
            except Exception as e:
                logging.error(f"Error loading document index: {e}")
                self.document_index = {
                    "documents": {},
                    "categories": ["Uncategorized"]
                }
        else:
            logging.info("No existing document index found, creating new one")
            self.document_index = {
                "documents": {},
                "categories": ["Uncategorized"]
            }
            # Save the initial index
            try:
                with open(self.index_file, 'w') as f:
                    json.dump(self.document_index, f)
                logging.info("Created new document index file")
            except Exception as e:
                logging.error(f"Error creating document index file: {e}")
        
        # Initialize or load categorization model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the categorization model"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file):
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logging.info("Loaded existing categorization model")
            else:
                # Initial model with default parameters
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english'
                )
                self.model = KMeans(n_clusters=5, random_state=42)
                logging.info("Created new categorization model")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            # Fallback to new model
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
            self.model = KMeans(n_clusters=5, random_state=42)
    
    def _extract_text_from_pdf(self, file_path, timeout=60):
        """Extract text content from PDF files with timeout protection"""
        text = ""
        result = {"text": "", "success": False, "error": None}
        
        def extract_with_timeout():
            try:
                # First try direct PDF text extraction
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        result["text"] += page.extract_text() or ""
                
                # If no text was extracted or text is too short, try OCR
                if len(result["text"].strip()) < 100:
                    images = convert_from_path(file_path)
                    for i, image in enumerate(images):
                        page_text = pytesseract.image_to_string(image)
                        result["text"] += page_text
                
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)
                logging.error(f"Error extracting text from PDF: {e}")
        
        # Run extraction in a separate thread with timeout
        thread = threading.Thread(target=extract_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            logging.error(f"PDF extraction timed out after {timeout} seconds: {file_path}")
            return f"Error: PDF extraction timed out after {timeout} seconds. The file may be too large or complex."
        
        if not result["success"] and result["error"]:
            logging.error(f"Failed to extract text from PDF: {result['error']}")
            return f"Error extracting text: {result['error']}"
        
        return result["text"]
    
    def _extract_text_from_image(self, file_path, timeout=30):
        """Extract text content from image files using OCR with timeout protection"""
        result = {"text": "", "success": False, "error": None}
        
        def extract_with_timeout():
            try:
                image = Image.open(file_path)
                result["text"] = pytesseract.image_to_string(image)
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)
                logging.error(f"Error extracting text from image: {e}")
        
        # Run extraction in a separate thread with timeout
        thread = threading.Thread(target=extract_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            logging.error(f"Image OCR timed out after {timeout} seconds: {file_path}")
            return f"Error: OCR timed out after {timeout} seconds. The image may be too large or complex."
        
        if not result["success"] and result["error"]:
            logging.error(f"Failed to extract text from image: {result['error']}")
            return f"Error extracting text: {result['error']}"
        
        return result["text"]
    
    def _preprocess_text(self, text):
        """Clean and preprocess the extracted text"""
        # Handle error messages
        if text.startswith("Error:"):
            return text
            
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            
            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize and remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word not in stop_words]
            
            return ' '.join(filtered_tokens)
        except Exception as e:
            logging.error(f"Error preprocessing text: {e}")
            return text
    
    def _categorize_text(self, text):
        """Determine categories for the document based on content"""
        # Handle error messages
        if text.startswith("Error:"):
            return ["Error"]
            
        try:
            preprocessed_text = self._preprocess_text(text)
            
            # If we don't have any documents yet, use a default category
            if len(self.document_index["documents"]) < 5:
                if not self.document_index["categories"]:
                    self.document_index["categories"] = ["Uncategorized"]
                return ["Uncategorized"]
            
            # Transform the text using the vectorizer
            try:
                # Check if we need to fit the vectorizer first
                if len(self.document_index["documents"]) == 5:
                    # Get all texts from the document index
                    all_texts = [doc["preprocessed_text"] for doc in self.document_index["documents"].values()]
                    # Fit the vectorizer and model
                    text_vectors = self.vectorizer.fit_transform(all_texts)
                    self.model.fit(text_vectors)
                    
                    # Save the model and vectorizer
                    with open(self.model_file, 'wb') as f:
                        pickle.dump(self.model, f)
                    with open(self.vectorizer_file, 'wb') as f:
                        pickle.dump(self.vectorizer, f)
                    
                    # Generate category names based on top terms per cluster
                    self._generate_category_names()
                
                # Transform the text and predict cluster
                text_vector = self.vectorizer.transform([preprocessed_text])
                cluster = self.model.predict(text_vector)[0]
                
                # Return the corresponding category
                if 0 <= cluster < len(self.document_index["categories"]):
                    return [self.document_index["categories"][cluster]]
                else:
                    return ["Uncategorized"]
                
            except Exception as e:
                logging.error(f"Categorization error: {e}")
                return ["Uncategorized"]
        except Exception as e:
            logging.error(f"Error in categorize_text: {e}")
            return ["Error"]
    
    def _generate_category_names(self):
        """Generate category names based on important terms in each cluster"""
        try:
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_centers = self.model.cluster_centers_
            
            categories = []
            for i in range(len(cluster_centers)):
                # Get the indices of the top terms for this cluster
                top_term_indices = cluster_centers[i].argsort()[-3:][::-1]
                # Get the actual terms
                top_terms = [feature_names[idx] for idx in top_term_indices if idx < len(feature_names)]
                # Create category name
                category_name = f"Category_{i+1}: {', '.join(top_terms)}"
                categories.append(category_name)
            
            self.document_index["categories"] = categories
        except Exception as e:
            logging.error(f"Error generating category names: {e}")
            self.document_index["categories"] = [f"Category_{i+1}" for i in range(self.model.n_clusters)]
    
    def process(self, file_path):
        """Process a document: OCR, categorize, and index it"""
        start_time = time.time()
        logging.info(f"Starting document processing: {file_path}")
        
        # Generate a unique document ID
        doc_id = str(uuid.uuid4())
        
        try:
            # Extract text based on file type
            if file_path.lower().endswith('.pdf'):
                text = self._extract_text_from_pdf(file_path)
                logging.info(f"PDF text extraction completed in {time.time() - start_time:.2f}s")
            else:  # Image file
                text = self._extract_text_from_image(file_path)
                logging.info(f"Image OCR completed in {time.time() - start_time:.2f}s")
            
            # Skip if no text was extracted
            if not text or (text.startswith("Error:") and len(text) < 100):
                logging.error(f"No text could be extracted from the document: {file_path}")
                raise ValueError("No text could be extracted from the document")
            
            # Preprocess the text
            preprocessed_text = self._preprocess_text(text)
            logging.info(f"Text preprocessing completed in {time.time() - start_time:.2f}s")
            
            # Categorize the document
            categories = self._categorize_text(preprocessed_text)
            logging.info(f"Categorization completed in {time.time() - start_time:.2f}s")
            
            # Store document data
            file_name = os.path.basename(file_path)
            document_data = {
                "id": doc_id,
                "filename": file_name,
                "path": file_path,
                "categories": categories,
                "preprocessed_text": preprocessed_text,
                "full_text": text
            }
            
            # Add to document index
            self.document_index["documents"][doc_id] = document_data
            
            # Save updated index
            with open(self.index_file, 'w') as f:
                json.dump(self.document_index, f)
            
            logging.info(f"Document processing completed in {time.time() - start_time:.2f}s: {doc_id}")
            return doc_id, categories
            
        except Exception as e:
            logging.error(f"Error processing document {file_path}: {e}")
            # Return a temporary ID and error category
            return doc_id, ["Error: " + str(e)]
    
    def get_categories(self):
        """Return all available categories"""
        # Ensure we always have at least the Uncategorized category
        if not self.document_index["categories"]:
            self.document_index["categories"] = ["Uncategorized"]
            # Save the updated index
            try:
                with open(self.index_file, 'w') as f:
                    json.dump(self.document_index, f)
            except Exception as e:
                logging.error(f"Error saving document index: {e}")
                
        return self.document_index["categories"] 