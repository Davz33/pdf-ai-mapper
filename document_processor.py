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

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DocumentProcessor:
    def __init__(self):
        self.processed_dir = "processed_data"
        self.index_file = os.path.join(self.processed_dir, "document_index.json")
        self.model_file = os.path.join(self.processed_dir, "category_model.pkl")
        self.vectorizer_file = os.path.join(self.processed_dir, "vectorizer.pkl")
        
        # Initialize or load document index
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                self.document_index = json.load(f)
        else:
            self.document_index = {
                "documents": {},
                "categories": []
            }
        
        # Initialize or load categorization model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the categorization model"""
        if os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file):
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_file, 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            # Initial model with default parameters
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
            self.model = KMeans(n_clusters=5, random_state=42)
    
    def _extract_text_from_pdf(self, file_path):
        """Extract text content from PDF files"""
        text = ""
        
        # First try direct PDF text extraction
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
        except Exception as e:
            print(f"Error extracting text directly from PDF: {e}")
        
        # If no text was extracted or text is too short, try OCR
        if len(text.strip()) < 100:
            try:
                images = convert_from_path(file_path)
                for i, image in enumerate(images):
                    page_text = pytesseract.image_to_string(image)
                    text += page_text
            except Exception as e:
                print(f"Error extracting text from PDF using OCR: {e}")
        
        return text
    
    def _extract_text_from_image(self, file_path):
        """Extract text content from image files using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""
    
    def _preprocess_text(self, text):
        """Clean and preprocess the extracted text"""
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
    
    def _categorize_text(self, text):
        """Determine categories for the document based on content"""
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
                self._generate_category_names(text_vectors)
            
            # Transform the text and predict cluster
            text_vector = self.vectorizer.transform([preprocessed_text])
            cluster = self.model.predict(text_vector)[0]
            
            # Return the corresponding category
            if 0 <= cluster < len(self.document_index["categories"]):
                return [self.document_index["categories"][cluster]]
            else:
                return ["Uncategorized"]
            
        except Exception as e:
            print(f"Categorization error: {e}")
            return ["Uncategorized"]
    
    def _generate_category_names(self, text_vectors):
        """Generate category names based on important terms in each cluster"""
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
    
    def process(self, file_path):
        """Process a document: OCR, categorize, and index it"""
        # Generate a unique document ID
        doc_id = str(uuid.uuid4())
        
        # Extract text based on file type
        if file_path.lower().endswith('.pdf'):
            text = self._extract_text_from_pdf(file_path)
        else:  # Image file
            text = self._extract_text_from_image(file_path)
        
        # Skip if no text was extracted
        if not text:
            raise ValueError("No text could be extracted from the document")
        
        # Preprocess the text
        preprocessed_text = self._preprocess_text(text)
        
        # Categorize the document
        categories = self._categorize_text(preprocessed_text)
        
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
        
        return doc_id, categories
    
    def get_categories(self):
        """Return all available categories"""
        return self.document_index["categories"] 