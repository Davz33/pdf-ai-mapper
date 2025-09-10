import os
import uuid
import json
import pytesseract
from PIL import Image
import pypdf
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
import traceback
import hashlib
import datetime

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
        
        self._pending_save = False
        self._save_lock = threading.Lock()
        
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
                self._save_document_index_immediate()
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
                self.model = KMeans(n_clusters=8, random_state=42)
                logging.info("Created new categorization model")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            # Fallback to new model
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
            self.model = KMeans(n_clusters=8, random_state=42)
    
    def _extract_text_from_pdf(self, file_path, timeout=120):
        """Extract text content from PDF files with timeout protection"""
        result = {"text": "", "success": False, "error": None}
        
        def extract_with_timeout():
            try:
                # First try direct PDF text extraction
                with open(file_path, 'rb') as f:
                    pdf_reader = pypdf.PdfReader(f)
                    logging.info(f"PDF has {len(pdf_reader.pages)} pages")
                    
                    # Process pages in batches to avoid memory issues
                    for page_num in range(len(pdf_reader.pages)):
                        if page_num % 10 == 0:
                            logging.info(f"Processing page {page_num} of {len(pdf_reader.pages)}")
                        try:
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text() or ""
                            result["text"] += page_text + "\n"
                        except Exception as e:
                            logging.error(f"Error extracting text from page {page_num}: {e}")
                
                # Log the amount of text extracted
                text_length = len(result["text"].strip())
                logging.info(f"Extracted {text_length} characters of text from PDF")
                
                # If no text was extracted or text is too short, try OCR on first few pages
                if text_length < 1000:
                    logging.info("Text extraction yielded limited results, trying OCR")
                    try:
                        # Only convert first 5 pages to images to save time
                        max_pages = min(5, len(pdf_reader.pages))
                        images = convert_from_path(file_path, first_page=1, last_page=max_pages)
                        logging.info(f"Converted {len(images)} pages to images for OCR")
                        
                        for i, image in enumerate(images):
                            logging.info(f"Running OCR on page {i+1}")
                            page_text = pytesseract.image_to_string(image)
                            result["text"] += page_text + "\n"
                    except Exception as e:
                        logging.error(f"Error during OCR processing: {e}")
                
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
            # Try to get whatever text was extracted before timeout
            if result["text"]:
                logging.info(f"Using partial text extracted before timeout ({len(result['text'])} characters)")
                return result["text"]
            return f"Error: PDF extraction timed out after {timeout} seconds. The file may be too large or complex."
        
        if not result["success"] and result["error"]:
            logging.error(f"Failed to extract text from PDF: {result['error']}")
            return f"Error extracting text: {result['error']}"
        
        # If we got some text, even if there were errors, return it
        if result["text"].strip():
            return result["text"]
        else:
            return "Error: No text could be extracted from the PDF"
    
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
            # Log the original text length
            logging.info(f"Preprocessing text of length {len(text)}")
            
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
            
            processed_text = ' '.join(filtered_tokens)
            logging.info(f"Preprocessed text length: {len(processed_text)}")
            
            return processed_text
        except Exception as e:
            logging.error(f"Error preprocessing text: {e}")
            return text
    
    def _categorize_text(self, text):
        """Determine categories for the document based on content"""
        # Handle error messages
        if text.startswith("Error:"):
            logging.warning(f"Categorizing text with error: {text[:100]}...")
            return ["Error"]
            
        try:
            preprocessed_text = self._preprocess_text(text)
            
            # Count documents
            doc_count = len(self.document_index["documents"])
            logging.info(f"Current document count: {doc_count}")
            
            # For the first few documents, create simple categories based on content
            if doc_count < 5:
                logging.info(f"Not enough documents for clustering ({doc_count}/5), creating simple category")
                
                # Create a simple category based on document content
                words = preprocessed_text.split()
                # Get the most common meaningful words (at least 4 characters)
                common_words = [word for word in words if len(word) >= 4]
                
                if common_words:
                    # Take up to 3 common words for the category name
                    from collections import Counter
                    word_counts = Counter(common_words)
                    top_words = [word for word, count in word_counts.most_common(3)]
                    
                    if top_words:
                        category_name = f"Topic: {', '.join(top_words)}"
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
            
            # If we have exactly 5 documents, fit the model
            if doc_count == 5:
                logging.info("Reached 5 documents, fitting vectorizer and model")
                try:
                    # Get all texts from the document index with fallback handling
                    all_texts = []
                    for doc in self.document_index["documents"].values():
                        if "preprocessed_text" in doc and doc["preprocessed_text"]:
                            all_texts.append(doc["preprocessed_text"])
                        elif "full_text" in doc and doc["full_text"]:
                            all_texts.append(doc["full_text"])
                        elif "content_file" in doc and doc["content_file"]:
                            try:
                                with open(doc["content_file"], 'r', encoding='utf-8') as f:
                                    content = f.read()
                                all_texts.append(content)
                            except Exception as e:
                                logging.error(f"Error loading content file {doc['content_file']}: {e}")
                        else:
                            logging.warning(f"Document {doc.get('id', 'unknown')} has no text content, skipping")
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
                except Exception as e:
                    logging.error(f"Error fitting model: {e}")
                    logging.error(traceback.format_exc())
            
            # If we have more than 5 documents and the model is fitted
            if doc_count >= 5 and hasattr(self.model, 'cluster_centers_'):
                try:
                    # Transform the text and predict cluster
                    text_vector = self.vectorizer.transform([preprocessed_text])
                    cluster = self.model.predict(text_vector)[0]
                    logging.info(f"Document assigned to cluster {cluster}")
                    
                    # Store cluster ID for later use in category generation (convert numpy int to Python int)
                    self._current_cluster_id = int(cluster)
                    
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
                    logging.error(traceback.format_exc())
                    return ["Uncategorized"]
            else:
                # Model not fitted yet, use Uncategorized
                logging.info("Model not fitted yet, using Uncategorized")
                return ["Uncategorized"]
                
        except Exception as e:
            logging.error(f"Error in categorize_text: {e}")
            logging.error(traceback.format_exc())
            return ["Error"]
    
    def _generate_category_names(self):
        """Generate meaningful category names based on document content analysis"""
        try:
            # Get all documents for analysis
            documents = self.document_index.get("documents", {})
            logging.info(f"Generating category names for {len(documents)} documents")
            if not documents:
                logging.warning("No documents found for category generation")
                return
            
            # Analyze document content to generate meaningful categories
            cluster_assignments = self.model.labels_
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_centers = self.model.cluster_centers_
            
            # Group documents by cluster using model labels
            cluster_docs = {}
            doc_list = list(documents.values())
            
            # Get all texts for clustering
            all_texts = []
            for doc in doc_list:
                if "preprocessed_text" in doc and doc["preprocessed_text"]:
                    all_texts.append(doc["preprocessed_text"])
                elif "full_text" in doc and doc["full_text"]:
                    all_texts.append(doc["full_text"])
                elif "content_file" in doc and doc["content_file"]:
                    try:
                        with open(doc["content_file"], 'r', encoding='utf-8') as f:
                            content = f.read()
                        all_texts.append(content)
                    except Exception as e:
                        logging.error(f"Error loading content file {doc['content_file']}: {e}")
                        all_texts.append("")
                else:
                    all_texts.append("")
            
            # Transform texts and get cluster assignments
            if all_texts:
                text_vectors = self.vectorizer.transform(all_texts)
                cluster_assignments = self.model.predict(text_vectors)
                
                # Group documents by cluster
                for i, doc in enumerate(doc_list):
                    cluster_id = cluster_assignments[i]
                    if cluster_id not in cluster_docs:
                        cluster_docs[cluster_id] = []
                    cluster_docs[cluster_id].append(doc)
            
            categories = []
            all_category_names = set()
            
            for cluster_id in range(len(cluster_centers)):
                if cluster_id not in cluster_docs:
                    continue
                    
                cluster_documents = cluster_docs[cluster_id]
                
                # Analyze the content of documents in this cluster
                category_info = self._analyze_cluster_content(cluster_documents, cluster_id, feature_names, cluster_centers)
                
                # Create meaningful category name
                category_name = f"{category_info['type']}: {category_info['description']}"
                
                # Ensure uniqueness
                base_name = category_name
                counter = 1
                while category_name in all_category_names:
                    category_name = f"{base_name} (Group {counter})"
                    counter += 1
                
                all_category_names.add(category_name)
                categories.append(category_name)
            
            logging.info(f"Generated meaningful categories: {categories}")
            self.document_index["categories"] = categories
            
        except Exception as e:
            logging.error(f"Error generating category names: {e}")
            logging.error(traceback.format_exc())
            # Create more descriptive default categories
            prefixes = ["Document", "Report", "Analysis", "Research", "Paper", 
                       "Publication", "Article", "Study"]
            n_clusters = getattr(self.model, 'n_clusters', 8)
            self.document_index["categories"] = [f"{prefixes[i % len(prefixes)]} Group {i+1}" 
                                               for i in range(n_clusters)]
    
    def _analyze_cluster_content(self, documents, cluster_id, feature_names, cluster_centers):
        """Analyze documents in a cluster to generate meaningful category information"""
        try:
            # Get top TF-IDF terms for this cluster
            top_term_indices = cluster_centers[cluster_id].argsort()[-10:][::-1]
            top_terms = [feature_names[idx] for idx in top_term_indices if idx < len(feature_names)]
            
            # Filter out meaningless short terms and common words
            meaningful_terms = []
            stop_words = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'])
            
            for term in top_terms:
                if len(term) > 2 and term.lower() not in stop_words and not term.isdigit():
                    meaningful_terms.append(term)
            
            # Take top 3-5 meaningful terms
            selected_terms = meaningful_terms[:5] if len(meaningful_terms) >= 5 else meaningful_terms
            
            # Analyze document types and content themes
            content_themes = self._identify_content_themes(documents)
            
            # Determine category type based on content analysis
            category_type = self._determine_category_type(documents, content_themes)
            
            # Create description from meaningful terms and themes
            if selected_terms:
                description = ', '.join(selected_terms[:3])
            else:
                description = content_themes.get('primary_theme', 'General Content')
            
            return {
                'type': category_type,
                'description': description,
                'keywords': selected_terms,
                'themes': content_themes
            }
            
        except Exception as e:
            logging.error(f"Error analyzing cluster content: {e}")
            return {
                'type': 'Document',
                'description': 'General Content',
                'keywords': [],
                'themes': {}
            }
    
    def _identify_content_themes(self, documents):
        """Identify common themes in a cluster of documents"""
        themes = {
            'primary_theme': 'General',
            'document_types': [],
            'subject_areas': []
        }
        
        try:
            # Analyze filenames and content for themes
            for doc in documents:
                filename = doc.get('filename', '').lower()
                content = doc.get('preprocessed_text', '').lower()
                
                # Identify document types from filename patterns
                if any(word in filename for word in ['report', 'analysis', 'study']):
                    themes['document_types'].append('Report')
                elif any(word in filename for word in ['manual', 'guide', 'instruction']):
                    themes['document_types'].append('Manual')
                elif any(word in filename for word in ['research', 'paper', 'article']):
                    themes['document_types'].append('Research')
                
                # Identify subject areas from content
                if any(word in content for word in ['philosophy', 'knowledge', 'theory', 'concept']):
                    themes['subject_areas'].append('Philosophy')
                elif any(word in content for word in ['history', 'historical', 'past', 'ancient']):
                    themes['subject_areas'].append('History')
                elif any(word in content for word in ['science', 'scientific', 'research', 'experiment']):
                    themes['subject_areas'].append('Science')
                elif any(word in content for word in ['literature', 'novel', 'story', 'fiction']):
                    themes['subject_areas'].append('Literature')
            
            # Determine primary theme
            if themes['subject_areas']:
                from collections import Counter
                most_common = Counter(themes['subject_areas']).most_common(1)[0][0]
                themes['primary_theme'] = most_common
            elif themes['document_types']:
                from collections import Counter
                most_common = Counter(themes['document_types']).most_common(1)[0][0]
                themes['primary_theme'] = most_common
                
        except Exception as e:
            logging.error(f"Error identifying content themes: {e}")
            
        return themes
    
    def _determine_category_type(self, documents, content_themes):
        """Determine the most appropriate category type based on document analysis"""
        try:
            # Count document types
            doc_type_counts = {}
            for doc in documents:
                filename = doc.get('filename', '').lower()
                if any(word in filename for word in ['report', 'analysis']):
                    doc_type_counts['Report'] = doc_type_counts.get('Report', 0) + 1
                elif any(word in filename for word in ['research', 'paper', 'study']):
                    doc_type_counts['Research'] = doc_type_counts.get('Research', 0) + 1
                elif any(word in filename for word in ['manual', 'guide']):
                    doc_type_counts['Guide'] = doc_type_counts.get('Guide', 0) + 1
                else:
                    doc_type_counts['Document'] = doc_type_counts.get('Document', 0) + 1
            
            # Return most common type, or use content theme
            if doc_type_counts:
                most_common_type = max(doc_type_counts, key=doc_type_counts.get)
                return most_common_type
            else:
                # Use content theme to determine type
                primary_theme = content_themes.get('primary_theme', 'General')
                theme_to_type = {
                    'Philosophy': 'Analysis',
                    'History': 'Research', 
                    'Science': 'Research',
                    'Literature': 'Document',
                    'General': 'Document'
                }
                return theme_to_type.get(primary_theme, 'Document')
                
        except Exception as e:
            logging.error(f"Error determining category type: {e}")
            return 'Document'
    
    def _generate_soft_categories(self, text_vector, threshold=0.3):
        """Generate soft categories allowing documents to belong to multiple categories"""
        try:
            if not hasattr(self.model, 'cluster_centers_'):
                return ["Uncategorized"]
            
            # Calculate distances to all cluster centers
            distances = []
            for i, center in enumerate(self.model.cluster_centers_):
                # Calculate cosine similarity (higher is better)
                similarity = np.dot(text_vector.toarray()[0], center) / (np.linalg.norm(text_vector.toarray()[0]) * np.linalg.norm(center))
                distances.append((i, similarity))
            
            # Sort by similarity (descending)
            distances.sort(key=lambda x: x[1], reverse=True)
            
            # Get categories above threshold
            categories = self.document_index.get("categories", ["Uncategorized"])
            selected_categories = []
            
            for cluster_id, similarity in distances:
                cluster_id = int(cluster_id)  # Convert numpy int to Python int
                if similarity >= threshold and cluster_id < len(categories):
                    selected_categories.append(categories[cluster_id])
            
            # If no categories meet threshold, return the top one
            if not selected_categories and distances:
                top_cluster = int(distances[0][0])  # Convert numpy int to Python int
                if top_cluster < len(categories):
                    selected_categories = [categories[top_cluster]]
            
            return selected_categories if selected_categories else ["Uncategorized"]
            
        except Exception as e:
            logging.error(f"Error in soft categorization: {e}")
            return ["Uncategorized"]
    
    def _separate_content_storage(self, document_data):
        """Separate content storage from metadata to reduce JSON file size"""
        try:
            # Extract content that should be stored separately
            content_data = {
                "full_text": document_data.get("full_text", ""),
                "preprocessed_text": document_data.get("preprocessed_text", "")
            }
            
            # Create content file path
            content_dir = os.path.join(self.processed_dir, "content")
            os.makedirs(content_dir, exist_ok=True)
            content_file = os.path.join(content_dir, f"{document_data['id']}.txt")
            
            # Store content in separate file
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(f"FULL_TEXT:\n{document_data.get('full_text', '')}\n\n")
                f.write(f"PREPROCESSED_TEXT:\n{document_data.get('preprocessed_text', '')}\n")
            
            # Remove content from document data, keep only references
            document_data.pop("full_text", None)
            document_data.pop("preprocessed_text", None)
            document_data["content_file"] = content_file
            
            return document_data
            
        except Exception as e:
            logging.error(f"Error separating content storage: {e}")
            return document_data
    
    def _load_content_from_file(self, content_file):
        """Load content from separate file"""
        try:
            if not os.path.exists(content_file):
                return {"full_text": "", "preprocessed_text": ""}
            
            with open(content_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the content
            parts = content.split("PREPROCESSED_TEXT:\n")
            full_text = parts[0].replace("FULL_TEXT:\n", "").strip()
            preprocessed_text = parts[1].strip() if len(parts) > 1 else ""
            
            return {
                "full_text": full_text,
                "preprocessed_text": preprocessed_text
            }
            
        except Exception as e:
            logging.error(f"Error loading content from file: {e}")
            return {"full_text": "", "preprocessed_text": ""}
            
            # For future enterprise use, also store structured category data
            # This doesn't affect current functionality but prepares for future enhancements
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
                        "created_at": datetime.datetime.now().isoformat()
                    }
                    structured_categories.append(structured_category)
                    logging.info(f"Created structured category: {structured_category}")
            
            # Store for future use (doesn't affect current functionality)
            self.document_index["structured_categories"] = structured_categories
            logging.info(f"Added {len(structured_categories)} structured categories to document index")
            
            self._mark_for_save()
            logging.info("Marked document index for save with structured categories")
            
        except Exception as e:
            logging.error(f"Error generating category names: {e}")
            logging.error(traceback.format_exc())
            # Create more descriptive default categories
            prefixes = ["Document", "Report", "Analysis", "Research", "Paper", 
                       "Publication", "Article", "Study"]
            n_clusters = getattr(self.model, 'n_clusters', 8)
            self.document_index["categories"] = [f"{prefixes[i % len(prefixes)]} Group {i+1}" 
                                               for i in range(n_clusters)]
    
    def _calculate_content_hash(self, text):
        """Calculate a hash of the document content for duplicate detection"""
        # Use a prefix of the text to create a hash
        # This allows for minor differences while still catching duplicates
        text_for_hash = text[:5000].strip().lower()
        return hashlib.md5(text_for_hash.encode('utf-8')).hexdigest()
    
    def _find_duplicate_document(self, file_name, text):
        """
        Check if a document with the same filename or similar content already exists
        Returns the document ID if a duplicate is found, None otherwise
        """
        if not text:
            return None
            
        # Calculate content hash
        content_hash = self._calculate_content_hash(text)
        logging.info(f"Content hash for potential duplicate check: {content_hash}")
        
        # First check for exact filename match
        for doc_id, doc in self.document_index["documents"].items():
            if doc["filename"] == file_name:
                logging.info(f"Found duplicate by filename: {file_name}")
                return doc_id
        
        # Then check for content similarity
        for doc_id, doc in self.document_index["documents"].items():
            if "content_hash" in doc and doc["content_hash"] == content_hash:
                logging.info(f"Found duplicate by content hash: {content_hash}")
                return doc_id
            
        # No duplicate found
        return None
    
    def clean_up_duplicates(self):
        """
        Clean up duplicate documents in the index
        Returns the number of duplicates removed
        """
        logging.info("Starting duplicate cleanup process")
        
        # First, ensure all documents have a content hash
        for doc_id, doc in list(self.document_index["documents"].items()):
            if "content_hash" not in doc and "full_text" in doc:
                doc["content_hash"] = self._calculate_content_hash(doc["full_text"])
                logging.info(f"Added content hash for document {doc_id}")
        
        # Track documents by content hash
        docs_by_hash = {}
        duplicates_to_remove = []
        
        # First pass: identify duplicates
        for doc_id, doc in self.document_index["documents"].items():
            if "content_hash" not in doc:
                logging.warning(f"Document {doc_id} has no content hash, skipping")
                continue
                
            content_hash = doc["content_hash"]
            
            if content_hash in docs_by_hash:
                # This is a duplicate
                duplicates_to_remove.append(doc_id)
                logging.info(f"Identified duplicate document: {doc_id} (same as {docs_by_hash[content_hash]})")
            else:
                # This is the first occurrence of this hash
                docs_by_hash[content_hash] = doc_id
        
        # Second pass: remove duplicates
        for doc_id in duplicates_to_remove:
            del self.document_index["documents"][doc_id]
            logging.info(f"Removed duplicate document: {doc_id}")
        
        if duplicates_to_remove:
            self._mark_for_save()
            logging.info(f"Marked document index for save after removing {len(duplicates_to_remove)} duplicates")
        
        return len(duplicates_to_remove)
    
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
            
            # Check for duplicate documents
            file_name = os.path.basename(file_path)
            existing_doc_id = self._find_duplicate_document(file_name, text)
            
            if existing_doc_id:
                logging.info(f"Duplicate document detected. Using existing document ID: {existing_doc_id}")
                
                # Update the existing document with new path if needed
                self.document_index["documents"][existing_doc_id]["path"] = file_path
                
                self._mark_for_save()
                
                # Return the existing document ID and its categories
                return existing_doc_id, self.document_index["documents"][existing_doc_id]["categories"]
            
            # Log a sample of the extracted text
            text_sample = text[:500].replace('\n', ' ').strip()
            logging.info(f"Extracted text sample: {text_sample}...")
            
            # Preprocess the text
            preprocessed_text = self._preprocess_text(text)
            logging.info(f"Text preprocessing completed in {time.time() - start_time:.2f}s")
            
            # Categorize the document
            categories = self._categorize_text(preprocessed_text)
            logging.info(f"Categorization completed in {time.time() - start_time:.2f}s")
            
            # Get cluster ID if available
            cluster_id = getattr(self, '_current_cluster_id', None)
            
            # Calculate content hash for future duplicate detection
            content_hash = self._calculate_content_hash(text)
            
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
            
            # Add cluster ID if available (convert numpy int to Python int)
            if cluster_id is not None:
                document_data["cluster_id"] = int(cluster_id)
            
            # Separate content storage to reduce JSON file size
            document_data = self._separate_content_storage(document_data)
            
            # Add to document index
            self.document_index["documents"][doc_id] = document_data
            
            self._mark_for_save()
            
            logging.info(f"Document processing completed in {time.time() - start_time:.2f}s: {doc_id}")
            return doc_id, categories
            
        except Exception as e:
            logging.error(f"Error processing document {file_path}: {e}")
            logging.error(traceback.format_exc())
            # Return a temporary ID and error category
            return doc_id, ["Error: " + str(e)]
    
    def get_categories(self):
        """Get all available categories"""
        try:
            return self.document_index["categories"]
        except Exception as e:
            logging.error(f"Error getting categories: {e}")
            return []
            
    def generate_structured_categories(self):
        """Generate structured categories using the new meaningful category generation"""
        try:
            # First, regenerate categories with the new logic
            self._generate_category_names()
            
            categories = self.document_index.get("categories", [])
            if not categories:
                logging.warning("No categories found to structure")
                return []
            
            # Get documents for analysis
            documents = self.document_index.get("documents", {})
            if not documents:
                logging.warning("No documents found for structured category generation")
                return []
            
            # Group documents by cluster for analysis
            cluster_docs = {}
            for doc_id, doc in documents.items():
                if 'cluster_id' in doc:
                    cluster_id = doc['cluster_id']
                    if cluster_id not in cluster_docs:
                        cluster_docs[cluster_id] = []
                    cluster_docs[cluster_id].append(doc)
            
            structured_categories = []
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_centers = self.model.cluster_centers_
            
            for i, category_name in enumerate(categories):
                # Analyze cluster content for meaningful keywords
                cluster_id = i
                if cluster_id in cluster_docs:
                    cluster_documents = cluster_docs[cluster_id]
                    category_info = self._analyze_cluster_content(cluster_documents, cluster_id, feature_names, cluster_centers)
                    
                    structured_category = {
                        "id": f"cat-{i+1:03d}",
                        "type": category_info['type'],
                        "keywords": category_info['keywords'],
                        "display_name": category_name,
                        "created_at": datetime.datetime.now().isoformat()
                    }
                else:
                    # Fallback to parsing the category name
                    parts = category_name.split(": ", 1)
                    if len(parts) == 2:
                        category_type = parts[0]
                        keywords = [k.strip() for k in parts[1].split(",")]
                    else:
                        category_type = "Document"
                        keywords = []
                    
                    structured_category = {
                        "id": f"cat-{i+1:03d}",
                        "type": category_type,
                        "keywords": keywords,
                        "display_name": category_name,
                        "created_at": datetime.datetime.now().isoformat()
                    }
                
                structured_categories.append(structured_category)
                logging.info(f"Created structured category: {structured_category}")
            
            # Store the structured categories
            self.document_index["structured_categories"] = structured_categories
            logging.info(f"Added {len(structured_categories)} structured categories to document index")
            
            self._mark_for_save()
            logging.info("Marked document index for save with structured categories")
            
            return structured_categories
        except Exception as e:
            logging.error(f"Error generating structured categories: {e}")
            logging.error(traceback.format_exc())
            return []
    
    def _mark_for_save(self):
        """Mark the document index for batched saving"""
        with self._save_lock:
            self._pending_save = True
    
    def _save_document_index_immediate(self):
        """Immediately save the document index to file"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.document_index, f)
            logging.info("Saved document index immediately")
        except Exception as e:
            logging.error(f"Error saving document index: {e}")
    
    def _save_document_index(self):
        """Save the document index to file (legacy method for compatibility)"""
        self._save_document_index_immediate()
    
    def flush_pending_saves(self):
        """Flush any pending saves to disk"""
        with self._save_lock:
            if self._pending_save:
                try:
                    with open(self.index_file, 'w') as f:
                        json.dump(self.document_index, f)
                    self._pending_save = False
                    logging.info("Flushed pending document index save")
                except Exception as e:
                    logging.error(f"Error flushing document index save: {e}")
                    raise     