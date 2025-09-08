"""
Manages document categorization and category generation.
"""
import logging
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle


class CategoryManager:
    """Handles document categorization and category management."""
    
    def __init__(self, model_file: str, vectorizer_file: str):
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file
        self.logger = logging.getLogger(__name__)
        self.vectorizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the categorization model."""
        try:
            if self._model_files_exist():
                self._load_existing_model()
                self.logger.info("Loaded existing categorization model")
            else:
                self._create_new_model()
                self.logger.info("Created new categorization model")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            self._create_new_model()
    
    def _model_files_exist(self) -> bool:
        """Check if model files exist."""
        import os
        return os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file)
    
    def _load_existing_model(self):
        """Load existing model and vectorizer from files."""
        with open(self.model_file, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_file, 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def _create_new_model(self):
        """Create new model and vectorizer with default parameters."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        self.model = KMeans(n_clusters=8, random_state=42)
    
    def fit_model(self, texts: list, n_clusters: int = None):
        """
        Fit the model with given texts.
        
        Args:
            texts: List of preprocessed texts
            n_clusters: Number of clusters (optional)
        """
        try:
            if n_clusters and n_clusters != self.model.n_clusters:
                from sklearn.cluster import KMeans
                self.model = KMeans(n_clusters=n_clusters, random_state=42)
            
            text_vectors = self.vectorizer.fit_transform(texts)
            self.model.fit(text_vectors)
            
            self._save_model()
            self.logger.info(f"Model fitted with {len(texts)} documents")
        except Exception as e:
            self.logger.error(f"Error fitting model: {e}")
            raise
    
    def predict_category(self, text: str) -> int:
        """
        Predict category for given text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Category cluster number
        """
        try:
            if not hasattr(self.model, 'cluster_centers_'):
                return 0  # Default category if model not fitted
            
            text_vector = self.vectorizer.transform([text])
            cluster = self.model.predict(text_vector)[0]
            return cluster
        except Exception as e:
            self.logger.error(f"Error predicting category: {e}")
            return 0
    
    def generate_category_names(self, document_index: dict):
        """
        Generate category names based on important terms in each cluster.
        
        Args:
            document_index: Document index to update with categories
        """
        try:
            if not hasattr(self.model, 'cluster_centers_'):
                self.logger.warning("Model not fitted, cannot generate category names")
                return
            
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_centers = self.model.cluster_centers_
            
            # Get top terms for each cluster
            cluster_terms = {}
            for i in range(len(cluster_centers)):
                top_term_indices = cluster_centers[i].argsort()[-5:][::-1]
                top_terms = [feature_names[idx] for idx in top_term_indices if idx < len(feature_names)]
                cluster_terms[i] = top_terms
            
            # Create category names
            categories = self._create_category_names(cluster_terms)
            document_index["categories"] = categories
            
            # Generate structured categories
            structured_categories = self._create_structured_categories(categories)
            document_index["structured_categories"] = structured_categories
            
            self.logger.info(f"Generated {len(categories)} categories")
            
        except Exception as e:
            self.logger.error(f"Error generating category names: {e}")
            self._create_default_categories(document_index)
    
    def _create_category_names(self, cluster_terms: dict) -> list:
        """Create category names from cluster terms."""
        domain_prefixes = ["Document", "Report", "Analysis", "Research", "Paper", 
                          "Publication", "Article", "Study", "Review", "Guide"]
        
        all_category_names = set()
        categories = []
        
        for i, terms in cluster_terms.items():
            selected_terms = terms[:3]
            category_type = domain_prefixes[i % len(domain_prefixes)]
            base_name = f"{category_type}: {', '.join(selected_terms)}"
            
            # Ensure uniqueness
            category_name = base_name
            counter = 1
            while category_name in all_category_names:
                if len(terms) > 3 and counter <= len(terms) - 3:
                    category_name = f"{category_type}: {', '.join(selected_terms + [terms[2 + counter]])}"
                else:
                    category_name = f"{base_name} (Group {counter})"
                counter += 1
            
            all_category_names.add(category_name)
            categories.append(category_name)
        
        return categories
    
    def _create_structured_categories(self, categories: list) -> list:
        """Create structured category data."""
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
        
        return structured_categories
    
    def _create_default_categories(self, document_index: dict):
        """Create default categories if generation fails."""
        prefixes = ["Document", "Report", "Analysis", "Research", "Paper", 
                   "Publication", "Article", "Study"]
        document_index["categories"] = [f"{prefixes[i % len(prefixes)]} Group {i+1}" 
                                       for i in range(self.model.n_clusters)]
    
    def _save_model(self):
        """Save model and vectorizer to files."""
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_file, 'wb') as f:
            pickle.dump(self.vectorizer, f)