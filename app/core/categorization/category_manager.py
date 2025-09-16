"""
Category management module for PDF AI Mapper.
Handles document categorization using LDA and topic analysis.
"""

import logging
import traceback
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import pickle


class CategoryManager:
    """Manages document categorization using LDA topic modeling."""
    
    def __init__(self, model_file, vectorizer_file):
        self.logger = logging.getLogger(__name__)
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file
        self.stop_words = set(stopwords.words('english'))
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the categorization model."""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file):
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.logger.info("Loaded existing categorization model")
            else:
                # Initial model with default parameters
                self.vectorizer = CountVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 3)
                )
                self.model = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=100)
                self.logger.info("Created new categorization model")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            # Fallback to new model
            self.vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3)
            )
            self.model = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=100)
    
    def categorize_text(self, text, document_index, preprocessor):
        """Determine categories for the document based on content."""
        # Handle error messages
        if text.startswith("Error:"):
            self.logger.warning(f"Categorizing text with error: {text[:100]}...")
            return ["Error"]
            
        try:
            preprocessed_text = preprocessor.preprocess_text(text)
            
            # Count documents
            doc_count = len(document_index["documents"])
            self.logger.info(f"Current document count: {doc_count}")
            
            # For the first few documents, create simple categories based on content
            if doc_count < 5:
                self.logger.info(f"Not enough documents for clustering ({doc_count}/5), creating simple category")
                return self._create_simple_category(preprocessed_text, document_index)
            
            # If we have exactly 5 documents, fit the model
            if doc_count == 5:
                self.logger.info("Reached 5 documents, fitting vectorizer and model")
                self._fit_model(document_index)
            
            # If we have more than 5 documents and the model is fitted
            if doc_count >= 5 and hasattr(self.model, 'components_'):
                return self._categorize_with_lda(preprocessed_text)
            else:
                # Model not fitted yet, use Uncategorized
                self.logger.info("Model not fitted yet, using Uncategorized")
                return ["Uncategorized"]
                
        except Exception as e:
            self.logger.error(f"Error in categorize_text: {e}")
            self.logger.error(traceback.format_exc())
            return ["Error"]
    
    def _create_simple_category(self, preprocessed_text, document_index):
        """Create a simple category based on document content."""
        words = preprocessed_text.split()
        # Get the most common meaningful words (at least 4 characters)
        common_words = [word for word in words if len(word) >= 4]
        
        if common_words:
            # Take up to 3 common words for the category name
            word_counts = Counter(common_words)
            top_words = [word for word, count in word_counts.most_common(3)]
            
            if top_words:
                category_name = f"Topic: {', '.join(top_words)}"
                self.logger.info(f"Created simple category: {category_name}")
                
                # Add this category if it doesn't exist
                if category_name not in document_index["categories"]:
                    document_index["categories"].append(category_name)
                    self.logger.info(f"Added new category: {category_name}")
                
                return [category_name]
        
        # Fallback to Uncategorized
        if not document_index["categories"]:
            document_index["categories"] = ["Uncategorized"]
        return ["Uncategorized"]
    
    def _fit_model(self, document_index):
        """Fit the LDA model with existing documents."""
        try:
            # Get all texts from the document index with fallback handling
            all_texts = []
            for doc in document_index["documents"].values():
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
                        self.logger.error(f"Error loading content file {doc['content_file']}: {e}")
                else:
                    self.logger.warning(f"Document {doc.get('id', 'unknown')} has no text content, skipping")
            
            # Fit the vectorizer and model
            text_vectors = self.vectorizer.fit_transform(all_texts)
            self.model.fit(text_vectors)
            
            # Save the model and vectorizer
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.vectorizer_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Generate category names based on top terms per cluster
            self._generate_category_names(document_index)
        except Exception as e:
            self.logger.error(f"Error fitting model: {e}")
            self.logger.error(traceback.format_exc())
    
    def _categorize_with_lda(self, preprocessed_text):
        """Categorize text using the fitted LDA model."""
        try:
            # Transform the text and get topic probabilities
            text_vector = self.vectorizer.transform([preprocessed_text])
            topic_probs = self.model.transform(text_vector)[0]
            
            # Get top topics with probability > threshold
            threshold = 0.1
            top_topics = []
            for i, prob in enumerate(topic_probs):
                if prob > threshold:
                    top_topics.append((i, prob))
            
            # Sort by probability
            top_topics.sort(key=lambda x: x[1], reverse=True)
            
            # Generate category names for top topics
            categories = []
            for topic_id, prob in top_topics[:3]:  # Top 3 topics
                category_name = self._get_lda_topic_name(topic_id)
                if category_name:
                    categories.append(category_name)
            
            if not categories:
                categories = ["Uncategorized"]
            
            self.logger.info(f"Document assigned to categories: {categories}")
            return categories
        except Exception as e:
            self.logger.error(f"Error in LDA categorization: {e}")
            self.logger.error(traceback.format_exc())
            return ["Uncategorized"]
    
    def _generate_category_names(self, document_index):
        """Generate meaningful category names based on LDA topic analysis."""
        try:
            # Get all documents for analysis
            documents = document_index.get("documents", {})
            self.logger.info(f"Generating category names for {len(documents)} documents")
            if not documents:
                self.logger.warning("No documents found for category generation")
                return
            
            # Check if we have LDA model with components
            if not hasattr(self.model, 'components_'):
                self.logger.warning("Model not fitted or not LDA model")
                return
            
            # Generate category names for each LDA topic
            feature_names = self.vectorizer.get_feature_names_out()
            n_topics = self.model.n_components
            
            # Clear existing categories
            document_index["categories"] = []
            
            # Generate meaningful names for each topic
            for topic_id in range(n_topics):
                category_name = self._get_lda_topic_name(topic_id)
                if category_name:
                    document_index["categories"].append(category_name)
                    self.logger.info(f"Generated category: {category_name}")
            
            self.logger.info(f"Generated {len(document_index['categories'])} categories")
            
        except Exception as e:
            self.logger.error(f"Error generating category names: {e}")
            self.logger.error(traceback.format_exc())
    
    def _get_lda_topic_name(self, topic_id):
        """Generate a meaningful name for an LDA topic."""
        try:
            if not hasattr(self.model, 'components_'):
                return None
            
            # Get top terms for this topic
            feature_names = self.vectorizer.get_feature_names_out()
            topic_weights = self.model.components_[topic_id]
            top_indices = topic_weights.argsort()[-10:][::-1]  # Top 10 terms
            top_terms = [feature_names[i] for i in top_indices]
            
            # Filter out meaningless terms
            meaningful_terms = []
            meaningless_words = {'like', 'ofthe', 'things', 'posterior', 'anterior', 'surface'}
            
            for term in top_terms:
                if (len(term) >= 3 and 
                    term.lower() not in self.stop_words and 
                    term.lower() not in meaningless_words and
                    not term.isdigit()):
                    meaningful_terms.append(term)
            
            # Take top 5-8 meaningful terms
            selected_terms = meaningful_terms[:8] if len(meaningful_terms) >= 8 else meaningful_terms
            
            if not selected_terms:
                return f"Topic {topic_id + 1}"
            
            # Determine topic type based on terms
            topic_type = self._determine_topic_type(selected_terms)
            
            # Create hierarchical category name
            if len(selected_terms) >= 3:
                category_name = f"{topic_type}: {', '.join(selected_terms[:5])}"
            else:
                category_name = f"{topic_type}: {', '.join(selected_terms)}"
            
            return category_name
            
        except Exception as e:
            self.logger.error(f"Error generating LDA topic name: {e}")
            return f"Topic {topic_id + 1}"
    
    def _determine_topic_type(self, terms):
        """Determine the type of topic based on terms."""
        try:
            # Convert to lowercase for comparison
            terms_lower = [term.lower() for term in terms]
            
            # Define topic type keywords
            topic_keywords = {
                'Philosophy': ['philosophy', 'philosophical', 'ethics', 'moral', 'virtue', 'justice', 'kant', 'aristotle', 'plato'],
                'Science': ['science', 'scientific', 'research', 'study', 'mathematics', 'geometry', 'theorem', 'proof', 'euclid', 'mathematical', 'physics', 'chemistry', 'biology'],
                'Literature': ['literature', 'literary', 'novel', 'story', 'fiction', 'poetry', 'poem', 'author', 'writer', 'book', 'chapter', 'character'],
                'History': ['history', 'historical', 'ancient', 'classical', 'empire', 'war', 'battle', 'century', 'period', 'civilization'],
                'Technology': ['technology', 'technical', 'programming', 'computer', 'software', 'hardware', 'algorithm', 'data', 'system', 'digital'],
                'Art': ['art', 'artistic', 'painting', 'sculpture', 'design', 'creative', 'aesthetic', 'beauty', 'artist', 'gallery'],
                'Medicine': ['medicine', 'medical', 'health', 'disease', 'treatment', 'patient', 'doctor', 'hospital', 'surgery', 'anatomy'],
                'Economics': ['economics', 'economic', 'financial', 'money', 'business', 'market', 'trade', 'commerce', 'industry', 'capital']
            }
            
            # Count matches for each topic type
            topic_scores = {}
            for topic_type, keywords in topic_keywords.items():
                score = sum(1 for term in terms_lower if any(keyword in term for keyword in keywords))
                topic_scores[topic_type] = score
            
            # Return the topic type with highest score, or default to "Document"
            if topic_scores:
                best_topic = max(topic_scores, key=topic_scores.get)
                if topic_scores[best_topic] > 0:
                    return best_topic
            
            return "Document"
            
        except Exception as e:
            self.logger.error(f"Error determining topic type: {e}")
            return "Document"