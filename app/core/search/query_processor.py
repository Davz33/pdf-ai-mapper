"""
Query processing utilities for search functionality.
"""
import re
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class QueryProcessor:
    """Handles search query preprocessing and processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_query(self, query: str) -> list:
        """
        Preprocess the search query similar to document text.
        
        Args:
            query: Raw search query
            
        Returns:
            List of processed tokens
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Tokenize
        tokens = word_tokenize(query)
        
        # Remove stopwords and apply stemming
        processed_tokens = [
            self.stemmer.stem(word) 
            for word in tokens 
            if word not in self.stop_words
        ]
        
        return processed_tokens
    
    def is_valid_query(self, tokens: list) -> bool:
        """
        Check if query has valid tokens after preprocessing.
        
        Args:
            tokens: List of processed tokens
            
        Returns:
            True if query is valid, False otherwise
        """
        return len(tokens) > 0