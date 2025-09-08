"""
Text preprocessing utilities for cleaning and normalizing text.
"""
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK data: {e}")


class TextPreprocessor:
    """Handles text preprocessing and cleaning."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text: str) -> str:
        """
        Clean and preprocess the extracted text.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Handle error messages
        if text.startswith("Error:"):
            return text
            
        try:
            self.logger.info(f"Preprocessing text of length {len(text)}")
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            
            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize and remove stopwords
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word not in self.stop_words]
            
            processed_text = ' '.join(filtered_tokens)
            self.logger.info(f"Preprocessed text length: {len(processed_text)}")
            
            return processed_text
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> list:
        """
        Extract keywords from preprocessed text.
        
        Args:
            text: Preprocessed text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        try:
            words = text.split()
            # Get meaningful words (at least 4 characters)
            meaningful_words = [word for word in words if len(word) >= 4]
            
            if meaningful_words:
                from collections import Counter
                word_counts = Counter(meaningful_words)
                return [word for word, count in word_counts.most_common(max_keywords)]
            
            return []
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []