import os
import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class SearchEngine:
    def __init__(self):
        self.processed_dir = "processed_data"
        self.index_file = os.path.join(self.processed_dir, "document_index.json")
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load document index
        self.load_index()
    
    def load_index(self):
        """Load the document index"""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                self.document_index = json.load(f)
        else:
            self.document_index = {"documents": {}, "categories": []}
    
    def _preprocess_query(self, query):
        """Preprocess the search query similar to document text"""
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
    
    def search(self, query, categories=None, max_results=10):
        """
        Search for documents matching the query
        
        Args:
            query (str): The search query
            categories (list): Optional list of categories to filter by
            max_results (int): Maximum number of results to return
            
        Returns:
            list: Matching documents with relevance scores
        """
        # Refresh the index in case new documents were added
        self.load_index()
        
        # Preprocess the query
        query_tokens = self._preprocess_query(query)
        
        # If no query tokens after preprocessing, return empty results
        if not query_tokens:
            return []
        
        results = []
        content_hashes_seen = set()  # Track content hashes to avoid duplicates
        
        # Get structured categories if they exist
        structured_categories_dict = {}
        if "structured_categories" in self.document_index:
            for cat in self.document_index["structured_categories"]:
                structured_categories_dict[cat["display_name"]] = cat
        
        # Search through all documents
        for doc_id, doc_data in self.document_index["documents"].items():
            # Skip if category filter is applied and document doesn't match
            if categories and not any(cat in doc_data["categories"] for cat in categories):
                continue
            
            # Skip duplicate content (if content_hash exists)
            content_hash = doc_data.get("content_hash")
            if content_hash:
                if content_hash in content_hashes_seen:
                    continue
                content_hashes_seen.add(content_hash)
            
            # Calculate relevance score
            score = self._calculate_relevance(query_tokens, doc_data["full_text"])
            
            if score > 0:
                # Get structured category information if available
                structured_categories = []
                for cat in doc_data["categories"]:
                    if cat in structured_categories_dict:
                        structured_categories.append(structured_categories_dict[cat])
                
                result = {
                    "document_id": doc_id,
                    "filename": doc_data["filename"],
                    "categories": doc_data["categories"],
                    "score": score,
                    "snippet": self._generate_snippet(query_tokens, doc_data["full_text"])
                }
                
                # Add structured categories if available
                if structured_categories:
                    result["structured_categories"] = structured_categories
                
                results.append(result)
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top results
        return results[:max_results]
    
    def _calculate_relevance(self, query_tokens, document_text):
        """Calculate relevance score of document for the given query"""
        # Simple term frequency scoring
        document_text = document_text.lower()
        
        score = 0
        for token in query_tokens:
            # Count occurrences of the token in the document
            count = document_text.count(token)
            
            # Increase score based on occurrences
            score += count
            
            # Bonus points for exact phrase matches
            if len(query_tokens) > 1:
                phrase = ' '.join(query_tokens)
                if phrase in document_text:
                    score += 10  # Bonus for exact phrase match
        
        return score
    
    def _generate_snippet(self, query_tokens, text, snippet_length=200):
        """Generate a relevant text snippet showing query context"""
        text = text.lower()
        
        # Try to find the best matching segment
        best_pos = 0
        highest_count = 0
        
        # Simple sliding window approach
        for i in range(0, len(text) - snippet_length, 50):
            window = text[i:i+snippet_length]
            
            # Count occurrences of query tokens in this window
            count = sum(window.count(token) for token in query_tokens)
            
            if count > highest_count:
                highest_count = count
                best_pos = i
        
        # If no matches found, return the beginning of the text
        if highest_count == 0:
            snippet = text[:snippet_length]
        else:
            snippet = text[best_pos:best_pos+snippet_length]
        
        # Clean up the snippet
        snippet = snippet.replace('\n', ' ')
        snippet = re.sub(r'\s+', ' ', snippet).strip()
        
        # Add ellipsis if we're not starting from the beginning
        if best_pos > 0:
            snippet = f"...{snippet}"
        
        # Add ellipsis if we're not ending at the end
        if best_pos + snippet_length < len(text):
            snippet = f"{snippet}..."
        
        return snippet 