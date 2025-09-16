"""
Relevance calculation for search results.
"""
import logging
from typing import List, Dict, Any


class RelevanceCalculator:
    """Calculates relevance scores for search results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_relevance(self, query_tokens: List[str], document_text: str) -> float:
        """
        Calculate relevance score of document for the given query.
        
        Args:
            query_tokens: List of processed query tokens
            document_text: Full text of the document
            
        Returns:
            Relevance score
        """
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
    
    def calculate_document_scores(self, query_tokens: List[str], documents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate relevance scores for multiple documents.
        
        Args:
            query_tokens: List of processed query tokens
            documents: Dictionary of documents to score
            
        Returns:
            List of documents with scores
        """
        scored_documents = []
        
        for doc_id, doc_data in documents.items():
            score = self.calculate_relevance(query_tokens, doc_data["full_text"])
            
            if score > 0:
                scored_documents.append({
                    "document_id": doc_id,
                    "score": score,
                    "document_data": doc_data
                })
        
        return scored_documents