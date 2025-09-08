"""
Snippet generation for search results.
"""
import re
import logging
from typing import List


class SnippetGenerator:
    """Generates relevant text snippets for search results."""
    
    def __init__(self, snippet_length: int = 200):
        self.snippet_length = snippet_length
        self.logger = logging.getLogger(__name__)
    
    def generate_snippet(self, query_tokens: List[str], text: str) -> str:
        """
        Generate a relevant text snippet showing query context.
        
        Args:
            query_tokens: List of processed query tokens
            text: Full document text
            
        Returns:
            Generated snippet
        """
        text = text.lower()
        
        # Try to find the best matching segment
        best_pos = 0
        highest_count = 0
        
        # Simple sliding window approach
        for i in range(0, len(text) - self.snippet_length, 50):
            window = text[i:i+self.snippet_length]
            
            # Count occurrences of query tokens in this window
            count = sum(window.count(token) for token in query_tokens)
            
            if count > highest_count:
                highest_count = count
                best_pos = i
        
        # If no matches found, return the beginning of the text
        if highest_count == 0:
            snippet = text[:self.snippet_length]
        else:
            snippet = text[best_pos:best_pos+self.snippet_length]
        
        # Clean up the snippet
        snippet = snippet.replace('\n', ' ')
        snippet = re.sub(r'\s+', ' ', snippet).strip()
        
        # Add ellipsis if we're not starting from the beginning
        if best_pos > 0:
            snippet = f"...{snippet}"
        
        # Add ellipsis if we're not ending at the end
        if best_pos + self.snippet_length < len(text):
            snippet = f"{snippet}..."
        
        return snippet
    
    def generate_snippets_for_results(self, query_tokens: List[str], results: List[dict]) -> List[dict]:
        """
        Generate snippets for multiple search results.
        
        Args:
            query_tokens: List of processed query tokens
            results: List of search results
            
        Returns:
            Results with added snippets
        """
        for result in results:
            if "document_data" in result:
                result["snippet"] = self.generate_snippet(
                    query_tokens, 
                    result["document_data"]["full_text"]
                )
        
        return results