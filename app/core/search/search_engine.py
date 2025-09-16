"""
Refactored search engine using modular components.
"""
import os
import json
import logging
from typing import List, Dict, Any

from .query_processor import QueryProcessor
from .relevance_calculator import RelevanceCalculator
from .snippet_generator import SnippetGenerator
from .filter_manager import FilterManager


class SearchEngine:
    """Main search engine using modular components."""
    
    def __init__(self):
        self.processed_dir = "processed_data"
        self.index_file = os.path.join(self.processed_dir, "document_index.json")
        
        # Initialize components
        self.query_processor = QueryProcessor()
        self.relevance_calculator = RelevanceCalculator()
        self.snippet_generator = SnippetGenerator()
        self.filter_manager = FilterManager()
        
        # Load document index
        self.document_index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load the document index."""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                return json.load(f)
        else:
            return {"documents": {}, "categories": []}
    
    def search(self, query: str, categories: List[str] = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.
        
        Args:
            query: The search query
            categories: Optional list of categories to filter by
            max_results: Maximum number of results to return
            
        Returns:
            List of matching documents with relevance scores
        """
        # Refresh the index in case new documents were added
        self.document_index = self._load_index()
        
        # Preprocess the query
        query_tokens = self.query_processor.preprocess_query(query)
        
        # If no query tokens after preprocessing, return empty results
        if not self.query_processor.is_valid_query(query_tokens):
            return []
        
        # Get all documents
        documents = self.document_index["documents"]
        
        # Apply category filters
        if categories:
            documents = self.filter_manager.apply_category_filters(documents, categories)
        
        # Remove duplicates
        documents = self.filter_manager.remove_duplicates(documents)
        
        # Calculate relevance scores
        scored_documents = self.relevance_calculator.calculate_document_scores(query_tokens, documents)
        
        # Generate snippets
        results = self.snippet_generator.generate_snippets_for_results(query_tokens, scored_documents)
        
        # Format results
        formatted_results = self._format_results(results)
        
        # Sort by relevance score (descending)
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top results
        return formatted_results[:max_results]
    
    def search_with_structured_filters(self, query: str, 
                                     category_types: List[str] = None, 
                                     keywords: List[str] = None,
                                     max_results: int = 10) -> Dict[str, Any]:
        """
        Search with structured category filters.
        
        Args:
            query: The search query
            category_types: List of category types to filter by
            keywords: List of keywords to filter by
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with results and available filters
        """
        # Refresh the index
        self.document_index = self._load_index()
        
        # Preprocess the query
        query_tokens = self.query_processor.preprocess_query(query)
        
        if not self.query_processor.is_valid_query(query_tokens):
            return {
                "results": [],
                "available_filters": self.filter_manager.get_available_filters()
            }
        
        # Get documents and structured categories
        documents = self.document_index["documents"]
        structured_categories = self.document_index.get("structured_categories", [])
        
        # Apply structured filters
        if category_types or keywords:
            documents = self.filter_manager.apply_structured_filters(
                documents, category_types, keywords, structured_categories
            )
        
        # Remove duplicates
        documents = self.filter_manager.remove_duplicates(documents)
        
        # Calculate relevance scores
        scored_documents = self.relevance_calculator.calculate_document_scores(query_tokens, documents)
        
        # Generate snippets
        results = self.snippet_generator.generate_snippets_for_results(query_tokens, scored_documents)
        
        # Format results
        formatted_results = self._format_results(results, structured_categories)
        
        # Sort by relevance score
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Get available filters
        available_filters = self.filter_manager.get_available_filters(structured_categories)
        
        return {
            "results": formatted_results[:max_results],
            "available_filters": available_filters
        }
    
    def _format_results(self, results: List[Dict[str, Any]], 
                       structured_categories: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Format search results for output.
        
        Args:
            results: List of scored documents
            structured_categories: List of structured category definitions
            
        Returns:
            Formatted results
        """
        formatted_results = []
        
        # Create structured categories lookup
        structured_categories_dict = {}
        if structured_categories:
            for cat in structured_categories:
                structured_categories_dict[cat["display_name"]] = cat
        
        for result in results:
            doc_data = result["document_data"]
            
            # Get structured category information if available
            structured_cats = []
            for cat in doc_data["categories"]:
                if cat in structured_categories_dict:
                    structured_cats.append(structured_categories_dict[cat])
            
            formatted_result = {
                "document_id": result["document_id"],
                "filename": doc_data["filename"],
                "categories": doc_data["categories"],
                "score": result["score"],
                "snippet": result["snippet"]
            }
            
            # Add structured categories if available
            if structured_cats:
                formatted_result["structured_categories"] = structured_cats
            
            formatted_results.append(formatted_result)
        
        return formatted_results