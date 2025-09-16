"""
Filter management for search functionality.
"""
import logging
from typing import List, Dict, Any, Set


class FilterManager:
    """Manages search filters and category filtering."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply_category_filters(self, documents: Dict[str, Any], categories: List[str] = None) -> Dict[str, Any]:
        """
        Apply category filters to documents.
        
        Args:
            documents: Dictionary of documents
            categories: List of categories to filter by
            
        Returns:
            Filtered documents
        """
        if not categories:
            return documents
        
        filtered_documents = {}
        for doc_id, doc_data in documents.items():
            if any(cat in doc_data["categories"] for cat in categories):
                filtered_documents[doc_id] = doc_data
        
        self.logger.info(f"Filtered {len(documents)} documents to {len(filtered_documents)} by categories")
        return filtered_documents
    
    def apply_structured_filters(self, documents: Dict[str, Any], 
                                category_types: List[str] = None, 
                                keywords: List[str] = None,
                                structured_categories: List[Dict] = None) -> Dict[str, Any]:
        """
        Apply structured category filters to documents.
        
        Args:
            documents: Dictionary of documents
            category_types: List of category types to filter by
            keywords: List of keywords to filter by
            structured_categories: List of structured category definitions
            
        Returns:
            Filtered documents
        """
        if not structured_categories or (not category_types and not keywords):
            return documents
        
        # Build filter categories from structured categories
        filter_categories = set()
        
        for cat in structured_categories:
            # Filter by category type
            if category_types and cat["type"] in category_types:
                filter_categories.add(cat["display_name"])
                continue
            
            # Filter by keywords
            if keywords and any(kw in cat["keywords"] for kw in keywords):
                filter_categories.add(cat["display_name"])
        
        return self.apply_category_filters(documents, list(filter_categories))
    
    def remove_duplicates(self, documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove duplicate documents based on content hash.
        
        Args:
            documents: Dictionary of documents
            
        Returns:
            Documents with duplicates removed
        """
        seen_hashes: Set[str] = set()
        unique_documents = {}
        
        for doc_id, doc_data in documents.items():
            content_hash = doc_data.get("content_hash")
            if content_hash:
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_documents[doc_id] = doc_data
            else:
                # If no content hash, include the document
                unique_documents[doc_id] = doc_data
        
        removed_count = len(documents) - len(unique_documents)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate documents")
        
        return unique_documents
    
    def get_available_filters(self, structured_categories: List[Dict] = None) -> Dict[str, List[str]]:
        """
        Get available category types and keywords for filtering.
        
        Args:
            structured_categories: List of structured category definitions
            
        Returns:
            Dictionary with available filters
        """
        available_filters = {
            "category_types": [],
            "keywords": []
        }
        
        if structured_categories:
            # Extract unique category types
            category_types = set()
            all_keywords = []
            
            for cat in structured_categories:
                category_types.add(cat["type"])
                all_keywords.extend(cat["keywords"])
            
            available_filters["category_types"] = sorted(list(category_types))
            available_filters["keywords"] = sorted(list(set(all_keywords)))
        
        return available_filters