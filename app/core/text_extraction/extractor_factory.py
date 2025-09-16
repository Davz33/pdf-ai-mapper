"""
Factory for creating appropriate text extractors based on file type.
"""
import os
from .pdf_extractor import PDFExtractor
from .image_extractor import ImageExtractor


class ExtractorFactory:
    """Factory class for creating text extractors."""
    
    def get_extractor(self, file_path: str):
        """
        Get appropriate extractor based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Appropriate extractor instance
            
        Raises:
            ValueError: If file type is not supported
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return PDFExtractor()
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return ImageExtractor()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    @staticmethod
    def create_extractor(file_path: str):
        """
        Create appropriate extractor based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Appropriate extractor instance
            
        Raises:
            ValueError: If file type is not supported
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return PDFExtractor()
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return ImageExtractor()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    @staticmethod
    def is_supported_file(file_path: str) -> bool:
        """
        Check if file type is supported for text extraction.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file type is supported, False otherwise
        """
        try:
            ExtractorFactory.create_extractor(file_path)
            return True
        except ValueError:
            return False