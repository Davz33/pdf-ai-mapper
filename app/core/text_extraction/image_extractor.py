"""
Image text extraction using OCR.
"""
import threading
import logging
from PIL import Image
import pytesseract


class ImageExtractor:
    """Handles text extraction from image files using OCR."""
    
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from image using OCR with timeout protection.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Extracted text or error message
        """
        result = {"text": "", "success": False, "error": None}
        
        def extract_with_timeout():
            try:
                image = Image.open(file_path)
                result["text"] = pytesseract.image_to_string(image)
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)
                self.logger.error(f"Error extracting text from image: {e}")
        
        # Run extraction in a separate thread with timeout
        thread = threading.Thread(target=extract_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(self.timeout)
        
        if thread.is_alive():
            self.logger.error(f"Image OCR timed out after {self.timeout} seconds: {file_path}")
            return f"Error: OCR timed out after {self.timeout} seconds. The image may be too large or complex."
        
        if not result["success"] and result["error"]:
            self.logger.error(f"Failed to extract text from image: {result['error']}")
            return f"Error extracting text: {result['error']}"
        
        return result["text"]