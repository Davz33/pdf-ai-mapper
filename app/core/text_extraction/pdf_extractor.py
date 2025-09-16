"""
PDF text extraction with OCR fallback.
"""
import os
import threading
import logging
import pypdf
from pdf2image import convert_from_path
import pytesseract


class PDFExtractor:
    """Handles text extraction from PDF files."""
    
    def __init__(self, timeout=120):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF with timeout protection.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text or error message
        """
        result = {"text": "", "success": False, "error": None}
        
        def extract_with_timeout():
            try:
                # Try direct PDF text extraction first
                with open(file_path, 'rb') as f:
                    pdf_reader = pypdf.PdfReader(f)
                    self.logger.info(f"PDF has {len(pdf_reader.pages)} pages")
                    
                    # Process pages in batches to avoid memory issues
                    for page_num in range(len(pdf_reader.pages)):
                        if page_num % 10 == 0:
                            self.logger.info(f"Processing page {page_num} of {len(pdf_reader.pages)}")
                        try:
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text() or ""
                            result["text"] += page_text + "\n"
                        except Exception as e:
                            self.logger.error(f"Error extracting text from page {page_num}: {e}")
                
                text_length = len(result["text"].strip())
                self.logger.info(f"Extracted {text_length} characters of text from PDF")
                
                # If no text or too little text, try OCR on first few pages
                if text_length < 1000:
                    self.logger.info("Text extraction yielded limited results, trying OCR")
                    self._try_ocr_fallback(file_path, pdf_reader, result)
                
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)
                self.logger.error(f"Error extracting text from PDF: {e}")
        
        # Run extraction in a separate thread with timeout
        thread = threading.Thread(target=extract_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(self.timeout)
        
        if thread.is_alive():
            self.logger.error(f"PDF extraction timed out after {self.timeout} seconds: {file_path}")
            if result["text"]:
                self.logger.info(f"Using partial text extracted before timeout ({len(result['text'])} characters)")
                return result["text"]
            return f"Error: PDF extraction timed out after {self.timeout} seconds. The file may be too large or complex."
        
        if not result["success"] and result["error"]:
            self.logger.error(f"Failed to extract text from PDF: {result['error']}")
            return f"Error extracting text: {result['error']}"
        
        if result["text"].strip():
            return result["text"]
        else:
            return "Error: No text could be extracted from the PDF"
    
    def _try_ocr_fallback(self, file_path: str, pdf_reader, result: dict):
        """Try OCR on first few pages as fallback."""
        try:
            # Only convert first 5 pages to images to save time
            max_pages = min(5, len(pdf_reader.pages))
            images = convert_from_path(file_path, first_page=1, last_page=max_pages)
            self.logger.info(f"Converted {len(images)} pages to images for OCR")
            
            for i, image in enumerate(images):
                self.logger.info(f"Running OCR on page {i+1}")
                page_text = pytesseract.image_to_string(image)
                result["text"] += page_text + "\n"
        except Exception as e:
            self.logger.error(f"Error during OCR processing: {e}")