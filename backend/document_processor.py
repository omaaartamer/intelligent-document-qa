import fitz
from typing import List, Dict
import re
import unicodedata
from datetime import datetime

class DocumentProcessor:
    """Handles PDF document processing and text extraction"""
    
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """Remove problematic Unicode characters and normalize text encoding"""
        text = ''.join(char for char in text if not (0xD800 <= ord(char) <= 0xDFFF))
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        return text
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract and clean text from PDF document"""
        try:
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text("text") + "\n"
            
            pdf_document.close()
            text = self.clean_text(text.strip())
            
            return text
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def extract_year_from_filename(self, filename: str) -> int:
        """Parse publication year from filename (expects YYYY_filename.pdf format)"""
        try:
            year_str = filename[:4]
            year = int(year_str)
            
            if 1900 <= year <= 2030:
                return year
            else:
                return datetime.now().year
                
        except (ValueError, IndexError):
            return datetime.now().year
        
    def process_document(self, pdf_content: bytes, filename: str) -> Dict:
        """Extract text and metadata from PDF document"""
        text = self.extract_text_from_pdf(pdf_content)
        year = self.extract_year_from_filename(filename)
        
        return {
            "filename": filename,
            "text": text,
            "year": year,
            "word_count": len(text.split()),
            "processed_at": datetime.now().isoformat()
        }