import fitz  # PyMuPDF
import logging
import asyncio

logger = logging.getLogger(__name__)


def _extract_text_from_pdf_sync(file_bytes: bytes) -> str:
    extracted_text = ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        # Iterate through all pages and extract text
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            extracted_text += page.get_text("text") + "\n" 
            
        doc.close()
        
        if not extracted_text.strip():
            raise ValueError("Document is empty or contains no readable text (might be a scanned image).")
            
        return extracted_text.strip()

    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise ValueError(f"Failed to parse PDF: {str(e)}")


def _extract_text_from_txt_sync(file_bytes: bytes) -> str:
    try:
        text = file_bytes.decode("utf-8", errors="strict")
        
        if not text.strip():
            raise ValueError("Text file is empty.")
            
        return text.strip()

    except UnicodeDecodeError as e:
        logger.error(f"Text file decoding failed: {str(e)}")
        raise ValueError(f"Failed to decode text file: {str(e)}")


async def extract_text_from_pdf(file_bytes: bytes) -> str:
    return await asyncio.to_thread(_extract_text_from_pdf_sync, file_bytes)


async def extract_text_from_txt(file_bytes: bytes) -> str:
    return await asyncio.to_thread(_extract_text_from_txt_sync, file_bytes)