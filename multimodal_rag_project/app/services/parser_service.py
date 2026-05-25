# title="app/services/parser_service.py"
import fitz  # PyMuPDF
import os
import uuid
from typing import List, Dict, Any
from app.config import settings

class ParserService:
    @staticmethod
    def extract_and_split(file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a document, partitions it by structural pages, and extracts embedded visual elements.
        """
        parsed_pages = []
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".pdf":
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                
                # Extract image matrices from the raw page structure
                image_list = page.get_images(full=True)
                extracted_images = []
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    img_filename = f"extracted_{uuid.uuid4()}.{image_ext}"
                    img_save_path = os.path.join(settings.UPLOAD_DIR, img_filename)
                    
                    with open(img_save_path, "wb") as f:
                        f.write(image_bytes)
                    
                    extracted_images.append(img_save_path)
                
                parsed_pages.append({
                    "page_index": page_num + 1,
                    "text_content": text,
                    "associated_images": extracted_images,
                    "source_file": os.path.basename(file_path)
                })
            doc.close()
        else:
            # Fallback handling for generic image files passed directly as documents
            if file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
                parsed_pages.append({
                    "page_index": 1,
                    "text_content": f"[Direct Image Upload: {os.path.basename(file_path)}]",
                    "associated_images": [file_path],
                    "source_file": os.path.basename(file_path)
                })
        return parsed_pages