# title="app/main.py"
import os
import re
import shutil
import uuid
import fitz
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional

from app.config import settings
from app.services.parser_service import ParserService
from app.services.ocr_layout_service import OCRLayoutVisionService
from app.services.vector_service import VectorDBService
from app.services.search_service import LiveWebSearchService
from app.services.llm_service import LLMOnyxOrchestrator
from app.services.generator_service import StructuredFileGeneratorService

app = FastAPI(title=settings.APP_NAME)

# In-memory storage context tracking active conversational session history
conversational_session_ledger = []

# Mock runtime registration layer to ensure directory alignment
templates = Jinja2Templates(directory=os.path.join(os.getcwd(), "app", "templates"))

class QueryModel(BaseModel):
    prompt: str
    enable_web_search: bool = True

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    # Pass 'request' as the first positional argument
    return templates.TemplateResponse(request, "index.html")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # === 1. AUTOMATICALLY WIPE PREVIOUS DATABASE ON NEW UPLOAD ===
    print("🧹 Wiping previous database for new upload...")
    VectorDBService.reset_vector_space()
    
    # 2. Setup temporary storage
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        chunks = []

        # === ROUTE 1: ADVANCED PDF PARSING (TEXT + EMBEDDED IMAGES) ===
        if file_extension == ".pdf":
            print(f"📄 Processing PDF document: {file.filename}")
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text().strip()
                
                # Look for images on this specific page
                image_descriptions = []
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_ext = base_image["ext"]
                    
                    # Save the extracted PDF image temporarily
                    tmp_img_path = os.path.join(temp_dir, f"page{page_num}_img{img_index}.{img_ext}")
                    with open(tmp_img_path, "wb") as f:
                        f.write(image_bytes)
                        
                    # Run Florence-2 on the extracted image!
                    print(f"🖼️ Analyzing embedded image {img_index + 1} on page {page_num + 1}...")
                    analysis_results = OCRLayoutVisionService.process_image_elements(tmp_img_path)
                    
                    img_desc = (
                        f"[Embedded Image {img_index + 1} Analysis]: {analysis_results['visual_caption']} | "
                        f"[OCR Text Inside Image]: {analysis_results['raw_ocr_text']}"
                    )
                    image_descriptions.append(img_desc)
                    
                    # Clean up the temp image
                    if os.path.exists(tmp_img_path):
                        os.remove(tmp_img_path)

                # Combine the text from the page with the descriptions of the images on that page
                combined_page_content = f"--- Page {page_num + 1} Text ---\n{page_text}\n"
                if image_descriptions:
                    combined_page_content += "\n--- Images found on Page " + str(page_num + 1) + " ---\n" + "\n".join(image_descriptions)
                
                # Only add if the page actually has text or images
                if page_text or image_descriptions:
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "content": combined_page_content,
                        "metadata": {"source_file": file.filename, "page_index": page_num + 1, "has_visuals": len(image_descriptions) > 0}
                    })
            doc.close()

        # === ROUTE 2: STANDALONE IMAGES ===
        elif file_extension in [".jpg", ".jpeg", ".png", ".webp"]:
            print(f"🖼️ Processing Standalone Image file: {file.filename}")
            analysis_results = OCRLayoutVisionService.process_image_elements(file_path)
            combined_text_context = (
                f"Image Context Analysis: {analysis_results['visual_caption']}\n"
                f"Extracted Image Text/OCR: {analysis_results['raw_ocr_text']}"
            )
            chunks.append({
                "id": str(uuid.uuid4()),
                "content": combined_text_context,
                "metadata": {"source_file": file.filename, "page_index": 1, "has_visuals": True}
            })
            
        # === ROUTE 3: SPREADSHEETS ===
        elif file_extension in [".xlsx", ".xls", ".csv"]:
            print(f"📊 Processing Spreadsheet: {file.filename}")
            import pandas as pd
            df = pd.read_csv(file_path) if file_extension == ".csv" else pd.read_excel(file_path)
            text_content = df.to_markdown(index=False)
            chunks.append({
                "id": str(uuid.uuid4()),
                "content": f"Spreadsheet Data from {file.filename}:\n\n{text_content}",
                "metadata": {"source_file": file.filename, "page_index": 1, "has_visuals": False}
            })

        else:
            return {"status": "error", "message": f"Unsupported file type: {file_extension}"}
        
        # === 3. PUSH CHUNKS TO LOCAL LANCEDB ===
        print("🧬 Formatting data and pushing to LanceDB...")
        if chunks:
            VectorDBService.index_chunks(chunks)
        
        print("✨ Successfully saved vectors to local DB!")
        return {"status": "success", "message": "File processed, old data wiped, and stored successfully."}

    except Exception as e:
        print(f"❌ Upload Pipeline Failed: {str(e)}")
        return {"status": "error", "message": f"Pipeline failed: {str(e)}"}
        
    finally:
        # Clean up the main uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            
@app.post("/api/chat")
async def execute_query_inference(data: QueryModel):
    """
    Orchestrates the vector space, interrogates live web search vectors, handles LLM reasoning context, and creates file exports.
    """
    global conversational_session_ledger
    user_query = data.prompt
    
    # 1. Coordinate localized document vector searches
    vector_matches = VectorDBService.query_vector_space(user_query, top_k=2)
    
    # 2. Interrogate live web data search loops conditionally
    web_triggers = ["search the web", "real-time", "real time", "internet", "online", "current", "latest news"]
        
    if any(trigger in user_query.lower() for trigger in web_triggers) and data.enable_web_search:
        print("🌐 Real-time web search triggered!")
        web_matches = LiveWebSearchService.search(user_query)
    else:
        web_matches = ""
        
    # 3. Assemble historical contexts and evaluate inference routes
    inference_payload = LLMOnyxOrchestrator.generate_response(
        query=user_query, 
        vector_context=vector_matches, 
        web_context=web_matches, 
        history=conversational_session_ledger[-2:]
    )
        
    raw_answer = inference_payload["answer"]
    
    # 4. Handle embedded explicit downstream layout document requests
    generated_file_url = None
    file_type_target = None
    
    pattern = r"\[\[FILE_GENERATION_REQUEST:\s*TYPE=(PDF|DOCX)\]\](.*)"
    match = re.search(pattern, raw_answer, re.DOTALL | re.IGNORECASE)
    
    if match:
        file_type_target = match.group(1).upper()
        document_markdown = match.group(2).strip()
        
        if file_type_target == "PDF":
            generated_file_url = StructuredFileGeneratorService.compile_pdf(document_markdown)
        elif file_type_target == "DOCX":
            generated_file_url = StructuredFileGeneratorService.compile_docx(document_markdown)
            
        raw_answer = re.sub(pattern, "", raw_answer, flags=re.DOTALL | re.IGNORECASE).strip()

    conversational_session_ledger.append({"role": "user", "content": user_query})
    conversational_session_ledger.append({"role": "assistant", "content": raw_answer})
    
    return JSONResponse(content={
        "response": raw_answer,
        "download_url": f"/api/download/{generated_file_url}" if generated_file_url else None,
        "file_type": file_type_target
    })

@app.get("/api/download/{filename}")
async def serve_generated_binary(filename: str):
    """
    Exposes dynamic data endpoints for retrieving compiled file downloads.
    """
    target_path = os.path.join(settings.OUTPUT_DIR, filename)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Requested intelligence generation asset not found.")
    return FileResponse(path=target_path, filename=filename, media_type="application/octet-stream")

@app.post("/api/reset")
async def clear_session_context():
    global conversational_session_ledger
    
    # 1. Clear the LLM's conversation history memory
    conversational_session_ledger.clear()
    
    # 2. Physically delete the local LanceDB vector data for a clean slate
    VectorDBService.reset_vector_space()
    
    return JSONResponse(content={"status": "Clear", "message": "State and Vector contexts completely reset."})