
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path

# Importing your logic from the src folder
from src.pipeline import GeminiRAG
from src.ingestion import IngestionPipeline

from fastapi import FastAPI, BackgroundTasks, HTTPException
from src.evaluation_utils import run_evaluation # Import your eval function

import sys
# Inject the parent root directory into Python's lookup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app = FastAPI(
    title="RAG Research API",
    description="Backend API for Transformer Research Assistant with Gemini & ChromaDB",
    version="1.0.0"
)

# Initialize the systems once at startup (Singletons)

rag_system = GeminiRAG()
ingestor = IngestionPipeline()

# Ensure temporary upload directory exists
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Data Models for API ---

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"status": "online", "message": "RAG Research API is running."}

@app.post("/ask", response_model=QueryResponse)
async def ask_rag(request: QueryRequest):
    """
    Endpoint to query the RAG system. 
    Accepts a prompt and returns an answer with retrieved context snippets.
    """
    try:
        # 1. Fetch the relevant docs FIRST (needed to return sources to the UI)
        relevant_docs = rag_system.retrieve_and_rerank(request.prompt)
        
        # 2. FIX: Call the parent query_system function to generate the answer.
        # This triggers your parent LangSmith span, nesting everything under it!
        answer = rag_system.generate(request.prompt, relevant_docs)
        
        # 3. Extract source text for the UI
        sources = [doc.page_content for doc in relevant_docs]
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Generation Error: {str(e)}")



@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload and index new PDF, TXT, or MD files.
    """
    # Validate file extension
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )

    temp_path = UPLOAD_DIR / file.filename

    try:
        # 1. Save the file to the temporary uploads folder
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Trigger the Ingestion Pipeline
        # This will load, chunk, and add to ChromaDB
        ingestor.run(str(temp_path))
        
        return {
            "status": "success", 
            "filename": file.filename,
            "message": "Document successfully indexed and added to the knowledge base."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion Error: {str(e)}")
    
    finally:
        # 3. Cleanup: Remove the temp file after processing to save space
        if temp_path.exists():
            os.remove(temp_path)




@app.post("/run-benchmark")
async def trigger_benchmark(background_tasks: BackgroundTasks):
    try:
        # We add the task to the background so the request returns 200 immediately
        background_tasks.add_task(run_evaluation)
        return {"status": "started", "message": "Benchmark running in background"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    # Run from the root directory as: python backend/main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)