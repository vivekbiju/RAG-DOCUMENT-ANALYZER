import os
import sys
import shutil
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware  # <-- ADD THIS IMPORT
from pydantic import BaseModel
from dotenv import load_dotenv

# Initialize environment configurations
load_dotenv()

BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))

effective_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if effective_key:
    os.environ["GEMINI_API_KEY"] = effective_key
    os.environ["GOOGLE_API_KEY"] = effective_key

for folder_name in ["data/uploads", "data/processed", "chroma_db"]:
    target_dir = BASE_PATH / folder_name
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Directory creation warning: {e}")

UPLOAD_DIR = BASE_PATH / "data/uploads"

try:
    from src.pipeline import GeminiRAG
    from src.ingestion import IngestionPipeline
    from src.evaluation_utils import run_evaluation
    rag_system = GeminiRAG()
    ingestor = IngestionPipeline()

except Exception as init_err:
    print(f"❌ HF BOOT ERROR: {init_err}") 
    rag_system = None
    ingestor = None


app = FastAPI(
    title="RAG Research API",
    description="Backend API for Transformer Research Assistant with Gemini & ChromaDB",
    version="1.0.0"
)

# 👇 ADD THIS CORS MIDDLEWARE BLOCK RIGHT HERE 👇
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins to connect inside the container space safely
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)


# --- Data Models ---
class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- API Endpoints ---

@app.get("/")
async def root():
    api_key_check = "Configured" if os.getenv("GEMINI_API_KEY") else "Missing"
    pipeline_check = "Active" if rag_system is not None else "Degraded (No Vector Store detected)"
    return {
        "status": "online", 
        "pipeline": pipeline_check,
        "credentials": api_key_check,
        "message": "RAG Research API is running successfully."
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_rag(request: QueryRequest):
    if not rag_system:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline is currently unavailable. Please upload a research document first to build the knowledge base."
        )
    try:
        relevant_docs = rag_system.retrieve_and_rerank(request.prompt)
        answer = rag_system.generate(request.prompt, relevant_docs)
        sources = [doc.page_content for doc in relevant_docs]
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Generation Error: {str(e)}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not ingestor:
        raise HTTPException(
            status_code=500, 
            detail="Ingestion pipeline failed to initialize during boot setup."
        )
        
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )

    temp_path = UPLOAD_DIR / file.filename

    try:
        # Stream file to the secure path location
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse, chunk, embed, and store
        ingestor.run(str(temp_path))
        
        # Dynamically mount the RAG interface if it was degraded at initialization
        global rag_system
        if rag_system is None:
            from src.pipeline import GeminiRAG
            rag_system = GeminiRAG()
            
        return {
            "status": "success", 
            "filename": file.filename,
            "message": "Document successfully indexed and added to the knowledge base."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion Error: {str(e)}")
    finally:
        # File removal cleanup layer
        if temp_path.exists():
            os.remove(temp_path)

@app.post("/run-benchmark")
async def trigger_benchmark(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_evaluation)
        return {"status": "started", "message": "Benchmark running in background"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)