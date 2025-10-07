"""
Fathom Prospector - Python API Server (Phase 2: Production Hardened)
Wraps prospect.py in a FastAPI server for remote calls from Next.js app
Refactored to use direct imports instead of subprocess for stability.
"""

import asyncio
import csv
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Add the current directory to the path to ensure prospect.py is importable
sys.path.insert(0, os.path.dirname(__file__))
from prospect import FathomProspector

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Fathom Prospector API", version="2.1.0")

# --- Rate Limiter & CORS Configuration ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fathomtech.net", "https://www.fathomtech.net", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals and Locks ---
API_KEY = os.getenv("FATHOM_API_KEY", "your-secret-api-key-change-this")
search_jobs = {}
search_jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="search_worker")
MAX_CONCURRENT_SEARCHES = int(os.getenv("MAX_CONCURRENT_SEARCHES", "5"))
search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    keywords: List[str]
    location: str
    radius: int = 25
    maxResults: int = 150

class SearchResponse(BaseModel):
    job_id: str
    status: str
    message: str

class SearchStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    csv_file: Optional[str] = None
    report_file: Optional[str] = None
    error: Optional[str] = None
    results: Optional[List[Dict]] = None


# --- Helper Functions ---
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

def parse_csv_results(csv_file: str):
    if not csv_file or not os.path.exists(csv_file): return []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except Exception as e:
        logger.error(f"Error parsing CSV {csv_file}: {e}")
        return []

# --- Core Search Logic ---
def _run_prospect_search_sync(job_id: str, request: SearchRequest):
    """
    Synchronous version of prospect search to run in thread pool.
    This directly calls the FathomProspector class.
    """
    try:
        with search_jobs_lock:
            search_jobs[job_id].update({"status": "running", "progress": 5, "message": "Initializing search..."})

        # Create progress callback to update job status
        def progress_callback(progress: int, message: str):
            """Update job progress in real-time"""
            with search_jobs_lock:
                search_jobs[job_id].update({
                    "progress": progress,
                    "message": message
                })
            logger.info(f"Job {job_id}: {progress}% - {message}")

        logger.info(f"Job {job_id}: Instantiating FathomProspector.")
        prospector = FathomProspector(progress_callback=progress_callback)

        with search_jobs_lock:
            search_jobs[job_id].update({"progress": 10, "message": "Configuring search parameters..."})
        
        start_time = time.time()
        
        results, csv_file, report_file = prospector.run_prospecting(
            keywords=request.keywords,
            location=request.location,
            radius=request.radius,
            max_results=request.maxResults
        )

        duration = time.time() - start_time
        logger.info(f"Job {job_id}: Prospecting complete in {duration:.2f} seconds.")
        
        parsed_results = parse_csv_results(csv_file)

        with search_jobs_lock:
            search_jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Search completed successfully.",
                "completed_at": datetime.now().isoformat(),
                "csv_file": csv_file,
                "report_file": report_file,
                "results": parsed_results,
                "duration": duration
            })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Job {job_id}: An exception occurred during prospecting: {e}\n{error_details}")
        with search_jobs_lock:
            search_jobs[job_id].update({
                "status": "failed",
                "progress": 0,
                "message": "An internal error occurred.",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            })

async def run_prospect_search(job_id: str, request: SearchRequest):
    """Async wrapper for the prospect search task."""
    async with search_semaphore:
        logger.info(f"Job {job_id}: Acquired semaphore. Running search in thread pool.")
        loop = asyncio.get_event_loop()
        try:
            # Run the synchronous, blocking function in the executor
            await loop.run_in_executor(executor, _run_prospect_search_sync, job_id, request)
        except Exception as e:
            logger.error(f"Job {job_id}: Async wrapper caught an unexpected exception: {e}")
            with search_jobs_lock:
                search_jobs[job_id].update({
                    "status": "failed", "error": f"Orchestration error: {e}"
                })

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"service": "Fathom Prospector API", "status": "operational", "version": app.version}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_searches": sum(1 for job in search_jobs.values() if job["status"] == "running"),
        "available_slots": search_semaphore._value,
    }

@app.post("/api/prospect/search", response_model=SearchResponse)
@limiter.limit("10/minute")
async def start_search(
    request: Request,
    search_request: SearchRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Header(..., alias="X-API-Key")
):
    verify_api_key(api_key)
    job_id = str(uuid.uuid4())
    
    with search_jobs_lock:
        search_jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Search is queued and will start shortly.",
            "started_at": datetime.now().isoformat(),
            "request": search_request.dict(),
        }
    
    background_tasks.add_task(run_prospect_search, job_id, search_request)
    logger.info(f"Queued search job {job_id} for location: {search_request.location}")
    
    return SearchResponse(job_id=job_id, status="queued", message="Search started successfully.")

@app.get("/api/prospect/status/{job_id}", response_model=SearchStatus)
async def get_search_status(job_id: str, api_key: str = Header(..., alias="X-API-Key")):
    verify_api_key(api_key)
    with search_jobs_lock:
        job = search_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # This corrected line adds the job_id to the response
    return SearchStatus(job_id=job_id, **job)

@app.get("/api/prospect/results/{job_id}")
async def get_search_results(job_id: str, api_key: str = Header(..., alias="X-API-Key")):
    """
    Get the detailed results of a completed search job.
    This endpoint is called by the Next.js frontend to fetch prospects.
    """
    verify_api_key(api_key)
    with search_jobs_lock:
        job = search_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed yet. Current status: {job['status']}"
        )

    return {
        "job_id": job_id,
        "status": "completed",
        "results": job.get("results", []),
        "csv_file": job.get("csv_file"),
        "report_file": job.get("report_file"),
        "duration": job.get("duration", 0)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
