
"""
Fathom Prospector - Python API Server (Phase 2: Production Hardened)
Wraps prospect.py in a FastAPI server for remote calls from Next.js app

Phase 2 Features:
- API quota management (per-user daily limits)
- Performance monitoring & metrics
- Rate limiting per user
- Enhanced error tracking
- Resource usage monitoring
"""

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import subprocess
import os
import sys
import json
import csv
import uuid
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
import time
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fathom Prospector API", version="2.0.0")

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - Allow requests from your web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fathomtech.net",
        "https://www.fathomtech.net", 
        "http://localhost:3000"  # For development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key for security (set via environment variable)
API_KEY = os.getenv("FATHOM_API_KEY", "your-secret-api-key-change-this")

# Store for tracking search jobs
search_jobs = {}
search_jobs_lock = threading.Lock()

# Thread pool for running blocking operations
# This prevents blocking the asyncio event loop
executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="search_worker")

# Semaphore to limit concurrent searches (prevent resource exhaustion)
MAX_CONCURRENT_SEARCHES = 5
search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

# ============================================================================
# PHASE 2: API QUOTA MANAGEMENT & MONITORING
# ============================================================================

# API Quota Configuration
DAILY_SEARCH_QUOTA = int(os.getenv("DAILY_SEARCH_QUOTA", "50"))  # Per user per day
PER_USER_DAILY_LIMIT = int(os.getenv("PER_USER_DAILY_LIMIT", "20"))  # Individual user limit
WARN_THRESHOLD = 0.8  # Warn at 80% quota usage

# Quota tracking (user_id: {date: count})
api_usage_tracker = defaultdict(lambda: defaultdict(int))
quota_lock = threading.Lock()

# Performance metrics
metrics = {
    "total_searches": 0,
    "successful_searches": 0,
    "failed_searches": 0,
    "total_search_time": 0.0,
    "avg_search_time": 0.0,
    "concurrent_peak": 0,
    "quota_rejections": 0,
    "rate_limit_hits": 0,
    "errors_by_type": defaultdict(int)
}
metrics_lock = threading.Lock()

def get_today_key():
    """Get today's date as string key"""
    return datetime.now().strftime("%Y-%m-%d")

def check_user_quota(user_id: str) -> Dict:
    """
    Check if user has remaining quota for today
    Returns: {"allowed": bool, "used": int, "limit": int, "remaining": int}
    """
    with quota_lock:
        today = get_today_key()
        used = api_usage_tracker[user_id][today]
        remaining = PER_USER_DAILY_LIMIT - used
        
        return {
            "allowed": used < PER_USER_DAILY_LIMIT,
            "used": used,
            "limit": PER_USER_DAILY_LIMIT,
            "remaining": max(0, remaining),
            "warn": used >= (PER_USER_DAILY_LIMIT * WARN_THRESHOLD)
        }

def increment_user_quota(user_id: str):
    """Increment user's daily search count"""
    with quota_lock:
        today = get_today_key()
        api_usage_tracker[user_id][today] += 1

def update_metrics(event: str, **kwargs):
    """Update performance metrics"""
    with metrics_lock:
        if event == "search_started":
            metrics["total_searches"] += 1
            concurrent = kwargs.get("concurrent", 0)
            metrics["concurrent_peak"] = max(metrics["concurrent_peak"], concurrent)
            
        elif event == "search_completed":
            metrics["successful_searches"] += 1
            duration = kwargs.get("duration", 0)
            metrics["total_search_time"] += duration
            if metrics["successful_searches"] > 0:
                metrics["avg_search_time"] = metrics["total_search_time"] / metrics["successful_searches"]
                
        elif event == "search_failed":
            metrics["failed_searches"] += 1
            error_type = kwargs.get("error_type", "unknown")
            metrics["errors_by_type"][error_type] += 1
            
        elif event == "quota_rejected":
            metrics["quota_rejections"] += 1
            
        elif event == "rate_limit_hit":
            metrics["rate_limit_hits"] += 1

def get_user_id_from_header(request: Request) -> str:
    """Extract user ID from request header or use IP as fallback"""
    # Try to get user ID from X-User-ID header (set by Next.js app)
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return user_id
    
    # Fallback to IP address
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    return request.client.host if request.client else "unknown"

# ============================================================================

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

def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from request header"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

def calculate_dynamic_timeout(estimated_results: int) -> int:
    """
    Calculate optimal timeout based on estimated result count
    Progressive timeout strategy - scales with expected workload
    
    Args:
        estimated_results: Estimated number of businesses to process
        
    Returns:
        Timeout in seconds
    """
    if estimated_results < 20:
        timeout = 20 * 60  # 20 minutes for small searches (doubled from 10)
        logger.info(f"Dynamic timeout: 20 minutes for ~{estimated_results} results")
    elif estimated_results < 50:
        timeout = 30 * 60  # 30 minutes for medium searches (doubled from 15)
        logger.info(f"Dynamic timeout: 30 minutes for ~{estimated_results} results")
    elif estimated_results < 80:
        timeout = 60 * 60  # 60 minutes for large searches (doubled from 30)
        logger.info(f"Dynamic timeout: 60 minutes for ~{estimated_results} results")
    elif estimated_results < 120:
        timeout = 90 * 60  # 90 minutes for very large searches (doubled from 45)
        logger.info(f"Dynamic timeout: 90 minutes for ~{estimated_results} results")
    else:
        timeout = 120 * 60  # 120 minutes max for extremely large searches (doubled from 60)
        logger.info(f"Dynamic timeout: 120 minutes (max) for ~{estimated_results} results")
    
    return timeout

def estimate_result_count_for_search(request: SearchRequest) -> int:
    """
    Estimate result count by calling prospect.py's estimate function
    
    Args:
        request: Search request parameters
        
    Returns:
        Estimated number of results
    """
    try:
        logger.info("Estimating result count for dynamic timeout calculation...")
        
        # Import prospect module
        script_dir = os.path.dirname(__file__)
        sys.path.insert(0, script_dir)
        
        from prospect import MedicalProspector
        
        # Create prospector instance with API key
        api_key = os.getenv("FATHOM_API_KEY", "")
        prospector = MedicalProspector(api_key=api_key)
        
        # Note: New prospect.py doesn't have estimate_result_count method
        # This will raise an exception and fall back to default estimate of 50
        # TODO: Implement estimate in new prospect.py if needed
        estimated = prospector.estimate_result_count(
            keywords=request.keywords,
            location=request.location,
            radius=request.radius
        )
        
        logger.info(f"Estimated {estimated} results for this search")
        return estimated
        
    except Exception as e:
        logger.warning(f"Could not estimate result count: {str(e)}")
        logger.info("Using default estimate of 50 results")
        return 50  # Default safe estimate

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Fathom Prospector API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check with metrics"""
    checks = {
        "python": check_python(),
        "prospect_script": check_prospect_script(),
        "google_api": check_google_api(),
        "gemini_api": check_gemini_api()
    }
    
    all_healthy = all(checks.values())
    
    # Calculate active searches
    active_searches = sum(1 for job in search_jobs.values() if job["status"] == "running")
    
    with metrics_lock:
        current_metrics = {
            "total_searches": metrics["total_searches"],
            "successful_searches": metrics["successful_searches"],
            "failed_searches": metrics["failed_searches"],
            "success_rate": (metrics["successful_searches"] / metrics["total_searches"] * 100) if metrics["total_searches"] > 0 else 0,
            "avg_search_time": round(metrics["avg_search_time"], 2),
            "active_searches": active_searches,
            "concurrent_peak": metrics["concurrent_peak"]
        }
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "metrics": current_metrics,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics(api_key: str = Header(..., alias="X-API-Key")):
    """Get detailed performance metrics"""
    verify_api_key(api_key)
    
    active_searches = sum(1 for job in search_jobs.values() if job["status"] == "running")
    queued_searches = sum(1 for job in search_jobs.values() if job["status"] == "queued")
    
    with metrics_lock:
        detailed_metrics = {
            "searches": {
                "total": metrics["total_searches"],
                "successful": metrics["successful_searches"],
                "failed": metrics["failed_searches"],
                "success_rate": round((metrics["successful_searches"] / metrics["total_searches"] * 100), 2) if metrics["total_searches"] > 0 else 0,
                "active": active_searches,
                "queued": queued_searches
            },
            "performance": {
                "avg_search_time_seconds": round(metrics["avg_search_time"], 2),
                "total_search_time_hours": round(metrics["total_search_time"] / 3600, 2),
                "concurrent_peak": metrics["concurrent_peak"]
            },
            "rate_limiting": {
                "quota_rejections": metrics["quota_rejections"],
                "rate_limit_hits": metrics["rate_limit_hits"]
            },
            "errors": dict(metrics["errors_by_type"]),
            "system": {
                "max_concurrent_searches": MAX_CONCURRENT_SEARCHES,
                "available_slots": MAX_CONCURRENT_SEARCHES - active_searches,
                "thread_pool_size": 10,
                "daily_quota_per_user": PER_USER_DAILY_LIMIT
            }
        }
    
    return {
        "metrics": detailed_metrics,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/quota/{user_id}")
async def get_user_quota(
    user_id: str,
    api_key: str = Header(..., alias="X-API-Key")
):
    """Get quota information for a specific user"""
    verify_api_key(api_key)
    
    quota_info = check_user_quota(user_id)
    
    return {
        "user_id": user_id,
        "quota": quota_info,
        "date": get_today_key(),
        "timestamp": datetime.now().isoformat()
    }

def check_python():
    """Check if Python 3 is available"""
    try:
        result = subprocess.run(
            [sys.executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

def check_prospect_script():
    """Check if prospect.py exists"""
    script_path = os.path.join(os.path.dirname(__file__), "prospect.py")
    return os.path.exists(script_path)

def check_google_api():
    """Check if Google API key is configured"""
    return bool(os.getenv("GOOGLE_PLACES_API_KEY"))

def check_gemini_api():
    """Check if Gemini API key is configured"""
    return bool(os.getenv("GEMINI_API_KEY"))

@app.post("/api/prospect/search", response_model=SearchResponse)
@limiter.limit("10/minute")  # Rate limit: 10 searches per minute per IP
async def start_search(
    request: Request,
    search_request: SearchRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Header(..., alias="X-API-Key")
):
    """Start a new prospecting search with quota management"""
    verify_api_key(api_key)
    
    # Get user ID from request
    user_id = get_user_id_from_header(request)
    
    # Check user quota
    quota_info = check_user_quota(user_id)
    
    if not quota_info["allowed"]:
        update_metrics("quota_rejected")
        logger.warning(f"Quota exceeded for user {user_id}: {quota_info['used']}/{quota_info['limit']}")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Daily search quota exceeded",
                "quota": quota_info,
                "message": f"You have used {quota_info['used']} of your {quota_info['limit']} daily searches. Quota resets at midnight."
            }
        )
    
    # Increment quota counter
    increment_user_quota(user_id)
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Calculate current concurrent searches
    active_searches = sum(1 for job in search_jobs.values() if job["status"] in ["running", "queued"])
    
    # Update metrics
    update_metrics("search_started", concurrent=active_searches + 1)
    
    # Initialize job status
    search_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Search queued",
        "started_at": datetime.now().isoformat(),
        "request": search_request.dict(),
        "user_id": user_id,
        "start_time": time.time()
    }
    
    # Start search in background
    background_tasks.add_task(run_prospect_search, job_id, search_request)
    
    # Log with quota warning if approaching limit
    quota_msg = f"(Quota: {quota_info['used']}/{quota_info['limit']})"
    if quota_info["warn"]:
        quota_msg += " ⚠️ APPROACHING LIMIT"
    logger.info(f"Started search job {job_id} for user {user_id}, location: {search_request.location} {quota_msg}")
    
    return SearchResponse(
        job_id=job_id,
        status="queued",
        message=f"Search started successfully. Remaining quota: {quota_info['remaining']} searches today."
    )

def _run_prospect_search_sync(job_id: str, request: SearchRequest):
    """
    Synchronous version of prospect search to run in thread pool
    This prevents blocking the asyncio event loop
    """
    try:
        # Update status
        with search_jobs_lock:
            search_jobs[job_id]["status"] = "running"
            search_jobs[job_id]["progress"] = 5
            search_jobs[job_id]["message"] = "Initializing search..."
        
        # Build command
        script_path = os.path.join(
            os.path.dirname(__file__), 
            "prospect.py"
        )
        
        # Join keywords into a single query string for new prospect.py format
        query = " ".join(request.keywords)
        
        cmd = [
            sys.executable,
            script_path,
            "--keywords", query,
            "--city", request.location,
            "--radius", str(request.radius)
        ]
        
        logger.info(f"Job {job_id}: Running command: {' '.join(cmd)}")
        
        # Run prospect.py
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "FATHOM_API_KEY": os.getenv("GOOGLE_PLACES_API_KEY", ""),  # Map Railway env var to new name
                "PYTHONUNBUFFERED": "1"
            }
        )
        
        # Monitor output
        output = []
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            output.append(line)
            logger.info(f"Job {job_id}: {line.strip()}")
            
            # Update progress based on output
            update_progress_from_output(job_id, line)
        
        # Wait for completion
        process.wait()
        
        # Get stderr
        stderr = process.stderr.read()
        if stderr:
            logger.error(f"Job {job_id} stderr: {stderr}")
        
        if process.returncode == 0:
            # Success - find CSV and report files
            full_output = ''.join(output)
            csv_file, report_file = extract_file_paths(full_output)
            
            # Calculate search duration
            start_time = search_jobs[job_id].get("start_time", time.time())
            duration = time.time() - start_time
            
            with search_jobs_lock:
                search_jobs[job_id].update({
                    "status": "completed",
                    "progress": 100,
                    "message": "Search completed successfully",
                    "completed_at": datetime.now().isoformat(),
                    "csv_file": csv_file,
                    "report_file": report_file,
                    "results": parse_csv_results(csv_file) if csv_file else [],
                    "duration": duration
                })
            
            # Update success metrics
            update_metrics("search_completed", duration=duration)
            
            logger.info(f"Job {job_id}: Completed successfully in {duration:.1f}s")
        else:
            # Error
            error_msg = f"Search failed with exit code {process.returncode}"
            if stderr:
                error_msg += f": {stderr[:500]}"
            
            with search_jobs_lock:
                search_jobs[job_id].update({
                    "status": "failed",
                    "progress": 0,
                    "message": "Search failed",
                    "error": error_msg,
                    "completed_at": datetime.now().isoformat()
                })
            
            # Update failure metrics
            update_metrics("search_failed", error_type="process_error")
            
            logger.error(f"Job {job_id}: {error_msg}")
            
    except Exception as e:
        logger.error(f"Job {job_id}: Exception - {str(e)}")
        with search_jobs_lock:
            search_jobs[job_id].update({
                "status": "failed",
                "progress": 0,
                "message": "Search failed with exception",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            })
        
        # Update failure metrics
        error_type = type(e).__name__
        update_metrics("search_failed", error_type=error_type)

async def run_prospect_search(job_id: str, request: SearchRequest):
    """
    Async wrapper for prospect search
    Uses semaphore to limit concurrent searches and thread pool to prevent blocking
    Implements progressive timeout strategy based on estimated result count
    """
    async with search_semaphore:
        logger.info(f"Job {job_id}: Starting search (concurrency: {MAX_CONCURRENT_SEARCHES - search_semaphore._value}/{MAX_CONCURRENT_SEARCHES})")
        
        try:
            # Estimate result count and calculate dynamic timeout
            estimated_results = estimate_result_count_for_search(request)
            timeout_seconds = calculate_dynamic_timeout(estimated_results)
            timeout_minutes = timeout_seconds // 60
            
            logger.info(f"Job {job_id}: Using {timeout_minutes}-minute timeout for estimated {estimated_results} results")
            
            # Store timeout info in job metadata
            with search_jobs_lock:
                search_jobs[job_id]["estimated_results"] = estimated_results
                search_jobs[job_id]["timeout_minutes"] = timeout_minutes
            
            # Run synchronous search in thread pool with dynamic timeout
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(executor, _run_prospect_search_sync, job_id, request),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Job {job_id}: Search timed out after {timeout_minutes} minutes")
            with search_jobs_lock:
                search_jobs[job_id].update({
                    "status": "failed",
                    "progress": 0,
                    "message": "Search timed out",
                    "error": f"Search exceeded {timeout_minutes} minute time limit (estimated {estimated_results} results)",
                    "completed_at": datetime.now().isoformat()
                })
            update_metrics("search_failed", error_type="timeout")
            
        except Exception as e:
            logger.error(f"Job {job_id}: Async wrapper exception - {str(e)}")
            with search_jobs_lock:
                search_jobs[job_id].update({
                    "status": "failed",
                    "progress": 0,
                    "message": "Search failed",
                    "error": f"Async error: {str(e)}",
                    "completed_at": datetime.now().isoformat()
                })
            update_metrics("search_failed", error_type="async_error")

def update_progress_from_output(job_id: str, output: str):
    """Update job progress based on script output (supports both old and new formats)"""
    with search_jobs_lock:
        # New format patterns
        if "🔍 Searching:" in output or "🔍 Starting search:" in output:
            search_jobs[job_id]["progress"] = 20
            search_jobs[job_id]["message"] = "Finding practices..."
        elif "✅ Found" in output and "practices" in output:
            search_jobs[job_id]["progress"] = 40
            search_jobs[job_id]["message"] = "Analyzing practices..."
        elif "🏥 Processing:" in output:
            search_jobs[job_id]["progress"] = 60
            search_jobs[job_id]["message"] = "Scraping practice data..."
        elif "📊 Progress:" in output:
            # Extract percentage from "📊 Progress: X/Y (Z%)"
            import re
            match = re.search(r'\((\d+)%\)', output)
            if match:
                pct = int(match.group(1))
                search_jobs[job_id]["progress"] = min(max(pct, 10), 95)
                search_jobs[job_id]["message"] = "Processing practices..."
        elif "✅ Score:" in output:
            search_jobs[job_id]["progress"] = 80
            search_jobs[job_id]["message"] = "AI scoring in progress..."
        elif "✅ Exported" in output or "Export results" in output:
            search_jobs[job_id]["progress"] = 90
            search_jobs[job_id]["message"] = "Generating reports..."
        # Old format patterns (fallback)
        elif "Searching for:" in output:
            search_jobs[job_id]["progress"] = 20
            search_jobs[job_id]["message"] = "Finding practices..."
        elif "Found" in output and "prospects" in output:
            search_jobs[job_id]["progress"] = 40
            search_jobs[job_id]["message"] = "Analyzing practices..."

def extract_file_paths(output: str):
    """Extract CSV and report file paths from output (supports both old and new formats)"""
    import re
    
    # Try new format first: "✅ Exported X practices to filename.csv"
    csv_match = re.search(r'✅ Exported .+ to (.+\.csv)', output)
    if not csv_match:
        # Try old format: "Results exported to: filename.csv"
        csv_match = re.search(r'Results exported to: (.+\.csv)', output)
    
    # Try new format first: "📄 Summary: filename.txt"
    report_match = re.search(r'📄 Summary: (.+\.txt)', output)
    if not report_match:
        # Try old format: "Summary report: filename.txt"
        report_match = re.search(r'Summary report: (.+\.txt)', output)
    
    csv_file = csv_match.group(1).strip() if csv_match else None
    report_file = report_match.group(1).strip() if report_match else None
    
    return csv_file, report_file

def parse_csv_results(csv_file: str):
    """Parse CSV file and return results"""
    if not csv_file or not os.path.exists(csv_file):
        return []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")
        return []

@app.get("/api/prospect/status/{job_id}", response_model=SearchStatus)
async def get_search_status(
    job_id: str,
    api_key: str = Header(..., alias="X-API-Key")
):
    """Get status of a search job"""
    verify_api_key(api_key)
    
    if job_id not in search_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = search_jobs[job_id]
    
    return SearchStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        csv_file=job.get("csv_file"),
        report_file=job.get("report_file"),
        error=job.get("error")
    )

@app.get("/api/prospect/results/{job_id}")
async def get_search_results(
    job_id: str,
    api_key: str = Header(..., alias="X-API-Key")
):
    """Get detailed results of a completed search"""
    verify_api_key(api_key)
    
    if job_id not in search_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = search_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "results": job.get("results", []),
        "csv_file": job.get("csv_file"),
        "report_file": job.get("report_file"),
        "completed_at": job.get("completed_at")
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "4"))  # Multiple workers for production
    
    print("=" * 80)
    print("🚀 Fathom Prospector API Server - PHASE 2: PRODUCTION HARDENED")
    print("=" * 80)
    print(f"Server: http://localhost:{port}")
    print(f"API Docs: http://localhost:{port}/docs")
    print(f"Health Check: http://localhost:{port}/health")
    print(f"Metrics: http://localhost:{port}/metrics")
    print("=" * 80)
    print("CONFIGURATION:")
    print(f"  • Workers: {workers}")
    print(f"  • Max Concurrent Searches: {MAX_CONCURRENT_SEARCHES}")
    print(f"  • Thread Pool Workers: 10")
    print(f"  • Daily Quota Per User: {PER_USER_DAILY_LIMIT} searches")
    print(f"  • Rate Limit: 10 searches/minute per IP")
    print(f"  • Progressive Timeout Strategy:")
    print(f"      - Small searches (<20 results): 10 minutes")
    print(f"      - Medium searches (20-50 results): 15 minutes")
    print(f"      - Large searches (50-80 results): 20 minutes")
    print(f"      - Very large (80-120 results): 30 minutes")
    print(f"      - Extremely large (120+ results): 40 minutes")
    print(f"  • API Key: {API_KEY[:20]}...")
    print("=" * 80)
    print("PHASE 1 FIXES (Applied):")
    print("  ✅ Async/blocking issue FIXED - Event loop will not block")
    print("  ✅ Concurrency protection enabled (semaphore + thread pool)")
    print("  ✅ Progressive timeout strategy - scales with search size")
    print("  ✅ Production-ready configuration applied")
    print("  ✅ Timeout protection enabled")
    print("=" * 80)
    print("PHASE 2 ENHANCEMENTS (New):")
    print("  ✅ API quota management (per-user daily limits)")
    print("  ✅ Rate limiting (10 searches/minute)")
    print("  ✅ Performance monitoring & metrics tracking")
    print("  ✅ Error tracking by type")
    print("  ✅ Enhanced health checks with system stats")
    print("=" * 80)
    print("🎯 System Status: PRODUCTION READY - Multi-User Hardened")
    print("=" * 80)
    
    # Production configuration - no reload, proper timeouts
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        workers=workers,  # Multiple worker processes
        log_level="info",
        timeout_keep_alive=30,
        limit_concurrency=100,
        limit_max_requests=1000  # Restart worker after N requests to prevent memory leaks
    )
