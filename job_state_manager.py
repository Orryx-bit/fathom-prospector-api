"""
Job State Manager for Multi-Replica Environments
Stores job state in persistent files that all replicas can access
"""

import os
import json
import fcntl
import threading
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Directory for storing job state files
JOB_STATE_DIR = Path("/tmp/search_jobs")
JOB_STATE_DIR.mkdir(exist_ok=True, parents=True)

# Lock for file operations
_file_lock = threading.Lock()

def save_job_state(job_id: str, state: Dict) -> None:
    """
    Save job state to persistent file (atomic write)
    All replicas can read/write to the same file system
    """
    file_path = JOB_STATE_DIR / f"{job_id}.json"
    temp_path = file_path.with_suffix('.tmp')
    
    try:
        with _file_lock:
            # Write to temp file first
            with open(temp_path, 'w') as f:
                # Lock the file for exclusive write
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(state, f, default=str)  # default=str handles datetime objects
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic rename (replaces old file)
            os.replace(temp_path, file_path)
            
    except Exception as e:
        logger.error(f"Error saving job state for {job_id}: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

def load_job_state(job_id: str) -> Optional[Dict]:
    """
    Load job state from persistent file
    Returns None if job doesn't exist
    """
    file_path = JOB_STATE_DIR / f"{job_id}.json"
    
    if not file_path.exists():
        return None
    
    try:
        with _file_lock:
            with open(file_path, 'r') as f:
                # Lock the file for shared read
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                state = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return state
    except Exception as e:
        logger.error(f"Error loading job state for {job_id}: {e}")
        return None

def update_job_state(job_id: str, updates: Dict) -> None:
    """
    Update specific fields in job state
    Loads existing state, applies updates, and saves
    """
    state = load_job_state(job_id)
    if state is None:
        logger.warning(f"Attempted to update non-existent job {job_id}")
        return
    
    state.update(updates)
    save_job_state(job_id, state)

def delete_job_state(job_id: str) -> None:
    """
    Delete job state file
    Used for cleanup of old jobs
    """
    file_path = JOB_STATE_DIR / f"{job_id}.json"
    
    try:
        with _file_lock:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted job state for {job_id}")
    except Exception as e:
        logger.error(f"Error deleting job state for {job_id}: {e}")

def list_all_jobs() -> Dict[str, Dict]:
    """
    List all jobs in the system
    Used for health checks and monitoring
    """
    jobs = {}
    
    try:
        with _file_lock:
            for file_path in JOB_STATE_DIR.glob("*.json"):
                job_id = file_path.stem
                state = load_job_state(job_id)
                if state:
                    jobs[job_id] = state
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
    
    return jobs

def cleanup_old_jobs(max_age_hours: int = 24) -> int:
    """
    Clean up job files older than max_age_hours
    Returns number of jobs deleted
    """
    import time
    
    deleted_count = 0
    cutoff_time = time.time() - (max_age_hours * 3600)
    
    try:
        with _file_lock:
            for file_path in JOB_STATE_DIR.glob("*.json"):
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Cleaned up old job: {file_path.stem}")
                except Exception as e:
                    logger.error(f"Error cleaning up {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    
    return deleted_count
