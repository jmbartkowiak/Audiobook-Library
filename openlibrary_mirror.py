"""
openlibrary_mirror.py
Version: 2.3.0
Last Update: 2025-06-20
Description: 
    Handles the download and conversion of the OpenLibrary TSV data dump into a 
    local SQLite database. This process is triggered automatically on the first 
    GUI launch and runs periodically. The SQLite mirror enables fast, offline 
    metadata enrichment for audiobooks and e-books.

Directory:
- lookup_book:line 81 - Queries the local SQLite mirror for book information.
- MirrorUpdater:line 107 - Main class for managing the OpenLibrary mirror.
- MirrorUpdater.needs_update:line 164 - Checks if the mirror needs to be downloaded/updated.
- MirrorUpdater.update_mirror:line 175 - Checks disk space, then downloads and converts the TSV dump.
- _convert_tsv_to_sqlite:line 47 - Helper to stream-convert the TSV dump, respecting `prefer_pigz` config.
- _download_file:line 203 - Helper to download a file with progress reporting.
"""
import sqlite3
import json
import time
import logging
import shutil
import subprocess
import tempfile
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Cache for storing failed requests to avoid repeated timeouts
TIMEOUT_CACHE = {}
TIMEOUT_CACHE_TTL = timedelta(hours=1)  # Cache timeouts for 1 hour

# Use the logging infrastructure from the main application
log = logging.getLogger(__name__)


# *=*=* PURPOSE: helper for fast TSV ➟ SQLite conversion *=*=* 
def _convert_tsv_to_sqlite(tsv_gz: Path, sqlite_path: Path, prefer_pigz: bool = True) -> None:
    """Stream-convert an OpenLibrary TSV dump into a SQLite DB."""
    # pigz is multi-core gzip.  Fallback to gzip if unavailable.
    decompressor = ["pigz", "-dc"] if prefer_pigz and shutil.which("pigz") else ["gzip", "-dc"]
    cmd = decompressor + [str(tsv_gz)]
    log.info(f"Starting conversion: {' '.join(cmd)} | openlibrary-to-sqlite ...")
    proc1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        # The check=True flag will raise CalledProcessError on non-zero exit codes.
        proc2 = subprocess.run(
            ["openlibrary-to-sqlite", str(sqlite_path), "-"],
            stdin=proc1.stdout,
            check=True,
            capture_output=True, text=True # Capture stdout/stderr for logging
        )
        log.info("openlibrary-to-sqlite finished successfully.")
        if proc2.stdout:
            log.info(f"[openlibrary-to-sqlite stdout]:\n{proc2.stdout}")
        if proc2.stderr:
            log.info(f"[openlibrary-to-sqlite stderr]:\n{proc2.stderr}")

    except subprocess.CalledProcessError as e:
        log.error(f"Conversion process failed with exit code {e.returncode}.")
        # Capture and log stderr/stdout from the failed process
        stderr = e.stderr or "(not captured)"
        stdout = e.stdout or "(not captured)"
        log.error(f"Stderr: {stderr.strip()}")
        log.error(f"Stdout: {stdout.strip()}")
        # Re-raise the error to be caught by the GUI thread
        raise e
    finally:
        # Ensure the first process's stdout is closed to allow it to terminate
        if proc1.stdout:
            proc1.stdout.close()
        # Wait for the decompressor process to finish
        proc1.wait()


# --- Existing lookup function (unchanged) ---
def lookup_book(db_path: str, title: str, author: str) -> Optional[dict]:
    """Lookup a book in the local OpenLibrary mirror by title and author."""
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        return None
    try:
        # Connect in read-only mode, ensuring the connection is closed
        with sqlite3.connect(f'file:{db_path_obj.as_uri()}?mode=ro', uri=True) as cx:
            cx.row_factory = sqlite3.Row
            query = f'{title} {author}'
            row = cx.execute(
                "SELECT payload FROM books WHERE books MATCH ? ORDER BY rank LIMIT 1",
                (query,)
            ).fetchone()
            if row:
                try:
                    return json.loads(row["payload"])
                except json.JSONDecodeError:
                    log.error(f"Failed to decode JSON payload for query: {query}")
                    return None
            return None
    except sqlite3.Error as e:
        log.error(f"Database error while looking up book: {e}")
        return None


# --- Mirror Updater ---
class MirrorUpdater:
    """
    Handles OpenLibrary mirror update logic:
    - Checks if the mirror exists or is outdated.
    - On first run or if outdated, downloads the TSV dump.
    - Converts the TSV dump to a SQLite database.
    - Reports progress for GUI integration.
    """
    def __init__(self, cfg: dict, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.cfg = cfg.get("openlibrary", {})
        self.progress_cb = progress_cb or (lambda x: None)
        
        # Configure paths using pathlib for OS independence
        db_path_str = self.cfg.get("sqlite_path", "data/openlibrary/mirror.sqlite")
        self.mirror_db_path = Path(db_path_str).resolve()
        self.mirror_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.last_update_file = self.mirror_db_path.with_suffix('.sqlite.lastupdate')
        
        self.dump_url = self.cfg.get("dump_url")
        self.chunk_size = 1024 * 1024  # 1MB
        self.min_days = self.cfg.get("min_update_days", 14)
        self.prefer_pigz = self.cfg.get("prefer_pigz", True)
        
        # Configure retry strategy for requests
        retry_strategy = Retry(
            total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"], respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({'User-Agent': 'AudiobookApp/2.3.0'})

    def get_last_update(self) -> Optional[float]:
        """Return last update timestamp, or None if never updated."""
        if not self.last_update_file.exists():
            return None
        try:
            return float(self.last_update_file.read_text().strip())
        except (IOError, ValueError):
            log.warning(f"Could not read or parse last update file: {self.last_update_file}")
            return None

    def set_last_update(self, t: float) -> None:
        """Write last update timestamp."""
        try:
            self.last_update_file.write_text(str(t))
        except IOError as e:
            log.error(f"Failed to write last update file: {e}")

    def needs_update(self) -> bool:
        """Return True if mirror DB doesn't exist or is older than min_days."""
        if not self.mirror_db_path.exists():
            return True
        last = self.get_last_update()
        if last is None:
            return True
        return (time.time() - last) > self.min_days * 86400

    def update_mirror(self) -> None:
        """
        Downloads TSV, converts to SQLite, and updates the timestamp.
        
        This method handles the entire mirror update process including:
        1. Validating configuration and disk space
        2. Downloading the TSV dump with progress reporting
        3. Converting the TSV to SQLite format
        4. Updating the last update timestamp
        5. Handling errors and reporting progress
        """
        if not self.dump_url:
            error_msg = "openlibrary.dump_url is not configured in settings.yaml"
            log.error(error_msg)
            self.progress_cb({"status": "error", "error": error_msg})
            raise RuntimeError(error_msg)

        # Check disk space before starting download
        try:
            free_space = shutil.disk_usage(self.mirror_db_path.parent).free
            required_space = 4 * 1024**3  # 4 GB
            if free_space < required_space:
                error_msg = (
                    f"Insufficient disk space. At least 4 GB is required, but only "
                    f"{free_space / 1024**3:.2f} GB is available in {self.mirror_db_path.parent}."
                )
                log.error(error_msg)
                self.progress_cb({
                    "status": "error",
                    "error": error_msg,
                    "free_space_gb": free_space / 1024**3,
                    "required_gb": 4.0
                })
                raise RuntimeError(error_msg)
        except FileNotFoundError as e:
            error_msg = f"Cannot check disk space: {e}"
            log.warning(error_msg)
            self.progress_cb({"status": "warning", "message": error_msg, "path": str(e)})
            # Continue with the download despite the warning

        try:
            # Create a temporary directory for the download
            with tempfile.TemporaryDirectory(prefix="ol_mirror_") as td:
                temp_dir = Path(td)
                dump_gz = temp_dir / "ol_dump_latest.txt.gz"
                
                # Download the TSV dump with progress reporting
                log.info(f"Starting download of OpenLibrary dump from {self.dump_url}")
                self.progress_cb({
                    "status": "downloading",
                    "url": self.dump_url,
                    "destination": str(dump_gz),
                    "start_time": time.time()
                })
                
                self._download_file(self.dump_url, dump_gz)
                
                # Verify the downloaded file
                if not dump_gz.exists() or dump_gz.stat().st_size == 0:
                    error_msg = f"Download failed: Empty or missing file at {dump_gz}"
                    log.error(error_msg)
                    self.progress_cb({"status": "error", "error": error_msg})
                    raise RuntimeError(error_msg)
                
                log.info(f"Download complete. Starting conversion to SQLite at {self.mirror_db_path}")
                self.progress_cb({
                    "status": "converting",
                    "source": str(dump_gz),
                    "destination": str(self.mirror_db_path),
                    "size_mb": dump_gz.stat().st_size / (1024 * 1024)
                })
                
                # Convert the TSV to SQLite
                start_time = time.time()
                _convert_tsv_to_sqlite(dump_gz, self.mirror_db_path, prefer_pigz=self.prefer_pigz)
                
                # Verify the SQLite database was created
                if not self.mirror_db_path.exists() or self.mirror_db_path.stat().st_size == 0:
                    error_msg = f"Conversion failed: Empty or missing SQLite database at {self.mirror_db_path}"
                    log.error(error_msg)
                    self.progress_cb({"status": "error", "error": error_msg})
                    raise RuntimeError(error_msg)
                
                # Update the last update timestamp
                update_time = time.time()
                self.set_last_update(update_time)
                
                # Log success with statistics
                db_size_mb = self.mirror_db_path.stat().st_size / (1024 * 1024)
                elapsed = time.time() - start_time
                log.info(
                    f"Successfully updated OpenLibrary mirror. "
                    f"Size: {db_size_mb:.1f} MB, Time: {elapsed:.1f}s"
                )
                
                # Send final progress update
                self.progress_cb({
                    "status": "complete",
                    "updated": True,
                    "path": str(self.mirror_db_path),
                    "size_mb": db_size_mb,
                    "elapsed_seconds": elapsed,
                    "timestamp": update_time,
                    "message": "OpenLibrary mirror update completed successfully"
                })
                
        except Exception as e:
            error_msg = f"Failed to update OpenLibrary mirror: {e}"
            log.error(error_msg, exc_info=True)
            self.progress_cb({
                "status": "error",
                "error": str(e),
                "type": type(e).__name__,
                "message": error_msg
            })
            raise RuntimeError(error_msg) from e

    def _is_request_cached(self, url: str) -> Tuple[bool, Optional[str]]:
        """Check if a request is in the timeout cache.
        
        Returns:
            Tuple of (is_cached, error_message) where error_message is None if not cached.
        """
        cache_key = hashlib.md5(url.encode()).hexdigest()
        if cache_key in TIMEOUT_CACHE:
            cached_time, error_msg = TIMEOUT_CACHE[cache_key]
            if datetime.now() - cached_time < TIMEOUT_CACHE_TTL:
                return True, error_msg
            del TIMEOUT_CACHE[cache_key]  # Cache expired
        return False, None

    def _cache_request_error(self, url: str, error_msg: str) -> None:
        """Cache a failed request to avoid repeated timeouts."""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        TIMEOUT_CACHE[cache_key] = (datetime.now(), error_msg)
        log.warning(f"Cached error for {url}: {error_msg}")

    def _download_file(self, url: str, dest_path: Path) -> None:
        """
        Download a file with progress reporting and timeout handling.
        
        Args:
            url: URL to download from
            dest_path: Local path to save the downloaded file
            
        Raises:
            RuntimeError: If download fails or is interrupted
        """
        # Check if this URL is in the timeout cache
        is_cached, cached_error = self._is_request_cached(url)
        if is_cached:
            error_msg = f"Skipping download (cached error): {cached_error}"
            log.warning(error_msg)
            self.progress_cb({"status": "error", "error": error_msg, "url": url})
            raise RuntimeError(error_msg)

        try:
            # Increased timeout for large file downloads
            with self.session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                
                # Get total size for progress reporting
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                start_time = time.time()
                last_update = 0
                
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=self.chunk_size):
                        if not chunk:  # Skip empty chunks
                            continue
                            
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Throttle progress updates to once per second
                        current_time = time.time()
                        if current_time - last_update >= 1.0 or downloaded == total_size:
                            elapsed = current_time - start_time
                            speed = downloaded / elapsed if elapsed > 0 else 0
                            percent = (downloaded / total_size * 100) if total_size > 0 else 0
                            
                            # Format speed as human-readable string
                            speed_str = f"{speed/1024/1024:.1f} MB/s" if speed > 1024*1024 else f"{speed/1024:.1f} KB/s"
                            
                            self.progress_cb({
                                "status": "downloading",
                                "downloaded": downloaded,
                                "total": total_size,
                                "speed": speed_str,
                                "percent": percent,
                                "elapsed": elapsed,
                                "url": url
                            })
                            last_update = current_time
                
                # Verify download size matches Content-Length
                if total_size > 0 and downloaded != total_size:
                    error_msg = f"Download incomplete. Expected {total_size} bytes, got {downloaded} bytes"
                    log.error(error_msg)
                    self.progress_cb({"status": "error", "error": error_msg, "url": url})
                    raise IOError(error_msg)
                
                log.info(f"Successfully downloaded {downloaded} bytes to {dest_path}")

        except requests.exceptions.Timeout as e:
            error_msg = f"Request timed out: {e}"
            self._cache_request_error(url, error_msg)
            log.error(error_msg)
            self.progress_cb({"status": "error", "error": error_msg, "url": url, "retryable": True})
            raise RuntimeError(error_msg) from e
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Download failed: {e}"
            log.error(error_msg)
            self.progress_cb({"status": "error", "error": error_msg, "url": url, "retryable": True})
            raise RuntimeError(error_msg) from e
            
        except IOError as e:
            error_msg = f"I/O error while saving file: {e}"
            log.error(error_msg)
            self.progress_cb({"status": "error", "error": error_msg, "url": url})
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error during download: {e}"
            log.error(error_msg, exc_info=True)
            self.progress_cb({"status": "error", "error": error_msg, "url": url})
            raise RuntimeError(error_msg) from e
