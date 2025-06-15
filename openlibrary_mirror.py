"""
openlibrary_mirror.py
Version: 2.2.0
Last Update: 2025-06-10
Description: 
    Helper for querying and updating a local SQLite FTS5 mirror of OpenLibrary. 
    Enables fast, offline metadata enrichment for audiobooks/e-books and provides 
    robust, fallback-enabled mirror updating with progress reporting for GUI integration.

Directory:
- MirrorUpdater:line 78 - Main class for handling OpenLibrary mirror updates.
- MirrorUpdater.update_mirror:line 282 - Main entry point to start the mirror update process.
- lookup_book:line 48 - Function to query the local mirror for book information.

Features:
- Fast SQLite FTS5-based search for title/author
- Returns full OpenLibrary work record as JSON
- Designed for local/offline enrichment
- Simple, one-shot lookup for integration
- Robust error handling for missing/corrupt DB
- Minimal dependencies, lightweight footprint
- Easy integration with core logic and DB
- Supports large-scale OpenLibrary dumps (12GB+)
- Out-of-band mirror build process
- Enables privacy-preserving enrichment
- MirrorUpdater: robust, fallback-enabled mirror update with progress, speed, and info for GUI
"""
import sqlite3
import json
import time
import logging
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Any, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Use the logging infrastructure from the main application
logger = logging.getLogger(__name__)

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
                    logger.error(f"Failed to decode JSON payload for query: {query}")
                    return None
            return None
    except sqlite3.Error as e:
        logger.error(f"Database error while looking up book: {e}")
        return None

# --- Mirror Updater ---
class MirrorUpdater:
    """
    Handles OpenLibrary mirror update logic:
    - Checks last update time (local file)
    - Checks if a new version is available on remote
    - Downloads with progress callback, using fallback mirrors
    - Only updates if >2 weeks since last update and a new version exists
    - Reports progress, speed, filename, file date, and active URL
    """
    def __init__(self, cfg: dict, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.cfg = cfg
        self.progress_cb = progress_cb or (lambda x: None)
        
        # Configure paths using pathlib for OS independence
        db_path_str = cfg["openlibrary"].get("mirror_db", "./ol_mirror.sqlite")
        self.mirror_db_path = Path(db_path_str).resolve()
        self.mirror_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.last_update_file = self.mirror_db_path.with_suffix('.sqlite.lastupdate')
        self.download_path = self.mirror_db_path.with_suffix('.sqlite.gz')
        self.temp_download_path = self.download_path.with_suffix('.gz.tmp')

        self.mirrors = [m["url"] for m in cfg["openlibrary"].get("mirrors", [])]
        self.chunk_size = 1024 * 1024  # 1MB
        self.min_days = cfg["openlibrary"].get("min_update_days", 14)
        
        # Configure retry strategy for requests
        self.retry_strategy = Retry(
            total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"], respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=self.retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({'User-Agent': 'AudiobookReorg/2.2.0'})

    def get_last_update(self, as_string: bool = False) -> Union[float, str]:
        """Return last update timestamp, or 'Never'/'0.0' if never updated."""
        if not self.last_update_file.exists():
            return "Never" if as_string else 0.0
        try:
            timestamp = float(self.last_update_file.read_text().strip())
            if timestamp == 0:
                return "Never" if as_string else 0.0
            if as_string:
                return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))
            return timestamp
        except (IOError, ValueError):
            logger.warning(f"Could not read or parse last update file: {self.last_update_file}")
            return "Never" if as_string else 0.0

    def set_last_update(self, t: float) -> None:
        """Write last update timestamp."""
        try:
            self.last_update_file.write_text(str(t))
        except IOError as e:
            logger.error(f"Failed to write last update file: {e}")

    def needs_update(self) -> bool:
        """Return True if >min_days since last update."""
        last = self.get_last_update()
        if not isinstance(last, float):
            return True
        return (time.time() - last) > self.min_days * 86400

    def get_remote_file_info(self, url: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Requesting remote file info from: {url}")
        self.progress_cb({"status": "checking", "url": url})
        try:
            resp = self.session.head(url, allow_redirects=True, timeout=15)
            resp.raise_for_status()
            
            last_modified_str = resp.headers.get('Last-Modified')
            last_modified_dt = None
            if last_modified_str:
                try:
                    last_modified_dt = datetime.strptime(last_modified_str, '%a, %d %b %Y %H:%M:%S %Z').replace(tzinfo=timezone.utc)
                except ValueError:
                    logger.warning(f"Could not parse Last-Modified date: {last_modified_str}")

            return {
                "last_modified": last_modified_dt,
                "content_length": int(resp.headers.get('Content-Length', 0)),
                "url": resp.url
            }
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to get remote file info from {url}: {e}"
            logger.error(error_msg)
            self.progress_cb({"status": "error", "error": error_msg, "url": url})
            return None

    def is_new_version(self, remote_info: Dict[str, Any]) -> bool:
        """Return True if remote file is newer than local or local does not exist."""
        if not self.mirror_db_path.exists():
            return True
        
        local_mtime = self.mirror_db_path.stat().st_mtime
        remote_last_modified = remote_info.get("last_modified")

        if remote_last_modified:
            remote_ts = remote_last_modified.timestamp()
            return remote_ts > local_mtime
        
        logger.warning("Remote 'Last-Modified' header not available. Assuming new version is available.")
        return True

    def download_with_progress(self, url: str, total_size: int) -> bool:
        logger.info(f"Starting download from {url} to {self.temp_download_path}")
        try:
            with self.session.get(url, stream=True, timeout=30) as r, open(self.temp_download_path, 'wb') as f:
                r.raise_for_status()
                downloaded = 0
                start_time = time.time()
                last_report_time = start_time

                self.progress_cb({
                    "status": "downloading", "filename": self.download_path.name,
                    "url": url, "downloaded": 0, "total": total_size, "percent": 0
                })

                for chunk in r.iter_content(chunk_size=self.chunk_size):
                    if not chunk: continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_report_time > 0.5:
                        elapsed = now - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        percent = (downloaded / total_size * 100) if total_size > 0 else 0
                        self.progress_cb({
                            "status": "downloading", "downloaded": downloaded, "total": total_size,
                            "speed": speed, "percent": percent
                        })
                        last_report_time = now
            
            if total_size and downloaded != total_size:
                raise IOError(f"Downloaded size mismatch. Expected {total_size}, got {downloaded}")

            self.download_path.unlink(missing_ok=True)
            self.temp_download_path.rename(self.download_path)
            logger.info(f"Successfully downloaded {self.download_path.name}")
            return True
        except (requests.exceptions.RequestException, IOError) as e:
            error_msg = f"Download failed: {e}"
            logger.error(error_msg)
            self.progress_cb({"status": "error", "error": error_msg, "url": url})
            self.temp_download_path.unlink(missing_ok=True)
            return False

    def _decompress_gz_file(self) -> bool:
        logger.info(f"Decompressing {self.download_path}")
        self.progress_cb({"status": "decompressing"})
        temp_decompressed_path = self.mirror_db_path.with_suffix('.sqlite.tmp')
        try:
            with gzip.open(self.download_path, 'rb') as f_in, open(temp_decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            
            if temp_decompressed_path.stat().st_size == 0:
                raise IOError("Decompressed file is empty.")

            self.mirror_db_path.unlink(missing_ok=True)
            temp_decompressed_path.rename(self.mirror_db_path)
            self.download_path.unlink()
            logger.info(f"Successfully decompressed to {self.mirror_db_path}")
            return True
        except (gzip.BadGzipFile, IOError, OSError) as e:
            error_msg = f"Failed to decompress {self.download_path}: {e}"
            logger.error(error_msg)
            self.progress_cb({"status": "error", "error": error_msg})
            temp_decompressed_path.unlink(missing_ok=True)
            self.download_path.unlink(missing_ok=True)
            return False

    def update_mirror(self) -> Dict[str, Any]:
        """
        Main update routine. Tries each mirror in order. Only updates if needed.
        """
        if not self.mirrors:
            msg = "No mirrors configured. Skipping update."
            logger.warning(msg)
            self.progress_cb({"status": "complete", "skipped": True, "message": msg, "done": True})
            return {"skipped": True, "error": msg}

        if not self.needs_update():
            msg = f"Skipping update, last updated {self.get_last_update(as_string=True)}."
            logger.info(msg)
            self.progress_cb({"status": "complete", "skipped": True, "message": msg, "done": True})
            return {"skipped": True, "message": msg}

        for i, url in enumerate(self.mirrors):
            logger.info(f"Trying mirror {i+1}/{len(self.mirrors)}: {url}")
            remote_info = self.get_remote_file_info(url)
            
            if not remote_info:
                logger.warning(f"Could not get info from mirror: {url}")
                continue

            if not self.is_new_version(remote_info):
                msg = f"Local mirror is up to date. Last checked against {url}."
                logger.info(msg)
                self.set_last_update(time.time())
                self.progress_cb({"status": "complete", "skipped": True, "message": msg, "done": True})
                return {"skipped": True, "message": msg, "url": url}

            if self.download_with_progress(url, remote_info.get("content_length", 0)):
                if self._decompress_gz_file():
                    self.set_last_update(time.time())
                    msg = f"Successfully updated mirror from {url}."
                    logger.info(msg)
                    self.progress_cb({"status": "complete", "updated": True, "message": msg, "done": True})
                    return {"updated": True, "url": url}
                else:
                    logger.error(f"Decompression failed for file from {url}. Continuing to next mirror.")
            else:
                logger.error(f"Download failed from {url}. Continuing to next mirror.")

        error_msg = "All mirrors failed or no new version could be processed."
        logger.error(error_msg)
        self.progress_cb({"status": "error", "error": error_msg, "done": True})
        return {"updated": False, "error": error_msg}
