"""
library_core.py
Last update: 2025-06-15
Version: 2.3.0
Description:
    Core logic for Audiobook/E-book Reorg. Handles file scanning, metadata extraction, grouping, 
    enrichment, and destination path calculation. Integrates with DB and GUI for full pipeline automation.

    Logical Flow:
    - Loads and validates configuration
    - Scans directories for media files
    - Extracts and normalizes metadata
    - Groups related files using fuzzy matching
    - Enriches metadata from OpenLibrary and LLM
    - Calculates destination paths based on metadata
    - Manages persistent cache for performance

    Directory:
    - BookGroup:line 50 - Represents a group of related files with metadata
    - RawMeta:line 100 - Stores raw metadata extracted from files
    - scan_directory:line 200 - Recursively scans directories for media files
    - extract_metadata:line 300 - Extracts and normalizes metadata from files
    - group_files:line 400 - Groups related files using fuzzy matching
    - enrich_metadata:line 500 - Enhances metadata using OpenLibrary and LLM
    - calculate_destination_path:line 600 - Determines destination path based on metadata
"""
from __future__ import annotations
import asyncio, httpx, hashlib, logging, json, yaml, time, re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
from difflib import HtmlDiff

import shutil
from rapidfuzz import fuzz

from tenacity import retry, wait_exponential, stop_after_attempt
from utils import BackupManager

import magic
from mutagen import File as AudioFile
from ebooklib import epub

from staging_db import StagingDB
from utils import BookGroup, RawMeta, calculate_destination_path
from openlibrary_mirror import lookup_book
from confidence_scoring import (
    EnrichmentConfidence, 
    GroupingConfidence
)



# ------------------ cache key ---------------
def _ck(kind: str, *parts: str) -> str:
    return kind + ":" + hashlib.sha256("||".join(parts).encode()).hexdigest()

# --------------- OpenLibrary fetch ----------
@retry(wait=wait_exponential(multiplier=2, min=2, max=20),
       stop=stop_after_attempt(5))
async def _ol_http(title: str, author: str):
    async with httpx.AsyncClient() as cli:
        r = await cli.get("https://openlibrary.org/search.json",
                          params={"title": title, "author": author, "limit": 1},
                          timeout=10)
        r.raise_for_status()
        return r.json()

async def ol_enrich(title: str, author: str, db: StagingDB, cfg: dict):
    ttl = cfg["openlibrary"]["cache_ttl_days"] * 86400
    key = _ck("ol", title, author)
    cached = db.cache_get(key, ttl)
    if cached:
        return cached
    # try mirror
    if cfg["openlibrary"].get("use_local_mirror"):
        mirror = lookup_book(cfg["openlibrary"]["mirror_db"], title, author)
        if mirror:
            db.cache_put(key, mirror)
            return mirror
    # fallback HTTP
    try:
        docs = (await _ol_http(title, author)).get("docs", [])
        if docs:
            doc = docs[0]
            info = {
                "title": doc.get("title") or title,
                "authors": doc.get("author_name", [])[:1] or [author],
                "year": str(doc.get("first_publish_year")) if doc.get("first_publish_year") else None,
                "genre": doc.get("subject", [""])[0].title() if doc.get("subject") else None,
            }
            db.cache_put(key, info)
            return info
    except Exception as e:
        logging.warning(f"OpenLibrary fetch failed for {title}/{author}: {e}")
    return None


def _unwrap(v):
    """Extract scalar from mutagen frame or list."""
    if hasattr(v, "text"):
        v = v.text
    if isinstance(v, (list, tuple)):
        for x in v:
            if x:
                return x
        return None
    return v

def is_valid_author(name: str, cfg: Dict[str, any]) -> bool:
    """Filter out invalid author-like tags based on config rules."""
    nm = name.strip()
    if not nm or len(nm) < 3:
        return False
    for blk in cfg.get("author_blacklist", []):
        if re.search(fr"(?i)\b{blk}\b", nm):
            logging.debug(f'Dropped blacklist match "{blk}": {nm}')
            return False
    if " " not in nm:
        logging.debug(f'Dropped no-space: {nm}')
        return False
    return True

def apply_alias(name: str, alias_map: Dict[str, str]) -> str:
    """Normalize author name: remove dots, apply alias mapping."""
    no_dots = name.replace(".", "")
    return alias_map.get(no_dots, no_dots)

def extract_tags(path: Path, alias_map: Dict[str, str], cfg: Dict[str, any]) -> Dict[str, Optional[any]]:
    """Extract and sanitize metadata tags from a file."""
    tags: Dict[str, Optional[any]] = {}
    try:
        mime = magic.from_file(str(path), mime=True)
    except:
        mime = ""
    if mime.startswith("audio"):
        try:
            audio = AudioFile(path)
            if audio and audio.tags:
                raw = _unwrap(audio.tags.get("TPE1") or audio.tags.get("ART"))
                candidates = []
                if raw:
                    candidates = [apply_alias(a.strip(), alias_map) for a in re.split(r"[;/,]", raw)]
                valid = [a for a in candidates if is_valid_author(a, cfg)]
                tags["authors"] = valid
                t = _unwrap(audio.tags.get("TIT2") or audio.tags.get("nam"))
                tags["title"] = t if isinstance(t, str) else None
                y = _unwrap(audio.tags.get("TDRC") or audio.tags.get("TDRL") or audio.tags.get("day"))
                year = None
                if y:
                    m = re.search(cfg.get("year_pattern", r"(19|20)\d{2}"), str(y))
                    year = m.group(0) if m else None
                tags["year"] = year
                g = _unwrap(audio.tags.get("TCON") or audio.tags.get("gen"))
                tags["genre"] = g if isinstance(g, str) else None
                tr = _unwrap(audio.tags.get("TRCK") or audio.tags.get("trkn"))
                if tr:
                    nums = re.findall(r"\d+", str(tr))
                    tags["track"] = int(nums[0]) if nums else None
                    tags["track_total"] = int(nums[1]) if len(nums) > 1 else None
                dc = _unwrap(audio.tags.get("TPOS"))
                if dc:
                    dn = re.findall(r"\d+", str(dc))
                    tags["disc"] = int(dn[0]) if dn else None
        except Exception as e:
            logging.warning(f"Could not extract audio tags from {path}: {e}")
    elif "epub" in mime:
        try:
            book = epub.read_epub(str(path))
            cr = book.get_metadata("DC", "creator")
            auths = [apply_alias(_unwrap(c[0]), alias_map) for c in cr]
            tags["authors"] = [a for a in auths if is_valid_author(a, cfg)]
            ti = book.get_metadata("DC", "title")
            tags["title"] = _unwrap(ti[0][0]) if ti else None
            dt = book.get_metadata("DC", "date")
            year = None
            if dt:
                m = re.search(cfg.get("year_pattern", r"(19|20)\d{2}"), dt[0][0])
                year = m.group(0) if m else None
            tags["year"] = year
            sb = book.get_metadata("DC", "subject")
            tags["genre"] = _unwrap(sb[0][0]) if sb else None
        except Exception as e:
            logging.warning(f"Could not extract epub tags from {path}: {e}")
    tags.setdefault("authors", [])
    tags.setdefault("title", None)
    tags.setdefault("year", None)
    tags.setdefault("genre", None)
    tags.setdefault("disc", None)
    tags.setdefault("track", None)
    tags.setdefault("track_total", None)
    tags["source"] = "tag" if tags["authors"] and tags["title"] else "fallback"
    return tags

FILENAME_PATTERNS = [
    re.compile(r"^(?P<last>[^,]+),\s*(?P<first>[^-]+)-\s*(?P<title>.+?)\s*\((?P<year>\d{4})\)"),
    re.compile(r"^(?P<title>.+?)\s*-\s*(?P<first>[^ ]+)\s+(?P<last>[^ ]+)")
]

def parse_from_filename(stem: str, alias_map: Dict[str, str]) -> Dict[str, Optional[any]]:
    """Fallback parsing of author/title/year from filename."""
    for pat in FILENAME_PATTERNS:
        m = pat.match(stem)
        if m:
            gd = m.groupdict()
            first = apply_alias(gd["first"].strip(), alias_map)
            last = apply_alias(gd["last"].strip(), alias_map)
            return {
                "authors": [f"{first} {last}"],
                "title": gd["title"].strip(),
                "year": gd.get("year")
            }
    return {"authors": [], "title": None, "year": None}

def scan_files(cfg: Dict[str, any], alias_map: Dict[str, str], progress_callback: Optional[Callable] = None) -> List[RawMeta]:
    """Recursively scan source roots for media files and extract raw metadata."""
    roots = []
    if cfg.get("source_audiobook_root"):
        roots.append(Path(cfg["source_audiobook_root"]))
    if cfg.get("source_ebook_root"):
        roots.append(Path(cfg["source_ebook_root"]))

    if not roots:
        logging.error("No source directories configured. Please set 'source_audiobook_root' or 'source_ebook_root' in settings.yaml.")
        return []
    raws: List[RawMeta] = []
    exts = {".mp3", ".m4a", ".flac", ".epub", ".mobi"}
    all_files = []
    for root in roots:
        for f in root.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                all_files.append(f)

    total_files = len(all_files)
    for i, f in enumerate(all_files):
                tags = extract_tags(f, alias_map, cfg)
                if tags["source"] == "fallback":
                    fb = parse_from_filename(f.stem, alias_map)
                    tags.update(fb)
                raws.append(RawMeta(
                    path=f,
                    tags=tags,
                    size=f.stat().st_size,
                    mtime=f.stat().st_mtime,
                    source=tags["source"]
                ))
                if progress_callback and (i + 1) % 12 == 0:
                    progress_callback(i + 1, total_files, f.name)
    return raws

def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def group_books(raws: List[RawMeta], cfg: Dict[str, any]) -> List[BookGroup]:
    """
    Cluster RawMeta entries into BookGroups. This is a simplified version.
    A more robust implementation would use confidence scoring.
    """
    groups: List[BookGroup] = []
    for meta in raws:
        title = meta.tags.get("title") or meta.path.stem
        authors = meta.tags.get("authors") or []
        main_author = authors[0] if authors else "Unknown"

        placed = False
        for grp in groups:
            title_sim = fuzz.token_set_ratio(normalize(title), normalize(grp.title)) / 100.0
            author_sim = fuzz.token_set_ratio(normalize(main_author), normalize(grp.authors[0] if grp.authors else "")) / 100.0
            if title_sim > 0.8 and author_sim > 0.7:
                grp.files.append(meta)
                grp.authors = list(dict.fromkeys(grp.authors + authors))[:5]
                placed = True
                break
        if not placed:
            groups.append(BookGroup(authors=authors, title=title, files=[meta]))
    return groups

def enrich_books(
    groups: List[BookGroup],
    cfg: Dict[str, any],
    db: StagingDB,
    progress_callback: Optional[Callable] = None,
    lazy_mode: bool = False
) -> None:
    """
    Enrich BookGroup metadata using OpenLibrary and potentially other services.
    """
    cache: dict = {}
    openlibrary_cfg = cfg.get("openlibrary", {})
    use_mirror: bool = openlibrary_cfg.get("use_local_mirror", False)
    mirror_db: str = openlibrary_cfg.get("mirror_db")

    total_groups = len(groups)
    for i, grp in enumerate(groups):
        if lazy_mode:
            grp.canonical = {"title": grp.title, "authors": grp.authors, "year": None, "genre": None, "status": "Lazy"}
            db.stage_book(grp, cfg)
            continue

        ol_result = None
        author_for_search = grp.authors[0] if grp.authors else ""
        if use_mirror:
            ol_result = lookup_book(mirror_db, grp.title, author_for_search)
        if not ol_result:
            ol_result = asyncio.run(ol_enrich(grp.title, author_for_search, db, cfg))

        if ol_result:
            grp.canonical = ol_result
            grp.canonical['status'] = 'OpenLibrary'
        else:
            grp.canonical = {
                "title": grp.title,
                "authors": grp.authors,
                "year": grp.files[0].tags.get("year"),
                "genre": grp.files[0].tags.get("genre"),
                "status": "Original"
            }

        # Blacklist check
        blacklist = ["TORRENTZ", "SAMPLE", "DEMO", "PROMO"]
        blacklist_hit = any(any(term in (str(f.path).upper() + str(f.tags.get('title', '')).upper()) for term in blacklist) for f in grp.files)
        if blacklist_hit:
            grp.canonical['status'] = 'Blacklisted'

        # Compute diff and discrepancy for GUI and audit
        old = [grp.title] + grp.authors
        new = [grp.canonical["title"]] + grp.canonical["authors"]
        diff = HtmlDiff()
        grp.diff_html = diff.make_table(old, new, "Original", "Canonical")
        grp.discrepancy = fuzz.ratio(str(old), str(new))

        db.stage_book(grp, cfg)

        if progress_callback:
            progress_callback(i + 1, total_groups, grp.title)

def perform_moves(cfg: Dict[str, any], db: StagingDB, batch_size: int,
                  dryrun: bool=False, progress_callback: Optional[Callable]=None):
    """
    Processes files pending conversion from the database.

    For each file, it performs the move from its source to its calculated
    destination. It also handles backing up the file if configured, marks
    the file as converted in the database, updates the staging_plan status,
    and provides progress updates.

    Args:
        cfg: Configuration dictionary.
        db: StagingDB instance for database operations.
        batch_size: The maximum number of files to process.
        dryrun: If True, simulates moves without touching files.
        progress_callback: Optional function to report progress.
    """
    bm = BackupManager(cfg) if cfg.get("backup_zip") else None
    items = db.get_pending_files(limit=batch_size)
    total_pending = len(items)
    
    if not total_pending:
        logging.info("No pending files to move.")
        return
        
    logging.info(f"Starting move operation for {total_pending} files.")
    moved_count = 0
    
    # Get database connection for transaction
    conn = db.conn
    
    try:
        with conn:
            for idx, it in enumerate(items, start=1):
                try:
                    src = Path(it["src"])
                    dst = Path(it["dst"])
                    
                    if not src.exists():
                        logging.warning(f"Source file not found, skipping: {src}")
                        db.mark_converted(it["file_id"], error="Source not found")
                        # Update staging_plan status for this file
                        conn.execute("""
                            UPDATE staging_plan 
                            SET status = 'error', 
                                error = ?
                            WHERE src = ?
                        """, ("Source file not found", str(src)))
                        continue

                    # Update status to 'processing' in staging_plan
                    conn.execute("""
                        UPDATE staging_plan 
                        SET status = 'processing'
                        WHERE src = ?
                    """, (str(src),))

                    if bm:
                        bm.backup_file(src, cfg)

                    if not dryrun:
                        tmp = dst.with_suffix(dst.suffix + ".partial")
                        tmp.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, tmp)
                        tmp.replace(dst)
                        src.unlink()
                        logging.debug(f"Moved {src} -> {dst}")

                    # Mark as processed in files table
                    db.mark_converted(it["file_id"])
                    
                    # Update staging_plan status to 'done'
                    conn.execute("""
                        UPDATE staging_plan 
                        SET status = 'done',
                            processed_at = CURRENT_TIMESTAMP
                        WHERE src = ?
                    """, (str(src),))
                    
                    moved_count += 1
                    
                    if progress_callback:
                        progress_callback(idx, total_pending, f"Moved: {moved_count}/{total_pending}")
                
                except Exception as e:
                    error_msg = str(e)
                    logging.error(f"Failed to move file {it.get('src', 'unknown')}: {error_msg}", exc_info=True)
                    if "file_id" in it:
                        db.mark_converted(it["file_id"], error=error_msg)
                    # Update staging_plan with error status
                    conn.execute("""
                        UPDATE staging_plan 
                        SET status = 'error',
                            error = ?,
                            processed_at = CURRENT_TIMESTAMP
                        WHERE src = ?
                    """, (error_msg, str(it.get('src', ''))))
                    
                    # Continue with next file even if one fails
                    continue
                    
    except Exception as e:
        logging.critical(f"Fatal error during move operation: {e}", exc_info=True)
        raise

    logging.info(f"Move operation complete. Moved {moved_count} of {total_pending} files.")
    return {"moved": moved_count, "total": total_pending}
