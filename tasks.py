"""
tasks.py
Version: 2.3.0
Last Update: 2025-06-20
Description:
    High-level task functions for the Audiobook Reorganization tool. These functions
    are designed to be called from a GUI in a separate thread and provide progress
    updates via a callback mechanism.
"""

import time
import json
from pathlib import Path
from typing import Callable, Optional, Dict, Any

from library_core import scan_files, group_books, enrich_books, perform_moves
from staging_db import StagingDB
from utils import get_library_snapshot, save_library_stats

def run_scan_task(cfg: Dict[str, Any], db: StagingDB, progress_callback: Optional[Callable] = None):
    """Runs the full scan, group, enrich, and stage process."""
    try:
        if progress_callback:
            progress_callback(0, 1, "Loading alias map...")
        
        alias_map = {}
        alias_map_path = cfg.get("alias_map")
        if alias_map_path:
            alias_file = Path(alias_map_path)
            if alias_file.exists():
                alias_map = json.loads(alias_file.read_text())

        if progress_callback:
            progress_callback(0, 1, "Scanning files...")

        raw_files = scan_files(cfg, alias_map, progress_callback)
        
        if progress_callback:
            progress_callback(0, 1, f"Grouping {len(raw_files)} files...")
        
        groups = group_books(raw_files, cfg)

        if progress_callback:
            progress_callback(0, 1, f"Enriching {len(groups)} books...")

        enrich_books(groups, cfg, db, progress_callback)

        if progress_callback:
            progress_callback(0, 1, "Staging books...")
        
        # Stage each book group in the database
        for i, group in enumerate(groups):
            if progress_callback:
                progress_callback(i, len(groups), f"Staging book {i+1}/{len(groups)}...")
            db.stage_book(group, cfg)
        
        # Commit all changes
        db.conn.commit()
        
        if progress_callback:
            progress_callback(0, 1, "Saving library statistics...")

        # Save library stats for smart scanning
        stats_path = Path(cfg["staging_db_path"]).parent / "library_stats.json"
        source_roots = []
        if cfg.get("source_audiobook_root"):
            source_roots.append(cfg["source_audiobook_root"])
        if cfg.get("source_ebook_root"):
            source_roots.append(cfg["source_ebook_root"])
        snapshot = get_library_snapshot(source_roots)
        save_library_stats(stats_path, snapshot)

        if progress_callback:
            progress_callback(1, 1, "Scan complete.")

        return {"status": "success", "message": f"Scan complete. Found {len(raw_files)} files and created {len(groups)} groups."}
    except Exception as e:
        import traceback
        return {"status": "error", "message": f"An error occurred during scan: {e}\n{traceback.format_exc()}"}

def run_enrich_task(cfg: Dict[str, Any], db: StagingDB, progress_callback: Optional[Callable] = None):
    """
    Runs the enrichment process on existing book groups in the database.
    
    Args:
        cfg: Application configuration dictionary
        db: StagingDB instance
        progress_callback: Optional callback for progress updates (current, total, message)
    
    Returns:
        dict: Status and message about the enrichment process
    """
    try:
        if progress_callback:
            progress_callback(0, 1, "Loading books from database...")
        
        # Get books that need enrichment (low confidence or missing enrichment data)
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT b.book_id, b.title, b.author, b.confidence_details
            FROM books b
            JOIN files f ON b.book_id = f.book_id
            WHERE b.enrich_confidence < 0.8  # Only enrich books with low confidence
            OR b.enrich_confidence IS NULL
            LIMIT 100  # Process in batches to avoid memory issues
        """)
        
        books_to_enrich = cursor.fetchall()
        
        if not books_to_enrich:
            msg = "No books need enrichment (all books have high confidence or are already enriched)."
            if progress_callback:
                progress_callback(1, 1, msg)
            return {"status": "success", "message": msg}
        
        # Convert to BookGroup objects expected by enrich_books
        from dataclasses import dataclass
        from typing import List
        
        @dataclass
        class BookGroup:
            book_id: str
            title: str
            author: str
            canonical: dict
            files: List[dict]
            group_confidence: float = 0.0
            enrich_confidence: float = 0.0
            chaptered: bool = False
            discrepancy: str = ""
        
        groups = []
        for i, book in enumerate(books_to_enrich):
            if progress_callback:
                progress_callback(i, len(books_to_enrich), f"Preparing book {i+1}/{len(books_to_enrich)} for enrichment...")
            
            # Get files for this book
            cursor.execute("""
                SELECT file_id, src, dst, title, authors, narrators, series, tags, 
                       chaptered, processed, enrich_confidence, confidence_details, discrepancy
                FROM files 
                WHERE book_id = ?
            """, (book['book_id'],))
            
            files = [dict(row) for row in cursor.fetchall()]
            
            # Parse confidence details
            try:
                canonical = json.loads(book['confidence_details']) if book['confidence_details'] else {}
            except json.JSONDecodeError:
                canonical = {}
            
            # Create book group
            group = BookGroup(
                book_id=book['book_id'],
                title=book['title'],
                author=book['author'],
                canonical=canonical,
                files=files
            )
            groups.append(group)
        
        # Process enrichment
        if progress_callback:
            progress_callback(0, len(groups), "Enriching book metadata...")
        
        # Call the enrichment function
        enrich_books(groups, cfg, db, progress_callback)
        
        # Update the database with enriched data
        with db.conn:
            for i, group in enumerate(groups):
                if progress_callback:
                    progress_callback(i, len(groups), f"Saving enrichment for book {i+1}/{len(groups)}...")
                
                # Update the book record with enriched data
                db.conn.execute("""
                    UPDATE books 
                    SET title = ?,
                        author = ?,
                        series = ?,
                        series_index = ?,
                        genre = ?,
                        year = ?,
                        enrich_confidence = ?,
                        confidence_details = ?
                    WHERE book_id = ?
                """, (
                    group.title,
                    group.author,
                    group.canonical.get('series'),
                    group.canonical.get('series_index'),
                    group.canonical.get('genre'),
                    group.canonical.get('year'),
                    group.enrich_confidence,
                    json.dumps(group.canonical),
                    group.book_id
                ))
                
                # Update files with new metadata
                for file in group.files:
                    db.conn.execute("""
                        UPDATE files 
                        SET title = ?,
                            authors = ?,
                            narrators = ?,
                            series = ?,
                            tags = ?,
                            chaptered = ?,
                            enrich_confidence = ?,
                            confidence_details = ?,
                            discrepancy = ?
                        WHERE file_id = ?
                    """, (
                        file.get('title'),
                        json.dumps(file.get('authors', [])),
                        json.dumps(file.get('narrators', [])),
                        file.get('series'),
                        json.dumps(file.get('tags', {})),
                        file.get('chaptered', False),
                        file.get('enrich_confidence', 0.0),
                        json.dumps(file.get('confidence_details', {})),
                        file.get('discrepancy', ''),
                        file['file_id']
                    ))
        
        msg = f"Successfully enriched {len(groups)} books."
        if progress_callback:
            progress_callback(len(groups), len(groups), msg)
            
        return {"status": "success", "message": msg}
        
    except Exception as e:
        import traceback
        error_msg = f"An error occurred during enrichment: {e}\n{traceback.format_exc()}"
        if progress_callback:
            progress_callback(0, 1, f"Error: {str(e)}")
        return {"status": "error", "message": error_msg}

def run_convert_task(cfg: Dict[str, Any], db: StagingDB, dryrun: bool, progress_callback: Optional[Callable] = None):
    """Runs the file conversion (move) process."""
    try:
        batch_size = cfg.get('batch_size', 100)
        result = perform_moves(cfg, db, batch_size, dryrun, progress_callback)
        return {"status": "success", "message": f"Conversion complete. Moved {result.get('moved', 0)} files."}
    except Exception as e:
        import traceback
        return {"status": "error", "message": f"An error occurred during conversion: {e}\n{traceback.format_exc()}"}

def run_undo_task(db: StagingDB, progress_callback: Optional[Callable] = None):
    """Runs the undo process for the last batch of conversions."""
    try:
        if progress_callback:
            progress_callback(0, 1, "Undoing last operation...")
        
        result = db.undo_last()

        if progress_callback:
            progress_callback(1, 1, "Undo complete.")

        return {"status": "success", "message": f"Undo successful. Reverted {result.get('reverted_count', 0)} files."}
    except Exception as e:
        import traceback
        return {"status": "error", "message": f"An error occurred during undo: {e}\n{traceback.format_exc()}"}
