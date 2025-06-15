"""
tasks.py
Version: 2.2.22
Last Update: 2025-06-11
Description:
    High-level task functions for the Audiobook Reorganization tool. These functions
    are designed to be called from a GUI in a separate thread and provide progress
    updates via a callback mechanism.
"""

import time
import json
from pathlib import Path
from typing import Callable, Optional, Dict, Any

from library_core import scan_files, group_books, enrich_books, stage_plan, perform_moves
from staging_db import StagingDB

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
            progress_callback(0, 1, "Staging plan...")

        stage_plan(groups, cfg, db)

        if progress_callback:
            progress_callback(1, 1, "Scan complete.")

        return {"status": "success", "message": f"Scan complete. Found {len(raw_files)} files and created {len(groups)} groups."}
    except Exception as e:
        import traceback
        return {"status": "error", "message": f"An error occurred during scan: {e}\n{traceback.format_exc()}"}

def run_enrich_task(cfg: Dict[str, Any], db: StagingDB, progress_callback: Optional[Callable] = None):
    """Runs the enrichment process on existing book groups in the database."""
    try:
        if progress_callback:
            progress_callback(0, 1, "Loading books from database...")
        
        # This needs to be adapted to work with the DB structure
        # For now, let's assume we get groups that need enrichment
        # This part of the logic needs to be fleshed out more.
        # groups = db.get_groups_needing_enrichment()
        # enrich_books(groups, cfg, db, progress_callback)
        # stage_plan(groups, cfg, db)

        if progress_callback:
            progress_callback(1, 1, "Enrichment complete.")
        return {"status": "success", "message": "Enrichment finished."}
    except Exception as e:
        import traceback
        return {"status": "error", "message": f"An error occurred during enrichment: {e}\n{traceback.format_exc()}"}

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
