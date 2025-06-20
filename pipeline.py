"""
pipeline.py
Version: 2.3.0
Last Update: 2025-06-20
Description:
    Entry-point and Command-Line Interface (CLI) for the Audiobook Reorganization
    tool. This script parses command-line arguments and orchestrates the high-level
    workflow by calling functions from other modules.

Directory:
- main:line 75 - Main entry point, parses args and config, and runs commands.
- parse_args:line 55 - Defines and parses command-line arguments.
- validate_config:line N/A - (Moved to utils.py)
"""

import argparse
import json
import sys
from pathlib import Path
from utils import load_and_prepare_config
from library_core import scan_files, group_books, enrich_books, perform_moves
from staging_db import StagingDB

def parse_args():
    """Define and parse command-line arguments for the application."""
    parser = argparse.ArgumentParser(description="Audiobook/Ebook Reorg v2.2.2b")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    scan_p = subparsers.add_parser("scan", help="Scan library, enrich, and stage a reorganization plan.")
    scan_p.add_argument("--dryrun", action="store_true", help="Scan only, skip enrichment and staging.")

    subparsers.add_parser("gui", help="Launch the graphical user interface.")

    convert_p = subparsers.add_parser("convert", help="Execute the staged file moves.")
    convert_p.add_argument("--batch", type=int, help="Override default batch size for this run.")
    convert_p.add_argument("--dryrun", action="store_true", help="Simulate moves without changing files.")

    undo_p = subparsers.add_parser("undo", help="Undo the most recent file operations.")
    undo_p.add_argument("--steps", type=int, default=1, help="Number of operations to undo.")

    return parser.parse_args()

def main():
    """Main entry point for the Audiobook Reorg application."""
    cfg = load_and_prepare_config()

    # Load alias map
    alias_map = {}
    alias_map_path = cfg.get("alias_map")
    if alias_map_path:
        alias_file = Path(alias_map_path)
        if alias_file.exists():
            try:
                alias_map = json.loads(alias_file.read_text())
            except json.JSONDecodeError:
                print(f"Warning: Could not parse alias map file: {alias_map_path}", file=sys.stderr)
        else:
            print(f"Warning: Alias map file not found: {alias_map_path}", file=sys.stderr)

    db = StagingDB(cfg['staging_db_path'])
    args = parse_args()

    if args.command == "scan":
        print("Scanning files...")
        raw_files = scan_files(cfg, alias_map)
        print(f"Found {len(raw_files)} audio files.")
        if not args.dryrun:
            groups = group_books(raw_files, cfg)
            enrich_books(groups, cfg, db)
            print("Plan staged successfully.")
        else:
            print("Dry run: Scan complete. No changes staged.")

    elif args.command == "gui":
        from ui_frontend import launch_gui
        print("Launching GUI...")
        launch_gui()

    elif args.command == "convert":
        batch_size = args.batch or cfg.get('batch_size', 100)
        # Use dryrun from args if present, otherwise from config
        dryrun = args.dryrun if args.dryrun else cfg.get('dryrun_default', True)
        perform_moves(cfg, db, batch_size, dryrun)

    elif args.command == "undo":
        print(f"Undoing last operation...")
        try:
            result = db.undo_last()
            if 'error' in result:
                print(f"Error during undo: {result['error']}", file=sys.stderr)
                sys.exit(1)
            
            reverted_count = result.get('reverted_count', 0)
            if reverted_count > 0:
                print(f"Successfully reverted {reverted_count} file(s).")
            else:
                print("No operations to undo.")
                
        except Exception as e:
            print(f"Error during undo operation: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
