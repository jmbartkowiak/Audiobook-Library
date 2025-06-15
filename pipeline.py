"""
pipeline.py
Version: 2.2.2
Last Update: 2025-06-10
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
import yaml
from pathlib import Path

# Core logic is now in library_core
from library_core import scan_files, group_books, enrich_books, stage_plan, perform_moves
from staging_db import StagingDB
from utils import validate_config

def parse_args():
    """Define and parse command-line arguments for the application."""
    parser = argparse.ArgumentParser(description="Audiobook/Ebook Reorg v2.2.0")
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
    cfg_path = Path("settings.yaml")
    if not cfg_path.exists():
        print("Error: settings.yaml not found.", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text())
    validate_config(cfg)

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
            enrich_books(groups, cfg)
            stage_plan(groups, cfg, db)
            print("Plan staged successfully.")
        else:
            print("Dry run: Scan complete. No changes staged.")

    elif args.command == "gui":
        from ui_frontend import launch_gui
        print("Launching GUI...")
        launch_gui(cfg, db)

    elif args.command == "convert":
        batch_size = args.batch or cfg.get('batch_size', 100)
        # Use dryrun from args if present, otherwise from config
        dryrun = args.dryrun if args.dryrun else cfg.get('dryrun_default', True)
        perform_moves(cfg, db, batch_size, dryrun)

    elif args.command == "undo":
        print(f"Undo command is not yet implemented.")

if __name__ == "__main__":
    main()
