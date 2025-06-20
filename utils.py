"""
utils.py
Last update: 2025-06-20
Version: 2.3.0
Description:
    A collection of shared utility functions used across the Audiobook/E-book Reorg application.
    This module helps to avoid circular dependencies and reduce code duplication by providing
    a central location for common logic, such as configuration validation and path calculations.

    Directory:
    - validate_config:line 91 - Validates settings.yaml, including the new OpenLibrary schema.
"""

from __future__ import annotations
import yaml
import sys
import re
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time
from zipfile import ZipFile, ZIP_DEFLATED
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_file: str = "reorg.log", level: int = logging.INFO) -> None:
    """Set up logging with both console and file handlers."""
    # Create log directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)
    
    # Create file handler with rotation
    fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    fh.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    # Set up logging for specific modules
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized. Log file: {log_file}")


def load_and_prepare_config():
    """
    Loads settings.yaml, defines project root, and resolves key paths.
    
    Supports both nested 'paths:' structure and legacy flat structure for backward compatibility.
    When both exist, the nested paths take precedence over the flat structure.
    """
    try:
        # Define project root as the directory containing the script's parent folder.
        project_root = Path(__file__).parent.resolve()
        cfg_path = project_root / "settings.yaml"

        if not cfg_path.exists():
            print(f"Error: settings.yaml not found at {cfg_path}", file=sys.stderr)
            sys.exit(1)

        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
            
        # Initialize paths dictionary if it doesn't exist
        if 'paths' not in cfg:
            cfg['paths'] = {}
            
        # List of path keys that should be moved under 'paths' if found at root level
        path_keys = [
            'source_audiobook_root', 'destination_audiobook_root',
            'source_ebook_root', 'destination_ebook_root',
            'staging_db_path', 'alias_map'
        ]
        
        # Migrate any root-level path keys to the paths dictionary if they don't exist there
        for key in path_keys:
            if key in cfg and key not in cfg['paths']:
                cfg['paths'][key] = cfg[key]
                if key not in ['alias_map']:  # Don't warn for alias_map as it's not strictly a path
                    print(f"Warning: Found legacy path key '{key}' at root level. "
                          f"Please move it under the 'paths:' section in settings.yaml", 
                          file=sys.stderr)
        
        # Resolve all paths to be absolute, relative to the project root
        for key, value in cfg['paths'].items():
            if value and isinstance(value, str):
                if not Path(value).is_absolute():
                    cfg['paths'][key] = str(project_root / value)
        
        # Handle OpenLibrary paths - ensure both mirror_db and sqlite_path are absolute and consistent
        if 'openlibrary' in cfg:
            ol_cfg = cfg['openlibrary']
            
            # If both mirror_db and sqlite_path exist, prefer mirror_db and sync them
            if 'mirror_db' in ol_cfg and ol_cfg['mirror_db']:
                if not Path(ol_cfg['mirror_db']).is_absolute():
                    ol_cfg['mirror_db'] = str(project_root / ol_cfg['mirror_db'])
                # Ensure sqlite_path matches mirror_db if not explicitly set
                if 'sqlite_path' not in ol_cfg or not ol_cfg['sqlite_path']:
                    ol_cfg['sqlite_path'] = ol_cfg['mirror_db']
            
            # If only sqlite_path exists, ensure it's absolute and copy to mirror_db
            elif 'sqlite_path' in ol_cfg and ol_cfg['sqlite_path']:
                if not Path(ol_cfg['sqlite_path']).is_absolute():
                    ol_cfg['sqlite_path'] = str(project_root / ol_cfg['sqlite_path'])
                ol_cfg['mirror_db'] = ol_cfg['sqlite_path']
            
            # Ensure the parent directory exists
            if 'mirror_db' in ol_cfg and ol_cfg['mirror_db']:
                Path(ol_cfg['mirror_db']).parent.mkdir(parents=True, exist_ok=True)

        # For backward compatibility, copy paths back to root level if they don't exist there
        # This ensures existing code that accesses paths directly from cfg still works
        for key in path_keys:
            if key in cfg['paths'] and key not in cfg:
                cfg[key] = cfg['paths'][key]

        # Validate the loaded config
        validate_config(cfg)
        return cfg

    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        print(f"Error loading or validating configuration: {e}", file=sys.stderr)
        raise
        sys.exit(1)

def validate_config(cfg: dict) -> None:
    """
    Validate configuration settings with enhanced error messages.
    Ensures all required sections and keys are present and valid.
    Supports both nested 'paths' structure and legacy flat structure.
    """
    # Ensure paths section exists (it's created in load_and_prepare_config if missing)
    if 'paths' not in cfg:
        cfg['paths'] = {}
    
    # Validate nested confidence structure
    if "confidence" not in cfg or not isinstance(cfg.get("confidence"), dict):
        raise ValueError("Missing 'confidence' section in config.")
    
    conf = cfg["confidence"]
    if "enrichment" not in conf or not isinstance(conf.get("enrichment"), dict):
        raise ValueError("Missing 'confidence.enrichment' section in config.")
    
    enrich_conf = conf["enrichment"]
    if "min_confidence" not in enrich_conf or not isinstance(enrich_conf.get("min_confidence"), (int, float)):
        raise ValueError("Missing or invalid 'confidence.enrichment.min_confidence' (should be a number).")
    if "llm_threshold" not in enrich_conf or not isinstance(enrich_conf.get("llm_threshold"), (int, float)):
        raise ValueError("Missing or invalid 'confidence.enrichment.llm_threshold' (should be a number).")

    if "grouping" not in conf or not isinstance(conf.get("grouping"), dict):
        raise ValueError("Missing 'confidence.grouping' section in config.")

    group_conf = conf["grouping"]
    if "min_confidence" not in group_conf or not isinstance(group_conf.get("min_confidence"), (int, float)):
        raise ValueError("Missing or invalid 'confidence.grouping.min_confidence' (should be a number).")
    
    # Define required configuration keys and their types
    required_keys = {
        "batch_size": int,
        "dryrun_default": bool,
        "offline_mode": bool,
        "backup_zip": bool,
        "backup_retention_days": int,
        "author_blacklist": list,
        "author_max_length": int,
        "year_pattern": str,
        "llm_provider": dict,
        "llm_batch_size": int,
        "self_hosted_llm": bool,
        "openlibrary": dict,
        "formatting": dict
    }
    
    # Define required paths and their types (can be in either root or under 'paths')
    required_paths = {
        "source_audiobook_root": str,
        "destination_audiobook_root": str,
        "source_ebook_root": str,
        "destination_ebook_root": str,
        "staging_db_path": str,
        "alias_map": str
    }
    # Validate required configuration keys
    for key, expected_type in required_keys.items():
        if key not in cfg:
            raise ValueError(
                f"Missing required config key: '{key}'. "
                f"Please ensure it is defined in the settings.yaml file."
            )
        if not isinstance(cfg[key], expected_type):
            raise TypeError(
                f"Config key '{key}' should be of type {expected_type.__name__}. "
                f"Found type {type(cfg[key]).__name__}."
            )
    
    # Validate required paths (can be in either root or under 'paths')
    missing_paths = []
    for path_key, expected_type in required_paths.items():
        # Check if path exists in either location
        path_value = cfg['paths'].get(path_key) if path_key in cfg['paths'] else cfg.get(path_key)
        
        if path_value is None:
            missing_paths.append(path_key)
        elif not isinstance(path_value, expected_type):
            location = 'paths' if path_key in cfg['paths'] else 'root level'
            raise TypeError(
                f"Config key '{path_key}' in {location} should be of type {expected_type.__name__}. "
                f"Found type {type(path_value).__name__}."
            )
    
    if missing_paths:
        raise ValueError(
            f"Missing required path configuration(s). Please ensure these are defined in the 'paths:' section "
            f"of settings.yaml: {', '.join(missing_paths)}"
        )
    
    # Validate llm_provider sub-keys
    llm_provider_keys = {
        "model": str, 
        "api_key_env": str, 
        "url": str, 
        "extra_lazy": bool
    }
    
    if "llm_provider" not in cfg:
        raise ValueError("Missing required 'llm_provider' section in config.")
        
    for key, expected_type in llm_provider_keys.items():
        if key not in cfg["llm_provider"]:
            raise ValueError(
                f"Missing required llm_provider config key: '{key}'. "
                f"Please ensure it is defined under llm_provider in settings.yaml."
            )
        if not isinstance(cfg["llm_provider"][key], expected_type):
            raise TypeError(
                f"Config key 'llm_provider.{key}' should be of type {expected_type.__name__}. "
                f"Found type {type(cfg['llm_provider'][key]).__name__}."
            )

    # Validate openlibrary configuration
    if "openlibrary" not in cfg:
        raise ValueError("Missing required 'openlibrary' section in config.")
        
    ol_cfg = cfg["openlibrary"]
    
    # Required OpenLibrary keys
    ol_required_keys = {
        "use_local_mirror": bool,
        "dump_url": str,
        "sqlite_path": str,
        "mirror_db": str,
        "cache_ttl_days": int,
        "min_update_days": int,
        "prefer_pigz": bool
    }
    
    # Check for required keys
    for key, expected_type in ol_required_keys.items():
        if key not in ol_cfg:
            # sqlite_path and mirror_db are required but might be auto-populated
            if key in ["sqlite_path", "mirror_db"] and ("sqlite_path" in ol_cfg or "mirror_db" in ol_cfg):
                continue
            raise ValueError(f"Missing required openlibrary config key: '{key}'.")
        if not isinstance(ol_cfg[key], expected_type):
            raise TypeError(
                f"Config key 'openlibrary.{key}' should be of type {expected_type.__name__}. "
                f"Found type {type(ol_cfg[key]).__name__}."
            )
    
    # Ensure dump_url is one of the expected values
    expected_urls = [
        "https://openlibrary.org/data/ol_dump_latest.txt.gz",
        "https://openlibrary.org/data/ol_dump_all_latest.txt.gz"
    ]
    if ol_cfg.get("dump_url") not in expected_urls:
        print(
            f"Warning: openlibrary.dump_url should be one of {expected_urls}. "
            f
                  "Using mirror_db for lookups and sqlite_path for updates.", file=sys.stderr)

@dataclass
class RawMeta:
    path: Path
    tags: Dict[str, Optional[any]]
    size: int
    mtime: float
    source: str

@dataclass
class BookGroup:
    authors: List[str]
    title: str
    files: List[RawMeta] = field(default_factory=list)
    canonical: Dict[str, any] = field(default_factory=dict)
    discrepancy: str = "Minor"
    diff_html: str = ""
    warning_html: str = ""
    group_confidence: float = 0.0
    enrich_confidence: float = 0.0
    chaptered: bool = False

    @property
    def confidence(self) -> float:
        """
        Deprecated: use group_confidence or enrich_confidence explicitly.
        Returns the grouping confidence for backward compatibility.
        """
        return self.group_confidence

def sanitize_path_component(name: str) -> str:
    """Sanitizes a string to be a valid path component."""
    if not name:
        return ""
    # Replace invalid characters with an underscore
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def calculate_destination_path(grp: BookGroup, raw: RawMeta, cfg: Dict[str, any]) -> Path:
    """
    Calculates the destination path for a file based on templates in the config.
    
    Args:
        grp: BookGroup containing canonical metadata
        raw: RawMeta with file information
        cfg: Configuration dictionary
        
    Returns:
        Path: The calculated destination path
        
    Raises:
        ValueError: If the file extension is not in any known format and no fallback is available
    """
    # Determine if it's an audiobook or ebook to select the correct destination root.
    file_ext = raw.path.suffix.lower()
    
    # Get format lists from config
    audio_exts = cfg.get('audio_formats', ['.mp3', '.m4a', '.m4b', '.flac'])
    ebook_exts = cfg.get('ebook_formats', ['.epub', '.mobi', '.pdf'])
    
    # Check if paths are nested under 'paths' key or at the root
    paths_cfg = cfg.get('paths', cfg)  # Handle both nested and flat config
    
    # Determine the appropriate library root based on file extension
    if file_ext in audio_exts:
        library_root = Path(paths_cfg['destination_audiobook_root'])
    elif file_ext in ebook_exts:
        library_root = Path(paths_cfg['destination_ebook_root'])
    else:
        # For unknown extensions, log a warning and raise an error
        error_msg = (
            f"Cannot determine destination for file with unknown extension '{file_ext}': {raw.path}\n"
            f"Known audio formats: {', '.join(audio_exts)}\n"
            f"Known ebook formats: {', '.join(ebook_exts)}"
        )
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Get templates from config
    formatting_cfg = cfg.get('formatting', {})
    
    # Select folder template, with a fallback for entries with no year
    if not grp.canonical.get('year') and 'folder_template_no_year' in formatting_cfg:
        folder_template = formatting_cfg['folder_template_no_year']
    else:
        folder_template = formatting_cfg.get('folder_template', "{author}/{title} ({year})")

    # Select filename template, with a series-specific fallback
    if grp.canonical.get('series') and 'filename_template_series' in formatting_cfg:
        filename_template = formatting_cfg['filename_template_series']
    else:
        filename_template = formatting_cfg.get('filename_template', "{filename}")

    # If the book is not part of a series, remove any lingering series placeholders
    if not grp.canonical.get('series'):
        filename_template = re.sub(r'({series}|{series_index(:[^}]+)?})\s*[-()]?\s*', '', filename_template, flags=re.IGNORECASE).strip()
        filename_template = filename_template.replace('()', '').replace('[]', '')

    # Prepare template data, adding original filename and stem for more flexible templates
    template_data = {
        'authors': ", ".join(grp.canonical.get('authors', ["Unknown Author"])),
        'author': (grp.canonical.get('authors') or ["Unknown Author"])[0],
        'title': grp.canonical.get('title', "Unknown Title"),
        'year': grp.canonical.get('year'),
        'series': grp.canonical.get('series'),
        'series_index': grp.canonical.get('series_index'),
        'ext': raw.path.suffix,
        'filename': raw.path.name,
        'stem': raw.path.stem
    }
    
    # Sanitize all components except the extension
    format_data = {k: sanitize_path_component(str(v)) if v is not None else '' for k, v in template_data.items()}
    format_data['ext'] = template_data['ext']

    # Handle special numeric formatting for year and series_index
    if template_data['year']:
        format_data['year'] = int(template_data['year'])
    if template_data['series_index']:
        format_data['series_index'] = int(template_data['series_index'])

    # Format the folder and filename
    folder_str = folder_template.format(**format_data)
    filename_str = filename_template.format(**format_data).strip()

    # Clean up the path string to remove empty parts and invalid characters
    folder_parts = [part for part in folder_str.split('/') if part]
    clean_folder_path = Path(*folder_parts)
    
    final_path = library_root / clean_folder_path / filename_str
    return resolve_collision(final_path)


def resolve_collision(dst: Path) -> Path:
    """Incrementally resolves filename collisions by appending '_1', '_2', etc."""
    if not dst.exists():
        return dst

    i = 1
    new_dst = dst.with_name(f"{dst.stem}_{i}{dst.suffix}")
    while new_dst.exists():
        i += 1
        new_dst = dst.with_name(f"{dst.stem}_{i}{dst.suffix}")
    return new_dst


def get_library_snapshot(library_roots: List[str]) -> Dict[str, Any]:
    """Calculate the current state of the library (file count, total size) across multiple roots."""
    total_files = 0
    total_size = 0
    for library_root in library_roots:
        if not Path(library_root).exists():
            logging.warning(f"Library snapshot path does not exist, skipping: {library_root}")
            continue
        for root, _, files in os.walk(library_root):
            total_files += len(files)
            for f in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass  # Ignore files that can't be accessed
    return {"total_files": total_files, "total_size": total_size, "last_scan_time": time.time()}


def load_library_stats(stats_path: Path) -> Optional[Dict[str, Any]]:
    """Load the last saved library statistics from a JSON file."""
    if not stats_path.exists():
        return None
    try:
        return json.loads(stats_path.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def save_library_stats(stats_path: Path, snapshot: Dict[str, Any]):
    """Save the current library statistics to a JSON file."""
    try:
        stats_path.write_text(json.dumps(snapshot, indent=4))
    except IOError:
        # Handle cases where the file can't be written
        pass

class BackupManager:
    def __init__(self, cfg):
        project_root = Path(__file__).parent.resolve()
        bdir = project_root / ".reorg_backup"
        bdir.mkdir(exist_ok=True)
        today = datetime.date.today().isoformat()
        self.zf = ZipFile(bdir / f"{today}.zip", "a", ZIP_DEFLATED)
        self.retention = cfg["backup_retention_days"]
        self._purge_old(bdir)

    def _purge_old(self, bdir: Path):
        cutoff = datetime.date.today() - datetime.timedelta(days=self.retention)
        for f in bdir.iterdir():
            if not f.is_file() or f.suffix != '.zip':
                continue
            try:
                d = datetime.date.fromisoformat(f.stem)
                if d < cutoff:
                    f.unlink()
            except ValueError:
                pass  # Ignore files with non-date names

    def backup_file(self, src: Path, cfg: dict):
        """Create a backup of the source file, storing it with a path relative to its source root."""
        source_audio_root = Path(cfg["source_audiobook_root"])
        source_ebook_root = Path(cfg["source_ebook_root"])
        
        arc_path = None
        try:
            # Path.relative_to raises ValueError if not a subpath
            arc_path = src.relative_to(source_audio_root)
        except ValueError:
            try:
                arc_path = src.relative_to(source_ebook_root)
            except ValueError:
                logging.error(f"Could not determine relative path for backup of {src}. It is not in a configured source root.")
                return

        if arc_path:
            self.zf.write(src, str(arc_path))
