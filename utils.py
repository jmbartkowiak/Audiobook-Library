from __future__ import annotations

"""
utils.py
Last update: 2025-06-11
Version: 2.2.2
Description:
    A collection of shared utility functions used across the Audiobook/E-book Reorg application.
    This module helps to avoid circular dependencies and reduce code duplication by providing
    a central location for common logic, such as configuration validation and path calculations.

    Directory:
    - validate_config:line 18 - Validates the structure and values in settings.yaml.
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import re

def validate_config(cfg: dict) -> None:
    """
    Validate configuration settings with enhanced error messages.
    Ensures the nested confidence structure and all required keys are present and valid.
    """
    # Validate nested confidence structure
    if "confidence" not in cfg or not isinstance(cfg.get("confidence"), dict):
        raise ValueError("Missing 'confidence' section in config.")
    
    conf = cfg["confidence"]
    if "enrichment" not in conf or not isinstance(conf.get("enrichment"), dict):
        raise ValueError("Missing 'confidence.enrichment' section in config.")
    
    enrich_conf = conf["enrichment"]
    if "min_confidence" not in enrich_conf or not isinstance(enrich_conf.get("min_confidence"), float):
        raise ValueError("Missing or invalid 'confidence.enrichment.min_confidence' (should be a float).")
    if "llm_threshold" not in enrich_conf or not isinstance(enrich_conf.get("llm_threshold"), float):
        raise ValueError("Missing or invalid 'confidence.enrichment.llm_threshold' (should be a float).")

    if "grouping" not in conf or not isinstance(conf.get("grouping"), dict):
        raise ValueError("Missing 'confidence.grouping' section in config.")

    group_conf = conf["grouping"]
    if "min_confidence" not in group_conf or not isinstance(group_conf.get("min_confidence"), float):
        raise ValueError("Missing or invalid 'confidence.grouping.min_confidence' (should be a float).")
    required_keys = {
        "library_root": str,
        "ebook_root": str,
        "staging_db_path": str,
        "batch_size": int,
        "dryrun_default": bool,
        "offline_mode": bool,
        "backup_zip": bool,
        "backup_retention_days": int,
        "alias_map": str,
        "author_blacklist": list,
        "author_max_length": int,
        "year_pattern": str,
        "llm_provider": dict,
        "llm_batch_size": int,
        "self_hosted_llm": bool
    }
    for key, expected_type in required_keys.items():
        if key not in cfg:
            raise ValueError(f"Missing required config key: '{key}'. Please ensure it is defined in the settings.yaml file.")
        if not isinstance(cfg[key], expected_type):
            raise TypeError(f"Config key '{key}' should be of type {expected_type.__name__}. Found type {type(cfg[key]).__name__}.")
    # Validate llm_provider sub-keys
    llm_provider_keys = {"model": str, "api_key_env": str, "url": str, "extra_lazy": bool}
    for key, expected_type in llm_provider_keys.items():
        if key not in cfg["llm_provider"]:
            raise ValueError(f"Missing required llm_provider config key: '{key}'. Please ensure it is defined under llm_provider in the settings.yaml file.")
        if not isinstance(cfg["llm_provider"][key], expected_type):
            raise TypeError(f"Config key 'llm_provider.{key}' should be of type {expected_type.__name__}. Found type {type(cfg['llm_provider'][key]).__name__}.")


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
    """Calculates the destination path for a file based on templates in the config."""
    library_root = Path(cfg['paths']['library_root'])
    
    # Get templates from config
    folder_template = cfg['formatting']['folder_template']
    filename_template = cfg['formatting']['filename_template']

    # Prepare template data from book group, handling missing values
    template_data = {
        'author': grp.canonical.get('authors', ["Unknown Author"])[0],
        'series': grp.canonical.get('series'),
        'title': grp.canonical.get('title', "Unknown Title"),
        'series_index': grp.canonical.get('series_index'),
        'ext': raw.path.suffix
    }
    
    # Sanitize path components, but keep original values for formatting
    format_data = {k: sanitize_path_component(str(v)) if v is not None else '' for k, v in template_data.items()}
    format_data['ext'] = template_data['ext'] # ext should not be sanitized

    # Handle special formatting for series_index
    if template_data['series_index'] is not None:
        # If series_index exists, use the original number for formatting, not the sanitized string.
        format_data['series_index'] = int(template_data['series_index'])
    else:
        # If index is None, remove the placeholder and any preceding separator
        filename_template = re.sub(r'\{series_index:[^}]+\}\s*-\s*', '', filename_template)
        filename_template = filename_template.replace('{series_index}', '')

    # Remove empty parts from the folder path to avoid double slashes
    folder_path_parts = [p for p in folder_template.format(**format_data).split('/') if p]
    folder_path = Path('/'.join(folder_path_parts))

    filename = filename_template.format(**format_data).strip()
    
    # Clean up filename if series index was missing
    if template_data['series_index'] is None:
        filename = filename.lstrip(" -")

    final_path = library_root / folder_path / filename
    return resolve_collision(final_path)

def resolve_collision(dst: Path) -> Path:
    base, ext = dst.with_suffix('').as_posix(), dst.suffix
    i=1
    new=dst
    while new.exists():
        new=Path(f"{base}_{i}{ext}")
    return new

