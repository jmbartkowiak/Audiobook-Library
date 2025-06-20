# Audiobook/E-book Reorganization System v2.3.X

**Last Update:** 2025-06-11  
**Version:** 2.2.0

---

## Overview & Intent

This project is a robust, production-grade system for organizing, cleaning, and managing large-scale audiobook and e-book libraries. It demonstrates advanced data engineering, data cleaning, intelligent grouping, and interactive GUI-driven workflows, all backed by a persistent and auto-migrating SQLite database. The system is designed to handle messy, real-world data at scale, with extensible support for LLM-based enrichment and OpenLibrary metadata integration.

## Key Features

- **Automated Grouping:** Uses fuzzy heuristics and metadata analysis to cluster files into books, even when tags are noisy or missing.
- **Audio Processing:** Supports audio duration detection and format handling for various audio formats.
- **Data Cleaning:** Applies alias mapping, blacklist filtering, and normalization to author/title fields.
- **Persistent Storage:** All operations are tracked in an SQLite database with robust undo and migration support.
- **GUI:** PySimpleGUI interface supports live dark mode, resizable/toggleable columns, context menus, and persistent user preferences.
- **LLM & OpenLibrary Integration:** Enriches metadata using OpenAI or self-hosted LLMs, plus local or remote OpenLibrary lookups.
- **Batch Processing:** Efficiently handles thousands of files with batch conversion, progress tracking, and error handling.
- **Extensible & Modular:** Codebase is organized for easy extension to new media types, enrichment sources, or deployment environments.
- **Limited AI Use:** Gemini 2.5 pro was used exclusively to track Revision History and ensure PEP8 compatibility throughout codebase post v2.2.0

## Project Evolution & Engineering Journey

Originally conceived as a quick-and-dirty helper script for renaming a handful of MP3s, this project has gradually matured into a fully-featured data-engineering pipeline. Along the way it has absorbed best-practice patterns drawn from large-scale data processing:

- **Extensive Data Cleaning** – Normalises titles/authors, deduplicates aliases, and enforces schema constraints before any file moves occur.
- **Algorithmic Grouping & Statistical Scoring** – Uses duration statistics, weighted string metrics, and naming-pattern detection to cluster files with quantifiable confidence.
- **Database-Centric Workflow** – All intermediate state lives in an auto-migrating SQLite store, enabling undo, auditing, and concurrent GUI paging.
- **Incremental Enrichment with LLMs & OpenLibrary** – Falls back to language-model inference only when confidence thresholds warrant it, keeping API usage efficient.
- **Modular, Test-Driven Engineering** – Shared utilities, dependency injection, and a growing suite of unit tests guard against regression as the codebase evolves.

This evolution illustrates how thoughtful refactors, layered abstractions, and incremental automation can transform an ad-hoc script into a production-ready application without sacrificing maintainability or clarity.

## Core Architecture & Data-Flow

The system follows a modular architecture with the following components:

1. **Core Engine** (`library_core.py`): Handles file scanning, metadata extraction, and grouping logic.
2. **GUI** (`ui_frontend.py`): Provides an interactive interface for managing the library.
3. **Database** (`staging_db.py`): Manages persistent storage and versioning.
4. **Metadata & Enrichment** (`openlibrary_mirror.py`, `confidence_scoring.py`): Manages metadata enrichment, including automatic OpenLibrary mirror updates via the `MirrorUpdater` class, and calculates confidence scores for grouping and enrichment decisions.
5. **Pipeline** (`pipeline.py`): Orchestrates the workflow between components.

Data flows from file scanning through grouping, enrichment, and finally to the database, with the GUI providing real-time feedback and control.

---

## Table of Contents

1. [Project Intent & Value Proposition](#overview--intent)
2. [Key Features](#key-features)
3. [Project Evolution & Engineering Journey](#project-evolution--engineering-journey)
4. [Core Architecture & Data-Flow](#core-architecture--data-flow)
5. [Installation & Quick-Start](#installation--setup)
6. [Configuration Reference](#configuration-reference)
7. [Advanced GUI Walk-through](GUI_GUIDE.md)
8. [Technical Details & Heuristics](TECHNICAL_DETAILS.md)
9. [Appendix & Glossary](APPENDIX.md)
10. [Limitations & Future Road-map](#limitations--future-road-map)

---

## Installation & Setup

### Requirements

- **Python 3.11+**
- **Dependencies:** Key dependencies are detailed in `APPENDIX.md`. For a complete, installable list, see `requirements.txt`.
- **FFmpeg:** For all audio processing features, FFmpeg must be installed and available in your system's PATH. See `APPENDIX.md` for installation instructions.

### Quick Start

1. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

   **Important Note on PySimpleGUI:** If you encounter issues with the GUI, particularly `AttributeError`, you may need to install `PySimpleGUI` from its private repository. Run these commands:

   ```sh
   pip uninstall PySimpleGUI
   pip cache purge
   pip install --upgrade --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
   ```


   Note: Some audio processing dependencies might require additional system libraries. If you encounter installation issues, please refer to the documentation for the respective libraries.

2. **Copy and edit `settings.yaml.example` to `settings.yaml`:**
   - Set `library_root` to your audiobook directory
   - Set LLM provider details if using enrichment

3. **Run the GUI:**

   ```sh
   python pipeline.py gui
   ```

4. **(Optional) Run CLI commands:**
   - `python pipeline.py scan` to stage a plan
   - `python pipeline.py convert` to apply moves
   - `python pipeline.py undo` to revert

---

## Configuration Reference

All configuration is in `settings.yaml`. Key fields:

- `library_root`: Path to your audiobook/e-book library (required)
- `batch_size`: Number of files to process per batch
- `dryrun_default`: If true, simulate actions by default
- `backup_retention_days`: How long to keep backup zips
- `alias_map`: Path to JSON alias map for author normalization
- `author_blacklist`: List of words to ignore in author fields
- `author_max_length`: Max length for valid author names
- `year_pattern`: Regex for filtering numeric/invalid authors
- `llm_provider`: Dict with `model`, `api_key_env`, `url`, etc.
- `gui`: Dict with `columns`, `dark_mode`, `page_size`, `book_view`
- `openlibrary`: Dict with `use_local_mirror`, `mirror_db`, `cache_ttl_days`

See `settings.yaml.example` for a template and detailed comments.

---

## Audio Processing

The application includes comprehensive audio processing capabilities:

### Supported Audio Formats

- MP3
- M4B/M4A (AAC)
- FLAC
- WAV
- OGG
- AIFF

### Audio Features

- **Duration Detection**: Automatically calculates and displays the duration of audio files
- **Metadata Extraction**: Reads ID3 and other standard audio metadata tags
- **Format Validation**: Verifies file integrity and format compatibility
- **Performance Optimization**: Caches audio metadata to improve processing speed

---

## Limitations & Future Road-map

See [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) for a full breakdown of analytics features and engineering demonstrations.

- **Edge Cases:** Some grouping errors may persist for extremely messy libraries; manual review is advised for low-confidence groups.
- **LLM Cost:** API usage may incur costs; consider self-hosted options for large libraries.
- **GUI Theming:** Some widgets may not fully restyle in dark mode; future PySimpleGUI releases may improve this.
- **Automated Testing:** Add more automated tests and CI integration.

---

## Contact & License

- **Author:** Jakub Bartkowiak
- **License:** MIT (see LICENSE file)
- **Issues/PRs:** Contributions welcome!

## Dependencies

Key dependencies used in this project:

### Core Dependencies

- `mutagen>=1.47.0` - Audio metadata handling
- `python-magic>=0.4.27` - File type detection
- `rapidfuzz>=2.0.0` - Fast string matching
- `httpx>=0.24.0` - Async HTTP client
- `openai>=1.0.0` - AI-powered metadata enhancement
- `PySimpleGUI>=4.60.0` - User interface

### Development Dependencies

- `pytest>=7.0.0` - Testing framework
- `mypy>=1.0.0` - Static type checking
- `black>=23.0.0` - Code formatting

For a complete list of dependencies with exact versions, see [requirements.txt](requirements.txt).
