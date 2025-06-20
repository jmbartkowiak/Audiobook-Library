# Revision History

---

### Version 2.3.0 (2025-06-20)

**Major Feature: OpenLibrary TSV to SQLite Migration & Pipeline Hardening**

This version introduces a complete overhaul of the OpenLibrary data integration, replacing the deprecated CSV mirror with a modern, robust TSV-to-SQLite pipeline. It also includes significant hardening of the download process and configuration.

-   **New Data Source**: The system now downloads the official OpenLibrary "all types" TSV dump (`ol_dump_latest.txt.gz`) directly from the source. The download URL has been corrected to resolve 404 errors from the previous endpoint.
-   **Automatic SQLite Conversion**: On the first launch of the GUI, the application automatically downloads the TSV dump and converts it into a local SQLite database (`mirror.sqlite`).
-   **Efficient Updates**: The mirror is now updated only if it is missing or older than a configurable number of days (`min_update_days`), significantly reducing unnecessary downloads.
-   **Disk Space Guard**: A new safety check ensures at least 4 GB of free disk space is available before initiating the large download, preventing disk-full errors.
-   **Performance Boost with `pigz`**: The system now automatically uses `pigz` (a parallel gzip implementation) for faster decompression if it is available on the system's PATH. This can be disabled with the `prefer_pigz: false` setting.
-   **New Dependency**: Added `openlibrary-to-sqlite` to handle the conversion process. This dependency has been added to `requirements.txt`.
-   **Configuration Overhaul**:
    -   The `settings.yaml` file has been updated to remove the obsolete `mirrors` list.
    -   New keys have been added under the `openlibrary` section: `dump_url`, `sqlite_path`, `min_update_days`, and `prefer_pigz`.
    -   The configuration validation schema in `utils.py` has been updated to enforce these new keys.
-   **Code Refactoring**: The `openlibrary_mirror.py` module has been extensively refactored to implement the new download, conversion, and safety-check logic.
-   **Documentation**: The `README.md` has been updated to document the new `pigz` recommendation and the updated `openlibrary` configuration settings.
-   **Version Synchronization**: All project files have been synchronized to version `2.3.0` to reflect this major update.

## Version 2.2.9 (2025-06-17)

### Fix

- **Critical**: Fixed `AttributeError: 'StagingDB' object has no attribute 'stage_book'` that occurred during library scan. Re-implemented the missing `stage_book` method in `staging_db.py` to correctly write scanned and enriched book data to the database, and updated calls in `library_core.py` to match. This resolves a major regression preventing the core scanning functionality from completing.

## Version 2.2.8 (2025-06-17)

### New Feature

- Implemented a non-destructive source/destination file processing workflow to protect original files.
- Added an integrated backup system for moved files, configured via the `backup_zip` key in `settings.yaml`.

### Refactor

- Overhauled `settings.yaml` to use new path keys: `source_audiobook_root`, `destination_audiobook_root`, `source_ebook_root`, and `destination_ebook_root`.
- Updated `library_core.py`, `utils.py`, `ui_frontend.py`, and `tasks.py` to support the new configuration and workflow.

### Fix

- Corrected `BackupManager` invocation in `perform_moves()` within `library_core.py` by passing the required `cfg` argument.

### Documentation

- Updated `README.md` to reflect the new workflow, configuration, and current version.

### Housekeeping

- Synchronized all module versions to `2.2.8`.

## Version 2.2.7 (2025-06-17)

### Fix

- `library_core.py`: Resolved critical `NameError: name 'queue_list' is not defined`.
- `staging_db.py`: Removed duplicate and unreachable `undo` logic.
- `utils.py`: Added missing imports, fixed an infinite loop in `resolve_collision`, and corrected a regex error.
- `confidence_scoring.py`: Fixed a bug where `blacklist_penalty` was applied twice.
- `tests/`: Corrected invalid mock configurations in `test_confidence_scoring.py` and `test_file_management.py`.

### Housekeeping

- Synchronized all active Python modules to version `2.2.7`.
- Confirmed the full test suite passes with 100% success.

## Version 2.2.5 (2025-06-16)

### Fix

- `ui_frontend.py`: Added `import threading` to resolve a `NameError` and used a fallback for `PySimpleGUI` theme setting for backward compatibility.
- `library_core.py`: Fixed a syntax error (`unterminated string literal`) and a broken implementation in `perform_moves`. Added missing imports.

## Version 2.2.2b (Beta) (2025-06-14)

### Documentation

- Updated `README.md`, `TECHNICAL_DETAILS.md`, and `LARGE_SCALE_TESTING.md` to version `2.2.2b`.
- Added a new "Task-Based GUI Architecture" section to `README.md`.
- Resolved over 100 markdown-lint warnings in `TECHNICAL_DETAILS.md`.

### Housekeeping

- Harmonized all documentation references to version `2.2.2b (Beta)`.

## Version 2.2.2 (2025-06-11)

### New Feature

- Integrated core commands (`scan`, `enrich`, `convert`, `undo`) as buttons in the main GUI.
- Added a progress bar and status text to the GUI for real-time feedback.
- Implemented threading to run core commands in the background, keeping the GUI responsive.
- Completed the `undo` functionality.

### Refactor

- Created a new `tasks.py` module to orchestrate background tasks.

- **Core Logic Refactoring**: Modified `library_core.py` and `staging_db.py` to support progress callbacks and the new undo feature.

### Version 2.2.0/2.2.1 (2025-06-10)

- **Documentation & Dependency Synchronization**: Performed a comprehensive audit and cleanup of project documentation and dependencies to align with the stable, production-ready state of the application.
- **Global Version Update**: Updated all Python scripts, configuration files, and documentation to version `2.2.0` to ensure consistency across the entire project.

### Version 2.2.0/2.2.1 Dependency & Documentation Cleanup

- **Dependency Cleanup**: Cleaned and deduplicated `requirements.txt` to remove unused packages (`pydub`, `auditokit`) and development-only dependencies, leaving only essential production packages.
- **Documentation Accuracy**: Removed all references to `pydub` and `auditokit` from `README.md`, `TECHNICAL_DETAILS.md`, and `APPENDIX.md` to reflect the current codebase.
- **PySimpleGUI Installation Fix**: Added detailed, critical installation instructions for `PySimpleGUI` to `README.md` and `APPENDIX.md`, guiding users to install from the private PyPI index to resolve known `AttributeError` issues.
- **Markdown Linting**: Fixed numerous markdown linting errors (spacing, blank lines) across all documentation files for improved readability and consistency.
- **Codebase Alignment**: Ensured that all version numbers referenced within the code (e.g., in user-agent strings, GUI titles) are consistent with the global version.

## Version 2.1.5 (2025-06-11)

### GUI & Core Logic Refactoring

- **GUI Overhaul**: Completely refactored `ui_frontend.py` to use a lazy-loading data model. The GUI now loads book overviews instantly and fetches file details on-demand when a book is expanded, resulting in a much faster and more responsive user experience, especially with large libraries.
- **Core Logic Refactoring**: Restructured the application's core logic by separating the command-line interface from the data processing functions. `pipeline.py` is now a clean entry-point, while all core data processing logic (scanning, grouping, enrichment, moving) is consolidated in `library_core.py`.
- **Optimized Metadata Enrichment**: The metadata enrichment process in `library_core.py` was refactored to use the local OpenLibrary mirror for fetching book details after finding the book ID via the live API. This significantly speeds up the enrichment process by reducing network requests.

### Version 2.1.5 GUI & Core Logic Stability

- **Stable GUI**: Fixed critical syntax and structural errors in `ui_frontend.py` that caused instability, ensuring the event loop and all UI components function correctly.
- **Code Cleanup**: Removed dead code and legacy patterns from `staging_db.py` and `pipeline.py`.
- **Documentation**: Updated all modified file headers to version `2.1.5` and improved comments and `Directory` sections for better code navigation and maintainability.

## Version 2.1.4 (2025-06-10)

### Database & Mirror Refactoring

- **Staging DB**: Refactored `staging_db.py` to remove dead code and add `get_books_overview` and `get_files_for_book` for efficient lazy-loading in the GUI.
- **OpenLibrary Mirror**: Overhauled `openlibrary_mirror.py` with a robust `MirrorUpdater` class, adding features like request retries, progress reporting, and atomic file writes to prevent corruption.
- **GUI**: Began refactoring `ui_frontend.py` for lazy-loading and integrated the OpenLibrary mirror update process with a progress popup.

## Version 2.1.0 (2025-06-09)

- **Initial Project Refactor**:  Continued major refactoring from a monolithic script to a modular, production-grade application.
- **Core Module Creation**: Established the foundational architecture by creating the initial versions of key modules:
  - `pipeline.py`: The main application entry point.
  - `library_core.py`: The central module for core data processing logic.
  - `staging_db.py`: The module for managing the SQLite database.

## Version 1.3.9 (Legacy)

- **Legacy Monolithic Script**: Represents the final version of the application before the major architectural refactoring.
- **Basic Functionality**: Contained basic, script-based functionality for scanning and reorganizing audiobooks without the advanced features, GUI, or modular design of later versions.

---

### Key for Developers

- **Major Version (X.0.0)**: Significant, potentially breaking changes to the architecture or core functionality.
- **Minor Version (0.X.0)**: Several new features, significant cross-module improvements, or major refactoring.
- **Patch Version (0.0.X)**: Bug fixes, minor improvements, single feature updates or additions, and documentation updates.