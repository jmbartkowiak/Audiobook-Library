# REVISION_HISTORY.md

## Version 2.2.2 (2025-06-11)

### Major Features & Refactoring

* **GUI Overhaul**: Integrated core commands (`scan`, `enrich`, `convert`, `undo`) as buttons in the main GUI.
* **Progress Feedback**: Added a progress bar and status text to the GUI for real-time feedback on long-running tasks.
* **Background Tasking**: Implemented threading to run core commands in the background, keeping the GUI responsive.
* **Undo Functionality**: Completed the `undo` functionality to revert the last file conversion batch.
* **Task Orchestration**: Created a new `tasks.py` module to orchestrate background tasks and their progress reporting.
* **Core Logic Refactoring**: Modified `library_core.py` and `staging_db.py` to support progress callbacks and the new undo feature.

### Version 2.2.0 (2025-06-10)

### Major Features & Refactoring

* **Documentation & Dependency Synchronization**: Performed a comprehensive audit and cleanup of project documentation and dependencies to align with the stable, production-ready state of the application.
* **Global Version Update**: Updated all Python scripts, configuration files, and documentation to version `2.2.0` to ensure consistency across the entire project.

### Fixes & Improvements

* **Dependency Cleanup**: Cleaned and deduplicated `requirements.txt` to remove unused packages (`pydub`, `auditokit`) and development-only dependencies, leaving only essential production packages.
* **Documentation Accuracy**: Removed all references to `pydub` and `auditokit` from `README.md`, `TECHNICAL_DETAILS.md`, and `APPENDIX.md` to reflect the current codebase.
* **PySimpleGUI Installation Fix**: Added detailed, critical installation instructions for `PySimpleGUI` to `README.md` and `APPENDIX.md`, guiding users to install from the private PyPI index to resolve known `AttributeError` issues.
* **Markdown Linting**: Fixed numerous markdown linting errors (spacing, blank lines) across all documentation files for improved readability and consistency.
* **Codebase Alignment**: Ensured that all version numbers referenced within the code (e.g., in user-agent strings, GUI titles) are consistent with the global version.

## Version 2.1.5 (2025-06-11)

### Major Features & Refactoring

* **GUI Overhaul**: Completely refactored `ui_frontend.py` to use a lazy-loading data model. The GUI now loads book overviews instantly and fetches file details on-demand when a book is expanded, resulting in a much faster and more responsive user experience, especially with large libraries.
* **Core Logic Refactoring**: Restructured the application's core logic by separating the command-line interface from the data processing functions. `pipeline.py` is now a clean entry-point, while all core data processing logic (scanning, grouping, enrichment, moving) is consolidated in `library_core.py`.
* **Optimized Metadata Enrichment**: The metadata enrichment process in `library_core.py` was refactored to use the local OpenLibrary mirror for fetching book details after finding the book ID via the live API. This significantly speeds up the enrichment process by reducing network requests.

### Fixes & Improvements

* **Stable GUI**: Fixed critical syntax and structural errors in `ui_frontend.py` that caused instability, ensuring the event loop and all UI components function correctly.
* **Code Cleanup**: Removed dead code and legacy patterns from `staging_db.py` and `pipeline.py`.
* **Documentation**: Updated all modified file headers to version `2.1.5` and improved comments and `Directory` sections for better code navigation and maintainability.

## Version 2.1.4 (2025-06-10)

### Fixes & Improvements

* **Staging DB**: Refactored `staging_db.py` to remove dead code and add `get_books_overview` and `get_files_for_book` for efficient lazy-loading in the GUI.
* **OpenLibrary Mirror**: Overhauled `openlibrary_mirror.py` with a robust `MirrorUpdater` class, adding features like request retries, progress reporting, and atomic file writes to prevent corruption.
* **GUI**: Began refactoring `ui_frontend.py` for lazy-loading and integrated the OpenLibrary mirror update process with a progress popup.

## Version 2.1.0 (2025-06-09)

* **Initial Project Refactor**:  Continued major refactoring from a monolithic script to a modular, production-grade application.
* **Core Module Creation**: Established the foundational architecture by creating the initial versions of key modules:
  * `pipeline.py`: The main application entry point.
  * `library_core.py`: The central module for core data processing logic.
  * `staging_db.py`: The module for managing the SQLite database.

## Version 1.3.9 (Legacy)

* **Legacy Monolithic Script**: Represents the final version of the application before the major architectural refactoring.
* **Basic Functionality**: Contained basic, script-based functionality for scanning and reorganizing audiobooks without the advanced features, GUI, or modular design of later versions.

---

### Key for Developers

* **Major Version (X.0.0)**: Significant, potentially breaking changes to the architecture or core functionality.
* **Minor Version (0.X.0)**: Several new features, significant cross-module improvements, or major refactoring.
* **Patch Version (0.0.X)**: Bug fixes, minor improvements, single feature updates or additions, and documentation updates.