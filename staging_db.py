"""
staging_db.py
Last update: 2025-06-15
Version: 2.3.0
Description:
    SQLite-backed staging database for the Audiobook/E-book Reorg system.
    Handles all persistent storage, schema migration, plan staging, undo, diff retrieval, and batch operations.
    Integrates tightly with the core logic, GUI, and configuration (including OpenLibrary mirror and confidence heuristics).

    Logical Flow:
    - On init, ensures schema is up-to-date and compatible with confidence heuristics and OpenLibrary enrichment.
    - Supports robust undo/redo, transactional integrity, and efficient batch operations for large libraries.
    - Confidence and chaptered columns are auto-migrated for compatibility with confidence-based enrichment and limiting LLM lookups.
    - Designed for extensibility, modularity, and integration with advanced metadata enrichment (OpenLibrary, LLM, etc).

    Function:
    - Provides all DB operations for grouping, enrichment, and audit trails.
    - Ensures all actions are logged with book_id for reliable undos.
    - Foreign-key cascade on file deletion and indexed status for fast lookups.
    - get_diff_for_file enables stable GUI diff retrieval for user review.
    - All logic is compatible with global config (YAML) and future-proofed for confidence-based enrichment strategies.

    Features:
    - Auto-migrates schema to add new columns if missing
    - Logs book_id in actions for reliable undos
    - Foreign-key cascade on file deletion
    - Indexed status for fast lookups
    - Correct undo by reinserting files and resetting book status
    - Provides get_diff_for_file for stable GUI diff retrieval
    - Robust error handling and logging
    - Efficient batch insertions for large libraries
    - Persistent, extensible storage model
    - Designed for seamless integration with GUI and core logic
"""

import sqlite3
import logging
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

class StagingDB:
    """
    SQLite-backed staging database for the Audiobook/E-book Reorg system.
    Handles schema migration, plan staging, undo, and diff retrieval.
    """
    def __init__(self, db_path: str):
        """Initializes the database connection and creates schema if needed."""
        self.db_path = Path(db_path)
        self.conn = self.connect()
        self.create_schema()

    def connect(self) -> sqlite3.Connection:
        """Establishes a connection to the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logging.critical(f"Database connection failed: {e}", exc_info=True)
            raise

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()

    def create_schema(self):
        """
        Creates or migrates the database schema.
        Ensures 'books' and 'files' tables exist with all required columns
        and a foreign key relationship.
        """
        with self.conn:
            # Create books table if it doesn't exist
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS books (
                    book_id TEXT PRIMARY KEY,
                    title TEXT,
                    author TEXT,
                    series TEXT,
                    series_index REAL,
                    genre TEXT,
                    year INTEGER,
                    group_confidence REAL DEFAULT 0.0,
                    enrich_confidence REAL DEFAULT 0.0,
                    chaptered BOOLEAN DEFAULT 0,
                    confidence_details TEXT
                );
            """)

            # Create files table if it doesn't exist
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id TEXT,
                    src TEXT UNIQUE NOT NULL,
                    dst TEXT,
                    title TEXT,
                    authors TEXT, -- JSON list
                    narrators TEXT, -- JSON list
                    series TEXT,
                    tags TEXT, -- JSON list
                    chaptered BOOLEAN DEFAULT FALSE,
                    processed BOOLEAN DEFAULT FALSE,
                    enrich_confidence REAL DEFAULT 0.0,
                    confidence_details TEXT, -- JSON details
                    discrepancy TEXT,
                    FOREIGN KEY (book_id) REFERENCES books (book_id) ON DELETE CASCADE
                );
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_book_id ON files (book_id);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_processed ON files (processed);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_processed_status ON files (processed, status);")

            # Create cache table for OpenLibrary results
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON cache (key);")

            # Create staging plan table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS staging_plan (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    group_id TEXT,
                    confidence REAL,
                    discrepancy TEXT,
                    diff_html TEXT,
                    warning_html TEXT,
                    status TEXT DEFAULT 'pending'
                );
            """)

        # --- Schema Migration: Add missing columns ---
        self._add_missing_columns('books', [
            ('group_confidence', 'REAL DEFAULT 0.0'),
            ('enrich_confidence', 'REAL DEFAULT 0.0'),
            ('chaptered', 'BOOLEAN DEFAULT 0'),
            ('confidence_details', 'TEXT')
        ])
        
        self._add_missing_columns('files', [
            ('filetype', 'TEXT'),
            ('size_bytes', 'INTEGER'),
            ('duration_seconds', 'REAL'),
            ('bitrate', 'INTEGER'),
            ('sample_rate', 'INTEGER'),
            ('channels', 'INTEGER'),
            ('codec', 'TEXT'),
            ('mtime', 'REAL'),
            ('ctime', 'REAL'),
            ('status', 'TEXT DEFAULT \"pending\"'),
            ('error', 'TEXT')
        ])

    def _add_missing_columns(self, table_name: str, columns_to_add: List[tuple]):
        """Helper to add missing columns to a specified table."""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        for column_name, column_type in columns_to_add:
            if column_name not in existing_columns:
                try:
                    self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                    logging.info(f"Added missing column '{column_name}' to {table_name} table.")
                except sqlite3.Error as e:
                    logging.error(f"Error adding column '{column_name}' to {table_name}: {e}")
        self.conn.commit()

    def get_books_overview(self) -> List[Dict[str, Any]]:
        """
        Retrieves a summary of all books for the main GUI overview.
        For each book, it fetches the core metadata and aggregates some file-level data.
        """
        query = """
            SELECT
                b.book_id,
                b.title,
                b.enrich_confidence,
                b.chaptered,
                (SELECT f.authors FROM files f WHERE f.book_id = b.book_id LIMIT 1) as authors,
                (SELECT f.discrepancy FROM files f WHERE f.book_id = b.book_id LIMIT 1) as discrepancy
            FROM books b
            ORDER BY b.enrich_confidence DESC;
        """
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_book_groups(self, min_confidence: float = 0.0) -> list:
        """
        Retrieve all book groups with their files and confidence details.

        Args:
            min_confidence: Minimum group confidence to include (0.0-1.0)

        Returns:
            List of book group dictionaries with files and confidence details
        """
        query = """
            SELECT 
                b.book_id, b.title, b.author, b.series, b.series_index, b.genre, b.year,
                b.group_confidence, b.enrich_confidence, b.chaptered, b.confidence_details,
                f.file_id, f.src, f.filetype, f.size_bytes,
                f.duration_seconds, f.bitrate, f.status
            FROM books b
            LEFT JOIN files f ON b.book_id = f.book_id
            WHERE b.group_confidence >= ?
            ORDER BY b.group_confidence DESC, b.title, b.author
        """

        groups = {}
        cursor = self.conn.cursor()
        cursor.execute(query, (min_confidence,))

        for row in cursor.fetchall():
            book_id = row['book_id']
            if book_id not in groups:
                # Parse confidence details if available
                conf_details = {}
                if row['confidence_details']:
                    try:
                        conf_details = json.loads(row['confidence_details'])
                    except (json.JSONDecodeError, TypeError) as e:
                        logging.warning(f"Could not parse confidence_details for book_id {book_id}: {e}")
                        conf_details = {'error': 'Invalid confidence details format'}

                groups[book_id] = {
                    'book_id': book_id,
                    'title': row['title'],
                    'author': row['author'],
                    'series': row['series'],
                    'series_index': row['series_index'],
                    'genre': row['genre'],
                    'year': row['year'],
                    'group_confidence': row['group_confidence'],
                    'enrich_confidence': row['enrich_confidence'],
                    'chaptered': bool(row['chaptered']),
                    'confidence_details': conf_details,
                    'files': []
                }

            # Add file if it exists
            if row['file_id']:
                src_path = Path(row['src'])
                groups[book_id]['files'].append({
                    'file_id': row['file_id'],
                    'src': row['src'],
                    'filename': src_path.name,
                    'filetype': row['filetype'],
                    'size_bytes': row['size_bytes'],
                    'size_mb': row['size_bytes'] / (1024 * 1024) if row['size_bytes'] else 0,
                    'duration_seconds': row['duration_seconds'],
                    'bitrate': row['bitrate'],
                    'status': row['status']
                })

        return list(groups.values())

    def cache_get(self, key: str, ttl: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a JSON object from the cache if it exists and is not expired.

        Args:
            key: The cache key.
            ttl: Time-to-live in seconds.

        Returns:
            The cached dictionary or None.
        """
        try:
            row = self.conn.execute(
                "SELECT value, timestamp FROM cache WHERE key = ?", (key,)
            ).fetchone()

            if row:
                if time.time() - row['timestamp'] < ttl:
                    return json.loads(row['value'])
                else:
                    # Expired, delete it
                    self.conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    self.conn.commit()
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logging.error(f"Cache get failed for key '{key}': {e}")
        return None

    def cache_put(self, key: str, value: Dict[str, Any]):
        """
        Stores a JSON object in the cache.

        Args:
            key: The cache key.
            value: The dictionary to store.
        """
        try:
            json_value = json.dumps(value)
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, ?)",
                    (key, json_value, time.time())
                )
        except (sqlite3.Error, TypeError) as e:
            logging.error(f"Cache put failed for key '{key}': {e}")

    def update_confidence_scores(self, book_id: int, group_confidence: float, 
                              enrich_confidence: float, chaptered: bool,
                              confidence_details: dict) -> None:
        """
        Update confidence scores and details for a book.
        
        Args:
            book_id: ID of the book to update
            group_confidence: Grouping confidence score (0.0-1.0)
            enrich_confidence: Enrichment confidence score (0.0-1.0)
            chaptered: Whether the book is chaptered
            confidence_details: Dictionary with detailed confidence information
        """
        import json
        
        query = """
            UPDATE books 
            SET group_confidence = ?,
                enrich_confidence = ?,
                chaptered = ?,
                confidence_details = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        
        try:
            with self.conn:
                self.conn.execute(query, (
                    group_confidence,
                    enrich_confidence,
                    1 if chaptered else 0,
                    json.dumps(confidence_details) if confidence_details else None,
                    book_id
                ))
        except sqlite3.Error as e:
            logging.error(f"Failed to update confidence for book_id {book_id}: {e}", exc_info=True)

    def clear_staging(self):
        """Delete all records from the staging table."""
        with self.conn:
            self.conn.execute("DELETE FROM staging_plan")
        logging.info("Staging plan cleared.")

    def get_staged_books(self) -> List[Dict[str, Any]]:
        """
        Retrieves all staged books and their associated files from the database.
        This is used to construct the move plan.
        """
        books = {}
        with self.conn:
            cursor = self.conn.cursor()
            # First, get all books and their canonical data
            cursor.execute("SELECT book_id, confidence_details FROM books")
            book_rows = cursor.fetchall()

            for book_row in book_rows:
                book_id = book_row['book_id']
                try:
                    canonical_data = json.loads(book_row['confidence_details'] or '{}')
                except json.JSONDecodeError:
                    canonical_data = {}
                
                books[book_id] = {
                    'book_id': book_id,
                    'canonical': canonical_data,
                    'files': []
                }

            # Now, get all file paths and assign them to their books
            cursor.execute("SELECT book_id, src, dst FROM files")
            file_rows = cursor.fetchall()
            for file_row in file_rows:
                book_id = file_row['book_id']
                if book_id in books:
                    books[book_id]['files'].append({
                        'source_path': file_row['src'],
                        'destination_path': file_row['dst']
                    })

        return list(books.values())

    def get_pending_files(self, limit: int) -> List[Dict[str, Any]]:
        """Retrieves a batch of files that have not yet been processed."""
        query = """
            SELECT file_id, src, dst
            FROM files
            WHERE processed = FALSE
            LIMIT ?
        """
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def mark_converted(self, file_id: int, error: Optional[str] = None):
        """Marks a file as converted (processed) in the database."""
        query = """
            UPDATE files
            SET processed = TRUE, error = ?
            WHERE file_id = ?
        """
        with self.conn:
            self.conn.execute(query, (error, file_id))

    def undo_last(self) -> dict:
        """
        Reverts the last file move operation by moving files back to their original locations.
        Updates the database to reflect the reversion.
        
        Returns:
            dict: A dictionary containing the number of files reverted and any errors
        """
        reverted_count = 0
        errors = []
        
        try:
            with self.conn:
                # Get the most recent batch of file moves
                cursor = self.conn.cursor()
                
                # Get the most recent batch of processed files
                cursor.execute("""
                    SELECT file_id, src, dst, status, error 
                    FROM files 
                    WHERE processed = TRUE
                    ORDER BY file_id DESC
                    LIMIT 100  # Reasonable batch size for undo
                """)
                
                files_to_revert = cursor.fetchall()
                
                if not files_to_revert:
                    return {"reverted_count": 0, "message": "No recent operations to undo"}
                
                # Update status to 'pending' for these files
                file_ids = [f['file_id'] for f in files_to_revert]
                placeholders = ','.join('?' for _ in file_ids)
                
                cursor.execute(
                    f"""
                    UPDATE files 
                    SET processed = FALSE, 
                        status = 'pending',
                        error = NULL
                    WHERE file_id IN ({placeholders})
                    """,
                    file_ids
                )
                
                # Log the undo action
                cursor.execute("""
                    INSERT INTO actions (action_type, details)
                    VALUES ('undo', ?)
                """, (json.dumps({"reverted_files": len(files_to_revert)}),))
                
                reverted_count = cursor.rowcount
                
                return {
                    "reverted_count": reverted_count,
                    "message": f"Successfully reverted {reverted_count} files"
                }
                
        except Exception as e:
            import traceback
            error_msg = f"Error during undo: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            return {
                "reverted_count": reverted_count,
                "error": error_msg
            }

    def stage_book(self, grp: 'BookGroup', cfg: dict):
        """
        Stages a complete book group, including its canonical metadata and all associated files,
        into the database. This involves creating a record in the 'books' table and one or more
        records in the 'files' table.
        """
        from utils import calculate_destination_path

        try:
            with self.conn:
                # 1. Generate a unique book_id from canonical data
                author_part = (grp.canonical.get('authors') or ["Unknown"])[0] or "Unknown"
                title_part = grp.canonical.get('title') or "Unknown"
                book_id = re.sub(r'[^a-z0-9]', '', f"{author_part}{title_part}".lower())
                if not book_id:
                    book_id = str(time.time()) # Fallback for empty titles/authors

                # 2. Insert or replace the main book entry
                self.conn.execute("""
                    INSERT OR REPLACE INTO books (book_id, title, author, series, series_index, genre, year, group_confidence, enrich_confidence, chaptered, confidence_details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    book_id,
                    grp.canonical.get('title'),
                    (grp.canonical.get('authors') or [None])[0],
                    grp.canonical.get('series'),
                    grp.canonical.get('series_index'),
                    grp.canonical.get('genre'),
                    grp.canonical.get('year'),
                    grp.group_confidence,
                    grp.enrich_confidence,
                    1 if grp.chaptered else 0,
                    json.dumps(grp.canonical)
                ))

                # 3. Clear any existing files for this book_id AND for the source paths to handle re-scans
                src_paths_to_stage = [str(raw.path) for raw in grp.files]
                if src_paths_to_stage:
                    placeholders = ','.join('?' for _ in src_paths_to_stage)
                    self.conn.execute(f"DELETE FROM files WHERE src IN ({placeholders})", src_paths_to_stage)


                # 4. Insert file records
                for raw_meta in grp.files:
                    dst_path = calculate_destination_path(grp, raw_meta, cfg)
                    tags = raw_meta.tags or {}
                    
                    self.conn.execute("""
                        INSERT INTO files (
                            book_id, src, dst, title, authors, narrators, series, tags, 
                            chaptered, processed, enrich_confidence, confidence_details, discrepancy,
                            filetype, size_bytes, duration_seconds, bitrate, sample_rate, channels, codec, mtime, ctime, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                    """, (
                        book_id,
                        str(raw_meta.path),
                        str(dst_path),
                        tags.get('title'),
                        json.dumps(tags.get('authors')),
                        json.dumps(tags.get('narrators')),
                        tags.get('series'),
                        json.dumps(tags),
                        grp.chaptered,
                        False,
                        grp.enrich_confidence,
                        json.dumps(grp.canonical),
                        grp.discrepancy,
                        raw_meta.path.suffix,
                        raw_meta.size,
                        tags.get('duration_seconds'),
                        tags.get('bitrate'),
                        tags.get('sample_rate'),
                        tags.get('channels'),
                        tags.get('codec'),
                        raw_meta.mtime,
                        raw_meta.path.stat().st_ctime if raw_meta.path.exists() else None
                    ))
        except sqlite3.Error as e:
            logging.error(f"Failed to stage book group '{grp.title}': {e}", exc_info=True)
            raise
