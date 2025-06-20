"""
ui_frontend.py
Last update: 2025-06-15
Version: 2.3.0
Description:
    Responsive PySimpleGUI interface for Audiobook/E-book Reorg. Provides an interactive, 
    user-friendly GUI for managing, reviewing, and converting large audiobook/e-book libraries. 
    Integrates with the core logic and database for real-time status, batch operations, and 
    persistent user preferences.

    Logical Flow:
    - Initializes UI components and loads user preferences
    - Starts background thread for OpenLibrary mirror update
    - Loads and displays book groups from the database
    - Handles user interactions (sorting, filtering, actions)
    - Manages batch operations with progress feedback
    - Persists UI state and preferences

    Directory:
    - launch_gui:line 72 - Main entry point for the GUI application
    - _create_confidence_bar:line 40 - Create a colored confidence bar visualization
    - _format_confidence_details:line 54 - Format confidence details for display in tooltip
    - show_ol_update_popup:line 114 - Show a non-blocking popup for OpenLibrary mirror update
    - run_ol_update_thread:line 127 - Run the OpenLibrary mirror update in a separate thread
    - save_gui_config:line 99 - Save GUI settings to settings.yaml
    - load_book_overview:line 173 - Loads and filters the main book list for the tree
    - load_files_for_book:line 199 - Lazily loads file details when a book is expanded
    - view_diff:line 216 - Shows a popup with source and destination paths for a file
"""
import queue
import PySimpleGUI as sg
from staging_db import StagingDB
from tasks import run_scan_task, run_enrich_task, run_convert_task, run_undo_task
from utils import setup_logging, BookGroup, RawMeta, load_and_prepare_config, get_library_snapshot, load_library_stats
from openlibrary_mirror import MirrorUpdater
import time
import threading
from typing import Any, Dict
from pathlib import Path
import logging
import tempfile
import json
import yaml
import os

def _create_confidence_bar(confidence: float) -> str:
    """Create a colored confidence bar visualization."""
    if confidence >= 0.7:
        color = '#4CAF50'  # Green
    elif confidence >= 0.5:
        color = '#FFC107'  # Amber
    else:
        color = '#F44336'  # Red
    
    # Create a simple bar using unicode block elements
    filled = '█' * int(confidence * 10)
    empty = '░' * (10 - len(filled))
    return f'[{color}]{filled}{empty} {confidence:.1f}[/]'

def _format_confidence_details(details: dict) -> str:
    """Format confidence details for display in tooltip."""
    lines = [f"Confidence: {details.get('final_score', 0):.2f}"]
    
    # Add base score components
    lines.append("\nScores:")
    for key, value in details.items():
        if key.endswith('_score') and key != 'final_score':
            lines.append(f"- {key.replace('_', ' ').title()}: {value:.2f}")
    
    # Add penalties if any
    if details.get('penalty_reasons'):
        lines.append("\nPenalties:")
        for reason in details.get('penalty_reasons', []):
            lines.append(f"- {reason}")
    
    return '\n'.join(lines)

def launch_gui() -> None:
    """
    Responsive PySimpleGUI interface for Audiobook/E-book Reorg.
    Features:
    - Live dark mode toggle
    - Robust column toggling (cannot hide all)
    - Tooltips for all controls
    - Persistent preferences (columns, dark mode, page size)
    - Expandable book/chapter tree with confidence indicators
    - Automatic OpenLibrary mirror update at startup if needed
    """
    cfg = load_and_prepare_config()
    # Access paths through the nested paths structure
    paths_cfg = cfg.get('paths', {})
    db = StagingDB(paths_cfg.get('staging_db_path'))

    # --- GUI Configuration ---
    gui_cfg = cfg.get("gui", {})
    dark_mode = gui_cfg.get("dark_mode", False)

    # --- Master loop to allow for window recreation (e.g., for theme/column changes) ---
    while True:
        # --- Theme and Styling ---
        theme_string = gui_cfg.get("theme", "DarkBlue3") if dark_mode else "LightGrey1"
        try:
            # Try the older method for backward compatibility
            sg.ChangeLookAndFeel(theme_string)
        except AttributeError:
            # Fallback to the modern method
            sg.theme(theme_string)
        
        # Get latest GUI config for this loop iteration
        visible_columns = gui_cfg.get("columns", ["ID", "Authors", "Book", "Conf", "Discrep", "Type"])
        all_columns = ["ID", "Authors", "Book", "Conf", "Discrep", "Type", "Source Path", "New Path"]
        page_size = gui_cfg.get("page_size", 35)
        book_view = gui_cfg.get("book_view", True)
        
        # --- Helper Functions ---
        def save_gui_config():
            """Save GUI settings to settings.yaml atomically."""
            cfg["gui"] = {
                "dark_mode": dark_mode,
                "page_size": page_size,
                "columns": visible_columns,
                "book_view": book_view,
            }

            settings_path = Path(__file__).parent / "settings.yaml"
            temp_path = None
            try:
                # Create a temporary file in the same directory to ensure atomic move
                with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=settings_path.parent, delete=False) as tf:
                    temp_path = tf.name
                    yaml.safe_dump(cfg, tf, allow_unicode=True, sort_keys=False)
                
                # Atomically replace the original file with the new one
                os.replace(temp_path, settings_path)
                temp_path = None  # Prevent deletion in finally block on success
            except Exception as e:
                logging.error(f"Failed to save settings.yaml: {e}", exc_info=True)
                sg.popup_error(f"Error saving GUI config: {e}")
            finally:
                # Clean up the temp file if it still exists (i.e., on error)
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        # --- OpenLibrary Mirror Update Popup --- #
        def show_ol_update_popup() -> sg.Window:
            """Show a non-blocking popup with progress bar for OpenLibrary mirror update."""
            layout = [
                [sg.Text("Checking for OpenLibrary Mirror updates...", key="-STATUS-")],
                [sg.Text("File:", size=(12,1)), sg.Text("", key="-FILENAME-", size=(60,1))],
                [sg.Text("URL:", size=(12,1)), sg.Text("", key="-URL-", size=(60,1))],
                [sg.Text("Speed:", size=(12,1)), sg.Text("", key="-SPEED-", size=(20,1))],
                [sg.Text("Last Update:", size=(12,1)), sg.Text("", key="-LAST-")],
                [sg.ProgressBar(1, orientation="h", size=(40,20), key="-PROG-")],
                [sg.Cancel()]
            ]
            return sg.Window("OpenLibrary Mirror Update", layout, modal=True, finalize=True, keep_on_top=True)

        def run_ol_update_thread(progress_cb):
            """Run the OpenLibrary mirror update in a separate thread."""
            try:
                updater = MirrorUpdater(cfg, progress_cb)
                result = updater.update_mirror()
                progress_cb({"status": "complete", **result, "done": True})
            except Exception as e:
                import traceback
                error_msg = f"Error updating mirror: {str(e)}\n\n{traceback.format_exc()}"
                progress_cb({"status": "error", "error": error_msg, "done": True})
        
        # --- Main Window Layout --- #
        lightbulb = "\U0001F4A1"
        bulb_btn = sg.Button(lightbulb, key="-BULB-", pad=((0,10),(0,0)), font=("Arial", 16), tooltip="Toggle dark mode")

        controls = [
            bulb_btn,
            sg.Checkbox("Dry Run", default=cfg.get("dryrun_default", False), key="-DRYRUN-", tooltip="Simulate actions without making changes"),
            sg.Text("Batch Size:"), sg.Spin(list(range(1, 51)), initial_value=cfg["batch_size"], key="-BATCH-", tooltip="Conversion batch size"),
            sg.Text("Page Size:"), sg.Spin(list(range(5, 101)), initial_value=page_size, key="-PAGE-", tooltip="Books per page"),
            sg.Checkbox("Book View", default=book_view, key="-BOOKVIEW-", tooltip="Hide low-confidence groups")
        ]

        hidden_cols = [c for c in all_columns if c not in visible_columns]
        right_click_menu = ['&Right', [f'Hide "{c}"' for c in visible_columns] + (['---'] if hidden_cols and visible_columns else []) + [f'Show "{c}"' for c in hidden_cols]]

        tree = sg.Tree(
            data=sg.TreeData(), headings=[c for c in all_columns if c in visible_columns],
            auto_size_columns=True, key="-TREE-", expand_x=True, expand_y=True,
            show_expanded=False, enable_events=True, num_rows=25,
            right_click_menu=right_click_menu, tooltip="Right-click header to show/hide columns."
        )
        
        action_buttons = {
            "-SCAN-": ("Scan Library", "Scan the library for new files and create a plan"),
            "-ENRICH-": ("Enrich Metadata", "Use online services to improve metadata for low-confidence groups"),
            "-CONVERT-": ("Convert Files", "Execute the planned file moves and renaming"),
            "-UNDO-": ("Undo Last Convert", "Revert the most recent batch of file moves"),
        }

        action_button_layout = [sg.Button(v[0], key=k, tooltip=v[1]) for k, v in action_buttons.items()]

        progress_bar = sg.ProgressBar(max_value=100, orientation="h", size=(60, 20), key="-ACTION_PROG-", visible=False)
        status_text = sg.Text("", key="-ACTION_STATUS-", size=(80, 1), visible=False)

        bottom_buttons = [sg.Button("Refresh", tooltip="Reload book list from database"), sg.Button("View Diff", tooltip="Show file differences"), sg.Button("Quit", tooltip="Exit the application")]

        # --- Auto-scan logic ---
        def check_and_run_scan(window, action_queue):
            # Get stats path from the same directory as the staging database
            stats_path = Path(paths_cfg.get('staging_db_path', 'staging.db')).parent / "library_stats.json"
            
            # Check if any source directories exist
            source_roots = []
            if paths_cfg.get('source_audiobook_root'):
                source_roots.append(paths_cfg['source_audiobook_root'])
            if paths_cfg.get('source_ebook_root'):
                source_roots.append(paths_cfg['source_ebook_root'])

            current_snapshot = get_library_snapshot(source_roots)

            needs_scan = False
            if not last_stats:
                reason = "No previous statistics found."
                needs_scan = True
            elif (last_stats.get("total_files") != current_snapshot.get("total_files") or
                  last_stats.get("total_size") != current_snapshot.get("total_size")):
                reason = "Library changes detected."
                needs_scan = True

            if needs_scan:
                window["-ACTION_STATUS-"].update(f"{reason} Starting automatic scan...", visible=True)
                window["-SCAN-"].update(disabled=True)
                # Use the main action_queue for consistency
                def task_callback(p, t, m):
                    # This callback is simple for the auto-scan, we just need to know it's running
                    pass
                threading.Thread(target=_run_task, args=(run_scan_task, cfg, db, task_callback), daemon=True).start()
            else:
                window["-ACTION_STATUS-"].update("No library changes detected. Scan skipped.", visible=True)

        layout = [
            controls,
            [tree],
            [sg.Column([action_button_layout], justification='center')],
            [sg.Column([[progress_bar]], justification='center')],
            [sg.Column([[status_text]], justification='center')],
            bottom_buttons
        ]
        window = sg.Window(f"Audiobook Reorg v{cfg.get('version', '2.2.0')}", layout, resizable=True, finalize=True)

        # Trigger initial data load and auto-scan check
        window.write_event_value('-AUTO_SCAN_CHECK-', None)
        window.write_event_value('-REFRESH-', None)

        # --- Data Loading and Caching --- #
        books_cache = {}
        files_cache = {}

        def load_book_overview():
            nonlocal books_cache, files_cache, page_size, book_view
            books_cache.clear()
            files_cache.clear()
            treedata = sg.TreeData()
            
            page_size = int(window["-PAGE-"].get())
            book_view = window["-BOOKVIEW-"].get()

            book_overviews = db.get_books_overview()
            filtered_books = sorted([b for b in book_overviews if not book_view or float(b['enrich_confidence']) >= 0.3], key=lambda x: -float(x['enrich_confidence']))[:page_size]

            for b in filtered_books:
                books_cache[b['book_id']] = b
                authors = json.loads(b['authors']) if isinstance(b['authors'], str) else b['authors']
                book_data = {
                    "ID": b['book_id'], "Authors": ", ".join(authors), "Book": b["title"],
                    "Conf": f"{b['enrich_confidence']:.2f}", "Discrep": b["discrepancy"],
                    "Type": "Chaptered" if b.get("chaptered") else "Monolithic"
                }
                row_values = [book_data.get(col, "") for col in visible_columns]
                treedata.insert("", b['book_id'], b["title"], row_values)
                treedata.insert(b['book_id'], f"_DUMMY_{b['book_id']}", "", [], icon=sg.EMOJI_BASE[0])

            window["-TREE-"].update(values=treedata)

        def load_files_for_book(book_id: str):
            if not book_id or book_id in files_cache:
                return

            window["-TREE-"].delete(f"_DUMMY_{book_id}")
            files = db.get_files_for_book(book_id)
            files_cache[book_id] = files

            for f in files:
                file_data = {
                    "ID": f["file_id"], "Book": Path(f.get("src", "")).name,
                    "Discrep": f.get("discrepancy", ""), "Source Path": f.get("src"), "New Path": f.get("dst")
                }
                file_row_values = [file_data.get(col, "") for col in visible_columns]
                window["-TREE-"].insert(book_id, f["file_id"], Path(f.get("src", "")).name, file_row_values)
        
        # --- Event Handlers --- #
        def view_diff(key: Any):
            if not key or "_DUMMY_" in str(key):
                sg.popup_error("Please select a valid file to view its diff.", title="Selection Error")
                return
                
            file_rec = next((f for book_files in files_cache.values() for f in book_files if f['file_id'] == key), None)

            if not file_rec:
                sg.popup_error("Please expand a book and select a specific file to view its diff.", title="Selection Error")
                return

            diff_layout = [[sg.Text(f"{label}:\n"), sg.Input(file_rec.get(k, 'N/A'), readonly=True, key=f"-{k.upper()}-")] for label, k in [("Source Path", 'src'), ("Destination Path", 'dst')]] + [[sg.Button("OK")]]
            sg.Window("View File Difference", diff_layout, modal=True).read(close=True)
        
        # --- Initial Load --- #
        load_book_overview()
        
        # --- OpenLibrary Mirror Update Logic --- #
        updater = MirrorUpdater(cfg)
        if updater.needs_update():
            ol_popup = show_ol_update_popup()
            update_data = {"done": False}
            
            def progress_cb(d: Dict[str, Any]):
                ol_popup.write_event_value("-OL_UPDATE-", d)
            
            threading.Thread(target=run_ol_update_thread, args=(progress_cb,), daemon=True).start()
            
            # Popup event loop
            while not update_data.get("done"):
                event, values = ol_popup.read(timeout=200)
                if event in (sg.WIN_CLOSED, "Cancel"):
                    # TODO: Add warning logic for cancellation
                    break
                
                if event == "-OL_UPDATE-":
                    update_data = values[event]
                    if "filename" in update_data: ol_popup["-FILENAME-"].update(update_data["filename"])
                    if "url" in update_data: ol_popup["-URL-"].update(update_data["url"])
                    if "speed" in update_data and update_data["speed"]: ol_popup["-SPEED-"].update(f"{update_data['speed']/1024/1024:.2f} MB/s")
                    if "last_update" in update_data: ol_popup["-LAST-"].update(time.strftime("%Y-%m-%d %H:%M", time.localtime(update_data["last_update"])) if update_data["last_update"] > 0 else "Never")
                    if "downloaded" in update_data and "total" in update_data: ol_popup["-PROG-"].update_bar(int(update_data["downloaded"]), max=max(1, int(update_data["total"])))
            
            ol_popup.close()

        # --- Worker Thread Management ---
        active_thread = None
        action_queue = queue.Queue()

        def _run_task(task_func, *args):
            result = task_func(*args)
            action_queue.put(result)

        def _update_progress_bar(current, total, text=""):
            window["-ACTION_PROG-"].update(current=current, max=total)
            window["-ACTION_STATUS-"].update(value=f"{text} ({current}/{total})")

        # --- Main Event Loop ---
        while True:
            event, values = window.read(timeout=100) # Timeout allows queue checking

            if event == sg.WIN_CLOSED or event == 'Quit' or event == sg.WINDOW_CLOSE_ATTEMPTED_EVENT:
                break

            if event == "-BULB-":
                dark_mode = not dark_mode
                save_gui_config()
                window.close()
                break  # Break from inner loop to recreate window

            if event == '-AUTO_SCAN_CHECK-':
                check_and_run_scan(window, action_queue)

            # --- Handle background task completion ---
            try:
                result = action_queue.get(block=False)
                sg.popup(result["message"], title=result["status"].title())
                # Re-enable buttons and hide progress bar
                for k in action_buttons:
                    window[k].update(disabled=False)
                window["-ACTION_PROG-"].update(visible=False)
                window["-ACTION_STATUS-"].update(visible=False)
                active_thread = None
                # Refresh data if needed
                if event in ("-SCAN-", "-CONVERT-", "-UNDO-", "-ENRICH-"):
                    tree.delete(*tree.get_children())
                    load_book_overview()
            except queue.Empty:
                pass

            # --- Handle Button Clicks ---
            if event in action_buttons and not active_thread:
                # Disable all action buttons
                for k in action_buttons:
                    window[k].update(disabled=True)
                
                # Show progress bar
                window["-ACTION_PROG-"].update(current=0, max=1, visible=True)
                window["-ACTION_STATUS-"].update(value="Starting...", visible=True)

                task_args = (cfg, db, _update_progress_bar)
                if event == "-SCAN-":
                    target_func = run_scan_task
                    task_args = (cfg, db, _update_progress_bar)
                elif event == "-CONVERT-":
                    target_func = run_convert_task
                    dryrun = sg.popup_yes_no("Perform a dry run? (No files will be moved)", title="Dry Run?") == "Yes"
                    task_args = (cfg, db, dryrun, _update_progress_bar)
                elif event == "-UNDO-":
                    target_func = run_undo_task
                    task_args = (db, _update_progress_bar)
                elif event == "-ENRICH-":
                    target_func = run_enrich_task
                    task_args = (cfg, db, _update_progress_bar)
                
                active_thread = threading.Thread(target=_run_task, args=(target_func,) + task_args, daemon=True)
                active_thread.start()

            if event == "-BULK_EDIT-":
                selected_keys = values["-TREE-"]
                if not selected_keys:
                    sg.popup_error("No items selected for bulk edit.")
                    continue
                
                # For simplicity, we'll just allow editing the genre for now
                new_genre = sg.popup_get_text("Enter new genre for selected items:", title="Bulk Edit Genre")
                if new_genre:
                    for key in selected_keys:
                        db.update_genre(key, new_genre)
                    load_book_overview() # Refresh view

            elif event == "Refresh":
                tree.delete(*tree.get_children())
                load_book_overview()

            elif event == "View Diff":
                if values["-TREE-"]:
                    item_key = values["-TREE-"][0]
                    show_diff_window(db, item_key)

            elif event == "tree_double_click":
                if values["-TREE-"]:
                    item_key = values["-TREE-"][0]
                    show_details_window(db, item_key, cfg)

            elif event.startswith("tree_heading_"):
                header, order = event.split('_')[-2:]
                sort_column(tree, header, order == 'asc')

            elif event == "-BULB-":
                dark_mode = not dark_mode
                break
            
            if event.startswith(('Hide "', 'Show "')):
                col = event.replace('Hide "', '').replace('Show "', '').strip('"')
                if col in visible_columns:
                    if len(visible_columns) > 1: visible_columns.remove(col)
                else: 
                    visible_columns.append(col)
                gui_cfg['columns'] = visible_columns # Update for recreation
                break
            
            if isinstance(event, tuple) and event[0] == '-TREE-' and event[1] == '+':
                load_files_for_book(event[2][0] if event[2] else None)

            if event in ("Refresh", "-BOOKVIEW-", "-PAGE-"):
                load_book_overview()
            elif event == "View Diff":
                if values["-TREE-"]: view_diff(values["-TREE-"][0])
                else: sg.popup_error("Please select an item to view the difference.", title="No Selection")
            elif event == "Enrich":
                selected = values["-TREE-"]
                if selected:
                    try:
                        # Run enrichment in a thread to keep UI responsive
                        threading.Thread(
                            target=run_enrich_task,
                            args=(cfg, db, lambda p, t, m: window.write_event_value("-PROGRESS-", (p, t, m))),
                            daemon=True
                        ).start()
                    except Exception as e:
                        logging.error(f"Error during enrichment: {e}", exc_info=True)
                        sg.popup_error(f"Error during enrichment: {e}", title="Error")
                else:
                    sg.popup_ok("Please select one or more items to enrich.", title="No Selection")
                    
            elif event == "Undo":
                try:
                    result = run_undo_task(cfg, db)
                    if 'error' in result:
                        sg.popup_error(f"Error during undo: {result['error']}", title="Error")
                    elif result.get('reverted_count', 0) > 0:
                        sg.popup_ok(f"Successfully reverted {result['reverted_count']} file(s).", title="Undo Complete")
                        load_book_overview()  # Refresh the view
                    else:
                        sg.popup_ok("No operations to undo.", title="Nothing to Undo")
                except Exception as e:
                    logging.error(f"Error during undo: {e}", exc_info=True)
                    sg.popup_error(f"Error during undo: {e}", title="Error")
                    
            elif event == "Convert":
                try:
                    # Get batch size from config or use default
                    batch_size = cfg.get('batch_size', 100)
                    dryrun = cfg.get('dryrun_default', True)
                    
                    # Run conversion in a thread to keep UI responsive
                    threading.Thread(
                        target=run_convert_task,
                        args=(cfg, db, batch_size, dryrun, 
                             lambda p, t, m: window.write_event_value("-PROGRESS-", (p, t, m))),
                        daemon=True
                    ).start()
                except Exception as e:
                    logging.error(f"Error during conversion: {e}", exc_info=True)
                    sg.popup_error(f"Error during conversion: {e}", title="Error")

        window.close()
        save_gui_config()
        if event in (sg.WIN_CLOSED, "Quit"): 
            break
