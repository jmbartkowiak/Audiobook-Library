# GUI Guide (v2.2.0)

*See [README.md](README.md) for project overview, installation, and configuration.*

## 1. GUI Overview

The graphical user interface provides a comprehensive and interactive way to manage your audiobook library. It is the human-in-the-loop validation step for the data pipeline, allowing you to review, edit, and approve the reorganization plan before any files are moved.

### 1.1. Main Window Layout

The main window is divided into three sections:

1. **Controls**: A toolbar at the top with buttons for common actions like refreshing the view, expanding/collapsing the tree, and toggling the theme.
2. **Book Tree**: The central area displaying your audiobook library in a hierarchical tree view. Each top-level item represents a book, which can be expanded to show its individual files.
3. **Pagination and Filtering**: Controls at the bottom for navigating through large libraries, adjusting the number of items per page, and filtering the view.

### 1.2. Key Features

- **Lazy Loading**: The GUI efficiently handles large libraries by loading data on demand as you scroll and expand items. This is crucial for performance with tens of thousands of files.
- **Confidence Scoring**: Visual indicators (color-coded bars and tooltips) provide at-a-glance information about the confidence of the automated grouping and metadata enrichment.
- **Interactive Tree**: Expand and collapse book entries to view individual files and their proposed new paths.
- **Customizable View**: Show, hide, and reorder columns to suit your workflow. Your preferences are saved to `settings.yaml`.
- **Dark/Light Theme**: Toggle between dark and light themes for comfortable viewing.

## 2. Understanding the Book Tree

The book tree is the core of the GUI. Each row represents a book or a file and contains several columns of information:

- **Title**: The title of the book.
- **Author**: The author of the book.
- **Type**: Whether the book is `Chaptered` (multiple files) or `Single` (a single file).
- **Confidence**: A visual representation of the confidence score for the book's grouping and metadata. Hover over the bar for a detailed breakdown.
- **Status**: The current status of the book (e.g., `High Conf.`, `Medium Conf.`, `Low Conf.`)
- **Files**: The number of files in the book group.

### 2.1. Confidence Visualization

The confidence score is represented by a colored bar:

- **Green**: High confidence (≥ 0.7)
- **Amber**: Medium confidence (≥ 0.5)
- **Red**: Low confidence (< 0.5)

Hovering over the confidence bar will display a tooltip with a detailed breakdown of the scores and any penalties that were applied. This transparency is key to trusting the automated decisions.

## 3. Common Operations

- **Expand/Collapse**: Use the `Expand All` and `Collapse All` buttons or click the arrows next to each book to show or hide its files.
- **Refresh**: Click the `Refresh` button to reload the library view from the database.
- **Filtering**: Use the `Filter` input at the bottom to search for specific books by title or author.
- **Pagination**: Use the pagination controls to navigate through your library if it spans multiple pages.
- **Actions**: The main pipeline actions (`scan`, `convert`, `undo`, `enrich`) are primarily run from the command line. The GUI serves as the primary tool for reviewing the staged plan created by `scan`.

## 4. The GUI's Role in the Data Pipeline

The GUI is more than just a user interface; it's a critical component of the data engineering workflow. It provides the essential "human-in-the-loop" capability, allowing for oversight and correction of the automated processes. By presenting the staged plan with clear confidence metrics, it empowers you to make informed decisions, ensuring data quality and preventing incorrect file operations before they happen.
