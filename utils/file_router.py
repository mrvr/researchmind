"""utils/file_router.py — Routes uploaded files to the correct processor."""

from pathlib import Path
from typing import Literal

FileCategory = Literal["text", "document", "audio", "video", "spreadsheet", "unsupported"]

# Extension → category mapping
EXTENSION_MAP: dict[str, FileCategory] = {
    # Plain text
    ".txt":  "text",
    ".md":   "text",
    ".rtf":  "text",
    ".html": "text",
    ".htm":  "text",
    ".xml":  "text",

    # Rich documents
    ".pdf":  "document",
    ".doc":  "document",
    ".docx": "document",
    ".odt":  "document",
    ".odp":  "document",
    ".ppt":  "document",
    ".pptx": "document",

    # Audio
    ".mp3":  "audio",
    ".wav":  "audio",
    ".m4a":  "audio",
    ".ogg":  "audio",
    ".flac": "audio",
    ".aac":  "audio",
    ".wma":  "audio",

    # Video
    ".mp4":  "video",
    ".mkv":  "video",
    ".avi":  "video",
    ".mov":  "video",
    ".webm": "video",
    ".wmv":  "video",
    ".flv":  "video",

    # Spreadsheets / Data
    ".csv":  "spreadsheet",
    ".xls":  "spreadsheet",
    ".xlsx": "spreadsheet",
    ".ods":  "spreadsheet",
    ".tsv":  "spreadsheet",
}


def get_file_category(filepath: str | Path) -> FileCategory:
    """Return the category string for a given file path."""
    ext = Path(filepath).suffix.lower()
    return EXTENSION_MAP.get(ext, "unsupported")


def route_files(filepaths: list[str | Path]) -> dict[FileCategory, list[Path]]:
    """
    Group a list of file paths by category.

    Returns:
        {
            "text": [Path, ...],
            "document": [Path, ...],
            "audio": [Path, ...],
            "video": [Path, ...],
            "spreadsheet": [Path, ...],
            "unsupported": [Path, ...],
        }
    """
    groups: dict[FileCategory, list[Path]] = {
        "text": [], "document": [], "audio": [],
        "video": [], "spreadsheet": [], "unsupported": [],
    }
    for fp in filepaths:
        category = get_file_category(fp)
        groups[category].append(Path(fp))
    return groups
