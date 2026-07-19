"""
processors/text_processor.py
Handles: .txt, .md, .rtf, .html, .htm, .xml
Returns raw extracted text.
"""

from pathlib import Path
import chardet
from utils.logger import get_logger

log = get_logger("TextProcessor")


def _read_with_encoding(filepath: Path) -> str:
    """Read a file, auto-detecting encoding if UTF-8 fails."""
    try:
        return filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = filepath.read_bytes()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "latin-1"
        log.warning(f"UTF-8 failed for {filepath.name}, using detected encoding: {encoding}")
        return raw.decode(encoding, errors="replace")


def _extract_rtf(filepath: Path) -> str:
    """Strip RTF markup and return plain text."""
    try:
        from striprtf.striprtf import rtf_to_text
        raw = _read_with_encoding(filepath)
        return rtf_to_text(raw)
    except ImportError:
        log.warning("striprtf not installed — returning raw RTF content")
        return _read_with_encoding(filepath)


def _extract_html(filepath: Path) -> str:
    """Parse HTML and return visible text only."""
    try:
        from html.parser import HTMLParser

        class _TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self._parts: list[str] = []
                self._skip_tags = {"script", "style", "head"}
                self._in_skip = False

            def handle_starttag(self, tag, attrs):
                if tag in self._skip_tags:
                    self._in_skip = True

            def handle_endtag(self, tag):
                if tag in self._skip_tags:
                    self._in_skip = False

            def handle_data(self, data):
                if not self._in_skip:
                    stripped = data.strip()
                    if stripped:
                        self._parts.append(stripped)

            def get_text(self):
                return "\n".join(self._parts)

        parser = _TextExtractor()
        parser.feed(_read_with_encoding(filepath))
        return parser.get_text()
    except Exception as e:
        log.warning(f"HTML parsing failed ({e}), returning raw text")
        return _read_with_encoding(filepath)


def process(filepath: str | Path) -> dict:
    """
    Extract text from a plain-text file.

    Returns:
        {
            "source": str,           # filename
            "type": "text",
            "content": str,          # full extracted text
            "char_count": int,
            "error": str | None
        }
    """
    filepath = Path(filepath)
    log.info(f"Processing text file: {filepath.name}")

    result = {
        "source": filepath.name,
        "type": "text",
        "content": "",
        "char_count": 0,
        "error": None,
    }

    try:
        ext = filepath.suffix.lower()

        if ext == ".rtf":
            text = _extract_rtf(filepath)
        elif ext in (".html", ".htm"):
            text = _extract_html(filepath)
        else:
            # .txt, .md, .xml, .csv treated as plain text
            text = _read_with_encoding(filepath)

        result["content"] = text.strip()
        result["char_count"] = len(result["content"])
        log.info(f"  ✓ Extracted {result['char_count']:,} chars from {filepath.name}")

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  ✗ Failed to process {filepath.name}: {e}")

    return result
