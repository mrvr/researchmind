"""
processors/document_processor.py
Handles: .pdf, .doc, .docx, .odt, .odp, .ppt, .pptx
Returns raw extracted text per file.
"""

from pathlib import Path
from utils.logger import get_logger

log = get_logger("DocumentProcessor")


# ── PDF ────────────────────────────────────────────────────────────────────────
def _extract_pdf(filepath: Path) -> str:
    """Extract all text from a PDF using PyMuPDF (fitz)."""
    import fitz  # pymupdf
    doc = fitz.open(str(filepath))
    pages_text = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        if text.strip():
            pages_text.append(f"[Page {page_num}]\n{text.strip()}")
    doc.close()
    return "\n\n".join(pages_text)


# ── DOCX ───────────────────────────────────────────────────────────────────────
def _extract_docx(filepath: Path) -> str:
    """Extract paragraphs and table content from a .docx file."""
    from docx import Document
    doc = Document(str(filepath))
    parts = []

    # Paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())

    # Tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)

    return "\n".join(parts)


# ── ODT (OpenDocument Text) ────────────────────────────────────────────────────
def _extract_odt(filepath: Path) -> str:
    """Extract text from ODT/ODP files using odfpy."""
    from odf.opendocument import load
    from odf.text import P
    from odf.element import Element

    doc = load(str(filepath))
    texts = []

    def _recurse(element):
        if isinstance(element, P):
            content = str(element)
            if content.strip():
                texts.append(content.strip())
        for child in element.childNodes:
            if isinstance(child, Element):
                _recurse(child)

    _recurse(doc.body)
    return "\n".join(texts)


# ── PPTX ───────────────────────────────────────────────────────────────────────
def _extract_pptx(filepath: Path) -> str:
    """Extract text from all slides in a .pptx file."""
    from pptx import Presentation
    prs = Presentation(str(filepath))
    slides_text = []
    for slide_num, slide in enumerate(prs.slides, 1):
        slide_parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        slide_parts.append(t)
        if slide_parts:
            slides_text.append(f"[Slide {slide_num}]\n" + "\n".join(slide_parts))
    return "\n\n".join(slides_text)


# ── DISPATCHER ─────────────────────────────────────────────────────────────────
def process(filepath: str | Path) -> dict:
    """
    Extract text from a rich document file.

    Returns:
        {
            "source": str,
            "type": "document",
            "content": str,
            "char_count": int,
            "error": str | None
        }
    """
    filepath = Path(filepath)
    log.info(f"Processing document: {filepath.name}")

    result = {
        "source": filepath.name,
        "type": "document",
        "content": "",
        "char_count": 0,
        "error": None,
    }

    try:
        ext = filepath.suffix.lower()

        if ext == ".pdf":
            text = _extract_pdf(filepath)
        elif ext in (".docx", ".doc"):
            text = _extract_docx(filepath)
        elif ext in (".odt", ".odp"):
            text = _extract_odt(filepath)
        elif ext in (".pptx", ".ppt"):
            text = _extract_pptx(filepath)
        else:
            raise ValueError(f"Unsupported document format: {ext}")

        result["content"] = text.strip()
        result["char_count"] = len(result["content"])
        log.info(f"  ✓ Extracted {result['char_count']:,} chars from {filepath.name}")

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  ✗ Failed: {filepath.name} — {e}")

    return result
