"""
processors/document_processor.py
Handles: .pdf, .doc, .docx, .odt, .odp, .ppt, .pptx

PDF extraction strategy (in order of preference):
  1. pdfplumber  — most reliable, handles complex layouts well
  2. pypdf        — lightweight fallback
  3. pymupdf      — final fallback (fitz), skip if import fails

Root cause of "Directory 'static/' does not exist":
  PyMuPDF >= 1.24 changed its internal font resource path and conflicts
  with Python 3.13. Using pdfplumber avoids this entirely.
"""

from pathlib import Path
from utils.logger import get_logger

log = get_logger("DocumentProcessor")


# ── PDF ────────────────────────────────────────────────────────────────────────

def _extract_pdf_pdfplumber(filepath: Path) -> str:
    """Primary PDF extractor — uses pdfplumber (most robust)."""
    import pdfplumber
    pages_text = []
    with pdfplumber.open(str(filepath)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(f"[Page {page_num}]\n{text.strip()}")
    return "\n\n".join(pages_text)


def _extract_pdf_pypdf(filepath: Path) -> str:
    """Fallback PDF extractor — uses pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(str(filepath))
    pages_text = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text and text.strip():
            pages_text.append(f"[Page {page_num}]\n{text.strip()}")
    return "\n\n".join(pages_text)


def _extract_pdf_pymupdf(filepath: Path) -> str:
    """Last-resort PDF extractor — uses PyMuPDF (fitz)."""
    import fitz  # pymupdf
    doc = fitz.open(str(filepath))
    pages_text = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        if text and text.strip():
            pages_text.append(f"[Page {page_num}]\n{text.strip()}")
    doc.close()
    return "\n\n".join(pages_text)


def _extract_pdf(filepath: Path) -> str:
    """
    Try PDF extractors in order of reliability.
    Falls through to the next if one raises an error.
    """
    extractors = [
        ("pdfplumber", _extract_pdf_pdfplumber),
        ("pypdf",      _extract_pdf_pypdf),
        ("pymupdf",    _extract_pdf_pymupdf),
    ]

    last_error = None
    for name, fn in extractors:
        try:
            log.info(f"  Trying PDF extractor: {name}")
            text = fn(filepath)
            if text.strip():
                log.info(f"  ✓ PDF extracted via {name}")
                return text
            else:
                log.warning(f"  {name} returned empty text, trying next extractor")
        except ImportError:
            log.warning(f"  {name} not installed, trying next extractor")
        except Exception as e:
            last_error = e
            log.warning(f"  {name} failed ({e}), trying next extractor")

    raise RuntimeError(
        f"All PDF extractors failed for {filepath.name}. "
        f"Last error: {last_error}\n"
        f"Install pdfplumber: pip install pdfplumber"
    )


# ── DOCX ───────────────────────────────────────────────────────────────────────
def _extract_docx(filepath: Path) -> str:
    """Extract paragraphs and table content from a .docx file."""
    from docx import Document
    doc = Document(str(filepath))
    parts = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                parts.append(row_text)

    return "\n".join(parts)


# ── ODT ────────────────────────────────────────────────────────────────────────
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
        "source":     filepath.name,
        "type":       "document",
        "content":    "",
        "char_count": 0,
        "error":      None,
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

        result["content"]    = text.strip()
        result["char_count"] = len(result["content"])
        log.info(f"  ✓ Extracted {result['char_count']:,} chars from {filepath.name}")

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  ✗ Failed: {filepath.name} — {e}")

    return result
