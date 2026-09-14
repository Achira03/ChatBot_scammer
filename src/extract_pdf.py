from __future__ import annotations
import io
import re
from collections import Counter
from pathlib import Path
import fitz
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from tqdm import tqdm

from . import settings

def _configure_tesseract() -> None:
    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

def _text_quality(text: str) -> float:
    """0..1 heuristic: printable Thai/English/digit ratio."""
    if not text:
        return 0.0
    useful = re.findall(r"[ก-๙A-Za-z0-9]", text)
    return len(useful) / max(len(text), 1)

def _needs_ocr(text: str, prefer_ocr: bool) -> bool:
    text = (text or "").strip()
    if prefer_ocr:
        # still keep embedded text if it is clearly substantial and usable
        return len(text) < 300 or _text_quality(text) < 0.35
    return len(text) < settings.MIN_TEXT_CHARS or _text_quality(text) < 0.25

def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    # Keep OCR preprocessing conservative; aggressive thresholding hurts Thai diacritics.
    img = img.filter(ImageFilter.SHARPEN)
    return img

def _ocr_page(page: fitz.Page) -> str:
    zoom = settings.OCR_DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img = _preprocess_for_ocr(img)
    return pytesseract.image_to_string(
        img,
        lang=settings.OCR_LANG,
        config="--oem 3 --psm 6"
    )

def _normalize_pdf_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_document(pdf_path: Path, document_id: str, prefer_ocr: bool = False) -> list[dict]:
    _configure_tesseract()
    doc = fitz.open(pdf_path)
    pages = []

    for page_idx in tqdm(range(len(doc)), desc=f"Extract {pdf_path.name}"):
        page = doc[page_idx]
        embedded = page.get_text("text") or ""
        use_ocr = _needs_ocr(embedded, prefer_ocr)
        if use_ocr:
            try:
                text = _ocr_page(page)
                method = "ocr"
            except Exception as exc:
                # Fallback allows the rest of the project to continue.
                text = embedded
                method = f"pdf_text_fallback_after_ocr_error:{type(exc).__name__}"
        else:
            text = embedded
            method = "pdf_text"

        pages.append({
            "document_id": document_id,
            "page": page_idx + 1,
            "method": method,
            "text": _normalize_pdf_text(text),
        })

    doc.close()
    return pages

def remove_repeated_headers_footers(pages: list[dict]) -> list[dict]:
    """
    Remove short lines that repeat on many pages (typical headers/footers).
    Conservative to avoid deleting actual recurring domain terms.
    """
    if len(pages) < 4:
        return pages

    page_line_sets = []
    for p in pages:
        lines = {
            re.sub(r"\s+", " ", line.strip())
            for line in p["text"].splitlines()
            if 4 <= len(line.strip()) <= 120
        }
        page_line_sets.append(lines)

    counts = Counter(line for s in page_line_sets for line in s)
    threshold = max(3, int(len(pages) * 0.45))
    repeated = {
        line for line, count in counts.items()
        if count >= threshold
        and not re.search(r"(OTP|SMS|LINE|มิจฉาชีพ|กลโกง|วิธีป้องกัน)", line, re.I)
    }

    cleaned = []
    for p in pages:
        kept = []
        for line in p["text"].splitlines():
            norm = re.sub(r"\s+", " ", line.strip())
            if norm in repeated:
                continue
            # bare page number
            if re.fullmatch(r"\d{1,3}", norm):
                continue
            kept.append(line)
        cp = dict(p)
        cp["text"] = _normalize_pdf_text("\n".join(kept))
        cleaned.append(cp)
    return cleaned
