from __future__ import annotations
import re
from dataclasses import dataclass
from . import settings
from .io_utils import sha256_text

HEADING_PATTERNS = [
    re.compile(r"^\s*บทที่\s*\d+\b.*$", re.I),
    re.compile(r"^\s*\d+(?:\.\d+)*\s+[^\n]{3,120}$"),
    re.compile(r"^\s*(?:แก๊งคอลเซ็นเตอร์|ซิมผี|บัญชีม้า|AI ปลอมเสียง|ลิงก์ข้อความหลอกลวง|App ดูดเงิน)\s*$", re.I),
    re.compile(r"^\s*(?:วิธีป้องกัน|วิธีปฏิบัติ|สัญญาณเตือน|ช่องทางการติดต่อ|กฎหมายที่เกี่ยวข้อง)\s*:?\s*$", re.I),
]

def _is_heading(line: str) -> bool:
    line = line.strip()
    if not (2 <= len(line) <= 140):
        return False
    return any(p.search(line) for p in HEADING_PATTERNS)

def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer paragraph/newline boundary.
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind("。", start, end),
                text.rfind(".", start, end),
            )
            if boundary > start + max_chars // 2:
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return parts

def make_chunks(pages: list[dict], document_meta: dict) -> list[dict]:
    chunks = []
    current_section = ""
    current_chapter = ""

    for page in pages:
        lines = [x.strip() for x in page["text"].splitlines() if x.strip()]
        blocks: list[tuple[str, str]] = []
        buffer = []

        def flush():
            nonlocal buffer
            if buffer:
                blocks.append((current_section, "\n".join(buffer).strip()))
                buffer = []

        for line in lines:
            if _is_heading(line):
                flush()
                if re.match(r"^\s*บทที่\s*\d+", line, re.I):
                    current_chapter = line
                current_section = line
            else:
                buffer.append(line)
        flush()

        if not blocks and page["text"].strip():
            blocks = [(current_section, page["text"].strip())]

        for section, block_text in blocks:
            pieces = _split_long_text(
                block_text,
                settings.MAX_CHUNK_CHARS,
                settings.CHUNK_OVERLAP_CHARS
            )
            for idx, piece in enumerate(pieces):
                if len(piece) < settings.MIN_CHUNK_CHARS and chunks:
                    # Attach very small tail to previous chunk from same doc/page when safe.
                    prev = chunks[-1]
                    if prev["document_id"] == page["document_id"] and prev["page"] == page["page"] \
                       and len(prev["text"]) + len(piece) + 2 <= settings.MAX_CHUNK_CHARS:
                        prev["text"] += "\n\n" + piece
                        prev["chunk_hash"] = sha256_text(prev["text"])
                        continue

                seed = f'{page["document_id"]}|{page["page"]}|{section}|{idx}|{piece}'
                chunks.append({
                    "chunk_id": "chunk_" + sha256_text(seed)[:16],
                    "chunk_hash": sha256_text(piece),
                    "document_id": page["document_id"],
                    "document_title": document_meta.get("title", ""),
                    "source": document_meta.get("source", ""),
                    "page": page["page"],
                    "chapter": current_chapter,
                    "section": section,
                    "text": piece,
                })
    return chunks
