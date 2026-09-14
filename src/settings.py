from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env", override=True)

def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

PDF_DIR = BASE_DIR / os.getenv("PDF_DIR", "data/pdf")

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
OCR_LANG = os.getenv("OCR_LANG", "tha+eng")
OCR_DPI = int(os.getenv("OCR_DPI", "250"))
MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "80"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "2"))
EXTRACTION_MODE = os.getenv("EXTRACTION_MODE", "single").strip().lower()

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "3500"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "250"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "250"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

FORCE_REEXTRACT = os.getenv("FORCE_REEXTRACT", "false").lower() == "true"
FORCE_RECHUNK = os.getenv("FORCE_RECHUNK", "false").lower() == "true"
FORCE_RELLM = os.getenv("FORCE_RELLM", "false").lower() == "true"
IMPORT_NEO4J = _bool("IMPORT_NEO4J", True)

DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHUNKS_DIR = DATA_DIR / "chunks"
CACHE_DIR = DATA_DIR / "llm_cache"
GRAPH_DIR = DATA_DIR / "graph"
CONFIG_DIR = BASE_DIR / "config"

for p in (EXTRACTED_DIR, CHUNKS_DIR, CACHE_DIR, GRAPH_DIR):
    p.mkdir(parents=True, exist_ok=True)
