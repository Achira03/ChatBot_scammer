from __future__ import annotations
import json
import hashlib
import re
from pathlib import Path
from typing import Any

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def stable_id(node_type: str, name: str) -> str:
    raw = f"{node_type}|{normalize_key(name)}"
    return f"{node_type.lower()}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"

def normalize_key(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”\"'`]+", "", text)
    text = text.strip(" .,:;!?()[]{}<>-–—")
    return text
