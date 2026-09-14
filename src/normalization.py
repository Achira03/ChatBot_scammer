from __future__ import annotations
import re
from rapidfuzz import fuzz
from . import settings
from .io_utils import read_json, normalize_key

ALIASES = read_json(settings.CONFIG_DIR / "aliases.json")

def canonical_name(node_type: str, name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip())
    alias_map = ALIASES.get(node_type, {})
    key = normalize_key(name)

    # exact alias
    for alias, canonical in alias_map.items():
        if normalize_key(alias) == key:
            return canonical

    # conservative fuzzy alias only on reasonably long strings
    if len(name) >= 7:
        best = None
        best_score = 0
        for alias, canonical in alias_map.items():
            score = fuzz.ratio(key, normalize_key(alias))
            if score > best_score:
                best_score, best = score, canonical
        if best is not None and best_score >= 94:
            return best

    # lightweight canonical formatting
    if node_type == "Channel":
        u = name.upper()
        if u in {"SMS", "LINE"}:
            return u
    if node_type == "Information" and "OTP" in name.upper():
        return "OTP"
    return name
