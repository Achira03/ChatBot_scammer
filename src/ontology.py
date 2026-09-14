from __future__ import annotations
from . import settings
from .io_utils import read_json

ONTOLOGY = read_json(settings.CONFIG_DIR / "ontology.json")
NODE_TYPES = set(ONTOLOGY["node_types"])
REL_SCHEMA = ONTOLOGY["relationship_schema"]

def relation_allowed(source_type: str, rel_type: str, target_type: str) -> bool:
    return target_type in REL_SCHEMA.get(source_type, {}).get(rel_type, [])

def allowed_relations_text() -> str:
    lines = []
    for s, rels in REL_SCHEMA.items():
        for rel, targets in rels.items():
            lines.append(f"- {s} -[{rel}]-> {', '.join(targets)}")
    return "\n".join(lines)
