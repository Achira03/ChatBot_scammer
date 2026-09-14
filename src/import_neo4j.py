from __future__ import annotations
from collections import defaultdict
from neo4j import GraphDatabase

from . import settings
from .ontology import NODE_TYPES

# Relations created by deterministic provenance layer in addition to ontology relations.
SYSTEM_REL_TYPES = {"SUPPORTED_BY", "HAS_SECTION", "DESCRIBES"}
ALLOWED_REL_TYPES = {
    "HAS_SECTION", "DESCRIBES", "RECOMMENDS", "REFERENCES",
    "USES_METHOD", "USES_CHANNEL", "HAS_SIGNAL", "REQUESTS_INFORMATION",
    "CAUSES", "PREVENTED_BY", "TARGETS", "RELATED_TO_LAW",
    "INSTANCE_OF", "INDICATES", "CONTACT", "HAS_CONTACT", "PROVIDES",
    "ADDRESSES", "SUPPORTED_BY"
}

def _safe_ident(value: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ValueError(f"Unsafe/unknown identifier: {value}")
    return value

def _batch(items, size=500):
    for i in range(0, len(items), size):
        yield items[i:i+size]

def _node_props(node: dict) -> dict:
    return {k: v for k, v in node.items() if k not in {"id", "type"} and v not in (None, "")}

def import_graph(graph: dict) -> None:
    if not settings.NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is empty. Set it in .env")

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            # One unique id constraint per label -> fast MERGE and duplicate protection.
            for label in sorted(NODE_TYPES):
                safe_label = _safe_ident(label, NODE_TYPES)
                cname = f"uniq_{safe_label.lower()}_id"
                session.run(
                    f"CREATE CONSTRAINT {cname} IF NOT EXISTS "
                    f"FOR (n:{safe_label}) REQUIRE n.id IS UNIQUE"
                ).consume()

            grouped_nodes = defaultdict(list)
            for n in graph["nodes"]:
                label = _safe_ident(n["type"], NODE_TYPES)
                grouped_nodes[label].append({
                    "id": n["id"],
                    "props": _node_props(n),
                })

            for label, rows in grouped_nodes.items():
                for batch in _batch(rows):
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MERGE (n:{label} {{id: row.id}})
                        SET n += row.props
                        """,
                        rows=batch
                    ).consume()

            grouped_rels = defaultdict(list)
            for r in graph["relationships"]:
                rel_type = _safe_ident(r["type"], ALLOWED_REL_TYPES)
                st = _safe_ident(r["source_type"], NODE_TYPES)
                tt = _safe_ident(r["target_type"], NODE_TYPES)
                props = {
                    k: v for k, v in r.items()
                    if k not in {"source", "source_type", "type", "target", "target_type"}
                    and v not in (None, "")
                }
                grouped_rels[(st, rel_type, tt)].append({
                    "source": r["source"],
                    "target": r["target"],
                    "props": props,
                })

            for (st, rel_type, tt), rows in grouped_rels.items():
                for batch in _batch(rows):
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MATCH (a:{st} {{id: row.source}})
                        MATCH (b:{tt} {{id: row.target}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r += row.props
                        """,
                        rows=batch
                    ).consume()

            result = session.run(
                "MATCH (n) WITH count(n) AS nodes "
                "MATCH ()-[r]->() RETURN nodes, count(r) AS relationships"
            ).single()
            print(f"Neo4j now has nodes={result['nodes']}, relationships={result['relationships']}")
    finally:
        driver.close()
