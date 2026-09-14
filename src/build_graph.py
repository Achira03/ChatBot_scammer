from __future__ import annotations
from collections import defaultdict
from .io_utils import stable_id, normalize_key, sha256_text
from .normalization import canonical_name
from .ontology import NODE_TYPES, relation_allowed

def _short_evidence(text: str, max_chars: int = 500) -> str:
    text = " ".join((text or "").split())
    return text[:max_chars]

def build_graph(documents: list[dict], chunks: list[dict], extractions: list[dict]) -> tuple[dict, dict]:
    """
    Returns (graph, validation_report)
    """
    nodes: dict[str, dict] = {}
    relationships: dict[tuple, dict] = {}
    report = {
        "accepted_entities": 0,
        "rejected_entities": [],
        "accepted_relations": 0,
        "rejected_relations": [],
    }

    doc_map = {d["document_id"]: d for d in documents}
    chunk_map = {c["chunk_id"]: c for c in chunks}

    # Document nodes
    for d in documents:
        node_id = stable_id("Document", d["document_id"])
        nodes[node_id] = {
            "id": node_id,
            "type": "Document",
            "name": d.get("title") or d["document_id"],
            "document_id": d["document_id"],
            "source": d.get("source", ""),
        }

    # Section nodes are deterministic, based on document + section text.
    section_ids = {}
    for c in chunks:
        section_name = (c.get("section") or c.get("chapter") or f"หน้า {c['page']}").strip()
        section_key = f"{c['document_id']}|{section_name}"
        sid = stable_id("Section", section_key)
        section_ids[(c["document_id"], section_name)] = sid
        if sid not in nodes:
            nodes[sid] = {
                "id": sid,
                "type": "Section",
                "name": section_name,
                "document_id": c["document_id"],
            }
        did = stable_id("Document", c["document_id"])
        rel_key = (did, "HAS_SECTION", sid)
        relationships[rel_key] = {
            "source": did, "source_type": "Document",
            "type": "HAS_SECTION",
            "target": sid, "target_type": "Section",
            "document_id": c["document_id"],
        }

    for ext in extractions:
        chunk = chunk_map.get(ext["chunk_id"])
        if not chunk:
            continue

        entity_lookup = {}
        for e in ext.get("entities", []):
            etype = e.get("type", "")
            raw_name = (e.get("name") or "").strip()
            if etype not in NODE_TYPES or etype in {"Document", "Section", "Evidence"} or not raw_name:
                report["rejected_entities"].append({"chunk_id": ext["chunk_id"], "entity": e})
                continue

            name = canonical_name(etype, raw_name)
            nid = stable_id(etype, name)
            entity_lookup[(etype, normalize_key(raw_name))] = nid
            entity_lookup[(etype, normalize_key(name))] = nid

            if nid not in nodes:
                nodes[nid] = {
                    "id": nid,
                    "type": etype,
                    "name": name,
                    "aliases": sorted({raw_name} - {name}),
                    "first_seen_document": chunk["document_id"],
                    "first_seen_page": chunk["page"],
                }
            else:
                aliases = set(nodes[nid].get("aliases", []))
                if raw_name != name:
                    aliases.add(raw_name)
                nodes[nid]["aliases"] = sorted(aliases)

            report["accepted_entities"] += 1

        # Ensure relation endpoints can resolve even if model omitted them from entities.
        def endpoint_id(etype: str, raw_name: str):
            if etype not in NODE_TYPES or etype in {"Document", "Section", "Evidence"}:
                return None
            cname = canonical_name(etype, raw_name.strip())
            nid = stable_id(etype, cname)
            if nid not in nodes:
                nodes[nid] = {
                    "id": nid,
                    "type": etype,
                    "name": cname,
                    "aliases": sorted({raw_name.strip()} - {cname}),
                    "first_seen_document": chunk["document_id"],
                    "first_seen_page": chunk["page"],
                }
            return nid

        for r in ext.get("relations", []):
            st = r.get("source_type", "")
            tt = r.get("target_type", "")
            rt = r.get("type", "")
            sn = (r.get("source_name") or "").strip()
            tn = (r.get("target_name") or "").strip()

            if not sn or not tn or not relation_allowed(st, rt, tt):
                report["rejected_relations"].append({
                    "chunk_id": ext["chunk_id"],
                    "reason": "schema_or_empty_endpoint",
                    "relation": r
                })
                continue

            sid = endpoint_id(st, sn)
            tid = endpoint_id(tt, tn)
            if not sid or not tid or sid == tid:
                report["rejected_relations"].append({
                    "chunk_id": ext["chunk_id"],
                    "reason": "invalid_endpoint",
                    "relation": r
                })
                continue

            evidence_text = _short_evidence(r.get("evidence_quote") or "")
            evidence_seed = f"{ext['chunk_id']}|{sid}|{rt}|{tid}|{evidence_text}"
            evidence_id = "evidence_" + sha256_text(evidence_seed)[:18]
            if evidence_id not in nodes:
                nodes[evidence_id] = {
                    "id": evidence_id,
                    "type": "Evidence",
                    "name": f"หลักฐาน หน้า {chunk['page']}",
                    "document_id": chunk["document_id"],
                    "page": chunk["page"],
                    "chunk_id": ext["chunk_id"],
                    "section": chunk.get("section", ""),
                    "text": evidence_text,
                }

            rel_key = (sid, rt, tid)
            rel_payload = relationships.get(rel_key)
            if rel_payload is None:
                rel_payload = {
                    "source": sid,
                    "source_type": st,
                    "type": rt,
                    "target": tid,
                    "target_type": tt,
                    "document_id": chunk["document_id"],
                    "page": chunk["page"],
                    "chunk_id": ext["chunk_id"],
                    "evidence_ids": [evidence_id],
                    "evidence_text": evidence_text,
                }
                relationships[rel_key] = rel_payload
            else:
                ids = set(rel_payload.get("evidence_ids", []))
                ids.add(evidence_id)
                rel_payload["evidence_ids"] = sorted(ids)

            # Evidence node attached to both endpoints for easy Browser tracing.
            for node_id, node_type in ((sid, st), (tid, tt)):
                ev_key = (node_id, "SUPPORTED_BY", evidence_id)
                relationships[ev_key] = {
                    "source": node_id,
                    "source_type": node_type,
                    "type": "SUPPORTED_BY",
                    "target": evidence_id,
                    "target_type": "Evidence",
                    "document_id": chunk["document_id"],
                    "page": chunk["page"],
                    "chunk_id": ext["chunk_id"],
                }

            # Section -> semantic object links help provenance/navigation.
            section_name = (chunk.get("section") or chunk.get("chapter") or f"หน้า {chunk['page']}").strip()
            section_id = section_ids[(chunk["document_id"], section_name)]
            for semantic_id, semantic_type in ((sid, st), (tid, tt)):
                if semantic_type in {"ScamType", "ScamCase", "ScamMethod"}:
                    skey = (section_id, "DESCRIBES", semantic_id)
                    relationships[skey] = {
                        "source": section_id,
                        "source_type": "Section",
                        "type": "DESCRIBES",
                        "target": semantic_id,
                        "target_type": semantic_type,
                        "document_id": chunk["document_id"],
                        "page": chunk["page"],
                    }

            report["accepted_relations"] += 1

    graph = {
        "nodes": list(nodes.values()),
        "relationships": list(relationships.values()),
        "stats": {
            "nodes": len(nodes),
            "relationships": len(relationships),
        }
    }
    return graph, report
