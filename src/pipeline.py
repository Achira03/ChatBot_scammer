from __future__ import annotations
from pathlib import Path
from tqdm import tqdm

from . import settings
from .io_utils import read_json, write_json
from .extract_pdf import extract_document, remove_repeated_headers_footers
from .chunk_document import make_chunks
from .llm_extract import extract_chunk
from .build_graph import build_graph
from .import_neo4j import import_graph

def _load_documents() -> list[dict]:
    cfg = read_json(settings.CONFIG_DIR / "documents.json")
    return cfg["documents"]

def stage_extract(documents: list[dict]) -> list[dict]:
    output_path = settings.EXTRACTED_DIR / "pages.json"
    if output_path.exists() and not settings.FORCE_REEXTRACT:
        print(f"[cache] extraction: {output_path}")
        return read_json(output_path)

    all_pages = []
    for d in documents:
        pdf_path = settings.PDF_DIR / d["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"ไม่พบ PDF: {pdf_path}\n"
                f"ให้นำไฟล์ PDF ไปไว้ใน {settings.PDF_DIR}"
            )
        pages = extract_document(
            pdf_path=pdf_path,
            document_id=d["document_id"],
            prefer_ocr=bool(d.get("prefer_ocr", False)),
        )
        pages = remove_repeated_headers_footers(pages)
        all_pages.extend(pages)

    write_json(output_path, all_pages)
    return all_pages

def stage_chunk(documents: list[dict], pages: list[dict]) -> list[dict]:
    output_path = settings.CHUNKS_DIR / "chunks.json"
    if output_path.exists() and not settings.FORCE_RECHUNK:
        print(f"[cache] chunks: {output_path}")
        return read_json(output_path)

    by_doc = {}
    for p in pages:
        by_doc.setdefault(p["document_id"], []).append(p)

    chunks = []
    for d in documents:
        chunks.extend(make_chunks(by_doc.get(d["document_id"], []), d))

    write_json(output_path, chunks)
    return chunks

def stage_llm(chunks: list[dict]) -> list[dict]:
    output_path = settings.GRAPH_DIR / "raw_extractions.json"
    # Individual chunk caching is always used; this aggregate file is for inspection.
    extractions = []
    for c in tqdm(chunks, desc=f"LLM extraction ({settings.LLM_MODEL})"):
        try:
            extractions.append(extract_chunk(c, force=settings.FORCE_RELLM))
        except Exception as exc:
            print(f"\n[WARN] chunk failed: {c['chunk_id']} -> {type(exc).__name__}: {exc}")
            extractions.append({
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "page": c["page"],
                "section": c.get("section", ""),
                "model": settings.LLM_MODEL,
                "entities": [],
                "relations": [],
                "error": f"{type(exc).__name__}: {exc}",
            })
    write_json(output_path, extractions)
    return extractions

def stage_graph(documents: list[dict], chunks: list[dict], extractions: list[dict]) -> dict:
    graph, report = build_graph(documents, chunks, extractions)
    write_json(settings.GRAPH_DIR / "validated_graph.json", graph)
    write_json(settings.GRAPH_DIR / "validation_report.json", report)

    nodes_by_type = {}
    for n in graph["nodes"]:
        nodes_by_type[n["type"]] = nodes_by_type.get(n["type"], 0) + 1

    print("\n=== Graph summary ===")
    print(f"Nodes: {graph['stats']['nodes']}")
    print(f"Relationships: {graph['stats']['relationships']}")
    print("Nodes by type:")
    for k, v in sorted(nodes_by_type.items()):
        print(f"  {k}: {v}")
    print(f"Rejected relations: {len(report['rejected_relations'])}")
    return graph

def run_pipeline():
    documents = _load_documents()

    print("[1/5] Extract PDF / OCR")
    pages = stage_extract(documents)

    print("[2/5] Structure-aware chunking")
    chunks = stage_chunk(documents, pages)
    print(f"Chunks: {len(chunks)}")

    print("[3/5] LLM entity + relation extraction")
    extractions = stage_llm(chunks)

    print("[4/5] Normalize + validate + build graph")
    graph = stage_graph(documents, chunks, extractions)

    if settings.IMPORT_NEO4J:
        print("[5/5] Import Neo4j")
        import_graph(graph)
    else:
        print("[5/5] Skip Neo4j import (IMPORT_NEO4J=false)")

    print("\nDone.")
    print(f"Graph JSON: {settings.GRAPH_DIR / 'validated_graph.json'}")
