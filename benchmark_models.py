"""
Benchmark candidate Ollama models on YOUR labeled KG extraction samples.

Metrics:
- Entity exact canonical Precision / Recall / F1
- Relation exact Precision / Recall / F1
- Mean latency per chunk

Gold format: data/eval/gold.json
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from statistics import mean

from ollama import Client
from src import settings
from src.schemas import ExtractionResult
from src.ontology import NODE_TYPES, allowed_relations_text
from src.normalization import canonical_name
from src.io_utils import normalize_key

client = Client(host=settings.OLLAMA_HOST)

SYSTEM = """คุณคือ Knowledge Graph Extractor ภาษาไทย
ใช้เฉพาะ SOURCE TEXT ห้ามใช้ความรู้ภายนอก ห้ามเดา
คืน JSON ตาม schema เท่านั้น
สร้าง relation ได้เฉพาะ relation schema ที่กำหนด
"""

def canon_entity(e):
    return (
        e["type"],
        normalize_key(canonical_name(e["type"], e["name"]))
    )

def canon_rel(r):
    return (
        r["source_type"],
        normalize_key(canonical_name(r["source_type"], r["source_name"])),
        r["type"],
        r["target_type"],
        normalize_key(canonical_name(r["target_type"], r["target_name"])),
    )

def prf(pred: set, gold: set):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else 1.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f

def run_model(model: str, samples: list[dict]):
    e_scores, r_scores, times = [], [], []
    failures = 0

    for s in samples:
        prompt = f"""ALLOWED NODE TYPES
{", ".join(sorted(NODE_TYPES - {"Document","Section","Evidence"}))}

ALLOWED RELATION SCHEMA
{allowed_relations_text()}

SOURCE TEXT
<<<
{s["text"]}
>>>

สกัด entities และ relations ที่มีหลักฐานชัดเจน
"""
        t0 = time.perf_counter()
        try:
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                format=ExtractionResult.model_json_schema(),
                options={"temperature": 0, "num_ctx": settings.LLM_NUM_CTX},
                stream=False,
            )
            pred = ExtractionResult.model_validate_json(resp.message.content).model_dump()
        except Exception as exc:
            failures += 1
            pred = {"entities": [], "relations": []}
            print(f"[{model}] sample failure: {exc}")
        times.append(time.perf_counter() - t0)

        pe = {canon_entity(x) for x in pred["entities"]}
        ge = {canon_entity(x) for x in s["gold"]["entities"]}
        pr = {canon_rel(x) for x in pred["relations"]}
        gr = {canon_rel(x) for x in s["gold"]["relations"]}

        e_scores.append(prf(pe, ge))
        r_scores.append(prf(pr, gr))

    return {
        "model": model,
        "entity_precision": mean(x[0] for x in e_scores),
        "entity_recall": mean(x[1] for x in e_scores),
        "entity_f1": mean(x[2] for x in e_scores),
        "relation_precision": mean(x[0] for x in r_scores),
        "relation_recall": mean(x[1] for x in r_scores),
        "relation_f1": mean(x[2] for x in r_scores),
        "mean_seconds": mean(times),
        "failures": failures,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["qwen3.5:4b", "qwen3:4b", "qwen3:8b"]
    )
    parser.add_argument("--gold", default="data/eval/gold.json")
    args = parser.parse_args()

    samples = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    if not samples:
        raise SystemExit("gold.json ว่างอยู่ ให้ใส่ labeled samples ก่อน")

    results = [run_model(m, samples) for m in args.models]
    results.sort(key=lambda x: (x["relation_f1"], x["entity_f1"], -x["mean_seconds"]), reverse=True)

    print("\nMODEL BENCHMARK")
    print("-" * 120)
    header = f"{'model':22} {'Ent F1':>8} {'Rel F1':>8} {'Ent P':>8} {'Ent R':>8} {'Rel P':>8} {'Rel R':>8} {'sec':>8} {'fail':>6}"
    print(header)
    print("-" * 120)
    for r in results:
        print(
            f"{r['model'][:22]:22} "
            f"{r['entity_f1']:8.3f} {r['relation_f1']:8.3f} "
            f"{r['entity_precision']:8.3f} {r['entity_recall']:8.3f} "
            f"{r['relation_precision']:8.3f} {r['relation_recall']:8.3f} "
            f"{r['mean_seconds']:8.2f} {r['failures']:6d}"
        )

if __name__ == "__main__":
    main()
