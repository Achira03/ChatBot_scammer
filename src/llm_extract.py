from __future__ import annotations
import json
from pathlib import Path
from ollama import Client
from pydantic import ValidationError

from . import settings
from .schemas import ExtractionResult, EntityOnlyResult, RelationOnlyResult
from .ontology import NODE_TYPES, allowed_relations_text
from .io_utils import read_json, write_json, sha256_text

client = Client(
    host=settings.OLLAMA_HOST,
    timeout=settings.LLM_TIMEOUT_SECONDS,
)

SYSTEM_SINGLE = """คุณคือ Knowledge Graph Information Extractor สำหรับเอกสารภาษาไทยด้านภัยมิจฉาชีพ

กติกาบังคับ:
1) ใช้เฉพาะข้อเท็จจริงที่มีหลักฐานอยู่ใน SOURCE TEXT เท่านั้น
2) ห้ามเติมความรู้ภายนอก ห้ามเดา ห้ามแต่งความสัมพันธ์
3) ชื่อ Entity ต้องสั้น ชัด และเป็นแนวคิดที่นำกลับมาใช้ซ้ำได้
4) ScamType = ประเภทกลโกงระดับแนวคิด, ScamCase = เหตุการณ์/ข้อความตัวอย่างเฉพาะกรณี
5) Signal = สิ่งที่เหยื่อสังเกตได้, ScamMethod = วิธีการที่มิจฉาชีพใช้
6) evidence_quote ต้องคัดข้อความสั้นที่สุดจาก SOURCE TEXT ที่รองรับรายการนั้น
7) สร้าง relation ได้เฉพาะ schema ที่กำหนด
8) ถ้าไม่มีข้อมูลที่รองรับ ให้คืน list ว่าง
9) ตอบตาม JSON schema เท่านั้น
10) เลือกเฉพาะ Entity และ Relation ที่สำคัญต่อ Knowledge Graph ไม่ต้องสกัดทุกประโยค
11) หลีกเลี่ยง Entity ซ้ำหรือชื่อที่มีความหมายเดียวกันใน chunk เดียว
12) evidence_quote ควรสั้นไม่เกิน 1-2 ประโยค
13) ต้องสร้าง JSON ให้จบสมบูรณ์ ห้ามหยุดกลาง JSON
"""

def _call(schema_cls, system: str, user: str):
    last_error = None

    for attempt in range(settings.LLM_RETRIES + 1):
        try:
            if attempt == 0:
                current_user = user
            else:
                current_user = user + f"""

IMPORTANT RETRY #{attempt}

คำตอบครั้งก่อนยาวเกินไปหรือ JSON ไม่สมบูรณ์

กติกาบังคับ:
- ตอบเฉพาะ JSON เท่านั้น
- เลือกเฉพาะ entity สำคัญสูงสุด 8 รายการ
- เลือกเฉพาะ relation สำคัญสูงสุด 10 รายการ
- evidence_quote ไม่เกิน 100 ตัวอักษร
- ห้ามสร้าง entity ซ้ำ
- ห้ามสร้าง relation ซ้ำ
- ห้ามอธิบายนอก JSON
- ห้ามเติมข้อมูลนอก SOURCE TEXT
- ถ้าข้อมูลมีจำนวนมาก ให้เลือกเฉพาะข้อที่สำคัญที่สุด
- ต้องสร้าง JSON ให้ครบสมบูรณ์ก่อนจบคำตอบ
"""

            response = client.chat(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": current_user},
                ],
                format=schema_cls.model_json_schema(),

                # สำคัญมากสำหรับ Qwen3.5
                think=False,

                options={
                    "temperature": 0,
                    "num_ctx": 16384,
                    "num_predict": 4096,
                },

                stream=False,
            )

            content = response.message.content or ""

            if not content.strip():
                done_reason = getattr(response, "done_reason", None)

                # debug เผื่อโมเดลใช้ token ไปกับ thinking
                thinking = getattr(response.message, "thinking", None)

                if thinking:
                    print(
                        f"\n[DEBUG] thinking chars = {len(thinking)}"
                    )

                raise ValueError(
                    f"Ollama returned empty response "
                    f"(done_reason={done_reason})"
                )

            return schema_cls.model_validate_json(content)

        except Exception as exc:
            last_error = exc

            if attempt < settings.LLM_RETRIES:
                print(
                    f"\n[RETRY {attempt + 1}/{settings.LLM_RETRIES}] "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                raise last_error
                
def _base_context(chunk: dict) -> str:
    return f"""METADATA
document_id: {chunk['document_id']}
page: {chunk['page']}
chapter: {chunk.get('chapter','')}
section: {chunk.get('section','')}

ALLOWED NODE TYPES
{", ".join(sorted(NODE_TYPES - {"Document", "Section", "Evidence"}))}

ALLOWED RELATION SCHEMA
{allowed_relations_text()}

SOURCE TEXT
<<<
{chunk['text']}
>>>
"""

def extract_single_pass(chunk: dict) -> ExtractionResult:
    return _call(
        ExtractionResult,
        SYSTEM_SINGLE,
        _base_context(chunk) + "\nสกัด entities และ relations ที่มีหลักฐานชัดเจน แล้วคืน JSON"
    )

def extract_two_pass(chunk: dict) -> ExtractionResult:
    ent_system = SYSTEM_SINGLE + "\nรอบนี้สกัดเฉพาะ entities เท่านั้น relations ต้องเป็น list ว่าง"
    ent = _call(
        EntityOnlyResult,
        ent_system,
        _base_context(chunk) + "\nสกัดเฉพาะ entities แล้วคืน JSON"
    )

    entities_json = json.dumps(
        [e.model_dump() for e in ent.entities],
        ensure_ascii=False
    )
    rel_system = SYSTEM_SINGLE + "\nรอบนี้สร้างเฉพาะ relations จาก entity ที่ให้มา"
    rel_user = _base_context(chunk) + f"""
CANDIDATE ENTITIES
{entities_json}

สร้าง relation เฉพาะเมื่อ source/target อยู่ใน candidate entities และมีข้อความรองรับ
"""
    rel = _call(RelationOnlyResult, rel_system, rel_user)
    return ExtractionResult(entities=ent.entities, relations=rel.relations)

def extract_chunk(chunk: dict, force: bool = False) -> dict:
    model_key = settings.LLM_MODEL.replace("/", "_").replace(":", "_")
    cache_key = sha256_text(
        f"{settings.LLM_MODEL}|{settings.EXTRACTION_MODE}|v3|{chunk['chunk_hash']}"
    )
    cache_path = settings.CACHE_DIR / model_key / f"{cache_key}.json"

    if cache_path.exists() and not force:
        return read_json(cache_path)

    if settings.EXTRACTION_MODE == "two_pass":
        result = extract_two_pass(chunk)
    else:
        result = extract_single_pass(chunk)

    payload = {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "page": chunk["page"],
        "section": chunk.get("section", ""),
        "model": settings.LLM_MODEL,
        "extraction_mode": settings.EXTRACTION_MODE,
        **result.model_dump(),
    }
    write_json(cache_path, payload)
    return payload
