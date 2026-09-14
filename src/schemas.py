from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

NodeType = Literal[
    "Document", "Section", "ScamType", "ScamCase", "ScamMethod",
    "Signal", "Information", "Risk", "Action", "Organization",
    "Channel", "Contact", "Law", "Statistic", "Evidence"
]

class ExtractedEntity(BaseModel):
    type: NodeType
    name: str = Field(min_length=1)
    evidence_quote: str = Field(
        default="",
        description="ข้อความสั้น ๆ จาก source ที่รองรับ entity นี้ ห้ามแต่งเพิ่ม"
    )

class ExtractedRelation(BaseModel):
    source_type: NodeType
    source_name: str
    type: str
    target_type: NodeType
    target_name: str
    evidence_quote: str = Field(
        default="",
        description="ข้อความสั้น ๆ จาก source ที่รองรับความสัมพันธ์นี้ ห้ามแต่งเพิ่ม"
    )

class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []

class EntityOnlyResult(BaseModel):
    entities: list[ExtractedEntity] = []

class RelationOnlyResult(BaseModel):
    relations: list[ExtractedRelation] = []
