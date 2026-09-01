from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(str, Enum):
    uploaded = "uploaded"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class DocumentType(str, Enum):
    lab_report = "lab_report"
    medication_list = "medication_list"
    radiology_report = "radiology_report"
    discharge_summary = "discharge_summary"
    unknown = "unknown"


class BoundingBox(BaseModel):
    page: int = Field(ge=1)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class EvidenceLocation(BaseModel):
    page: int = Field(ge=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    bounding_box: BoundingBox | None = None
    quote: str | None = Field(default=None, max_length=500)


class FactCreate(BaseModel):
    patient_id: str = Field(min_length=1, max_length=128)
    fact_type: str = Field(min_length=1, max_length=64)
    test_or_finding: str = Field(min_length=1, max_length=255)
    normalized_code: str | None = Field(default=None, max_length=64)
    value: str | None = Field(default=None, max_length=512)
    unit: str | None = Field(default=None, max_length=64)
    reference_range: str | None = Field(default=None, max_length=128)
    status: str | None = Field(default=None, max_length=64)
    observed_date: date
    evidence_location: EvidenceLocation
    confidence: float = Field(ge=0, le=1)
    details: dict = Field(default_factory=dict)


class FactRead(FactCreate):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    source_document_id: UUID
    created_at: datetime


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    patient_id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    status: DocumentStatus
    document_type: DocumentType = DocumentType.unknown
    extraction_error: str | None = None
    created_at: datetime
    facts: list[FactRead] = []


class DocumentList(BaseModel):
    items: list[DocumentRead]
    total: int


class HealthRead(BaseModel):
    status: str
    database: str
    object_store: str
    model: str


class ExtractionJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    document_id: UUID
    status: str
    attempts: int
    error: str | None
    created_at: datetime
    completed_at: datetime | None


class ModelSubmissionRead(BaseModel):
    document_id: UUID
    provider: str
    accepted: bool
    facts_created: int
    message: str


class TimelineEntry(BaseModel):
    id: UUID
    observed_date: date
    fact_type: str
    test_or_finding: str
    value: str | None
    unit: str | None
    status: str | None
    source_document_id: UUID
    source_filename: str
    evidence_location: EvidenceLocation
    confidence: float
    details: dict = Field(default_factory=dict)
    prior_value: str | None = None
    numeric_delta: float | None = None


class TimelineRead(BaseModel):
    patient_id: str
    groups: dict[str, list[TimelineEntry]]
    total: int
