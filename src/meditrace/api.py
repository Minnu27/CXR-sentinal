from __future__ import annotations

from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from .config import get_settings
from .database import create_schema, get_session
from .extraction import HttpModelProvider, extract_text
from .models import Document, ExtractionJob, Fact
from .schemas import (
    DocumentList,
    DocumentRead,
    DocumentStatus,
    ExtractionJobRead,
    FactCreate,
    FactRead,
    HealthRead,
    ModelSubmissionRead,
    TimelineEntry,
    TimelineRead,
)
from .storage import LocalObjectStore

DISCLAIMER = "Decision-support prototype on synthetic/de-identified research data — not a diagnostic device."
ALLOWED_MEDIA_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "text/plain",
    "text/csv",
}
settings = get_settings()
store = LocalObjectStore(settings.object_store_path)


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_schema()
    yield


app = FastAPI(
    title="MediTrace AI",
    version="0.1.0",
    description=f"Evidence-first clinical document workspace. {DISCLAIMER}",
    lifespan=lifespan,
)


@app.get("/api/health", response_model=HealthRead)
def health() -> HealthRead:
    return HealthRead(
        status="ok",
        database="connected",
        object_store="connected",
        model="configured" if settings.model_endpoint else "not_configured",
    )


@app.post("/api/documents", response_model=DocumentRead, status_code=201)
async def upload_document(
    patient_id: str = Form(min_length=1, max_length=128),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> Document:
    media_type = file.content_type or "application/octet-stream"
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(415, f"Unsupported media type: {media_type}")
    content = await file.read(settings.max_upload_bytes + 1)
    if not content:
        raise HTTPException(400, "The uploaded document is empty")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, "Document exceeds the upload limit")

    document_id = uuid.uuid4()
    safe_name = Path(file.filename or "document").name
    object_key = f"{patient_id}/{document_id}/{safe_name}"
    store.put(object_key, content)
    document = Document(
        id=document_id,
        patient_id=patient_id,
        filename=safe_name,
        media_type=media_type,
        object_key=object_key,
        size_bytes=len(content),
        sha256=sha256(content).hexdigest(),
        status=DocumentStatus.uploaded,
    )
    try:
        session.add(document)
        session.commit()
    except Exception:
        session.rollback()
        store.delete(object_key)
        raise
    return document


@app.post(
    "/api/documents/{document_id}/extract",
    response_model=ExtractionJobRead,
    status_code=202,
)
def enqueue_extraction(
    document_id: uuid.UUID, session: Session = Depends(get_session)
) -> ExtractionJob:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")
    existing = session.scalar(
        select(ExtractionJob).where(
            ExtractionJob.document_id == document_id,
            ExtractionJob.status.in_(("queued", "processing")),
        )
    )
    if existing:
        return existing
    job = ExtractionJob(document_id=document_id)
    document.status = DocumentStatus.processing
    session.add(job)
    session.commit()
    return job


@app.get("/api/jobs/{job_id}", response_model=ExtractionJobRead)
def get_job(
    job_id: uuid.UUID, session: Session = Depends(get_session)
) -> ExtractionJob:
    job = session.get(ExtractionJob, job_id)
    if job is None:
        raise HTTPException(404, "Extraction job not found")
    return job


@app.post(
    "/api/documents/{document_id}/submit-to-model", response_model=ModelSubmissionRead
)
def submit_to_model(
    document_id: uuid.UUID, session: Session = Depends(get_session)
) -> ModelSubmissionRead:
    """Send a source to the configured gateway and persist only schema-valid facts."""
    if not settings.model_endpoint:
        raise HTTPException(503, "MODEL_ENDPOINT is not configured")
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")
    text = extract_text(store.get(document.object_key), document.media_type)
    provider = HttpModelProvider(
        settings.model_endpoint, settings.model_name, settings.model_api_key
    )
    try:
        facts = []
        for item in provider.extract(
            {
                "patient_id": document.patient_id,
                "document_id": str(document.id),
                "document_type": document.document_type.value,
                "text": text,
            }
        ):
            item["patient_id"] = document.patient_id
            validated = FactCreate.model_validate(item)
            facts.append(Fact(source_document_id=document.id, **validated.model_dump()))
        session.add_all(facts)
        document.status = DocumentStatus.ready
        session.commit()
    except Exception as exc:
        session.rollback()
        raise HTTPException(502, f"Model submission failed: {exc}") from exc
    return ModelSubmissionRead(
        document_id=document.id,
        provider=provider.name,
        accepted=True,
        facts_created=len(facts),
        message="Validated model facts persisted",
    )


@app.get("/api/documents", response_model=DocumentList)
def list_documents(
    patient_id: str | None = None, session: Session = Depends(get_session)
) -> DocumentList:
    query = (
        select(Document)
        .options(selectinload(Document.facts))
        .order_by(Document.created_at.desc())
    )
    count_query = select(func.count(Document.id))
    if patient_id:
        query = query.where(Document.patient_id == patient_id)
        count_query = count_query.where(Document.patient_id == patient_id)
    return DocumentList(
        items=list(session.scalars(query)), total=session.scalar(count_query) or 0
    )


@app.get("/api/documents/{document_id}", response_model=DocumentRead)
def get_document(
    document_id: uuid.UUID, session: Session = Depends(get_session)
) -> Document:
    document = session.scalar(
        select(Document)
        .where(Document.id == document_id)
        .options(selectinload(Document.facts))
    )
    if document is None:
        raise HTTPException(404, "Document not found")
    return document


@app.get("/api/documents/{document_id}/content")
def get_document_content(
    document_id: uuid.UUID, session: Session = Depends(get_session)
) -> Response:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")
    return Response(store.get(document.object_key), media_type=document.media_type)


@app.post(
    "/api/documents/{document_id}/facts", response_model=FactRead, status_code=201
)
def create_fact(
    document_id: uuid.UUID, payload: FactCreate, session: Session = Depends(get_session)
) -> Fact:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")
    if payload.patient_id != document.patient_id:
        raise HTTPException(409, "Fact patient does not match document patient")
    fact = Fact(source_document_id=document_id, **payload.model_dump())
    session.add(fact)
    document.status = DocumentStatus.ready
    session.commit()
    return fact


@app.get("/api/patients/{patient_id}/timeline", response_model=TimelineRead)
def patient_timeline(
    patient_id: str, group_by: str = "month", session: Session = Depends(get_session)
) -> TimelineRead:
    if group_by not in {"month", "visit"}:
        raise HTTPException(422, "group_by must be month or visit")
    rows = session.execute(
        select(Fact, Document)
        .join(Document, Fact.source_document_id == Document.id)
        .where(Fact.patient_id == patient_id)
        .order_by(Fact.observed_date, Fact.created_at)
    ).all()
    groups: dict[str, list[TimelineEntry]] = {}
    prior: dict[tuple[str, str | None], Fact] = {}
    for fact, document in rows:
        key = (
            fact.observed_date.strftime("%Y-%m")
            if group_by == "month"
            else fact.observed_date.isoformat()
        )
        comparison = prior.get((fact.test_or_finding.lower(), fact.unit))
        delta = None
        if comparison and fact.value is not None and comparison.value is not None:
            try:
                delta = round(float(fact.value) - float(comparison.value), 6)
            except ValueError:
                pass
        entry = TimelineEntry(
            id=fact.id,
            observed_date=fact.observed_date,
            fact_type=fact.fact_type,
            test_or_finding=fact.test_or_finding,
            value=fact.value,
            unit=fact.unit,
            status=fact.status,
            source_document_id=document.id,
            source_filename=document.filename,
            evidence_location=fact.evidence_location,
            confidence=fact.confidence,
            details=fact.details or {},
            prior_value=comparison.value if comparison else None,
            numeric_delta=delta,
        )
        groups.setdefault(key, []).append(entry)
        prior[(fact.test_or_finding.lower(), fact.unit)] = fact
    return TimelineRead(
        patient_id=patient_id,
        groups=dict(reversed(list(groups.items()))),
        total=len(rows),
    )


frontend = Path(__file__).parent / "web"
app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
