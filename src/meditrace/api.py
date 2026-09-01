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
from .models import Document, Fact
from .schemas import DocumentList, DocumentRead, DocumentStatus, FactCreate, FactRead, HealthRead
from .storage import LocalObjectStore

DISCLAIMER = "Decision-support prototype on synthetic/de-identified research data — not a diagnostic device."
ALLOWED_MEDIA_TYPES = {"application/pdf", "image/png", "image/jpeg", "text/plain", "text/csv"}
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
    return HealthRead(status="ok", database="connected", object_store="connected")


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


@app.get("/api/documents", response_model=DocumentList)
def list_documents(patient_id: str | None = None, session: Session = Depends(get_session)) -> DocumentList:
    query = select(Document).options(selectinload(Document.facts)).order_by(Document.created_at.desc())
    count_query = select(func.count(Document.id))
    if patient_id:
        query = query.where(Document.patient_id == patient_id)
        count_query = count_query.where(Document.patient_id == patient_id)
    return DocumentList(items=list(session.scalars(query)), total=session.scalar(count_query) or 0)


@app.get("/api/documents/{document_id}", response_model=DocumentRead)
def get_document(document_id: uuid.UUID, session: Session = Depends(get_session)) -> Document:
    document = session.scalar(
        select(Document).where(Document.id == document_id).options(selectinload(Document.facts))
    )
    if document is None:
        raise HTTPException(404, "Document not found")
    return document


@app.get("/api/documents/{document_id}/content")
def get_document_content(document_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")
    return Response(store.get(document.object_key), media_type=document.media_type)


@app.post("/api/documents/{document_id}/facts", response_model=FactRead, status_code=201)
def create_fact(document_id: uuid.UUID, payload: FactCreate, session: Session = Depends(get_session)) -> Fact:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")
    if payload.patient_id != document.patient_id:
        raise HTTPException(409, "Fact patient does not match document patient")
    fact = Fact(source_document_id=document_id, **payload.model_dump(mode="json"))
    session.add(fact)
    document.status = DocumentStatus.ready
    session.commit()
    return fact


frontend = Path(__file__).parent / "web"
app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
