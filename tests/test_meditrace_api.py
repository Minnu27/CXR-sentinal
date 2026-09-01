from __future__ import annotations

import importlib
import os

from fastapi.testclient import TestClient


def build_client(tmp_path):
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'test.db'}"
    os.environ["OBJECT_STORE_PATH"] = str(tmp_path / "objects")
    from src.meditrace import config

    config.get_settings.cache_clear()
    import src.meditrace.database as database
    import src.meditrace.api as api

    importlib.reload(database)
    api = importlib.reload(api)
    return TestClient(api.app)


def test_upload_list_download_and_evidence_fact(tmp_path):
    with build_client(tmp_path) as client:
        upload = client.post(
            "/api/documents",
            data={"patient_id": "SYN-1048"},
            files={"file": ("lab.txt", b"HbA1c 7.2 %", "text/plain")},
        )
        assert upload.status_code == 201
        document = upload.json()
        assert document["sha256"]
        assert document["status"] == "uploaded"

        listing = client.get("/api/documents").json()
        assert listing["total"] == 1
        assert listing["items"][0]["patient_id"] == "SYN-1048"

        content = client.get(f"/api/documents/{document['id']}/content")
        assert content.content == b"HbA1c 7.2 %"

        fact = client.post(
            f"/api/documents/{document['id']}/facts",
            json={
                "patient_id": "SYN-1048",
                "fact_type": "lab",
                "test_or_finding": "HbA1c",
                "normalized_code": "4548-4",
                "value": "7.2",
                "unit": "%",
                "reference_range": "4.0-5.6",
                "status": "high",
                "observed_date": "2026-08-30",
                "evidence_location": {
                    "page": 1,
                    "line_start": 1,
                    "line_end": 1,
                    "quote": "HbA1c 7.2 %",
                },
                "confidence": 0.98,
            },
        )
        assert fact.status_code == 201
        assert fact.json()["source_document_id"] == document["id"]


def test_rejects_unsupported_media_type(tmp_path):
    with build_client(tmp_path) as client:
        rejected = client.post(
            "/api/documents",
            data={"patient_id": "SYN-1"},
            files={"file": ("bad.exe", b"x", "application/x-msdownload")},
        )
        assert rejected.status_code == 415


def test_extraction_queue_and_deterministic_timeline(tmp_path):
    with build_client(tmp_path) as client:
        document = client.post(
            "/api/documents",
            data={"patient_id": "SYN-TIME"},
            files={
                "file": (
                    "lab.txt",
                    b"Laboratory report\nHbA1c 7.2 % high",
                    "text/plain",
                )
            },
        ).json()
        queued = client.post(f"/api/documents/{document['id']}/extract")
        assert queued.status_code == 202
        assert queued.json()["status"] == "queued"
        assert client.get(f"/api/jobs/{queued.json()['id']}").status_code == 200

        for observed_date, value in (("2026-07-01", "7.0"), ("2026-08-01", "7.5")):
            result = client.post(
                f"/api/documents/{document['id']}/facts",
                json={
                    "patient_id": "SYN-TIME",
                    "fact_type": "lab",
                    "test_or_finding": "HbA1c",
                    "value": value,
                    "unit": "%",
                    "observed_date": observed_date,
                    "evidence_location": {"page": 1, "line_start": 2, "quote": "HbA1c"},
                    "confidence": 0.9,
                },
            )
            assert result.status_code == 201
        timeline = client.get("/api/patients/SYN-TIME/timeline").json()
        assert timeline["total"] == 2
        assert timeline["groups"]["2026-08"][0]["numeric_delta"] == 0.5
        assert timeline["groups"]["2026-08"][0]["source_document_id"] == document["id"]
