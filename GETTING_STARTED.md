# Getting Started with MediTrace AI

This is a from-scratch walkthrough for a new machine: clone the repo, run the
service, upload a document, and see an evidence-linked fact come back. It
complements the shorter reference in [`README.md`](README.md) — start here if
this is your first time in the repository.

> **Reminder:** this is a decision-support prototype for synthetic and
> de-identified research data only. Never upload identifiable patient
> information or use it for clinical care.

## 1. Prerequisites

- Python 3.10+ (`python3 --version`)
- `pip` and `venv` (bundled with most Python installs)
- Git
- Optional, only for Postgres mode: Docker and Docker Compose

You do **not** need PyTorch, a GPU, or Docker to run the core MediTrace
workflow — those are only required for the legacy CXR research modules
(Section 7) and the optional Postgres database (Section 5).

## 2. Clone and create a virtual environment

```bash
git clone <this-repository-url>
cd CXR-sentinal
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

`setup.sh` automates this exact sequence plus the install in the next step,
if you'd rather run one script.

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, SQLAlchemy, the Postgres driver, and test
tooling — everything the MediTrace API and worker need. It intentionally
skips PyTorch and other ML packages; those live in `requirements-ml.txt` and
are only needed for the legacy CXR classifier notebooks (Section 7).

## 4. Run MediTrace in zero-infrastructure mode

This is the fastest path: no Docker, no external database. Open two
terminals in the project root (both with `.venv` activated).

**Terminal 1 — API server:**

```bash
uvicorn src.meditrace.api:app --reload
```

**Terminal 2 — background extraction worker:**

```bash
python -m src.meditrace.worker
```

The worker is a separate process on purpose — extraction never runs inline
in the request path. Leave both running for the rest of this guide.

Now open <http://127.0.0.1:8000> in a browser. You'll see the "Industry"
upload/review interface with the research-only disclaimer always visible.
Metadata is written to `meditrace.db` (SQLite) and source files to
`data/documents/` — both are git-ignored and safe to delete if you want a
clean slate.

## 5. Walk through the workflow

You can do this from the web UI, or the same steps via `curl`/`/docs`:

1. **Upload a document.** In the UI, upload a small text file with something
   like `HbA1c 7.2 %` in it, tagged with a synthetic patient ID (e.g.
   `SYN-1048`). Via the API:

   ```bash
   curl -F "patient_id=SYN-1048" \
        -F "file=@lab.txt;type=text/plain" \
        http://127.0.0.1:8000/api/documents
   ```

2. **Queue extraction** for the returned document ID:

   ```bash
   curl -X POST http://127.0.0.1:8000/api/documents/<id>/extract
   ```

   Check `GET /api/jobs/<job_id>` for status — the worker process you started
   in Terminal 2 picks this up and runs deterministic lab/medication/
   radiology/discharge extraction against the text.

3. **Inspect the fact.** Every extracted (or manually added) fact carries a
   page/line/quote/bounding-box evidence location back to the source
   document — nothing is displayed without it. See the example fact shape in
   `README.md`'s "Connect your database and model" section.

4. **View the timeline** for the patient:

   ```bash
   curl http://127.0.0.1:8000/api/patients/SYN-1048/timeline
   ```

   Facts are grouped by month/visit with deterministic prior-value deltas —
   no model call involved in the sorting or arithmetic.

Full interactive API documentation (all routes, request/response schemas) is
always available at <http://127.0.0.1:8000/docs> while the server is running.

## 6. Verify your setup

```bash
pytest -q
```

This runs the MediTrace upload → extract → evidence-fact → timeline
integration tests against synthetic text, using a temporary SQLite database
per test — nothing you do here touches `meditrace.db`.

## 7. Optional: Postgres instead of SQLite

If you want the service backed by a real database (closer to how it runs in
Phase 0's target deployment):

```bash
cp .env.example .env
docker compose up -d postgres
set -a; source .env; set +a
uvicorn src.meditrace.api:app --reload
```

`docker-compose.yml` starts a local Postgres 16 container with throwaway
credentials (`meditrace` / `meditrace-dev`) — fine for local development,
**never use those credentials outside an isolated dev machine.** `config.py`
normalizes a `postgresql://` URL to the installed `psycopg` driver
automatically.

## 8. Optional: connect a model gateway

By default, no source text is ever sent anywhere — extraction is fully
deterministic. If you have an OpenAI-compatible MedGemma-style endpoint and
want to try model-assisted extraction on a specific document, set in `.env`:

```bash
MODEL_ENDPOINT=<your endpoint>
MODEL_API_KEY=<your key>
MODEL_NAME=medgemma
```

Submission is always an explicit, per-document call —
`POST /api/documents/{id}/submit-to-model` — never automatic. Returned facts
are schema-validated before they're persisted.

## 9. Optional: the legacy CXR research stack

The pre-existing chest X-ray classifier code under `src/` (outside
`src/meditrace/`) is kept for reference but is **not** wired into the
MediTrace document workflow and isn't represented as clinical functionality.
If you want to explore it:

```bash
pip install -r requirements-ml.txt   # adds torch, torchvision, pandas, etc.
python -m src.selftest               # exercises the pipeline on generated images only
```

`notebooks/` and `configs/phase1.yaml` are the entry points for that
research track; `scripts/download_nih_chestxray14.py` fetches the (public)
NIH ChestX-ray14 dataset if you want real training data rather than the
self-test's generated images.

## 10. Where to go next

- [`README.md`](README.md) — condensed reference: full API table, scope
  boundaries, and the example evidence-fact JSON shape.
- [`PROJECT_PLAN.md`](PROJECT_PLAN.md) — the phase-by-phase execution
  checklist. Phase 0 (foundations) and the Phase 1–3 text workflow you just
  exercised are done; unchecked items (S3/MinIO storage, Alembic migrations,
  document-type classification, trends/contradictions, imaging) are open
  work, tracked honestly as not-yet-implemented.
- `src/meditrace/` — the FastAPI app (`api.py`), extraction logic
  (`extraction.py`), background `worker.py`, SQLAlchemy models
  (`models.py`), Pydantic schemas (`schemas.py`), and the bundled web UI
  (`web/`).
- `tests/` — start with `tests/test_meditrace_api.py` for a runnable example
  of the full upload-to-timeline loop.

If something in this guide doesn't match what you see (a moved file, a
changed command), trust the code and open an issue — this document should
track the repository, not the other way around.
