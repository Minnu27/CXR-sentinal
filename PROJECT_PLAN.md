# MediTrace AI — Execution Checklist

This checklist turns the product roadmap into repository-sized increments. A checked item is implemented and testable here; unchecked work is not presented as shipped.

## Phase 0 — Foundations (current)

- [x] Canonical document and evidence-linked fact schemas.
- [x] FastAPI ingestion, document listing, source download, and fact creation.
- [x] SQLAlchemy persistence with Postgres configuration and local SQLite fallback.
- [x] Durable local object-store adapter with atomic writes.
- [x] Upload and source-register web interface using the Industry visual language.
- [x] Persistent in-product and README research-use disclaimer.
- [x] Synthetic API integration test for the complete upload → fact → evidence loop.
- [ ] S3/MinIO adapter (local filesystem is the intentional first adapter).
- [ ] Alembic migrations before the schema begins evolving.

**Gate:** files go in, can be retrieved byte-for-byte, metadata persists, and facts cannot exist without document-scoped evidence.

## Phase 1 — Lab extraction

- [ ] Introduce a background job boundary; never run parsing in the request process.
- [ ] Parse text-native and scanned PDF lab reports with Docling.
- [ ] Preserve page, line, and bounding-box coordinates through extraction.
- [ ] Add a model-provider interface and MedGemma structured-output implementation.
- [ ] Normalize lab tests against a versioned LOINC reference subset.
- [ ] Add the split document/evidence review UI.
- [ ] Hand-check a fixed 20-document synthetic/de-identified evaluation set and publish the rubric.

## Phase 2 — Additional text documents

- [ ] Document-type classification with explicit `unknown` behavior.
- [ ] Medication schema and RxNorm normalization.
- [ ] Radiology report text schema.
- [ ] Discharge summary handling through shared fact types.

## Phase 3 — Timeline

- [ ] Patient-scoped chronological query and visit/month grouping.
- [ ] Deterministic prior-value deltas; no model call for sorting or arithmetic.
- [ ] Timeline UI with source links on every entry.

## Phase 4 — Trends and contradictions

- [ ] Versioned threshold table for a small, reviewed set of common lab trends.
- [ ] Candidate grouping followed by deterministic conflict checks.
- [ ] Medication-dose and allergy contradiction rules.
- [ ] Flag cards cite both sources and never auto-resolve a conflict.

## Phase 5 — Evidence-grounded Q&A (v1 gate)

- [ ] Retrieve from structured facts, not unbounded raw prose.
- [ ] Require fact IDs for every generated claim.
- [ ] Deterministic verification/suppression pass with “not enough evidence” fallback.
- [ ] Fixed 15–20 question evaluation set in CI.

## Phase 6 — Chest X-ray module (v1.5)

- [ ] CXR Foundation embedding adapter and separately evaluated classifier head.
- [ ] MedGemma vision adapter with model/version provenance.
- [ ] Imaging facts use the same evidence and timeline contracts.
- [ ] No CT, MRI, pathology, dermatology, or retinal claims in this phase.

## Phases 7–9 — Optional differentiation and packaging

- [ ] Relational fact graph only when a demonstrated query requires it.
- [ ] Static, visible document reliability tiers.
- [ ] Append-only query, answer, view, evidence, outcome, and model-version audit events.
- [ ] Consistent confidence and insufficient-evidence displays.
- [ ] Three-minute walkthrough using conflicting synthetic sources only.

## Non-negotiable release rules

1. Never ingest real identifiable patient records.
2. Never represent an unchecked phase as implemented.
3. Every displayed fact must retain a source document and precise evidence location.
4. Never rank conflicting records or silently choose a winner.
5. Imaging is cut before trends, contradictions, or grounded answers if schedule slips.
