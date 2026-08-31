# MediTrace AI — Phased Execution Plan

*A build roadmap for a solo/portfolio project. Scope is calibrated for one developer working in evenings/weekends, using pretrained models instead of training anything from scratch, and public/synthetic data instead of real patient records.*

> **How this relates to CXR Sentinel:** this repo's existing code (`src/`, `PROJECT_PLAN.md`, the Gradio demo) is a standalone chest X-ray classifier — supervised classification, longitudinal comparison, calibration, Grad-CAM, and claim-verified report drafting. MediTrace AI is the larger evidence-tracing platform this roadmap describes: multi-document clinical records (labs, prescriptions, radiology reports, discharge summaries), a patient timeline, cross-document contradiction detection, and evidence-grounded Q&A, with chest X-ray imaging as one module (Phase 6 below) rather than the whole product. CXR Sentinel's classifier/Grad-CAM/calibration work is the natural seed for that Phase 6 module — it doesn't need to be rebuilt, just wired into MediTrace's document schema and timeline when that phase comes up.

## Framing decisions before Phase 0

Two calls shape everything below, and it's worth stating them up front rather than discovering them mid-build.

**Data.** This project can never touch real patient data — not because of squeamishness, but because doing so triggers HIPAA obligations and, if the "claim verification" and "contradiction detection" features are ever presented as usable in a real clinical setting, FDA Software-as-a-Medical-Device regulation. None of that is buildable solo. So the entire project runs on public and synthetic data: [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) and [MIMIC-IV](https://physionet.org/content/mimiciv/) (free via PhysioNet credentialed access — a short CITI ethics course plus a data use agreement, no institutional affiliation required), the [Indiana University Chest X-ray collection](https://openi.nlm.nih.gov/faq) (fully open, no credentialing, good for a fast start), and [Synthea](https://synthea.mitre.org/) for synthetic longitudinal patient records (labs, meds, encounters) when you need messy multi-document timelines that MIMIC doesn't give you directly. Every screen and every README should say "decision-support prototype on synthetic/de-identified research data — not a diagnostic device" — that line is doing real legal work, not just covering you.

**Model strategy.** The original brainstorm sketches OCR → layout AI → medical NER → a separate CV model → a separate VLM as distinct pipeline stages. As of this year that's more infrastructure than a solo build needs: Google's [MedGemma 1.5](https://huggingface.co/google/medgemma-1.5-4b-it) is a single open-weight multimodal model (4B params, runs on one consumer GPU or via Model Garden) that natively handles medical text extraction, EHR/lab-report parsing, *and* 2D imaging including chest X-rays and bounding-box localization. Pairing it with [CXR Foundation](https://huggingface.co/google/cxr-foundation) (pretrained chest X-ray embeddings for data-efficient or zero-shot classification, also from Google's Health AI Developer Foundations) collapses what would have been three separate model-training efforts into "call an existing model with the right prompt/embedding." That's the single biggest scope reduction available to you, and the plan below leans on it deliberately — it's what makes cross-document reasoning and evidence-linking (the actual differentiator) reachable in a reasonable timeframe instead of the imaging CV stack eating the whole project.

---

## Phase 0 — Foundations (1–2 weeks)

Goal: nothing clinical yet, just the skeleton that every later phase plugs into.

- Lock the document schema: every extracted fact carries `{test/finding, value, unit, reference_range, status, date, source_document_id, evidence_location, confidence}`. Decide this once — every phase from here on writes into it.
- Stand up the skeleton: Postgres for structured facts + documents metadata, object storage (S3-compatible, even local MinIO) for raw files, FastAPI backend, a bare Next.js (or plain React) frontend with an upload screen and a document list. No intelligence yet — just prove files go in and come back out.
- Pull down MIMIC-CXR credentialed access (start this immediately — approval can take days) and grab the Indiana University set as a no-wait fallback so Phase 1 isn't blocked on paperwork.
- Write the disclaimer/positioning language once (README + in-app banner) and reuse it everywhere.

**Cut for now:** auth beyond a single dev login, any deployment story, any model calls at all.

## Phase 1 — Single document type, end to end (2–3 weeks)

Goal: prove the full extraction → structured JSON → evidence-linked answer loop on *one* document type before spreading wide. Lab reports are the right first target — numeric, structured, low ambiguity.

- OCR/parsing: [Docling](https://github.com/docling-project/docling) (open source, self-hosted, free) for text-native and scanned PDFs. Keep AWS Textract + Comprehend Medical Insights as a documented fallback for messy scans, but don't wire it up unless Docling's output is visibly bad on your sample set — it adds an AWS dependency and cost for a portfolio project.
- Structured extraction: prompt MedGemma 1.5 4B (self-hosted via vLLM/Transformers, or Model Garden if you'd rather not manage a GPU) to turn parsed text into the Phase 0 JSON schema. Normalize test names against [LOINC](https://loinc.org/) codes — free, and it's the detail that makes this look like a real health-tech project rather than a demo.
- Evidence linking: every extracted value must carry a pointer back to page/line/bounding box in the source document, not just a document ID. This is cheap to do now and expensive to retrofit later — Docling's output preserves layout coordinates, so carry them through.
- UI: a single-document view showing the source PDF side by side with its extracted structured facts, each fact clickable back to its location in the source.

**Cut for now:** everything except lab reports. No timeline, no cross-document anything, no imaging, no LLM Q&A yet — this phase is "can I trust one extraction," full stop.

**Milestone check:** upload a MIMIC-IV or Synthea lab report PDF, get back correct structured values with working evidence links, on ~90%+ of a 20-document sample you've hand-checked yourself.

## Phase 2 — More document types, same pipe (2 weeks)

Goal: generalize Phase 1's pipeline to prescriptions, radiology reports (text), and discharge summaries — prove the schema and extraction approach hold up outside labs, not that you need four separate pipelines.

- Add document-type detection (a MedGemma classification prompt is enough at this scale — no separate classifier model needed).
- Extend the schema per type: medications get `{drug, dose, frequency, route, start_date, prescriber}`; radiology reports get `{modality, finding, severity, comparison_to_prior}`.
- Normalize medications against [RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/) the same way labs got normalized against LOINC.

**Cut for now:** imaging (pixels, not just radiology report *text*) stays out until Phase 6. Discharge summaries can be treated as "prescriptions + free text" rather than getting bespoke handling.

## Phase 3 — Timeline engine (1–2 weeks)

Goal: this is the first genuinely differentiating feature from the brainstorm doc — turn a pile of independently-extracted documents into one chronological view per patient.

- Group all extracted facts by patient ID and sort by date; render as the timeline view (the doc's own mockup — grouped by month/visit, with up/down arrows against the prior value of the same test).
- This is mostly a query + UI problem now that Phase 1–2 give you clean structured facts with dates — resist the urge to add ML here. A sort and a group-by is the whole feature.

**Milestone check:** load a Synthea patient with 6–12 months of synthetic visits and confirm the timeline reads the way a clinician skimming a chart would expect.

## Phase 4 — Cross-document reasoning & contradiction detection (2–3 weeks)

Goal: the signature feature. Two sub-features, build trend detection first since it's a strict subset of contradiction detection.

- **Trend detection:** for any test appearing more than once for a patient, compute direction and magnitude of change, flag clinically notable jumps (start with a hardcoded threshold table for common labs — HbA1c, creatinine, eGFR — rather than trying to learn thresholds).
- **Contradiction detection:** compare facts of the same *type* (same drug, same allergy, same diagnosis) across documents for direct conflicts (different Metformin doses in two documents; "no known allergies" vs. an earlier "penicillin allergy" entry). Do this as a structured comparison over your Phase 0 schema, not by asking an LLM to "spot contradictions" freeform — the LLM proposes candidate comparisons, code does the actual conflict check, matching the doc's own "don't let the LLM invent an answer" principle.
- Surface both as the flagged-card UI from the brainstorm doc, always citing the two conflicting source documents.

**Cut for now:** don't try to resolve contradictions automatically or rank which record is "more correct" — the whole point is surfacing for human review, so resist scope creep toward auto-resolution.

## Phase 5 — Evidence-grounded Q&A (2 weeks)

Goal: let a user ask "what was my latest HbA1c" in plain language and get an answer that cites its source, instead of browsing the timeline manually.

- Standard RAG over your structured facts (not raw document text — you already have clean JSON, querying that is more reliable than re-parsing prose): retrieve relevant facts for the question, have MedGemma (or a cheaper text-only model, since this step is pure language, not vision) compose an answer strictly from the retrieved facts, and require every claim in the answer to trace to a specific fact ID.
- Claim verification pass: before showing an LLM-composed answer, check each sentence against the retrieved facts it claims to be based on; if a sentence can't be matched to supporting evidence, suppress it or mark it "not enough evidence" rather than showing it. This is the architecture the brainstorm doc lays out (retrieval → extraction → reasoning → verification → answer) and it's what separates this from "upload PDF, ChatGPT explains it."

**Milestone check:** ask 15–20 hand-written questions against a loaded patient's records; every answer should either cite correct evidence or explicitly decline.

## Phase 6 — Imaging module (2–3 weeks)

Goal: add chest X-ray understanding as one module, not the center of the project — deliberately sequenced *after* the reasoning engine works, since imaging was the brainstorm doc's explicit "don't let this eat the project" warning.

- Use CXR Foundation embeddings for finding classification against MIMIC-CXR's labeled findings (data-efficient route — you don't need to train a CNN from scratch, you're training a small classifier head on pretrained embeddings, or using its zero-shot text-prompt mode directly).
- Use MedGemma's vision capability for a natural-language description of a chest X-ray and to compare against a prior image ("increased compared with previous scan") — this is a documented MedGemma capability, not a custom feature you'd need to build.
- Wire imaging findings into the same Phase 0 schema and timeline so a chest X-ray finding shows up alongside lab trends for the same date, not in a separate UI.
- This is where CXR Sentinel's existing classifier, Grad-CAM, calibration, and longitudinal-comparison code (see `PROJECT_PLAN.md` and `src/`) plugs in directly — the delta/overlap-score approach in `src/temporal.py` (Phase 2 of that plan) is the same "worsening/improving/unchanged" signal this phase needs, just re-emitted into MediTrace's document schema instead of a standalone Gradio demo.

**Cut for now:** every other modality (CT, MRI, pathology, dermatology, retinal) stays as "the architecture supports plugging this in" rather than being built — say that explicitly in the portfolio writeup rather than trying to cover it.

## Phase 7 — Knowledge graph & document reliability (1–2 weeks, optional)

Goal: two "nice to have, genuinely differentiating if you have time" features from the brainstorm doc, sized down for solo scope.

- Knowledge graph: don't stand up Neo4j for this. A `facts` table with `(patient, entity_type, entity_id, related_entity_id, relationship, date)` in the Postgres you already have supports the doc's example query ("documents related to kidney function after medication X started") via SQL joins. Reach for a real graph database only if traversal queries get painful — likely won't happen at portfolio scale.
- Document reliability tiers: a static lookup table (signed pathology/radiology reports = high, scanned handwritten notes = medium, patient-entered = low) attached to each document, surfaced in the UI and factored into which evidence claim verification prefers when sources conflict.

**Cut entirely if time-constrained:** duplicate/version detection (brainstorm section 14) — real but lowest-leverage feature for a portfolio audience.

## Phase 8 — Audit trail & polish (1 week)

Goal: the detail that makes this read as "built by someone who understands healthcare software," not the differentiator itself.

- Log every query/answer/document-view event (`user, action, documents_accessed, model_version, evidence_ids, timestamp, outcome`) to its own table, surfaced as a simple audit log view.
- Add the "not enough evidence" and confidence-tier displays consistently across every screen, not just Q&A.

## Phase 9 — Packaging (ongoing)

Not a build phase — a presentation one. Record a walkthrough that demonstrates the actual differentiator in under 3 minutes: upload 3–4 conflicting synthetic documents for one patient, show the timeline build itself, show a contradiction get flagged with both sources cited, ask one evidence-grounded question. That sequence *is* the pitch from the brainstorm doc — lead with it over a feature list.

---

## What ships in v1 vs. what's aspirational

| | Included | Explicitly deferred |
|---|---|---|
| **v1 (Phases 0–5, ~10–13 weeks part-time)** | Lab reports, prescriptions, radiology report text, discharge summaries; timeline; trend + contradiction detection; evidence-grounded Q&A | Imaging pixels, knowledge graph, audit system, duplicate detection |
| **v1.5 (Phase 6, +2–3 weeks)** | Chest X-ray findings via CXR Foundation + MedGemma vision | CT/MRI/pathology/dermatology/retinal |
| **Stretch (Phases 7–8)** | Lightweight knowledge graph, document reliability tiers, audit trail | Neo4j, auto-resolution of contradictions, any real-time/multi-user features |

If the timeline slips, the phase to cut is 6 (imaging) before touching 4 or 5 — trend/contradiction detection and evidence-grounded Q&A are the actual differentiators the brainstorm doc identifies; a chest X-ray classifier alone is what "thousands of projects already do."

## One practical aside

You already have a finished design system in this project space ("Industry" — steel-blue, blueprint-style cards with corner registration marks, Barlow/Barlow Condensed) sitting unused for this build. Its clinical, technical-drawing aesthetic — hairline borders, evidence "cards" with corner marks, duotoned imagery — would suit a medical evidence-tracing product's visual language unusually well if you want a head start on Phase 0's frontend instead of designing MediTrace's UI from scratch.

---

### Sources consulted
- [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) — Hugging Face
- [google/cxr-foundation](https://huggingface.co/google/cxr-foundation) — Hugging Face
- [Best AI for Clinical Document Parsing (2026)](https://www.llamaindex.ai/insights/best-ai-for-clinical-document-parsing) — LlamaIndex
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) and [MIMIC-IV](https://physionet.org/content/mimiciv/) — PhysioNet
- [Synthea downloads](https://synthea.mitre.org/downloads) — MITRE
- [Docling](https://github.com/docling-project/docling) — GitHub (docling-project)
- [Indiana University Chest X-ray collection](https://openi.nlm.nih.gov/faq) — Open-i, National Library of Medicine
