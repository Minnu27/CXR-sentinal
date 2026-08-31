# CXR Sentinel

Evidence-aware, longitudinal chest X-ray decision-support pipeline. Instead of "pneumonia: 82%", the goal
is: what's present, what changed since the prior study, where exactly, how confident the model actually is
(calibrated, not raw softmax), and when it should defer to a human instead of guessing.

This repo is also the seed of a larger project, **MediTrace AI** — a multi-document clinical evidence
platform (labs, prescriptions, radiology reports, a patient timeline, cross-document contradiction
detection, evidence-grounded Q&A) with chest X-ray imaging as one module rather than the whole product. See
[`MEDITRACE_ROADMAP.md`](MEDITRACE_ROADMAP.md) for that phased plan; everything below documents CXR
Sentinel as it stands today.

**Implemented and tested, Phase 1-4:** supervised classifier (Phase 1), longitudinal/history comparison
(Phase 2), an unsupervised OOD autoencoder, template + LLM-upgradeable report drafting with real claim
verification (Phase 3), a contextual bandit for the abstention threshold (the one place RL has a real role
here), and a Gradio demo app wired to actual model output (Phase 4). See `notebooks/CXR_Sentinel_Full.ipynb`
for the single-notebook version of all of it. `PROJECT_PLAN.md` has the full phase-by-phase checklist.

**Still intentionally out of scope:** MedSAM/segmentation, multi-model consensus, an evidence-graph
database, deep ensembles/conformal prediction. These need trained models or running infrastructure this
repo doesn't fabricate — add them deliberately, the same way Phase 2-4 got added, not all at once.

## Why scoped this way

The full "11-layer" version of this system (temporal modeling, evidence graphs, multi-model consensus,
counterfactual explanations, disagreement learning, a full MLOps safety observatory) is a multi-quarter,
small-team build. Trying to build all of it at once produces a lot of half-finished pieces and nothing
demoable. This scaffold exists so Phase 1 is *solid and demoable on its own*, with the data schema and
module boundaries already set up so Phase 2+ don't require rewriting anything.

## Quickstart

### VS Code / local terminal

```bash
./setup.sh                       # creates .venv, installs requirements
source .venv/bin/activate
python -m src.selftest           # runs the whole pipeline on synthetic data — no real data needed
```

Open the folder in VS Code, select `.venv/bin/python` as the interpreter (Cmd/Ctrl+Shift+P -> "Python:
Select Interpreter"), then use the **Run and Debug** panel — five configs are preloaded in
`.vscode/launch.json`: self-test, dataset download, train, evaluate, and "current file". Recommended
extensions (Python, Pylance, Black, Jupyter) are in `.vscode/extensions.json` — VS Code will prompt to
install them on open.

If `selftest.py` prints `ALL CHECKS PASSED`, the code is wired correctly and any issues after that are data
issues, not pipeline issues. Do this before spending a single day waiting on dataset access.

**Suggested split:** write and debug locally in VS Code against a small downloaded subset (CPU is fine for
that), then push to GitHub and pull into Colab for full-GPU training runs against the full dataset — Colab's
free T4 will train Phase 1 far faster than a local CPU.

### Colab

`git clone` your repo (after pushing it to GitHub) or upload the folder, open `notebooks/phase1_colab.ipynb`,
run top to bottom.

## Data

Every dataset gets converted into one common CSV schema (see the docstring in `src/data.py`):

```
image_path, patient_id, study_id, study_date, cardiomegaly, pleural_effusion, lung_opacity
```

`patient_id` / `study_id` / `study_date` aren't used by Phase 1's model, but they're in the schema from day
one so Phase 2 (pairing a study with its most recent prior for the same patient) doesn't require touching
this file again.

### Download NIH ChestX-ray14 now — no registration required

```bash
python -m scripts.download_nih_chestxray14 --num_batches 1    # ~9,300 images, ~3-4GB, good for local dev
python -m scripts.download_nih_chestxray14 --num_batches 12   # full 112,120 images, ~42GB — do this in Colab
```

This pulls directly from the Hugging Face mirror of NIH's own release (same files as
[nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC) — no account, no
credentialing, per NIH's usage terms), and writes `data/train.csv` + `data/val.csv` already in this repo's
schema, split **by patient** (not by image) so no patient's images leak across train/val.

**Label caveat, read before trusting numbers:** NIH ChestX-ray14 has no native "Lung Opacity" label — that's
a CheXpert-specific finding. The script maps NIH's "Infiltration" label to `lung_opacity` as an
approximation, documented in the script itself. Treat Phase 1 `lung_opacity` results trained on this data as
a pipeline-validation placeholder; swap in CheXpert Plus (or the RSNA Pneumonia Detection Challenge dataset,
which has a radiologist-defined Lung Opacity label) once access clears — same CSV schema, no code changes
needed elsewhere.

**Apply for these two in parallel, now, since access lag is the real critical path for anything beyond this
placeholder:**

| Dataset | Access | Notes |
|---|---|---|
| CheXpert Plus (Stanford AIMI) | Dataset Research Use Agreement, no human-subjects training required — historically faster | ~223K report/image pairs, 187K studies, 64.7K patients, many patients have multiple studies — usable for Phase 2 |
| MIMIC-CXR (PhysioNet) | Registration + CITI human-subjects training course + signed data use agreement — budget 1-2 weeks | 377K images, 227K studies, 65K patients, repeat visits, plus free-text reports — useful for Phase 3 (VLM) |

Both licenses are **research-use-only, non-commercial**. If this ends up shipping as an actual product
rather than staying a research/demo prototype, that's a real blocker to plan around early — either a
different data-sourcing arrangement (e.g. a hospital data partnership under a BAA) or a licensing
conversation, plus note that software which outputs findings a clinician could act on falls under FDA
Software-as-a-Medical-Device regulation in the US. Worth a five-minute conversation with whoever owns
compliance at your company before this goes past a demo, not after.

## What's implemented

- `scripts/download_nih_chestxray14.py` — downloads + converts NIH ChestX-ray14 into this repo's schema,
  with a patient-level train/val split.
- `src/data.py` — dataset schema + loading. **Note:** no horizontal flip augmentation — left/right laterality
  is clinically meaningful on a chest X-ray, flipping would corrupt the label.
- `src/model.py` — DenseNet121 backbone (ImageNet-pretrained), linear multi-label head. DenseNet121 is the
  standard baseline in CXR literature (CheXNet and most follow-ups), so your numbers are comparable to
  published results.
- `src/gradcam.py` — Grad-CAM via forward/backward hooks on the last conv block.
- `src/calibrate.py` — temperature scaling (fit post-hoc on a val set, doesn't touch AUROC) + Expected
  Calibration Error + reliability diagram data.
- `src/train.py` — training loop, per-epoch AUROC, best-checkpoint saving.
- `src/evaluate.py` — standalone script: checkpoint in, per-finding AUROC + ECE (raw vs. calibrated) +
  reliability diagram PNGs + a `report.json` out.
- `src/selftest.py` — synthetic end-to-end smoke test, described above.

## Roadmap after Phase 1

- **Phase 2 — longitudinal.** Add a `prior_study_id` join on `patient_id` + `study_date` in a new
  `src/temporal.py`. Reuse `GradCAM` on both the current and prior study, compute a probability delta and an
  IoU/overlap score between the two heatmaps, and turn that into "worsening / improving / unchanged."
  Don't build a custom temporal deep-learning model for this — the delta-based approach is enough for a
  demoable feature, and it's the actual differentiator versus a plain classifier project.
- **Phase 3 — VLM + claim verification.** A VLM drafts a sentence per finding; the Phase 1 classifiers
  independently check it (probability threshold + Grad-CAM region present); mismatches get flagged instead
  of shipped in the report.
- **Phase 4 — pick 1-2, not all of:** proper uncertainty (MC dropout / deep ensembles / conformal
  prediction) with selective abstention, multi-model consensus, an evidence graph, an error-analysis
  dashboard. Build whichever best supports the story once Phases 1-3 exist and are demoable.
