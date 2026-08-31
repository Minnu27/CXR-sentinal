# CXR Sentinel — Full Project Plan (Phase 0 → Phase 4)

A working checklist, not a status report. Check things off as you go in Colab. Each phase ends with a
"definition of done" — don't move to the next phase until you can check that box for real, on real data,
not synthetic.

Rough timeline estimate, part-time alongside coursework: **5-8 weeks** total. Phase 2's start date depends
on when CheXpert Plus / MIMIC-CXR access clears, so start those applications *today* regardless of which
phase you're actually working on — see Phase 0.

---

## Phase 0 — Setup (do this before writing any new code)

- [ ] Push the `cxr-sentinel` repo to GitHub (private repo is fine). This is what you'll `git clone` into
      Colab, and what lets VS Code and Colab share the same code without manually re-uploading files.
- [ ] Apply for **CheXpert Plus** access (Stanford AIMI / Redivis, Dataset Research Use Agreement, no CITI
      training needed) — this unlocks Phase 2 (longitudinal) and is usually the faster of the two to clear.
- [ ] Start the **MIMIC-CXR** CITI human-subjects training course in parallel (PhysioNet credentialing,
      budget 1-2 weeks). MIMIC's free-text reports are what Phase 3 (VLM report drafting) benefits from most.
- [ ] In Colab: `Runtime -> Change runtime type -> T4 GPU`. Mount Google Drive and decide now where
      checkpoints/data will live persistently (Colab's local disk is wiped every session):
      `/content/drive/MyDrive/cxr-sentinel/`.
- [ ] Confirm licensing reality with whoever owns compliance at your company: CheXpert Plus and MIMIC-CXR
      are both **research-use-only, non-commercial** licenses. Fine for a research prototype/demo; not fine
      if this ships as a product without a separate data arrangement. Also flag that anything outputting
      findings a clinician could act on falls under FDA Software-as-a-Medical-Device territory in the US —
      doesn't block building the prototype, but should be on someone's radar before this goes past a demo.

**Definition of done:** GitHub repo exists and clones cleanly into a fresh Colab notebook. Both dataset
applications are submitted. You know where Drive-persisted checkpoints will live.

---

## Phase 1 — Core Classifier (single timepoint, 3 findings)

This is the code already in the repo (`src/data.py`, `model.py`, `gradcam.py`, `calibrate.py`, `train.py`,
`evaluate.py`, `scripts/download_nih_chestxray14.py`). This section is the *execution* checklist for it.

- [ ] `git clone` your repo into Colab, `!pip install -q -r requirements.txt`.
- [ ] Run `!python -m src.selftest`. Must print `ALL CHECKS PASSED` before you touch real data.
- [ ] Download a small NIH ChestX-ray14 subset to validate on real images while CheXpert Plus is pending:
      `!python -m scripts.download_nih_chestxray14 --num_batches 1 --out_dir /content/drive/MyDrive/cxr-sentinel/data`
- [ ] Point `configs/phase1.yaml` at that `data/train.csv` / `data/val.csv`, run a short training job
      (5 epochs) and sanity-check: does loss go down, is AUROC above 0.5 for all three findings? If AUROC
      is stuck at ~0.5, something's wrong (label mapping, data loading) — debug before scaling up.
- [ ] Download the full NIH dataset (`--num_batches 12`, do this once, save to Drive so you don't repeat
      the ~42GB download every session) and run a full training job (10-15 epochs, checkpoint each epoch to
      Drive in case the session disconnects).
- [ ] Run `src/evaluate.py` on the trained checkpoint: record per-finding AUROC, ECE before/after
      calibration, and look at the reliability diagrams — do they visually look better calibrated after
      temperature scaling?
- [ ] Spot-check Grad-CAM on 10-15 individual images per finding: does the heatmap actually land on the
      relevant anatomy (heart border for cardiomegaly, costophrenic angle for effusion)? If it's
      highlighting random corners of the image, the model may be learning shortcuts (image artifacts,
      device markers) rather than the actual finding — worth a closer look before trusting it.
- [ ] Once CheXpert Plus access clears: write a converter (same pattern as
      `scripts/download_nih_chestxray14.py`) mapping CheXpert Plus's native format into the same CSV
      schema, and retrain. CheXpert Plus has a *real* Lung Opacity label — this replaces the NIH
      "Infiltration" placeholder used above.

**Definition of done:** a checkpoint trained on CheXpert Plus (not just the NIH placeholder), with
documented per-finding AUROC, calibrated ECE, and Grad-CAM spot-checks that look anatomically sane. This is
the artifact you'd actually show your company as "Phase 1 complete."

---

## Phase 2 — Longitudinal Comparison

The actual differentiator. Requires a dataset with multiple studies per patient — CheXpert Plus works
(many patients have repeat studies); NIH's subset mostly won't (repeats are sparse and undated).

- [ ] New file: `src/temporal.py`.
  - [ ] `pair_studies(manifest_df)` — group by `patient_id`, sort by `study_date`, for every patient with
        2+ studies yield `(current_row, prior_row)` pairs (most recent prior only, for v1).
  - [ ] For each pair: run the Phase 1 classifier + `GradCAM` on both images independently.
  - [ ] Compute a **probability delta** per finding: `prob_current - prob_prior`.
  - [ ] Compute a **heatmap overlap score**: threshold each Grad-CAM heatmap into a binary "active region"
        mask, compute IoU between current and prior masks. High IoU = same anatomical area involved across
        both studies (supports "the same finding is evolving," not "a new unrelated finding appeared").
  - [ ] Combine into a status label per finding: `new` (prior negative, current positive), `resolved`
        (prior positive, current negative), `worsening` (prob delta above a threshold, region overlaps),
        `improving` (prob delta below a negative threshold), `unchanged` (small delta).
- [ ] **Registration caveat, don't skip this:** two X-rays of the same patient aren't pixel-aligned —
      different rotation, zoom, patient positioning. Naive pixel-space heatmap IoU will be noisy. For v1,
      a consistent resize + center-crop is enough to get a usable signal; full affine registration
      (aligning the two images before comparing) is a real improvement but treat it as a stretch goal, not
      a Phase 2 blocker.
- [ ] Test on 10-20 real patient pairs with known repeat studies. Manually sanity-check: do the "worsening"
      calls look plausible against the actual images? This is the step most worth spending real time on —
      it's the feature the whole project is differentiated by.
- [ ] Add a Grad-CAM diff visualization: current image, prior image, and the two heatmaps side by side,
      with the computed status label. This is your best demo asset.

**Definition of done:** `src/temporal.py` producing `{finding, current_prob, prior_prob, status,
overlap_score}` for real patient pairs, plus a handful of manually-verified before/after visualizations you
trust enough to show someone.

---

## Phase 3 — VLM Report Drafting + Claim Verification

Recommended architecture: **don't** ask a VLM to diagnose the raw image from scratch — that's where
hallucination risk is highest and hardest to check. Instead, feed the VLM your own Phase 1/2 model's
*already-computed, calibrated* findings (probability, status, region) and ask it only to draft natural
report language from that structured input. This makes verification straightforward: you're checking
whether the VLM's sentence matches the numbers you already trust, not whether it correctly diagnosed a
chest X-ray on its own.

- [ ] Pick a VLM/LLM API you have access to (this doesn't need to be a specialized medical VLM for v1 —
      you're using it for grounded language generation, not diagnosis).
- [ ] New file: `src/report_draft.py` — given a finding's structured output (probability, calibrated
      confidence, temporal status if available, rough anatomical region from Grad-CAM), construct a prompt
      and get back a **structured JSON response** (not free text) with one object per finding: `{finding,
      claim_text, location}`. Structured output sidesteps fragile text-parsing in the next step.
- [ ] New file: `src/claim_verify.py` — for each claim, check it against your own model's outputs:
      classifier probability above threshold? Grad-CAM active region present and roughly matching the
      claimed location? Temporal status (if claim mentions change) consistent with Phase 2's output?
      Tag each claim `SUPPORTED` or `UNSUPPORTED`, drop or flag unsupported ones before they'd reach a
      report.
- [ ] Run this on 10-20 real cases. Deliberately try to break it — feed it a borderline/ambiguous case and
      see whether verification correctly catches an overreaching claim.

**Definition of done:** an end-to-end function that takes an image (+ prior, if available) and returns a
draft report with each sentence tagged supported/unsupported — tested against real cases, not just the
happy path.

---

## Phase 4 — Demo Layer (pick 1-2, don't try all of these)

Everything here is optional polish. Pick what best supports what your company actually needs to see —
don't build all of it just because it's listed.

- [ ] **Interactive demo UI.** For a Colab-friendly demo, use **Gradio** (`pip install gradio`, runs
      directly in Colab with a shareable public link) rather than building the Next.js frontend from the
      original architecture doc — that's a much bigger lift than a v1 demo needs. Upload an image (+
      optional prior), show findings, Grad-CAM overlay, calibrated confidence, and the draft report with
      verification tags.
- [ ] **FastAPI endpoint.** `src/api.py` — a `/predict` route wrapping the trained model, for anyone who
      wants to call this programmatically rather than through a UI. `fastapi` + `uvicorn` are already in
      `requirements.txt`.
- [ ] **Selective prediction / abstention.** Using the calibrated confidence from Phase 1, add a threshold
      below which the system returns "confidence too low, human review required" instead of a number. Easy
      to add now that calibration already exists, and it's the single most defensible "responsible AI"
      talking point from the original plan.
- [ ] **Minimal error-analysis view.** A couple of matplotlib panels (or a Gradio tab): AUROC and ECE by
      finding, maybe broken down by image quality or AP-vs-PA if that metadata is available. Don't build
      the full multi-model-consensus / evidence-graph / safety-observatory stack from the original plan —
      pick this one thing if error analysis is what's actually useful to show.

**Definition of done:** whichever 1-2 items you picked are working end-to-end on real cases and demoable
in a single Colab session without manual setup steps.

---

## Colab operational notes (read once, save yourself pain later)

- **Sessions disconnect.** Free tier: ~12hr hard cap, and can drop after ~90min of inactivity. Checkpoint
  every epoch to Drive (`output_dir='/content/drive/MyDrive/cxr-sentinel/checkpoints'`), not just the best
  one — if the session dies mid-run you want to resume, not restart.
- **Don't re-download 42GB every session.** Point `scripts/download_nih_chestxray14.py --out_dir` at your
  Drive path once; subsequent sessions just `drive.mount()` and the data's already there.
- **Free-tier T4 has limited memory.** If you hit OOM at `batch_size=32`, drop to 16, and/or wrap the
  training step in `torch.autocast('cuda')` for mixed precision — meaningfully faster on a T4 too.
- **Don't leave a session idle mid-training just to "watch it."** Kick off training, come back later — an
  active browser tab doesn't prevent the idle-disconnect timer in the way people assume.

---

## What to actually report to your company at each checkpoint

- **After Phase 1:** "Working classifier for 3 findings, AUROC X/Y/Z, calibrated (ECE went from A to B
  after temperature scaling), Grad-CAM explanations spot-checked against real cases."
- **After Phase 2:** "System detects worsening/improving/unchanged between a patient's studies, not just a
  single snapshot — validated against N real repeat-study cases."
- **After Phase 3:** "Draft report language is generated from the model's own calibrated findings, and every
  claim is independently checked before it would reach a report — this is the anti-hallucination story."
- **After Phase 4:** a live, clickable demo — this is what actually lands in a meeting, more than any
  metric on a slide.
