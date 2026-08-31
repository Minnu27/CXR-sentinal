"""
Runs the ENTIRE pipeline — Phase 1 (data -> train -> Grad-CAM -> calibration),
Phase 2 (temporal/history comparison), unsupervised OOD, Phase 3 (report
drafting + claim verification, including a deliberately hallucinated claim
to prove the verifier catches it), and the RL threshold bandit — on synthetic
data. Proves every module is wired together correctly before you've
downloaded a single real X-ray. Run this first, in Colab or locally.

    python -m src.selftest
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.calibrate import TemperatureScaler, expected_calibration_error
from src.data import CXRDataset, CXRDatasetConfig, DEFAULT_TARGETS
from src.gradcam import GradCAM
from src.model import CXRClassifier
from src.train import evaluate, run_training


def make_synthetic_dataset(root: str, n_images: int = 40, image_size: int = 320):
    image_dir = os.path.join(root, "images")
    os.makedirs(image_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_images):
        arr = rng.integers(0, 255, size=(image_size, image_size, 3), dtype=np.uint8)
        fname = f"synthetic_{i:03d}.png"
        Image.fromarray(arr).save(os.path.join(image_dir, fname))

        rows.append(
            {
                "image_path": os.path.join("images", fname),
                "patient_id": f"P{i % 10:04d}",
                "study_id": f"S{i:05d}",
                "study_date": "2026-01-01",
                "cardiomegaly": int(rng.random() < 0.3),
                "pleural_effusion": int(rng.random() < 0.3),
                "lung_opacity": int(rng.random() < 0.3),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(root, "manifest.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, image_dir


def main():
    tmp_root = tempfile.mkdtemp(prefix="cxr_selftest_")
    try:
        csv_path, image_root = make_synthetic_dataset(tmp_root, n_images=40, image_size=224)

        train_cfg = CXRDatasetConfig(csv_path=csv_path, image_root=tmp_root, image_size=224, train=True)
        val_cfg = CXRDatasetConfig(csv_path=csv_path, image_root=tmp_root, image_size=224, train=False)
        train_ds = CXRDataset(train_cfg)
        val_ds = CXRDataset(val_cfg)
        print(f"[ok] datasets built: train={len(train_ds)} val={len(val_ds)}")

        image, labels, meta = train_ds[0]
        print(f"[ok] sample item: image {tuple(image.shape)}, labels {labels.tolist()}, meta keys {list(meta.keys())}")

        output_dir = os.path.join(tmp_root, "checkpoints")
        # pretrained=False here only because this sandbox's network can't reach
        # download.pytorch.org; in Colab (or anywhere with normal internet) leave
        # the default pretrained=True — ImageNet init matters a lot at this data scale.
        model, history = run_training(
            train_ds, val_ds, DEFAULT_TARGETS, output_dir, epochs=2, batch_size=8, num_workers=0, pretrained=False
        )
        print(f"[ok] training ran for {len(history)} epochs, checkpoint saved: "
              f"{os.path.exists(os.path.join(output_dir, 'best_model.pt'))}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        sample_image = image.unsqueeze(0).to(device)
        cam = GradCAM(model)
        heatmap = cam(sample_image, target_index=0)
        assert heatmap.shape == (224, 224), heatmap.shape
        print(f"[ok] grad-cam heatmap shape {heatmap.shape}, range [{heatmap.min():.3f}, {heatmap.max():.3f}]")

        import torch.nn as nn
        from torch.utils.data import DataLoader

        val_loader = DataLoader(val_ds, batch_size=8, num_workers=0)
        _, _, val_logits, val_labels = evaluate(model, val_loader, nn.BCEWithLogitsLoss(), device, DEFAULT_TARGETS)

        scaler = TemperatureScaler(num_targets=len(DEFAULT_TARGETS))
        learned_temp = scaler.fit(val_logits, val_labels)
        print(f"[ok] temperature scaling fit, learned temperatures: {learned_temp.tolist()}")

        raw_probs = torch.sigmoid(val_logits)[:, 0].numpy()
        calibrated_probs = torch.sigmoid(scaler(val_logits))[:, 0].detach().numpy()
        labels_np = val_labels[:, 0].numpy()

        ece_raw = expected_calibration_error(raw_probs, labels_np, n_bins=5)
        ece_calibrated = expected_calibration_error(calibrated_probs, labels_np, n_bins=5)
        print(f"[ok] ECE raw={ece_raw:.4f} calibrated={ece_calibrated:.4f}")

        # --- Phase 2: temporal / history comparison ---
        from src.temporal import compare_studies, pair_studies

        toy_manifest = pd.DataFrame({
            "patient_id": ["P1", "P1"], "study_id": ["s1", "s2"], "study_date": ["", ""],
        })
        pairs = pair_studies(toy_manifest)
        assert len(pairs) == 1
        prior_image, _, _ = train_ds[1]
        change_results = compare_studies(model, cam, sample_image.squeeze(0), prior_image, DEFAULT_TARGETS, device=device)
        assert len(change_results) == len(DEFAULT_TARGETS)
        print(f"[ok] Phase 2 temporal comparison ran: {[(r.finding, r.status) for r in change_results]}")

        # --- Unsupervised: OOD autoencoder ---
        from torch.utils.data import DataLoader as _DL

        from src.ood import ConvAutoencoder, fit_ood_threshold, reconstruction_error, train_autoencoder

        ae = ConvAutoencoder()
        ae_loader = _DL(torch.rand(16, 3, 64, 64), batch_size=8)
        ae_history = train_autoencoder(ae, ae_loader, epochs=2, device=device)
        assert len(ae_history) == 2
        errs = reconstruction_error(ae, torch.rand(4, 3, 64, 64), device)
        thresh = fit_ood_threshold(errs, percentile=90)
        print(f"[ok] unsupervised OOD autoencoder trained, threshold={thresh:.4f}")

        # --- Phase 3: report drafting + claim verification (incl. a bad claim) ---
        from src.claim_verify import verify_claims
        from src.report_draft import Claim, draft_report_templated

        toy_findings = [{"finding": "cardiomegaly", "probability": 0.9, "status": "worsening"}]
        toy_claims = draft_report_templated(toy_findings)
        toy_verification = verify_claims(toy_claims, toy_findings)
        assert toy_verification[0].verdict == "SUPPORTED"

        # A finding the model never even assessed — the classic hallucination case
        fake_claim = Claim("pneumothorax", "There is a large pneumothorax.", "right lung", 0.9, None)
        bad_verification = verify_claims([fake_claim], toy_findings)
        assert bad_verification[0].verdict == "UNSUPPORTED", "verifier failed to catch a claim about an unassessed finding"
        print("[ok] Phase 3 report drafting + claim verification correctly flags a hallucinated claim")

        # --- RL: threshold bandit ---
        from src.threshold_bandit import ThresholdBandit, simulate_feedback_round

        bandit = ThresholdBandit(seed=0)
        for _ in range(50):
            simulate_feedback_round(bandit, model_prob=0.9, true_reviewer_accepts=True)
        assert bandit.counts[bandit.select_threshold()] >= 0  # just confirm no crash across many pulls
        print(f"[ok] threshold bandit ran 50 feedback rounds, current best={bandit.best_threshold()}")

        print("\nALL CHECKS PASSED — full pipeline (Phase 1, 2, 3, unsupervised OOD, RL bandit) wired correctly end to end.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
