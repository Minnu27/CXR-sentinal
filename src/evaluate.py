"""
CXR Sentinel — Phase 1 evaluation.

Run after training to get the numbers that actually go in a portfolio
writeup: per-finding AUROC, ECE before/after calibration, and reliability
diagrams. This is also where "the model should refuse" gets decided later
(Phase 4 selective prediction) — the ECE/confidence numbers computed here
are what that threshold would be tuned against.

Usage:
    python -m src.evaluate --checkpoint checkpoints/best_model.pt \
        --csv data/val.csv --image_root data/images --out_dir reports/
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.calibrate import TemperatureScaler, expected_calibration_error, reliability_diagram_data
from src.data import CXRDataset, CXRDatasetConfig
from src.model import CXRClassifier
from src.train import evaluate as run_eval


def plot_reliability_diagram(probs, labels, name: str, out_path: str, n_bins: int = 10):
    centers, acc, conf, counts = reliability_diagram_data(probs, labels, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.bar(centers, acc, width=1.0 / n_bins, edgecolor="black", alpha=0.7, label="accuracy in bin")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability diagram — {name}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--out_dir", default="reports")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    target_names = ckpt["target_names"]

    model = CXRClassifier(num_targets=len(target_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cfg = CXRDatasetConfig(csv_path=args.csv, image_root=args.image_root, image_size=args.image_size, train=False)
    dataset = CXRDataset(cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    _, per_class_auroc, logits, labels = run_eval(model, loader, nn.BCEWithLogitsLoss(), device, target_names)

    scaler = TemperatureScaler(num_targets=len(target_names))
    scaler.fit(logits, labels)

    report = {"per_class_auroc": per_class_auroc, "per_class": {}}

    for i, name in enumerate(target_names):
        raw_probs = torch.sigmoid(logits)[:, i].numpy()
        calibrated_probs = torch.sigmoid(scaler(logits))[:, i].detach().numpy()
        labels_np = labels[:, i].numpy()

        ece_raw = expected_calibration_error(raw_probs, labels_np)
        ece_cal = expected_calibration_error(calibrated_probs, labels_np)

        plot_reliability_diagram(raw_probs, labels_np, f"{name} (raw)", os.path.join(args.out_dir, f"{name}_raw.png"))
        plot_reliability_diagram(
            calibrated_probs, labels_np, f"{name} (calibrated)", os.path.join(args.out_dir, f"{name}_calibrated.png")
        )

        report["per_class"][name] = {
            "auroc": per_class_auroc[name],
            "ece_raw": ece_raw,
            "ece_calibrated": ece_cal,
            "temperature": scaler.log_temperature.exp()[i].item(),
        }
        print(f"{name}: AUROC={per_class_auroc[name]:.3f}  ECE raw={ece_raw:.3f}  ECE calibrated={ece_cal:.3f}")

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    torch.save(scaler.state_dict(), os.path.join(args.out_dir, "temperature_scaler.pt"))
    print(f"\nSaved report + reliability diagrams to {args.out_dir}/")


if __name__ == "__main__":
    main()
