"""
CXR Sentinel — calibration.

Raw sigmoid outputs from a freshly trained classifier are usually
overconfident. Temperature scaling fixes this with a single learned scalar
per model (fit on a held-out validation set, after training is finished),
without changing the model's ranking of predictions (AUROC is unaffected).

This is Phase 1's uncertainty story: "calibrated confidence", not the full
Monte-Carlo-dropout / deep-ensembles / conformal-prediction stack from the
original plan — that's a Phase 4 add-on once the core pipeline is solid.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """Wraps a trained model; learns one temperature per output (finding)."""

    def __init__(self, num_targets: int):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(num_targets))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature.exp()
        return logits / temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 200):
        """val_logits, val_labels: (N, num_targets), collected once from a trained model in eval mode."""
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.log_temperature.exp().detach()


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Standard ECE for a single binary target: bins predictions by confidence,
    compares average confidence to actual accuracy in each bin.
    probs, labels: 1D arrays of equal length.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        bin_count = in_bin.sum()
        if bin_count == 0:
            continue
        bin_confidence = probs[in_bin].mean()
        bin_accuracy = labels[in_bin].mean()
        ece += (bin_count / n) * abs(bin_confidence - bin_accuracy)

    return float(ece)


def reliability_diagram_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Returns (bin_centers, bin_accuracies, bin_confidences, bin_counts) for plotting."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_acc, bin_conf, bin_counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        count = in_bin.sum()
        bin_centers.append((lo + hi) / 2)
        bin_counts.append(int(count))
        if count > 0:
            bin_acc.append(labels[in_bin].mean())
            bin_conf.append(probs[in_bin].mean())
        else:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)

    return (
        np.array(bin_centers),
        np.array(bin_acc),
        np.array(bin_conf),
        np.array(bin_counts),
    )
