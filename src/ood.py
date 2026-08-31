"""
CXR Sentinel — unsupervised OOD / image-quality flagging.

A convolutional autoencoder trained ONLY on reconstruction loss against
"No Finding" images — no disease labels touch this training loop at all,
which is what makes it genuinely unsupervised (as opposed to Phase 1's
classifier, which is supervised on finding labels).

The idea: a model trained to reconstruct normal chest X-rays will reconstruct
other normal X-rays well, and reconstruct things unlike its training
distribution poorly — badly-rotated images, wrong body part, heavy artifacts,
scanner types it's never seen. Reconstruction error becomes an anomaly score.
This is the "OOD risk" line from the original Uncertainty Engine plan, done
for real rather than asserted.

This is deliberately simple (no VAE, no adversarial training) — a plain
autoencoder is enough to demonstrate the concept and is fast enough to train
on a free Colab GPU. Swap in something fancier only if the simple version's
reconstruction errors don't separate normal from anomalous images well on
your real data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # H/2
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),            # H/4
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),            # H/8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),   # H/4
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),   # H/2
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1), nn.Sigmoid(),   # H
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(model: ConvAutoencoder, loader, epochs: int, device: str, lr: float = 1e-3):
    """
    `loader` should yield ONLY normal ("No Finding") images — filter your
    dataset for that before building this loader. No labels are used here;
    the input image is also the target.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)

            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n += images.size(0)

        epoch_loss = running_loss / n
        print(f"[ood autoencoder] epoch {epoch:02d} | reconstruction MSE {epoch_loss:.5f}")
        history.append(epoch_loss)

    return history


@torch.no_grad()
def reconstruction_error(model: ConvAutoencoder, images: torch.Tensor, device: str) -> np.ndarray:
    """Per-image mean-squared reconstruction error. Higher = more anomalous / more OOD."""
    model.eval()
    images = images.to(device)
    recon = model(images)
    error = ((recon - images) ** 2).mean(dim=(1, 2, 3))
    return error.cpu().numpy()


def fit_ood_threshold(calibration_errors: np.ndarray, percentile: float = 95.0) -> float:
    """
    Call this once, after training, on reconstruction errors from a held-out
    set of KNOWN-NORMAL images. Returns the error value at `percentile` —
    images scoring above this at inference are flagged as OOD/anomalous.
    """
    return float(np.percentile(calibration_errors, percentile))


def is_ood(error: float, threshold: float) -> bool:
    return error > threshold
