"""
CXR Sentinel — Phase 1 training loop.

Kept deliberately plain: one optimizer, one loss, per-epoch AUROC on a val
split, best-checkpoint saving. Get this correct and boring before adding
anything from Phase 2+.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.model import CXRClassifier


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels, _meta in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, target_names: list[str]):
    model.eval()
    running_loss = 0.0
    all_logits, all_labels = [], []

    for images, labels, _meta in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    labels_np = all_labels.numpy()

    per_class_auroc = {}
    for i, name in enumerate(target_names):
        # AUROC is undefined if a validation split has only one class present;
        # this happens easily on small subsets, so guard instead of crashing.
        if len(set(labels_np[:, i])) < 2:
            per_class_auroc[name] = float("nan")
        else:
            per_class_auroc[name] = roc_auc_score(labels_np[:, i], probs[:, i])

    val_loss = running_loss / len(loader.dataset)
    return val_loss, per_class_auroc, all_logits, all_labels


def run_training(
    train_dataset,
    val_dataset,
    target_names: list[str],
    output_dir: str,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    num_workers: int = 2,
    device: str | None = None,
    pretrained: bool = True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = CXRClassifier(num_targets=len(target_names), pretrained=pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_mean_auroc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, per_class_auroc, _, _ = evaluate(model, val_loader, criterion, device, target_names)

        valid_aurocs = [v for v in per_class_auroc.values() if v == v]  # drop NaNs
        mean_auroc = sum(valid_aurocs) / len(valid_aurocs) if valid_aurocs else float("nan")

        print(
            f"epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
            f"| mean_auroc {mean_auroc:.4f} | per_class {per_class_auroc}"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "mean_auroc": mean_auroc}
        )

        if mean_auroc == mean_auroc and mean_auroc > best_mean_auroc:  # mean_auroc == mean_auroc filters NaN
            best_mean_auroc = mean_auroc
            torch.save(
                {"model_state_dict": model.state_dict(), "target_names": target_names, "epoch": epoch},
                os.path.join(output_dir, "best_model.pt"),
            )

    return model, history
