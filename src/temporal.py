"""
CXR Sentinel — Phase 2: longitudinal comparison ("old history retrieval").

Given a manifest with repeat studies per patient, this pairs each patient's
current study with their most recent prior, runs the trained classifier +
Grad-CAM on both, and turns the difference into a status label a radiologist
actually cares about: new / worsening / improving / resolved / unchanged.

This is real, not templated — every number here comes from an actual forward
pass through your trained model on both images, not from a lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def pair_studies(manifest_df, patient_col: str = "patient_id", date_col: str = "study_date", id_col: str = "study_id"):
    """
    Groups by patient, orders studies chronologically, and yields the most
    recent (current, prior) pair per patient with 2+ studies.

    Falls back to ordering by `id_col` if `date_col` is missing/empty for a
    patient (NIH's subset doesn't carry real dates — study_id, built from
    Image Index, is at least a stable tiebreaker, though it isn't a true
    chronological signal. CheXpert Plus / MIMIC-CXR do carry real dates, so
    this fallback stops mattering once you're on either of those.)
    """
    df = manifest_df.copy()
    has_dates = date_col in df.columns and df[date_col].astype(str).str.strip().ne("").any()
    sort_col = date_col if has_dates else id_col

    pairs = []
    for patient_id, group in df.groupby(patient_col):
        if len(group) < 2:
            continue
        group = group.sort_values(sort_col)
        current_row = group.iloc[-1]
        prior_row = group.iloc[-2]
        pairs.append({"patient_id": patient_id, "current": current_row, "prior": prior_row})
    return pairs


def compute_heatmap_iou(heatmap_a: np.ndarray, heatmap_b: np.ndarray, threshold: float = 0.5) -> float:
    """IoU between two Grad-CAM heatmaps after binarizing each at `threshold`."""
    mask_a = heatmap_a > threshold
    mask_b = heatmap_b > threshold

    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0  # neither heatmap has an active region — nothing to overlap
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection / union)


@dataclass
class ChangeResult:
    finding: str
    prob_current: float
    prob_prior: float
    delta: float
    heatmap_iou: float
    status: str


def classify_change(
    prob_current: float,
    prob_prior: float,
    heatmap_iou: float,
    positive_threshold: float = 0.5,
    delta_threshold: float = 0.15,
) -> str:
    """
    Turns two probabilities + a spatial overlap score into one of:
    new / resolved / worsening / improving / unchanged.

    `heatmap_iou` isn't used to gate new/resolved (there's nothing to overlap
    when a finding wasn't there before), but for worsening/improving it's a
    real check that the change is happening in the same anatomical region,
    not that an unrelated new finding happened to nudge the probability.
    """
    current_positive = prob_current >= positive_threshold
    prior_positive = prob_prior >= positive_threshold
    delta = prob_current - prob_prior

    if current_positive and not prior_positive:
        return "new"
    if prior_positive and not current_positive:
        return "resolved"
    if current_positive and prior_positive:
        if delta >= delta_threshold:
            return "worsening" if heatmap_iou >= 0.1 else "worsening (region shifted — verify)"
        if delta <= -delta_threshold:
            return "improving" if heatmap_iou >= 0.1 else "improving (region shifted — verify)"
        return "unchanged"
    return "unchanged"  # negative in both


@torch.no_grad()
def _predict_probs(model, image_tensor: torch.Tensor, device: str) -> np.ndarray:
    model.eval()
    logits = model(image_tensor.unsqueeze(0).to(device))
    return torch.sigmoid(logits).squeeze(0).cpu().numpy()


def compare_studies(model, gradcam, current_tensor: torch.Tensor, prior_tensor: torch.Tensor, target_names: list[str], device: str = "cpu") -> list[ChangeResult]:
    """
    Runs the classifier + Grad-CAM on both images (each already preprocessed
    the same way as training — same resize/normalize) and returns one
    ChangeResult per finding.
    """
    probs_current = _predict_probs(model, current_tensor, device)
    probs_prior = _predict_probs(model, prior_tensor, device)

    results = []
    for i, name in enumerate(target_names):
        heatmap_current = gradcam(current_tensor.unsqueeze(0).to(device), target_index=i)
        heatmap_prior = gradcam(prior_tensor.unsqueeze(0).to(device), target_index=i)
        iou = compute_heatmap_iou(heatmap_current, heatmap_prior)

        status = classify_change(float(probs_current[i]), float(probs_prior[i]), iou)

        results.append(
            ChangeResult(
                finding=name,
                prob_current=float(probs_current[i]),
                prob_prior=float(probs_prior[i]),
                delta=float(probs_current[i] - probs_prior[i]),
                heatmap_iou=iou,
                status=status,
            )
        )
    return results
