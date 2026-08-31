"""
CXR Sentinel — Grad-CAM.

This is the "evidence" half of Phase 1: for a given predicted finding,
produce a heatmap over the input image showing which regions drove that
prediction. In Phase 2 this same module is reused to diff heatmaps between
the current and prior study (see notes in README under Phase 2).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _get_submodule(model: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    module = model
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str | None = None):
        self.model = model
        self.target_layer_name = target_layer_name or model.last_conv_layer_name
        self.target_layer = _get_submodule(model, self.target_layer_name)

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, image: torch.Tensor, target_index: int) -> np.ndarray:
        """
        image: single image tensor, shape (1, C, H, W), already normalized.
        target_index: index into the model's output logits for the finding
                      you want an explanation for.
        Returns a (H, W) heatmap normalized to [0, 1], resized to the input
        image's spatial size.
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        score = logits[0, target_index]
        score.backward()

        # Global-average-pool the gradients to get per-channel importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam
