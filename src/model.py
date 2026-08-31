"""
CXR Sentinel — Phase 1 model.

DenseNet121 is the standard backbone in the CXR classification literature
(CheXNet and most follow-ups use it) so results are comparable to published
baselines, and it's small enough to fine-tune on a free Colab GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class CXRClassifier(nn.Module):
    def __init__(self, num_targets: int, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)

        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Multi-label head: one logit per finding, sigmoid applied via loss (BCEWithLogitsLoss)
        self.head = nn.Linear(in_features, num_targets)

        # Kept as an attribute so gradcam.py can register a hook on it without
        # digging through torchvision's internal DenseNet structure.
        self.last_conv_layer_name = "backbone.features.denseblock4.denselayer16.conv2"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits
