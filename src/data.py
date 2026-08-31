"""
CXR Sentinel — Phase 1 data loading.

Expects a CSV with (at minimum) these columns:
    image_path      -- path to the image file, relative to `image_root`
    patient_id       -- used later in Phase 2 for longitudinal pairing
    study_id         -- used later in Phase 2 for longitudinal pairing
    study_date       -- ISO date string, used later in Phase 2

Plus one column per target finding, values in {0, 1}.
Default targets (Phase 1 scope): cardiomegaly, pleural_effusion, lung_opacity.

This schema is deliberately dataset-agnostic. Write a small converter script
per source dataset (NIH ChestX-ray14, CheXpert Plus, MIMIC-CXR) that maps
their native label formats into this CSV. Keeping patient_id/study_id/date
in the schema from day one means Phase 2 (longitudinal pairing) doesn't
require touching this file again.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_TARGETS = ["cardiomegaly", "pleural_effusion", "lung_opacity"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class CXRDatasetConfig:
    csv_path: str
    image_root: str
    targets: list[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    image_size: int = 320
    train: bool = True


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.0),  # CXRs: DO NOT flip L/R, laterality matters clinically
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class CXRDataset(Dataset):
    """Multi-label chest X-ray dataset driven by a CSV manifest."""

    def __init__(self, config: CXRDatasetConfig):
        self.config = config
        self.df = pd.read_csv(config.csv_path)

        missing = [c for c in ["image_path", *config.targets] if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"CSV at {config.csv_path} is missing required columns: {missing}. "
                f"See src/data.py docstring for the expected schema."
            )

        self.transform = build_transforms(config.image_size, config.train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.config.image_root, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        labels = row[self.config.targets].astype("float32").values.copy()
        import torch

        labels = torch.from_numpy(labels)

        meta = {
            "image_path": row["image_path"],
            "patient_id": row.get("patient_id", None),
            "study_id": row.get("study_id", None),
            "study_date": row.get("study_date", None),
        }
        return image, labels, meta
