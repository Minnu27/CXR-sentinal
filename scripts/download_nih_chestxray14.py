"""
CXR Sentinel — dataset download + converter (NIH ChestX-ray14).

Downloads directly from the Hugging Face mirror of the official NIH release
(same files as https://nihcc.app.box.com/v/ChestXray-NIHCC — no account, no
credentialing, no login required, per NIH's own usage terms). Then converts
into this repo's common CSV schema (see src/data.py).

The full dataset is 12 zip batches, ~9,300 images / ~3-4 GB each, ~42 GB
total. You almost never need all 12 for Phase 1 development — this script
lets you grab just a few batches for a fast local/VS Code iteration subset,
then pull the rest later (in Colab, with more disk/bandwidth) for a full
training run.

IMPORTANT — label caveat:
NIH ChestX-ray14 does NOT have a native "Lung Opacity" label (that's a
CheXpert-specific finding). This script maps "Infiltration" -> lung_opacity
as an approximation, since NIH labels were NLP-mined from reports and
Infiltration is the closest available concept. Treat Phase 1 lung_opacity
numbers trained on this data as a placeholder to validate the pipeline —
swap in CheXpert Plus or the RSNA Pneumonia Detection Challenge dataset
(which has a radiologist-defined "Lung Opacity" label) once you have it,
without changing any other code.

Usage:
    python -m scripts.download_nih_chestxray14 --num_batches 1
    python -m scripts.download_nih_chestxray14 --num_batches 12   # full dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

REPO = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data"
LABELS_CSV_URL = f"{REPO}/Data_Entry_2017_v2020.csv"
IMAGE_BATCH_URL = REPO + "/images/images_{batch:03d}.zip"

# NIH's 15 original NLP-mined labels (14 findings + "No Finding").
# "Lung Opacity" is NOT one of them — see module docstring.
NIH_TO_PROJECT_LABELS = {
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "pleural_effusion",
    "Infiltration": "lung_opacity",  # approximation, see module docstring
}


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists():
        print(f"  already have {dest.name}, skipping download")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {dest.name}: {downloaded / 1e6:.0f}MB / {total / 1e6:.0f}MB ({pct:.0f}%)", end="")
    print()
    tmp.rename(dest)


def download_and_extract_batch(batch_num: int, raw_dir: Path, images_out_dir: Path) -> int:
    """Downloads one images_XXX.zip, extracts pngs into images_out_dir (flat), returns count added."""
    zip_path = raw_dir / f"images_{batch_num:03d}.zip"
    print(f"[batch {batch_num:03d}] downloading...")
    download_file(IMAGE_BATCH_URL.format(batch=batch_num), zip_path)

    extract_dir = raw_dir / f"images_{batch_num:03d}_extracted"
    print(f"[batch {batch_num:03d}] extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    images_out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for png_path in extract_dir.rglob("*.png"):
        target = images_out_dir / png_path.name
        if not target.exists():
            shutil.move(str(png_path), str(target))
            count += 1

    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"[batch {batch_num:03d}] added {count} images")
    return count


def build_manifest(labels_csv_path: Path, images_dir: Path) -> pd.DataFrame:
    """Cross-references downloaded images against the NIH label CSV and maps to this repo's schema."""
    labels_df = pd.read_csv(labels_csv_path)
    labels_df = labels_df.rename(columns={"Image Index": "image_filename", "Patient ID": "patient_id"})

    available = {p.name for p in images_dir.glob("*.png")}
    labels_df = labels_df[labels_df["image_filename"].isin(available)].copy()

    for nih_label, project_label in NIH_TO_PROJECT_LABELS.items():
        labels_df[project_label] = labels_df["Finding Labels"].apply(
            lambda findings, nl=nih_label: int(nl in str(findings).split("|"))
        )

    labels_df["image_path"] = "images/" + labels_df["image_filename"]
    # NIH doesn't release real study dates or study IDs; Image Index is unique
    # per study here (each row is one image/study), so it doubles as study_id.
    labels_df["study_id"] = labels_df["image_filename"].str.replace(".png", "", regex=False)
    labels_df["study_date"] = ""

    keep_cols = ["image_path", "patient_id", "study_id", "study_date", *NIH_TO_PROJECT_LABELS.values()]
    return labels_df[keep_cols].reset_index(drop=True)


def patient_level_split(manifest: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int):
    """
    Splits by patient_id into train/val/test so no patient's images leak across splits.

    `test` is genuinely held out: nothing in the training or evaluation loop
    (including checkpoint selection, which uses val AUROC) ever looks at it
    until you deliberately run evaluate.py against data/test.csv at the end.
    That's what makes it "unseen data" rather than a second validation set.
    """
    patients = manifest["patient_id"].unique()
    shuffled = pd.Series(patients).sample(frac=1.0, random_state=seed).values

    n_val = max(1, int(len(shuffled) * val_fraction))
    n_test = max(1, int(len(shuffled) * test_fraction))

    val_patients = set(shuffled[:n_val])
    test_patients = set(shuffled[n_val:n_val + n_test])
    # everyone else -> train

    val_mask = manifest["patient_id"].isin(val_patients)
    test_mask = manifest["patient_id"].isin(test_patients)
    train_mask = ~val_mask & ~test_mask

    return (
        manifest[train_mask].reset_index(drop=True),
        manifest[val_mask].reset_index(drop=True),
        manifest[test_mask].reset_index(drop=True),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_batches", type=int, default=1, help="How many of the 12 image batches to download (1 batch ~= 9,300 images, ~3-4GB)")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15, help="Held out entirely — never used for training or checkpoint selection, only for final evaluation on unseen data.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 1 <= args.num_batches <= 12:
        raise ValueError("--num_batches must be between 1 and 12")

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "_raw"
    images_dir = out_dir / "images"

    print("Downloading label metadata (Data_Entry_2017_v2020.csv, ~9MB)...")
    labels_csv_path = raw_dir / "Data_Entry_2017_v2020.csv"
    download_file(LABELS_CSV_URL, labels_csv_path)

    total_images = 0
    for batch in range(1, args.num_batches + 1):
        total_images += download_and_extract_batch(batch, raw_dir, images_dir)

    print(f"\nDownloaded {total_images} new images. Building manifest...")
    manifest = build_manifest(labels_csv_path, images_dir)
    print(f"Manifest covers {len(manifest)} images across {manifest['patient_id'].nunique()} patients.")

    train_df, val_df, test_df = patient_level_split(manifest, args.val_fraction, args.test_fraction, args.seed)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(
        f"\nWrote {out_dir/'train.csv'} ({len(train_df)} rows), {out_dir/'val.csv'} ({len(val_df)} rows), "
        f"{out_dir/'test.csv'} ({len(test_df)} rows, held out — don't touch until final evaluation)."
    )
    print("Positive rates (train):")
    for label in NIH_TO_PROJECT_LABELS.values():
        print(f"  {label}: {train_df[label].mean():.1%}")
    print(
        "\nNote: lung_opacity is approximated from NIH's 'Infiltration' label — see this "
        "script's module docstring before trusting Phase 1 lung_opacity numbers."
    )


if __name__ == "__main__":
    main()
