"""Deterministic in-code sample data: ADE20K demonstration scene and the custom adaptation dataset.

Nothing here is downloaded and nothing needs external dependencies beyond Pillow and numpy,
so the tutorial's default path has zero network dataset dependencies and the same generators
are exercised by the repository's unit tests.

Two separate label vocabularies live here and must not be conflated:

* ``tutorial_scene`` returns an RGB scene for demonstrating the pretrained ADE20K 150-class model.
* ``synthetic_segmentation_dataset`` returns records labelled with ``ADAPT_CLASSES``, a three-class
  vocabulary (``background``, ``road``, ``structure``) that represents a custom target deployment domain.
  A model fine-tuned on this dataset predicts only these three classes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .metrics import IGNORE_INDEX

ADE20K_SCENE_SIZE = (512, 384)
ADAPT_SCENE_SIZE = (512, 512)

# The custom adaptation vocabulary. Deliberately distinct from ADE20K indices.
# A model re-headed and fine-tuned on this dataset predicts these three classes.
ADAPT_CLASSES: tuple[str, ...] = ("background", "road", "structure")


def tutorial_scene(
    width: int = ADE20K_SCENE_SIZE[0], height: int = ADE20K_SCENE_SIZE[1]
) -> Image.Image:
    """The ADE20K demonstration scene: deterministic gradient background plus flat shapes.

    Matches the fixed tutorial fixture used to demonstrate pretrained inference.
    """
    ramp = np.linspace(0.0, 255.0, width)
    red = np.tile(ramp, (height, 1))
    green = np.tile(np.linspace(0.0, 255.0, height)[:, None], (1, width))
    blue = (red + green) / 2.0
    array = np.rint(np.stack([red, green, blue], axis=-1)).astype(np.uint8)
    scene = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(scene)
    draw.rectangle([40, int(height * 0.625), 220, int(height * 0.9375)], fill=(20, 20, 20))
    draw.ellipse([300, int(height * 0.156), 460, int(height * 0.573)], fill=(240, 240, 240))
    draw.polygon(
        [(260, int(height * 0.963)), (330, int(height * 0.651)), (400, int(height * 0.963))],
        fill=(30, 90, 200),
    )
    return scene


def generate_scene(
    index: int,
    *,
    width: int = ADAPT_SCENE_SIZE[0],
    height: int = ADAPT_SCENE_SIZE[1],
    seed: int = 20260915,
) -> tuple[Image.Image, np.ndarray]:
    """Generate a single synthetic image and its exact pixel-aligned ground truth mask.

    Classes:
      0: background (sky / environment)
      1: road (traversable ground plane)
      2: structure (geometric buildings / block entities)
    """
    rng = np.random.default_rng(seed + index * 101)

    # Base background (class 0)
    bg_r = int(rng.integers(120, 160))
    bg_g = int(rng.integers(170, 210))
    bg_b = int(rng.integers(210, 250))
    img = Image.new("RGB", (width, height), (bg_r, bg_g, bg_b))
    d = ImageDraw.Draw(img)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Road / ground plane (class 1)
    road_top = int(height * rng.uniform(0.55, 0.65))
    road_color = int(rng.integers(60, 90))
    d.rectangle([0, road_top, width, height], fill=(road_color, road_color, road_color))
    mask[road_top:height, :] = 1

    # Add 1 to 3 structures (class 2) seated on or above the ground plane
    n_structures = int(rng.integers(1, 4))
    for s_idx in range(n_structures):
        sw = max(4, int(rng.integers(max(4, int(width * 0.12)), max(5, int(width * 0.28)))))
        sh = max(4, int(rng.integers(max(4, int(height * 0.15)), max(5, int(height * 0.35)))))
        low_x = int(width * 0.05 + s_idx * width * 0.28)
        high_x = int(min(width - sw - 2, (s_idx + 1) * width * 0.32))
        if high_x <= low_x:
            sx = max(2, min(low_x, max(2, width - sw - 2)))
        else:
            sx = int(rng.integers(low_x, high_x))
        sx = max(2, min(sx, max(2, width - sw - 2)))
        sy = road_top - int(sh * rng.uniform(0.6, 0.9))
        sy = max(5, min(sy, road_top - 5))
        sy_bottom = min(height - 2, sy + sh)

        st_r = int(rng.integers(160, 220))
        st_g = int(rng.integers(30, 80))
        st_b = int(rng.integers(30, 80))
        d.rectangle([sx, sy, sx + sw, sy_bottom], fill=(st_r, st_g, st_b), outline=(255, 255, 255), width=2)
        mask[sy:sy_bottom, sx : sx + sw] = 2

        # Optional roof triangle
        if rng.random() > 0.4:
            peak_y = max(5, sy - int(sh * 0.3))
            peak_x = sx + sw // 2
            roof_pts = [(sx - 4, sy), (peak_x, peak_y), (sx + sw + 4, sy)]
            d.polygon(roof_pts, fill=(max(0, st_r - 40), max(0, st_g - 20), max(0, st_b - 20)))
            # Compute triangle mask using polygon rasterization
            roof_img = Image.new("L", (width, height), 0)
            roof_draw = ImageDraw.Draw(roof_img)
            roof_draw.polygon(roof_pts, fill=1)
            roof_arr = np.array(roof_img, dtype=bool)
            mask[roof_arr] = 2

    return img, mask


def synthetic_segmentation_dataset(
    n_samples: int = 24,
    *,
    width: int = ADAPT_SCENE_SIZE[0],
    height: int = ADAPT_SCENE_SIZE[1],
    seed: int = 20260915,
) -> list[dict[str, Any]]:
    """Produce a deterministic multi-image segmentation adaptation dataset.

    Returns a list of dicts with keys:
    - ``id``: unique string identifier
    - ``image``: PIL.Image in RGB mode
    - ``mask``: 2D uint8 numpy array with pixel values in 0..len(ADAPT_CLASSES)-1
    """
    records = []
    for i in range(n_samples):
        img, mask = generate_scene(i, width=width, height=height, seed=seed)
        records.append({
            "id": f"scene_{i:03d}",
            "image": img,
            "mask": mask,
        })
    return records


def split_dataset(
    records: Sequence[dict[str, Any]],
    *,
    val_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically partition records into train and validation splits."""
    n = len(records)
    if n < 2:
        raise ValueError(f"Need at least 2 records to partition; got {n}.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}.")

    n_val = max(1, math.floor(n * val_fraction))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    val_idx = set(indices[:n_val])

    train_split = [r for i, r in enumerate(records) if i not in val_idx]
    val_split = [r for i, r in enumerate(records) if i in val_idx]
    return train_split, val_split


def validate_dataset(
    records: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    *,
    epochs: int = 1,
    max_pixels: int = 64_000_000,
) -> dict[str, Any]:
    """Validate that ``records`` adhere to the ``core.dataset.vision.raster-mask`` contract.

    Raises ValueError on schema violations, shape mismatches, or invalid class indices.
    """
    if not records:
        raise ValueError("Dataset cannot be empty; at least 1 record required.")
    if not class_names:
        raise ValueError("class_names cannot be empty.")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")

    num_classes = len(class_names)
    observed_classes: set[int] = set()
    validated_records = []

    for idx, record in enumerate(records):
        rec_id = str(record.get("id", f"sample_{idx}"))
        if "image" not in record or "mask" not in record:
            raise ValueError(f"Record {rec_id} missing 'image' or 'mask' field.")

        img_raw = record["image"]
        if isinstance(img_raw, (str, Path)):
            img_path = Path(img_raw)
            if not img_path.is_file():
                raise ValueError(f"Record {rec_id}: image file not found: {img_path}")
            with Image.open(img_path) as opened:
                img = opened.convert("RGB")
        elif isinstance(img_raw, Image.Image):
            img = img_raw.convert("RGB")
        else:
            raise TypeError(f"Record {rec_id}: unexpected image type {type(img_raw)}.")

        width, height = img.size
        if width < 1 or height < 1:
            raise ValueError(f"Record {rec_id}: dimensions must be positive; got {width}x{height}.")
        if width * height > max_pixels:
            raise ValueError(f"Record {rec_id}: {width * height} pixels exceeds ceiling {max_pixels}.")

        mask_raw = record["mask"]
        if isinstance(mask_raw, (str, Path)):
            mask_path = Path(mask_raw)
            if not mask_path.is_file():
                raise ValueError(f"Record {rec_id}: mask file not found: {mask_path}")
            with Image.open(mask_path) as opened_m:
                mask = np.array(opened_m)
        elif isinstance(mask_raw, Image.Image):
            mask = np.array(mask_raw)
        elif isinstance(mask_raw, np.ndarray):
            mask = mask_raw
        else:
            raise TypeError(f"Record {rec_id}: unexpected mask type {type(mask_raw)}.")

        if mask.ndim != 2:
            raise ValueError(f"Record {rec_id}: mask must be 2-D; got shape {mask.shape}.")
        if mask.shape != (height, width):
            raise ValueError(
                f"Record {rec_id}: mask shape {mask.shape} != image height/width ({height}, {width})."
            )

        unique_vals = np.unique(mask)
        for val in unique_vals:
            if val != IGNORE_INDEX:
                if val < 0 or val >= num_classes:
                    raise ValueError(
                        f"Record {rec_id}: mask contains class index {val} "
                        f"outside valid range 0..{num_classes - 1}."
                    )
                observed_classes.add(int(val))

        validated_records.append({
            "id": rec_id,
            "width": width,
            "height": height,
            "classes_present": [int(v) for v in unique_vals if v != IGNORE_INDEX],
        })

    return {
        "n_records": len(records),
        "class_names": list(class_names),
        "observed_classes": sorted(observed_classes),
        "schema": "core.dataset.vision.raster-mask",
        "verdict": "accepted",
    }
