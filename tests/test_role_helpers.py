"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dimer_swin_segmentation import (
    INPUT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)
from dimer_swin_segmentation.runtime import SegmentationResult


def _image(tmp_path: Path, name: str = "a.png", size=(32, 24)) -> Path:
    path = tmp_path / name
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


def test_validate_inputs_returns_manifest_with_schema_and_identity(tmp_path: Path) -> None:
    a, b = _image(tmp_path), _image(tmp_path, "b.jpg", (48, 48))
    manifest = validate_inputs([a, b], names=["first", "second"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert [entry["id"] for entry in manifest["inputs"]] == ["first", "second"]
    assert manifest["inputs"][0]["width"] == 32 and manifest["inputs"][0]["height"] == 24
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_single_path_default_ids(tmp_path: Path) -> None:
    manifest = validate_inputs(_image(tmp_path, "solo.png"))
    assert [entry["id"] for entry in manifest["inputs"]] == ["solo.png"]


def test_validate_inputs_rejects_like_predict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Image does not exist"):
        validate_inputs(tmp_path / "missing.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="not a readable image"):
        validate_inputs(bad)
    with pytest.raises(ValueError, match="names must have one entry per image"):
        validate_inputs(_image(tmp_path), names=["a", "b"])


def test_evaluation_report_is_always_not_measurable() -> None:
    results = [
        SegmentationResult("a.png", np.zeros((4, 4), dtype=np.uint8), (0,)),
        {"image_id": "b.png", "shape": [4, 4], "classes_present": [1, 2], "num_classes_present": 2},
    ]
    report = evaluation_report(results, sample_kind="synthetic")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_images"] == 2
    assert "mean IoU" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
