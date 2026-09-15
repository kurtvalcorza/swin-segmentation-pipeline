"""Offline tests for Swin-T UPerNet adaptation, dataset contract, re-heading, and artifact round-trip."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from dimer_swin_segmentation import (
    ADAPT_CLASSES,
    ARTIFACT_FORMAT,
    MODEL_ID,
    MODEL_REVISION,
    DimerSwinSegmenter,
    freeze_backbone,
    rehead_model,
    split_dataset,
    synthetic_segmentation_dataset,
    validate_dataset,
)


class DummyHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.num_classes = num_classes
        self.out_channels = num_classes


class DummySegmentor(nn.Module):
    def __init__(self, num_classes: int = 150):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        self.decode_head = DummyHead(512, num_classes)
        self.auxiliary_head = DummyHead(256, num_classes)
        self.dataset_meta = {"classes": tuple(f"cls_{i}" for i in range(num_classes))}

    def data_preprocessor(self, data: dict, training: bool = True) -> dict:
        return data

    def forward(self, inputs=None, data_samples=None, mode: str = "loss"):
        if mode == "loss":
            loss = sum(p.sum() * 0.0 + 0.5 for p in self.decode_head.parameters())
            return {"loss_decode": loss}
        return None

    def parse_losses(self, losses: dict) -> tuple[torch.Tensor, dict]:
        total = sum(losses.values())
        return total, {"loss": float(total.detach().cpu())}


def test_rehead_model_replaces_conv_seg_and_updates_classes() -> None:
    model = DummySegmentor(150)
    assert model.decode_head.conv_seg.out_channels == 150
    assert model.auxiliary_head.conv_seg.out_channels == 150

    new_classes = ("bg", "road", "structure")
    in_dec, in_aux = rehead_model(model, new_classes, seed=123)

    assert in_dec == 512
    assert in_aux == 256
    assert model.decode_head.conv_seg.out_channels == 3
    assert model.decode_head.num_classes == 3
    assert model.auxiliary_head.conv_seg.out_channels == 3
    assert model.auxiliary_head.num_classes == 3
    assert model.dataset_meta["classes"] == ("bg", "road", "structure")


def test_freeze_backbone_disables_gradients() -> None:
    model = DummySegmentor(3)
    for p in model.backbone.parameters():
        assert p.requires_grad is True

    frozen = freeze_backbone(model)
    assert frozen > 0
    for p in model.backbone.parameters():
        assert p.requires_grad is False
    # Decode head parameters should remain trainable
    assert any(p.requires_grad for p in model.decode_head.parameters())


def test_synthetic_segmentation_dataset_generation_and_split() -> None:
    records = synthetic_segmentation_dataset(8, width=64, height=64, seed=42)
    assert len(records) == 8

    for rec in records:
        assert isinstance(rec["image"], Image.Image)
        assert isinstance(rec["mask"], np.ndarray)
        assert rec["mask"].shape == (64, 64)
        assert set(np.unique(rec["mask"])).issubset({0, 1, 2})

    train_recs, val_recs = split_dataset(records, val_fraction=0.25, seed=42)
    assert len(train_recs) == 6
    assert len(val_recs) == 2
    assert set(r["id"] for r in train_recs).isdisjoint(set(r["id"] for r in val_recs))


def test_validate_dataset_contract() -> None:
    records = synthetic_segmentation_dataset(4, width=32, height=32, seed=10)
    summary = validate_dataset(records, ADAPT_CLASSES)
    assert summary["verdict"] == "accepted"
    assert summary["n_records"] == 4
    assert summary["class_names"] == list(ADAPT_CLASSES)

    # Empty dataset rejects
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_dataset([], ADAPT_CLASSES)

    # Missing mask rejects
    with pytest.raises(ValueError, match="missing 'image' or 'mask'"):
        validate_dataset([{"image": records[0]["image"]}], ADAPT_CLASSES)

    # Shape mismatch rejects
    bad_shape_mask = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        validate_dataset([{"image": records[0]["image"], "mask": bad_shape_mask}], ADAPT_CLASSES)

    # Out of range class index rejects
    bad_class_mask = np.zeros((32, 32), dtype=np.uint8)
    bad_class_mask[0, 0] = 5  # valid is 0..2
    with pytest.raises(ValueError, match="outside valid range"):
        validate_dataset([{"image": records[0]["image"], "mask": bad_class_mask}], ADAPT_CLASSES)


def test_artifact_save_and_load_roundtrip(tmp_path: Path, monkeypatch) -> None:
    # Build a stub segmenter
    pipe = types.SimpleNamespace()
    pipe.model = DummySegmentor(3)
    pipe.classes = ("bg", "road", "structure")
    pipe.adapted = True
    pipe.base_state_digest = "e380ad3e" * 8
    pipe.checkpoint = tmp_path / "dummy.pth"
    pipe.checkpoint.write_bytes(b"\x00" * 32)
    pipe.device = "cpu"

    # Bind save_artifact
    save_func = DimerSwinSegmenter.save_artifact.__get__(pipe, DimerSwinSegmenter)
    artifact_path = tmp_path / "swin-segmentation-adapter-v1.pt"
    desc = save_func(artifact_path, notes="unit test adapter")

    assert Path(desc["path"]).is_file()
    assert desc["format"] == ARTIFACT_FORMAT
    assert desc["class_names"] == ["bg", "road", "structure"]
    assert desc["tensors"] > 0
    assert desc["model_id"] == MODEL_ID
    assert desc["model_revision"] == MODEL_REVISION

    # Test load_artifact with stubbed from_pretrained
    dummy_reloaded = DummySegmentor(3)
    stub_pipe = types.SimpleNamespace(
        model=dummy_reloaded,
        classes=("bg", "road", "structure"),
        adapted=False,
        source="local",
    )
    monkeypatch.setattr(DimerSwinSegmenter, "from_pretrained", classmethod(lambda cls, **kw: stub_pipe))

    loaded = DimerSwinSegmenter.load_artifact(artifact_path)
    assert loaded.adapted is True
    assert loaded.source == f"artifact:{artifact_path.name}"


def test_artifact_load_rejects_corrupt_or_mismatched(tmp_path: Path) -> None:
    bad_format = tmp_path / "bad_format.pt"
    torch.save({"format": "other_format"}, bad_format)
    with pytest.raises(ValueError, match="artifact format"):
        DimerSwinSegmenter.load_artifact(bad_format)

    bad_model = tmp_path / "bad_model.pt"
    torch.save(
        {"format": ARTIFACT_FORMAT, "model_id": "wrong_model", "model_revision": MODEL_REVISION},
        bad_model,
    )
    with pytest.raises(ValueError, match="artifact built on"):
        DimerSwinSegmenter.load_artifact(bad_model)

    bad_key = tmp_path / "bad_key.pt"
    torch.save(
        {
            "format": ARTIFACT_FORMAT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_key": "wrong_key",
        },
        bad_key,
    )
    with pytest.raises(ValueError, match="artifact model_key"):
        DimerSwinSegmenter.load_artifact(bad_key)


def test_finetune_loop_mock(tmp_path: Path, monkeypatch) -> None:
    import sys

    # Stub mmengine and mmseg structures for offline unit testing
    monkeypatch.setitem(
        sys.modules,
        "mmengine.structures",
        types.SimpleNamespace(PixelData=lambda data: types.SimpleNamespace(data=data)),
    )
    monkeypatch.setitem(
        sys.modules,
        "mmseg.structures",
        types.SimpleNamespace(
            SegDataSample=lambda: types.SimpleNamespace(
                gt_sem_seg=None, set_metainfo=lambda info: None
            )
        ),
    )

    # Setup mock pipeline
    pipe = types.SimpleNamespace()
    pipe.model = DummySegmentor(3)
    pipe.classes = ("bg", "road", "structure")
    pipe.adapted = False
    pipe.device = "cpu"

    records = synthetic_segmentation_dataset(4, width=32, height=32, seed=1)

    finetune_func = DimerSwinSegmenter.finetune.__get__(pipe, DimerSwinSegmenter)
    progress_calls = []
    result = finetune_func(
        records,
        epochs=2,
        batch_size=2,
        learning_rate=1e-4,
        progress=lambda p: progress_calls.append(p),
    )

    assert result["epochs"] == 2
    assert result["batch_size"] == 2
    assert len(result["epoch_losses"]) == 2
    assert len(progress_calls) == 2
    assert pipe.adapted is True
