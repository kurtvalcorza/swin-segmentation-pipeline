"""Offline tests for Swin-T UPerNet adaptation, dataset contract, re-heading, and artifact round-trip."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save as safetensors_save

from dimer_swin_segmentation import (
    ADAPT_CLASSES,
    ARTIFACT_FILENAME,
    ARTIFACT_FORMAT,
    ARTIFACT_METADATA_KEY,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    MODEL_SPEC,
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


def _stub_pipe(tmp_path: Path, model: nn.Module) -> types.SimpleNamespace:
    pipe = types.SimpleNamespace()
    pipe.model = model
    pipe.classes = ("bg", "road", "structure")
    pipe.adapted = True
    pipe.base_state_digest = "e380ad3e" * 8
    pipe.checkpoint = tmp_path / "dummy.pth"
    pipe.checkpoint.write_bytes(b"\x00" * 32)
    pipe.device = "cpu"
    return pipe


def _save(pipe: types.SimpleNamespace, path: Path, notes: str = "unit test adapter") -> dict:
    return DimerSwinSegmenter.save_artifact.__get__(pipe, DimerSwinSegmenter)(path, notes=notes)


def _write_safetensors(path: Path, metadata: dict | None) -> Path:
    header = None if metadata is None else {ARTIFACT_METADATA_KEY: json.dumps(metadata, sort_keys=True)}
    path.write_bytes(safetensors_save({"w": torch.zeros(2)}, metadata=header))
    return path


def test_artifact_save_and_load_roundtrip(tmp_path: Path, monkeypatch) -> None:
    torch.manual_seed(0)
    source = DummySegmentor(3)
    pipe = _stub_pipe(tmp_path, source)
    artifact_path = tmp_path / ARTIFACT_FILENAME
    desc = _save(pipe, artifact_path)

    assert ARTIFACT_FILENAME == "swin-segmentation-adapter-v2.safetensors"
    assert Path(desc["path"]).is_file()
    assert desc["format"] == ARTIFACT_FORMAT == "dimer_swin_segmentation_adapter_v2"
    assert desc["serialization"] == "safetensors"
    assert desc["class_names"] == ["bg", "road", "structure"]
    assert desc["tensors"] == len(source.state_dict()) > 0
    assert desc["model_id"] == MODEL_ID
    assert desc["model_revision"] == MODEL_REVISION

    # Every non-tensor field is plain data in the header metadata (SER11).
    metadata, _ = DimerSwinSegmenter.read_artifact(artifact_path)
    assert metadata == {
        "format": ARTIFACT_FORMAT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_key": MODEL_KEY,
        "checkpoint_sha256": MODEL_SPEC["checkpoint_sha256"],
        "class_names": ["bg", "road", "structure"],
        "adapted": True,
        "base_state_digest": "e380ad3e" * 8,
        "notes": "unit test adapter",
    }

    # Reload into a fresh, differently initialised model: tensors and outputs must match exactly.
    torch.manual_seed(1)
    fresh = DummySegmentor(3)
    assert not torch.equal(fresh.backbone[0].weight, source.backbone[0].weight)
    stub_pipe = types.SimpleNamespace(
        model=fresh, classes=("bg", "road", "structure"), adapted=False, source="local"
    )
    seen: dict = {}

    def fake_from_pretrained(cls, **kw):
        seen.update(kw)
        return stub_pipe

    monkeypatch.setattr(DimerSwinSegmenter, "from_pretrained", classmethod(fake_from_pretrained))
    loaded = DimerSwinSegmenter.load_artifact(artifact_path)
    assert loaded.adapted is True
    assert loaded.source == f"artifact:{artifact_path.name}"
    assert seen["class_names"] == ("bg", "road", "structure")
    original, reloaded = source.state_dict(), loaded.model.state_dict()
    assert original.keys() == reloaded.keys()
    for key in original:
        assert original[key].dtype == reloaded[key].dtype, key
        assert torch.equal(original[key], reloaded[key]), key
    x = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(7))
    feats = torch.ones(1, 512, 2, 2)
    with torch.no_grad():
        assert torch.equal(source.backbone(x), loaded.model.backbone(x))
        assert torch.equal(source.decode_head.conv_seg(feats), loaded.model.decode_head.conv_seg(feats))


def test_artifact_bytes_are_deterministic_and_pickle_free(tmp_path: Path) -> None:
    torch.manual_seed(0)
    pipe = _stub_pipe(tmp_path, DummySegmentor(3))
    first, second = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    d1, d2 = _save(pipe, first), _save(pipe, second)
    raw = first.read_bytes()
    assert raw == second.read_bytes()
    assert d1["sha256"] == d2["sha256"]

    # A safetensors file: little-endian u64 header length, a JSON header, then raw tensor bytes.
    header_len = int.from_bytes(raw[:8], "little")
    header_bytes = raw[8 : 8 + header_len]
    header = json.loads(header_bytes)
    assert list(header["__metadata__"]) == [ARTIFACT_METADATA_KEY]
    encoded = header["__metadata__"][ARTIFACT_METADATA_KEY]
    assert encoded == json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":"))
    assert sum(1 for k in header if k != "__metadata__") == d1["tensors"]

    # No pickle stream and no torch.save zip container: the header is JSON and the rest is raw tensor data,
    # whose length the header's offsets account for exactly.
    assert not raw.startswith(b"PK\x03\x04") and b"data.pkl" not in raw
    assert header_bytes.lstrip().startswith(b"{") and b"\x80" not in header_bytes
    data_end = max(info["data_offsets"][1] for k, info in header.items() if k != "__metadata__")
    assert len(raw) == 8 + header_len + data_end


def test_artifact_load_rejects_corrupt_or_mismatched(tmp_path: Path) -> None:
    # A legacy torch.save pickle (artifact v1) is refused without being unpickled.
    legacy = tmp_path / "swin-segmentation-adapter-v1.pt"
    torch.save({"format": "dimer_swin_segmentation_adapter_v1", "state_dict": {}}, legacy)
    with pytest.raises(ValueError, match="not a readable safetensors file"):
        DimerSwinSegmenter.load_artifact(legacy)

    # A tampered (truncated) artifact is refused.
    torch.manual_seed(0)
    good = tmp_path / "good.safetensors"
    _save(_stub_pipe(tmp_path, DummySegmentor(3)), good)
    raw = good.read_bytes()
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(raw[:-16])
    with pytest.raises(ValueError, match="not a readable safetensors file"):
        DimerSwinSegmenter.load_artifact(truncated)

    # A tampered header length is refused.
    bad_header = tmp_path / "bad_header.safetensors"
    bad_header.write_bytes((int.from_bytes(raw[:8], "little") + 9).to_bytes(8, "little") + raw[8:])
    with pytest.raises(ValueError, match="not a readable safetensors file"):
        DimerSwinSegmenter.load_artifact(bad_header)

    # A safetensors file without DIMER metadata is refused.
    with pytest.raises(ValueError, match="artifact format None"):
        DimerSwinSegmenter.load_artifact(_write_safetensors(tmp_path / "no_meta.safetensors", None))

    with pytest.raises(ValueError, match="artifact format"):
        DimerSwinSegmenter.load_artifact(
            _write_safetensors(tmp_path / "bad_format.safetensors", {"format": "other_format"})
        )

    with pytest.raises(ValueError, match="artifact built on"):
        DimerSwinSegmenter.load_artifact(
            _write_safetensors(
                tmp_path / "bad_model.safetensors",
                {"format": ARTIFACT_FORMAT, "model_id": "wrong_model", "model_revision": MODEL_REVISION},
            )
        )

    identity = {"format": ARTIFACT_FORMAT, "model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    with pytest.raises(ValueError, match="artifact model_key"):
        DimerSwinSegmenter.load_artifact(
            _write_safetensors(
                tmp_path / "bad_key.safetensors",
                {**identity, "model_key": "wrong_key"},
            )
        )

    with pytest.raises(ValueError, match="class_names"):
        DimerSwinSegmenter.load_artifact(
            _write_safetensors(
                tmp_path / "bad_classes.safetensors",
                {**identity, "model_key": MODEL_KEY},
            )
        )

    with pytest.raises(FileNotFoundError):
        DimerSwinSegmenter.load_artifact(tmp_path / "missing.safetensors")


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
