"""Offline tests for the fleet snapshot scheme on the OpenMMLab loader: verification, staging, loading."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from dimer_swin_segmentation import (
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    MODEL_SPEC,
    DimerSwinSegmenter,
    runtime,
    stage_missing_files,
    verify_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = ROOT / "weights" / MODEL_KEY / runtime.MANIFEST_NAME
PAYLOAD = b"\x00" * 64


def _write_snapshot(root: Path, *, model_id: str = MODEL_ID, sha256: str | None = None, size: int | None = None) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    entry = {
        "path": runtime.WEIGHTS_FILE,
        "bytes": len(PAYLOAD) if size is None else size,
        "sha256": hashlib.sha256(PAYLOAD).hexdigest() if sha256 is None else sha256,
    }
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": model_id,
        "revision": MODEL_REVISION,
        "files": [entry],
        "totalBytes": entry["bytes"],
    }
    (root / runtime.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _pin_spec_to_payload(monkeypatch) -> None:
    """The tests use a 64-byte stand-in checkpoint; MODEL_SPEC must agree with it for verify_snapshot."""
    monkeypatch.setitem(runtime.MODEL_SPEC, "checkpoint_sha256", hashlib.sha256(PAYLOAD).hexdigest())
    monkeypatch.setitem(runtime.MODEL_SPEC, "checkpoint_size_bytes", len(PAYLOAD))


def _stub_openmmlab(monkeypatch, calls: list) -> None:
    """Stand-ins for MMSegmentation's initialiser and safetensors' reader, recording how the runtime calls them."""
    monkeypatch.setattr(runtime, "verify_runtime_versions", lambda: {"stub": "0"})
    monkeypatch.setattr(runtime, "resolve_packaged_config", lambda: Path("stub-config.py"))
    monkeypatch.setattr(
        runtime, "ade20k_dataset_meta", lambda: {"classes": tuple(f"c{i}" for i in range(150)), "palette": []}
    )

    class Model:
        def load_state_dict(self, state, strict):
            calls.append(("load_state_dict", state, strict))

    def init(config, checkpoint, device="cpu"):
        calls.append((config, checkpoint, device))
        return Model()

    def load_file(path):
        return {"read": path}

    apis = types.SimpleNamespace(init_model=init)
    monkeypatch.setitem(sys.modules, "mmseg", types.SimpleNamespace(apis=apis))
    monkeypatch.setitem(sys.modules, "mmseg.apis", apis)
    reader = types.SimpleNamespace(load_file=load_file)
    monkeypatch.setitem(sys.modules, "safetensors", types.SimpleNamespace(torch=reader))
    monkeypatch.setitem(sys.modules, "safetensors.torch", reader)


CONVERTED_PAYLOAD = b"\x02" * 32


def _pin_converted(monkeypatch) -> None:
    """The stand-in converted file and the digest the runtime must find for it."""
    monkeypatch.setattr(runtime, "CONVERTED_SHA256", hashlib.sha256(CONVERTED_PAYLOAD).hexdigest())


def test_committed_manifest_pins_the_model_spec_checkpoint() -> None:
    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (MODEL_ID, MODEL_REVISION, MODEL_KEY)
    assert [f["path"] for f in manifest["files"]] == [runtime.WEIGHTS_FILE]
    entry = manifest["files"][0]
    assert entry["sha256"] == MODEL_SPEC["checkpoint_sha256"]
    assert entry["bytes"] == MODEL_SPEC["checkpoint_size_bytes"]
    assert Path(MODEL_SPEC["checkpoint_url"]).name == runtime.WEIGHTS_FILE


def test_verify_snapshot_accepts_matching_file(monkeypatch, tmp_path: Path) -> None:
    _pin_spec_to_payload(monkeypatch)
    manifest = _write_snapshot(tmp_path)
    (tmp_path / runtime.WEIGHTS_FILE).write_bytes(PAYLOAD)
    result = verify_snapshot(tmp_path)
    assert result["path"] == str(tmp_path)
    assert result["files"] == manifest["files"]


def test_verify_snapshot_rejects_tampered_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _pin_spec_to_payload(monkeypatch)
    _write_snapshot(tmp_path)
    (tmp_path / runtime.WEIGHTS_FILE).write_bytes(b"\x01" * 64)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_manifest_that_disagrees_with_model_spec(monkeypatch, tmp_path: Path) -> None:
    _pin_spec_to_payload(monkeypatch)
    _write_snapshot(tmp_path, sha256="f" * 64)
    (tmp_path / runtime.WEIGHTS_FILE).write_bytes(PAYLOAD)
    with pytest.raises(ValueError, match="pinned in MODEL_SPEC"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(monkeypatch, tmp_path: Path) -> None:
    _pin_spec_to_payload(monkeypatch)
    _write_snapshot(tmp_path, model_id="someone/else")
    (tmp_path / runtime.WEIGHTS_FILE).write_bytes(PAYLOAD)
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(tmp_path)
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_stage_missing_files_fetches_only_the_absent_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _pin_spec_to_payload(monkeypatch)
    _write_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched: list[str] = []

    def downloader(relative_path: str, root: Path) -> None:
        fetched.append(relative_path)
        (root / relative_path).write_bytes(PAYLOAD)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == [runtime.WEIGHTS_FILE]
    assert fetched == [runtime.WEIGHTS_FILE]
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == []
    verify_snapshot(tmp_path)


def test_from_pretrained_converts_the_verified_checkpoint_then_serves_the_conversion(monkeypatch, tmp_path: Path) -> None:
    _pin_spec_to_payload(monkeypatch)
    _pin_converted(monkeypatch)
    _write_snapshot(tmp_path)
    (tmp_path / runtime.WEIGHTS_FILE).write_bytes(PAYLOAD)
    calls: list = []
    _stub_openmmlab(monkeypatch, calls)
    conversions: list = []

    def convert(source, target_dir):
        conversions.append((source, target_dir))
        (Path(target_dir) / runtime.CONVERTED_WEIGHTS_FILE).write_bytes(CONVERTED_PAYLOAD)

    monkeypatch.setattr(runtime, "convert_checkpoint", convert)
    model = DimerSwinSegmenter.from_pretrained(weights_dir=tmp_path)
    converted = tmp_path / runtime.CONVERTED_WEIGHTS_FILE
    assert conversions == [(tmp_path / runtime.WEIGHTS_FILE, tmp_path)]
    assert model.source == "local-snapshot"
    assert model.checkpoint == converted
    # Built from the config with no checkpoint (no MMEngine pickle load), then loaded strictly from safetensors.
    assert calls == [("stub-config.py", None, "cpu"), ("load_state_dict", {"read": str(converted)}, True)]
    assert len(model.classes) == 150
    # Once converted, the source is neither staged, verified nor read again.
    (tmp_path / runtime.WEIGHTS_FILE).unlink()
    calls.clear()
    DimerSwinSegmenter.from_pretrained(weights_dir=tmp_path)
    assert len(conversions) == 1 and calls[0] == ("stub-config.py", None, "cpu")


def test_from_pretrained_refuses_without_manifest_or_download(monkeypatch, tmp_path: Path) -> None:
    _stub_openmmlab(monkeypatch, [])
    with pytest.raises(FileNotFoundError, match="no snapshot manifest"):
        DimerSwinSegmenter.from_pretrained(weights_dir=tmp_path)
    _pin_spec_to_payload(monkeypatch)
    _write_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        DimerSwinSegmenter.from_pretrained(weights_dir=tmp_path)


def test_cache_path_serves_the_converted_weights(monkeypatch, tmp_path: Path) -> None:
    _pin_converted(monkeypatch)
    calls: list = []
    _stub_openmmlab(monkeypatch, calls)
    acquired = tmp_path / runtime.CONVERTED_WEIGHTS_FILE
    acquired.write_bytes(CONVERTED_PAYLOAD)
    monkeypatch.setattr(runtime, "acquire_converted_weights", lambda cache_dir: acquired)
    model = DimerSwinSegmenter(cache_dir=tmp_path)
    assert model.source == "openmmlab-cache"
    assert calls == [("stub-config.py", None, "cpu"), ("load_state_dict", {"read": str(acquired)}, True)]
