"""The segmenter is served from the pinned safetensors conversion; the pickled `.pth` is read only to convert.

The conversion tests swap the Swin-T UPerNet for a two-layer stand-in, so they need torch and safetensors
but not the OpenMMLab stack. The last test runs only where the real checkpoint and its conversion are staged
under weights/ and MMSegmentation is installed.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import types
from pathlib import Path

import pytest

from dimer_swin_segmentation import runtime
from dimer_swin_segmentation.runtime import (
    CONVERTED_SHA256,
    CONVERTED_WEIGHTS_FILE,
    DEFAULT_WEIGHTS_DIR,
    WEIGHTS_FILE,
    DimerSwinSegmenter,
)

# A stand-in for the checkpoint's ADE20K metadata (the real one comes from the pinned MMSegmentation package).
META = {"classes": tuple(f"class {i}" for i in range(150)), "palette": [[i, i, i] for i in range(150)]}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_converted_file_is_pinned_safetensors() -> None:
    assert CONVERTED_WEIGHTS_FILE.endswith(".safetensors")
    assert CONVERTED_WEIGHTS_FILE.removesuffix(".safetensors") == WEIGHTS_FILE.removesuffix(".pth")
    assert len(CONVERTED_SHA256) == 64 and all(c in "0123456789abcdef" for c in CONVERTED_SHA256)


def test_serving_path_never_unpickles() -> None:
    init = inspect.getsource(DimerSwinSegmenter.__init__)
    load = inspect.getsource(DimerSwinSegmenter.from_pretrained)
    build = inspect.getsource(runtime.build_unloaded_model)
    for source in (init, load, build):
        assert "torch.load" not in source and "load_checkpoint" not in source
    assert "load_file" in init and "strict=True" in init and "verify_converted" in init
    assert "init_model(str(config), None" in build


def test_validator_declares_the_converted_digest() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tools" / "validate_release_assets.py"
    spec = importlib.util.spec_from_file_location("validate_release_assets", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    assert CONVERTED_SHA256 in validator.EXTERNAL_WEIGHT_DIGESTS


def test_a_pickle_or_altered_file_is_refused_before_any_model_is_built(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime, "verify_runtime_versions", lambda: {"stub": "0"})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the model must not be built from unverified weights")

    monkeypatch.setattr(runtime, "build_unloaded_model", forbidden)
    pickle = tmp_path / WEIGHTS_FILE
    pickle.write_bytes(b"\x80\x02")
    with pytest.raises(ValueError, match="serves only"):
        DimerSwinSegmenter(checkpoint=pickle)
    altered = tmp_path / CONVERTED_WEIGHTS_FILE
    altered.write_bytes(b"not the pinned conversion")
    with pytest.raises(ValueError, match="sha256"):
        DimerSwinSegmenter(checkpoint=altered)
    with pytest.raises(ValueError, match="sha256"):
        DimerSwinSegmenter.from_pretrained(weights_dir=tmp_path)


def test_build_unloaded_model_passes_no_checkpoint_and_sets_the_ade20k_metadata(monkeypatch) -> None:
    calls: list = []

    def init(config, checkpoint, device="cpu"):
        calls.append((config, checkpoint, device))
        return types.SimpleNamespace()

    apis = types.SimpleNamespace(init_model=init)
    monkeypatch.setitem(sys.modules, "mmseg", types.SimpleNamespace(apis=apis))
    monkeypatch.setitem(sys.modules, "mmseg.apis", apis)
    meta = {"classes": ("wall",) * 150, "palette": [[0, 0, 0]] * 150}
    monkeypatch.setattr(runtime, "ade20k_dataset_meta", lambda: meta)
    model = runtime.build_unloaded_model(Path("cfg.py"))
    assert calls == [("cfg.py", None, "cpu")]
    assert model.dataset_meta == meta


@pytest.fixture
def stand_in(tmp_path: Path, monkeypatch):
    """A pinned two-layer MMEngine-style checkpoint ({meta, state_dict}) and its conversion's digest."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    def build(config=None, device="cpu"):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
        model.dataset_meta = dict(META)
        return model

    torch.manual_seed(0)
    state = build().state_dict()
    root = tmp_path / "weights"
    root.mkdir()
    source = root / WEIGHTS_FILE
    meta = {"CLASSES": META["classes"], "PALETTE": META["palette"], "iter": 1}
    torch.save({"meta": meta, "state_dict": state}, source)
    monkeypatch.setitem(runtime.MODEL_SPEC, "checkpoint_sha256", _sha(source))
    monkeypatch.setitem(runtime.MODEL_SPEC, "checkpoint_size_bytes", source.stat().st_size)
    reference = tmp_path / "reference.safetensors"
    runtime._write_converted(state, reference)
    monkeypatch.setattr(runtime, "CONVERTED_SHA256", _sha(reference))
    monkeypatch.setattr(runtime, "build_unloaded_model", build)
    monkeypatch.setattr(runtime, "resolve_packaged_config", lambda: Path("stub-config.py"))
    monkeypatch.setattr(runtime, "verify_runtime_versions", lambda: {"stub": "0"})
    return torch, root, source, state


def test_checkpoint_state_dict_matches_mmengine_key_handling() -> None:
    torch = pytest.importorskip("torch")
    weight = torch.zeros(2)
    wrapped = {"meta": {}, "state_dict": {"module.a.w": weight, "b.module.w": weight}}
    assert runtime.checkpoint_state_dict(wrapped) == {"a.w": weight, "b.module.w": weight}
    assert runtime.checkpoint_state_dict({"a.w": weight}) == {"a.w": weight}
    with pytest.raises(ValueError, match="not a tensor"):
        runtime.checkpoint_state_dict({"state_dict": {"a": 1}})


def test_conversion_is_byte_reproducible_and_strict(stand_in) -> None:
    _torch, root, source, _state = stand_in
    first = _sha(runtime.convert_checkpoint(source, root))
    (root / CONVERTED_WEIGHTS_FILE).unlink()
    second = _sha(runtime.convert_checkpoint(source, root))
    assert first == second == runtime.CONVERTED_SHA256
    assert not list(root.glob("*.partial"))


def test_conversion_refuses_a_wrong_digest_or_class_list_and_leaves_nothing(stand_in, monkeypatch) -> None:
    torch, root, source, state = stand_in
    monkeypatch.setattr(runtime, "CONVERTED_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        runtime.convert_checkpoint(source, root)
    assert not (root / CONVERTED_WEIGHTS_FILE).exists() and not list(root.glob("*.partial"))
    renamed = root / "renamed.pth"
    repainted = {"CLASSES": META["classes"], "PALETTE": [[9, 9, 9]] * len(META["palette"])}
    torch.save({"meta": repainted, "state_dict": state}, renamed)
    monkeypatch.setitem(runtime.MODEL_SPEC, "checkpoint_sha256", _sha(renamed))
    monkeypatch.setitem(runtime.MODEL_SPEC, "checkpoint_size_bytes", renamed.stat().st_size)
    with pytest.raises(ValueError, match="class names or palette"):
        runtime.convert_checkpoint(renamed, root)


def test_conversion_refuses_an_unpinned_source(stand_in, monkeypatch) -> None:
    _torch, root, source, _state = stand_in
    monkeypatch.setitem(runtime.MODEL_SPEC, "checkpoint_sha256", "0" * 64)
    with pytest.raises(ValueError, match="pinned in MODEL_SPEC"):
        runtime.convert_checkpoint(source, root)


def test_cache_path_converts_once_then_serves_without_the_source(stand_in, monkeypatch) -> None:
    torch, root, source, state = stand_in
    monkeypatch.setattr(runtime, "acquire_verified_checkpoint", lambda cache_dir: source)
    first = DimerSwinSegmenter(cache_dir=root)
    assert first.checkpoint == root / CONVERTED_WEIGHTS_FILE
    source.unlink()

    def refuse(*_args, **_kwargs):
        raise AssertionError("torch.load called on the serving path")

    monkeypatch.setattr(torch, "load", refuse)
    monkeypatch.setattr(runtime, "acquire_verified_checkpoint", refuse)
    served = DimerSwinSegmenter(cache_dir=root).model.state_dict()
    assert set(served) == set(state) and all(torch.equal(served[k], state[k]) for k in state)


@pytest.mark.skipif(
    not all((DEFAULT_WEIGHTS_DIR / name).is_file() for name in (WEIGHTS_FILE, CONVERTED_WEIGHTS_FILE)),
    reason="real checkpoint and conversion not staged",
)
def test_real_conversion_equals_the_source_checkpoint() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("mmseg")
    from safetensors.torch import load_file

    runtime.verify_snapshot(DEFAULT_WEIGHTS_DIR)
    runtime.verify_converted(DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_FILE)
    checkpoint = torch.load(DEFAULT_WEIGHTS_DIR / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    source = runtime.checkpoint_state_dict(checkpoint)
    converted = load_file(str(DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_FILE))
    assert set(source) == set(converted)
    assert all(torch.equal(source[k], converted[k]) and source[k].dtype == converted[k].dtype for k in source)
    assert tuple(checkpoint["meta"]["CLASSES"]) == runtime.ade20k_dataset_meta()["classes"]
