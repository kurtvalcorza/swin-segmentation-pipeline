from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import mmseg
import torch
from mmseg.apis import MMSegInferencer

MODEL_CONFIG = "swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py"
CHECKPOINT_URL = "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth"
SAMPLE_URL = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/850d349e5038f291284e7999fcacbedc0922534b/ADE_val_00000001.jpg"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


root = Path(".runtime-probe")
root.mkdir(exist_ok=True)
checkpoint = root / "model.pth"
sample = root / "ADE_val_00000001.jpg"
if not checkpoint.exists():
    urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
if not sample.exists():
    urllib.request.urlretrieve(SAMPLE_URL, sample)

config = Path(mmseg.__file__).resolve().parent / ".mim" / "configs" / "swin" / MODEL_CONFIG
if not config.is_file():
    raise RuntimeError(f"Packaged MMSegmentation config not found: {config}")

digest = sha256(checkpoint)
print(json.dumps({
    "torch": torch.__version__,
    "mmseg": mmseg.__version__,
    "config": str(config),
    "checkpoint_bytes": checkpoint.stat().st_size,
    "checkpoint_sha256": digest,
}, indent=2))
assert checkpoint.stat().st_size == 240154742
assert digest == "e380ad3e5d94060d89e4b62b5d393cdcc7f1f3406b1d46bcab547d3c276b6064"

inferencer = MMSegInferencer(model=str(config), weights=str(checkpoint), device="cpu")
result = inferencer(str(sample), return_datasamples=True)
pred = result["predictions"]
if isinstance(pred, (list, tuple)):
    pred = pred[0]
mask = pred.pred_sem_seg.data.squeeze(0)
print(json.dumps({
    "prediction_shape": list(mask.shape),
    "min_label": int(mask.min().item()),
    "max_label": int(mask.max().item()),
    "unique_labels": int(torch.unique(mask).numel()),
}, indent=2))
assert mask.ndim == 2 and mask.numel() > 0
print("TASK-INFERENCE RUNTIME PROBE: PASS")
