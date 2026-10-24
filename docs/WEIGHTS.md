# Weight provenance and hosting

The task-inference runtime uses one checkpoint, pinned by the fleet snapshot scheme and by the package's
own `MODEL_SPEC`; the two must agree and `verify_snapshot` refuses a manifest that disagrees with `MODEL_SPEC`.

| Item | Value |
|---|---|
| `MODEL_ID` | `open-mmlab/mmsegmentation:swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512` — the config recipe `configs/swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py` inside the pinned MMSegmentation 1.2.2 package (`resolve_packaged_config`) |
| `MODEL_REVISION` | `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8` — the `v1.2.2` release-tag commit of `open-mmlab/mmsegmentation`, i.e. the config source. The checkpoint bytes carry no git revision of their own; the manifest digest below is what pins them. |
| `MODEL_KEY` | `swin-t-upernet-ade20k` → `weights/swin-t-upernet-ade20k/` |
| Checkpoint host | `download.openmmlab.com` (not the Hugging Face Hub): `https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth` |
| Manifest | `weights/swin-t-upernet-ade20k/dimer-base-manifest.json` — 1 file, 240,154,742 bytes in total |
| `upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth` | 240,154,742 bytes, SHA-256 `e380ad3e5d94060d89e4b62b5d393cdcc7f1f3406b1d46bcab547d3c276b6064` (equal to `MODEL_SPEC["checkpoint_sha256"]`) |
| Weights licence | Apache-2.0 (OpenMMLab MMSegmentation model zoo) |
| Format / trust boundary | the upstream `.pth` is a pickled PyTorch checkpoint and is only the conversion source. After its size and SHA-256 are verified, `convert_checkpoint` reads it once with `torch.load(..., weights_only=True)`, takes `state_dict` (dropping the `meta` training record) with mmengine's own key handling (strip a leading `module.`; none of the 271 keys carries one), checks its `meta` `CLASSES`/`PALETTE` against the pinned MMSegmentation `ADE20KDataset.METAINFO`, loads it into the config-built model with `strict=True`, and writes a safetensors file. A matching digest proves byte identity with the pinned distribution, not publisher authenticity. |
| Served file: `upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.safetensors` | 240,079,592 bytes, SHA-256 `a91744b53b1bb48e12ba2fc64a6a0bee57c42ae6007b004b5806a9b4b0d31cf9` (`CONVERTED_SHA256`; byte-reproducible: two conversions hash the same). Written next to the `.pth` in whichever directory the acquisition path uses (below); git-ignored (`weights/**/*.safetensors`). The segmenter is built with `init_model(config, None)` (no checkpoint, so MMEngine's pickle loader is never called), given the checkpoint's class names and palette, and loaded from this file with `load_state_dict(strict=True)`. |

## Acquisition paths

- **Fleet snapshot path (standalone notebook, `from_pretrained(weights_dir=…, allow_download=…)`):**
  `stage_missing_files` fetches only manifest entries that are absent (the checkpoint, from the pinned URL above,
  and only with `allow_download=True`), `verify_snapshot` re-hashes every entry against the manifest and
  `MODEL_SPEC`, and `convert_checkpoint` writes the served safetensors file into the same directory; then
  `DimerSwinSegmenter(checkpoint=<converted .safetensors>, source="local-snapshot")` serves it. Once the converted
  file exists, the `.pth` is not staged, verified or read again. Both weight files are git-ignored
  (`weights/**/*.pth`, `weights/**/*.safetensors`); only the manifest is committed. In a repository checkout this
  directory is `weights/swin-t-upernet-ade20k/`; the standalone notebook uses `./weights/swin-t-upernet-ade20k/` under
  its working directory.
- **Cache path (`DimerSwinSegmenter(cache_dir=…)`, used by the CLI):** `acquire_converted_weights` serves the
  converted file from `.dimer-models/` when it is there; otherwise `acquire_verified_checkpoint` downloads the `.pth`
  into `.dimer-models/`, verifies size + SHA-256 against `MODEL_SPEC`, and `convert_checkpoint` converts it there
  (`source="openmmlab-cache"`).

## Adaptation lineage (separate, not hosted)

The Microsoft/SwinTransformer release asset `upernet_swin_tiny_patch4_window7_512x512.pth` (SHA-256
`c26408bb5ddee935dcc709ad7815fa039f2db2b41134cd702da7da277d1a7d89`) recorded in `provenance/open-weights.json` belongs
to the still-scaffolded gradient-adaptation lineage, is **not** byte-identical to the inference checkpoint above, and
remains `dimer_hosting: BLOCKED` while its weight licence is `unknown`.
