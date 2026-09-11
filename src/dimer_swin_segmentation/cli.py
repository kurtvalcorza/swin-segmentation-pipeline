from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import DimerSwinSegmenter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pinned DIMER Swin-T UPerNet semantic segmenter.")
    parser.add_argument("images", nargs="+", help="Image path(s) to segment")
    parser.add_argument("--output-dir", required=True, help="Directory for semantic-mask PNGs and predictions.json")
    parser.add_argument("--provenance", help="Optional provenance JSON path")
    parser.add_argument("--cache-dir", default=".dimer-models")
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segmenter = DimerSwinSegmenter(cache_dir=args.cache_dir, device=args.device)
    summaries = []
    for image in args.images:
        result = segmenter.predict(image)
        mask_name = f"{Path(image).stem}.semantic.png"
        result.save_mask(output_dir / mask_name)
        summary = result.summary()
        summary["mask_file"] = mask_name
        summaries.append(summary)
    (output_dir / "predictions.json").write_text(json.dumps(summaries, indent=2) + "\n")
    if args.provenance:
        segmenter.write_provenance(args.provenance)
    print(json.dumps({"images": len(args.images), "output_dir": str(output_dir), "predictions": str(output_dir / 'predictions.json')}))


if __name__ == "__main__":
    main()
