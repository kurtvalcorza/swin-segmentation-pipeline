#!/usr/bin/env python3
"""Fail-closed verifier for the semantic-segmentation pipeline scaffold."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    surface = load(ROOT / "spec" / "pipeline-surface.json")
    provenance = load(ROOT / "provenance" / "open-weights.json")
    errors: list[str] = []

    if surface.get("pipelineId") != provenance.get("pipelineId"):
        errors.append("pipelineId mismatch")
    if surface.get("status") != "BLOCKED_PENDING_CONTRACT_AND_WORKERS":
        errors.append("scaffold must remain blocked until contract and worker releases exist")

    contract = surface.get("contract", {})
    if contract.get("candidateTaskProfile") != "core.task.vision.semantic-segmentation":
        errors.append("unexpected semantic-segmentation task profile id")
    if contract.get("taskProfileStatus") != "NOT_PRESENT_ON_REVIEWED_BRANCH":
        errors.append("task profile status must fail closed")
    if contract.get("datasetRepresentationStatus") != "NOT_PRESENT_ON_REVIEWED_BRANCH":
        errors.append("dataset representation status must fail closed")

    blockers = set(surface.get("blockers", []))
    required_blockers = {
        "CONTRACT_SEMANTIC_SEGMENTATION_TASK_PROFILE_MISSING",
        "CONTRACT_RASTER_MASK_REPRESENTATION_MISSING",
        "VALIDATOR_WORKER_RELEASE_MISSING",
        "FINETUNER_WORKER_RELEASE_MISSING",
        "CHECKPOINT_SHA256_PENDING",
    }
    missing = sorted(required_blockers - blockers)
    if missing:
        errors.append(f"missing blockers: {missing}")

    source = provenance.get("architectureSourceOfRecord", {})
    if not HEX40.fullmatch(source.get("sourceRevision", "")):
        errors.append("source revision is not an immutable 40-hex commit")

    model = provenance.get("canonicalV1", {})
    if not HEX40.fullmatch(model.get("configBlobSha", "")):
        errors.append("config blob sha is not a 40-hex git object id")
    digest = model.get("checkpointSha256")
    if digest is not None and not HEX64.fullmatch(digest):
        errors.append("checkpointSha256 must be null or 64 lowercase hex characters")
    if digest is None and model.get("status") == "QUALIFIED":
        errors.append("checkpoint cannot be QUALIFIED without SHA-256")
    if provenance.get("policy", {}).get("runtimeNetworkFetch") != "DENY":
        errors.append("runtime network fetch must be denied")

    if errors:
        for error in errors:
            print(f"FAIL  {error}")
        return 1

    print("PASS  segmentation scaffold is internally consistent and fail-closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
