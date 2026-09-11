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
    }
    missing = sorted(required_blockers - blockers)
    if missing:
        errors.append(f"missing blockers: {missing}")

    source = provenance.get("architectureSourceOfRecord", {})
    if not HEX40.fullmatch(source.get("referenceImplementationRevision", "")):
        errors.append("reference implementation revision is not an immutable 40-hex commit")

    model = provenance.get("canonicalV1", {})
    if not HEX40.fullmatch(model.get("configBlobSha", "")):
        errors.append("config blob sha is not a 40-hex git object id")
    digest = model.get("checkpointSha256")
    if not digest or not HEX64.fullmatch(digest):
        errors.append("checkpointSha256 must be a verified 64 lowercase hex characters digest")

    evidence_path = ROOT / "evidence" / "open-weights-kaggle.json"
    if not evidence_path.is_file():
        errors.append("missing evidence/open-weights-kaggle.json")
    else:
        evidence = load(evidence_path)
        if evidence.get("releaseAsset", {}).get("sha256") != digest:
            errors.append("evidence sha256 does not match provenance checkpointSha256")
        if evidence.get("releaseAsset", {}).get("observedSizeBytes") != model.get("checkpointSizeBytes"):
            errors.append("evidence observedSizeBytes does not match provenance checkpointSizeBytes")

    # DIMER Pipeline Specification 1.0 lifecycle / topology / capability metadata (GEN8, GEN15, DOC13/14).
    spec_meta = surface.get("dimerPipelineSpec", {})
    if spec_meta.get("pipeline_spec") != "1.0":
        errors.append("dimerPipelineSpec.pipeline_spec must be '1.0'")
    if spec_meta.get("lifecycle_status") != "scaffold":
        errors.append("lifecycle_status must remain 'scaffold' while blockers exist; promotion is not a field edit")
    if blockers and spec_meta.get("lifecycle_status") in ("candidate", "release"):
        errors.append("a repository with open blockers cannot be candidate or release")
    if spec_meta.get("implementation_topology") not in ("PACKAGE", "COMPOSED-WORKERS"):
        errors.append("implementation_topology must be PACKAGE or COMPOSED-WORKERS")
    modes = spec_meta.get("capability_modes") or []
    allowed_modes = {"GRADIENT-ADAPTATION", "CONTEXT-CONDITIONING", "PRETRAINED-INFERENCE", "MULTI-CAPABILITY-INFERENCE"}
    if not modes or not set(modes) <= allowed_modes:
        errors.append("capability_modes must be a non-empty subset of the Pipeline Spec modes")
    for forbidden in ("components", "release"):
        if forbidden in surface:
            errors.append(f"a scaffold must not declare {forbidden!r}")
    composition = surface.get("composition", {})
    if composition.get("manifestStatus") != "NOT_EMITTED" or composition.get("releaseStatus") != "NOT_EMITTED":
        errors.append("a scaffold must record composition manifest/release as NOT_EMITTED")

    # Weight licensing / redistribution gate (LIC4, LIC6, LIC7): unknown or prohibited blocks DIMER hosting.
    licensing = model.get("weightLicensing", {})
    status = licensing.get("redistribution_status")
    if status not in ("permitted", "conditional", "prohibited", "unknown"):
        errors.append("weightLicensing.redistribution_status must be permitted|conditional|prohibited|unknown")
    hosting = licensing.get("dimer_hosting")
    if status == "permitted":
        if not licensing.get("license") or not licensing.get("license_source"):
            errors.append("permitted redistribution requires license and license_source")
    elif status == "conditional":
        if hosting != "BLOCKED" and not licensing.get("condition_satisfied"):
            errors.append("conditional redistribution must keep dimer_hosting BLOCKED until condition_satisfied is recorded")
    else:
        if hosting != "BLOCKED":
            errors.append(f"redistribution_status {status!r} requires dimer_hosting BLOCKED")
    review = licensing.get("review", {})
    if not all(review.get(k) for k in ("date", "reviewer", "determination")):
        errors.append("weightLicensing.review needs date, reviewer and determination")

    policy = provenance.get("policy", {})
    if policy.get("runtimeNetworkFetch") != "DENY":
        errors.append("runtime network fetch must be denied")
    if policy.get("releaseAssetPublishesSourceCommitBinding") is not False:
        errors.append("release/source commit binding must not be claimed unless upstream publishes one")

    if errors:
        for error in errors:
            print(f"FAIL  {error}")
        return 1

    print("PASS  segmentation scaffold is internally consistent and fail-closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
