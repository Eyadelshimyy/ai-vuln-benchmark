"""Synchronization functions for datasets, checkpoints, and benchmark evidence.

Provides push and pull logic against Cloudflare R2 and Cloudflare D1,
governed by AC-DATA-02 through AC-DATA-08.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from clave.data.dataset import (
    DESCRIPTION_FILE,
    DatasetDescription,
    DatasetError,
)
from clave.data.dataset import (
    read as read_dataset,
)
from clave.storage.d1 import D1Client
from clave.storage.r2 import R2Client

logger = logging.getLogger(__name__)


def push_dataset(
    dataset_dir: Path,
    r2: R2Client,
    d1: D1Client | None = None,
) -> tuple[int, int]:
    """Upload a local dataset to R2 and register in D1 (AC-DATA-02, AC-DATA-04).

    Verifies the local dataset digest before uploading. Never overwrites
    existing objects.

    Args:
        dataset_dir: Path to the local dataset directory.
        r2: Connected R2 client.
        d1: Optional D1 client for metadata registration.

    Returns:
        A tuple of (uploaded_count, skipped_count).

    Raises:
        DatasetError: If local verification fails.
        R2Error: If R2 upload fails.
        D1Error: If D1 registration fails.
    """
    description = read_dataset(dataset_dir)
    digest = description.digest

    uploaded = 0
    skipped = 0

    files_to_upload: list[Path] = [dataset_dir / DESCRIPTION_FILE]
    files_to_upload.extend(sorted(dataset_dir.glob("*.npz")))

    for local_file in files_to_upload:
        if not local_file.is_file():
            continue
        key = f"datasets/{digest}/{local_file.name}"
        if r2.head_object(key) is not None:
            skipped += 1
            continue

        content_type = (
            "application/json"
            if local_file.name.endswith(".json")
            else "application/octet-stream"
        )
        r2.upload_file(local_file, key, content_type=content_type)
        uploaded += 1

    if d1 is not None:
        train_count = 0
        val_count = 0
        test_count = 0
        if "train" in description.composition:
            train_part = description.composition["train"]
            cnt = train_part.get("example_count", 0)
            train_count = cnt if isinstance(cnt, int) else 0
        if "validation" in description.composition:
            val_part = description.composition["validation"]
            cnt = val_part.get("example_count", 0)
            val_count = cnt if isinstance(cnt, int) else 0
        if "test" in description.composition:
            test_part = description.composition["test"]
            cnt = test_part.get("example_count", 0)
            test_count = cnt if isinstance(cnt, int) else 0

        d1.record_dataset(
            digest=digest,
            seed=description.seed,
            config_digest=description.config_digest,
            example_count=description.example_count,
            train_examples=train_count,
            validation_examples=val_count,
            test_examples=test_count,
            role=description.role,
            campaign_id=description.campaign_id,
        )
        for archived in description.files:
            d1.record_dataset_file(
                digest=digest,
                name=archived.name,
                sha256=archived.sha256,
                byte_count=archived.byte_count,
                frame_count=archived.frame_count,
                seed=archived.seed,
                belt_speed_meters_per_second=archived.belt_speed_meters_per_second,
                spacing_meters=archived.spacing_meters,
                camera_ids=",".join(archived.camera_ids),
                r2_key=f"datasets/{digest}/{archived.name}",
            )

    return uploaded, skipped


def pull_dataset(
    digest: str,
    destination: Path,
    r2: R2Client,
) -> DatasetDescription:
    """Download dataset from R2 and verify digest locally (AC-DATA-03).

    Zero silent corruption: if downloaded rollouts do not match the expected
    digest, the downloaded files are removed and DatasetError is raised.

    Args:
        digest: Expected dataset SHA-256 digest.
        destination: Local directory where files will be saved.
        r2: Connected R2 client.

    Returns:
        The verified DatasetDescription.

    Raises:
        DatasetError: If dataset is not found or fails local digest verification.
    """
    prefix = f"datasets/{digest}/"
    remote_objects = r2.list_objects(prefix=prefix)
    if not remote_objects:
        raise DatasetError(f"dataset {digest} not found in remote storage")

    destination.mkdir(parents=True, exist_ok=True)
    downloaded_paths: list[Path] = []

    try:
        for obj in remote_objects:
            key = obj["key"]
            filename = key.split("/")[-1]
            if not filename:
                continue
            target_file = destination / filename
            r2.download_file(key, target_file)
            downloaded_paths.append(target_file)

        # Verify composite digest against dataset.json
        description = read_dataset(destination)
        if description.digest != digest:
            raise DatasetError(
                f"downloaded dataset {destination} has digest "
                f"{description.digest}, expected {digest}"
            )
        return description
    except Exception as err:
        # Clean up any downloaded files to prevent silent corruptions
        for p in downloaded_paths:
            if p.is_file():
                with contextlib.suppress(OSError):
                    p.unlink()
        if isinstance(err, DatasetError):
            raise
        raise DatasetError(
            f"failed to pull and verify dataset {digest}: {err}"
        ) from err


def push_training_run(
    checkpoint_path: Path,
    run_record_path: Path,
    r2: R2Client,
    d1: D1Client | None = None,
) -> str:
    """Upload checkpoint and run record to R2 and register in D1 (AC-DATA-05).

    Args:
        checkpoint_path: Path to the local .pt model checkpoint.
        run_record_path: Path to the corresponding .run.json file.
        r2: Connected R2 client.
        d1: Optional D1 client.

    Returns:
        The R2 object key of the uploaded checkpoint.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not run_record_path.is_file():
        raise FileNotFoundError(f"run record not found: {run_record_path}")

    raw_record: dict[str, Any] = json.loads(run_record_path.read_text(encoding="utf-8"))
    candidate = str(raw_record.get("candidate", "unknown"))
    config_digest = str(raw_record.get("config_digest", "unknown"))
    dataset_digest = str(raw_record.get("dataset_digest", "unknown"))
    machine = str(raw_record.get("machine", "unknown"))

    epochs_list = raw_record.get("epochs", [])
    epochs_count = len(epochs_list)
    final_loss = float(epochs_list[-1]["loss"]) if epochs_list else 0.0

    r2_dir = f"checkpoints/{candidate}/{config_digest}_{dataset_digest}"
    checkpoint_key = f"{r2_dir}/{checkpoint_path.name}"
    record_key = f"{r2_dir}/{run_record_path.name}"

    r2.upload_file(
        checkpoint_path,
        checkpoint_key,
        content_type="application/octet-stream",
    )
    r2.upload_file(run_record_path, record_key, content_type="application/json")

    if d1 is not None:
        run_id = f"{candidate}_{config_digest[:8]}_{dataset_digest[:8]}"
        d1.record_training_run(
            run_id=run_id,
            candidate=candidate,
            dataset_digest=dataset_digest,
            config_digest=config_digest,
            epochs=epochs_count,
            final_loss=final_loss,
            machine=machine,
            checkpoint_r2_key=checkpoint_key,
        )

    return checkpoint_key


def index_scored_model(
    checkpoint: Path,
    *,
    candidate: str,
    name: str,
    validation_dataset_digest: str,
    overall_agreement: float,
    mean_iou: float | None,
    frames_scored: int,
    r2: R2Client,
    d1: D1Client,
    campaign_id: str | None = None,
) -> str:
    """Upload a scored checkpoint and attach the validation metrics.

    Args:
        checkpoint: The `.pt` file that was scored.
        candidate: Architecture name from the registry.
        name: Label for this scored checkpoint.
        validation_dataset_digest: Digest of the corpus the score used.
        overall_agreement: Fraction of class bits or boxes that matched.
        mean_iou: Mean best box overlap, or none for a classifier.
        frames_scored: Frames the score covered.
        r2: Connected R2 client.
        d1: Connected D1 client.
        campaign_id: Campaign that paired the two corpora, when known.

    Returns:
        The R2 object key of the checkpoint.
    """
    model_id = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    run_path = checkpoint.with_suffix(".run.json")
    checkpoint_key = f"checkpoints/{candidate}/{model_id}.pt"
    train_fields: tuple[str, str, int, float, str] | None = None
    if run_path.is_file():
        raw: dict[str, Any] = json.loads(run_path.read_text(encoding="utf-8"))
        config_digest = str(raw.get("config_digest", "unknown"))
        dataset_digest = str(raw.get("dataset_digest", "unknown"))
        checkpoint_key = (
            f"checkpoints/{candidate}/{config_digest}_{dataset_digest}/"
            f"{checkpoint.name}"
        )
        epochs_list = raw.get("epochs", [])
        if not isinstance(epochs_list, list):
            epochs_list = []
        train_fields = (
            dataset_digest,
            config_digest,
            len(epochs_list),
            float(epochs_list[-1]["loss"]) if epochs_list else 0.0,
            str(raw.get("machine", "unknown")),
        )
    r2.upload_file(checkpoint, checkpoint_key, content_type="application/octet-stream")
    if train_fields is not None:
        dataset_digest, config_digest, epochs, final_loss, machine = train_fields
        d1.record_model(
            model_id=model_id,
            name=name,
            candidate=candidate,
            format="pt",
            checkpoint_r2_key=checkpoint_key,
            train_dataset_digest=dataset_digest,
            config_digest=config_digest,
            campaign_id=campaign_id,
            trained_at=_utc_now(),
            epochs=epochs,
            final_loss=final_loss,
            machine=machine,
        )
    d1.record_model_score(
        model_id=model_id,
        name=name,
        candidate=candidate,
        format="pt",
        checkpoint_r2_key=checkpoint_key,
        validation_dataset_digest=validation_dataset_digest,
        overall_agreement=overall_agreement,
        mean_iou=mean_iou,
        frames_scored=frames_scored,
    )
    return checkpoint_key


def _utc_now() -> str:
    """UTC timestamp for a publication, independent of the machine's zone."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def push_benchmark(
    evidence_pack_path: Path,
    r2: R2Client,
    d1: D1Client | None = None,
) -> str:
    """Upload benchmark evidence to R2 and register in D1 (AC-DATA-06).

    Args:
        evidence_pack_path: Path to benchmark.json.
        r2: Connected R2 client.
        d1: Optional D1 client.

    Returns:
        The R2 object key of the uploaded evidence pack.
    """
    if not evidence_pack_path.is_file():
        raise FileNotFoundError(f"benchmark file not found: {evidence_pack_path}")

    raw_pack: dict[str, Any] = json.loads(
        evidence_pack_path.read_text(encoding="utf-8")
    )
    world_digest = str(raw_pack.get("world_digest", "unknown"))
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")

    pack_key = f"benchmarks/{world_digest}/{timestamp}_benchmark.json"
    r2.upload_file(evidence_pack_path, pack_key, content_type="application/json")

    if d1 is not None:
        scored_list = raw_pack.get("scored", [])
        for row in scored_list:
            config_name = str(row.get("name", "unknown"))
            accuracy = row.get("overall_accuracy")
            overall_acc = float(accuracy) if accuracy is not None else 0.0
            p99 = row.get("decision_latency_p99_seconds")
            p99_sec = float(p99) if p99 is not None else 0.0

            gates = row.get("gates", {})
            misroute = 0.0
            if "misroute_rate" in gates and "observed" in gates["misroute_rate"]:
                misroute = float(gates["misroute_rate"]["observed"])

            row_passed = (
                all(
                    bool(gate_info.get("passed", False)) for gate_info in gates.values()
                )
                if gates
                else False
            )

            benchmark_id = f"{config_name}_{timestamp}_{world_digest[:8]}"
            d1.record_benchmark(
                benchmark_id=benchmark_id,
                world_digest=world_digest,
                configuration_name=config_name,
                overall_accuracy=overall_acc,
                misroute_rate=misroute,
                decision_latency_p99_seconds=p99_sec,
                passed=row_passed,
                pack_r2_key=pack_key,
            )

    return pack_key


def pull_training_checkpoint(
    candidate: str,
    config_digest: str,
    dataset_digest: str,
    destination: Path,
    r2: R2Client,
) -> tuple[Path, Path]:
    """Download a versioned model checkpoint and run record from R2.

    Args:
        candidate: Model architecture name (e.g. 'act', 'resnet50').
        config_digest: Hyperparameter configuration digest.
        dataset_digest: Dataset composite SHA-256 digest.
        destination: Local directory to save checkpoint and record into.
        r2: Connected R2 client.

    Returns:
        Tuple of (checkpoint_path, run_record_path).
    """
    destination.mkdir(parents=True, exist_ok=True)
    r2_dir = f"checkpoints/{candidate}/{config_digest}_{dataset_digest}"
    ckpt_key = f"{r2_dir}/{candidate}.pt"
    record_key = f"{r2_dir}/{candidate}.run.json"

    local_ckpt = destination / f"{candidate}.pt"
    local_record = destination / f"{candidate}.run.json"

    r2.download_file(ckpt_key, local_ckpt)
    r2.download_file(record_key, local_record)
    return local_ckpt, local_record


def restore_latest_checkpoint(
    candidate: str,
    destination: Path,
    r2: R2Client,
    d1: D1Client,
) -> tuple[Path, Path]:
    """Look up the most recent run for a candidate in D1 and download weights.

    Args:
        candidate: Model architecture name.
        destination: Local directory where weights will be saved.
        r2: Connected R2 client.
        d1: Connected D1 client.

    Returns:
        Tuple of (checkpoint_path, run_record_path).
    """
    sql = """
    SELECT config_digest, dataset_digest
    FROM training_runs
    WHERE candidate = ?
    ORDER BY created_at DESC
    LIMIT 1
    """
    rows = d1.execute(sql, [candidate])
    if not rows:
        raise FileNotFoundError(
            f"no recorded training runs found in D1 for candidate {candidate!r}"
        )
    row = rows[0]
    config_digest = str(row["config_digest"])
    dataset_digest = str(row["dataset_digest"])
    return pull_training_checkpoint(
        candidate=candidate,
        config_digest=config_digest,
        dataset_digest=dataset_digest,
        destination=destination,
        r2=r2,
    )
