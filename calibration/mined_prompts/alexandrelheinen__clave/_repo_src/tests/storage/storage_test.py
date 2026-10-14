"""Unit tests for Cloudflare R2 and D1 storage integration.

Guards AC-DATA-01 through AC-DATA-08 defined in docs/requirements/data-storage.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clave.cli import _runs_list
from clave.data.dataset import (
    DESCRIPTION_FILE,
    DatasetError,
    _archive_digest,
)
from clave.storage.config import (
    R2Config,
    StorageConfigError,
    load_d1_config,
    load_r2_config,
    load_storage_config,
)
from clave.storage.d1 import D1Client
from clave.storage.r2 import R2Client
from clave.storage.sync import (
    pull_dataset,
    push_benchmark,
    push_dataset,
    push_training_run,
)


def test_ac_data_01_config_missing_keys_raises_clear_error(tmp_path: Path) -> None:
    """AC-DATA-01: System fails with a clear error naming missing keys."""
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")

    with pytest.raises(StorageConfigError) as r2_exc:
        load_r2_config(empty_env, environ={})
    msg = str(r2_exc.value)
    assert "R2_BUCKET_NAME" in msg
    assert "R2_ACCESS_KEY_ID" in msg
    assert "R2_SECRET_ACCESS_KEY" in msg
    assert "R2_ENDPOINT_URL" in msg

    with pytest.raises(StorageConfigError) as d1_exc:
        load_d1_config(empty_env, environ={})
    d1_msg = str(d1_exc.value)
    assert "CLOUDFLARE_ACCOUNT_ID" in d1_msg
    assert "CLOUDFLARE_API_TOKEN" in d1_msg
    assert "D1_DATABASE_ID" in d1_msg


def test_ac_data_01_loads_credentials_from_env_file(tmp_path: Path) -> None:
    """AC-DATA-01: Valid credentials load from a .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
        R2_BUCKET_NAME="my-bucket"
        R2_ACCESS_KEY_ID="test-access-key"
        R2_SECRET_ACCESS_KEY="test-secret-key"
        R2_ENDPOINT_URL="https://endpoint.r2.cloudflarestorage.com"
        CLOUDFLARE_ACCOUNT_ID="acc123"
        CLOUDFLARE_API_TOKEN="tok456"
        D1_DATABASE_ID="db789"
        D1_DATABASE_NAME="mydb"
        """
    )
    r2_cfg = load_r2_config(env_file, environ={})
    assert r2_cfg.bucket == "my-bucket"
    assert r2_cfg.access_key_id == "test-access-key"
    assert r2_cfg.secret_access_key == "test-secret-key"
    assert r2_cfg.endpoint_url == "https://endpoint.r2.cloudflarestorage.com"

    d1_cfg = load_d1_config(env_file, environ={})
    assert d1_cfg.account_id == "acc123"
    assert d1_cfg.api_token == "tok456"
    assert d1_cfg.database_id == "db789"
    assert d1_cfg.database_name == "mydb"

    combined = load_storage_config(env_file, environ={})
    assert combined.r2 is not None
    assert combined.d1 is not None


def test_ac_data_02_sigv4_signing_structure() -> None:
    """AC-DATA-02: Zero-dependency SigV4 generates compliant AWS headers."""
    cfg = R2Config(
        bucket="test-bucket",
        access_key_id="test_key",
        secret_access_key="test_secret",
        endpoint_url="https://test-account.r2.cloudflarestorage.com",
    )
    client = R2Client(cfg)
    req = client._build_request(
        "GET",
        "/test-bucket/some-key.json",
        query_params={"prefix": "test"},
    )
    auth_header = req.get_header("Authorization")
    assert auth_header is not None
    assert auth_header.startswith("AWS4-HMAC-SHA256 Credential=test_key/")
    assert "SignedHeaders=host;x-amz-content-sha256;x-amz-date" in auth_header
    assert "Signature=" in auth_header
    assert req.get_header("X-amz-date") is not None
    assert req.get_header("X-amz-content-sha256") is not None


def test_ac_data_02_and_04_dataset_push(tmp_path: Path) -> None:
    """AC-DATA-02, AC-DATA-04: Dataset push verifies local files and uploads missing."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    npz_file = dataset_dir / "rollout_000.npz"
    npz_file.write_bytes(b"dummy_npz_data")

    expected_digest = _archive_digest([npz_file])
    desc_json = {
        "digest": expected_digest,
        "seed": 42,
        "config_digest": "config_digest_123",
        "example_count": 1,
        "parts": {"train": ["rollout_000"]},
        "composition": {
            "train": {
                "example_count": 1,
                "class_counts": {},
                "absent_classes": [],
            }
        },
    }
    (dataset_dir / DESCRIPTION_FILE).write_text(json.dumps(desc_json))

    r2_mock = MagicMock(spec=R2Client)

    def fake_head(key: str) -> dict[str, str] | None:
        if key.endswith(DESCRIPTION_FILE):
            return {"content-length": "100"}
        return None

    r2_mock.head_object.side_effect = fake_head
    d1_mock = MagicMock(spec=D1Client)

    uploaded, skipped = push_dataset(dataset_dir, r2_mock, d1_mock)
    assert uploaded == 1  # rollout_000.npz
    assert skipped == 1  # dataset.json
    assert r2_mock.upload_file.call_count == 1

    d1_mock.record_dataset.assert_called_once()
    kwargs = d1_mock.record_dataset.call_args.kwargs
    assert kwargs["digest"] == expected_digest
    assert kwargs["seed"] == 42
    assert kwargs["config_digest"] == "config_digest_123"


def test_ac_data_03_dataset_pull_verification(tmp_path: Path) -> None:
    """AC-DATA-03: Pull downloads from R2 and verifies local digest."""
    dataset_dir = tmp_path / "source_dataset"
    dataset_dir.mkdir()
    npz_file = dataset_dir / "rollout_000.npz"
    npz_file.write_bytes(b"sample_binary_payload")

    digest = _archive_digest([npz_file])
    desc_json = {
        "digest": digest,
        "seed": 10,
        "config_digest": "cfg_abc",
        "example_count": 1,
        "parts": {"train": ["rollout_000"]},
        "composition": {
            "train": {
                "example_count": 1,
                "class_counts": {},
                "absent_classes": [],
            }
        },
    }
    (dataset_dir / DESCRIPTION_FILE).write_text(json.dumps(desc_json))

    dest_dir = tmp_path / "pulled_dataset"
    r2_mock = MagicMock(spec=R2Client)
    r2_mock.list_objects.return_value = [
        {"key": f"datasets/{digest}/dataset.json", "size": 100},
        {"key": f"datasets/{digest}/rollout_000.npz", "size": 50},
    ]

    def fake_download(key: str, local_path: Path) -> None:
        filename = Path(key).name
        local_path.write_bytes((dataset_dir / filename).read_bytes())

    r2_mock.download_file.side_effect = fake_download

    pulled_desc = pull_dataset(digest, dest_dir, r2_mock)
    assert pulled_desc.digest == digest
    assert (dest_dir / "dataset.json").is_file()
    assert (dest_dir / "rollout_000.npz").is_file()


def test_ac_data_03_dataset_pull_digest_mismatch_fails_and_cleans_up(
    tmp_path: Path,
) -> None:
    """AC-DATA-03: Mismatch deletes files and raises DatasetError."""
    dest_dir = tmp_path / "corrupted_pulled"
    digest = "expected_digest_000000000000000000000000000000000000000000000000000000"

    r2_mock = MagicMock(spec=R2Client)
    r2_mock.list_objects.return_value = [
        {"key": f"datasets/{digest}/dataset.json", "size": 100},
        {"key": f"datasets/{digest}/rollout_000.npz", "size": 50},
    ]

    def fake_download_corrupted(key: str, local_path: Path) -> None:
        if key.endswith("dataset.json"):
            fake_json = {
                "digest": digest,
                "seed": 0,
                "config_digest": "cfg",
                "example_count": 1,
                "parts": {"train": ["rollout_000"]},
                "composition": {},
            }
            local_path.write_text(json.dumps(fake_json))
        else:
            local_path.write_bytes(b"corrupted_bytes")

    r2_mock.download_file.side_effect = fake_download_corrupted

    with pytest.raises(DatasetError):
        pull_dataset(digest, dest_dir, r2_mock)

    assert not (dest_dir / "rollout_000.npz").exists()


def test_ac_data_05_push_training_run(tmp_path: Path) -> None:
    """AC-DATA-05: Push checkpoint and .run.json to R2 and D1."""
    ckpt_file = tmp_path / "act.pt"
    ckpt_file.write_bytes(b"model_weights")

    run_file = tmp_path / "act.run.json"
    run_file.write_text(
        json.dumps(
            {
                "candidate": "act",
                "config_digest": "cfg111",
                "dataset_digest": "data222",
                "machine": "x86_64, 16 threads",
                "epochs": [{"index": 0, "loss": 0.05, "seconds": 1.2}],
            }
        )
    )

    r2_mock = MagicMock(spec=R2Client)
    d1_mock = MagicMock(spec=D1Client)

    key = push_training_run(ckpt_file, run_file, r2_mock, d1_mock)
    assert key == "checkpoints/act/cfg111_data222/act.pt"
    assert r2_mock.upload_file.call_count == 2
    d1_mock.record_training_run.assert_called_once()
    kwargs = d1_mock.record_training_run.call_args.kwargs
    assert kwargs["candidate"] == "act"
    assert kwargs["dataset_digest"] == "data222"
    assert kwargs["config_digest"] == "cfg111"
    assert kwargs["epochs"] == 1
    assert kwargs["final_loss"] == 0.05
    d1_mock.record_model.assert_not_called()


def test_ac_data_09_a_score_records_the_model(tmp_path: Path) -> None:
    """AC-DATA-09: scoring a checkpoint records the model and its result."""
    from clave.storage.sync import index_scored_model

    checkpoint = tmp_path / "resnet50-baseline.pt"
    checkpoint.write_bytes(b"weights")
    (tmp_path / "resnet50-baseline.run.json").write_text(
        json.dumps(
            {
                "candidate": "resnet50-baseline",
                "config_digest": "cfg",
                "dataset_digest": "traindigest",
                "machine": "x86_64, 8 threads",
                "epochs": [{"index": 0, "loss": 0.2, "seconds": 1.0}],
            }
        )
    )
    r2_mock = MagicMock(spec=R2Client)
    d1_mock = MagicMock(spec=D1Client)
    key = index_scored_model(
        checkpoint,
        candidate="resnet50-baseline",
        name="gate classifier",
        validation_dataset_digest="valdigest",
        campaign_id="campaign",
        overall_agreement=0.81,
        mean_iou=None,
        frames_scored=12,
        r2=r2_mock,
        d1=d1_mock,
    )
    assert key == ("checkpoints/resnet50-baseline/cfg_traindigest/resnet50-baseline.pt")
    scored = d1_mock.record_model_score.call_args.kwargs
    assert scored["name"] == "gate classifier"
    assert scored["candidate"] == "resnet50-baseline"
    assert scored["validation_dataset_digest"] == "valdigest"
    assert scored["overall_agreement"] == 0.81
    assert scored["mean_iou"] is None
    assert scored["frames_scored"] == 12
    assert scored["checkpoint_r2_key"] == key
    trained = d1_mock.record_model.call_args.kwargs
    assert trained["name"] == "gate classifier"
    assert trained["candidate"] == "resnet50-baseline"
    assert trained["campaign_id"] == "campaign"
    assert trained["train_dataset_digest"] == "traindigest"


def test_recording_a_model_leaves_the_score_columns_in_place() -> None:
    """AC-DATA-09: republishing weights does not clear a validation score."""
    from clave.storage.d1 import D1Client

    client = D1Client.__new__(D1Client)
    with patch.object(D1Client, "execute", return_value=[]) as execute:
        client.record_model(
            model_id="a" * 64,
            name="gate classifier",
            candidate="resnet50-baseline",
            format="pt",
            checkpoint_r2_key="checkpoints/resnet50-baseline/cfg_data/model.pt",
            train_dataset_digest="data",
            config_digest="cfg",
            campaign_id="campaign",
            trained_at="2026-09-25T16:00:00Z",
            epochs=10,
            final_loss=0.1,
            machine="x86_64, 8 threads",
        )
    update = execute.call_args.args[0].split("DO UPDATE SET", maxsplit=1)[1]
    assert "overall_agreement" not in update
    assert "validation_dataset_digest" not in update


def test_ac_data_06_push_benchmark(tmp_path: Path) -> None:
    """AC-DATA-06: Push benchmark pack to R2 and register configurations in D1."""
    pack_file = tmp_path / "benchmark.json"
    pack_file.write_text(
        json.dumps(
            {
                "world_digest": "world_digest_999",
                "scored": [
                    {
                        "name": "resnet50 + act",
                        "overall_accuracy": 0.98,
                        "decision_latency_p99_seconds": 0.024,
                        "gates": {
                            "accuracy": {"observed": 0.98, "passed": True},
                            "misroute_rate": {"observed": 0.01, "passed": True},
                        },
                    }
                ],
            }
        )
    )

    r2_mock = MagicMock(spec=R2Client)
    d1_mock = MagicMock(spec=D1Client)

    key = push_benchmark(pack_file, r2_mock, d1_mock)
    assert key.startswith("benchmarks/world_digest_999/")
    assert r2_mock.upload_file.call_count == 1
    d1_mock.record_benchmark.assert_called_once()
    kwargs = d1_mock.record_benchmark.call_args.kwargs
    assert kwargs["world_digest"] == "world_digest_999"
    assert kwargs["configuration_name"] == "resnet50 + act"
    assert kwargs["overall_accuracy"] == 0.98
    assert kwargs["passed"] is True


def test_ac_data_07_runs_list(caplog: pytest.LogCaptureFixture) -> None:
    """AC-DATA-07: Query and render formatted tables for runs and benchmarks."""
    d1_mock = MagicMock(spec=D1Client)
    d1_mock.list_training_runs.return_value = [
        {
            "run_id": "act_cfg_data",
            "candidate": "act",
            "epochs": 10,
            "final_loss": 0.0215,
            "created_at": "2026-09-20 14:00:00",
        }
    ]
    d1_mock.list_benchmarks.return_value = [
        {
            "benchmark_id": "bench_01",
            "configuration_name": "resnet50 + act",
            "overall_accuracy": 0.992,
            "decision_latency_p99_seconds": 0.021,
            "passed": 1,
            "created_at": "2026-09-20 14:10:00",
        }
    ]

    with (
        patch("clave.storage.D1Client", return_value=d1_mock),
        patch("clave.storage.load_d1_config", return_value=MagicMock()),
        caplog.at_level(logging.INFO),
    ):
        exit_code = _runs_list(limit=5)

    assert exit_code == 0
    assert "Training Runs:" in caplog.text
    assert "act_cfg_data" in caplog.text
    assert "0.0215" in caplog.text
    assert "Benchmarks:" in caplog.text
    assert "resnet50 + act" in caplog.text
    assert "PASSED" in caplog.text


def test_ac_data_08_offline_train_and_benchmark(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """AC-DATA-08: Offline network failure logs warning and retains local files."""
    with patch(
        "clave.storage.config.load_r2_config",
        side_effect=StorageConfigError("missing R2_BUCKET_NAME"),
    ):
        from clave.cli import _train

        fake_run = MagicMock()
        fake_run.unavailable_reason = None
        fake_run.completed = True
        fake_run.candidate = "act"
        fake_run.machine = "x86_64"
        fake_run.threads = 8
        fake_run.dataset_digest = "data123"
        fake_run.epochs = []

        with (
            patch("clave.training.runner.train", return_value=fake_run),
            patch(
                "clave.training.config.TrainingConfig.load", return_value=MagicMock()
            ),
            patch("clave.training.memory.MemoryBudget.load"),
            caplog.at_level(logging.WARNING),
        ):
            code = _train(tmp_path, Path("config.yml"), None, sync=True)
            assert code == 0
            assert "WARNING  remote sync failed" in caplog.text


def test_pull_and_restore_training_checkpoint(tmp_path: Path) -> None:
    """Download checkpoint by digests and restore latest from D1."""
    dest_dir = tmp_path / "ckpts"
    r2_mock = MagicMock(spec=R2Client)
    d1_mock = MagicMock(spec=D1Client)

    from clave.storage.sync import (
        pull_training_checkpoint,
        restore_latest_checkpoint,
    )

    ckpt, rec = pull_training_checkpoint("act", "cfg1", "data2", dest_dir, r2_mock)
    assert ckpt == dest_dir / "act.pt"
    assert rec == dest_dir / "act.run.json"
    assert r2_mock.download_file.call_count == 2

    d1_mock.execute.return_value = [
        {"config_digest": "cfg_latest", "dataset_digest": "data_latest"}
    ]
    ckpt_l, rec_l = restore_latest_checkpoint("act", dest_dir, r2_mock, d1_mock)
    assert ckpt_l == dest_dir / "act.pt"
    assert rec_l == dest_dir / "act.run.json"
    assert r2_mock.download_file.call_count == 4
