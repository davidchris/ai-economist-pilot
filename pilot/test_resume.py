"""Exact CPU continuation and crash-safe checkpoint contracts, exercised by pytest."""

import json
import os
import pickle
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import checkpoint
import numpy as np
import pytest
import torch


def equal_state(left, right, seen=None):
    """Compare complete simulator object graphs, including shared/cyclic objects."""
    if seen is None:
        seen = set()
    pair = (id(left), id(right))
    if pair in seen:
        return
    seen.add(pair)
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key], seen)
    elif isinstance(left, (list, tuple, deque)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_state(a, b, seen)
    elif hasattr(left, "__dict__") and not isinstance(left, type):
        equal_state(vars(left), vars(right), seen)
    else:
        assert left == right


def run_train(path, updates, *extra):
    command = [
        sys.executable,
        str(checkpoint.ROOT / "pilot/train.py"),
        "--device",
        "cpu",
        "--output",
        str(path),
        "--updates",
        str(updates),
        "--checkpoint-every",
        "2",
        "--keep-checkpoints",
        "2",
        *extra,
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_exact_restart(tmp_path):
    # A persistent path is optional for retaining measured acceptance artifacts.
    base = Path(os.environ.get("RESUME_TEST_RESULTS", str(tmp_path)))
    base.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    whole, split, resumed = (base / name for name in ("whole", "split", "resumed"))
    run_train(whole, 9, "--workers", "1", "--epochs", "1")
    run_train(split, 4, "--workers", "1", "--epochs", "1")
    source = split / "checkpoint-000004.pt"
    middle = checkpoint.load(source)
    env = pickle.loads(middle["workers"][0])[0]
    assert env.world.timestep == 512
    assert env._completions == 0
    # Model a crash with records beyond the last durable checkpoint, including a torn line.
    checkpoint.append_log(split / "metrics.jsonl", '{"update":5}')
    with (split / "metrics.jsonl").open("ab") as stream:
        stream.write(b'{"update":')
    checkpoint.append_log(split / "episodes.jsonl", '{"uncommitted":true}')
    run_train(resumed, 9, "--resume", str(source))
    a = checkpoint.load(whole / "checkpoint-000009.pt")
    b = checkpoint.load(resumed / "checkpoint-000009.pt")
    for key in (
        "model",
        "optimizer",
        "progress",
        "recurrent",
        "starts",
        "observation",
        "rng",
        "schedules",
    ):
        equal_state(a[key], b[key])
    equal_state(pickle.loads(a["workers"][0]), pickle.loads(b["workers"][0]))
    final_env = pickle.loads(b["workers"][0])[0]
    assert final_env._completions == 1
    assert final_env.world.timestep == 152
    assert final_env._auto_warmup_integrator > env._auto_warmup_integrator
    records = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [record["update"] for record in records] == list(range(1, 10))
    whole_records = [
        json.loads(line) for line in (whole / "metrics.jsonl").read_text().splitlines()
    ]
    for x, y in zip(whole_records, records):
        equal_state(
            {k: v for k, v in x.items() if not k.endswith("seconds") and k != "rss_bytes"},
            {k: v for k, v in y.items() if not k.endswith("seconds") and k != "rss_bytes"},
        )
    assert len((resumed / "episodes.jsonl").read_text().splitlines()) == 1
    for path in (whole, split, resumed):
        assert len(list(path.glob("checkpoint-*.pt"))) == 2
    # Configuration changes must fail before creating an output directory.
    for flag, value in (("--workers", "2"), ("--sequence-length", "8"), ("--horizon", "64")):
        rejected = base / "rejected"
        result = subprocess.run(
            [
                sys.executable,
                str(checkpoint.ROOT / "pilot/train.py"),
                "--device",
                "cpu",
                "--resume",
                str(source),
                "--output",
                str(rejected),
                flag,
                value,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Incompatible resume configuration" in result.stderr
        assert not rejected.exists()
    (base / "comparison.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "comparison": "bitwise CPU state equality",
                "updates": 9,
                "restart_update": 4,
                "restart_episode_step": 512,
                "final_episode_step": 152,
                "episode_completions": final_env._completions,
                "labor_integrator_before": env._auto_warmup_integrator,
                "labor_integrator_after": final_env._auto_warmup_integrator,
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("failure", ["write", "rename"])
def test_atomic_failure_and_retention(tmp_path, monkeypatch, failure):
    def snapshot(update):
        return {"progress": {"update": update}}

    first = checkpoint.save(tmp_path, snapshot(1), 1)
    original = first.read_bytes()
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            if failure == "write":
                args[1].write(b"partial checkpoint")
            raise OSError("injected write failure")

        if failure == "write":
            patch.setattr(torch, "save", fail)
        else:
            patch.setattr(os, "replace", fail)
        with pytest.raises(OSError, match="injected"):
            checkpoint.save(tmp_path, snapshot(2), 1)
    assert first.read_bytes() == original
    assert list(tmp_path.iterdir()) == [first]
    second = checkpoint.save(tmp_path, snapshot(2), 1)
    assert list(tmp_path.iterdir()) == [second]


def test_legacy_and_incompatible_format(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model": {}}, path)
    with pytest.raises(ValueError, match="legacy"):
        checkpoint.load(path)
    torch.save({"version": checkpoint.VERSION, "compatibility": {}}, path)
    with pytest.raises(ValueError, match="incompatible"):
        checkpoint.load(path)


def test_unique_directories_and_log_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "ROOT", tmp_path)
    first, second = checkpoint.output_directory(), checkpoint.output_directory()
    assert first != second
    with pytest.raises(FileExistsError):
        checkpoint.output_directory(str(first))
    with pytest.raises(ValueError, match="truncated"):
        checkpoint.recover_logs(first, second, {"metrics.jsonl": 2})


def test_interrupted_run_recovers_committed_metrics(tmp_path):
    output = tmp_path / "interrupted"
    # Keep enough distance to the next checkpoint to terminate deterministically.
    command = [
        sys.executable,
        str(checkpoint.ROOT / "pilot/train.py"),
        "--device",
        "cpu",
        "--workers",
        "1",
        "--updates",
        "100",
        "--horizon",
        "16",
        "--sequence-length",
        "16",
        "--epochs",
        "1",
        "--checkpoint-every",
        "5",
        "--output",
        str(output),
    ]
    with (tmp_path / "process.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail((tmp_path / "process.log").read_text())
                metrics = output / "metrics.jsonl"
                if metrics.exists() and len(metrics.read_text().splitlines()) >= 6:
                    break
                time.sleep(0.02)
            else:
                pytest.fail("Trainer did not reach the interruption boundary")
        finally:
            # Terminate the whole spawned process group, including environment workers.
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
    source = output / "checkpoint-000005.pt"
    assert source.exists()
    assert not (output / "checkpoint-000010.pt").exists()
    resumed = tmp_path / "recovered"
    run_train(resumed, 7, "--resume", str(source))
    records = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [r["update"] for r in records] == list(range(1, 8))
    assert checkpoint.load(resumed / "checkpoint-000007.pt")["progress"]["update"] == 7


@pytest.mark.skipif(
    os.environ.get("RUN_CUDA_RESUME_TEST") != "1", reason="Opt-in bounded CUDA smoke test"
)
def test_cuda_resume(tmp_path):
    base = Path(os.environ.get("CUDA_TEST_RESULTS", str(tmp_path)))
    base.mkdir(parents=True, exist_ok=True)
    first, resumed = base / "cuda-start", base / "cuda-resumed"
    run_train(first, 2, "--device", "cuda", "--workers", "4", "--horizon", "32", "--epochs", "1")
    source = first / "checkpoint-000002.pt"
    run_train(resumed, 4, "--device", "cuda", "--resume", str(source))
    saved = checkpoint.load(resumed / "checkpoint-000004.pt")
    assert saved["progress"]["environment_steps"] == 512
    assert len(saved["workers"]) == 4
    assert len(saved["rng"]["cuda"]) > 0
    assert all(t.device.type == "cpu" for t in saved["model"].values())
    summary = json.loads((resumed / "summary.json").read_text())
    assert summary["checkpoint_cpu_portability"]
    assert summary["cpu_device_forward_agreement"]
