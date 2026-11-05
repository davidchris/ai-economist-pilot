"""Trusted local snapshots with atomic publication and bounded retention."""

import hashlib
import os
import pickle
import platform
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

VERSION = 3
ROOT = Path(__file__).resolve().parents[1]


def compatibility():
    """Reject code/runtime drift rather than promising a portable simulator pickle."""
    digest = hashlib.sha256()
    paths = sorted((ROOT / "upstream/ai_economist/foundation").rglob("*.py"))
    paths += [
        ROOT / f"upstream/tutorials/rllib/{phase}/config.yaml" for phase in ("phase1", "phase2")
    ]
    paths += [
        ROOT / "pilot" / name
        for name in ("model.py", "environment.py", "train.py", "checkpoint.py")
    ]
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "layout": "worker-4x1xNx128-planner-4x1xNx256",
    }


def rng_state(device):
    return {
        "python_numpy": pickle.dumps((random.getstate(), np.random.get_state())),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def restore_rng(state, device):
    python, numpy = pickle.loads(state["python_numpy"])
    random.setstate(python)
    np.random.set_state(numpy)
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def load(path):
    # The outer format uses safe torch primitives; worker blobs are trusted pickle.
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    if snapshot.get("version") != VERSION:
        raise ValueError("Unsupported/legacy checkpoint: exact resume requires format version 3")
    if snapshot["compatibility"] != compatibility():
        raise ValueError("Checkpoint source/runtime/recurrent layout is incompatible")
    return snapshot


def output_directory(explicit=None):
    if explicit:
        path = Path(explicit)
        path.mkdir(parents=True, exist_ok=False)
        return path
    root = ROOT / "results"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="train-", dir=root))


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(directory, snapshot, keep):
    if keep < 1:
        raise ValueError("Checkpoint retention must be positive")
    target = directory / f"checkpoint-{snapshot['progress']['update']:06d}.pt"
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(snapshot, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        sync_directory(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    for old in sorted(directory.glob("checkpoint-[0-9][0-9][0-9][0-9][0-9][0-9].pt"))[:-keep]:
        old.unlink()
    return target


def append_log(path, text):
    with path.open("ab") as stream:
        stream.write((text + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def log_offsets(directory):
    return {
        name: (directory / name).stat().st_size if (directory / name).exists() else 0
        for name in ("metrics.jsonl", "episodes.jsonl")
    }


def recover_logs(source, destination, offsets):
    """Copy just the committed prefix; leave the parent run untouched."""
    for name, length in offsets.items():
        if length < 0 or (
            length and (not (source / name).exists() or (source / name).stat().st_size < length)
        ):
            raise ValueError(f"Missing or truncated committed log: {name}")
        with (destination / name).open("wb") as output:
            if length:
                with (source / name).open("rb") as original:
                    remaining = length
                    while remaining:
                        block = original.read(min(remaining, 1024 * 1024))
                        if not block:
                            raise ValueError(f"Committed log truncated during recovery: {name}")
                        output.write(block)
                        remaining -= len(block)
            output.flush()
            os.fsync(output.fileno())
