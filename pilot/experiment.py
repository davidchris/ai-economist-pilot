"""Run the prespecified matched diagnostic in sequential, resource-limited segments.

One training seed, 200k phase-one steps, then 200k per phase-two treatment.
This tests learning progress with compressed curricula; it is not paper replication.
Every child invocation has its own systemd scope and ten-minute timeout.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("free-market", "fixed-tax", "learned-tax")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resources():
    readings = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for label, command in (
        ("disk", ["df", "-h", str(ROOT)]),
        ("ram", ["free", "-h"]),
        ("load", ["uptime"]),
        (
            "gpu",
            ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
        ),
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        readings[label] = {"exit_code": result.returncode, "output": result.stdout + result.stderr}
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError("Less than 15 GiB disk reserve")
    return readings


def invoke(root, label, arguments, unit_prefix="ai-econ-medium"):
    dump(root / f"{label}.resources.json", resources())
    unit = f"{unit_prefix}-{os.getpid()}-{label}"
    command = [
        "systemd-run",
        "--user",
        "--scope",
        f"--unit={unit}",
        "-p",
        "CPUQuota=200%",
        "-p",
        "MemoryMax=8G",
        "-p",
        "MemorySwapMax=0",
        "timeout",
        "--signal=TERM",
        "--kill-after=10s",
        "600s",
        "env",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "uv",
        "run",
        "--locked",
        "--no-sync",
        "python",
        *arguments,
    ]
    with (root / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"label": label, "command": command}) + "\n")
    started = time.monotonic()
    print(f"START {label}", flush=True)
    with (root / f"{label}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    dump(
        root / f"{label}.exit.json",
        {
            "returncode": result.returncode,
            "seconds": time.monotonic() - started,
        },
    )
    if result.returncode:
        raise RuntimeError(f"{label} failed; inspect {root / f'{label}.log'}")
    print(f"PASS {label}: {time.monotonic() - started:.1f}s", flush=True)


def train_phase(root, label, device, phase1=None, regime="free-market"):
    source = None
    selected = {}
    # 25 updates = 20,000 environment steps. Preserve both 100k/200k checkpoints.
    for target in range(25, 251, 25):
        name = f"{label}-{target:06d}"
        directory = root / name
        arguments = [
            "pilot/train.py",
            "--device",
            device,
            "--updates",
            str(target),
            "--checkpoint-every",
            "25",
            "--keep-checkpoints",
            "2",
            "--output",
            str(directory),
        ]
        if source is not None:
            arguments += ["--resume", str(source)]
        elif phase1 is not None:
            arguments += ["--phase2-from", str(phase1), "--tax-regime", regime]
        else:
            arguments += [
                "--profile",
                "paper2020",
                "--workers",
                "4",
                "--seed",
                "42",
                "--minibatch-sequences",
                "4",
                "--tax-anneal-steps",
                "100000",
                "--labor-warmup-constant",
                "100",
            ]
        invoke(root, name, arguments)
        summary = json.loads((directory / "summary.json").read_text())
        if summary["status"] != "passed" or summary["environment_steps"] != target * 800:
            raise RuntimeError("Training summary failed protocol checks")
        source = directory / summary["checkpoint"]
        if target in (125, 250):
            selected[target] = source
    return selected


def report(root, selected):
    records = []
    aggregates = []
    for name in selected:
        episodes = []
        for start in range(10000, 10010, 2):
            directory = root / f"eval-{name}-{start}"
            episodes += [
                json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()
            ]
        for episode in episodes:
            records.append(
                {
                    "policy": name,
                    **{
                        k: episode[k]
                        for k in (
                            "seed",
                            "productivity",
                            "equality",
                            "welfare",
                            "taxes_collected",
                            "return_mean",
                            "labor_weight",
                        )
                    },
                }
            )
        row = {"policy": name, "episodes": len(episodes)}
        for metric in ("productivity", "equality", "welfare", "taxes_collected", "return_mean"):
            values = [e[metric] for e in episodes]
            mean = sum(values) / len(values)
            row[metric] = {
                "mean": mean,
                "sd": (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5,
            }
        aggregates.append(row)
    with (root / "evaluation-records.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    dump(root / "aggregates.json", aggregates)
    indexed = {(row["policy"], row["seed"]): row for row in records}
    differences = []
    for update in (125, 250):
        for baseline in ("free-market", "fixed-tax"):
            for seed in range(10000, 10010):
                learned = indexed[(f"learned-tax-{update}", seed)]
                control = indexed[(f"{baseline}-{update}", seed)]
                differences.append(
                    {
                        "phase_steps": update * 800,
                        "seed": seed,
                        "contrast": f"learned-tax minus {baseline}",
                        **{
                            key: learned[key] - control[key]
                            for key in (
                                "welfare",
                                "productivity",
                                "equality",
                                "taxes_collected",
                                "return_mean",
                            )
                        },
                    }
                )
    dump(root / "paired-differences.json", differences)
    curves = []
    for phase in ("pretrain", *REGIMES):
        directory = root / f"{phase}-000250"
        episodes = [
            json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()
        ]
        for end in range(25, 251, 25):
            block = [e for e in episodes if end - 25 < e["update"] <= end]
            curves.append(
                {
                    "phase": phase,
                    "environment_steps": end * 800,
                    "episodes": len(block),
                    "welfare": sum(
                        e["metrics"]["social_welfare/coin_eq_times_productivity"] for e in block
                    )
                    / len(block),
                    "productivity": sum(e["metrics"]["social/productivity"] for e in block)
                    / len(block),
                    "labor_weight": sum(e["labor_weight"] for e in block) / len(block),
                    "tax_ceiling": sum(e["tax_ceiling"] for e in block) / len(block),
                }
            )
    dump(root / "training-curves.json", curves)


def archive_source(root):
    """Record the exact Python source, dependency lock and working-tree diff for this run."""
    (root / "implementation.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    )
    archive = root / "source"
    archive.mkdir()
    for path in (ROOT / "pilot").glob("*.py"):
        shutil.copy2(path, archive / path.name)
    for name in ("uv.lock", "pyproject.toml"):
        shutil.copy2(ROOT / name, archive / name)
    dump(
        root / "provenance.json",
        {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "sources": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.iterdir()
            },
        },
    )


def main(output, device):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    protocol = {
        "training_seed": 42,
        "workers": 4,
        "profile": "paper2020",
        "device": device,
        "pretraining_steps": 200000,
        "phase2_steps_per_regime": 200000,
        "evaluation_phase_steps": [100000, 200000],
        "evaluation_seeds": list(range(10000, 10010)),
        "tax_anneal_steps": 100000,
        "labor_warmup_constant": 100,
        "minibatch_transitions": 200,
        "regimes": REGIMES,
        "interpretation": "Single-training-seed diagnostic with compressed curricula and reduced batches; not a convergence guarantee or paper replication.",
    }
    dump(root / "protocol.json", protocol)
    archive_source(root)
    pretraining = train_phase(root, "pretrain", device)
    selected = {f"pretrain-{update}": path for update, path in pretraining.items()}
    for regime in REGIMES:
        policies = train_phase(root, regime, device, pretraining[250], regime)
        selected.update({f"{regime}-{update}": path for update, path in policies.items()})
    manifest = {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in selected.items()
    }
    dump(root / "selected-checkpoints.json", manifest)
    invoke(
        root,
        "curriculum-audit",
        [
            "pilot/audit.py",
            "--manifest",
            str(root / "selected-checkpoints.json"),
            "--output",
            str(root / "curriculum-audit.json"),
        ],
    )
    for name, source in selected.items():
        for start in range(10000, 10010, 2):
            label = f"eval-{name}-{start}"
            invoke(
                root,
                label,
                [
                    "pilot/evaluate.py",
                    "--checkpoint",
                    str(source),
                    "--seeds",
                    str(start),
                    str(start + 1),
                    "--output",
                    str(root / label),
                ],
            )
    for name, source in selected.items():
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest[name]["sha256"]:
            raise RuntimeError("Evaluation changed checkpoint")
    report(root, selected)
    dump(root / "final-resources.json", resources())
    print(f"COMPLETE {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    main(args.output, args.device)
