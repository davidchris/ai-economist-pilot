"""Matched phase-two tax comparison from gated pretraining checkpoints.

Prespecified: every treatment continues each seed's final pretraining checkpoint through
the real phase transition, for the same steps, with the same tax-ceiling setting. Both
tax treatments ramp under that ceiling. Policies are evaluated on common held-out
episode seeds at the end of the ramp and at the end of the run.

Every child invocation has its own systemd scope and ten-minute timeout. Rerunning an
identical command continues completed work. A failed arm stops alone; the others go on.
This compares bounded policies; it is not paper replication.
"""

import argparse
import hashlib
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from experiment import REGIMES, ROOT, archive_source, dump
from pretrain import (
    ENV_STEPS_PER_UPDATE,
    SEGMENT_UPDATES,
    WORKERS,
    Paused,
    episode_row,
    evaluate_checkpoint,
    gate,
    guarded_invoke,
    references,
    retire,
    summarize,
    training_curve,
)

PRETRAINING_ARMS = {42: "labor40k", 142: "labor40k-s142", 242: "labor40k-s242"}
PRIMARY_METRICS = ("welfare", "productivity", "equality", "taxes_collected", "return_mean")
CONTRASTS = (
    ("learned-tax", "free-market"),
    ("learned-tax", "fixed-tax"),
    ("fixed-tax", "free-market"),
)
# Two-sided 95% Student t for 2 degrees of freedom (three training seeds).
T_95_N3 = 4.303
CRITIC_CHECK = re.compile(r"assert_close\((?:gpu_)?values\.cpu\(\), cpu_values")


def source_checkpoints(pretraining):
    """Each seed's checkpoint, only if its prespecified pretraining gate passed."""
    sources = {}
    for seed, name in PRETRAINING_ARMS.items():
        final = json.loads((Path(pretraining) / f"final-{name}.json").read_text())
        if not final["hold_gate"]["passed"]:
            raise ValueError(f"Pretraining arm {name} did not pass its gate")
        path = Path(final["source_checkpoint"])
        sources[seed] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return sources


def accepted_failure(root, name, directory, target):
    """Return a published checkpoint whose failure is an approved critic check.

    Rerunning would repeat the failure, because restarts are exact. Approval names
    segments, in advance for an arm already found collapsed, and covers no other failure.
    The worker policy check precedes this one; later checks do not run, so accepted
    checkpoints must have every check replayed afterwards.
    """
    log = (root / f"{name}.log").read_text()
    path = directory / f"checkpoint-{target:06d}.pt"
    if "Tensor-likes are not close" not in log or not CRITIC_CHECK.search(log) or not path.exists():
        raise RuntimeError(f"{name} is not a critic agreement failure with a checkpoint")
    return path


def train_segment(root, arm, target, previous, accepted):
    name = f"{arm['name']}-{target:06d}"
    directory = root / name
    summary_path = directory / "summary.json"
    if name in accepted and directory.exists() and not summary_path.exists():
        try:
            return directory, accepted_failure(root, name, directory, target)
        except RuntimeError:
            pass  # Interrupted, not rejected. An exact rerun decides; other failures still stop.
    if not summary_path.exists():
        retire(directory)
        arguments = [
            "pilot/train.py",
            "--updates",
            str(target),
            "--checkpoint-every",
            str(SEGMENT_UPDATES),
            "--keep-checkpoints",
            "1",
            "--output",
            str(directory),
        ]
        if previous is not None:
            arguments += ["--resume", str(previous)]
        else:
            arguments += [
                "--device",
                arm["device"],
                "--phase2-from",
                arm["source"],
                "--tax-regime",
                arm["regime"],
                "--tax-anneal-steps",
                str(arm["tax_anneal_steps"]),
            ]
        try:
            guarded_invoke(root, name, arguments)
        except RuntimeError:
            if name not in accepted:
                raise
            path = accepted_failure(root, name, directory, target)
            print(f"ACCEPTED critic agreement failure {name}", flush=True)
            return directory, path
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "passed" or summary["completed_updates"] != target:
        raise RuntimeError(f"{name} failed protocol checks")
    return directory, directory / summary["checkpoint"]


def evaluation_row(episodes):
    schedules = [
        schedule["marginal_rates"] for episode in episodes for schedule in episode["tax_schedules"]
    ]
    return {
        **episode_row(episodes),
        **{key: summarize([e[key] for e in episodes]) for key in PRIMARY_METRICS[0:4]},
        "labor_weight": float(np.mean([e["labor_weight"] for e in episodes])),
        "mean_marginal_rates": np.mean(schedules, axis=0).tolist() if schedules else None,
        "action_shares": {
            key: float(np.mean([e["action_shares"][key] for e in episodes]))
            for key in episodes[0]["action_shares"]
        },
        "per_seed": {str(e["seed"]): {key: e[key] for key in PRIMARY_METRICS} for e in episodes},
    }


def monitoring_curve(directory):
    """Per-segment training curve, adding the tax ceiling and planner diagnostics."""
    curve = training_curve(directory)
    episodes = [json.loads(x) for x in (directory / "episodes.jsonl").read_text().splitlines()]
    records = [json.loads(x) for x in (directory / "metrics.jsonl").read_text().splitlines()]
    for row in curve:
        end = row["environment_steps"] // ENV_STEPS_PER_UPDATE
        begin = end - SEGMENT_UPDATES
        block = [e for e in episodes if begin < e["update"] <= end]
        planners = [r["planner"] for r in records if begin < r["update"] <= end and "planner" in r]
        if block:
            row["tax_ceiling"] = float(np.mean([e["tax_ceiling"] for e in block]))
        if planners:
            row["planner_entropy"] = float(np.mean([p["entropy"] for p in planners]))
            row["planner_reward"] = float(np.mean([p["reward_mean"] for p in planners]))
    return curve


def run_arm(root, arm, seeds, reference, accepted, deadline):
    updates = arm["steps"] // ENV_STEPS_PER_UPDATE
    evaluate_at = {steps // ENV_STEPS_PER_UPDATE for steps in arm["evaluation_steps"]}
    previous = None
    directory = None
    evaluations = []
    for target in range(SEGMENT_UPDATES, updates + 1, SEGMENT_UPDATES):
        name = f"{arm['name']}-{target:06d}"
        published = root / name / f"checkpoint-{target:06d}.pt"
        done = (root / name / "summary.json").exists() or (name in accepted and published.exists())
        if not done and time.monotonic() > deadline:
            raise Paused(arm["name"])
        directory, previous = train_segment(root, arm, target, previous, accepted)
        if target in evaluate_at:
            row = evaluation_row(evaluate_checkpoint(root, arm, target, previous, seeds))
            evaluations.append(
                {
                    "environment_steps": target * ENV_STEPS_PER_UPDATE,
                    "checkpoint": str(previous),
                    "trained_labor_weight": row["labor_weight"],
                    **row,
                }
            )
    assert directory is not None
    report = {
        **arm,
        "training_curve": monitoring_curve(directory),
        "evaluations": evaluations,
        "gate": gate(
            evaluations,
            reference["uniform"]["productivity"]["mean"],
            reference["noop"]["return_mean"]["mean"],
        ),
    }
    dump(root / f"report-{arm['name']}.json", report)
    return report


def contrasts(reports):
    """Paired differences by episode seed, then variation across training seeds.

    `reports[seed][regime]["evaluations"]` holds common-seed rows in the same order.
    The interval uses three training seeds, so it is wide and only descriptive.
    """
    results = []
    any_seed = next(iter(reports))
    points = [row["environment_steps"] for row in reports[any_seed][REGIMES[0]]["evaluations"]]
    for index, steps in enumerate(points):
        for treated, control in CONTRASTS:
            for metric in PRIMARY_METRICS:
                by_seed = {}
                for seed, arms in reports.items():
                    left = arms[treated]["evaluations"][index]["per_seed"]
                    right = arms[control]["evaluations"][index]["per_seed"]
                    if left.keys() != right.keys():
                        raise ValueError("Contrasts require common evaluation seeds")
                    by_seed[seed] = summarize([left[k][metric] - right[k][metric] for k in left])
                means = np.array([row["mean"] for row in by_seed.values()])
                half = (
                    T_95_N3 * means.std(ddof=1) / np.sqrt(len(means)) if len(means) == 3 else None
                )
                results.append(
                    {
                        "environment_steps": steps,
                        "contrast": f"{treated} minus {control}",
                        "metric": metric,
                        "by_training_seed": {str(k): v for k, v in by_seed.items()},
                        "mean": float(means.mean()),
                        "interval_95": None
                        if half is None
                        else [float(means.mean() - half), float(means.mean() + half)],
                        "same_sign_all_seeds": bool((means > 0).all() or (means < 0).all()),
                    }
                )
    return results


def build_arms(args, sources):
    return [
        {
            "name": f"{regime}-s{seed}",
            "regime": regime,
            "seed": seed,
            "source": sources[seed]["path"],
            "steps": args.steps,
            "evaluation_steps": args.evaluate_at,
            "tax_anneal_steps": args.tax_anneal_steps,
            "device": args.device,
        }
        for seed in args.training_seeds
        for regime in REGIMES
    ]


def main(args):
    deadline = time.monotonic() + args.max_seconds
    segment = SEGMENT_UPDATES * ENV_STEPS_PER_UPDATE
    if args.steps % segment or any(s % segment or s > args.steps for s in args.evaluate_at):
        raise ValueError("Steps and evaluation points must align with 40,000-step segments")
    if args.steps not in args.evaluate_at or len(args.evaluate_at) < 2:
        raise ValueError("Evaluate the final checkpoint and at least one earlier point")
    root = Path(args.output).resolve()
    sources = source_checkpoints(args.pretraining)
    seeds = list(range(args.first_eval_seed, args.first_eval_seed + 10))
    arms = build_arms(args, sources)
    protocol = {
        "arms": arms,
        "sources": sources,
        "workers": WORKERS,
        "evaluation_seeds": seeds,
        "primary_metric": "welfare",
        "contrasts": [f"{a} minus {b}" for a, b in CONTRASTS],
        "interpretation": "Bounded three-seed comparison; not a convergence or paper claim.",
    }
    if root.exists():
        # Compare in JSON form: the saved file has string seed keys.
        if json.loads((root / "protocol.json").read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError("Existing output has a different protocol")
    else:
        root.mkdir(parents=True)
        dump(root / "protocol.json", protocol)
        archive_source(root)
    if args.accept_critic_failure:
        approvals = root / "approved-critic-failures.jsonl"
        with approvals.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "segments": args.accept_critic_failure,
                    }
                )
                + "\n"
            )
    accepted = set(args.accept_critic_failure)
    reference = references(root, seeds)
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = {
            arm["name"]: pool.submit(run_arm, root, arm, seeds, reference, accepted, deadline)
            for arm in arms
        }
    errors = {name: future.exception() for name, future in futures.items()}
    failed = {k: v for k, v in errors.items() if v and not isinstance(v, Paused)}
    for name, error in failed.items():
        print(f"FAILED {name}: {error}", flush=True)
    if failed:
        raise SystemExit(1)
    if any(errors.values()):
        print(f"PAUSED {root}; rerun the same command to continue", flush=True)
        return
    reports = {}
    for arm in arms:
        reports.setdefault(arm["seed"], {})[arm["regime"]] = futures[arm["name"]].result()
    baseline = {}
    for seed, source in sources.items():
        if seed in args.training_seeds:
            arm = {"name": f"pretrained-s{seed}"}
            baseline[str(seed)] = evaluation_row(
                evaluate_checkpoint(root, arm, 0, source["path"], seeds)
            )
    dump(
        root / "report.json",
        {
            "references": reference,
            "pretrained_baseline": baseline,
            "gates": {arm["name"]: futures[arm["name"]].result()["gate"] for arm in arms},
            "contrasts": contrasts(reports),
        },
    )
    print(f"COMPLETE {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretraining", default=str(ROOT / "results/pretrain-stage-b-20261104"))
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=list(PRETRAINING_ARMS),
        choices=[42, 142, 242],
    )
    parser.add_argument("--steps", type=int, default=4_400_000)
    parser.add_argument("--evaluate-at", type=int, nargs="+", default=[3_600_000, 4_400_000])
    parser.add_argument("--tax-anneal-steps", type=int, default=3_600_000)
    parser.add_argument("--first-eval-seed", type=int, default=40000, help="Ten consecutive seeds")
    parser.add_argument("--concurrency", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--max-seconds", type=float, default=1e9, help="Launch no new segment after this time"
    )
    parser.add_argument(
        "--accept-critic-failure",
        action="extend",
        nargs="+",
        default=[],
        help="Segment names that may continue past a critic agreement failure only",
    )
    arguments = parser.parse_args()
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError("Less than 15 GiB disk reserve")
    main(arguments)
