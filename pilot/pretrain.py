"""Worker-pretraining screen: resumable arms, reference policies and a productivity gate.

Each arm is a phase-one free-market worker run in bounded segments. Every child
invocation has its own systemd scope and ten-minute timeout. Rerunning an
identical command continues completed work; failed directories are preserved.
This screens pretraining settings; it is not paper replication.
"""

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch
from environment import configuration
from experiment import ROOT, archive_source, dump, invoke
from torch.distributions import Categorical

WORKERS = 4
HORIZON = 200
ENV_STEPS_PER_UPDATE = WORKERS * HORIZON
SEGMENT_UPDATES = 50
EVAL_SEEDS_PER_INVOCATION = 5
MIN_AVAILABLE_RAM = 12 * 1024**3
UNIT_PREFIX = "ai-econ-pretrain"
# Two-sided 95% Student t for 9 degrees of freedom (ten held-out episodes).
T_95_N10 = 2.262
MIN_TRAINED_LABOR_WEIGHT = 0.95
HOLD_UPDATES = 300  # 240,000 environment steps of phase-two continuation.


class Paused(Exception):
    """The invocation's time budget ended between segments; rerun to continue."""


class ReferencePolicy:
    """Stateless masked policy with the evaluator's ActorCritic calling convention."""

    def __init__(self, kind):
        if kind not in ("uniform", "noop"):
            raise ValueError(f"Unknown reference policy: {kind}")
        self.kind = kind

    def initial_state(self, batch, device):
        return ()

    def __call__(self, observations, state, starts):
        mask = observations["action_mask"]
        logits = torch.zeros_like(mask).masked_fill(mask <= 0, -1e9)
        if self.kind == "noop":
            logits = torch.full_like(mask, -1e9)
            logits[..., 0] = 0
        return Categorical(logits=logits), torch.zeros(mask.shape[:2]), state


def reference_config():
    """The evaluator's stationary objective: full labor cost, free market."""
    config = configuration("free-market", "paper2020", WORKERS, False, None, 0)
    config["energy_warmup_constant"] = 0
    return config


def run_reference(kind, seeds, output):
    from evaluate import episode

    torch.set_num_threads(1)
    policy = ReferencePolicy(kind)
    records = []
    for seed in seeds:
        record = episode(policy, reference_config(), seed, "sample")
        record.pop("shapes")
        records.append(record)
    Path(output).write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))


def parse_arm(text):
    name, seed, labor, minibatch = text.split(":")
    if not name.replace("-", "").isalnum():
        raise ValueError(f"Arm name must be alphanumeric/hyphens: {name}")
    return {
        "name": name,
        "seed": int(seed),
        "labor_warmup_constant": float(labor),
        "minibatch_sequences": int(minibatch),
    }


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
    }


def episode_row(episodes):
    """Aggregate episode records from training or held-out evaluation."""
    metric = lambda e, key: e["metrics"][key]  # noqa: E731
    return {
        "episodes": len(episodes),
        "productivity": summarize([metric(e, "social/productivity") for e in episodes]),
        "return_mean": summarize([e["return_mean"] for e in episodes]),
        "labor_per_worker": float(
            np.mean([[metric(e, f"endogenous/{i}/Labor") for i in range(4)] for e in episodes])
        ),
        "builds": float(np.mean([metric(e, "Build/total_builds") for e in episodes])),
        "trades": float(np.mean([metric(e, "Trade/n_trades") for e in episodes])),
        "resources_gathered": float(np.mean([e["resources_gathered"] for e in episodes])),
    }


def gate(evaluations, uniform_productivity, noop_return):
    """Prespecified: sustained, full-cost productive behavior at the last two checkpoints.

    The return interval must exceed the measured no-op return, so production beats
    inactivity at full labor cost. A no-op return is slightly negative, not zero:
    Foundation charges collection labor when a resource respawns under a stationary
    agent. Productivity must also exceed the uniform-random reference, and the
    policy must have been trained under nearly full labor cost, not merely
    evaluated there.
    """
    checks = []
    for row in evaluations[-2:]:
        lower = row["return_mean"]["mean"] - T_95_N10 * row["return_mean"]["sem"]
        checks.append(
            {
                "environment_steps": row["environment_steps"],
                "return_lower_95": lower,
                "return_above_noop": lower > noop_return,
                "productivity_above_uniform": row["productivity"]["mean"] > uniform_productivity,
                "builds_occur": row["builds"] > 0,
                "trained_near_full_labor": row["trained_labor_weight"] >= MIN_TRAINED_LABOR_WEIGHT,
            }
        )
    passed = len(checks) == 2 and all(
        value for check in checks for key, value in check.items() if isinstance(value, bool)
    )
    return {"passed": passed, "checkpoints": checks}


def retire(directory):
    """Preserve an incomplete directory from an interrupted attempt."""
    if directory.exists():
        directory.rename(directory.with_name(f"{directory.name}.incomplete-{int(time.time())}"))


def guarded_invoke(root, label, arguments):
    if psutil.virtual_memory().available < MIN_AVAILABLE_RAM:
        raise RuntimeError("Available RAM fell below the 12 GiB launch guard")
    invoke(root, label, arguments, unit_prefix=UNIT_PREFIX)


def references(root, seeds):
    results = {}
    for kind in ("noop", "uniform"):
        episodes = []
        for start in range(0, len(seeds), EVAL_SEEDS_PER_INVOCATION):
            batch = seeds[start : start + EVAL_SEEDS_PER_INVOCATION]
            path = root / f"reference-{kind}-{batch[0]}.jsonl"
            if not path.exists():
                guarded_invoke(
                    root,
                    f"reference-{kind}-{batch[0]}",
                    [
                        "pilot/pretrain.py",
                        "--reference",
                        kind,
                        "--reference-output",
                        str(path),
                        "--seeds",
                        *map(str, batch),
                    ],
                )
            episodes += [json.loads(line) for line in path.read_text().splitlines()]
        results[kind] = episode_row(episodes)
    dump(root / "references.json", results)
    return results


def train_segment(root, arm, target, previous, device, phase2_from=None):
    name = f"{arm['name']}-{target:06d}"
    directory = root / name
    summary_path = directory / "summary.json"
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
        elif phase2_from is not None:
            arguments += ["--phase2-from", str(phase2_from), "--tax-regime", "free-market"]
        else:
            arguments += [
                "--device",
                device,
                "--profile",
                "paper2020",
                "--workers",
                str(WORKERS),
                "--seed",
                str(arm["seed"]),
                "--minibatch-sequences",
                str(arm["minibatch_sequences"]),
                "--labor-warmup-constant",
                repr(arm["labor_warmup_constant"]),
            ]
        guarded_invoke(root, name, arguments)
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "passed" or summary["completed_updates"] != target:
        raise RuntimeError(f"{name} failed protocol checks")
    return directory, directory / summary["checkpoint"]


def evaluate_checkpoint(root, arm, target, checkpoint, seeds):
    episodes = []
    for start in range(0, len(seeds), EVAL_SEEDS_PER_INVOCATION):
        batch = seeds[start : start + EVAL_SEEDS_PER_INVOCATION]
        label = f"eval-{arm['name']}-{target:06d}-{batch[0]}"
        directory = root / label
        if not (directory / "summary.json").exists():
            retire(directory)
            guarded_invoke(
                root,
                label,
                [
                    "pilot/evaluate.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--seeds",
                    *map(str, batch),
                    "--output",
                    str(directory),
                ],
            )
        episodes += [
            json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()
        ]
    return episodes


def training_curve(directory, block_updates=SEGMENT_UPDATES):
    """Summarize the child run's committed logs, which include all parent records."""
    episodes = [json.loads(x) for x in (directory / "episodes.jsonl").read_text().splitlines()]
    records = [json.loads(x) for x in (directory / "metrics.jsonl").read_text().splitlines()]
    curve = []
    for end in range(block_updates, records[-1]["update"] + 1, block_updates):
        block = [e for e in episodes if end - block_updates < e["update"] <= end]
        updates = [r for r in records if end - block_updates < r["update"] <= end]
        row = {
            "environment_steps": end * ENV_STEPS_PER_UPDATE,
            "entropy": float(np.mean([r["entropy"] for r in updates])),
            "approx_kl": float(np.mean([r["approx_kl"] for r in updates])),
            "explained_variance": float(np.mean([r["explained_variance"] or 0 for r in updates])),
            "action_shares": {
                key: float(np.mean([r["action_shares"][key] for r in updates]))
                for key in updates[0]["action_shares"]
            },
        }
        if block:
            row |= episode_row(block)
            row["labor_weight"] = float(np.mean([e["labor_weight"] for e in block]))
        curve.append(row)
    return curve


def run_arm(root, arm, steps, eval_every, seeds, device, reference, deadline):
    updates = steps // ENV_STEPS_PER_UPDATE
    eval_updates = eval_every // ENV_STEPS_PER_UPDATE
    previous = None
    directory = None
    evaluations = []
    for target in range(SEGMENT_UPDATES, updates + 1, SEGMENT_UPDATES):
        segment_done = (root / f"{arm['name']}-{target:06d}" / "summary.json").exists()
        if not segment_done and time.monotonic() > deadline:
            raise Paused(arm["name"])
        directory, previous = train_segment(root, arm, target, previous, device)
        if target % eval_updates == 0:
            episodes = evaluate_checkpoint(root, arm, target, previous, seeds)
            trained = [
                json.loads(x) for x in (directory / "episodes.jsonl").read_text().splitlines()
            ]
            evaluations.append(
                {
                    "environment_steps": target * ENV_STEPS_PER_UPDATE,
                    "checkpoint": str(previous),
                    "trained_labor_weight": float(
                        np.mean([e["labor_weight"] for e in trained[-WORKERS:]])
                    ),
                    **episode_row(episodes),
                    "action_shares": {
                        key: float(np.mean([e["action_shares"][key] for e in episodes]))
                        for key in episodes[0]["action_shares"]
                    },
                }
            )
    assert directory is not None
    report = {
        **arm,
        "steps": updates * ENV_STEPS_PER_UPDATE,
        "training_curve": training_curve(directory),
        "evaluations": evaluations,
        "gate": gate(
            evaluations,
            reference["uniform"]["productivity"]["mean"],
            reference["noop"]["return_mean"]["mean"],
        ),
    }
    dump(root / f"report-{arm['name']}.json", report)
    return report


def finalize_arm(root, name, final_seeds, reference):
    """Selected in advance: the arm's final checkpoint, new seeds, then a full-cost hold.

    The hold enters phase two through the real transition: fresh economies and
    recurrent state, transferred model and Adam state, labor weight exactly one.
    """
    report = json.loads((root / f"report-{name}.json").read_text())
    source = Path(report["evaluations"][-1]["checkpoint"])
    uniform = reference["uniform"]["productivity"]["mean"]
    noop = reference["noop"]["return_mean"]["mean"]
    pretrained = episode_row(evaluate_checkpoint(root, {"name": name}, 0, source, final_seeds))
    hold = {"name": f"hold-{name}"}
    previous = None
    evaluations = []
    for target in range(SEGMENT_UPDATES, HOLD_UPDATES + 1, SEGMENT_UPDATES):
        phase2_from = None if previous is not None else source
        directory, previous = train_segment(root, hold, target, previous, None, phase2_from)
        if target % (HOLD_UPDATES // 2) == 0:
            episodes = evaluate_checkpoint(root, hold, target, previous, final_seeds)
            trained = [
                json.loads(x) for x in (directory / "episodes.jsonl").read_text().splitlines()
            ]
            evaluations.append(
                {
                    "environment_steps": target * ENV_STEPS_PER_UPDATE,
                    "checkpoint": str(previous),
                    "trained_labor_weight": float(
                        np.mean([e["labor_weight"] for e in trained[-WORKERS:]])
                    ),
                    **episode_row(episodes),
                }
            )
    final = {
        "arm": name,
        "source_checkpoint": str(source),
        "final_seeds": final_seeds,
        "pretrained": pretrained,
        "pretrained_above_noop": pretrained["return_mean"]["mean"]
        - T_95_N10 * pretrained["return_mean"]["sem"]
        > noop,
        "pretrained_above_uniform": pretrained["productivity"]["mean"] > uniform,
        "hold": evaluations,
        "hold_curve": training_curve(directory),
        "hold_gate": gate(evaluations, uniform, noop),
    }
    dump(root / f"final-{name}.json", final)
    print(f"FINAL {name}: hold gate {final['hold_gate']['passed']}", flush=True)


def extend_protocol(existing, requested):
    """Allow more steps and new arms, but never redefine settings of an existing arm."""
    fixed = [k for k in requested if k not in ("arms", "steps")]
    if any(existing[k] != requested[k] for k in fixed):
        raise ValueError("Existing output has a different protocol")
    if requested["steps"] < existing["steps"]:
        raise ValueError("A protocol cannot reduce its step target")
    arms = {arm["name"]: arm for arm in existing["arms"]}
    for arm in requested["arms"]:
        if arms.setdefault(arm["name"], arm) != arm:
            raise ValueError(f"Arm {arm['name']} already has different settings")
    return {**existing, "steps": requested["steps"], "arms": list(arms.values())}


def main(args):
    deadline = time.monotonic() + args.max_seconds
    root = Path(args.output).resolve()
    seeds = list(range(args.first_eval_seed, args.first_eval_seed + 10))
    protocol = {
        "arms": [parse_arm(text) for text in args.arm],
        "steps": args.steps,
        "eval_every": args.eval_every,
        "device": args.device,
        "workers": WORKERS,
        "horizon": HORIZON,
        "profile": "paper2020",
        "evaluation_seeds": seeds,
        "gate": "See pilot/pretrain.py:gate; last two checkpoints, full labor cost, sampled actions.",
    }
    if args.steps % (SEGMENT_UPDATES * ENV_STEPS_PER_UPDATE) or args.eval_every % (
        SEGMENT_UPDATES * ENV_STEPS_PER_UPDATE
    ):
        raise ValueError("Steps and evaluation interval must align with 40,000-step segments")
    if root.exists():
        # Earlier arms stay on record; this invocation runs only the requested arms.
        existing = json.loads((root / "protocol.json").read_text())
        dump(root / "protocol.json", extend_protocol(existing, protocol))
    else:
        root.mkdir(parents=True)
        dump(root / "protocol.json", protocol)
        archive_source(root)
    reference = references(root, seeds)
    if args.finalize:
        final_seeds = list(range(args.final_eval_seed, args.final_eval_seed + 10))
        with ThreadPoolExecutor(args.concurrency) as pool:
            for future in [
                pool.submit(finalize_arm, root, name, final_seeds, reference)
                for name in args.finalize
            ]:
                future.result()
        return
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = [
            pool.submit(
                run_arm,
                root,
                arm,
                args.steps,
                args.eval_every,
                seeds,
                args.device,
                reference,
                deadline,
            )
            for arm in protocol["arms"]
        ]
        try:
            reports = [future.result() for future in futures]
        except Paused:
            for future in futures:
                future.exception()
            failures = [f.exception() for f in futures if not isinstance(f.exception(), Paused)]
            if any(failures):
                raise next(f for f in failures if f) from None
            print(f"PAUSED {root}; rerun the same command to continue", flush=True)
            return
    dump(root / "report.json", {"references": reference, "arms": reports})
    print(f"COMPLETE {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="name:seed:labor_warmup_constant:minibatch_sequences",
    )
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--eval-every", type=int, default=200_000)
    parser.add_argument("--first-eval-seed", type=int, default=20000, help="Ten consecutive seeds")
    parser.add_argument("--finalize", nargs="+", help="Arms to evaluate on final seeds and hold")
    parser.add_argument("--final-eval-seed", type=int, default=30000, help="Ten consecutive seeds")
    parser.add_argument("--concurrency", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--max-seconds", type=float, default=1e9, help="Launch no new segment after this time"
    )
    parser.add_argument("--reference", choices=("noop", "uniform"))
    parser.add_argument("--reference-output")
    parser.add_argument("--seeds", type=int, nargs="+")
    arguments = parser.parse_args()
    if arguments.reference:
        sys.path.insert(0, str(ROOT / "pilot"))
        run_reference(arguments.reference, arguments.seeds, arguments.reference_output)
    else:
        if shutil.disk_usage(ROOT).free < 15 * 1024**3:
            raise RuntimeError("Less than 15 GiB disk reserve")
        main(arguments)
