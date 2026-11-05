"""Held-out, full-episode CPU evaluation of trusted worker and optional planner checkpoints."""

import argparse
import copy
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

import checkpoint
import numpy as np
import torch
from environment import (
    ROOT,
    action_shares,
    configuration_from_run,
    pack,
    pack_planner,
    resources_gathered,
)
from model import ActorCritic, PlannerActorCritic

METRICS = {
    "productivity": "social/productivity",
    "equality": "social/equality",
    "welfare": "social_welfare/coin_eq_times_productivity",
}


def evaluation_config(saved):
    expected = configuration_from_run(saved["config"])
    if saved["config"]["environment"] != expected:
        raise ValueError("Unsupported checkpoint environment")
    config = copy.deepcopy(expected)
    # A common, stationary economic objective across checkpoints and episodes.
    config["energy_warmup_constant"] = 0
    tax = next(c["PeriodicBracketTax"] for c in config["components"] if "PeriodicBracketTax" in c)
    tax["tax_annealing_schedule"] = None
    return config


def training_seeds(saved):
    config = saved["config"]
    seeds = set(range(config["seed"], config["seed"] + config["workers"]))
    # Phase two may override the pretraining seed/worker count. Require its
    # provenance checkpoint rather than silently calling a pretraining seed held-out.
    if config.get("phase2_from"):
        seeds.update(training_seeds(checkpoint.load(config["phase2_from"])))
    return sorted(seeds)


def validate_seeds(saved, seeds):
    if not 1 <= len(seeds) <= 20 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide 1..20 distinct held-out seeds")
    used_seeds = set(training_seeds(saved))
    if any(seed < 1 or seed >= 2**32 for seed in seeds):
        raise ValueError("Seeds must be in 1..2**32-1")
    if used_seeds.intersection(seeds):
        raise ValueError("Held-out seeds must exclude all training worker seeds")


def check_disk(path):
    if shutil.disk_usage(path).free < 15 * 1024**3:
        raise RuntimeError("Evaluation requires at least 15 GiB free")


@torch.inference_mode()
def episode(model, config, seed, action_mode, planner=None, expected_shapes=None):
    from ai_economist import foundation

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    foundation.scenarios.get(config["scenario_name"]).seed(seed)
    env = foundation.make_env_instance(**copy.deepcopy(config))
    raw = env.reset()
    observation = pack(raw)
    shapes = {key: list(value.shape[1:]) for key, value in observation.items()}
    planner_observation = pack_planner(raw) if planner is not None else None
    planner_shapes = (
        {key: list(value.shape[1:]) for key, value in planner_observation.items()}
        if planner_observation is not None
        else None
    )
    if expected_shapes is not None and (shapes, planner_shapes) != expected_shapes:
        raise ValueError("Evaluation observation layout differs from checkpoint")
    if planner is not None and env.world.planner.action_spaces.tolist() != planner.action_sizes:
        raise ValueError("Evaluation planner action layout differs from checkpoint")
    planner_state = planner.initial_state(1, "cpu") if planner is not None else None
    planner_starts = torch.ones(1, 1)
    planner_return = 0.0
    tax_schedules = []
    tax = env.get_component("PeriodicBracketTax")
    rates = np.zeros(tax.n_brackets) if tax.disable_taxes else tax.curr_marginal_rates.copy()
    state = model.initial_state(4, "cpu")
    starts = torch.ones(1, 4)
    returns = np.zeros(4)
    action_counts = np.zeros(shapes["action_mask"][0], dtype=np.int64)
    for step in range(config["episode_length"]):
        tensors = {key: torch.from_numpy(value).unsqueeze(0) for key, value in observation.items()}
        distribution, values, state = model(tensors, state, starts)
        if not torch.isfinite(values).all() or not torch.isfinite(distribution.probs).all():
            raise ValueError("Nonfinite policy output")
        actions = (
            distribution.sample() if action_mode == "sample" else distribution.probs.argmax(-1)
        )[0].numpy()
        if not (observation["action_mask"][np.arange(4), actions] > 0).all():
            raise ValueError("Invalid masked action")
        np.add.at(action_counts, actions, 1)
        joint_actions = {str(i): int(a) for i, a in enumerate(actions)}
        if planner is not None:
            assert planner_observation is not None
            tensors = {
                key: torch.from_numpy(value).unsqueeze(0)
                for key, value in planner_observation.items()
            }
            distribution, values, planner_state = planner(tensors, planner_state, planner_starts)
            if not torch.isfinite(values).all() or not torch.isfinite(distribution.probs).all():
                raise ValueError("Nonfinite planner policy output")
            chosen = (
                distribution.sample()
                if action_mode == "sample"
                else torch.stack([part.probs.argmax(-1) for part in distribution.parts], dim=-1)
            )[0, 0].numpy()
            offset = 0
            for action, size in zip(chosen, planner.action_sizes, strict=True):
                if (
                    not 0 <= action < size
                    or planner_observation["action_mask"][0, offset + action] <= 0
                ):
                    raise ValueError("Invalid masked planner action")
                offset += size
            joint_actions["p"] = chosen
        new_period = tax.tax_cycle_pos == 1
        raw, rewards, done, _ = env.step(joint_actions)
        if new_period:
            tax_schedules.append(
                {
                    "start_step": step + 1,
                    "marginal_rates": (
                        np.zeros(tax.n_brackets) if tax.disable_taxes else tax.curr_marginal_rates
                    ).tolist(),
                }
            )
        if planner is not None:
            if not np.isfinite(rewards["p"]):
                raise ValueError("Nonfinite planner reward")
            planner_return += float(rewards["p"])
            planner_observation = pack_planner(raw)
            planner_starts.zero_()
        reward = np.array([rewards[str(i)] for i in range(4)])
        if not np.isfinite(reward).all():
            raise ValueError("Nonfinite reward")
        returns += reward
        observation = pack(raw)
        starts.zero_()
        if done["__all__"]:
            if step + 1 != config["episode_length"]:
                raise ValueError("Unexpected episode termination")
            break
    else:
        raise ValueError("Episode did not terminate")
    raw_metrics = env.metrics
    economic = {name: float(raw_metrics[key]) for name, key in METRICS.items()}
    economic["taxes_collected"] = float(tax.total_collected_taxes)
    economic["return_mean"] = float(returns.mean())
    if not all(np.isfinite(value) for value in economic.values()):
        raise ValueError("Nonfinite economic metric")
    # Some upstream diagnostics (e.g. averages with no trades) are undefined.
    metrics = {
        key: float(value) if np.isfinite(value) else None for key, value in raw_metrics.items()
    }
    return {
        "seed": seed,
        "steps": step + 1,
        "returns": returns.tolist(),
        **economic,
        "tax_model": tax.tax_model,
        "disable_taxes": tax.disable_taxes,
        "bracket_cutoffs": tax.bracket_cutoffs.tolist(),
        "marginal_rates": rates.tolist() if planner is None else None,
        "tax_schedules": tax_schedules,
        "planner_return": planner_return if planner is not None else None,
        "labor_weight": float(env.energy_weight),
        "resources_gathered": resources_gathered(env),
        "action_shares": action_shares(action_counts),
        "metrics": metrics,
        "shapes": shapes,
    }


def evaluate(source, seeds, output=None, action_mode="sample"):
    if action_mode not in ("sample", "greedy"):
        raise ValueError("Unknown action mode")
    saved = checkpoint.load(source)
    validate_seeds(saved, seeds)
    config = evaluation_config(saved)
    torch.set_num_threads(1)
    model = ActorCritic(saved["config"]["shapes"])
    model.load_state_dict(saved["model"])
    model.eval()
    model.requires_grad_(False)
    planner = None
    if saved["config"]["tax_regime"] == "learned-tax":
        if not saved.get("planner"):
            raise ValueError("Joint checkpoint is missing planner policy")
        planner = PlannerActorCritic(
            saved["config"]["planner_shapes"], saved["config"]["planner_action_sizes"]
        )
        planner.load_state_dict(saved["planner"]["model"])
        planner.eval()
        planner.requires_grad_(False)
    root = ROOT / "results"
    root.mkdir(exist_ok=True)
    check_disk(root if output is None else Path(output).parent)
    directory = (
        checkpoint.output_directory(output)
        if output is not None
        else Path(tempfile.mkdtemp(prefix="eval-", dir=root))
    )
    metadata = {
        "source_checkpoint": str(Path(source).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "compatibility": checkpoint.compatibility(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_progress": saved["progress"],
        "training_profile": saved["config"].get("profile", "pilot"),
        "phase_one_progress": saved["config"].get("phase_one_progress"),
        "tax_regime": saved["config"]["tax_regime"],
        "training_seeds": training_seeds(saved),
        "planner_evaluated": planner is not None,
        "tax_policy": "full-rate-range-no-annealing",
        "schema_version": 2,
        "seeds": seeds,
        "action_mode": action_mode,
        "device": "cpu",
        "labor_policy": "full-cost",
        "environment": config,
    }
    (directory / "config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    records = []
    for seed in seeds:
        check_disk(directory)
        record = episode(
            model,
            config,
            seed,
            action_mode,
            planner,
            (saved["config"]["shapes"], saved["config"].get("planner_shapes")),
        )
        if record.pop("shapes") != saved["config"]["shapes"]:
            raise ValueError("Evaluation observation layout differs from checkpoint")
        checkpoint.append_log(directory / "episodes.jsonl", json.dumps(record, allow_nan=False))
        records.append(record)
    aggregates = {}
    for key in (
        *METRICS,
        "taxes_collected",
        "return_mean",
        *(("planner_return",) if planner is not None else ()),
    ):
        values = np.array([record[key] for record in records])
        aggregates[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
        }
    summary = {
        "episodes": len(records),
        "environment_steps": sum(record["steps"] for record in records),
        "aggregates": aggregates,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Evaluation directory: {directory}", flush=True)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Trusted current-format worker or joint checkpoint"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[10000, 10001])
    parser.add_argument("--action-mode", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--output")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.seeds, args.output, args.action_mode)
