"""Held-out evaluation and explicit Foundation fixed-tax acceptance checks."""

import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import checkpoint
import evaluate as evaluation
import numpy as np
import pytest
from environment import configuration
from evaluate import evaluate, validate_seeds
from test_resume import equal_state, run_train


@pytest.mark.parametrize("regime", ["free-market", "fixed-tax"])
def test_tax_mechanics(regime):
    from ai_economist import foundation

    config = configuration(regime)
    foundation.scenarios.get(config["scenario_name"]).seed(123)
    env = foundation.make_env_instance(**config)
    env.reset()
    tax = env.get_component("PeriodicBracketTax")
    if regime == "fixed-tax":
        assert tax.tax_model == "us-federal-single-filer-2018-scaled"
        assert tax.tax_annealing_schedule is None
        np.testing.assert_allclose(
            tax.curr_marginal_rates, [0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
        )
    # Known income, idle workers: verify collection and equal redistribution.
    env.world.agents[0].state["inventory"]["Coin"] = 100
    expected = tax.taxes_due(100) if regime == "fixed-tax" else 0
    for _ in range(tax.period):
        env.step({str(i): 0 for i in range(4)})
    assert tax.total_collected_taxes == pytest.approx(expected)
    coins = [a.total_endowment("Coin") for a in env.world.agents]
    np.testing.assert_allclose(coins, [100 - expected * 0.75] + [expected / 4] * 3)
    if regime == "fixed-tax":
        assert expected > 0


def test_held_out_seed_validation():
    saved = {"config": {"seed": 42, "workers": 4}}
    for seeds in ([42], [45], [100, 100], [], [0], [-1], [2**32], list(range(100, 121))):
        with pytest.raises(ValueError):
            validate_seeds(saved, seeds)
    validate_seeds(saved, [10000, 10001])


@pytest.mark.parametrize("regime", ["free-market", "fixed-tax", "learned-tax"])
def test_evaluation_and_fixed_resume(tmp_path, monkeypatch, regime):
    base = Path(os.environ.get("EVALUATION_TEST_RESULTS", str(tmp_path))) / regime
    base.mkdir(parents=True, exist_ok=True)
    whole, split, resumed = (base / name for name in ("whole", "split", "resumed"))
    options = ("--workers", "1", "--horizon", "16", "--epochs", "1", "--tax-regime", regime)
    if regime == "learned-tax":
        phase1 = base / "phase1"
        run_train(phase1, 1, "--workers", "1", "--horizon", "16", "--epochs", "1", "--seed", "99")
        options += ("--phase2-from", str(phase1 / "checkpoint-000001.pt"), "--seed", "42")
    run_train(whole, 2, *options)
    source = whole / "checkpoint-000002.pt"
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    saved = checkpoint.load(source)
    if regime == "fixed-tax":
        run_train(split, 1, *options)
        run_train(resumed, 2, "--resume", str(split / "checkpoint-000001.pt"))
        continued = checkpoint.load(resumed / "checkpoint-000002.pt")
        for key in ("model", "optimizer", "recurrent", "rng", "progress"):
            equal_state(saved[key], continued[key])
        equal_state(pickle.loads(saved["workers"][0]), pickle.loads(continued["workers"][0]))
        rejected = subprocess.run(
            [
                sys.executable,
                str(checkpoint.ROOT / "pilot/train.py"),
                "--resume",
                str(source),
                "--tax-regime",
                "free-market",
                "--output",
                str(base / "rejected"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert rejected.returncode != 0
        assert "Incompatible resume configuration: tax_regime" in rejected.stderr
        assert not (base / "rejected").exists()
    planners = []
    planner_factory = evaluation.PlannerActorCritic

    def capture_planner(shapes, sizes):
        planner = planner_factory(shapes, sizes)
        planners.append(planner)
        return planner

    monkeypatch.setattr(evaluation, "PlannerActorCritic", capture_planner)
    models = []
    factory = evaluation.ActorCritic

    def capture_model(shapes):
        model = factory(shapes)
        models.append(model)
        return model

    monkeypatch.setattr(evaluation, "ActorCritic", capture_model)
    first = evaluate(source, [10000, 10001], base / "eval")
    if regime == "learned-tax":
        equal_state(saved["planner"]["model"], planners[0].state_dict())
        assert not planners[0].training
        assert all(p.grad is None and not p.requires_grad for p in planners[0].parameters())
        with pytest.raises(ValueError, match="training worker"):
            evaluate(source, [99], base / "pretraining-seed")
    equal_state(saved["model"], models[0].state_dict())
    assert not models[0].training
    assert all(parameter.grad is None for parameter in models[0].parameters())
    # Fresh-process repeat tests RNG/recurrent resets and checkpoint-only inference.
    result = subprocess.run(
        [
            sys.executable,
            str(checkpoint.ROOT / "pilot/evaluate.py"),
            "--checkpoint",
            str(source),
            "--seeds",
            "10000",
            "10001",
            "--output",
            str(base / "repeat"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (first / "episodes.jsonl").read_bytes() == (base / "repeat/episodes.jsonl").read_bytes()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    records = [json.loads(line) for line in (first / "episodes.jsonl").read_text().splitlines()]
    assert [r["seed"] for r in records] == [10000, 10001]
    assert all(r["steps"] == 1000 and r["labor_weight"] == 1 for r in records)
    assert all(0 <= r["equality"] <= 1 for r in records)
    assert all(r["disable_taxes"] == (regime == "free-market") for r in records)
    for record in records:
        assert [s["start_step"] for s in record["tax_schedules"]] == list(range(1, 1001, 100))
    if regime == "learned-tax":
        assert all(
            r["marginal_rates"] is None and np.isfinite(r["planner_return"]) for r in records
        )
        assert any(max(s["marginal_rates"]) > 0.1 for r in records for s in r["tax_schedules"])
        metadata = json.loads((first / "config.json").read_text())
        assert metadata["training_seeds"] == [42, 99]
        assert metadata["planner_evaluated"]
    summary = json.loads((first / "summary.json").read_text())
    assert summary["environment_steps"] == 2000
    assert summary["aggregates"]["productivity"]["mean"] == np.mean(
        [r["productivity"] for r in records]
    )
    with pytest.raises(ValueError, match="training worker"):
        evaluate(source, [42], base / "bad-seed")
    assert not (base / "bad-seed").exists()
    with pytest.raises(FileExistsError):
        evaluate(source, [10000], first)


def test_joint_greedy_tax_schedule():
    """Each head selects its own rate; idle masks override logits for 99 steps."""
    import torch
    from ai_economist import foundation
    from environment import pack, pack_planner
    from model import ActorCritic, PlannerActorCritic

    config = evaluation.evaluation_config(
        {"config": {"tax_regime": "learned-tax", "environment": configuration("learned-tax")}}
    )
    foundation.scenarios.get(config["scenario_name"]).seed(123)
    env = foundation.make_env_instance(**config)
    raw = env.reset()
    shapes = {k: list(v.shape[1:]) for k, v in pack(raw).items()}
    planner_shapes = {k: list(v.shape[1:]) for k, v in pack_planner(raw).items()}
    worker = ActorCritic(shapes).eval()
    planner = PlannerActorCritic(planner_shapes, [22] * 7).eval()
    with torch.no_grad():
        planner.policy.output.weight.zero_()
        planner.policy.output.bias.zero_()
        for i in range(7):
            planner.policy.output.bias[i * 22 + i + 2] = 100
    before = copy.deepcopy(planner.state_dict())
    a = evaluation.episode(worker, config, 10000, "greedy", planner, (shapes, planner_shapes))
    b = evaluation.episode(worker, config, 10000, "greedy", planner, (shapes, planner_shapes))
    assert a == b
    for schedule in a["tax_schedules"]:
        np.testing.assert_allclose(schedule["marginal_rates"], np.arange(1, 8) * 0.05)
    equal_state(before, planner.state_dict())
    with pytest.raises(ValueError, match="observation layout"):
        evaluation.episode(worker, config, 10000, "greedy", planner, (shapes, {}))
