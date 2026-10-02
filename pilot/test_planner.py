"""Planner masks, Foundation taxes, phase transition, and exact joint continuation."""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import checkpoint
import numpy as np
import pytest
import torch
from environment import configuration, pack_planner
from evaluate import evaluation_config
from model import ActorCritic, MultiCategorical, PlannerActorCritic
from test_resume import equal_state, run_train
from train import planner_entropy, transfer_workers


def test_planner_distribution_and_recurrence():
    torch.set_num_threads(1)
    shapes = {"flat": [5], "action_mask": [7]}
    model = PlannerActorCritic(shapes, [3, 4])
    obs = {"flat": torch.randn(3, 2, 5), "action_mask": torch.zeros(3, 2, 7)}
    obs["action_mask"][..., [1, 5]] = 1
    with torch.no_grad():
        distribution, values, state = model(
            obs,
            tuple(torch.ones(1, 2, 256) for _ in range(4)),
            torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
        )
        assert (distribution.sample() == torch.tensor([1, 2])).all()
        torch.testing.assert_close(distribution.entropy(), torch.zeros(3, 2))
        _, fresh_values, fresh_state = model(
            {k: v[2:] for k, v in obs.items()}, model.initial_state(2, "cpu"), torch.ones(1, 2)
        )
        torch.testing.assert_close(values[2:], fresh_values)
        equal_state(state, fresh_state)
    dist = MultiCategorical(torch.zeros(2, 7), [3, 4])
    torch.testing.assert_close(
        dist.log_prob(torch.zeros(2, 2, dtype=torch.long)),
        torch.full((2,), -np.log(12), dtype=torch.float32),
    )
    torch.testing.assert_close(dist.entropy(), torch.full((2,), np.log(12), dtype=torch.float32))
    obs["action_mask"][..., :3] = 0
    with pytest.raises(ValueError, match="no valid action"):
        model(obs, model.initial_state(2, "cpu"), torch.ones(3, 2))
    assert planner_entropy(0) == 2.0
    assert planner_entropy(25_000_000) == 1.0625
    assert planner_entropy(50_000_000) == planner_entropy(60_000_000) == 0.125


def test_foundation_planner_taxes_and_annealing():
    from ai_economist import foundation

    config = configuration("learned-tax")
    foundation.scenarios.get(config["scenario_name"]).seed(123)
    env = foundation.make_env_instance(**config)
    raw = env.reset()
    packed = pack_planner(raw)
    assert packed["flat"].size == sum(
        np.asarray(v).size for k, v in raw["p"].items() if k != "action_mask"
    )
    tax = env.get_component("PeriodicBracketTax")
    sizes = env.world.planner.action_spaces
    assert sizes.tolist() == [22] * 7
    masks = np.split(packed["action_mask"][0], np.cumsum(sizes)[:-1])
    assert all(mask[:4].all() and not mask[4:].any() for mask in masks)
    # Index 3 means 10%, index 1 means 0%; each includes its own no-op at 0.
    raw, _, _, _ = env.step({"p": [3] * 7})
    np.testing.assert_allclose(tax.curr_marginal_rates, [0.1] * 7)
    idle_masks = np.split(raw["p"]["action_mask"], np.cumsum(sizes)[:-1])
    assert all(mask[0] and not mask[1:].any() for mask in idle_masks)
    env.world.agents[0].state["inventory"]["Coin"] = 100
    for _ in range(tax.period - 1):
        raw, _, _, _ = env.step({"p": [0] * 7})
    assert tax.total_collected_taxes == pytest.approx(10)
    np.testing.assert_allclose(
        [a.total_endowment("Coin") for a in env.world.agents], [92.5, 2.5, 2.5, 2.5]
    )
    assert tax.tax_cycle_pos == 1
    # Exercise a later curriculum point without a costly thousand-episode run.
    old_max = tax._annealed_rate_max
    env._completions = 1000
    raw = env.reset()
    assert tax._annealed_rate_max > old_max
    assert raw["p"]["action_mask"].sum() > packed["action_mask"].sum()
    assert env.energy_weight == 1


def test_joint_transition_and_exact_restart(tmp_path):
    base = Path(os.environ.get("PLANNER_TEST_RESULTS", str(tmp_path)))
    base.mkdir(parents=True, exist_ok=True)
    phase1, whole, split, resumed = (
        base / n for n in ("phase1", "joint-whole", "joint-split", "joint-resumed")
    )
    run_train(phase1, 1, "--workers", "1", "--epochs", "1")
    source = phase1 / "checkpoint-000001.pt"
    before = checkpoint.load(source)
    transferred = ActorCritic(before["config"]["shapes"])
    transferred_optimizer = torch.optim.Adam(transferred.parameters(), lr=3e-4)
    transfer_workers(transferred, transferred_optimizer, before["config"]["shapes"], before)
    equal_state(before["model"], transferred.state_dict())
    equal_state(before["optimizer"], transferred_optimizer.state_dict())
    with pytest.raises(ValueError, match="observation layout"):
        transfer_workers(transferred, transferred_optimizer, {}, before)
    invalid = {**before, "config": {**before["config"], "tax_regime": "fixed-tax"}}
    invalid_source = base / "invalid-source.pt"
    torch.save(invalid, invalid_source)
    for index, (options, message) in enumerate(
        [
            (("--tax-regime", "learned-tax"), "require --phase2-from"),
            (("--phase2-from", str(invalid_source)), "free-market phase-one"),
            (("--phase2-from", str(source), "--tax-regime", "fixed-tax"), "requires learned-tax"),
            (("--phase2-from", str(source), "--resume", str(source)), "mutually exclusive"),
        ]
    ):
        rejected = base / f"rejected-{index}"
        result = subprocess.run(
            [
                sys.executable,
                str(checkpoint.ROOT / "pilot/train.py"),
                "--device",
                "cpu",
                "--output",
                str(rejected),
                *options,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0 and message in result.stderr
        assert not rejected.exists()
    options = ("--phase2-from", str(source))
    run_train(whole, 9, *options)
    run_train(split, 4, *options)
    middle = checkpoint.load(split / "checkpoint-000004.pt")
    assert middle["config"]["phase_one_progress"] == before["progress"]
    assert middle["config"]["phase2_from"] == str(source.resolve())
    env = pickle.loads(middle["workers"][0])[0]
    assert env.world.timestep == 512 and env._completions == 0
    assert env.energy_weight == 1
    # Worker optimizer's Adam step continues from pretraining; both networks learn.
    steps = [v["step"].item() for v in middle["optimizer"]["state"].values()]
    initial_steps = [v["step"].item() for v in before["optimizer"]["state"].values()]
    assert min(steps) > max(initial_steps)
    checkpoint.append_log(split / "metrics.jsonl", '{"uncommitted":true}')
    run_train(resumed, 9, "--resume", str(split / "checkpoint-000004.pt"))
    a = checkpoint.load(whole / "checkpoint-000009.pt")
    b = checkpoint.load(resumed / "checkpoint-000009.pt")
    for key in (
        "model",
        "optimizer",
        "planner",
        "progress",
        "recurrent",
        "starts",
        "observation",
        "rng",
        "schedules",
    ):
        equal_state(a[key], b[key])
    equal_state(pickle.loads(a["workers"][0]), pickle.loads(b["workers"][0]))
    env = pickle.loads(b["workers"][0])[0]
    assert env._completions == 1 and env.world.timestep == 152
    assert env.get_component("PeriodicBracketTax")._last_completions == 1
    assert b["config"]["phase2_from"] == str(source.resolve())
    assert b["schedules"]["phase_steps"] == 1152
    assert b["schedules"]["planner_entropy"] == planner_entropy(1152)
    records = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    originals = [json.loads(line) for line in (whole / "metrics.jsonl").read_text().splitlines()]
    for x, y in zip(originals, records, strict=True):
        equal_state(
            {k: v for k, v in x.items() if not k.endswith("seconds") and k != "rss_bytes"},
            {k: v for k, v in y.items() if not k.endswith("seconds") and k != "rss_bytes"},
        )
    assert [r["update"] for r in records] == list(range(1, 10))
    for directory in (whole, split, resumed):
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["parameter_change_l2"] > 0
        assert summary["planner_parameter_change_l2"] > 0
    assert evaluation_config(b)["energy_warmup_constant"] == 0


@pytest.mark.skipif(os.environ.get("RUN_CUDA_RESUME_TEST") != "1", reason="Opt-in CUDA smoke test")
def test_joint_cuda_resume(tmp_path):
    base = Path(os.environ.get("PLANNER_CUDA_TEST_RESULTS", str(tmp_path)))
    base.mkdir(parents=True, exist_ok=True)
    phase1, first, resumed = (base / n for n in ("phase1", "joint-start", "joint-resumed"))
    run_train(phase1, 1, "--workers", "4", "--horizon", "32", "--epochs", "1")
    run_train(first, 2, "--device", "cuda", "--phase2-from", str(phase1 / "checkpoint-000001.pt"))
    run_train(resumed, 4, "--device", "cuda", "--resume", str(first / "checkpoint-000002.pt"))
    saved = checkpoint.load(resumed / "checkpoint-000004.pt")
    assert len(saved["workers"]) == 4
    assert saved["progress"]["environment_steps"] == 512
    assert saved["planner"]["recurrent"][0].shape == (1, 4, 256)
    assert len(saved["rng"]["cuda"]) > 0
    assert all(t.device.type == "cpu" for t in saved["planner"]["model"].values())
    summary = json.loads((resumed / "summary.json").read_text())
    assert summary["checkpoint_cpu_portability"] and summary["cpu_device_forward_agreement"]
    assert summary["planner_parameter_change_l2"] > 0
