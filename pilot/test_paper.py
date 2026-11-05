"""Paper-profile observations, curricula, matched phases and restart acceptance."""

import json
import os
import pickle
import subprocess
import sys

import checkpoint
import numpy as np
import pytest
import torch
from environment import configuration, pack, pack_planner
from evaluate import episode, evaluation_config
from model import ActorCritic, PlannerActorCritic
from test_resume import equal_state, run_train
from train import planner_entropy


def test_spatial_planner_and_replica_invariant_tax_curriculum():
    from ai_economist import foundation

    for workers in (1, 4):
        config = configuration("learned-tax", "paper2020", workers, True, 4000, 100)
        foundation.scenarios.get(config["scenario_name"]).seed(42)
        env = foundation.make_env_instance(**config)
        raw = env.reset()
        obs = pack_planner(raw)
        assert obs["world-map"].shape[-2:] == (25, 25)
        assert pack(raw)["world-map"].shape[-2:] == (11, 11)
        for key in ("world-map", "world-idx_map"):
            np.testing.assert_array_equal(obs[key][0], raw["p"][key])
        # Public per-worker fields must survive separately from the map tensors.
        np.testing.assert_array_equal(
            obs["flat"][0],
            np.concatenate(
                [
                    np.asarray(raw["p"][k]).reshape(-1)
                    for k in sorted(raw["p"])
                    if k not in ("world-map", "world-idx_map", "action_mask")
                ]
            ),
        )
        tax = env.get_component("PeriodicBracketTax")
        assert tax.curr_rate_max == pytest.approx(0.1)
        assert env.energy_weight == 1
        assert env.get_component("ContinuousDoubleAuction").order_labor * env.energy_cost == 0.05
        # Same aggregate step count (4,000) reaches the same endpoint at either count.
        env._completions = 4000 // (1000 * workers)
        env.reset()
        assert tax.curr_rate_max == 1
        assert planner_entropy(0, "paper2020") == planner_entropy(54_000_000, "paper2020") == 0.1
    shapes = {k: list(v.shape[1:]) for k, v in obs.items()}
    model = PlannerActorCritic(shapes, [22] * 7)
    tensors = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
    _, values, _ = model(tensors, model.initial_state(1, "cpu"), torch.ones(1, 1))
    values.sum().backward()
    gradient = dict(model.named_parameters())["value.convolution.0.weight"].grad
    assert gradient is not None and gradient.abs().sum() > 0
    with torch.no_grad():
        state = tuple(torch.ones(1, 1, 256) for _ in range(4))
        a = model(tensors, state, torch.ones(1, 1))
        b = model(tensors, model.initial_state(1, "cpu"), torch.ones(1, 1))
        equal_state(a[1:], b[1:])


@pytest.mark.parametrize("regime", ["free-market", "fixed-tax", "learned-tax"])
def test_paper_matched_transition_restart_and_evaluation(tmp_path, regime):
    phase1, whole, split, resumed = (tmp_path / n for n in ("phase1", "whole", "split", "resumed"))
    run_train(
        phase1,
        1,
        "--profile",
        "paper2020",
        "--workers",
        "1",
        "--minibatch-sequences",
        "4",
        "--labor-warmup-constant",
        "100",
        "--tax-anneal-steps",
        "1000",
    )
    source = phase1 / "checkpoint-000001.pt"
    options = ("--phase2-from", str(source), "--tax-regime", regime)
    run_train(whole, 6, *options)
    run_train(split, 2, *options)
    middle = checkpoint.load(split / "checkpoint-000002.pt")
    assert middle["config"]["phase_one_progress"]["environment_steps"] == 200
    assert middle["config"]["sequence_length"] == 50
    assert middle["config"]["epochs"] == 1
    env = pickle.loads(middle["workers"][0])[0]
    assert env.energy_weight == 1 and env.world.timestep == 400
    run_train(resumed, 6, "--resume", str(split / "checkpoint-000002.pt"))
    a = checkpoint.load(whole / "checkpoint-000006.pt")
    b = checkpoint.load(resumed / "checkpoint-000006.pt")
    for key in (
        "model",
        "optimizer",
        "planner",
        "recurrent",
        "starts",
        "observation",
        "rng",
        "schedules",
        "progress",
    ):
        equal_state(a[key], b[key])
    equal_state(pickle.loads(a["workers"][0]), pickle.loads(b["workers"][0]))
    env = pickle.loads(b["workers"][0])[0]
    assert env._completions == 1 and env.world.timestep == 200
    if regime != "free-market":
        assert env.get_component("PeriodicBracketTax").curr_rate_max == 1
    planner = None
    if regime == "learned-tax":
        planner = PlannerActorCritic(
            b["config"]["planner_shapes"], b["config"]["planner_action_sizes"]
        )
        planner.load_state_dict(b["planner"]["model"])
        assert b["config"]["planner_shapes"]["world-map"][-2:] == [25, 25]
    model = ActorCritic(b["config"]["shapes"])
    model.load_state_dict(b["model"])
    model.eval()
    if planner is not None:
        planner.eval()
    config = evaluation_config(b)
    expected = (b["config"]["shapes"], b["config"]["planner_shapes"])
    first = episode(model, config, 10000, "sample", planner, expected)
    second = episode(model, config, 10000, "sample", planner, expected)
    assert first == second
    assert first["steps"] == 1000 and first["labor_weight"] == 1
    for flag, value in (
        ("--tax-anneal-steps", "2000"),
        ("--profile", "pilot"),
        ("--labor-warmup-constant", "0"),
    ):
        rejected = subprocess.run(
            [
                sys.executable,
                str(checkpoint.ROOT / "pilot/train.py"),
                "--resume",
                str(resumed / "checkpoint-000006.pt"),
                flag,
                value,
                "--output",
                str(tmp_path / "rejected"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert rejected.returncode != 0 and "Incompatible resume configuration" in rejected.stderr
        assert not (tmp_path / "rejected").exists()


def test_paper_continues_beyond_old_total_cap(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run_train(
        first,
        100,
        "--profile",
        "paper2020",
        "--workers",
        "1",
        "--horizon",
        "1",
        "--sequence-length",
        "1",
        "--minibatch-sequences",
        "4",
        "--checkpoint-every",
        "100",
    )
    run_train(second, 102, "--resume", str(first / "checkpoint-000100.pt"))
    saved = checkpoint.load(second / "checkpoint-000102.pt")
    assert saved["progress"]["update"] == 102
    records = [json.loads(line) for line in (second / "metrics.jsonl").read_text().splitlines()]
    assert [r["update"] for r in records] == list(range(1, 103))
    rejected = subprocess.run(
        [
            sys.executable,
            str(checkpoint.ROOT / "pilot/train.py"),
            "--resume",
            str(second / "checkpoint-000102.pt"),
            "--updates",
            "203",
            "--output",
            str(tmp_path / "rejected"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rejected.returncode != 0 and "100 additional" in rejected.stderr
    assert not (tmp_path / "rejected").exists()


@pytest.mark.skipif(os.environ.get("RUN_CUDA_RESUME_TEST") != "1", reason="Opt-in CUDA smoke test")
def test_spatial_cuda_resume(tmp_path):
    phase1, first, resumed = (tmp_path / name for name in ("phase1", "first", "resumed"))
    run_train(
        phase1,
        1,
        "--profile",
        "paper2020",
        "--workers",
        "4",
        "--horizon",
        "50",
        "--minibatch-sequences",
        "4",
    )
    run_train(first, 2, "--device", "cuda", "--phase2-from", str(phase1 / "checkpoint-000001.pt"))
    run_train(resumed, 4, "--device", "cuda", "--resume", str(first / "checkpoint-000002.pt"))
    saved = checkpoint.load(resumed / "checkpoint-000004.pt")
    assert saved["config"]["planner_shapes"]["world-map"][-2:] == [25, 25]
    assert saved["progress"]["environment_steps"] == 800
    summary = json.loads((resumed / "summary.json").read_text())
    assert summary["checkpoint_cpu_portability"] and summary["cpu_device_forward_agreement"]
