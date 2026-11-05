"""Pretraining gate, reference policies and production diagnostics."""

import json

import numpy as np
import pytest
import torch
from environment import (
    WORKER_ACTION_GROUPS,
    action_shares,
    configuration,
    pack,
    resources_gathered,
)
from pretrain import (
    ReferencePolicy,
    extend_protocol,
    gate,
    parse_arm,
    reference_config,
    run_reference,
)


def evaluation(steps=200_000, weight=0.99, returned=5.0, sem=1.0, productivity=900.0, builds=20.0):
    return {
        "environment_steps": steps,
        "trained_labor_weight": weight,
        "return_mean": {"mean": returned, "sem": sem},
        "productivity": {"mean": productivity},
        "builds": builds,
    }


def test_gate_requires_two_sustained_full_cost_productive_checkpoints():
    good = evaluation()
    assert gate([good, good], 700.0, -1.0)["passed"]
    assert not gate([good], 700.0, -1.0)["passed"]
    # Only the last two checkpoints count; an early failure can be outgrown.
    assert gate([evaluation(returned=-9.0), good, good], 700.0, -1.0)["passed"]
    for bad in (
        evaluation(returned=1.0, sem=1.0),  # interval includes the no-op return
        evaluation(productivity=600.0),
        evaluation(builds=0.0),
        evaluation(weight=0.5),  # labor-free productivity is not enough
    ):
        assert not gate([good, bad], 700.0, -1.0)["passed"]
        assert not gate([bad, good], 700.0, -1.0)["passed"]


def test_action_groups_match_foundation_layout():
    from ai_economist import foundation

    config = configuration("free-market", "paper2020", 4, False, None, 100)
    foundation.scenarios.get(config["scenario_name"]).seed(42)
    env = foundation.make_env_instance(**config)
    env.reset()
    agent = env.world.agents[0]
    assert agent.action_spaces == 50
    groups = {"Build": "build", "ContinuousDoubleAuction": "trade", "Gather": "move"}
    for action, (component, _) in agent.single_action_map.items():
        start, stop = WORKER_ACTION_GROUPS[groups[component.split(".")[0]]]
        assert start <= action < stop
    counts = np.zeros(50)
    counts[[0, 1, 2, 46]] = [1, 1, 1, 1]
    assert action_shares(counts) == {"noop": 0.25, "build": 0.25, "trade": 0.25, "move": 0.25}


def test_reference_policies_and_gather_diagnostic(tmp_path):
    from ai_economist import foundation

    mask = torch.zeros(1, 4, 50)
    mask[..., [0, 3, 46]] = 1
    observations = {"action_mask": mask}
    distribution, values, state = ReferencePolicy("noop")(observations, (), torch.ones(1, 4))
    assert (distribution.sample() == 0).all() and values.shape == (1, 4) and state == ()
    distribution, _, _ = ReferencePolicy("uniform")(observations, (), torch.ones(1, 4))
    np.testing.assert_allclose(distribution.probs[0, 0, [0, 3, 46]].numpy(), 1 / 3)

    assert parse_arm("a-1:42:1e+12:16") == {
        "name": "a-1",
        "seed": 42,
        "labor_warmup_constant": 1e12,
        "minibatch_sequences": 16,
    }

    path = tmp_path / "noop.jsonl"
    run_reference("noop", [20000], path)
    record = json.loads(path.read_text())
    # Respawns under stationary agents still charge collection labor, so return <= 0.
    assert record["return_mean"] <= 0 and record["productivity"] == 0
    assert record["metrics"]["Build/total_builds"] == 0 and record["action_shares"]["noop"] == 1

    # The Gather log counts collected units, including skill bonuses.
    config = reference_config()
    foundation.scenarios.get(config["scenario_name"]).seed(20000)
    env = foundation.make_env_instance(**config)
    env.reset()
    env.get_component("Gather").gathers = [[], [{"agent": 0, "resource": "Wood", "n": 2}]]
    assert resources_gathered(env) == 2


def test_worker_flat_observation_includes_episode_time():
    from ai_economist import foundation

    config = reference_config()
    foundation.scenarios.get(config["scenario_name"]).seed(20000)
    env = foundation.make_env_instance(**config)
    raw = env.reset()
    for _ in range(3):
        raw, _, _, _ = env.step({str(i): 0 for i in range(4)})
    packed = pack(raw)["flat"]
    assert packed.shape == (4, raw["0"]["flat"].size + 1)
    np.testing.assert_allclose(packed[:, -1], 3 / config["episode_length"])
    np.testing.assert_array_equal(packed[0, :-1], raw["0"]["flat"])


def test_protocol_extension_cannot_redefine_existing_arms():
    arm = parse_arm("a:42:40000:4")
    existing = {"arms": [arm], "steps": 100, "device": "cuda"}
    other = parse_arm("b:142:40000:4")
    extended = extend_protocol(existing, {"arms": [other], "steps": 200, "device": "cuda"})
    assert extended["steps"] == 200 and extended["arms"] == [arm, other]
    with pytest.raises(ValueError):
        extend_protocol(existing, {"arms": [arm], "steps": 50, "device": "cuda"})
    with pytest.raises(ValueError):
        extend_protocol(existing, {"arms": [arm], "steps": 200, "device": "cpu"})
    with pytest.raises(ValueError):
        changed = parse_arm("a:42:10000:4")
        extend_protocol(existing, {"arms": [changed], "steps": 200, "device": "cuda"})
