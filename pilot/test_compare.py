"""Tax comparison driver: arm matching, paired contrasts and failure approval."""

import argparse
import json

import compare
import pytest
from compare import (
    PRETRAINING_ARMS,
    accepted_failure,
    build_arms,
    contrasts,
    train_segment,
)
from environment import configuration
from experiment import REGIMES


def test_rerun_accepts_its_own_saved_protocol(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    final = {"hold_gate": {"passed": True}, "source_checkpoint": str(tmp_path / "source.pt")}
    (tmp_path / "source.pt").write_bytes(b"checkpoint")
    for name in PRETRAINING_ARMS.values():
        (tmp_path / f"final-{name}.json").write_text(json.dumps(final))
    args = argparse.Namespace(
        output=str(root),
        pretraining=str(tmp_path),
        training_seeds=[42],
        steps=80_000,
        evaluate_at=[40_000, 80_000],
        tax_anneal_steps=3_600_000,
        first_eval_seed=40000,
        device="cuda",
        concurrency=1,
        max_seconds=-1,
        accept_critic_failure=[],
    )
    monkeypatch.setattr(compare, "references", lambda *_: {})
    monkeypatch.setattr(compare, "archive_source", lambda _: None)
    root.rmdir()
    compare.main(args)  # Creates the protocol, then pauses at the expired deadline.
    compare.main(args)  # A different saved protocol would raise here.
    args.steps = 120_000
    args.evaluate_at = [40_000, 120_000]
    with pytest.raises(ValueError, match="different protocol"):
        compare.main(args)


def row(steps, offsets, base):
    return {
        "environment_steps": steps,
        "per_seed": {
            str(seed): {
                "welfare": base + offset,
                "productivity": base,
                "equality": base,
                "taxes_collected": base,
                "return_mean": base,
            }
            for seed, offset in zip((40000, 40001), offsets, strict=True)
        },
    }


def test_arms_differ_only_in_regime_within_a_training_seed():
    args = argparse.Namespace(
        training_seeds=[42, 142],
        steps=4_400_000,
        evaluate_at=[3_600_000, 4_400_000],
        tax_anneal_steps=3_600_000,
        device="cuda",
    )
    arms = build_arms(args, {42: {"path": "a"}, 142: {"path": "b"}})
    assert len(arms) == 6
    for seed, path in ((42, "a"), (142, "b")):
        group = [arm for arm in arms if arm["seed"] == seed]
        assert [arm["regime"] for arm in group] == list(REGIMES)
        shared = [{k: v for k, v in arm.items() if k not in ("name", "regime")} for arm in group]
        assert shared[0] == shared[1] == shared[2] and shared[0]["source"] == path


def test_tax_treatments_share_the_tutorial_ceiling_and_free_market_has_no_taxes():
    def schedule(regime):
        config = configuration(regime, "paper2020", 4, True, 3_600_000)
        tax = next(
            c["PeriodicBracketTax"] for c in config["components"] if "PeriodicBracketTax" in c
        )
        assert config["energy_warmup_constant"] == 0
        return tax, tax["tax_annealing_schedule"]

    assert schedule("learned-tax")[1] == pytest.approx([-100, 0.001])
    assert schedule("fixed-tax")[1] == schedule("learned-tax")[1]
    assert schedule("free-market")[0]["disable_taxes"]


def test_contrasts_pair_episode_seeds_and_keep_training_seeds_separate():
    reports = {
        seed: {
            "free-market": {"evaluations": [row(800, (0, 0), 10)]},
            "fixed-tax": {"evaluations": [row(800, (0, 0), 10)]},
            "learned-tax": {"evaluations": [row(800, (delta, delta + 2), 10)]},
        }
        for seed, delta in ((42, 1), (142, 2), (242, 3))
    }
    results = contrasts(reports)
    found = next(
        r
        for r in results
        if r["metric"] == "welfare" and r["contrast"] == "learned-tax minus free-market"
    )
    assert [found["by_training_seed"][k]["mean"] for k in ("42", "142", "242")] == [2, 3, 4]
    assert found["mean"] == 3 and found["same_sign_all_seeds"]
    assert found["interval_95"][0] < 3 < found["interval_95"][1]

    single = contrasts({42: reports[42]})
    assert all(r["interval_95"] is None for r in single)
    json.dumps(single, allow_nan=False)

    reports[42]["learned-tax"]["evaluations"][0]["per_seed"].pop("40001")
    with pytest.raises(ValueError, match="common evaluation seeds"):
        contrasts(reports)


FAILURE = """Traceback (most recent call last):
    torch.testing.assert_close({call}, cpu_values, rtol=1e-4, atol=1e-5)
AssertionError: Tensor-likes are not close!
"""


def segment(tmp_path, log, checkpoint=True):
    directory = tmp_path / "learned-tax-s42-000050"
    directory.mkdir()
    (tmp_path / "learned-tax-s42-000050.log").write_text(log)
    if checkpoint:
        (directory / "checkpoint-000050.pt").write_bytes(b"")
    return directory


@pytest.mark.parametrize("call", ["gpu_values.cpu()", "values.cpu()"])
def test_approved_critic_failure_continues_from_published_checkpoint(tmp_path, call):
    directory = segment(tmp_path, FAILURE.format(call=call))
    arm = {"name": "learned-tax-s42"}
    accepted = {"learned-tax-s42-000050"}
    assert train_segment(tmp_path, arm, 50, None, accepted) == (
        directory,
        directory / "checkpoint-000050.pt",
    )
    assert directory.exists()


@pytest.mark.parametrize(
    "log, checkpoint",
    [
        (FAILURE.format(call="gpu_policy.probs.cpu()"), True),  # policy checks stay fatal
        ("RuntimeError: replay mismatch", True),
        (FAILURE.format(call="gpu_values.cpu()"), False),
    ],
)
def test_approval_cannot_cover_other_failures(tmp_path, log, checkpoint):
    directory = segment(tmp_path, log, checkpoint)
    with pytest.raises(RuntimeError, match="not a critic agreement failure"):
        accepted_failure(tmp_path, "learned-tax-s42-000050", directory, 50)


def preapproved(tmp_path, monkeypatch, log, accepted):
    def failing_invoke(root, label, arguments):
        directory = root / label
        directory.mkdir()
        (directory / "checkpoint-000050.pt").write_bytes(b"")
        (root / f"{label}.log").write_text(log)
        raise RuntimeError(f"{label} failed")

    monkeypatch.setattr(compare, "guarded_invoke", failing_invoke)
    arm = {"name": "learned-tax-s42"}
    return train_segment(tmp_path, arm, 50, tmp_path / "parent.pt", accepted)


def test_preapproved_segment_continues_only_past_a_critic_failure(tmp_path, monkeypatch):
    name = "learned-tax-s42-000050"
    critic = FAILURE.format(call="gpu_values.cpu()")
    _, path = preapproved(tmp_path, monkeypatch, critic, {name})
    assert path == tmp_path / name / "checkpoint-000050.pt"


def test_preapproval_stops_other_and_unapproved_failures(tmp_path, monkeypatch):
    name = "learned-tax-s42-000050"
    policy = FAILURE.format(call="gpu_policy.probs.cpu()")
    with pytest.raises(RuntimeError, match="not a critic agreement failure"):
        preapproved(tmp_path, monkeypatch, policy, {name})
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(RuntimeError, match=f"{name} failed"):
        preapproved(other, monkeypatch, FAILURE.format(call="gpu_values.cpu()"), set())


def test_interrupted_approved_segment_is_retired_and_rerun(tmp_path, monkeypatch):
    name = "learned-tax-s42-000050"
    (tmp_path / name).mkdir()
    (tmp_path / f"{name}.log").write_text("Run directory: interrupted\n")
    calls = []

    def passing_invoke(root, label, arguments):
        calls.append(arguments)
        (root / label).mkdir()
        (root / label / "summary.json").write_text(
            json.dumps({"status": "passed", "completed_updates": 50, "checkpoint": "c.pt"})
        )

    monkeypatch.setattr(compare, "guarded_invoke", passing_invoke)
    arm = {"name": "learned-tax-s42"}
    _, path = train_segment(tmp_path, arm, 50, tmp_path / "parent.pt", {name})
    assert path == tmp_path / name / "c.pt" and len(calls) == 1
    assert "--resume" in calls[0] and list(tmp_path.glob(f"{name}.incomplete-*"))


def test_pending_approval_does_not_bypass_the_time_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "guarded_invoke", pytest.fail)
    arm = {"name": "a", "steps": 40_000, "evaluation_steps": []}
    with pytest.raises(compare.Paused):
        compare.run_arm(tmp_path, arm, [], {}, {"a-000050"}, deadline=-1)
