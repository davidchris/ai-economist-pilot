"""Bounded subprocess workers for the pinned Salesforce Foundation environment."""

import multiprocessing as mp
import pickle
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream"))
KEYS = ("flat", "world-map", "world-idx_map", "action_mask")


TAX_REGIMES = ("free-market", "fixed-tax", "learned-tax")


def configuration(tax_regime="free-market"):
    with (ROOT / "upstream/tutorials/rllib/phase1/config.yaml").open() as source:
        config = yaml.safe_load(source)["env"]
    if tax_regime not in TAX_REGIMES:
        raise ValueError(f"Unsupported tax regime: {tax_regime}")
    tax = next(c["PeriodicBracketTax"] for c in config["components"] if "PeriodicBracketTax" in c)
    # Spacing selects cutoffs, not rates. Fixed taxes must also bypass annealing.
    tax["tax_model"] = (
        "us-federal-single-filer-2018-scaled" if tax_regime == "fixed-tax" else "model_wrapper"
    )
    tax["disable_taxes"] = tax_regime == "free-market"
    if tax_regime == "fixed-tax":
        tax["tax_annealing_schedule"] = None
    if tax_regime == "learned-tax":
        tax["rate_disc"] = 0.05
        config["energy_warmup_constant"] = 0
    config["dense_log_frequency"] = None
    return config


def pack(observation):
    result = {key: np.stack([observation[str(i)][key] for i in range(4)]) for key in KEYS}
    assert all(np.isfinite(value).all() for value in result.values())
    assert (result["action_mask"].sum(axis=1) > 0).all()
    return result


def pack_planner(observation):
    """Include public per-worker observations appended by Foundation after flattening."""
    raw = observation["p"]
    result = {
        "flat": np.concatenate(
            [np.asarray(raw[k]).reshape(-1) for k in sorted(raw) if k != "action_mask"]
        )[None].astype(np.float32),
        "action_mask": np.asarray(raw["action_mask"], dtype=np.float32)[None],
    }
    assert all(np.isfinite(value).all() for value in result.values())
    return result


def worker(connection, seed, profile=False, snapshot=None, tax_regime="free-market"):
    try:
        from ai_economist import foundation

        if snapshot is None:
            foundation.scenarios.get(configuration(tax_regime)["scenario_name"]).seed(seed)
            env = foundation.make_env_instance(**configuration(tax_regime))
            observation = env.reset()
            episode_return = np.zeros(5 if tax_regime == "learned-tax" else 4)
        else:
            env, observation, episode_return, python_rng, numpy_rng = pickle.loads(snapshot)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        joint = tax_regime == "learned-tax"
        initial = pack(observation)
        connection.send(
            (initial, pack_planner(observation), env.world.planner.action_spaces.tolist())
            if joint
            else initial
        )
        echo_payload = (initial, np.zeros(4, dtype=np.float32), False, None, None)
        while True:
            action = connection.recv()
            if action is None:
                break
            if isinstance(action, str) and action == "snapshot":
                connection.send(
                    pickle.dumps(
                        (
                            env,
                            observation,
                            episode_return,
                            random.getstate(),
                            np.random.get_state(),
                        ),
                        protocol=5,
                    )
                )
                continue
            if isinstance(action, str) and action == "profile_echo":
                connection.send(echo_payload)
                continue
            work_started = time.perf_counter() if profile else 0
            planner_action = None
            if joint:
                action, planner_action = action
                offset = 0
                for value, size in zip(
                    planner_action, env.world.planner.action_spaces, strict=True
                ):
                    assert 0 <= value < size
                    assert observation["p"]["action_mask"][offset + value] > 0
                    offset += size
            for i, value in enumerate(action):
                assert observation[str(i)]["action_mask"][value] > 0
            simulator_started = time.perf_counter() if profile else 0
            actions: dict[str, Any] = {str(i): int(value) for i, value in enumerate(action)}
            if joint:
                actions["p"] = planner_action
            observation, rewards, done, _ = env.step(actions)
            simulator_seconds = time.perf_counter() - simulator_started if profile else 0
            reward = np.array([rewards[str(i)] for i in range(4)], dtype=np.float32)
            assert np.isfinite(reward).all()
            episode_return[:4] += reward
            if joint:
                assert np.isfinite(rewards["p"])
                episode_return[4] += rewards["p"]
            terminal = done["__all__"]
            metrics = None
            if terminal:
                metrics = {
                    "return_mean": float(episode_return[:4].mean()),
                    "metrics": env.metrics,
                    "worker_rss_bytes": psutil.Process().memory_info().rss,
                }
                if joint:
                    metrics["planner_return"] = float(episode_return[4])
                reset_started = time.perf_counter() if profile else 0
                observation = env.reset()
                simulator_seconds += time.perf_counter() - reset_started if profile else 0
                episode_return[:] = 0
            packed = pack(observation)
            timing = None
            if profile:
                timing = {
                    "simulator_seconds": simulator_seconds,
                    "worker_processing_seconds": time.perf_counter() - work_started,
                }
            payload = (packed, reward, terminal, metrics, timing)
            if joint:
                assert np.isfinite(rewards["p"])
                payload += (pack_planner(observation), float(rewards["p"]))
            connection.send(payload)
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


class Environments:
    def __init__(self, count, seed, profile=False, snapshots=None, tax_regime="free-market"):
        self.joint = tax_regime == "learned-tax"
        self.planner_observation = {}
        self.planner_reward = None
        self.planner_action_sizes = []
        self.profile = profile
        self.timings = {}
        context = mp.get_context("spawn")
        self.connections = []
        self.processes = []
        if snapshots is not None and len(snapshots) != count:
            raise ValueError("Checkpoint worker count is incompatible")
        try:
            for index in range(count):
                parent, child = context.Pipe()
                process = context.Process(
                    target=worker,
                    args=(
                        child,
                        seed + index,
                        profile,
                        snapshots[index] if snapshots else None,
                        tax_regime,
                    ),
                    daemon=True,
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            initial = [self.receive(c) for c in self.connections]
            if self.joint:
                self.planner_action_sizes = initial[0][2]
                assert all(item[2] == self.planner_action_sizes for item in initial)
                self.planner_observation = self.stack([item[1] for item in initial])
                initial = [item[0] for item in initial]
            self.observation = self.stack(initial)
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        for connection in self.connections:
            connection.send("snapshot")
        return [self.receive(c) for c in self.connections]

    @staticmethod
    def receive(connection):
        if not connection.poll(60):
            raise TimeoutError("Foundation worker did not respond within 60 seconds")
        result = connection.recv()
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result

    @staticmethod
    def stack(observations):
        return {key: np.concatenate([o[key] for o in observations]) for key in observations[0]}

    def step(self, actions, planner_actions=None):
        started = time.perf_counter() if self.profile else 0
        if self.joint and (
            planner_actions is None
            or planner_actions.shape != (len(self.connections), len(self.planner_action_sizes))
        ):
            raise ValueError("Planner actions must match economies and action subspaces")
        for index, (connection, action) in enumerate(zip(self.connections, actions.reshape(-1, 4))):
            if self.joint:
                assert planner_actions is not None
                connection.send((action, planner_actions[index]))
            else:
                connection.send(action)
        results = [self.receive(c) for c in self.connections]
        if self.joint:
            self.planner_observation = self.stack([r[5] for r in results])
            self.planner_reward = np.array([r[6] for r in results], dtype=np.float32)
        received = time.perf_counter() if self.profile else 0
        self.observation = self.stack([r[0] for r in results])
        if self.profile:
            self.timings = {
                "worker_roundtrip_seconds": received - started,
                "observation_stack_seconds": time.perf_counter() - received,
                "simulator_sum_seconds": sum(r[4]["simulator_seconds"] for r in results),
                "worker_processing_sum_seconds": sum(
                    r[4]["worker_processing_seconds"] for r in results
                ),
                "worker_processing_max_seconds": max(
                    r[4]["worker_processing_seconds"] for r in results
                ),
            }
        return (
            self.observation,
            np.concatenate([r[1] for r in results]),
            np.repeat([r[2] for r in results], 4).astype(np.float32),
            [r[3] for r in results if r[3] is not None],
        )

    def benchmark_transport(self, iterations=128):
        """Round-trip a representative initial payload without advancing the economy."""
        started = time.perf_counter()
        for _ in range(iterations):
            for connection in self.connections:
                connection.send("profile_echo")
            for connection in self.connections:
                self.receive(connection)
        return {
            "iterations": iterations,
            "seconds": time.perf_counter() - started,
            "scope": "Initial observation payload echo; includes serialization and scheduling, excludes simulation and stacking.",
        }

    def close(self):
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError, OSError):
                    pass
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            connection.close()
