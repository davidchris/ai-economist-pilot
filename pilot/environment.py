"""Bounded subprocess workers for the pinned Salesforce Foundation environment."""

import multiprocessing as mp
import pickle
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream"))
KEYS = ("flat", "world-map", "world-idx_map", "action_mask")


def configuration():
    with (ROOT / "upstream/tutorials/rllib/phase1/config.yaml").open() as source:
        config = yaml.safe_load(source)["env"]
    config["dense_log_frequency"] = None
    return config


def pack(observation):
    result = {key: np.stack([observation[str(i)][key] for i in range(4)]) for key in KEYS}
    assert all(np.isfinite(value).all() for value in result.values())
    assert (result["action_mask"].sum(axis=1) > 0).all()
    return result


def worker(connection, seed, profile=False, snapshot=None):
    try:
        from ai_economist import foundation

        if snapshot is None:
            foundation.scenarios.get(configuration()["scenario_name"]).seed(seed)
            env = foundation.make_env_instance(**configuration())
            observation = env.reset()
            episode_return = np.zeros(4)
        else:
            env, observation, episode_return, python_rng, numpy_rng = pickle.loads(snapshot)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        initial = pack(observation)
        connection.send(initial)
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
            for i, value in enumerate(action):
                assert observation[str(i)]["action_mask"][value] > 0
            simulator_started = time.perf_counter() if profile else 0
            observation, rewards, done, _ = env.step(
                {str(i): int(value) for i, value in enumerate(action)}
            )
            simulator_seconds = time.perf_counter() - simulator_started if profile else 0
            reward = np.array([rewards[str(i)] for i in range(4)], dtype=np.float32)
            assert np.isfinite(reward).all()
            episode_return += reward
            terminal = done["__all__"]
            metrics = None
            if terminal:
                metrics = {
                    "return_mean": float(episode_return.mean()),
                    "metrics": env.metrics,
                    "worker_rss_bytes": psutil.Process().memory_info().rss,
                }
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
            connection.send((packed, reward, terminal, metrics, timing))
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


class Environments:
    def __init__(self, count, seed, profile=False, snapshots=None):
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
                    args=(child, seed + index, profile, snapshots[index] if snapshots else None),
                    daemon=True,
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.observation = self.stack([self.receive(c) for c in self.connections])
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
        return {key: np.concatenate([o[key] for o in observations]) for key in KEYS}

    def step(self, actions):
        started = time.perf_counter() if self.profile else 0
        for connection, action in zip(self.connections, actions.reshape(-1, 4)):
            connection.send(action)
        results = [self.receive(c) for c in self.connections]
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
