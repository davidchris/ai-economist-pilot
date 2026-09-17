"""Finite, resource-measured worker-only recurrent PPO pilot.

Run from the project root: .venv/bin/python pilot/train.py --device cuda
The calling systemd scope supplies the CPU/RAM/time limits.
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import TypedDict

import checkpoint as checkpoints
import numpy as np
import psutil
import torch
from environment import Environments, configuration
from model import ActorCritic, advantages
from profiling import CollectionProfile


class Totals(TypedDict):
    rollout_seconds: float
    update_seconds: float
    sampled_peak_rss_bytes: int
    collection_profile: dict[str, float]


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def tensors(observation, device, sequence=False):
    return {
        key: torch.as_tensor(value, device=device, dtype=torch.float32).unsqueeze(0)
        if sequence
        else torch.as_tensor(value, device=device, dtype=torch.float32)
        for key, value in observation.items()
    }


def sequences(value, length):
    # [time, agents, ...] -> [sequence length, chunks * agents, ...]
    chunks = value.shape[0] // length
    shaped = value.reshape(chunks, length, *value.shape[1:]).transpose(0, 1)
    return shaped.reshape(length, -1, *value.shape[2:])


def rss_bytes():
    process = psutil.Process()
    return sum(
        p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running()
    )


def train(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    saved = checkpoints.load(args.resume) if args.resume else None
    args.device = args.device or (saved["config"]["device"] if saved else "cuda")
    if args.resume:
        args.resume = str(Path(args.resume).resolve())
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(0.25)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if saved:
        for key in (
            "workers",
            "seed",
            "horizon",
            "sequence_length",
            "epochs",
            "minibatch_sequences",
        ):
            if getattr(args, key) is None:
                setattr(args, key, saved["config"][key])
            elif getattr(args, key) != saved["config"][key]:
                raise ValueError(f"Incompatible resume configuration: {key}")
        if saved["config"]["environment"] != configuration():
            raise ValueError("Incompatible environment configuration")
        if args.updates <= saved["progress"]["update"]:
            raise ValueError("--updates is a total target and must exceed the saved update")
    else:
        for key, value in {
            "workers": 2,
            "seed": 42,
            "horizon": 128,
            "sequence_length": 16,
            "epochs": 2,
            "minibatch_sequences": 16,
        }.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    if not (1 <= args.workers <= 4 and 1 <= args.updates <= 100):
        raise ValueError("workers must be 1..4 and total updates 1..100")
    if (
        min(
            args.horizon,
            args.sequence_length,
            args.epochs,
            args.minibatch_sequences,
            args.checkpoint_every,
            args.keep_checkpoints,
            args.seed,
        )
        < 1
    ):
        raise ValueError("Training sizes, seed, and checkpoint settings must be positive")
    if args.horizon % args.sequence_length:
        raise ValueError("horizon must be divisible by sequence length")
    # Seed only after inheriting the effective configuration.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = checkpoints.output_directory(args.output)
    print(f"Run directory: {output}", flush=True)
    if saved:
        checkpoints.recover_logs(Path(args.resume).resolve().parent, output, saved["log_offsets"])
    disk_before = shutil.disk_usage(output).free
    if disk_before < 15 * 1024**3:
        raise RuntimeError("Pilot requires at least 15 GiB free after dependency installation")
    started = time.perf_counter()
    environments = Environments(
        args.workers,
        args.seed,
        profile=args.profile_collection,
        snapshots=saved["workers"] if saved else None,
    )
    try:
        transport = environments.benchmark_transport() if args.profile_collection else None
        observation = environments.observation
        shapes = {key: list(value.shape[1:]) for key, value in observation.items()}
        model = ActorCritic(shapes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        initial_parameters = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
        agents = args.workers * 4
        state = model.initial_state(agents, device)
        starts = torch.ones(agents, device=device)
        config = {
            **vars(args),
            "output": str(output.resolve()),
            "environment": configuration(),
            "shapes": shapes,
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "parameter_count": initial_parameters.numel(),
            "gamma": 0.998,
            "gae_lambda": 0.98,
            "learning_rate": 3e-4,
            "ppo_clip": 0.3,
            "entropy_coefficient": 0.025,
            "value_coefficient": 0.05,
            "value_clip": 50.0,
            "gradient_clip": 10.0,
        }
        (output / "config.json").write_text(json.dumps(config, indent=2))
        progress = {
            "update": 0,
            "environment_steps": 0,
            "agent_transitions": 0,
            "episodes_completed": 0,
        }
        totals: Totals = {
            "rollout_seconds": 0.0,
            "update_seconds": 0.0,
            "sampled_peak_rss_bytes": 0,
            "collection_profile": {},
        }
        schedules = {}  # Future parent-owned tax/entropy schedules must be saved here.
        if saved:
            if saved["config"]["shapes"] != shapes:
                raise ValueError("Incompatible observation shapes")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            state = tuple(value.to(device) for value in saved["recurrent"])
            if len(state) != 4 or any(value.shape != (1, agents, 128) for value in state):
                raise ValueError("Incompatible recurrent state layout")
            starts = saved["starts"].to(device)
            for key, value in observation.items():
                np.testing.assert_array_equal(value, saved["observation"][key].numpy())
            progress = saved["progress"].copy()
            totals = saved["totals"]
            schedules = saved["schedules"]
            initial_parameters = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
            checkpoints.restore_rng(saved["rng"], device)
        first_update = progress["update"]
        checkpoint = None
        run_started = time.perf_counter()
        for update in range(first_update, args.updates):
            if shutil.disk_usage(output).free < 15 * 1024**3:
                raise RuntimeError("Free space fell below the pilot's 15 GiB reserve")
            profile = CollectionProfile(args.profile_collection, device)
            rollout_started = time.perf_counter()
            observation_buffer = {key: [] for key in observation}
            action_buffer, logprob_buffer, value_buffer = [], [], []
            reward_buffer, done_buffer, start_buffer, state_buffer = [], [], [], []
            for step in range(args.horizon):
                if step % args.sequence_length == 0:
                    state_buffer.append(tuple(value.detach().clone() for value in state))
                for key in observation:
                    observation_buffer[key].append(observation[key])
                start_buffer.append(starts)
                with profile.measure("observation_to_device_seconds"):
                    policy_observation = tensors(observation, device, sequence=True)
                with profile.measure("policy_seconds"), torch.no_grad():
                    distribution, values, state = model(policy_observation, state, starts[None])
                    actions = distribution.sample()
                    logprob_buffer.append(distribution.log_prob(actions)[0])
                    value_buffer.append(values[0])
                action_buffer.append(actions[0])
                with profile.measure("action_to_cpu_seconds"):
                    cpu_actions = actions[0].cpu().numpy()
                with profile.measure("environment_seconds"):
                    observation, reward, done, completed = environments.step(cpu_actions)
                if args.profile_collection:
                    for key, value in environments.timings.items():
                        profile.seconds[key] += value
                for episode in completed:
                    checkpoints.append_log(
                        output / "episodes.jsonl",
                        json.dumps({"update": update + 1, **episode}, default=json_default),
                    )
                progress["episodes_completed"] += len(completed)
                reward_buffer.append(reward)
                done_buffer.append(done)
                with profile.measure("done_to_device_seconds"):
                    starts = torch.as_tensor(done, device=device)
            with profile.measure("bootstrap_seconds"), torch.no_grad():
                _, bootstrap, _ = model(
                    tensors(observation, device, sequence=True), state, starts[None]
                )
            rollout_seconds = time.perf_counter() - rollout_started
            update_started = time.perf_counter()
            rewards = torch.as_tensor(np.stack(reward_buffer), device=device)
            dones = torch.as_tensor(np.stack(done_buffer), device=device)
            old_values = torch.stack(value_buffer)
            advantage, returns = advantages(rewards, old_values, dones, bootstrap[0])
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
            length = args.sequence_length
            data = {
                key: sequences(value, length)
                for key, value in tensors(
                    {key: np.stack(value) for key, value in observation_buffer.items()}, device
                ).items()
            }
            old_logprob = sequences(torch.stack(logprob_buffer), length)
            old_value = sequences(old_values, length)
            chosen_actions = sequences(torch.stack(action_buffer), length)
            batch_starts = sequences(torch.stack(start_buffer), length)
            batch_advantage = sequences(advantage, length)
            batch_return = sequences(returns, length)
            batch_states = tuple(
                torch.cat([state[i] for state in state_buffer], dim=1) for i in range(4)
            )
            sequence_count = chosen_actions.shape[1]
            losses, entropies, divergences, norms = [], [], [], []
            # The pre-update ratio must equal one. This checks chunk ordering,
            # saved recurrent states, episode resets, and action alignment.
            with torch.no_grad():
                old_distribution, _, _ = model(data, batch_states, batch_starts)
                ratio_error = (
                    (old_distribution.log_prob(chosen_actions) - old_logprob).abs().max().item()
                )
                if ratio_error > 1e-4:
                    raise AssertionError(f"Recurrent rollout replay mismatch: {ratio_error}")
            for _ in range(args.epochs):
                order = torch.randperm(sequence_count, device=device)
                for selection in order.split(args.minibatch_sequences):
                    distribution, value, _ = model(
                        {key: value[:, selection] for key, value in data.items()},
                        tuple(value[:, selection] for value in batch_states),
                        batch_starts[:, selection],
                    )
                    log_ratio = (
                        distribution.log_prob(chosen_actions[:, selection])
                        - old_logprob[:, selection]
                    )
                    ratio = log_ratio.exp()
                    adv = batch_advantage[:, selection]
                    policy_loss = -torch.minimum(ratio * adv, ratio.clamp(0.7, 1.3) * adv).mean()
                    value_clipped = old_value[:, selection] + (
                        value - old_value[:, selection]
                    ).clamp(-50, 50)
                    value_loss = (
                        0.5
                        * torch.maximum(
                            (value - batch_return[:, selection]).square(),
                            (value_clipped - batch_return[:, selection]).square(),
                        ).mean()
                    )
                    entropy = distribution.entropy().mean()
                    loss = policy_loss + 0.05 * value_loss - 0.025 * entropy
                    if not torch.isfinite(loss):
                        raise AssertionError("Non-finite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 10, error_if_nonfinite=True
                    )
                    optimizer.step()
                    losses.append(loss.item())
                    entropies.append(entropy.item())
                    divergences.append(((ratio - 1) - log_ratio).mean().item())
                    norms.append(norm.item())
            if device.type == "cuda":
                torch.cuda.synchronize()
            update_seconds = time.perf_counter() - update_started
            record = {
                "update": update + 1,
                "environment_steps": (update + 1) * args.horizon * args.workers,
                "agent_transitions": (update + 1) * args.horizon * agents,
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "loss": float(np.mean(losses)),
                "entropy": float(np.mean(entropies)),
                "approx_kl": float(np.mean(divergences)),
                "gradient_norm_max": max(norms),
                "replay_logprob_max_error": ratio_error,
                "episodes_completed": progress["episodes_completed"],
                "reward_mean": rewards.mean().item(),
                "rss_bytes": rss_bytes(),
            }
            if args.profile_collection:
                record["collection_profile"] = dict(profile.seconds)
            checkpoints.append_log(output / "metrics.jsonl", json.dumps(record))
            progress.update(
                update=update + 1,
                environment_steps=(update + 1) * args.horizon * args.workers,
                agent_transitions=(update + 1) * args.horizon * agents,
            )
            totals["rollout_seconds"] += rollout_seconds
            totals["update_seconds"] += update_seconds
            totals["sampled_peak_rss_bytes"] = max(totals["sampled_peak_rss_bytes"], rss_bytes())
            for key, value in profile.seconds.items():
                totals["collection_profile"][key] = totals["collection_profile"].get(key, 0) + value
            if (update + 1) % args.checkpoint_every == 0 or update + 1 == args.updates:
                checkpoint = checkpoints.save(
                    output,
                    {
                        "version": checkpoints.VERSION,
                        "compatibility": checkpoints.compatibility(),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                        "progress": progress,
                        "totals": totals,
                        "schedules": schedules,
                        "recurrent": tuple(value.detach().cpu() for value in state),
                        "starts": starts.cpu(),
                        "observation": {
                            key: torch.from_numpy(value) for key, value in observation.items()
                        },
                        "workers": environments.snapshot(),
                        "rng": checkpoints.rng_state(device),
                        "log_offsets": checkpoints.log_offsets(output),
                    },
                    args.keep_checkpoints,
                )
            if (update + 1) % 5 == 0:
                print(json.dumps(record), flush=True)
        training_seconds = time.perf_counter() - run_started
        parameters = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
        parameter_change = torch.linalg.vector_norm(parameters - initial_parameters).item()
        assert parameter_change > 0, "Optimizer did not change model parameters"
        assert checkpoint is not None
        # Verify CPU checkpoint portability and CPU/CUDA forward agreement.
        restored = ActorCritic(shapes)
        restored.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        )
        model.eval()
        restored.eval()
        with torch.no_grad():
            test_obs = tensors(observation, device, sequence=True)
            gpu_policy, gpu_values, _ = model(
                test_obs, model.initial_state(agents, device), torch.ones(1, agents, device=device)
            )
            cpu_policy, cpu_values, _ = restored(
                {key: value.cpu() for key, value in test_obs.items()},
                restored.initial_state(agents, "cpu"),
                torch.ones(1, agents),
            )
            torch.testing.assert_close(
                gpu_policy.probs.cpu(), cpu_policy.probs, rtol=1e-4, atol=1e-5
            )
            torch.testing.assert_close(gpu_values.cpu(), cpu_values, rtol=1e-4, atol=1e-5)
        summary = {
            "status": "passed",
            "device": str(device),
            "environment_steps": progress["environment_steps"],
            "agent_transitions": progress["agent_transitions"],
            "completed_updates": progress["update"],
            "source_checkpoint": args.resume,
            "checkpoint": checkpoint.name,
            "training_seconds": training_seconds,
            "environment_steps_per_second": (args.updates - first_update)
            * args.horizon
            * args.workers
            / training_seconds,
            "rollout_seconds": totals["rollout_seconds"],
            "update_seconds": totals["update_seconds"],
            "episodes_completed": progress["episodes_completed"],
            "parameter_change_l2": parameter_change,
            "sampled_peak_rss_bytes": totals["sampled_peak_rss_bytes"],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else 0,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
            if device.type == "cuda"
            else 0,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_cpu_portability": True,
            "cpu_device_forward_agreement": True,
            "disk_free_before_bytes": disk_before,
            "disk_free_after_bytes": shutil.disk_usage(output).free,
            "total_seconds": time.perf_counter() - started,
            "interpretation": "Engineering pilot only. No claim of economic convergence or paper reproduction.",
        }
        if args.profile_collection:
            summary["transport_probe"] = transport
            summary["collection_profile"] = totals["collection_profile"]
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        environments.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-collection", action="store_true")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), help="Defaults to cuda, or inherits the resumed device"
    )
    parser.add_argument(
        "--output", help="New directory; defaults to a unique results/train-* directory"
    )
    parser.add_argument("--resume", help="Trusted checkpoint; creates a linked child run")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--minibatch-sequences", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()
    train(arguments)
