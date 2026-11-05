"""Finite, resource-measured worker and joint worker–planner recurrent PPO pilot.

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
from environment import TAX_REGIMES, Environments, action_shares, configuration_from_run
from model import ActorCritic, PlannerActorCritic, advantages
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


def ppo_update(
    model,
    optimizer,
    args,
    device,
    observation_buffer,
    action_buffer,
    logprob_buffer,
    value_buffer,
    reward_buffer,
    done_buffer,
    start_buffer,
    state_buffer,
    bootstrap,
    entropy_coefficient,
):
    rewards = torch.as_tensor(np.stack(reward_buffer), device=device)
    dones = torch.as_tensor(np.stack(done_buffer), device=device)
    old_values = torch.stack(value_buffer)
    advantage, returns = advantages(rewards, old_values, dones, bootstrap[0])
    return_variance = returns.var(unbiased=False).item()
    explained_variance = (
        1 - (returns - old_values).var(unbiased=False).item() / return_variance
        if return_variance > 0
        else None
    )
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
    batch_states = tuple(torch.cat([state[i] for state in state_buffer], dim=1) for i in range(4))
    sequence_count = chosen_actions.shape[1]
    losses, entropies, divergences, norms = [], [], [], []
    # The pre-update ratio must equal one. This checks chunk ordering,
    # saved recurrent states, episode resets, and action alignment.
    with torch.no_grad():
        old_distribution, _, _ = model(data, batch_states, batch_starts)
        ratio_error = (old_distribution.log_prob(chosen_actions) - old_logprob).abs().max().item()
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
                distribution.log_prob(chosen_actions[:, selection]) - old_logprob[:, selection]
            )
            ratio = log_ratio.exp()
            adv = batch_advantage[:, selection]
            policy_loss = -torch.minimum(ratio * adv, ratio.clamp(0.7, 1.3) * adv).mean()
            value_clipped = old_value[:, selection] + (value - old_value[:, selection]).clamp(
                -50, 50
            )
            value_loss = (
                0.5
                * torch.maximum(
                    (value - batch_return[:, selection]).square(),
                    (value_clipped - batch_return[:, selection]).square(),
                ).mean()
            )
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.05 * value_loss - entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise AssertionError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            entropies.append(entropy.item())
            divergences.append(((ratio - 1) - log_ratio).mean().item())
            norms.append(norm.item())
    return {
        "loss": float(np.mean(losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(divergences)),
        "gradient_norm_max": max(norms),
        "replay_logprob_max_error": ratio_error,
        "reward_mean": rewards.mean().item(),
        "explained_variance": explained_variance,
        "entropy_coefficient": entropy_coefficient,
    }


def planner_entropy(steps, profile="pilot"):
    if profile == "paper2020":
        return 0.1
    # Upstream phase-two schedule, in aggregate environment steps of this phase.
    return 2.0 + (0.125 - 2.0) * min(steps / 50_000_000, 1.0)


class PlannerRollout:
    def __init__(self, model, state, starts):
        self.model, self.state, self.starts = model, state, starts
        self.observations = {}
        self.actions, self.logprobs, self.values = [], [], []
        self.rewards, self.dones, self.start_buffer, self.states = [], [], [], []

    @torch.no_grad()
    def act(self, observation, device, step, length):
        if step % length == 0:
            self.states.append(tuple(v.detach().clone() for v in self.state))
        for key, value in observation.items():
            self.observations.setdefault(key, []).append(value)
        self.start_buffer.append(self.starts)
        distribution, value, self.state = self.model(
            tensors(observation, device, sequence=True), self.state, self.starts[None]
        )
        action = distribution.sample()[0]
        self.actions.append(action)
        self.logprobs.append(distribution.log_prob(action[None])[0])
        self.values.append(value[0])
        return action.cpu().numpy()

    def update(self, optimizer, args, device, observation, coefficient):
        with torch.no_grad():
            _, bootstrap, _ = self.model(
                tensors(observation, device, sequence=True), self.state, self.starts[None]
            )
        return ppo_update(
            self.model,
            optimizer,
            args,
            device,
            self.observations,
            self.actions,
            self.logprobs,
            self.values,
            self.rewards,
            self.dones,
            self.start_buffer,
            self.states,
            bootstrap,
            coefficient,
        )


def transfer_workers(model, optimizer, shapes, source):
    """Transfer learning state; phase transition deliberately resets rollout state."""
    if source["config"]["shapes"] != shapes:
        raise ValueError("Phase-one worker observation layout is incompatible")
    model.load_state_dict(source["model"])
    optimizer.load_state_dict(source["optimizer"])


def train(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.resume and args.phase2_from:
        raise ValueError("--resume and --phase2-from are mutually exclusive")
    saved = checkpoints.load(args.resume) if args.resume else None
    phase_one = checkpoints.load(args.phase2_from) if args.phase2_from else None
    if phase_one:
        args.phase2_from = str(Path(args.phase2_from).resolve())
        if phase_one["config"]["tax_regime"] != "free-market" or phase_one["config"].get(
            "phase2_from"
        ):
            raise ValueError("Phase two requires a free-market phase-one checkpoint")
        args.tax_regime = args.tax_regime or "learned-tax"
        if args.profile is not None and args.profile != phase_one["config"]["profile"]:
            raise ValueError("Phase transition requires the same training profile")
        for key in (
            "profile",
            "tax_anneal_steps",
            "labor_warmup_constant",
            "workers",
            "seed",
            "horizon",
            "sequence_length",
            "epochs",
            "minibatch_sequences",
        ):
            if getattr(args, key) is None:
                setattr(args, key, phase_one["config"][key])
    args.device = args.device or (saved["config"]["device"] if saved else "cuda")
    if args.resume:
        assert saved is not None
        args.resume = str(Path(args.resume).resolve())
        args.phase2_from = saved["config"].get("phase2_from")
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
            "profile",
            "tax_anneal_steps",
            "labor_warmup_constant",
            "tax_regime",
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
        if saved["config"]["environment"] != configuration_from_run(vars(args)):
            raise ValueError("Incompatible environment configuration")
        if args.updates <= saved["progress"]["update"]:
            raise ValueError("--updates is a total target and must exceed the saved update")
    else:
        args.profile = args.profile or "pilot"
        paper = args.profile == "paper2020"
        for key, value in {
            "tax_regime": "free-market",
            "workers": 2,
            "seed": 42,
            "horizon": 200 if paper else 128,
            "sequence_length": 50 if paper else 16,
            "epochs": 1 if paper else 2,
            "minibatch_sequences": 60 if paper else 16,
            "tax_anneal_steps": 54_000_000 if paper else None,
            "labor_warmup_constant": 10000 if paper else None,
        }.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    if args.tax_regime == "learned-tax" and not (saved or phase_one):
        raise ValueError("Learned taxes require --phase2-from or --resume")
    joint = args.tax_regime == "learned-tax"
    first_update = saved["progress"]["update"] if saved else 0
    max_updates = 999999 if args.profile == "paper2020" else 100
    if not (
        1 <= args.workers <= 4
        and 1 <= args.updates <= max_updates
        and 1 <= args.updates - first_update <= 100
    ):
        raise ValueError(
            "workers must be 1..4; at most 100 additional updates per invocation; "
            f"total updates must be 1..{max_updates}"
        )
    if args.profile == "pilot" and (
        args.tax_anneal_steps is not None or args.labor_warmup_constant is not None
    ):
        raise ValueError("Explicit curricula require --profile paper2020")
    if args.profile == "paper2020" and (
        not np.isfinite(args.labor_warmup_constant)
        or args.labor_warmup_constant < 0
        or args.tax_anneal_steps < 1
    ):
        raise ValueError("Labor warmup must be finite/nonnegative and tax duration positive")
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
        tax_regime=args.tax_regime,
        profile=args.profile_collection,
        snapshots=saved["workers"] if saved else None,
        config=configuration_from_run(vars(args)),
    )
    try:
        transport = environments.benchmark_transport() if args.profile_collection else None
        observation = environments.observation
        shapes = {key: list(value.shape[1:]) for key, value in observation.items()}
        model = ActorCritic(shapes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        if phase_one:
            transfer_workers(model, optimizer, shapes, phase_one)
        initial_parameters = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
        agents = args.workers * 4
        state = model.initial_state(agents, device)
        starts = torch.ones(agents, device=device)
        planner = planner_optimizer = planner_state = planner_starts = None
        planner_shapes = None
        planner_initial_parameters = None
        if joint:
            planner_shapes = {
                k: list(v.shape[1:]) for k, v in environments.planner_observation.items()
            }
            planner = PlannerActorCritic(planner_shapes, environments.planner_action_sizes).to(
                device
            )
            planner_optimizer = torch.optim.Adam(planner.parameters(), lr=1e-4)
            planner_state = planner.initial_state(args.workers, device)
            planner_starts = torch.ones(args.workers, device=device)
        config = {
            **vars(args),
            "output": str(output.resolve()),
            "environment": configuration_from_run(vars(args)),
            "shapes": shapes,
            "planner_shapes": planner_shapes,
            "planner_parameter_count": sum(p.numel() for p in planner.parameters())
            if planner is not None
            else 0,
            "planner_action_sizes": environments.planner_action_sizes,
            "planner_learning_rate": 1e-4 if joint else None,
            "planner_entropy_schedule": (
                [[0, 0.1]] if args.profile == "paper2020" else [[0, 2.0], [50_000_000, 0.125]]
            )
            if joint
            else None,
            "phase_one_progress": phase_one["progress"]
            if phase_one
            else saved["config"].get("phase_one_progress")
            if saved
            else None,
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
        schedules = {
            "phase": "phase2" if args.phase2_from else "worker-only",
            "phase_steps": 0,
            "worker_entropy": 0.025,
            "planner_entropy": planner_entropy(0, args.profile) if joint else None,
        }
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
            if joint:
                assert planner is not None and planner_optimizer is not None
                if (
                    saved["config"]["planner_shapes"] != planner_shapes
                    or saved["config"]["planner_action_sizes"] != environments.planner_action_sizes
                ):
                    raise ValueError("Incompatible planner observation/action layout")
                planner.load_state_dict(saved["planner"]["model"])
                planner_optimizer.load_state_dict(saved["planner"]["optimizer"])
                planner_state = tuple(v.to(device) for v in saved["planner"]["recurrent"])
                if len(planner_state) != 4 or any(
                    v.shape != (1, args.workers, 256) for v in planner_state
                ):
                    raise ValueError("Incompatible planner recurrent layout")
                planner_starts = saved["planner"]["starts"].to(device)
                if planner_starts.shape != (args.workers,):
                    raise ValueError("Incompatible planner start-mask layout")
                for k, v in environments.planner_observation.items():
                    np.testing.assert_array_equal(v, saved["planner"]["observation"][k].numpy())
            expected_schedules = {
                "phase": "phase2" if args.phase2_from else "worker-only",
                "phase_steps": progress["environment_steps"],
                "worker_entropy": 0.025,
                "planner_entropy": planner_entropy(progress["environment_steps"], args.profile)
                if joint
                else None,
            }
            if schedules != expected_schedules:
                raise ValueError("Incompatible schedule state")
            checkpoints.restore_rng(saved["rng"], device)
        if joint:
            assert planner is not None
            planner_initial_parameters = torch.cat(
                [p.detach().flatten().cpu() for p in planner.parameters()]
            )
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
            planner_rollout = (
                PlannerRollout(planner, planner_state, planner_starts) if joint else None
            )
            action_counts = np.zeros(shapes["action_mask"][0], dtype=np.int64)
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
                np.add.at(action_counts, cpu_actions, 1)
                planner_actions = None
                if planner_rollout is not None:
                    with profile.measure("planner_policy_seconds"):
                        planner_actions = planner_rollout.act(
                            environments.planner_observation, device, step, args.sequence_length
                        )
                with profile.measure("environment_seconds"):
                    observation, reward, done, completed = environments.step(
                        cpu_actions, planner_actions
                    )
                if planner_rollout is not None:
                    planner_done = done[::4]
                    planner_rollout.rewards.append(environments.planner_reward)
                    planner_rollout.dones.append(planner_done)
                    planner_rollout.starts = torch.as_tensor(planner_done, device=device)
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
            worker_metrics = ppo_update(
                model,
                optimizer,
                args,
                device,
                observation_buffer,
                action_buffer,
                logprob_buffer,
                value_buffer,
                reward_buffer,
                done_buffer,
                start_buffer,
                state_buffer,
                bootstrap,
                0.025,
            )
            planner_metrics = None
            if planner_rollout is not None:
                planner_metrics = planner_rollout.update(
                    planner_optimizer,
                    args,
                    device,
                    environments.planner_observation,
                    planner_entropy(progress["environment_steps"], args.profile),
                )
                planner_state, planner_starts = planner_rollout.state, planner_rollout.starts
            if device.type == "cuda":
                torch.cuda.synchronize()
            update_seconds = time.perf_counter() - update_started
            record = {
                "update": update + 1,
                "environment_steps": (update + 1) * args.horizon * args.workers,
                "agent_transitions": (update + 1) * args.horizon * agents,
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                **worker_metrics,
                "action_shares": action_shares(action_counts),
                "episodes_completed": progress["episodes_completed"],
                "rss_bytes": rss_bytes(),
            }
            if planner_metrics is not None:
                record["planner"] = planner_metrics
            if args.profile_collection:
                record["collection_profile"] = dict(profile.seconds)
            checkpoints.append_log(output / "metrics.jsonl", json.dumps(record))
            progress.update(
                update=update + 1,
                environment_steps=(update + 1) * args.horizon * args.workers,
                agent_transitions=(update + 1) * args.horizon * agents,
            )
            schedules.update(
                phase_steps=progress["environment_steps"],
                planner_entropy=planner_entropy(progress["environment_steps"], args.profile)
                if joint
                else None,
            )
            totals["rollout_seconds"] += rollout_seconds
            totals["update_seconds"] += update_seconds
            totals["sampled_peak_rss_bytes"] = max(totals["sampled_peak_rss_bytes"], rss_bytes())
            for key, value in profile.seconds.items():
                totals["collection_profile"][key] = totals["collection_profile"].get(key, 0) + value
            planner_snapshot = None
            if joint:
                assert planner is not None and planner_optimizer is not None
                assert planner_state is not None and planner_starts is not None
                planner_snapshot = {
                    "model": planner.state_dict(),
                    "optimizer": planner_optimizer.state_dict(),
                    "recurrent": tuple(v.detach().cpu() for v in planner_state),
                    "starts": planner_starts.cpu(),
                    "observation": {
                        k: torch.from_numpy(v) for k, v in environments.planner_observation.items()
                    },
                }
            if (update + 1) % args.checkpoint_every == 0 or update + 1 == args.updates:
                checkpoint = checkpoints.save(
                    output,
                    {
                        "version": checkpoints.VERSION,
                        "compatibility": checkpoints.compatibility(),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "planner": planner_snapshot,
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
        planner_change = None
        if joint:
            assert planner is not None and planner_initial_parameters is not None
            planner_change = torch.linalg.vector_norm(
                torch.cat([p.detach().flatten().cpu() for p in planner.parameters()])
                - planner_initial_parameters
            ).item()
            assert planner_change > 0, "Planner optimizer did not change parameters"
            restored_planner = PlannerActorCritic(planner_shapes, environments.planner_action_sizes)
            restored_planner.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)["planner"]["model"]
            )
            with torch.no_grad():
                test_obs = tensors(environments.planner_observation, device, sequence=True)
                policy, values, _ = planner(
                    test_obs,
                    planner.initial_state(args.workers, device),
                    torch.ones(1, args.workers, device=device),
                )
                cpu_policy, cpu_values, _ = restored_planner(
                    {k: v.cpu() for k, v in test_obs.items()},
                    restored_planner.initial_state(args.workers, "cpu"),
                    torch.ones(1, args.workers),
                )
                torch.testing.assert_close(
                    policy.probs.cpu(), cpu_policy.probs, rtol=1e-4, atol=1e-5
                )
                torch.testing.assert_close(values.cpu(), cpu_values, rtol=1e-4, atol=1e-5)
        summary = {
            "planner_parameter_change_l2": planner_change,
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
    parser.add_argument(
        "--phase2-from", help="Warm-start any tax regime from a free-market phase-one checkpoint"
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
    parser.add_argument("--tax-regime", choices=TAX_REGIMES)
    parser.add_argument("--profile", choices=("pilot", "paper2020"))
    parser.add_argument(
        "--tax-anneal-steps", type=int, help="Aggregate phase-two steps to reach 100%% tax support"
    )
    parser.add_argument(
        "--labor-warmup-constant",
        type=float,
        help="Foundation automatic warmup constant, per economy (0 = full cost)",
    )
    arguments = parser.parse_args()
    train(arguments)
