"""Small recurrent PPO model using Foundation's spatial and private observations.

Each policy/value tower uses index embeddings, two convolutions, two dense
layers, layer normalization, and an LSTM, following the upstream model layout.
This is a new implementation, not a validated conversion of TensorFlow weights.
"""

from typing import cast

import torch
from torch import nn
from torch.distributions import Categorical


class Tower(nn.Module):
    def __init__(self, shapes, outputs, hidden=128):
        super().__init__()
        self.hidden = hidden
        channels, height, width = shapes["world-map"]
        index_channels = shapes["world-idx_map"][0]
        self.embedding = nn.Embedding(100, 4)
        self.convolution = nn.Sequential(
            nn.Conv2d(channels + index_channels * 4, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            size = self.convolution(
                torch.zeros(1, channels + index_channels * 4, height, width)
            ).shape[-1]
        self.dense = nn.Sequential(
            nn.Linear(size + shapes["flat"][0], hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.recurrent = nn.LSTM(hidden, hidden)
        self.output = nn.Linear(hidden, outputs)

    def forward(self, observations, state, starts):
        time, batch = observations["flat"].shape[:2]
        indices = observations["world-idx_map"].long().flatten(0, 1)
        embedded = self.embedding(indices).permute(0, 1, 4, 2, 3).flatten(1, 2)
        spatial = torch.cat((observations["world-map"].flatten(0, 1), embedded), dim=1)
        encoded = self.dense(
            torch.cat((self.convolution(spatial), observations["flat"].flatten(0, 1)), dim=-1)
        )
        encoded = encoded.reshape(time, batch, self.hidden)
        state = tuple(value * (1 - starts[0]).view(1, batch, 1) for value in state)
        if time == 1 or not starts[1:].any():
            output, state = self.recurrent(encoded, state)
        else:
            pieces = []
            for t in range(time):
                state = tuple(value * (1 - starts[t]).view(1, batch, 1) for value in state)
                output, state = self.recurrent(encoded[t : t + 1], state)
                pieces.append(output)
            output = torch.cat(pieces)
        return self.output(output), state


class ActorCritic(nn.Module):
    def __init__(self, shapes):
        super().__init__()
        self.policy = Tower(shapes, shapes["action_mask"][0])
        self.value = Tower(shapes, 1)

    def initial_state(self, batch, device):
        return tuple(torch.zeros(1, batch, 128, device=device) for _ in range(4))

    def forward(self, observations, state, starts):
        logits, policy_state = self.policy(observations, state[:2], starts)
        values, value_state = self.value(observations, state[2:], starts)
        logits = logits.masked_fill(observations["action_mask"] <= 0, -1e9)
        return Categorical(logits=logits), values.squeeze(-1), (*policy_state, *value_state)


def advantages(rewards, values, dones, bootstrap, gamma=0.998, lam=0.98):
    """GAE with true episode termination and bootstrap at rollout boundaries."""
    result = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for t in reversed(range(len(rewards))):
        continuation = 1 - dones[t]
        delta = rewards[t] + gamma * continuation * next_value - values[t]
        carry = delta + gamma * lam * continuation * carry
        result[t] = carry
        next_value = values[t]
    return result, result + values


class MultiCategorical:
    """Independent bracket actions, with joint log probability and summed entropy."""

    def __init__(self, logits, sizes):
        self.parts = [Categorical(logits=part) for part in logits.split(sizes, dim=-1)]

    def sample(self):
        return torch.stack([part.sample() for part in self.parts], dim=-1)

    def log_prob(self, actions):
        return torch.stack(
            [part.log_prob(actions[..., i]) for i, part in enumerate(self.parts)], dim=-1
        ).sum(-1)

    def entropy(self):
        return torch.stack([part.entropy() for part in self.parts], dim=-1).sum(-1)

    @property
    def probs(self):
        return torch.cat([cast(torch.Tensor, part.probs) for part in self.parts], dim=-1)


class PlannerTower(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256)
        )
        self.recurrent = nn.LSTM(256, 256)
        self.output = nn.Linear(256, outputs)

    def forward(self, observations, state, starts):
        encoded = self.dense(observations["flat"])
        pieces = []
        for t in range(len(encoded)):
            state = tuple(value * (1 - starts[t]).view(1, -1, 1) for value in state)
            output, state = self.recurrent(encoded[t : t + 1], state)
            pieces.append(output)
        return self.output(torch.cat(pieces)), state


class PlannerActorCritic(nn.Module):
    def __init__(self, shapes, action_sizes):
        super().__init__()
        if sum(action_sizes) != shapes["action_mask"][0] or min(action_sizes) < 1:
            raise ValueError("Incompatible planner action layout")
        self.action_sizes = action_sizes
        if "world-map" in shapes:
            self.policy = Tower(shapes, sum(action_sizes), hidden=256)
            self.value = Tower(shapes, 1, hidden=256)
        else:
            self.policy = PlannerTower(shapes["flat"][0], sum(action_sizes))
            self.value = PlannerTower(shapes["flat"][0], 1)

    def initial_state(self, batch, device):
        return tuple(torch.zeros(1, batch, 256, device=device) for _ in range(4))

    def forward(self, observations, state, starts):
        logits, policy_state = self.policy(observations, state[:2], starts)
        values, value_state = self.value(observations, state[2:], starts)
        for mask in observations["action_mask"].split(self.action_sizes, dim=-1):
            if not (mask > 0).any(-1).all():
                raise ValueError("Planner action subspace has no valid action")
        logits = logits.masked_fill(observations["action_mask"] <= 0, -1e9)
        return (
            MultiCategorical(logits, self.action_sizes),
            values.squeeze(-1),
            (*policy_state, *value_state),
        )
