"""Small recurrent PPO model using Foundation's spatial and private observations.

Each policy/value tower uses index embeddings, two convolutions, two dense
layers, layer normalization, and an LSTM, following the upstream model layout.
This is a new implementation, not a validated conversion of TensorFlow weights.
"""

import torch
from torch import nn
from torch.distributions import Categorical


class Tower(nn.Module):
    def __init__(self, shapes, outputs):
        super().__init__()
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
            nn.Linear(size + shapes["flat"][0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )
        self.recurrent = nn.LSTM(128, 128)
        self.output = nn.Linear(128, outputs)

    def forward(self, observations, state, starts):
        time, batch = observations["flat"].shape[:2]
        indices = observations["world-idx_map"].long().flatten(0, 1)
        embedded = self.embedding(indices).permute(0, 1, 4, 2, 3).flatten(1, 2)
        spatial = torch.cat((observations["world-map"].flatten(0, 1), embedded), dim=1)
        encoded = self.dense(
            torch.cat((self.convolution(spatial), observations["flat"].flatten(0, 1)), dim=-1)
        )
        encoded = encoded.reshape(time, batch, 128)
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
