"""Checks for return targets, action validity, and recurrent episode boundaries."""

import unittest

import torch
from model import ActorCritic, advantages


class PilotChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_terminal_return_ignores_next_episode_and_bootstrap(self):
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.tensor([[0.5], [0.25]])
        _, returns = advantages(
            rewards, values, torch.tensor([[1.0], [0.0]]), torch.tensor([3.0]), gamma=1, lam=1
        )
        torch.testing.assert_close(returns, torch.tensor([[1.0], [5.0]]))

    def test_nonterminal_return_includes_bootstrap(self):
        _, returns = advantages(
            torch.tensor([[1.0], [2.0]]),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.tensor([3.0]),
            gamma=1,
            lam=1,
        )
        torch.testing.assert_close(returns, torch.tensor([[6.0], [5.0]]))

    def test_mask_and_episode_reset(self):
        shapes = {
            "flat": [5],
            "world-map": [3, 11, 11],
            "world-idx_map": [2, 11, 11],
            "action_mask": [4],
        }
        model = ActorCritic(shapes)
        observation = {key: torch.zeros(3, 2, *shape) for key, shape in shapes.items()}
        observation["action_mask"][..., 2] = 1
        initial = tuple(torch.ones(1, 2, 128) for _ in range(4))
        with torch.no_grad():
            distribution, value, state = model(
                observation, initial, torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
            )
            self.assertTrue((distribution.sample() == 2).all())
            _, fresh_value, fresh_state = model(
                {k: v[2:] for k, v in observation.items()},
                model.initial_state(2, "cpu"),
                torch.ones(1, 2),
            )
            torch.testing.assert_close(value[2:], fresh_value)
            for actual, expected in zip(state, fresh_state):
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
