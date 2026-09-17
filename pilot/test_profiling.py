"""Profiling must preserve environment trajectories and the random stream."""

import unittest

import numpy as np
from environment import Environments


class ProfilingChecks(unittest.TestCase):
    def trajectory(self, profile):
        environments = Environments(1, 42, profile=profile)
        try:
            if profile:
                probe = environments.benchmark_transport(iterations=4)
                self.assertGreater(probe["seconds"], 0)
            trajectory = []
            for _ in range(8):
                actions = environments.observation["action_mask"].argmax(axis=1)
                observation, reward, done, completed = environments.step(actions)
                trajectory.append((observation, reward, done, completed))
                if profile:
                    timings = environments.timings
                    self.assertGreater(timings["simulator_sum_seconds"], 0)
                    self.assertGreaterEqual(
                        timings["worker_processing_sum_seconds"],
                        timings["simulator_sum_seconds"],
                    )
            return trajectory
        finally:
            environments.close()

    def test_profile_and_echo_preserve_trajectory(self):
        plain = self.trajectory(False)
        profiled = self.trajectory(True)
        for (obs_a, reward_a, done_a, completed_a), (obs_b, reward_b, done_b, completed_b) in zip(
            plain, profiled
        ):
            for key in obs_a:
                np.testing.assert_array_equal(obs_a[key], obs_b[key])
            np.testing.assert_array_equal(reward_a, reward_b)
            np.testing.assert_array_equal(done_a, done_b)
            self.assertEqual(completed_a, completed_b)


if __name__ == "__main__":
    unittest.main()
