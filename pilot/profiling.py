"""Opt-in wall-clock timings; synchronize CUDA at measured boundaries."""

import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class CollectionProfile:
    def __init__(self, enabled, device):
        self.enabled = enabled
        self.device = device
        self.seconds = defaultdict(float)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds[name] += time.perf_counter() - started
