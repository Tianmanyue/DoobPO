"""
Running observation normalization using Welford's online algorithm.

Matches the FPO Playground ``RunningStats`` implementation.
"""

import numpy as np


class RunningObsStats:
    """Tracks running mean/std of observations using Welford's method."""

    def __init__(self, obs_dim: int, clip: float = 10.0):
        self.count = 0.0
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var_sum = np.zeros(obs_dim, dtype=np.float64)
        self.std = np.ones(obs_dim, dtype=np.float32)
        self.clip = clip

    def update(self, batch: np.ndarray):
        """Update stats with a batch of observations ``(N, obs_dim)``."""
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, self.mean.shape[0])
        n = batch.shape[0]
        if n == 0:
            return
        new_count = self.count + n
        batch_mean = batch.mean(axis=0)
        diff_old = batch - self.mean
        self.mean += diff_old.sum(axis=0) / new_count
        diff_new = batch - self.mean
        self.var_sum += (diff_old * diff_new).sum(axis=0)
        self.count = new_count
        var = np.clip(self.var_sum / max(self.count, 1.0), 1e-12, 1e12)
        self.std = np.sqrt(var).astype(np.float32)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations: ``(obs - mean) / std``, clipped."""
        out = (obs - self.mean.astype(np.float32)) / self.std
        return np.clip(out, -self.clip, self.clip).astype(np.float32)
