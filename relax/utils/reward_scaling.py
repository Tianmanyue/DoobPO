"""
Running reward normalization via discounted-return variance.

Divides rewards by sqrt(running_var(discounted_returns)).
Based on OpenAI PPG / Baselines implementation.
Reference: https://arxiv.org/pdf/2005.12729.pdf
"""

import numpy as np


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = M2 / (tot_count - 1)
        self.count = tot_count


class RunningRewardScaler:
    """Normalize rewards by sqrt(running variance of discounted returns).

    Tracks the running variance of backward-discounted returns and divides
    rewards by sqrt(var + eps).  Does NOT subtract the mean — only scales.
    """

    def __init__(self, num_envs: int, gamma: float = 0.99,
                 clip_reward: float = 10.0, epsilon: float = 1e-8):
        self.ret_rms = RunningMeanStd()
        self.clip_reward = clip_reward
        self.ret = np.zeros(num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def __call__(self, reward: np.ndarray, first: np.ndarray) -> np.ndarray:
        """Scale rewards in-place for a full rollout fragment.

        Parameters
        ----------
        reward : (num_envs, T)
        first  : (num_envs, T)  —  1 at episode boundaries, 0 otherwise
        """
        _, T = reward.shape
        rets = np.zeros_like(reward)
        prevret = self.ret.copy()
        for t in range(T):
            prevret = rets[:, t] = reward[:, t] + (1 - first[:, t]) * self.gamma * prevret
        self.ret = rets[:, -1]
        self.ret_rms.update(rets.reshape(-1))
        return np.clip(
            reward / np.sqrt(self.ret_rms.var + self.epsilon),
            -self.clip_reward, self.clip_reward,
        )
