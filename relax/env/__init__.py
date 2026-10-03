import numpy as np
from gymnasium import Env, Wrapper, make
from gymnasium.spaces import Box

from relax.env.vector import VectorEnv, SerialVectorEnv, GymProcessVectorEnv, PipeProcessVectorEnv, SpinlockProcessVectorEnv, FutexProcessVectorEnv


class MultiStepVectorWrapper(VectorEnv):
    """Execute ``act_steps`` sub-actions per environment step.

    The wrapper exposes a flat action space of dimension
    ``act_steps * original_act_dim``.  Internally it reshapes the flat
    action into ``(act_steps, original_act_dim)`` and steps the inner
    ``VectorEnv`` sequentially, accumulating rewards.  If an episode
    terminates during the chunk the remaining sub-steps still execute
    (the worker auto-resets) but their rewards are masked out.

    This mirrors the original ReinFlow ``MultiStep`` wrapper but works
    at the vectorised-environment level.
    """

    def __init__(self, inner: VectorEnv, act_steps: int):
        assert act_steps >= 1
        self._inner = inner
        self._act_steps = act_steps
        self._orig_act_dim = inner.act_dim

        self.num_envs = inner.num_envs
        self.obs_dim = inner.obs_dim
        self.act_dim = act_steps * inner.act_dim

        self.single_observation_space = inner.single_observation_space
        self.observation_space = inner.observation_space

        self.single_action_space = Box(
            low=np.tile(inner.single_action_space.low, act_steps),
            high=np.tile(inner.single_action_space.high, act_steps),
            dtype=inner.single_action_space.dtype,
        )
        self.action_space = Box(
            low=np.tile(inner.action_space.low, (1, act_steps)),
            high=np.tile(inner.action_space.high, (1, act_steps)),
            dtype=inner.action_space.dtype,
        )

        if hasattr(inner, "spec"):
            self.spec = inner.spec

    def reset(self, *, seed=None, options=None):
        return self._inner.reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        sub_acts = action.reshape(self.num_envs, self._act_steps, self._orig_act_dim)

        cum_reward = np.zeros(self.num_envs, dtype=np.float64)
        alive = np.ones(self.num_envs, dtype=bool)
        chunk_terminated = np.zeros(self.num_envs, dtype=bool)
        chunk_truncated = np.zeros(self.num_envs, dtype=bool)
        chunk_obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)

        for k in range(self._act_steps):
            obs, reward, terminated, truncated, info = self._inner.step(sub_acts[:, k])

            cum_reward += reward * alive.astype(np.float64)

            just_done = (terminated | truncated) & alive
            chunk_terminated |= terminated & alive
            chunk_truncated |= truncated & alive & ~chunk_terminated

            chunk_obs[just_done] = obs[just_done]
            alive &= ~(terminated | truncated)

            if np.any(just_done) and k < self._act_steps - 1:
                self._inner.reset()

        still_alive = ~(chunk_terminated | chunk_truncated)
        chunk_obs[still_alive] = obs[still_alive]

        return chunk_obs, cum_reward, chunk_terminated, chunk_truncated, info

    def close(self):
        self._inner.close()

    @property
    def unwrapped(self):
        return self._inner.unwrapped


class RelaxWrapper(Wrapper):
    def __init__(self, env: Env, action_seed: int = 0):
        super().__init__(env)
        self.env: Env[np.ndarray, np.ndarray]

        assert isinstance(env.observation_space, Box)
        assert isinstance(env.action_space, Box) and env.action_space.is_bounded()
        if isinstance(env, VectorEnv):
            _, self.obs_dim = env.observation_space.shape
            _, self.act_dim = env.action_space.shape
            single_action_space = env.single_action_space
        else:
            self.obs_dim, = env.observation_space.shape
            self.act_dim, = env.action_space.shape
            single_action_space = env.action_space

        if np.any(single_action_space.low != -1.0) or np.any(single_action_space.high != 1.0):
            print(f"NOTE: The action space is not normalized, but {single_action_space.low} to {single_action_space.high}, will be rescaled.")
            self.needs_rescale = True
            self.original_action_center = (single_action_space.low + single_action_space.high) * 0.5
            self.original_action_half_range = (single_action_space.high - single_action_space.low) * 0.5
        else:
            self.needs_rescale = False
        self.original_action_dtype = env.action_space.dtype

        self._action_space = Box(
            low=-1,
            high=1,
            shape=env.action_space.shape,
            dtype=np.float32,
            seed=action_seed
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.astype(np.float32, copy=False), info

    def step(self, action: np.ndarray):
        action = action.astype(self.original_action_dtype)
        if self.needs_rescale:
            action *= self.original_action_half_range
            action += self.original_action_center
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32, copy=False), reward, terminated, truncated, info

def create_env(name: str, seed: int, action_seed: int = 0):
    env = make(name)
    env.reset(seed=seed)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim

def create_vector_env(name: str, num_envs: int, seed: int, action_seed: int = 0, mode: str = "serial", act_steps: int = 1, **kwargs):
    Impl = {
        "serial": SerialVectorEnv,
        "gym": GymProcessVectorEnv,
        "pipe": PipeProcessVectorEnv,
        "spinlock": SpinlockProcessVectorEnv,
        "futex": FutexProcessVectorEnv,
    }[mode]
    env = Impl(name, num_envs, seed, **kwargs)
    if act_steps > 1:
        env = MultiStepVectorWrapper(env, act_steps)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim
