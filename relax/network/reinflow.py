"""
ExploreNoiseNet for ReinFlow.

Predicts the per-step noise standard deviation σ(s,t) for SDE sampling.

Matches the original ReinFlow implementation:
  - Outputs per-dimension σ ∈ [σ_min, σ_max]  (B, act_dim)
  - Hidden dims [64, 64], Tanh activation
  - σ bounded via tanh → logvar → exp(0.5·logvar)
"""

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, sinusoidal_pos_emb


def _mish(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jax.nn.softplus(x))


def create_explore_noise_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    *,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: Activation = jax.nn.tanh,
    time_dim: int = 16,
    min_noise_std: float = 0.10,
    max_noise_std: float = 0.24,
) -> Tuple[hk.Transformed, hk.Params]:
    """Create a Haiku noise-prediction network: (obs, t) -> sigma in [min, max].

    The time input is processed through SinusoidalPosEmb -> Linear -> Mish ->
    Linear (matching the policy's time embedding), then concatenated with obs.
    """
    logvar_min = float(jnp.log(jnp.float32(min_noise_std ** 2)))
    logvar_max = float(jnp.log(jnp.float32(max_noise_std ** 2)))

    def _noise_fn(obs: jax.Array, t: jax.Array) -> jax.Array:
        te = sinusoidal_pos_emb(t, dim=time_dim, batch_shape=obs.shape[:-1])
        te = hk.Linear(time_dim * 2)(te)
        te = _mish(te)
        te = hk.Linear(time_dim)(te)
        x = jnp.concatenate([obs, te], axis=-1)
        for h in hidden_sizes:
            x = hk.Linear(h)(x)
            x = activation(x)
        raw = hk.Linear(act_dim)(x)
        logvar = logvar_min + (logvar_max - logvar_min) * (jnp.tanh(raw) + 1.0) / 2.0
        return jnp.exp(0.5 * logvar)

    noise_net = hk.without_apply_rng(hk.transform(_noise_fn))
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_t = jnp.zeros((1,))
    params = noise_net.init(key, dummy_obs, dummy_t)
    return noise_net, params


