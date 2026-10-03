"""
Network factory for LFGPO.

Re-uses the existing Diffv2Net (diffusion policy + Q-networks) and adds a
lightweight ratio network r_β: (obs, act) → ℝ₊.

The ratio network outputs exp(clip(f_θ(obs, act))) which is:
  - Always positive, with logits clipped before exp to avoid late-training blow-ups
  - Equal to 1 when f_θ = 0 (achieved with standard zero-bias initialisation)
This ensures the ratio starts near the "no update" reference of 1.
"""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation
from relax.network.diffv2 import Diffv2Net, Diffv2Params, create_diffv2_net

LOGIT_CLIP_MIN = -7.5
LOGIT_CLIP_MAX = 7.5


def create_ratio_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
):
    """Create a ratio network r_β(obs, act) → ℝ₊ using exp(MLP(obs, act))."""

    def _ratio_fn(obs: jax.Array, act: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, act], axis=-1)
        for h in hidden_sizes:
            x = hk.Linear(h)(x)
            x = activation(x)
        # exp ensures r_β > 0; zero-init bias ⟹ exp(0) = 1 at initialisation.
        # Clip logits so exp() cannot blow up late in training.
        logit = hk.Linear(1, w_init=jnp.zeros, b_init=jnp.zeros)(x)[..., 0]
        logit = jnp.clip(logit, LOGIT_CLIP_MIN, LOGIT_CLIP_MAX)
        return jnp.exp(logit)  # shape (...,), in [exp(-7.5), exp(7.5)]

    ratio_net = hk.without_apply_rng(hk.transform(_ratio_fn))

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    ratio_params = ratio_net.init(key, sample_obs, sample_act)

    return ratio_net, ratio_params


def create_ratio_net_resnet(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_dim: int,
    num_blocks: int,
    activation: Activation = jax.nn.relu,
):
    """Ratio network with residual trunk (same block pattern as ``ResidualPolicyNet``).

    ``hidden_dim`` is the width of every trunk layer; ``num_blocks`` is how many
    skip blocks (each: Linear→act→Linear→act + residual). With ``hidden_dim=256``,
    ``num_blocks=1`` matches the **parameter count** of an MLP ``[256,256,256]``
    ratio (stem + two 256→256 maps + head); deeper blocks add ~2×256² params each.
    """

    def _ratio_fn(obs: jax.Array, act: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, act], axis=-1)
        x = hk.Linear(hidden_dim)(x)
        for _ in range(num_blocks):
            residual = x
            x = hk.Linear(hidden_dim)(activation(x))
            x = hk.Linear(hidden_dim)(activation(x))
            x = x + residual
        logit = hk.Linear(1, w_init=jnp.zeros, b_init=jnp.zeros)(x)[..., 0]
        logit = jnp.clip(logit, LOGIT_CLIP_MIN, LOGIT_CLIP_MAX)
        return jnp.exp(logit)

    ratio_net = hk.without_apply_rng(hk.transform(_ratio_fn))
    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    ratio_params = ratio_net.init(key, sample_obs, sample_act)
    return ratio_net, ratio_params


def create_lfgpo_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    *,
    num_timesteps: int = 20,
    num_particles: int = 4,
    noise_scale: float = 0.05,
    target_entropy_scale: float = 0.9,
    beta_schedule_scale: float = 0.3,
    ratio_hidden_sizes: Optional[Sequence[int]] = None,
) -> Tuple[Diffv2Net, Diffv2Params, hk.Transformed, hk.Params]:
    """
    Create all networks needed for LFGPO.

    Returns
    -------
    agent          : Diffv2Net (diffusion policy + double-Q)
    params         : Diffv2Params (initial parameters)
    ratio_net      : hk.Transformed for the ratio network
    ratio_params   : initial ratio-network parameters
    """
    diffv2_key, ratio_key = jax.random.split(key)

    agent, params = create_diffv2_net(
        diffv2_key,
        obs_dim,
        act_dim,
        hidden_sizes,
        diffusion_hidden_sizes,
        activation,
        num_timesteps=num_timesteps,
        num_particles=num_particles,
        noise_scale=noise_scale,
        target_entropy_scale=target_entropy_scale,
        beta_schedule_scale=beta_schedule_scale,
    )

    if ratio_hidden_sizes is None:
        ratio_hidden_sizes = hidden_sizes

    ratio_net, ratio_params = create_ratio_net(
        ratio_key, obs_dim, act_dim, ratio_hidden_sizes, activation
    )

    return agent, params, ratio_net, ratio_params
