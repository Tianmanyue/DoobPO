"""
Network factory for DoobPO-Flow.

Combines a FlowPPONet (flow matching policy + V(s) value network) with
a lightweight ratio network r_β: (obs, act) → ℝ₊.

Uses the same FlowPPONet as the baselines (FPO, ReinFlow, PiRL), ensuring
a unified network architecture across all flow methods.
"""

from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, QNet
from relax.network.flow_ppo import FlowPPONet, FlowPPOParams, create_flow_ppo_net
from relax.network.doobpo import create_ratio_net, create_ratio_net_resnet


def create_doobpo_flow_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    policy_hidden_sizes: Sequence[int],
    value_hidden_sizes: Sequence[int],
    policy_activation: Activation = jax.nn.relu,
    value_activation: Activation = jax.nn.relu,
    *,
    num_steps: int = 10,
    ratio_hidden_sizes: Optional[Sequence[int]] = None,
    ratio_net_type: str = "mlp",
    # ~same param count as MLP 256×3: one ResBlock = two 256→256 layers (see doobpo.py doc).
    ratio_resnet_hidden_dim: int = 256,
    ratio_resnet_num_blocks: int = 1,
    include_twin_q: bool = True,
) -> Tuple[
    FlowPPONet,
    FlowPPOParams,
    hk.Transformed,
    hk.Params,
    Optional[Callable[[hk.Params, jax.Array, jax.Array], jax.Array]],
    Optional[hk.Params],
    Optional[hk.Params],
    Optional[hk.Params],
    Optional[hk.Params],
]:
    """Create all networks for DoobPO-Flow.

    Returns
    -------
    agent, flow_ppo_params, ratio_net, ratio_params
    If ``include_twin_q`` (default True): also
    q_apply, q1_params, q2_params, target_q1_params, target_q2_params
    for off-policy TD3/SAC-style twin critics. Otherwise four Nones.
    """
    ratio_key = jax.random.fold_in(key, 0x1F6F0)

    agent, params = create_flow_ppo_net(
        key, obs_dim, act_dim,
        policy_hidden_sizes, value_hidden_sizes,
        policy_activation, value_activation,
        num_steps=num_steps,
    )

    if ratio_net_type == "resnet":
        ratio_net, ratio_params = create_ratio_net_resnet(
            ratio_key,
            obs_dim,
            act_dim,
            ratio_resnet_hidden_dim,
            ratio_resnet_num_blocks,
            policy_activation,
        )
    else:
        if ratio_hidden_sizes is None:
            ratio_hidden_sizes = list(value_hidden_sizes)
        ratio_net, ratio_params = create_ratio_net(
            ratio_key, obs_dim, act_dim, ratio_hidden_sizes, policy_activation,
        )

    if not include_twin_q:
        return agent, params, ratio_net, ratio_params, None, None, None, None, None

    q_key = jax.random.fold_in(key, 0x0F1CE)
    q = hk.without_apply_rng(hk.transform(
        lambda obs, act: QNet(value_hidden_sizes, policy_activation)(obs, act)
    ))
    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    q1_key, q2_key = jax.random.split(q_key)
    q1_params = q.init(q1_key, sample_obs, sample_act)
    q2_params = q.init(q2_key, sample_obs, sample_act)
    return (
        agent,
        params,
        ratio_net,
        ratio_params,
        q.apply,
        q1_params,
        q2_params,
        q1_params,
        q2_params,
    )
