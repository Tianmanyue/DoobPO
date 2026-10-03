"""
PPO-style flow policy network for on-policy baselines (FPO, ReinFlow, PiRL).

Unlike the SAC-style FlowNet (which bundles twin Q-networks, target networks,
and log_alpha), FlowPPONet pairs a flow velocity policy with a simple
state-value network V(s).  GAE advantages are computed externally by the
OnPolicySampler — the algorithm only needs to provide ``get_value(obs)``.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import (
    Activation, DACERPolicyNet, FPOPolicyNet, ResidualPolicyNet,
    ResidualValueNet, ValueNet,
)
from relax.utils.flow_matching import FlowMatching
from relax.utils.jax_utils import random_key_from_data


class FlowPPOParams(NamedTuple):
    policy: hk.Params
    value: hk.Params


@dataclass
class FlowPPONet:
    """Flow matching policy + V(s) value network (PPO-style)."""
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    value: Callable[[hk.Params, jax.Array], jax.Array]
    num_steps: int
    act_dim: int

    @property
    def flow(self) -> FlowMatching:
        return FlowMatching(self.num_steps)

    def get_action(
        self, key: jax.Array, policy_params: hk.Params, obs: jax.Array,
    ) -> jax.Array:
        """ODE sample → clip to [-1, 1].  No Q-based selection, no alpha noise."""
        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)
        act = self.flow.ode_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
        return act.clip(-1, 1)

    def get_deterministic_action(
        self, policy_params: hk.Params, obs: jax.Array,
    ) -> jax.Array:
        key = random_key_from_data(obs)
        return self.get_action(key, policy_params, obs)


def create_flow_ppo_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    policy_hidden_sizes: Sequence[int],
    value_hidden_sizes: Sequence[int],
    policy_activation: Activation = jax.nn.relu,
    value_activation: Activation = jax.nn.relu,
    *,
    num_steps: int = 10,
    use_residual_policy: bool = False,
    use_fpo_policy: bool = False,
    use_residual_value: bool = False,
    output_scale: float = 1.0,
) -> Tuple[FlowPPONet, FlowPPOParams]:
    """Create flow policy + V(s) for PPO-style on-policy training.

    Parameters
    ----------
    use_fpo_policy : if True, use FPOPolicyNet with 8-dim 2^k time embedding
        and the given ``output_scale`` (matching FPO's network architecture).
    use_residual_policy : if True, use ResidualPolicyNet (matching ReinFlow).
    output_scale : multiply network output by this factor (FPO uses 0.25).
    """
    if use_fpo_policy:
        policy = hk.without_apply_rng(hk.transform(
            lambda obs, act, t: FPOPolicyNet(
                policy_hidden_sizes, policy_activation,
                output_scale=output_scale,
            )(obs, act, t)
        ))
    elif use_residual_policy:
        hidden_dim = policy_hidden_sizes[0]
        num_blocks = max(1, (len(policy_hidden_sizes) - 1) // 2)
        policy = hk.without_apply_rng(hk.transform(
            lambda obs, act, t: ResidualPolicyNet(
                hidden_dim, num_blocks, policy_activation,
            )(obs, act, t)
        ))
    else:
        policy = hk.without_apply_rng(hk.transform(
            lambda obs, act, t: DACERPolicyNet(
                policy_hidden_sizes, policy_activation,
            )(obs, act, t)
        ))

    if use_residual_value:
        v_hidden = value_hidden_sizes[0]
        v_blocks = max(1, (len(value_hidden_sizes) - 1) // 2)
        value = hk.without_apply_rng(hk.transform(
            lambda obs: ResidualValueNet(v_hidden, v_blocks, value_activation)(obs)
        ))
    else:
        value = hk.without_apply_rng(hk.transform(
            lambda obs: ValueNet(value_hidden_sizes, value_activation)(obs)
        ))

    @jax.jit
    def init(key, obs, act):
        p_key, v_key = jax.random.split(key)
        p_params = policy.init(p_key, obs, act, jnp.float32(0.0))
        v_params = value.init(v_key, obs)
        return FlowPPOParams(p_params, v_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = FlowPPONet(
        policy=policy.apply,
        value=value.apply,
        num_steps=num_steps,
        act_dim=act_dim,
    )
    return net, params
