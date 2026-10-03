"""
FPO-Flow: Flow Policy Optimization for flow matching policies.

Faithful reimplementation of the original FPO codebase
(github.com/akanazawa/fpo) adapted for Gymnasium MuJoCo.

Key design choices matching the original Playground implementation:
  - V(s) + GAE for advantage estimation (standard PPO, no Q-networks).
  - The implicit policy ratio ρ = exp(mean(L_old) − mean(L_new)) uses
    no intermediate clipping (``average_losses_before_exp=True`` mode).
  - Fixed advantages: GAE advantages are computed once per rollout and
    kept frozen across all PPO epochs (matching original FPO).
  - Standard PPO clipped-surrogate objective.
  - Value loss: MSE(V(s), GAE returns) with value_loss_coeff.
  - FPOPolicyNet: 8-dim 2^k time embedding, output_scale=0.25.
  - Observation normalization via Welford RunningStats.
  - Discrete time sampling from the ODE integration grid for consistency.
  - CFM loss weighted by t² (our convention, equivalent to (1−t)² in the
    original reverse-time convention) — "u_but_supervise_as_eps" mode.
  - Advantages recomputed from current V(s) inside each update step.
  - Actions passed to the environment via tanh; raw (pre-tanh) actions
    used for CFM loss, matching the original.
  - Minibatch support: data shuffled and split per epoch.
  - Constant learning rate (no schedule) shared by policy and value.

Hyperparameters default to the FPO MuJoCo Playground values:
  ppo_eps=0.05, gamma=0.995, n_epochs=16, n_cfm_samples=8,
  lr=3e-4, gae_lambda=0.95, reward_scale=10.0, value_loss_coeff=0.25,
  output_scale=0.25, num_minibatches=1
"""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.flow_ppo import FlowPPONet, FlowPPOParams
from relax.utils.experience import Experience, GAEExperience
from relax.utils.obs_normalize import RunningObsStats
from relax.utils.typing_utils import Metric


class FPOFlowOptStates(NamedTuple):
    policy: optax.OptState
    value: optax.OptState


class FPOFlowTrainState(NamedTuple):
    params: FlowPPOParams
    opt_state: FPOFlowOptStates
    step: int


class FPOFlow(Algorithm):
    """On-policy flow PPO with implicit CFM-loss ratio."""

    def __init__(
        self,
        agent: FlowPPONet,
        params: FlowPPOParams,
        *,
        n_epochs: int = 16,
        n_cfm_samples: int = 8,
        ppo_eps: float = 0.05,
        gamma: float = 0.995,
        lr: float = 3e-4,
        value_lr: float = 1e-4,
        reward_scale: float = 10.0,
        gae_lambda: float = 0.95,
        value_loss_coeff: float = 0.25,
        normalize_obs: bool = True,
        discretize_t: bool = True,
        num_minibatches: int = 1,
        obs_dim: int = 0,
        max_grad_norm: float = 1.0,
        huber_delta: float = 10.0,
        value_clip: float = 0.2,
        cfm_loss_scale: float = 1.0,
        cfm_diff_clip: float = 3.0,
        target_kl: float = 0.0,
        value_updates_per_batch: int = 10,
        policy_updates_per_batch: int = 1,
    ):
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_scale = reward_scale
        self._n_epochs = n_epochs
        self._n_cfm_samples = n_cfm_samples
        self._num_minibatches = num_minibatches
        self._epoch_counter = 0
        self._target_kl = target_kl
        self._kl_early_stop = False

        self._eps_buffer: list = []
        self._t_buffer: list = []
        self._init_loss_buffer: list = []
        self._raw_action_buffer: list = []
        self._consolidated_eps = None
        self._consolidated_t = None
        self._consolidated_init_loss = None
        self._consolidated_raw_actions = None

        self._do_obs_norm = normalize_obs
        self._obs_stats = RunningObsStats(obs_dim) if normalize_obs and obs_dim > 0 else None

        agent_ref = self.agent
        N = n_cfm_samples
        _discretize_t = discretize_t
        _num_steps = agent.num_steps

        _t_grid = jnp.linspace(0.0, 1.0 - 1.0 / _num_steps, _num_steps)

        @jax.jit
        def raw_ode_sample(key, policy_params, obs):
            """ODE sample returning raw (pre-tanh) action."""
            B = obs.shape[0]

            def model_fn(t, x):
                return agent_ref.policy(policy_params, obs, x, t)

            return agent_ref.flow.ode_sample(key, model_fn, (B, agent_ref.act_dim))

        self._raw_ode_sample = raw_ode_sample

        @jax.jit
        def compute_cfm_info(key, policy_params, obs, action_raw):
            """Compute CFM info using raw (pre-tanh) actions.

            Returns eps, t_vals, and t²-weighted init_loss.
            """
            B = obs.shape[0]
            act_dim_local = action_raw.shape[1]
            obs_dim_local = obs.shape[1]
            eps_key, t_key = jax.random.split(key)

            eps = jax.random.normal(eps_key, (B, N, act_dim_local))

            if _discretize_t:
                t_idx = jax.random.randint(t_key, (B, N), 0, _num_steps)
                t_vals = _t_grid[t_idx]
            else:
                t_vals = jax.random.uniform(t_key, (B, N))

            BN = B * N
            obs_flat = jnp.broadcast_to(
                obs[:, None, :], (B, N, obs_dim_local)
            ).reshape(BN, obs_dim_local)
            act_exp = action_raw[:, None, :]
            t_3d = t_vals[:, :, None]
            x_t = (1.0 - t_3d) * eps + t_3d * act_exp
            vel_target = act_exp - eps

            x_t_flat = x_t.reshape(BN, act_dim_local)
            t_flat = t_vals.reshape(BN)

            vel_pred = agent_ref.policy(
                policy_params, obs_flat, x_t_flat, t_flat
            ).reshape(B, N, act_dim_local)

            mse = jnp.mean((vel_pred - vel_target) ** 2, axis=-1)
            init_loss = t_vals ** 2 * mse
            return eps, t_vals, init_loss

        self._compute_cfm_info = compute_cfm_info

        if max_grad_norm > 0:
            self.policy_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))
            self.value_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(value_lr))
        else:
            self.policy_optim = optax.adam(lr)
            self.value_optim = optax.adam(value_lr)

        self.state = FPOFlowTrainState(
            params=params,
            opt_state=FPOFlowOptStates(
                policy=self.policy_optim.init(params.policy),
                value=self.value_optim.init(params.value),
            ),
            step=jnp.int32(0),
        )

        _ppo_eps = ppo_eps
        _vloss_coef = value_loss_coeff
        _huber_delta = huber_delta
        _value_clip = value_clip
        _cfm_scale = cfm_loss_scale
        _cfm_clip = cfm_diff_clip
        _n_val = value_updates_per_batch
        _n_p = policy_updates_per_batch

        @jax.jit
        def stateless_update(
            key: jax.Array,
            state: FPOFlowTrainState,
            data: GAEExperience,
            stored_eps: jax.Array,
            stored_t: jax.Array,
            stored_init_loss: jax.Array,
            raw_actions: jax.Array,
            old_values: jax.Array,
        ) -> Tuple[FPOFlowTrainState, Metric]:
            obs = data.obs
            ret = data.ret
            trunc_mask = 1.0 - data.truncated.astype(jnp.float32)
            pp, vp = state.params
            p_os, v_os = state.opt_state
            step = state.step

            B = obs.shape[0]
            obs_dim = obs.shape[1]
            act_dim = raw_actions.shape[1]

            adv_norm = (data.adv - data.adv.mean()) / (data.adv.std() + 1e-8)
            adv_sg = jax.lax.stop_gradient(adv_norm)

            BN = B * N
            t_3d = stored_t[:, :, None]
            act_exp = raw_actions[:, None, :]
            x_t = (1.0 - t_3d) * stored_eps + t_3d * act_exp
            vel_target = act_exp - stored_eps

            obs_flat = jnp.broadcast_to(
                obs[:, None, :], (B, N, obs_dim)).reshape(BN, obs_dim)
            x_t_flat = x_t.reshape(BN, act_dim)
            t_flat = stored_t.reshape(BN)

            def policy_loss_fn(pp_):
                new_vel = agent_ref.policy(
                    pp_, obs_flat, x_t_flat, t_flat,
                ).reshape(B, N, act_dim)
                new_mse = jnp.mean((new_vel - vel_target) ** 2, axis=-1)
                new_loss = stored_t ** 2 * new_mse

                cfm_diff = (
                    stored_init_loss.mean(axis=1)
                    - new_loss.mean(axis=1)
                ) / _cfm_scale
                cfm_diff = jnp.clip(cfm_diff, -_cfm_clip, _cfm_clip)
                rho = jnp.exp(cfm_diff)
                surr1 = rho * adv_sg
                surr2 = jnp.clip(
                    rho, 1.0 - _ppo_eps, 1.0 + _ppo_eps
                ) * adv_sg
                return -jnp.minimum(surr1, surr2).mean(), rho

            def value_loss_fn(vp_):
                v_pred = agent_ref.value(vp_, obs)
                v_clipped = old_values + jnp.clip(
                    v_pred - old_values, -_value_clip, _value_clip
                )
                loss_orig = _huber_loss(ret - v_pred, _huber_delta)
                loss_clipped = _huber_loss(ret - v_clipped, _huber_delta)
                return _vloss_coef * jnp.mean(jnp.maximum(loss_orig, loss_clipped) * trunc_mask)

            # ---- Value updates: _n_val times ----
            v_loss = 0.0
            for _ in range(_n_val):
                v_loss, v_grads = jax.value_and_grad(value_loss_fn)(vp)
                v_updates, v_os = self.value_optim.update(v_grads, v_os)
                vp = optax.apply_updates(vp, v_updates)

            # ---- Policy update: _n_p times ----
            p_loss = 0.0
            rho_val = jnp.ones((B,))
            for _ in range(_n_p):
                (p_loss, rho_val), p_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(pp)
                p_updates, p_os = self.policy_optim.update(p_grads, p_os)
                pp = optax.apply_updates(pp, p_updates)

            new_state = FPOFlowTrainState(
                params=FlowPPOParams(pp, vp),
                opt_state=FPOFlowOptStates(p_os, v_os),
                step=step + 1,
            )
            approx_kl = jnp.mean((rho_val - 1.0) - jnp.log(jnp.maximum(rho_val, 1e-8)))
            info = {
                "policy_loss": p_loss,
                "value_loss": v_loss,
                "rho_mean": jnp.mean(rho_val),
                "rho_std": jnp.std(rho_val),
                "rho_max": jnp.max(rho_val),
                "advantage_mean": jnp.mean(data.adv),
                "approx_kl": approx_kl,
            }
            return new_state, info

        self._implement_common_behavior(
            stateless_update,
            self.agent.get_action,
            self.agent.get_deterministic_action,
        )

    # ---- Observation normalization helpers ----

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_stats is not None:
            return self._obs_stats.normalize(obs)
        return obs

    # ---- Rollout buffer management ----

    def _consolidate_cfm_data(self):
        eps = np.stack(self._eps_buffer, axis=1).reshape(
            -1, self._n_cfm_samples, self._eps_buffer[0].shape[2])
        t = np.stack(self._t_buffer, axis=1).reshape(-1, self._n_cfm_samples)
        init_loss = np.stack(self._init_loss_buffer, axis=1).reshape(
            -1, self._n_cfm_samples)
        raw_act = np.stack(self._raw_action_buffer, axis=1).reshape(
            -1, self._raw_action_buffer[0].shape[-1])
        self._consolidated_eps = jnp.array(eps)
        self._consolidated_t = jnp.array(t)
        self._consolidated_init_loss = jnp.array(init_loss)
        self._consolidated_raw_actions = jnp.array(raw_act)
        self._eps_buffer.clear()
        self._t_buffer.clear()
        self._init_loss_buffer.clear()
        self._raw_action_buffer.clear()

    # ---- Param accessors ----

    def get_policy_params(self):
        return self.state.params.policy

    def get_policy_params_to_save(self):
        return self.state.params.policy

    def get_value_params(self):
        return self.state.params.value

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        obs_n = self._norm_obs(obs)
        return np.asarray(self.agent.value(self.state.params.value, obs_n))

    # ---- Rollout: generate action + store CFM info ----

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        obs_n = self._norm_obs(obs)
        action_key, cfm_key = jax.random.split(key)
        pp = self.get_policy_params()

        action_raw = self._raw_ode_sample(action_key, pp, obs_n)

        eps, t_vals, init_loss = self._compute_cfm_info(
            cfm_key, pp, obs_n, action_raw,
        )

        self._eps_buffer.append(np.asarray(eps))
        self._t_buffer.append(np.asarray(t_vals))
        self._init_loss_buffer.append(np.asarray(init_loss))
        self._raw_action_buffer.append(np.asarray(action_raw))

        action_env = np.asarray(jnp.tanh(action_raw))
        return action_env

    # ---- Training: consolidate + update with minibatches ----

    def update(self, key: jax.Array, data: GAEExperience) -> Tuple[dict, dict]:
        if self._epoch_counter % self._n_epochs == 0:
            self._consolidate_cfm_data()
            if self._obs_stats is not None:
                self._obs_stats.update(data.obs)
            obs_for_val = self._norm_obs(data.obs) if self._obs_stats is not None else data.obs
            self._old_values = jnp.array(
                self.agent.value(self.state.params.value, obs_for_val)
            )
            self._kl_early_stop = False

        self._epoch_counter += 1

        if self._kl_early_stop:
            return ({}, {})

        obs_n = self._norm_obs(data.obs) if self._obs_stats is not None else data.obs
        data_n = GAEExperience(obs_n, data.action, data.reward, data.done,
                               data.next_obs, data.ret, data.adv, data.truncated)

        B = data_n.obs.shape[0]
        M = self._num_minibatches

        if M <= 1 or B < M:
            self.state, info = self._update(
                key, self.state, data_n,
                self._consolidated_eps,
                self._consolidated_t,
                self._consolidated_init_loss,
                self._consolidated_raw_actions,
                self._old_values,
            )
            if self._target_kl > 0 and float(info.get("approx_kl", 0)) > self._target_kl:
                self._kl_early_stop = True
        else:
            mb_size = B // M
            perm = np.random.permutation(B)
            info = None
            for m in range(M):
                if self._kl_early_stop:
                    break
                idx = perm[m * mb_size:(m + 1) * mb_size]
                mb_data = GAEExperience(
                    data_n.obs[idx], data_n.action[idx], data_n.reward[idx],
                    data_n.done[idx], data_n.next_obs[idx],
                    data_n.ret[idx], data_n.adv[idx], data_n.truncated[idx],
                )
                mb_eps = self._consolidated_eps[idx]
                mb_t = self._consolidated_t[idx]
                mb_init = self._consolidated_init_loss[idx]
                mb_raw = self._consolidated_raw_actions[idx]
                mb_old_vals = self._old_values[idx]
                mb_key = jax.random.fold_in(key, m)
                self.state, info = self._update(
                    mb_key, self.state, mb_data,
                    mb_eps, mb_t, mb_init, mb_raw, mb_old_vals,
                )
                if self._target_kl > 0 and float(info.get("approx_kl", 0)) > self._target_kl:
                    self._kl_early_stop = True

        return (
            {k: float(v) for k, v in info.items() if not k.startswith("hist")},
            {k: v for k, v in info.items() if k.startswith("hist")},
        )

    def warmup(self, data: GAEExperience) -> None:
        key = jax.random.key(0)
        B = data.obs.shape[0]
        act_dim = data.action.shape[1]
        N = self._n_cfm_samples

        dummy_eps = jnp.zeros((B, N, act_dim))
        dummy_t = jnp.zeros((B, N))
        dummy_init_loss = jnp.zeros((B, N))
        dummy_raw = jnp.zeros((B, act_dim))
        dummy_old_values = jnp.zeros((B,))

        self._update(
            key, self.state, data,
            dummy_eps, dummy_t, dummy_init_loss, dummy_raw,
            dummy_old_values,
        )
        pp = self.get_policy_params()
        obs = data.obs[0]
        self._get_action(key, pp, obs)
        self._get_deterministic_action(pp, obs)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)


def _huber_loss(error: jax.Array, delta: float) -> jax.Array:
    abs_error = jnp.abs(error)
    return jnp.where(
        abs_error < delta,
        0.5 * error ** 2,
        delta * (abs_error - 0.5 * delta),
    )
