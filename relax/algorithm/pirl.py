"""
PiRL (π-RL / RLinf): RL Fine-Tuning via Analytic ODE → SDE Conversion.

Faithful reimplementation adapted for Gymnasium MuJoCo.

Key design choices matching the original RLinf algorithm:
  - V(s) + GAE for advantage estimation (standard PPO).
  - Single-step SDE: noise injected at exactly ONE randomly chosen
    denoising step per trajectory (all other steps are deterministic ODE).
  - Analytic noise: σ_raw(t) = noise_level · √((1−t) / t),
    σ_eff(t) = √dt · σ_raw(t).
  - Score-based drift correction for the SDE mean.
  - Log-prob computed at the SDE step only.
  - Value loss: Huber loss with value clipping (PPO-style).
  - Dual-clip PPO (clip_ratio_c=3.0).
  - Gradient clipping (max_grad_norm).
  - Optional noise annealing (linear from noise_start to noise_end).
  - Optional ignore_last (skip last denoising step for noise injection).
  - No alpha / temperature mechanism.

Hyperparameters:
  ppo_eps=0.2, n_epochs=4, gamma=0.99, gae_lambda=0.95,
  lr=1e-4, value_loss_coeff=0.5, ent_coef=0.0,
  reward_scale=1.0, max_grad_norm=1.0, huber_delta=10.0,
  value_clip=0.2
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
from relax.utils.typing_utils import Metric


class PiRLOptStates(NamedTuple):
    policy: optax.OptState
    value: optax.OptState


class PiRLTrainState(NamedTuple):
    params: FlowPPOParams
    opt_state: PiRLOptStates
    step: int


class PiRL(Algorithm):
    """On-policy flow PPO with analytic ODE→SDE noise injection (single step)."""

    def __init__(
        self,
        agent: FlowPPONet,
        params: FlowPPOParams,
        *,
        n_epochs: int = 4,
        noise_level: float = 0.5,
        min_sigma: float = 1e-3,
        ppo_eps: float = 0.2,
        gamma: float = 0.99,
        lr: float = 1e-4,
        value_lr: float = None,
        reward_scale: float = 1.0,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.0,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 1.0,
        huber_delta: float = 10.0,
        value_clip: float = 0.2,
        clip_ratio_c: float = 3.0,
        num_minibatches: int = 1,
        noise_anneal: bool = False,
        noise_start: float = 0.7,
        noise_end: float = 0.3,
        noise_anneal_steps: int = 400,
        ignore_last: bool = False,
        value_updates_per_batch: int = 10,
        policy_updates_per_batch: int = 1,
    ):
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_scale = reward_scale
        self.noise_level = noise_level
        self.min_sigma = min_sigma
        self._n_epochs = n_epochs
        self._num_minibatches = num_minibatches
        self._epoch_counter = 0

        self._noise_anneal = noise_anneal
        self._noise_start = noise_start
        self._noise_end = noise_end
        self._noise_anneal_steps = noise_anneal_steps
        self._ignore_last = ignore_last
        self._global_step = 0

        self._traj_buffer = []
        self._logprob_buffer = []
        self._sde_step_buffer = []
        self._num_envs = 0  # set in get_action

        agent_ref = self.agent

        _vlr = value_lr if value_lr is not None else lr
        if max_grad_norm > 0:
            policy_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))
            value_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(_vlr))
        else:
            policy_optim = optax.adam(lr)
            value_optim = optax.adam(_vlr)
        self.policy_optim = policy_optim
        self.value_optim = value_optim

        self.state = PiRLTrainState(
            params=params,
            opt_state=PiRLOptStates(
                policy=policy_optim.init(params.policy),
                value=value_optim.init(params.value),
            ),
            step=jnp.int32(0),
        )

        _min_sigma = min_sigma
        _dt = 1.0 / agent.num_steps
        _max_sde_step = agent.num_steps - 1 if ignore_last else agent.num_steps

        def analytic_sigma_fn_dynamic(noise_lvl, t, x):
            t_safe = jnp.maximum(t, _dt)
            sigma_raw = noise_lvl * jnp.sqrt((1.0 - t_safe) / t_safe)
            sigma_eff = jnp.sqrt(_dt) * sigma_raw
            return jnp.maximum(sigma_eff, _min_sigma)

        @jax.jit
        def sde_get_action(key, policy_params, obs, noise_lvl):
            B = obs.shape[0]

            def model_fn(t, x):
                return agent_ref.policy(policy_params, obs, x, jnp.full((B,), t))

            def sigma_fn(t, x):
                return analytic_sigma_fn_dynamic(noise_lvl, t, x)

            x_final, trajectory, logprob, sde_step = agent_ref.flow.sde_sample_single_step(
                key, model_fn, sigma_fn, (B, agent_ref.act_dim),
                use_drift_correction=True,
                max_sde_step=_max_sde_step,
            )
            action = x_final.clip(-1, 1)
            return action, trajectory, logprob, sde_step

        self._sde_get_action = sde_get_action

        _ppo_eps = ppo_eps
        _ent_coef = ent_coef
        _vloss_coef = value_loss_coeff
        _num_steps = agent.num_steps
        _huber_delta = huber_delta
        _value_clip = value_clip
        _clip_c = clip_ratio_c
        _n_val = value_updates_per_batch
        _n_p = policy_updates_per_batch

        @jax.jit
        def stateless_update(
            key: jax.Array,
            state: PiRLTrainState,
            data: GAEExperience,
            trajectories: jax.Array,
            old_logprobs: jax.Array,
            sde_step_indices: jax.Array,
            old_values: jax.Array,
            noise_lvl: jax.Array,
        ) -> Tuple[PiRLTrainState, Metric]:

            obs = data.obs
            action = data.action
            adv = data.adv
            ret = data.ret
            pp, vp = state.params
            p_os, v_os = state.opt_state
            step = state.step
            B = obs.shape[0]

            adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv_sg = jax.lax.stop_gradient(adv_norm)

            def sigma_fn(t, x):
                return analytic_sigma_fn_dynamic(noise_lvl, t, x)

            # ---- Value updates: _n_val times ----
            def value_loss_fn(vp_):
                v_pred = agent_ref.value(vp_, obs)
                v_clipped = old_values + jnp.clip(
                    v_pred - old_values, -_value_clip, _value_clip
                )
                loss_orig = _huber_loss(ret - v_pred, _huber_delta)
                loss_clipped = _huber_loss(ret - v_clipped, _huber_delta)
                return _vloss_coef * jnp.mean(jnp.maximum(loss_orig, loss_clipped))

            v_loss = 0.0
            for _ in range(_n_val):
                v_loss, v_grads = jax.value_and_grad(value_loss_fn)(vp)
                v_updates, v_os = self.value_optim.update(v_grads, v_os)
                vp = optax.apply_updates(vp, v_updates)

            # ---- Policy update: _n_p times ----
            traj_T = jnp.moveaxis(trajectories, 1, 0)

            def policy_loss_fn(pp_):
                def model_fn(t_batch, x):
                    return agent_ref.policy(pp_, obs, x, t_batch)

                new_lp = agent_ref.flow.compute_single_step_logprob(
                    model_fn, sigma_fn, traj_T,
                    sde_step_indices,
                    use_drift_correction=True,
                )
                ratio = jnp.exp(new_lp - old_logprobs)
                surr1 = ratio * adv_sg
                surr2 = jnp.clip(ratio, 1.0 - _ppo_eps, 1.0 + _ppo_eps) * adv_sg
                clipped_obj = jnp.minimum(surr1, surr2)
                dual_clip_obj = jnp.where(
                    adv_sg < 0,
                    jnp.maximum(clipped_obj, _clip_c * adv_sg),
                    clipped_obj,
                )
                ppo_loss = -dual_clip_obj.mean()

                entropy_bonus = _ent_coef * new_lp.mean() / _num_steps

                return ppo_loss + entropy_bonus, (ratio, new_lp)

            (p_loss, (ratio_val, new_lp_val)), p_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True)(pp)

            p_updates, new_p_os = self.policy_optim.update(p_grads, p_os)
            pp = optax.apply_updates(pp, p_updates)

            new_state = PiRLTrainState(
                params=FlowPPOParams(pp, vp),
                opt_state=PiRLOptStates(p_os, v_os),
                step=step + 1,
            )
            info = {
                "value_loss": v_loss,
                "policy_loss": p_loss,
                "ratio_mean": jnp.mean(ratio_val),
                "ratio_std": jnp.std(ratio_val),
                "ratio_max": jnp.max(ratio_val),
                "new_logprob_mean": jnp.mean(new_lp_val),
                "old_logprob_mean": jnp.mean(old_logprobs),
                "advantage_mean": jnp.mean(adv),
                "noise_level": noise_lvl,
            }
            return new_state, info

        self._implement_common_behavior(
            stateless_update,
            self.agent.get_action,
            self.agent.get_deterministic_action,
        )

    # ----------------------------------------------------------------
    # Param accessors
    # ----------------------------------------------------------------
    def get_policy_params(self):
        return self.state.params.policy

    def get_policy_params_to_save(self):
        return self.state.params.policy

    def get_value_params(self):
        return self.state.params.value

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.agent.value(self.state.params.value, obs))

    # ----------------------------------------------------------------
    # Noise annealing helper
    # ----------------------------------------------------------------
    def _current_noise_level(self) -> float:
        if not self._noise_anneal:
            return self.noise_level
        progress = min(self._global_step, self._noise_anneal_steps) / self._noise_anneal_steps
        return self._noise_start + (self._noise_end - self._noise_start) * progress

    # ----------------------------------------------------------------
    # Rollout: single-step SDE sampling + trajectory storage
    # ----------------------------------------------------------------
    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        p = self.state.params
        noise_lvl = jnp.float32(self._current_noise_level())
        action, trajectory, logprob, sde_step = self._sde_get_action(
            key, p.policy, obs, noise_lvl,
        )
        self._num_envs = obs.shape[0]
        self._traj_buffer.append(np.asarray(jnp.moveaxis(trajectory, 0, 1)))
        self._logprob_buffer.append(np.asarray(logprob))
        self._sde_step_buffer.append(int(sde_step))
        return np.asarray(action)

    def _consolidate_trajectories(self):
        K_plus_1 = self._traj_buffer[0].shape[1]
        act_dim = self._traj_buffer[0].shape[2]
        num_envs = self._num_envs if self._num_envs > 0 else self._traj_buffer[0].shape[0]
        L = len(self._traj_buffer)
        trajs = np.stack(self._traj_buffer, axis=1).reshape(-1, K_plus_1, act_dim)
        logps = np.stack(self._logprob_buffer, axis=1).reshape(-1)
        sde_arr = np.array(self._sde_step_buffer, dtype=np.int32)  # (L,)
        sde_per_sample = np.tile(sde_arr, num_envs)  # (num_envs * L,)
        self._consolidated_trajs = jnp.array(trajs)
        self._consolidated_logps = jnp.array(logps)
        self._consolidated_sde_steps = jnp.array(sde_per_sample)
        self._traj_buffer.clear()
        self._logprob_buffer.clear()
        self._sde_step_buffer.clear()

    # ----------------------------------------------------------------
    # Update
    # ----------------------------------------------------------------
    def update(self, key: jax.Array, data: GAEExperience) -> Tuple[dict, dict]:
        if self._epoch_counter % self._n_epochs == 0:
            self._consolidate_trajectories()
            self._old_values = jnp.array(
                self.agent.value(self.state.params.value, data.obs)
            )
        self._epoch_counter += 1

        noise_lvl = jnp.float32(self._current_noise_level())
        B = data.obs.shape[0]
        M = self._num_minibatches

        if M <= 1 or B < M:
            self.state, info = self._update(
                key, self.state, data,
                self._consolidated_trajs,
                self._consolidated_logps,
                self._consolidated_sde_steps,
                self._old_values,
                noise_lvl,
            )
        else:
            mb_size = B // M
            perm = np.random.permutation(B)
            info = {}
            for m in range(M):
                idx = perm[m * mb_size:(m + 1) * mb_size]
                mb_data = GAEExperience(
                    data.obs[idx], data.action[idx], data.reward[idx],
                    data.done[idx], data.next_obs[idx],
                    data.ret[idx], data.adv[idx], data.truncated[idx],
                )
                mb_trajs = self._consolidated_trajs[idx]
                mb_logps = self._consolidated_logps[idx]
                mb_sde_steps = self._consolidated_sde_steps[idx]
                mb_old_vals = self._old_values[idx]
                mb_key = jax.random.fold_in(key, m)
                self.state, info = self._update(
                    mb_key, self.state, mb_data,
                    mb_trajs, mb_logps,
                    mb_sde_steps,
                    mb_old_vals,
                    noise_lvl,
                )

        self._global_step += 1

        return (
            {k: float(v) for k, v in info.items() if not k.startswith("hist")},
            {k: v for k, v in info.items() if k.startswith("hist")},
        )

    def warmup(self, data: GAEExperience) -> None:
        key = jax.random.key(0)
        B = data.obs.shape[0]
        K = self.agent.num_steps

        dummy_trajs = jnp.zeros((B, K + 1, self.agent.act_dim))
        dummy_logps = jnp.zeros((B,))
        dummy_sde_steps = jnp.int32(0)
        dummy_old_values = jnp.zeros((B,))
        dummy_noise_lvl = jnp.float32(self._current_noise_level())

        self._update(
            key, self.state, data,
            dummy_trajs, dummy_logps,
            dummy_sde_steps, dummy_old_values,
            dummy_noise_lvl,
        )
        pp = self.get_policy_params()
        obs = data.obs[0]
        self._get_action(key, pp, obs)
        self._get_deterministic_action(pp, obs)
        self._sde_get_action(key, self.state.params.policy, data.obs[:2], dummy_noise_lvl)

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
