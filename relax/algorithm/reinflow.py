"""
ReinFlow: Online RL Fine-Tuning of Flow Policies via Noise Injection.

Faithful reimplementation of the original ReinFlow codebase
(github.com/ReinFlow/ReinFlow) adapted for Gymnasium MuJoCo.

Key design choices matching the original:
  - V(s) + GAE for advantage estimation (standard PPO).
  - SDE sampling with learned per-step noise σ(s,t) from ExploreNoiseNet.
  - Log-prob normalized by (num_steps+1) * act_dim before clamping.
  - Intermediate x_t clipped to ±1 between SDE steps.
  - Noise decay schedule: max_std decays via cosine from 0.24 to ~0.198.
  - Critic warmup: only update V(s) for the first ``n_critic_warmup``
    training iterations before starting policy updates.
  - AdamW for critic (weight_decay=1e-4 matching original).
  - Periodic cosine restart LR schedule (cycle=100 iters, warmup=10).
  - Minibatch: data shuffled and split per epoch.
  - KL early stopping: Schulman approx KL, target_kl=1.0.
  - Running reward normalization.
  - Entropy bonus in the policy loss.
  - Value loss: MSE(V(s), GAE returns) with value_loss_coeff.

Hyperparameters default to the ReinFlow HalfCheetah config:
  ppo_eps=0.01, n_epochs=5, gamma=0.99, gae_lambda=0.95,
  actor_lr=3e-5, critic_lr=3e-4, noise_lr=3e-5,
  ent_coef=0.03, logprob_min/max=[-1,1], randn_clip=3,
  reward_scale=1.0, value_loss_coeff=0.5
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
from relax.utils.reward_scaling import RunningRewardScaler
from relax.utils.typing_utils import Metric


class ReinFlowParams(NamedTuple):
    policy: hk.Params
    value: hk.Params
    noise: hk.Params


class ReinFlowOptStates(NamedTuple):
    policy: optax.OptState
    value: optax.OptState
    noise: optax.OptState


class ReinFlowTrainState(NamedTuple):
    params: ReinFlowParams
    opt_state: ReinFlowOptStates
    step: int


def _periodic_cosine_lr(max_lr: float, min_lr: float,
                        cycle_steps: int, warmup_steps: int = 0):
    """Periodic cosine annealing with warm restarts (CosineAnnealingWarmRestarts).

    ``cycle_steps`` and ``warmup_steps`` are measured in **optimizer-update**
    units (i.e. already scaled by ``n_epochs * num_minibatches``).
    """
    _cycle = max(cycle_steps, 1)
    _warmup = max(warmup_steps, 0)
    _cosine_len = max(_cycle - _warmup, 1)

    def schedule_fn(count):
        pos = count % _cycle
        warmup_frac = jnp.minimum(pos / jnp.maximum(_warmup, 1), 1.0)
        cosine_pos = jnp.maximum(pos - _warmup, 0)
        cosine_frac = cosine_pos / _cosine_len
        cosine_val = min_lr + (max_lr - min_lr) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_frac))
        warmup_val = min_lr + (max_lr - min_lr) * warmup_frac
        return jnp.where(pos < _warmup, warmup_val, cosine_val)

    return schedule_fn


class ReinFlow(Algorithm):
    """On-policy flow PPO with learned noise injection (SDE)."""

    def __init__(
        self,
        agent: FlowPPONet,
        params: FlowPPOParams,
        noise_net,
        noise_params,
        *,
        n_epochs: int = 5,
        ppo_eps: float = 0.01,
        gamma: float = 0.99,
        actor_lr: float = 3e-5,
        critic_lr: float = 3e-4,
        noise_lr: float = 3e-5,
        critic_weight_decay: float = 1e-4,
        reward_scale: float = 1.0,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.03,
        value_loss_coeff: float = 0.5,
        logprob_min: float = -1.0,
        logprob_max: float = 1.0,
        randn_clip: float = 3.0,
        clip_intermediate: float = 1.0,
        normalize_logprob: bool = True,
        n_critic_warmup: int = 10,
        noise_decay_hold_ratio: float = 0.35,
        noise_decay_ratio: float = 0.7,
        total_train_iters: int = 1000,
        min_noise_std: float = 0.10,
        max_noise_std: float = 0.24,
        reward_scale_running: bool = True,
        num_envs: int = 5,
        num_minibatches: int = 1,
        lr_cycle_steps: int = 100,
        lr_warmup_steps: int = 10,
        critic_min_lr: float = 1.5e-4,
        actor_min_lr: float = 1.5e-5,
        target_kl: float = 1.0,
        max_grad_norm: float = 0.0,
        huber_delta: float = 10.0,
        value_clip: float = 0.2,
        value_updates_per_batch: int = 10,
        policy_updates_per_batch: int = 1,
    ):
        self.agent = agent
        self.noise_net = noise_net
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_scale = reward_scale
        if reward_scale_running:
            self.reward_scaler = RunningRewardScaler(
                num_envs=num_envs, gamma=gamma,
            )
        else:
            self.reward_scaler = None
        self._n_epochs = n_epochs
        self._num_minibatches = num_minibatches
        self._epoch_counter = 0
        self._train_iter = 0
        self._n_critic_warmup = n_critic_warmup
        self._target_kl = target_kl
        self._n_value_updates = value_updates_per_batch
        self._n_policy_updates = policy_updates_per_batch

        self._traj_buffer = []
        self._logprob_buffer = []

        self._noise_decay_hold_ratio = noise_decay_hold_ratio
        self._noise_decay_ratio = noise_decay_ratio
        self._total_train_iters = total_train_iters
        self._min_noise_std = min_noise_std
        self._max_noise_std = max_noise_std
        self._current_max_noise = max_noise_std

        agent_ref = self.agent
        noise_net_ref = self.noise_net

        updates_per_iter = n_epochs * max(1, num_minibatches)
        cycle_sched = lr_cycle_steps * updates_per_iter
        warmup_sched = lr_warmup_steps * updates_per_iter

        actor_schedule = _periodic_cosine_lr(actor_lr, actor_min_lr, cycle_sched, warmup_sched)
        critic_schedule = _periodic_cosine_lr(critic_lr, critic_min_lr, cycle_sched, warmup_sched)
        noise_schedule = _periodic_cosine_lr(noise_lr, noise_lr * 0.1, cycle_sched, warmup_sched)

        if max_grad_norm > 0:
            self.policy_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(actor_schedule))
            self.value_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adamw(critic_schedule, weight_decay=critic_weight_decay))
            self.noise_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(noise_schedule))
        else:
            self.policy_optim = optax.adam(actor_schedule)
            self.value_optim = optax.adamw(critic_schedule, weight_decay=critic_weight_decay)
            self.noise_optim = optax.adam(noise_schedule)

        rf_params = ReinFlowParams(
            policy=params.policy,
            value=params.value,
            noise=noise_params,
        )

        self.state = ReinFlowTrainState(
            params=rf_params,
            opt_state=ReinFlowOptStates(
                policy=self.policy_optim.init(rf_params.policy),
                value=self.value_optim.init(rf_params.value),
                noise=self.noise_optim.init(rf_params.noise),
            ),
            step=jnp.int32(0),
        )

        _lp_min = logprob_min
        _lp_max = logprob_max
        _randn = randn_clip
        _clip_interm = clip_intermediate
        _normalize_lp = normalize_logprob

        @jax.jit
        def sde_get_action(key, policy_params, noise_params, obs, max_noise_clamp):
            B = obs.shape[0]

            def model_fn(t, x):
                return agent_ref.policy(policy_params, obs, x, jnp.full((B,), t))

            def sigma_fn(t, x):
                sigma = noise_net_ref.apply(noise_params, obs, jnp.full((B,), t))
                return jnp.minimum(sigma, max_noise_clamp)

            x_final, trajectory, logprob = agent_ref.flow.sde_sample(
                key, model_fn, sigma_fn, (B, agent_ref.act_dim),
                randn_clip=_randn,
                clip_intermediate=_clip_interm,
                normalize_logprob=_normalize_lp,
            )
            logprob = jnp.clip(logprob, _lp_min, _lp_max)
            action = x_final.clip(-1, 1)
            return action, trajectory, logprob

        self._sde_get_action = sde_get_action

        _ppo_eps = ppo_eps
        _ent_coef = ent_coef
        _vloss_coef = value_loss_coeff
        _num_steps = agent.num_steps
        _huber_delta = huber_delta
        _value_clip = value_clip
        _n_val = value_updates_per_batch
        _n_p = policy_updates_per_batch

        _act_dim = agent.act_dim

        @jax.jit
        def stateless_update(
            key: jax.Array,
            state: ReinFlowTrainState,
            data: GAEExperience,
            trajectories: jax.Array,
            old_logprobs: jax.Array,
            do_policy_update: bool,
            max_noise_clamp: jax.Array,
            old_values: jax.Array,
        ) -> Tuple[ReinFlowTrainState, Metric]:

            obs = data.obs
            action = data.action
            adv = data.adv
            ret = data.ret
            pp, vp, noise_p = state.params
            p_os, v_os, n_os = state.opt_state
            step = state.step
            B = obs.shape[0]

            adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv_sg = jax.lax.stop_gradient(adv_norm)

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
                v_updates, v_os = self.value_optim.update(v_grads, v_os, params=vp)
                vp = optax.apply_updates(vp, v_updates)

            # ---- Policy + noise update: _n_p times when do_policy_update ----
            traj_T = jnp.moveaxis(trajectories, 1, 0)

            def policy_loss_fn(pp_, np_):
                def model_fn(t, x):
                    return agent_ref.policy(pp_, obs, x, jnp.full((B,), t))

                def sigma_fn(t, x):
                    sigma = noise_net_ref.apply(np_, obs, jnp.full((B,), t))
                    return jnp.minimum(sigma, max_noise_clamp)

                new_lp = agent_ref.flow.compute_trajectory_logprob(
                    model_fn, sigma_fn, traj_T,
                    clip_intermediate=_clip_interm,
                    normalize_logprob=_normalize_lp,
                )
                new_lp = jnp.clip(new_lp, _lp_min, _lp_max)

                ratio = jnp.exp(new_lp - old_logprobs)
                surr1 = ratio * adv_sg
                surr2 = jnp.clip(ratio, 1.0 - _ppo_eps, 1.0 + _ppo_eps) * adv_sg
                ppo_loss = -jnp.minimum(surr1, surr2).mean()

                # True Gaussian entropy of SDE transitions (matching original)
                dt = 1.0 / _num_steps
                t_vals = jnp.linspace(0.0, 1.0 - dt, _num_steps)

                def _ent_step(_, t_val):
                    sigma = noise_net_ref.apply(np_, obs, jnp.full((B,), t_val))
                    sigma = jnp.minimum(sigma, max_noise_clamp)
                    h = 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * jnp.e * sigma ** 2 + 1e-16), axis=-1)
                    return None, h

                _, step_ents = jax.lax.scan(_ent_step, None, t_vals)
                joint_ent = step_ents.sum(axis=0)
                init_ent = 0.5 * _act_dim * jnp.log(2.0 * jnp.pi * jnp.e)
                joint_ent = joint_ent + init_ent
                if _normalize_lp:
                    entropy_rate = joint_ent / (_num_steps + 1) / _act_dim
                else:
                    entropy_rate = joint_ent / (_num_steps + 1)

                entropy_loss = -_ent_coef * entropy_rate.mean()

                return ppo_loss + entropy_loss, (ratio, new_lp)

            (p_loss, (ratio_val, new_lp_val)), (p_grads, n_grads) = (
                jax.value_and_grad(policy_loss_fn, argnums=(0, 1), has_aux=True)(pp, noise_p)
            )

            ratio_for_kl = jnp.exp(new_lp_val - old_logprobs)
            approx_kl = jnp.mean((ratio_for_kl - 1.0) - (new_lp_val - old_logprobs))

            def apply_policy_update(pp_, p_os_, noise_p_, n_os_):
                p_up, new_p_os_ = self.policy_optim.update(p_grads, p_os_)
                pp_new = optax.apply_updates(pp_, p_up)
                n_up, new_n_os_ = self.noise_optim.update(n_grads, n_os_)
                noise_p_new = optax.apply_updates(noise_p_, n_up)
                return pp_new, new_p_os_, noise_p_new, new_n_os_

            def skip_policy_update(pp_, p_os_, noise_p_, n_os_):
                return pp_, p_os_, noise_p_, n_os_

            pp, p_os, noise_p, n_os = jax.lax.cond(
                do_policy_update,
                apply_policy_update,
                skip_policy_update,
                pp, p_os, noise_p, n_os,
            )

            new_state = ReinFlowTrainState(
                params=ReinFlowParams(pp, vp, noise_p),
                opt_state=ReinFlowOptStates(p_os, v_os, n_os),
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
                "approx_kl": approx_kl,
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
    # Rollout: SDE sampling + trajectory storage
    # ----------------------------------------------------------------
    def _update_noise_decay(self):
        """Cosine decay of max noise std: hold → cosine → target."""
        itr = self._train_iter
        total = self._total_train_iters
        hold = self._noise_decay_hold_ratio
        ratio = self._noise_decay_ratio
        max_std = self._max_noise_std
        min_std = self._min_noise_std
        target = min_std * (1 - ratio) + max_std * ratio
        hold_iters = int(hold * total)
        if itr <= hold_iters:
            self._current_max_noise = max_std
        else:
            remaining = total - hold_iters
            if remaining <= 0:
                self._current_max_noise = target
            else:
                import math
                progress = min(1.0, (itr - hold_iters) / remaining)
                self._current_max_noise = target + (max_std - target) * (1 + math.cos(math.pi * progress)) / 2

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        p = self.state.params
        action, trajectory, logprob = self._sde_get_action(
            key, p.policy, p.noise, obs,
            jnp.float32(self._current_max_noise),
        )
        self._traj_buffer.append(np.asarray(jnp.moveaxis(trajectory, 0, 1)))
        self._logprob_buffer.append(np.asarray(logprob))
        return np.asarray(action)

    def _consolidate_trajectories(self):
        K_plus_1 = self._traj_buffer[0].shape[1]
        act_dim = self._traj_buffer[0].shape[2]
        trajs = np.stack(self._traj_buffer, axis=1).reshape(-1, K_plus_1, act_dim)
        logps = np.stack(self._logprob_buffer, axis=1).reshape(-1)
        self._consolidated_trajs = jnp.array(trajs)
        self._consolidated_logps = jnp.array(logps)
        self._traj_buffer.clear()
        self._logprob_buffer.clear()

    # ----------------------------------------------------------------
    # Update
    # ----------------------------------------------------------------
    def update(self, key: jax.Array, data: GAEExperience) -> Tuple[dict, dict]:
        if self._epoch_counter % self._n_epochs == 0:
            self._consolidate_trajectories()
            self._train_iter += 1
            self._update_noise_decay()
            self._kl_early_stop = False
            self._old_values = jnp.array(
                self.agent.value(self.state.params.value, data.obs)
            )
        self._epoch_counter += 1

        if self._kl_early_stop:
            return ({}, {})

        do_policy_update = self._train_iter >= self._n_critic_warmup

        B = data.obs.shape[0]
        M = self._num_minibatches

        if M <= 1 or B < M:
            self.state, info = self._update(
                key, self.state, data,
                self._consolidated_trajs,
                self._consolidated_logps,
                do_policy_update,
                jnp.float32(self._current_max_noise),
                self._old_values,
            )
            if self._target_kl > 0 and float(info.get("approx_kl", 0)) > self._target_kl:
                self._kl_early_stop = True
        else:
            mb_size = B // M
            perm = np.random.permutation(B)
            info = {}
            for m in range(M):
                if self._kl_early_stop:
                    break
                idx = perm[m * mb_size:(m + 1) * mb_size]
                mb_data = GAEExperience(
                    data.obs[idx], data.action[idx], data.reward[idx],
                    data.done[idx], data.next_obs[idx],
                    data.ret[idx], data.adv[idx], data.truncated[idx],
                )
                mb_trajs = self._consolidated_trajs[idx]
                mb_logps = self._consolidated_logps[idx]
                mb_old_vals = self._old_values[idx]
                mb_key = jax.random.fold_in(key, m)
                self.state, info = self._update(
                    mb_key, self.state, mb_data,
                    mb_trajs, mb_logps,
                    do_policy_update,
                    jnp.float32(self._current_max_noise),
                    mb_old_vals,
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
        K = self.agent.num_steps

        dummy_trajs = jnp.zeros((B, K + 1, self.agent.act_dim))
        dummy_logps = jnp.zeros((B,))
        dummy_old_values = jnp.zeros((B,))

        self._update(
            key, self.state, data,
            dummy_trajs, dummy_logps, True,
            jnp.float32(self._current_max_noise),
            dummy_old_values,
        )
        pp = self.get_policy_params()
        obs = data.obs[0]
        self._get_action(key, pp, obs)
        self._get_deterministic_action(pp, obs)
        p = self.state.params
        self._sde_get_action(key, p.policy, p.noise, data.obs[:2], jnp.float32(self._current_max_noise))

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
