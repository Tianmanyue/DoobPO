"""
DoobPO-Flow-GRPO: Flow Policy Optimization with Group-Relative Advantage.

Same as DoobPO-Flow (PPO) but replaces the single-sample baseline advantage
    A(s,a) = Q(s,a) - Q(s, pi(s))
with a group-relative advantage (GRPO-style):
    sample G actions {a_1,...,a_G} ~ pi(.|s)
    A(s,a) = (Q(s,a) - mean(group_Q)) / (std(group_Q) + eps)

Everything else is identical: ratio network with PPO clipping, regularizer,
and weighted velocity matching distillation.
"""

from typing import NamedTuple, Optional, Tuple, Union
import math

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


class DoobPOFlowGRPOParams(NamedTuple):
    policy: hk.Params
    value: hk.Params
    ratio: hk.Params


class DoobPOFlowGRPOParamsWithQ(NamedTuple):
    policy: hk.Params
    value: hk.Params
    ratio: hk.Params
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    target_policy: hk.Params
    log_alpha: jax.Array


class DoobPOFlowGRPOOptStates(NamedTuple):
    policy: optax.OptState
    value: optax.OptState
    ratio: optax.OptState


class DoobPOFlowGRPOOptStatesWithQ(NamedTuple):
    policy: optax.OptState
    value: optax.OptState
    ratio: optax.OptState
    q1: optax.OptState
    q2: optax.OptState
    log_alpha: optax.OptState


class DoobPOFlowGRPOTrainState(NamedTuple):
    params: Union[DoobPOFlowGRPOParams, DoobPOFlowGRPOParamsWithQ]
    opt_state: Union[DoobPOFlowGRPOOptStates, DoobPOFlowGRPOOptStatesWithQ]
    step: int


class DoobPOFlowGRPO(Algorithm):
    """Flow DoobPO with GRPO-style group-relative advantage."""

    def __init__(
        self,
        agent: FlowPPONet,
        params: FlowPPOParams,
        ratio_net,
        ratio_params,
        *,
        gamma: float = 0.99,
        lr: float = 3e-4,
        value_lr: float = 1e-4,
        ratio_lr: float = 3e-4,
        q_lr: float = 3e-4,
        reward_scale: float = 1.0,
        ppo_eps: float = 0.2,
        max_ratio_weight: float = 5.0,
        ratio_regularizer_lambda: float = 0.01,
        gae_lambda: float = 0.95,
        value_loss_coeff: float = 0.5,
        n_epochs: int = 4,
        num_minibatches: int = 1,
        max_grad_norm: float = 1.0,
        huber_delta: float = 10.0,
        value_clip: float = 0.2,
        value_updates_per_batch: int = 10,
        ratio_updates_per_batch: int = 1,
        policy_updates_per_batch: int = 1,
        off_policy: bool = False,
        q_apply=None,
        q1_params: Optional[hk.Params] = None,
        q2_params: Optional[hk.Params] = None,
        target_q1_params: Optional[hk.Params] = None,
        target_q2_params: Optional[hk.Params] = None,
        tau: float = 0.005,
        delay_q_target_update: int = 2,
        delay_policy_update: int = 2,
        lr_schedule_end: float = 3e-5,
        lr_schedule_steps: int = 50000,
        alpha_lr: float = 7e-3,
        delay_alpha_update: int = 250,
        noise_scale: float = 0.1,
        target_entropy_scale: float = 0.9,
        num_grpo_samples: int = 16,
        adv_clip: float = 0.0,
        q_backup_clip: float = 0.0,
        grpo_batch_norm_adv: bool = False,
    ):
        self.agent = agent
        self.ratio_net = ratio_net
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_scale = reward_scale
        self.ppo_eps = ppo_eps
        self.max_ratio_weight = max_ratio_weight
        self._n_epochs = n_epochs
        self._num_minibatches = num_minibatches
        self._epoch_counter = 0
        self._n_value_updates = value_updates_per_batch
        self._n_ratio_updates = ratio_updates_per_batch
        self._n_policy_updates = policy_updates_per_batch
        self._off_policy = off_policy
        self._tau = tau
        self.q_apply = q_apply
        self._delay_q_target_update = max(1, int(delay_q_target_update))
        self._delay_policy_update = max(1, int(delay_policy_update))
        self._delay_alpha_update = max(1, int(delay_alpha_update))
        self._noise_scale = noise_scale
        self._target_entropy = -agent.act_dim * target_entropy_scale
        self._num_grpo_samples = num_grpo_samples

        if off_policy:
            policy_lr = optax.schedules.linear_schedule(
                init_value=lr,
                end_value=lr_schedule_end,
                transition_steps=lr_schedule_steps,
                transition_begin=lr_schedule_steps // 2,
            )
        else:
            policy_lr = lr

        if max_grad_norm > 0:
            self.policy_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(policy_lr))
            self.value_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(value_lr))
            self.ratio_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(ratio_lr))
            self.q_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(q_lr))
            self.alpha_optim = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(alpha_lr))
        else:
            self.policy_optim = optax.adam(policy_lr)
            self.value_optim = optax.adam(value_lr)
            self.ratio_optim = optax.adam(ratio_lr)
            self.q_optim = optax.adam(q_lr)
            self.alpha_optim = optax.adam(alpha_lr)

        if off_policy:
            assert q_apply is not None and all(
                x is not None
                for x in (q1_params, q2_params, target_q1_params, target_q2_params)
            )
            log_alpha_init = jnp.float32(math.log(5))
            grpo_params = DoobPOFlowGRPOParamsWithQ(
                policy=params.policy,
                value=params.value,
                ratio=ratio_params,
                q1=q1_params,
                q2=q2_params,
                target_q1=target_q1_params,
                target_q2=target_q2_params,
                target_policy=params.policy,
                log_alpha=log_alpha_init,
            )
            self.state = DoobPOFlowGRPOTrainState(
                params=grpo_params,
                opt_state=DoobPOFlowGRPOOptStatesWithQ(
                    policy=self.policy_optim.init(grpo_params.policy),
                    value=self.value_optim.init(grpo_params.value),
                    ratio=self.ratio_optim.init(grpo_params.ratio),
                    q1=self.q_optim.init(grpo_params.q1),
                    q2=self.q_optim.init(grpo_params.q2),
                    log_alpha=self.alpha_optim.init(log_alpha_init),
                ),
                step=jnp.int32(0),
            )
        else:
            grpo_params = DoobPOFlowGRPOParams(
                policy=params.policy,
                value=params.value,
                ratio=ratio_params,
            )
            self.state = DoobPOFlowGRPOTrainState(
                params=grpo_params,
                opt_state=DoobPOFlowGRPOOptStates(
                    policy=self.policy_optim.init(grpo_params.policy),
                    value=self.value_optim.init(grpo_params.value),
                    ratio=self.ratio_optim.init(grpo_params.ratio),
                ),
                step=jnp.int32(0),
            )

        agent_ref = self.agent
        _ppo_eps = ppo_eps
        _max_rw = max_ratio_weight
        _reg_lambda = ratio_regularizer_lambda
        _vloss_coef = value_loss_coeff
        _huber_delta = huber_delta
        _value_clip = value_clip
        _n_val = value_updates_per_batch
        _n_r = ratio_updates_per_batch
        _n_p = policy_updates_per_batch
        _G = num_grpo_samples

        # ---- On-policy (GAE) update ----
        # On-policy already has GAE advantages; we use group-relative normalization
        # by sampling G actions per state and comparing.
        def stateless_update_on(
            key: jax.Array,
            state: DoobPOFlowGRPOTrainState,
            data: GAEExperience,
            old_values: jax.Array,
            old_policy_params: hk.Params,
        ) -> Tuple[DoobPOFlowGRPOTrainState, Metric]:
            obs = data.obs
            action = data.action
            adv = data.adv
            ret = data.ret
            pp, vp, rp = state.params.policy, state.params.value, state.params.ratio
            p_os, v_os, r_os = (
                state.opt_state.policy,
                state.opt_state.value,
                state.opt_state.ratio,
            )
            step = state.step

            # GRPO-style: use GAE advantage but normalize group-relative
            # (on-policy already has per-trajectory advantages; normalize as usual)
            adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv_sg = jax.lax.stop_gradient(adv_norm)

            flow_t_key, flow_noise_key, ratio_reg_keys = jax.random.split(key, 3)
            ratio_reg_keys = jax.random.split(ratio_reg_keys, _n_r * 2 + 1)[: _n_r * 2]

            trunc_mask = 1.0 - data.truncated.astype(jnp.float32)

            def value_loss_fn(vp_):
                v_pred = agent_ref.value(vp_, obs)
                v_clipped = old_values + jnp.clip(
                    v_pred - old_values, -_value_clip, _value_clip
                )
                loss_orig = _huber_loss(ret - v_pred, _huber_delta)
                loss_clipped = _huber_loss(ret - v_clipped, _huber_delta)
                return _vloss_coef * jnp.mean(jnp.maximum(loss_orig, loss_clipped) * trunc_mask)

            v_loss = 0.0
            for _ in range(_n_val):
                v_loss, v_grads = jax.value_and_grad(value_loss_fn)(vp)
                v_updates, v_os = self.value_optim.update(v_grads, v_os)
                vp = optax.apply_updates(vp, v_updates)

            rp_new = rp
            r_beta_val = self.ratio_net.apply(rp, obs, action)
            ratio_loss_val = 0.0
            for i in range(_n_r):
                action_g1 = jax.lax.stop_gradient(
                    agent_ref.get_action(ratio_reg_keys[i * 2], old_policy_params, obs)
                )
                action_g2 = jax.lax.stop_gradient(
                    agent_ref.get_action(ratio_reg_keys[i * 2 + 1], old_policy_params, obs)
                )

                def ratio_loss_fn(rp_):
                    r_beta = self.ratio_net.apply(rp_, obs, action)
                    r_clip = jnp.clip(r_beta, 1.0 - _ppo_eps, 1.0 + _ppo_eps)
                    ppo_obj = jnp.minimum(r_beta * adv_sg, r_clip * adv_sg)

                    r_beta_g1 = self.ratio_net.apply(rp_, obs, action_g1)
                    r_beta_g2 = self.ratio_net.apply(rp_, obs, action_g2)
                    g1 = r_beta_g1.mean() - 1.0
                    g2 = r_beta_g2.mean() - 1.0
                    regularizer = _reg_lambda * g1 * g2

                    return -ppo_obj.mean() + regularizer, r_beta

                (ratio_loss_val, r_beta_val), ratio_grads = jax.value_and_grad(
                    ratio_loss_fn, has_aux=True
                )(rp_new)

                r_updates, r_os = self.ratio_optim.update(ratio_grads, r_os)
                rp_new = optax.apply_updates(rp_new, r_updates)

            def policy_loss_fn(pp_):
                r_raw = jax.lax.stop_gradient(
                    self.ratio_net.apply(rp_new, obs, action)
                )
                r_norm = r_raw / (r_raw.mean() + 1e-8)
                r_sg = jnp.clip(r_norm, 0.0, _max_rw)

                def velocity_model(t, x):
                    return agent_ref.policy(pp_, obs, x, t)

                t = jax.random.uniform(flow_t_key, (obs.shape[0],))
                return agent_ref.flow.weighted_velocity_loss(
                    flow_noise_key,
                    r_sg,
                    velocity_model,
                    t,
                    jax.lax.stop_gradient(action),
                )

            p_loss = 0.0
            for _ in range(_n_p):
                p_loss, p_grads = jax.value_and_grad(policy_loss_fn)(pp)
                p_updates, p_os = self.policy_optim.update(p_grads, p_os)
                pp = optax.apply_updates(pp, p_updates)

            new_state = DoobPOFlowGRPOTrainState(
                params=DoobPOFlowGRPOParams(pp, vp, rp_new),
                opt_state=DoobPOFlowGRPOOptStates(p_os, v_os, r_os),
                step=step + 1,
            )
            info = {
                "policy_loss": p_loss,
                "value_loss": v_loss,
                "ratio_loss": ratio_loss_val,
                "r_beta_mean": jnp.mean(r_beta_val),
                "r_beta_std": jnp.std(r_beta_val),
                "r_beta_max": jnp.max(r_beta_val),
                "advantage_mean": jnp.mean(adv),
            }
            return new_state, info

        # ---- Off-policy (twin Q) update with GRPO advantage ----
        q_apply = self.q_apply
        _gamma = gamma
        _tau = tau
        _rs = reward_scale
        _delay_q_tgt = self._delay_q_target_update
        _delay_p = self._delay_policy_update
        _delay_alpha = self._delay_alpha_update
        _ns = self._noise_scale
        _target_ent = self._target_entropy
        _act_dim = agent.act_dim
        _adv_clip = adv_clip
        _q_backup_clip = q_backup_clip
        _batch_norm_adv = grpo_batch_norm_adv

        def stateless_update_off(
            key: jax.Array,
            state: DoobPOFlowGRPOTrainState,
            data: Experience,
        ) -> Tuple[DoobPOFlowGRPOTrainState, Metric]:
            obs, action, reward, next_obs, done = (
                data.obs,
                data.action,
                data.reward,
                data.next_obs,
                data.done,
            )
            reward = reward * _rs
            p = state.params
            pp, rp = p.policy, p.ratio
            q1p, q2p, tq1, tq2 = p.q1, p.q2, p.target_q1, p.target_q2
            log_alpha = p.log_alpha
            os = state.opt_state
            p_os, r_os, q1_os, q2_os = os.policy, os.ratio, os.q1, os.q2
            la_os = os.log_alpha
            step = state.step

            (
                next_act_key,
                grpo_key,
                flow_t_key,
                flow_noise_key,
                ratio_reg_base,
            ) = jax.random.split(key, 5)
            ratio_reg_keys = jax.random.split(ratio_reg_base, _n_r + 1)[:_n_r]

            def min_q(t1, t2, s, a):
                return jnp.minimum(q_apply(t1, s, a), q_apply(t2, s, a))

            # --- Noisy action for Q-backup ---
            next_act_rng, next_noise_rng = jax.random.split(next_act_key)
            next_action = agent_ref.get_action(next_act_rng, pp, next_obs)
            next_action = next_action + jax.random.normal(
                next_noise_rng, next_action.shape
            ) * jnp.exp(log_alpha) * _ns
            next_action = next_action.clip(-1, 1)

            q_target = jax.lax.stop_gradient(
                min_q(tq1, tq2, next_obs, next_action)
            )
            q_target = jnp.nan_to_num(q_target, nan=0.0, posinf=1e4, neginf=-1e4)
            q_backup = reward + (1.0 - done.astype(jnp.float32)) * _gamma * q_target
            q_backup = jax.lax.cond(
                _q_backup_clip > 0,
                lambda: jnp.clip(q_backup, -_q_backup_clip, _q_backup_clip),
                lambda: q_backup,
            )

            def q_loss_fn(qp):
                q_val = q_apply(qp, obs, action)
                return jnp.mean((q_val - q_backup) ** 2), q_val

            (q1_loss, q1_val), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1p)
            (q2_loss, q2_val), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2p)

            q1_updates, q1_os = self.q_optim.update(q1_grads, q1_os)
            q1p = optax.apply_updates(q1p, q1_updates)
            q2_updates, q2_os = self.q_optim.update(q2_grads, q2_os)
            q2p = optax.apply_updates(q2p, q2_updates)

            # --- GRPO-style group-relative advantage ---
            q_sa = jax.lax.stop_gradient(min_q(tq1, tq2, obs, action))

            grpo_keys = jax.random.split(grpo_key, _G)

            def _sample_and_eval(k):
                a_rng, n_rng = jax.random.split(k)
                a = agent_ref.get_action(a_rng, pp, obs)
                a = a + jax.random.normal(n_rng, a.shape) * jnp.exp(log_alpha) * _ns
                a = a.clip(-1, 1)
                return min_q(tq1, tq2, obs, a)

            sampled_q = jax.vmap(_sample_and_eval)(grpo_keys)  # (G, batch)

            all_q = jnp.concatenate([sampled_q, q_sa[None]], axis=0)  # (G+1, batch)
            group_mean = all_q.mean(axis=0)
            group_std = all_q.std(axis=0) + 1e-8

            if _batch_norm_adv:
                adv_raw = q_sa - group_mean
                adv = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)
            else:
                adv = (q_sa - group_mean) / group_std
            adv = jax.lax.cond(
                _adv_clip > 0,
                lambda: jnp.clip(adv, -_adv_clip, _adv_clip),
                lambda: adv,
            )
            adv_sg = jax.lax.stop_gradient(adv)

            # --- Ratio update (PPO clipped, same as DoobPO-Flow) ---
            rp_new = rp
            ratio_loss_val = 0.0
            r_beta_val = self.ratio_net.apply(rp_new, obs, action)
            for ri in range(_n_r):
                action_prime = jax.lax.stop_gradient(
                    agent_ref.get_action(ratio_reg_keys[ri], pp, obs)
                )

                def ratio_loss_fn(rp_):
                    r_beta = self.ratio_net.apply(rp_, obs, action)
                    r_clip = jnp.clip(r_beta, 1.0 - _ppo_eps, 1.0 + _ppo_eps)
                    ppo_obj = jnp.minimum(r_beta * adv_sg, r_clip * adv_sg)
                    r_beta_prime = self.ratio_net.apply(rp_, obs, action_prime)
                    g_hat_1 = r_beta.mean() - 1.0
                    g_hat_2 = r_beta_prime.mean() - 1.0
                    regularizer = _reg_lambda * g_hat_1 * g_hat_2
                    return -ppo_obj.mean() + regularizer, r_beta

                (ratio_loss_val, r_beta_val), ratio_grads = jax.value_and_grad(
                    ratio_loss_fn, has_aux=True
                )(rp_new)
                r_updates, r_os = self.ratio_optim.update(ratio_grads, r_os)
                rp_new = optax.apply_updates(rp_new, r_updates)

            # --- Policy update (velocity matching, same as DoobPO-Flow) ---
            def policy_loss_fn(pp_):
                r_raw = jax.lax.stop_gradient(
                    self.ratio_net.apply(rp_new, obs, action)
                )
                r_norm = r_raw / (r_raw.mean() + 1e-8)
                r_sg = jnp.clip(r_norm, 0.0, _max_rw)

                def velocity_model(t, x):
                    return agent_ref.policy(pp_, obs, x, t)

                t = jax.random.uniform(flow_t_key, (obs.shape[0],))
                return agent_ref.flow.weighted_velocity_loss(
                    flow_noise_key,
                    r_sg,
                    velocity_model,
                    t,
                    jax.lax.stop_gradient(action),
                )

            tp = p.target_policy

            def _do_policy_update(pp_, p_os_):
                p_loss_, p_grads_ = jax.value_and_grad(policy_loss_fn)(pp_)
                p_updates_, p_os_ = self.policy_optim.update(p_grads_, p_os_)
                pp_ = optax.apply_updates(pp_, p_updates_)
                return pp_, p_os_, p_loss_

            pp_new, p_os_new, p_loss = jax.lax.cond(
                (step % _delay_p) == 0,
                lambda: _do_policy_update(pp, p_os),
                lambda: (pp, p_os, jnp.float32(0.0)),
            )

            do_target_update = (step % _delay_q_tgt) == 0
            tq1 = jax.lax.cond(
                do_target_update,
                lambda: optax.incremental_update(q1p, tq1, _tau),
                lambda: tq1,
            )
            tq2 = jax.lax.cond(
                do_target_update,
                lambda: optax.incremental_update(q2p, tq2, _tau),
                lambda: tq2,
            )
            tp = jax.lax.cond(
                do_target_update,
                lambda: optax.incremental_update(pp_new, tp, _tau),
                lambda: tp,
            )

            # --- Alpha update ---
            def log_alpha_loss_fn(la):
                approx_entropy = 0.5 * _act_dim * jnp.log(
                    2 * jnp.pi * jnp.exp(1) * (_ns * jnp.exp(la)) ** 2
                )
                return -la * (jax.lax.stop_gradient(-approx_entropy) + _target_ent)

            def _do_alpha_update(la_, la_os_):
                la_grad = jax.grad(log_alpha_loss_fn)(la_)
                la_upd, la_os_new_ = self.alpha_optim.update(la_grad, la_os_)
                return optax.apply_updates(la_, la_upd), la_os_new_

            la_new, la_os_new = jax.lax.cond(
                (step % _delay_alpha) == 0,
                lambda: _do_alpha_update(log_alpha, la_os),
                lambda: (log_alpha, la_os),
            )

            new_state = DoobPOFlowGRPOTrainState(
                params=DoobPOFlowGRPOParamsWithQ(
                    pp_new, p.value, rp_new, q1p, q2p, tq1, tq2, tp, la_new
                ),
                opt_state=DoobPOFlowGRPOOptStatesWithQ(
                    p_os_new, state.opt_state.value, r_os, q1_os, q2_os, la_os_new
                ),
                step=step + 1,
            )
            info = {
                "policy_loss": p_loss,
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "ratio_loss": ratio_loss_val,
                "r_beta_mean": jnp.mean(r_beta_val),
                "r_beta_std": jnp.std(r_beta_val),
                "r_beta_max": jnp.max(r_beta_val),
                "advantage_mean": jnp.mean(adv),
                "advantage_std": jnp.std(adv),
                "q1_mean": jnp.mean(q1_val),
                "alpha": jnp.exp(la_new),
                "grpo_group_std": jnp.mean(group_std),
            }
            return new_state, info

        self._update_on = jax.jit(stateless_update_on)
        if off_policy:
            self._update_off = jax.jit(stateless_update_off)
            self._get_min_q = jax.jit(
                lambda t1, t2, o, a: jnp.minimum(
                    q_apply(t1, o, a), q_apply(t2, o, a)
                )
            )

            def _min_q_value(params_bundle, obs_, act_):
                pr = params_bundle
                return jnp.minimum(
                    q_apply(pr[0], obs_, act_), q_apply(pr[1], obs_, act_)
                )

            @jax.jit
            def _noisy_get_action(rng, pp_, la_, obs_):
                act_rng, noise_rng = jax.random.split(rng)
                act_ = agent_ref.get_action(act_rng, pp_, obs_)
                act_ = act_ + jax.random.normal(noise_rng, act_.shape) * jnp.exp(la_) * _ns
                return act_.clip(-1, 1)

            self._get_noisy_action = _noisy_get_action

            self._implement_common_behavior(
                stateless_update_off,
                self.agent.get_action,
                self.agent.get_deterministic_action,
                stateless_get_value=_min_q_value,
            )
        else:
            self._update_off = None
            self._get_noisy_action = None
            self._implement_common_behavior(
                stateless_update_on,
                self.agent.get_action,
                self.agent.get_deterministic_action,
            )

    def get_policy_params(self):
        return self.state.params.policy

    def get_policy_params_to_save(self):
        if self._off_policy:
            return self.state.params.target_policy
        return self.state.params.policy

    def get_value_params(self):
        if self._off_policy:
            return (self.state.params.target_q1, self.state.params.target_q2)
        return self.state.params.value

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        if not self._off_policy:
            return np.asarray(self.agent.value(self.state.params.value, obs))
        action = self.get_deterministic_action(obs)
        p = self.state.params
        out = self._get_min_q(p.target_q1, p.target_q2, obs, action)
        return np.asarray(out)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        if self._off_policy:
            action = self._get_noisy_action(
                key, self.get_policy_params_to_save(),
                self.state.params.log_alpha, obs,
            )
        else:
            action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)

    def update(
        self, key: jax.Array, data: Union[Experience, GAEExperience]
    ) -> Tuple[dict, dict]:
        if self._off_policy:
            self.state, info = self._update_off(key, self.state, data)
            return (
                {k: float(v) for k, v in info.items() if not k.startswith("hist")},
                {k: v for k, v in info.items() if k.startswith("hist")},
            )

        assert isinstance(data, GAEExperience)
        if self._epoch_counter % self._n_epochs == 0:
            self._old_values = jnp.array(
                self.agent.value(self.state.params.value, data.obs)
            )
            self._old_policy_params = self.state.params.policy
        self._epoch_counter += 1

        B = data.obs.shape[0]
        M = self._num_minibatches

        if M <= 1 or B < M:
            self.state, info = self._update_on(
                key, self.state, data, self._old_values, self._old_policy_params
            )
        else:
            mb_size = B // M
            perm = np.random.permutation(B)
            info = {}
            for m in range(M):
                idx = perm[m * mb_size : (m + 1) * mb_size]
                mb_data = GAEExperience(
                    data.obs[idx],
                    data.action[idx],
                    data.reward[idx],
                    data.done[idx],
                    data.next_obs[idx],
                    data.ret[idx],
                    data.adv[idx],
                    data.truncated[idx],
                )
                mb_old_vals = self._old_values[idx]
                mb_key = jax.random.fold_in(key, m)
                self.state, info = self._update_on(
                    mb_key,
                    self.state,
                    mb_data,
                    mb_old_vals,
                    self._old_policy_params,
                )

        return (
            {k: float(v) for k, v in info.items() if not k.startswith("hist")},
            {k: v for k, v in info.items() if k.startswith("hist")},
        )

    def warmup(self, data: Union[Experience, GAEExperience]) -> None:
        key = jax.random.key(0)
        if self._off_policy:
            assert isinstance(data, Experience)
            self._old_policy_params = self.state.params.policy
            self.state, _ = self._update_off(key, self.state, data)
        else:
            assert isinstance(data, GAEExperience)
            B = data.obs.shape[0]
            dummy_old_values = jnp.zeros((B,))
            self._old_policy_params = self.state.params.policy
            self.state, _ = self._update_on(
                key, self.state, data, dummy_old_values, self._old_policy_params
            )
        pp = self.get_policy_params()
        obs = data.obs[0]
        self._get_action(key, pp, obs)
        self._get_deterministic_action(pp, obs)
        if self._off_policy:
            self._get_noisy_action(key, pp, self.state.params.log_alpha, obs)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)


def _huber_loss(error: jax.Array, delta: float) -> jax.Array:
    abs_error = jnp.abs(error)
    return jnp.where(
        abs_error < delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta),
    )
