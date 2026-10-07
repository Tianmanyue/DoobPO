"""
DoobGRPO: same off-policy backbone as DoobPO, but ratio-network training uses a
GRPO-style *group-relative* advantage; the ratio surrogate matches DoobPO's PPO-style
clipping (evaluated at each (s, a_k) over the K on-policy samples).

For each replay state s, sample K actions from the current policy π(·|s) (stop-grad).
Let Q_k = min_j Q_target(s, a_k). Define group baseline V_group(s) = mean_k Q_k and
group-normalised advantages

    A_grpo_k = (Q_k - mean_j Q_j) / (std_j Q_j + ε)   (normalisation across the K samples)

Ratio loss (PPO clipping on r_β, same form as DoobPO):

    L_ratio = - E[ min( r_β A, clip(r_β, 1±ε) A ) ]
              + λ · (mean_{group1} r_β - 1) · (mean_{group2} r_β - 1)

The regulariser follows the same unbiased double-mean trick as DoobPO, but both means
are taken over independent on-policy action groups (instead of buffer vs π).

Policy drift-matching still uses replay (s, a_buf) with stop-grad weights r_β(s, a_buf),
identical to DoobPO.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.algorithm.doobpo import DoobPOOptStates, DoobPOParams, DoobPOTrainState
from relax.network.diffv2 import Diffv2Net, Diffv2Params
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


class DoobPOGRPO(Algorithm):
    """Off-policy DoobPO with GRPO-style group-relative ratio learning."""

    def __init__(
        self,
        agent: Diffv2Net,
        params: Diffv2Params,
        ratio_net,
        ratio_params,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        ratio_lr: float = 3e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        lr_schedule_steps: int = 50000,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        grpo_group_size: int = 8,
        max_ratio_weight: float = 5.0,
        ratio_regularizer_lambda: float = 0.01,
        ppo_eps: float = 0.2,
        use_ema: bool = True,
    ):
        self.agent = agent
        self.ratio_net = ratio_net
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.grpo_group_size = int(grpo_group_size)
        self.max_ratio_weight = max_ratio_weight
        self.ratio_regularizer_lambda = ratio_regularizer_lambda
        self.ppo_eps = ppo_eps

        self.q_optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=lr_schedule_steps,
            transition_begin=lr_schedule_steps // 2,
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.ratio_optim = optax.adam(ratio_lr)
        self.alpha_optim = optax.adam(alpha_lr)

        doobpo_params = DoobPOParams(
            q1=params.q1,
            q2=params.q2,
            target_q1=params.target_q1,
            target_q2=params.target_q2,
            policy=params.policy,
            target_policy=params.target_poicy,
            ratio=ratio_params,
            log_alpha=params.log_alpha,
        )

        self.state = DoobPOTrainState(
            params=doobpo_params,
            opt_state=DoobPOOptStates(
                q1=self.q_optim.init(doobpo_params.q1),
                q2=self.q_optim.init(doobpo_params.q2),
                policy=self.policy_optim.init(doobpo_params.policy),
                ratio=self.ratio_optim.init(doobpo_params.ratio),
                log_alpha=self.alpha_optim.init(doobpo_params.log_alpha),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0),
        )

        K = self.grpo_group_size
        max_ratio_weight_const = max_ratio_weight
        ratio_reg_lambda = self.ratio_regularizer_lambda
        ppo_eps_const = self.ppo_eps

        @jax.jit
        def stateless_update(
            key: jax.Array,
            state: DoobPOTrainState,
            data: Experience,
        ) -> Tuple[DoobPOTrainState, Metric]:
            obs, action, reward, next_obs, done = (
                data.obs, data.action, data.reward, data.next_obs, data.done
            )
            (q1_params, q2_params, target_q1, target_q2,
             policy_params, target_policy, ratio_params, log_alpha) = state.params
            (q1_opt_state, q2_opt_state, policy_opt_state,
             ratio_opt_state, log_alpha_opt_state) = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std

            reward = reward * self.reward_scale

            n_split = 3 + 2 * K
            keys = jax.random.split(key, n_split)
            next_eval_key = keys[0]
            diffusion_t_key = keys[1]
            diffusion_noise_key = keys[2]
            grpo_keys_1 = keys[3 : 3 + K]
            grpo_keys_2 = keys[3 + K : 3 + 2 * K]

            def min_q(p1, p2, s, a):
                return jnp.minimum(self.agent.q(p1, s, a), self.agent.q(p2, s, a))

            pol = (policy_params, log_alpha, q1_params, q2_params)

            next_action = self.agent.get_action(
                next_eval_key,
                pol,
                next_obs,
            )
            q_target = jax.lax.stop_gradient(
                min_q(target_q1, target_q2, next_obs, next_action)
            )
            q_backup = reward + (1.0 - done) * self.gamma * q_target

            def q_loss_fn(qp):
                q_val = self.agent.q(qp, obs, action)
                return jnp.mean((q_val - q_backup) ** 2), q_val

            (q1_loss, q1_val), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2_val), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)

            # Running stats (same as DoobPO: use buffer Q(s,a))
            q_sa = jax.lax.stop_gradient(min_q(target_q1, target_q2, obs, action))

            def sample_policy_actions(subkeys):
                acts = jax.vmap(
                    lambda sk: self.agent.get_action(sk, pol, obs)
                )(subkeys)
                return jax.lax.stop_gradient(acts)

            actions_g1 = sample_policy_actions(grpo_keys_1)  # (K, B, act_dim)
            actions_g2 = sample_policy_actions(grpo_keys_2)

            def q_on_actions(actions_kb):
                return jax.vmap(
                    lambda a: min_q(target_q1, target_q2, obs, a)
                )(actions_kb)

            q_vals = q_on_actions(actions_g1)  # (K, B)
            adv_mean_kb = jnp.mean(q_vals, axis=0, keepdims=True)
            adv_std_kb = jnp.std(q_vals, axis=0, keepdims=True) + 1e-8
            adv_grpo = (q_vals - adv_mean_kb) / adv_std_kb
            adv_grpo_sg = jax.lax.stop_gradient(adv_grpo)

            obs_k = jnp.broadcast_to(obs, (K,) + obs.shape)  # (K, B, obs_dim)
            flat_obs = jnp.reshape(obs_k, (K * obs.shape[0], obs.shape[-1]))
            flat_act1 = jnp.reshape(actions_g1, (K * obs.shape[0], actions_g1.shape[-1]))
            flat_act2 = jnp.reshape(actions_g2, (K * obs.shape[0], actions_g2.shape[-1]))

            def ratio_loss_fn(rp):
                r_flat1 = self.ratio_net.apply(rp, flat_obs, flat_act1)
                r_kb = jnp.reshape(r_flat1, (K, obs.shape[0]))
                r_clip = jnp.clip(
                    r_kb, 1.0 - ppo_eps_const, 1.0 + ppo_eps_const
                )
                ppo_obj = jnp.minimum(
                    r_kb * adv_grpo_sg, r_clip * adv_grpo_sg
                )

                r_flat2 = self.ratio_net.apply(rp, flat_obs, flat_act2)
                r_kb2 = jnp.reshape(r_flat2, (K, obs.shape[0]))
                g_hat_1 = r_kb.mean() - 1.0
                g_hat_2 = r_kb2.mean() - 1.0
                regularizer = ratio_reg_lambda * g_hat_1 * g_hat_2

                return -jnp.mean(ppo_obj) + regularizer, r_kb

            (ratio_loss_val, r_beta_kb), ratio_grads = jax.value_and_grad(
                ratio_loss_fn, has_aux=True
            )(ratio_params)

            ratio_update, ratio_opt_state_new = self.ratio_optim.update(ratio_grads, ratio_opt_state)
            ratio_params_new = optax.apply_updates(ratio_params, ratio_update)

            def policy_loss_fn(pp):
                r_raw = jax.lax.stop_gradient(
                    self.ratio_net.apply(ratio_params_new, obs, action)
                )
                r_norm = r_raw / (r_raw.mean() + 1e-8)
                r_beta_sg = jnp.clip(r_norm, 0.0, max_ratio_weight_const)

                def denoiser(t, x):
                    return self.agent.policy(pp, obs, x, t)

                t = jax.random.randint(
                    diffusion_t_key,
                    (obs.shape[0],),
                    0,
                    self.agent.num_timesteps,
                )
                return self.agent.diffusion.weighted_p_loss(
                    diffusion_noise_key, r_beta_sg, denoiser, t,
                    jax.lax.stop_gradient(action),
                )

            policy_loss_val, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)

            def log_alpha_loss_fn(la):
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log(
                    2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(la)) ** 2
                )
                return -la * (-jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)

            def param_update(optim, params, grads, opt_state):
                updates, new_opt_state = optim.update(grads, opt_state)
                return optax.apply_updates(params, updates), new_opt_state

            def delay_update_fn(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda p, s: param_update(optim, p, grads, s),
                    lambda p, s: (p, s),
                    params, opt_state,
                )

            def delay_alpha_update_fn(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda p, s: param_update(
                        optim, p, jax.grad(log_alpha_loss_fn)(p), s
                    ),
                    lambda p, s: (p, s),
                    params, opt_state,
                )

            def target_update_fn(params, target, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda t: optax.incremental_update(params, t, tau),
                    lambda t: t,
                    target,
                )

            q1_params, q1_opt_state = param_update(self.q_optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.q_optim, q2_params, q2_grads, q2_opt_state)

            policy_params, policy_opt_state = delay_update_fn(
                self.policy_optim, policy_params, policy_grads, policy_opt_state
            )
            log_alpha, log_alpha_opt_state = delay_alpha_update_fn(
                self.alpha_optim, log_alpha, log_alpha_opt_state
            )

            target_q1 = target_update_fn(q1_params, target_q1, self.tau)
            target_q2 = target_update_fn(q2_params, target_q2, self.tau)
            target_policy = target_update_fn(policy_params, target_policy, self.tau)

            q_mean = jnp.mean(q_sa)
            q_std = jnp.std(q_sa)
            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            r_flat_for_hist = jnp.reshape(r_beta_kb, (-1,))

            new_state = DoobPOTrainState(
                params=DoobPOParams(
                    q1=q1_params,
                    q2=q2_params,
                    target_q1=target_q1,
                    target_q2=target_q2,
                    policy=policy_params,
                    target_policy=target_policy,
                    ratio=ratio_params_new,
                    log_alpha=log_alpha,
                ),
                opt_state=DoobPOOptStates(
                    q1=q1_opt_state,
                    q2=q2_opt_state,
                    policy=policy_opt_state,
                    ratio=ratio_opt_state_new,
                    log_alpha=log_alpha_opt_state,
                ),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std,
            )

            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1_mean": jnp.mean(q1_val),
                "q1_max": jnp.max(q1_val),
                "q1_min": jnp.min(q1_val),
                "policy_loss": policy_loss_val,
                "ratio_loss": ratio_loss_val,
                "r_beta_mean": jnp.mean(r_beta_kb),
                "r_beta_std": jnp.std(r_beta_kb),
                "r_beta_min": jnp.min(r_beta_kb),
                "r_beta_max": jnp.max(r_beta_kb),
                "advantage_mean": jnp.mean(adv_grpo),
                "advantage_std": jnp.std(adv_grpo),
                "alpha": jnp.exp(log_alpha),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                "hist_r_beta": r_flat_for_hist,
            }
            return new_state, info

        self._implement_common_behavior(
            stateless_update,
            self.agent.get_action,
            self.agent.get_deterministic_action,
            stateless_get_value=self.agent.q,
        )

    def get_policy_params(self):
        p = self.state.params
        return (p.policy, p.log_alpha, p.q1, p.q2)

    def get_policy_params_to_save(self):
        p = self.state.params
        return (p.target_policy, p.log_alpha, p.q1, p.q2)

    def get_value_params(self):
        return self.state.params.q1

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
