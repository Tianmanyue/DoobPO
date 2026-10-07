"""
DoobPO: Learning via Doob h-transform (off-policy implementation)

Architecture (Algorithm 1 from the paper):
  1. Q-network update: standard Bellman backup (single sample).
  2. Advantage computation: A(s,a) = Q(s,a) - V(s), where V(s) is estimated by
     sampling a single action from the current policy.
  3. Ratio-network update: train r_β(s,a) with a PPO clipped-surrogate loss
     so that r_β approximates the density ratio needed for the Doob h-transform
     while staying close to 1 (the "no-update" reference).
  4. Policy update: ratio-reweighted drift matching on replay-buffer (s, a) pairs,
     with weights stop_gradient(r_β(s,a)).
  5. Alpha (entropy-temperature) update: same as DPMD.
  6. Target-network soft updates: same as DPMD.

Key difference from DPMD:
  DPMD  → weights = exp(normalised_Q / alpha)  (direct advantage exponentiation)
  DoobPO → weights = r_β (learned ratio net, PPO-clipped for stability)
"""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.diffv2 import Diffv2Net, Diffv2Params
from relax.utils.experience import Experience, GAEExperience
from relax.utils.typing_utils import Metric


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------

class DoobPOParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_policy: hk.Params
    ratio: hk.Params          # ratio-network parameters
    log_alpha: jax.Array


class DoobPOOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    ratio: optax.OptState     # ratio-network optimiser state
    log_alpha: optax.OptState


class DoobPOTrainState(NamedTuple):
    params: DoobPOParams
    opt_state: DoobPOOptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float


# ---------------------------------------------------------------------------
# Main algorithm class
# ---------------------------------------------------------------------------

class DoobPO(Algorithm):
    """Off-policy diffusion RL via Doob h-transform with a learned ratio net."""

    def __init__(
        self,
        agent: Diffv2Net,
        params: Diffv2Params,
        ratio_net,          # hk.Transformed (without_apply_rng) for the ratio network
        ratio_params,       # initial haiku params for the ratio network
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
        ppo_eps: float = 0.2,        # PPO clipping coefficient ε
        max_ratio_weight: float = 5.0,  # upper cap for r_β used in drift-matching weights
        ratio_regularizer_lambda: float = 0.01,  # λ for ratio network regularizer: -λ (E[r_β] - 1)^2
        gae_lambda: float = 0.95,  # for on-policy: GAE advantage (when data has .adv)
        use_ema: bool = True,
    ):
        self.agent = agent
        self.ratio_net = ratio_net
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.ppo_eps = ppo_eps
        self.max_ratio_weight = max_ratio_weight
        self.ratio_regularizer_lambda = ratio_regularizer_lambda

        # V(s) = min_Q(s, deterministic_action(s)) for on-policy GAE
        agent_ref = self.agent
        def _min_q_fn(q1, q2, o, a):
            return jnp.minimum(agent_ref.q(q1, o, a), agent_ref.q(q2, o, a))
        self._get_min_q = jax.jit(_min_q_fn)

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

        # Build combined parameter / opt-state objects
        doobpo_params = DoobPOParams(
            q1=params.q1,
            q2=params.q2,
            target_q1=params.target_q1,
            target_q2=params.target_q2,
            policy=params.policy,
            target_policy=params.target_poicy,   # note: typo in Diffv2Params
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

        # ---------------------------------------------------------------
        # JIT-compiled stateless update
        # ---------------------------------------------------------------
        ppo_eps_const = ppo_eps
        max_ratio_weight_const = max_ratio_weight
        ratio_reg_lambda = self.ratio_regularizer_lambda

        @jax.jit
        def stateless_update(
            key: jax.Array,
            state: DoobPOTrainState,
            data: Experience,
            use_precomputed_adv: bool,
            precomputed_adv: jax.Array,
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

            (next_eval_key, diffusion_t_key,
             diffusion_noise_key, log_alpha_key,
             v_estimation_key, ratio_reg_key) = jax.random.split(key, 6)

            # ----------------------------------------------------------
            # Helpers
            # ----------------------------------------------------------
            def min_q(p1, p2, s, a):
                return jnp.minimum(self.agent.q(p1, s, a), self.agent.q(p2, s, a))

            # ----------------------------------------------------------
            # 1. Q-network update (Bellman backup)
            # ----------------------------------------------------------
            # Sample single next_action and compute Q-target
            next_action = self.agent.get_action(
                next_eval_key,
                (policy_params, log_alpha, q1_params, q2_params),
                next_obs,
            )
            q_target = jax.lax.stop_gradient(
                min_q(target_q1, target_q2, next_obs, next_action)
            )  # (B,)
            q_backup = reward + (1.0 - done) * self.gamma * q_target

            def q_loss_fn(qp):
                q_val = self.agent.q(qp, obs, action)
                return jnp.mean((q_val - q_backup) ** 2), q_val

            (q1_loss, q1_val), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2_val), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)

            # ----------------------------------------------------------
            # 2. Advantage: precomputed A^{π^k} (on-policy GAE) or Q(s,a)-V(s) (off-policy)
            # ----------------------------------------------------------
            q_sa = jax.lax.stop_gradient(
                min_q(target_q1, target_q2, obs, action)
            )  # (B,) for logging

            def adv_from_qv(v_key):
                v_action = self.agent.get_action(
                    v_key,
                    (policy_params, log_alpha, q1_params, q2_params),
                    obs,
                )
                v_s = jax.lax.stop_gradient(
                    min_q(target_q1, target_q2, obs, v_action)
                )
                adv = q_sa - v_s
                m, s = adv.mean(), adv.std() + 1e-8
                return (adv - m) / s, m, s

            def use_precomputed(adv_arr):
                m = adv_arr.mean()
                s = adv_arr.std() + 1e-8
                return (adv_arr - m) / s, m, s

            adv_norm, adv_mean, adv_std = jax.lax.cond(
                use_precomputed_adv,
                lambda: use_precomputed(precomputed_adv),
                lambda: adv_from_qv(v_estimation_key),
            )

            # ----------------------------------------------------------
            # 3. Ratio-network update (PPO clipped surrogate)
            # ----------------------------------------------------------
            adv_norm_sg = jax.lax.stop_gradient(adv_norm)

            # Double sampling for ratio regularizer:
            # Sample 2: independent actions from π_old (stop_grad, not differentiating policy)
            action_prime = jax.lax.stop_gradient(
                self.agent.get_action(
                    ratio_reg_key,
                    (policy_params, log_alpha, q1_params, q2_params),
                    obs,
                )
            )  # (B, act_dim)

            def ratio_loss_fn(rp):
                # r_β > 0 by construction (exp output from the network)
                r_beta = self.ratio_net.apply(rp, obs, action)  # (B,) — sample 1: buffer actions
                r_clip = jnp.clip(r_beta, 1.0 - ppo_eps_const, 1.0 + ppo_eps_const)
                ppo_obj = jnp.minimum(r_beta * adv_norm_sg, r_clip * adv_norm_sg)

                # Unbiased ratio regularizer via double sampling (per paper):
                #   g := E_{a'~π_old}[r_β(s,a')] - 1
                #   naive (r_beta.mean()-1)^2 is biased; use g_hat1 * g_hat2 instead.
                #   Sample 1: buffer (s,a) → g_hat_1 = mean(r_beta) - 1
                #   Sample 2: newly sampled a'~π_old → g_hat_2 = mean(r_beta') - 1
                #   E[g_hat_1 * g_hat_2] = g^2  (unbiased, since samples are independent)
                r_beta_prime = self.ratio_net.apply(rp, obs, action_prime)  # (B,) — sample 2
                g_hat_1 = r_beta.mean() - 1.0
                g_hat_2 = r_beta_prime.mean() - 1.0
                regularizer = ratio_reg_lambda * g_hat_1 * g_hat_2

                return -ppo_obj.mean() + regularizer, r_beta

            (ratio_loss_val, r_beta_val), ratio_grads = jax.value_and_grad(
                ratio_loss_fn, has_aux=True
            )(ratio_params)

            # ----------------------------------------------------------
            # 4. Policy update: ratio-reweighted drift matching
            #    Use buffer (obs, action) pairs; r_β as stop-grad weights.
            #
            #  Engineering fix 3: normalise r_β by its batch mean so the
            #  effective policy learning-rate stays stable as r_β drifts.
            #
            #  Engineering fix 4: hard-cap normalised weights at
            #  max_ratio_weight to prevent catastrophic policy updates from
            #  outlier (s,a) pairs with extremely high ratios.
            # ----------------------------------------------------------
            ratio_update, ratio_opt_state_new = self.ratio_optim.update(ratio_grads, ratio_opt_state)
            ratio_params_new = optax.apply_updates(ratio_params, ratio_update)

            def policy_loss_fn(pp):
                r_raw = jax.lax.stop_gradient(
                    self.ratio_net.apply(ratio_params_new, obs, action)
                )  # (B,)
                # Normalise then cap
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

            # ----------------------------------------------------------
            # 5. Alpha (entropy temperature) update
            # ----------------------------------------------------------
            def log_alpha_loss_fn(la):
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log(
                    2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(la)) ** 2
                )
                return -la * (-jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)

            # ----------------------------------------------------------
            # 6. Apply gradient updates
            # ----------------------------------------------------------
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

            # Running stats for Q values (for logging)
            q_mean = jnp.mean(q_sa)
            q_std = jnp.std(q_sa)
            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

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
                "r_beta_mean": jnp.mean(r_beta_val),
                "r_beta_std": jnp.std(r_beta_val),
                "r_beta_min": jnp.min(r_beta_val),
                "r_beta_max": jnp.max(r_beta_val),
                "advantage_mean": adv_mean,
                "advantage_std": adv_std,
                "alpha": jnp.exp(log_alpha),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                # 以 hist_ 开头 → base.py 会将其分流到 dist_info，
                # 用于可视化 ratio net 的输出分布（一整个 batch 的原始值）
                "hist_r_beta": r_beta_val,
            }
            return new_state, info

        # ---------------------------------------------------------------
        # Register with base class machinery
        # ---------------------------------------------------------------
        self._implement_common_behavior(
            stateless_update,
            self.agent.get_action,
            self.agent.get_deterministic_action,
            stateless_get_value=self.agent.q,
        )

    # ---------------------------------------------------------------
    # Param accessors (required by base class / persistence)
    # ---------------------------------------------------------------
    def get_policy_params(self):
        p = self.state.params
        return (p.policy, p.log_alpha, p.q1, p.q2)

    def get_policy_params_to_save(self):
        p = self.state.params
        return (p.target_policy, p.log_alpha, p.q1, p.q2)

    def get_value_params(self):
        return self.state.params.q1

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        """V(s) = min_Q(s, deterministic_action(s)) for on-policy GAE."""
        action = self.get_deterministic_action(obs)
        p = self.state.params
        out = self._get_min_q(p.target_q1, p.target_q2, obs, action)
        return np.asarray(out)

    def update(self, key: jax.Array, data: Experience) -> Tuple[dict, dict]:
        """Support on-policy: when data has .adv (GAEExperience), use it for ratio PPO."""
        obs, action = data.obs, data.action
        use_precomputed_adv = hasattr(data, "adv") and getattr(data, "adv") is not None
        precomputed_adv = (
            jnp.array(data.adv) if use_precomputed_adv
            else jnp.zeros(obs.shape[0], dtype=jnp.float32)
        )
        self.state, info = self._update(
            key, self.state, data, use_precomputed_adv, precomputed_adv
        )
        return (
            {k: float(v) for k, v in info.items() if not k.startswith("hist")},
            {k: v for k, v in info.items() if k.startswith("hist")},
        )

    def warmup(self, data: Experience) -> None:
        key = jax.random.key(0)
        obs = data.obs[0]
        policy_params = self.get_policy_params()
        use_precomputed_adv = hasattr(data, "adv") and getattr(data, "adv") is not None
        precomputed_adv = (
            jnp.array(data.adv) if use_precomputed_adv
            else jnp.zeros(data.obs.shape[0], dtype=jnp.float32)
        )
        self.state, _ = self._update(
            key, self.state, data, use_precomputed_adv, precomputed_adv
        )
        self._get_action(key, policy_params, obs)
        self._get_deterministic_action(policy_params, obs)

    # ---------------------------------------------------------------
    # Override save_policy to use target policy
    # ---------------------------------------------------------------
    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
