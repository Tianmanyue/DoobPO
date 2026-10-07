"""
DoobPO-Diffusion (off-policy): Likelihood-Free Generative Policy Optimization for
diffusion policies via the Doob h-transform with a learned PPO-clipped ratio net.

PyTorch port of relax/algorithm/doobpo.py (JAX) into the ReinFlow/DPPO codebase for
robomimic. Structure mirrors QSMDiffusion (twin-Q critic + target Q + replay buffer)
but the policy update is RWR-style ratio-reweighted drift matching (reuses
RWRDiffusion.p_losses), and the weights come from a learned ratio net r_beta(s,a).

Pipeline per off-policy update on a replay minibatch:
  1. critic: twin-Q TD target  r + gamma * min_i targetQ_i(s', a'),  a'~pi        (loss_critic)
  2. advantage: A(s,a) = min_i targetQ_i(s,a) - min_i targetQ_i(s, a'~pi), normalized (compute_advantage)
  3. ratio net: r_beta = exp(MLP(s,a)); PPO-clip surrogate on A + unbiased
     double-sample regularizer  lambda*(E[r_beta]-1)(E[r_beta']-1)                  (loss_ratio)
  4. policy: weight = clip(r_beta/mean(r_beta), 0, max_ratio_weight).detach();
     weighted denoising (drift matching) on buffer actions                          (loss_actor)
  5. Polyak update target Q (and target policy for a' sampling)

Weights follow doobpo.py: DoobPO -> r_beta (learned, PPO-clipped);  DPMD -> exp(Q/alpha).
"""

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from model.diffusion.diffusion_rwr import RWRDiffusion

log = logging.getLogger(__name__)


class RatioNet(nn.Module):
    """r_beta(s, a) = exp(clip(f_theta(s, a))): (obs, act) -> R_+.

    Port of relax/network/doobpo.py:create_ratio_net. The head is zero-initialized so
    the network outputs exp(0)=1 at init -- i.e. starts at the "no update" reference.
    Input is the flattened conditioning state concatenated with the flattened action.
    """

    def __init__(self, obs_dim, action_dim, cond_steps, horizon_steps,
                 hidden_sizes=(256, 256, 256), logit_clip=10.0,
                 activation_type="ReLU"):
        super().__init__()
        self.cond_dim = obs_dim * cond_steps
        self.act_dim = action_dim * horizon_steps
        self.logit_clip = logit_clip
        try:
            activation_cls = getattr(nn, activation_type)
        except AttributeError as exc:
            raise ValueError(f"Unknown ratio activation {activation_type!r}") from exc
        layers = []
        last = self.cond_dim + self.act_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), activation_cls()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(last, 1)
        # zero-init head -> logit 0 -> r_beta = 1 at init
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, cond, action):
        # cond: {"state": (B, cond_steps, obs_dim)}; action: (B, horizon_steps, act_dim)
        state = cond["state"] if isinstance(cond, dict) else cond
        x = torch.cat(
            [state.reshape(state.shape[0], -1), action.reshape(action.shape[0], -1)],
            dim=-1,
        )
        logit = self.head(self.trunk(x))[..., 0]
        logit = torch.clamp(logit, -self.logit_clip, self.logit_clip)
        return torch.exp(logit)  # (B,) > 0


class DoobPODiffusion(RWRDiffusion):

    def __init__(
        self,
        actor,
        critic,
        ratio_net,
        ppo_eps=0.2,
        max_ratio_weight=5.0,
        ratio_reg_lambda=0.01,
        bc_anchor_coef=0.0,
        adv_norm=True,
        use_target_policy=True,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        assert not self.use_ddim, "DoobPO drift-matching uses the DDPM denoising loss"
        self.critic_q = critic.to(self.device)
        self.target_q = copy.deepcopy(critic)
        self.ratio_net = ratio_net.to(self.device)
        self.actor = self.network

        self.ppo_eps = ppo_eps
        self.max_ratio_weight = max_ratio_weight
        self.ratio_reg_lambda = ratio_reg_lambda
        self.bc_anchor_coef = bc_anchor_coef
        self.adv_norm = adv_norm

        # Polyak target policy used to sample a' for the advantage baseline (doobpo.py).
        self.use_target_policy = use_target_policy
        if use_target_policy:
            self.target_actor = copy.deepcopy(self.network)
        self.base_actor = None
        if self.bc_anchor_coef > 0:
            self.base_actor = copy.deepcopy(self.network).eval()
            for param in self.base_actor.parameters():
                param.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # sampling helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _sample_actions(self, cond, use_target):
        """Sample a fresh action from the (target) policy for advantage / regularizer."""
        if use_target and self.use_target_policy:
            orig = self.network
            self.network = self.target_actor
            try:
                a = self.forward(cond=cond, deterministic=False)
            finally:
                self.network = orig
            return a
        return self.forward(cond=cond, deterministic=False)

    # ------------------------------------------------------------------ #
    # 1. critic: twin-Q TD (identical to QSM / doobpo.py q_loss)
    # ------------------------------------------------------------------ #
    def loss_critic(self, obs, next_obs, actions, rewards, terminated, gamma):
        current_q1, current_q2 = self.critic_q(obs, actions)
        next_actions = self._sample_actions(next_obs, use_target=True)
        with torch.no_grad():
            next_q1, next_q2 = self.target_q(next_obs, next_actions)
        next_q = torch.min(next_q1, next_q2)

        mask = 1 - terminated
        rewards = rewards.view(-1)
        next_q = next_q.view(-1)
        mask = mask.view(-1)
        target_q = rewards + gamma * next_q * mask

        loss_critic = torch.mean((current_q1.view(-1) - target_q) ** 2) + torch.mean(
            (current_q2.view(-1) - target_q) ** 2
        )
        return loss_critic

    # ------------------------------------------------------------------ #
    # 2. advantage: A(s,a) = minQ(s,a) - minQ(s, a'~pi), normalized
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_advantage(self, obs, actions):
        q1, q2 = self.target_q(obs, actions)
        q_sa = torch.min(q1, q2).view(-1)
        v_action = self._sample_actions(obs, use_target=True)
        vq1, vq2 = self.target_q(obs, v_action)
        v_s = torch.min(vq1, vq2).view(-1)
        adv = q_sa - v_s
        if self.adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv

    # ------------------------------------------------------------------ #
    # 3. ratio net: PPO-clip surrogate + unbiased double-sample regularizer
    # ------------------------------------------------------------------ #
    def loss_ratio(self, obs, actions, adv_norm):
        adv_sg = adv_norm.detach()
        r_beta = self.ratio_net(obs, actions)  # (B,) buffer actions
        r_clip = torch.clamp(r_beta, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
        ppo_obj = torch.minimum(r_beta * adv_sg, r_clip * adv_sg)

        # unbiased ratio regularizer via double sampling (per paper):
        #   g := E_{a'~pi}[r_beta(s,a')] - 1 ; use g_hat1 * g_hat2 (independent samples)
        action_prime = self._sample_actions(obs, use_target=True)
        r_beta_prime = self.ratio_net(obs, action_prime)
        g_hat_1 = r_beta.mean() - 1.0
        g_hat_2 = r_beta_prime.mean() - 1.0
        regularizer = self.ratio_reg_lambda * g_hat_1 * g_hat_2

        loss = -ppo_obj.mean() + regularizer
        return loss, r_beta.detach().mean()

    # ------------------------------------------------------------------ #
    # 4. policy: ratio-reweighted drift matching (RWR weighted denoising)
    # ------------------------------------------------------------------ #
    def loss_actor(self, obs, actions):
        with torch.no_grad():
            r_raw = self.ratio_net(obs, actions)             # (B,)
            r_norm = r_raw / (r_raw.mean() + 1e-8)            # engineering fix: normalize by batch mean
            weight = torch.clamp(r_norm, 0.0, self.max_ratio_weight)  # engineering fix: hard cap
        B = actions.shape[0]
        t = torch.randint(0, self.denoising_steps, (B,), device=actions.device).long()
        noise = torch.randn_like(actions)
        x_noisy = self.q_sample(x_start=actions, t=t, noise=noise)
        prediction = self.network(x_noisy, t, cond=obs)
        target = noise if self.predict_epsilon else actions
        per_sample = einops.reduce(
            F.mse_loss(prediction, target, reduction="none"), "b h d -> b", "mean"
        )
        loss = (weight * per_sample).mean()
        if self.base_actor is not None:
            with torch.no_grad():
                base_prediction = self.base_actor(x_noisy, t, cond=obs)
            loss = loss + self.bc_anchor_coef * F.mse_loss(
                prediction, base_prediction
            )
        return loss

    # ------------------------------------------------------------------ #
    # 5. Polyak target updates
    # ------------------------------------------------------------------ #
    def update_target_critic(self, tau):
        for tp, sp in zip(self.target_q.parameters(), self.critic_q.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)

    def update_target_policy(self, tau):
        if not self.use_target_policy:
            return
        for tp, sp in zip(self.target_actor.parameters(), self.network.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)
