"""
DoobPO-Diffusion (off-policy) fine-tuning agent for robomimic.

Port of relax/algorithm/doobpo.py into the ReinFlow/DPPO training loop. Structure
follows agent/finetune/diffusion_baselines/train_qsm_diffusion_agent.py (FIFO replay
buffer, twin-Q critic, off-policy updates), but each replay minibatch runs the DoobPO
update: critic TD -> advantage -> ratio-net PPO-clip -> ratio-weighted drift matching
-> Polyak target updates. See model/diffusion/diffusion_doobpo.py.
"""

import os
import pickle
import logging
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import einops
import wandb

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_agent import TrainAgent
from util.scheduler import CosineAnnealingWarmupRestarts


class TrainDoobPODiffusionAgent(TrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.gamma = cfg.train.gamma
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # optimizers: actor (diffusion) / critic (twin Q) / ratio net
        optimizer_type = cfg.train.get("optimizer_type", "adamw").lower()
        if optimizer_type not in {"adam", "adamw"}:
            raise ValueError(f"Unknown train.optimizer_type={optimizer_type!r}")
        optimizer_cls = torch.optim.Adam if optimizer_type == "adam" else torch.optim.AdamW
        self.actor_optimizer = optimizer_cls(
            self.model.actor.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_optimizer = optimizer_cls(
            self.model.critic_q.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.ratio_optimizer = optimizer_cls(
            self.model.ratio_net.parameters(),
            lr=cfg.train.ratio_lr,
            weight_decay=cfg.train.get("ratio_weight_decay", 0.0),
        )
        self.alpha_optimizer = None
        if hasattr(self.model, "log_alpha"):
            self.alpha_optimizer = torch.optim.Adam(
                [self.model.log_alpha], lr=cfg.train.get("alpha_lr", 7e-3)
            )
        self.alpha_update_freq = cfg.train.get("alpha_update_freq", 250)
        self.freeze_actor = cfg.train.get("freeze_actor", False)
        self.freeze_ratio = cfg.train.get("freeze_ratio", False)
        self.use_lr_scheduler = cfg.train.get("use_lr_scheduler", True)
        self.positive_replay_fraction = cfg.train.get("positive_replay_fraction", 0.0)
        if not 0.0 <= self.positive_replay_fraction <= 1.0:
            raise ValueError("train.positive_replay_fraction must be in [0, 1]")

        # buffer / update schedule
        self.buffer_size = cfg.train.buffer_size
        self.scale_reward_factor = cfg.train.scale_reward_factor
        self.replay_ratio = cfg.train.replay_ratio
        self.critic_tau = cfg.train.critic_tau
        self.policy_tau = cfg.train.get("policy_tau", self.critic_tau)
        # ratio-net gradient steps per policy step (default 1:1, cf. doobpo.py delay)
        self.ratio_updates_per_batch = cfg.train.get("ratio_updates_per_batch", 1)
        # Flow policies are substantially more sensitive to repeated weighted
        # flow-matching steps than diffusion policies.  Keep the default at 1
        # for backwards compatibility, but allow sweeps to decouple the critic
        # update rate from actor/ratio updates.
        self.policy_update_freq = cfg.train.get("policy_update_freq", 1)
        self.target_update_freq = cfg.train.get("target_update_freq", 1)
        if self.policy_update_freq < 1:
            raise ValueError("train.policy_update_freq must be >= 1")
        if self.target_update_freq < 1:
            raise ValueError("train.target_update_freq must be >= 1")
        if self.alpha_update_freq < 1:
            raise ValueError("train.alpha_update_freq must be >= 1")

    def run(self):
        # FIFO replay buffer
        obs_buffer = deque(maxlen=self.buffer_size)
        next_obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        terminated_buffer = deque(maxlen=self.buffer_size)

        timer = Timer()
        run_results = []
        cnt_train_step = 0
        gradient_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:

            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # ----- rollout -----
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            for step in range(self.n_steps):
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                    }
                    samples = self.model(cond=cond, deterministic=eval_mode).cpu().numpy()
                action_venv = samples[:, : self.act_steps]

                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv

                if not eval_mode:
                    obs_venv_copy = obs_venv.copy()
                    for i in range(self.n_envs):
                        if truncated_venv[i]:
                            obs_venv_copy["state"][i] = info_venv[i]["final_obs"]["state"]
                    obs_buffer.append(prev_obs_venv["state"])
                    next_obs_buffer.append(obs_venv_copy["state"])
                    action_buffer.append(action_venv)
                    reward_buffer.append(reward_venv * self.scale_reward_factor)
                    terminated_buffer.append(terminated_venv)

                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # ----- episode stats (same as QSM/DPPO) -----
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start, end = env_steps[i], env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array([np.sum(r) for r in reward_trajs_split])
                # MultiStep already sums primitive rewards within each action chunk.
                episode_best_reward = np.array([np.max(r) for r in reward_trajs_split])
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                num_episode_finished = 0
                avg_episode_reward = avg_best_reward = success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # ----- DoobPO off-policy updates -----
            if not eval_mode:
                num_batch = int(
                    self.n_steps * self.n_envs / self.batch_size * self.replay_ratio
                )
                obs_trajs = einops.rearrange(np.array(deepcopy(obs_buffer)), "s e h d -> (s e) h d")
                next_obs_trajs = einops.rearrange(np.array(deepcopy(next_obs_buffer)), "s e h d -> (s e) h d")
                action_trajs = einops.rearrange(np.array(deepcopy(action_buffer)), "s e h d -> (s e) h d")
                reward_trajs_flat = np.array(deepcopy(reward_buffer)).reshape(-1)
                terminated_trajs = np.array(deepcopy(terminated_buffer)).reshape(-1)

                loss_actor = loss_critic = loss_ratio = torch.tensor(0.0)
                for batch_idx in range(num_batch):
                    if self.positive_replay_fraction > 0:
                        positive = np.flatnonzero(reward_trajs_flat > 0)
                        nonpositive = np.flatnonzero(reward_trajs_flat <= 0)
                        n_positive = min(
                            int(self.batch_size * self.positive_replay_fraction),
                            self.batch_size if len(positive) else 0,
                        )
                        n_other = self.batch_size - n_positive
                        positive_inds = (
                            np.random.choice(positive, n_positive) if n_positive else np.empty(0, dtype=int)
                        )
                        other_pool = nonpositive if len(nonpositive) else np.arange(len(obs_trajs))
                        inds = np.concatenate([
                            positive_inds,
                            np.random.choice(other_pool, n_other),
                        ])
                        np.random.shuffle(inds)
                    else:
                        inds = np.random.choice(len(obs_trajs), self.batch_size)
                    obs_b = {"state": torch.from_numpy(obs_trajs[inds]).float().to(self.device)}
                    next_obs_b = {"state": torch.from_numpy(next_obs_trajs[inds]).float().to(self.device)}
                    actions_b = torch.from_numpy(action_trajs[inds]).float().to(self.device)
                    rewards_b = torch.from_numpy(reward_trajs_flat[inds]).float().to(self.device)
                    terminated_b = torch.from_numpy(terminated_trajs[inds]).float().to(self.device)

                    # 1. critic (twin Q TD)
                    loss_critic = self.model.loss_critic(
                        obs_b, next_obs_b, actions_b, rewards_b, terminated_b, self.gamma
                    )
                    self.critic_optimizer.zero_grad()
                    loss_critic.backward()
                    self.critic_optimizer.step()

                    # During critic warmup, keep both the ratio network and actor
                    # frozen. Updating the ratio from an untrained Q/advantage
                    # contaminates the first post-warmup policy updates.
                    if (
                        self.itr >= self.n_critic_warmup_itr
                        and gradient_step % self.policy_update_freq == 0
                        and not (self.freeze_actor and self.freeze_ratio)
                    ):
                        # 2. advantage from (target) twin Q
                        adv = self.model.compute_advantage(obs_b, actions_b)

                        # 3. ratio net (PPO-clip surrogate + regularizer)
                        if not self.freeze_ratio:
                            for _ in range(self.ratio_updates_per_batch):
                                loss_ratio, _ = self.model.loss_ratio(obs_b, actions_b, adv)
                                self.ratio_optimizer.zero_grad()
                                loss_ratio.backward()
                                self.ratio_optimizer.step()

                        # 4. ratio-reweighted policy matching after critic warmup
                        if not self.freeze_actor:
                            loss_actor = self.model.loss_actor(obs_b, actions_b)
                            self.actor_optimizer.zero_grad()
                            loss_actor.backward()
                            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.actor.parameters(),
                                self.max_grad_norm if self.max_grad_norm is not None else float("inf"),
                            )
                            diagnostics = getattr(self.model, "last_diagnostics", None)
                            if diagnostics is not None:
                                diagnostics["actor_grad_norm"] = float(
                                    actor_grad_norm.detach()
                                )
                            self.actor_optimizer.step()

                    # 5. Polyak target updates
                    if gradient_step % self.target_update_freq == 0:
                        self.model.update_target_critic(self.critic_tau)
                        self.model.update_target_policy(self.policy_tau)

                    if (
                        self.alpha_optimizer is not None
                        and gradient_step % self.alpha_update_freq == 0
                    ):
                        loss_alpha = self.model.loss_alpha()
                        self.alpha_optimizer.zero_grad()
                        loss_alpha.backward()
                        self.alpha_optimizer.step()
                    gradient_step += 1

            if self.use_lr_scheduler:
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()

            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    diagnostics = getattr(self.model, "last_diagnostics", {})
                    diagnostic_text = " ".join(
                        f"{key} {value:.4f}" for key, value in diagnostics.items()
                    )
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss actor {loss_actor:8.4f} | loss critic {loss_critic:8.4f} | loss ratio {loss_ratio:8.4f} | reward {avg_episode_reward:8.4f} | {diagnostic_text} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss - actor": loss_actor,
                                "loss - critic": loss_critic,
                                "loss - ratio": loss_ratio,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
