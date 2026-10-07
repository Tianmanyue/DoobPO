import argparse
import os
import os.path
import sys
import time
from functools import partial
from pathlib import Path

# Make `relax` resolve from this repo when the script is invoked as
# `python scripts/train_mujoco.py`.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import yaml

import jax, jax.numpy as jnp

from relax.algorithm.sac import SAC
from relax.algorithm.dsact import DSACT
from relax.algorithm.dacer import DACER
from relax.algorithm.dacer_doubleq import DACERDoubleQ
from relax.algorithm.qsm import QSM
from relax.algorithm.dipo import DIPO
from relax.algorithm.qvpo import QVPO
from relax.algorithm.sdac import SDAC
from relax.algorithm.dpmd import DPMD
from relax.algorithm.idem import IDEM
from relax.algorithm.doobpo import DoobPO
from relax.algorithm.doobpo_grpo import DoobPOGRPO
from relax.algorithm.doobpo_flow import DoobPOFlow
from relax.algorithm.doobpo_flow_grpo import DoobPOFlowGRPO
from relax.algorithm.fpmd import FPMD
from relax.algorithm.fpo_flow import FPOFlow
from relax.algorithm.reinflow import ReinFlow
from relax.algorithm.pirl import PiRL
import haiku as hk
from relax.network.blocks import QNet
from relax.network.doobpo import create_doobpo_net
from relax.network.doobpo_flow import create_doobpo_flow_net
from relax.network.flow_ppo import create_flow_ppo_net
from relax.network.reinflow import create_explore_noise_net
from relax.buffer import TreeBuffer
from relax.network.sac import create_sac_net
from relax.network.dsact import create_dsact_net
from relax.network.dacer import create_dacer_net
from relax.network.dacer_doubleq import create_dacer_doubleq_net
from relax.network.qsm import create_qsm_net
from relax.network.dipo import create_dipo_net
from relax.network.diffv2 import create_diffv2_net
from relax.network.qvpo import create_qvpo_net
from relax.trainer.off_policy import OffPolicyTrainer
from relax.trainer.on_policy import OnPolicyTrainer
from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience, GAEExperience, ObsActionPair
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.task_hparams import get_task_hparams, REINFLOW_HPARAMS, FPO_HPARAMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="sdac")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--suffix", type=str, default="test_use_atp1")
    parser.add_argument("--num_vec_envs", type=int, default=5)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(3e4)) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=7e-3)
    parser.add_argument("--delay_alpha_update", type=float, default=250)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--beta_schedule_scale", type=float, default=0.8)
    parser.add_argument("--beta_schedule_type", type=str, default='linear')
    # Trainer
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--utd", type=int, default=1, help="update-to-data ratio (update_per_iteration)")
    # DoobPO-specific
    parser.add_argument("--ppo_eps", type=float, default=0.2)
    parser.add_argument("--ratio_lr", type=float, default=3e-4)
    parser.add_argument("--reward_scale", type=float, default=0.2)
    parser.add_argument("--max_ratio_weight", type=float, default=5.0)
    parser.add_argument("--ratio_regularizer_lambda", type=float, default=0.01,
                        help="DoobPO: λ for ratio network regularizer -λ (E[r_β] - 1)^2")
    parser.add_argument("--lr_schedule_steps", type=int, default=50000,
                        help="gradient steps over which policy lr decays from lr to lr_schedule_end")
    parser.add_argument("--n_epochs", type=int, default=4,
                        help="PPO epochs per rollout batch (on-policy flow baselines only)")
    # DoobGRPO (diffusion)
    parser.add_argument("--grpo_group_size", type=int, default=8,
                        help="DoobGRPO: number of action samples per state for group-relative advantage")
    # ---- Flow-matching methods (all _flow suffix, won't conflict with diffusion args) ----
    parser.add_argument("--flow_steps", type=int, default=20,
                        help="Flow: ODE Euler integration steps")
    parser.add_argument("--lr_flow", type=float, default=1e-4,
                        help="Flow: policy & Q learning rate")
    parser.add_argument("--num_minibatches_flow", type=int, default=1,
                        help="Flow: number of minibatches per epoch (1=full batch)")
    parser.add_argument("--ppo_eps_flow", type=float, default=0.1,
                        help="Flow: PPO clipping epsilon (unified: 0.1)")
    parser.add_argument("--reward_scale_flow", type=float, default=1.0,
                        help="Flow: reward scaling (default 1.0)")
    parser.add_argument("--n_epochs_flow", type=int, default=4,
                        help="Flow: PPO epochs per rollout batch (FPO paper: 16)")
    parser.add_argument("--num_particles_flow", type=int, default=32,
                        help="Flow: number of action particles for best-of-N")
    parser.add_argument("--noise_scale_flow", type=float, default=0.1,
                        help="Flow: exploration noise scale")
    parser.add_argument("--gamma_flow", type=float, default=0.99,
                        help="Flow: discount factor (unified: 0.99)")
    # DoobPO-Flow specific
    parser.add_argument("--ratio_lr_flow", type=float, default=3e-4,
                        help="DoobPO-Flow: ratio network learning rate")
    parser.add_argument("--max_ratio_weight_flow", type=float, default=5.0,
                        help="DoobPO-Flow: max ratio clipping for drift matching")
    parser.add_argument("--ratio_regularizer_lambda_flow", type=float, default=0.01,
                        help="DoobPO-Flow: ratio regularizer lambda")
    parser.add_argument("--ratio_updates_per_batch_flow", type=int, default=1,
                        help="DoobPO-Flow: ratio-net gradient steps per replay batch (off-policy; vs policy_updates_per_batch_flow)")
    parser.add_argument("--policy_updates_per_batch_flow", type=int, default=1,
                        help="DoobPO-Flow: policy (flow) gradient steps per replay batch (off-policy)")
    parser.add_argument("--delay_q_target_update_flow", type=int, default=2,
                        help="DoobPO-Flow off-policy: Polyak-update target Q1/Q2 every N algorithm steps "
                             "(same idea as diffusion DoobPO delay_update for targets; 1=every step)")
    parser.add_argument("--alpha_lr_flow", type=float, default=7e-3,
                        help="DoobPO-Flow: alpha (entropy temperature) learning rate")
    parser.add_argument("--delay_alpha_update_flow", type=int, default=250,
                        help="DoobPO-Flow: update alpha every N algorithm steps")
    parser.add_argument("--target_entropy_scale_flow", type=float, default=0.9,
                        help="DoobPO-Flow: target_entropy = -act_dim * scale")
    # DoobPO-Flow-GRPO specific
    parser.add_argument("--num_grpo_samples", type=int, default=16,
                        help="DoobPO-Flow-GRPO: number of actions sampled per state for group-relative advantage")
    parser.add_argument(
        "--ratio_net_type",
        type=str,
        default="mlp",
        choices=("mlp", "resnet"),
        help="DoobPO-Flow / GRPO-Flow: ratio backbone (mlp | resnet residual trunk).",
    )
    parser.add_argument(
        "--ratio_mlp_hidden_sizes",
        type=str,
        default="",
        help="DoobPO-Flow: comma-separated MLP ratio hidden widths (e.g. 256,256,256). "
             "Empty: use value net widths [hidden_dim]*hidden_num.",
    )
    parser.add_argument(
        "--ratio_resnet_hidden_dim",
        type=int,
        default=256,
        help="DoobPO-Flow: ResNet ratio trunk width (all ResBlocks use this dim).",
    )
    parser.add_argument(
        "--ratio_resnet_num_blocks",
        type=int,
        default=1,
        help="DoobPO-Flow: number of residual blocks (1 block ≈ param budget of MLP 256×3).",
    )
    # FPO-Flow specific
    parser.add_argument("--n_cfm_samples_flow", type=int, default=8,
                        help="FPO-Flow: number of (eps,t) samples per action")
    # ReinFlow specific
    parser.add_argument("--noise_lr_flow", type=float, default=3e-5,
                        help="ReinFlow: ExploreNoiseNet learning rate (paper: same as actor)")
    parser.add_argument("--q_lr_flow", type=float, default=3e-4,
                        help="ReinFlow: Q-network learning rate (paper: critic_lr=3e-4)")
    parser.add_argument("--min_noise_std_flow", type=float, default=0.10,
                        help="ReinFlow: minimum noise sigma (paper: 0.10)")
    parser.add_argument("--max_noise_std_flow", type=float, default=0.24,
                        help="ReinFlow: maximum noise sigma (paper: 0.24)")
    parser.add_argument("--ent_coef_flow", type=float, default=0.0,
                        help="Flow: entropy bonus coefficient (unified: 0.0)")
    parser.add_argument("--logprob_min_flow", type=float, default=-1.0,
                        help="ReinFlow: log-prob clamping lower bound")
    parser.add_argument("--logprob_max_flow", type=float, default=1.0,
                        help="ReinFlow: log-prob clamping upper bound")
    parser.add_argument("--randn_clip_flow", type=float, default=3.0,
                        help="ReinFlow: randn clip for SDE noise (paper: 3)")
    parser.add_argument("--clip_intermediate_flow", type=float, default=1.0,
                        help="ReinFlow: clip denoised mean at each SDE step (paper: 1.0)")
    parser.add_argument("--normalize_logprob_flow", type=int, default=1,
                        help="ReinFlow: normalize log-prob by steps*act_dim (paper: 1)")
    # PiRL specific
    parser.add_argument("--noise_level_flow", type=float, default=0.5,
                        help="PiRL: analytic sigma coefficient")
    parser.add_argument("--min_sigma_flow", type=float, default=1e-3,
                        help="PiRL: minimum clamped sigma")
    parser.add_argument("--noise_anneal_flow", type=int, default=0,
                        help="PiRL: enable noise annealing (0=off, 1=on)")
    parser.add_argument("--noise_start_flow", type=float, default=0.7,
                        help="PiRL: noise annealing start value")
    parser.add_argument("--noise_end_flow", type=float, default=0.3,
                        help="PiRL: noise annealing end value")
    parser.add_argument("--noise_anneal_steps_flow", type=int, default=400,
                        help="PiRL: noise annealing steps")
    parser.add_argument("--ignore_last_flow", type=int, default=0,
                        help="PiRL: ignore last denoise step for noise injection (0=off)")
    # Action chunking (multi-step)
    parser.add_argument("--act_steps_flow", type=int, default=1,
                        help="Flow: action chunk size (original ReinFlow uses 4)")
    # Gradient clipping
    parser.add_argument("--max_grad_norm_flow", type=float, default=0.0,
                        help="Flow: max gradient norm (unified: 0.0=disabled)")
    # PPO stability clipping
    parser.add_argument("--adv_clip_flow", type=float, default=0.0,
                        help="Clip normalized advantages to [-c, c] (0=disabled)")
    parser.add_argument("--q_backup_clip_flow", type=float, default=0.0,
                        help="Clip Q-backup targets to [-c, c] (0=disabled)")
    parser.add_argument("--grpo_batch_norm_adv", action="store_true",
                        help="GRPO: use batch-level advantage normalization instead of per-state group normalization")
    parser.add_argument("--reinflow_legacy_hparams", action="store_true",
                        help="ReinFlow: use old DoobPO settings (5e-6 actor, target_kl=0.05, no running reward norm, ...). "
                             "Default: official ReinFlow ShortCut yaml per task family (Ant-v4 same as ant-v2, etc.).")
    args = parser.parse_args()

    _ratio_mlp_spec = (args.ratio_mlp_hidden_sizes or "").strip()
    ratio_hidden_sizes_kw = None
    if _ratio_mlp_spec:
        ratio_hidden_sizes_kw = tuple(
            int(x.strip()) for x in _ratio_mlp_spec.split(",") if x.strip()
        )

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    act_steps = 1
    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex", act_steps=act_steps)
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    gelu = partial(jax.nn.gelu, approximate=False)
    
    print(f"Algorithm: {args.alg}")

    if args.alg == 'sdac':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_diffv2_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps, 
                                          num_particles=args.num_particles, 
                                          noise_scale=args.noise_scale,
                                          beta_schedule_scale=args.beta_schedule_scale)
        algorithm = SDAC(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update, lr_schedule_end=args.lr_schedule_end)
    
    elif args.alg == 'dpmd':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_diffv2_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps, 
                                          num_particles=args.num_particles, 
                                          noise_scale=args.noise_scale,
                                          beta_schedule_scale=args.beta_schedule_scale)
        algorithm = DPMD(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update, lr_schedule_end=args.lr_schedule_end)

    elif args.alg == 'doobpo':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params, ratio_net, ratio_params = create_doobpo_net(
            init_network_key, obs_dim, act_dim,
            hidden_sizes, diffusion_hidden_sizes, mish,
            num_timesteps=args.diffusion_steps,
            num_particles=args.num_particles,
            noise_scale=args.noise_scale,
            beta_schedule_scale=args.beta_schedule_scale,
        )
        algorithm = DoobPO(
            agent, params, ratio_net, ratio_params,
            lr=args.lr,
            ratio_lr=args.ratio_lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
            reward_scale=args.reward_scale,
            ppo_eps=args.ppo_eps,
            max_ratio_weight=args.max_ratio_weight,
            ratio_regularizer_lambda=args.ratio_regularizer_lambda,
            lr_schedule_steps=args.lr_schedule_steps,
            gae_lambda=0.95,
        )

    elif args.alg == 'doobpo_grpo':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params, ratio_net, ratio_params = create_doobpo_net(
            init_network_key, obs_dim, act_dim,
            hidden_sizes, diffusion_hidden_sizes, mish,
            num_timesteps=args.diffusion_steps,
            num_particles=args.num_particles,
            noise_scale=args.noise_scale,
            beta_schedule_scale=args.beta_schedule_scale,
        )
        algorithm = DoobPOGRPO(
            agent, params, ratio_net, ratio_params,
            lr=args.lr,
            ratio_lr=args.ratio_lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
            reward_scale=args.reward_scale,
            grpo_group_size=args.grpo_group_size,
            max_ratio_weight=args.max_ratio_weight,
            ratio_regularizer_lambda=args.ratio_regularizer_lambda,
            lr_schedule_steps=args.lr_schedule_steps,
            ppo_eps=args.ppo_eps,
        )

    elif args.alg == 'doobpo_flow':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        (
            agent,
            params,
            ratio_net,
            ratio_params,
            q_apply,
            q1_p,
            q2_p,
            tq1_p,
            tq2_p,
        ) = create_doobpo_flow_net(
            init_network_key, obs_dim, act_dim,
            policy_hidden_sizes=diffusion_hidden_sizes,
            value_hidden_sizes=hidden_sizes,
            policy_activation=mish,
            value_activation=mish,
            num_steps=args.flow_steps,
            include_twin_q=True,
            ratio_net_type=args.ratio_net_type,
            ratio_hidden_sizes=ratio_hidden_sizes_kw,
            ratio_resnet_hidden_dim=args.ratio_resnet_hidden_dim,
            ratio_resnet_num_blocks=args.ratio_resnet_num_blocks,
        )
        # Off-policy: match diffusion DoobPO (--alg doobpo) lr / reward / PPO-ratio hyperparameters.
        algorithm = DoobPOFlow(
            agent, params, ratio_net, ratio_params,
            lr=args.lr,
            value_lr=args.lr,
            ratio_lr=args.ratio_lr,
            q_lr=args.lr,
            reward_scale=args.reward_scale,
            ppo_eps=args.ppo_eps,
            max_ratio_weight=args.max_ratio_weight,
            ratio_regularizer_lambda=args.ratio_regularizer_lambda,
            gamma=args.gamma_flow,
            gae_lambda=0.95,
            value_loss_coeff=0.5,
            n_epochs=args.n_epochs_flow,
            num_minibatches=args.num_minibatches_flow,
            max_grad_norm=args.max_grad_norm_flow,
            huber_delta=10.0,
            value_clip=0.2,
            value_updates_per_batch=10,
            ratio_updates_per_batch=args.ratio_updates_per_batch_flow,
            policy_updates_per_batch=args.policy_updates_per_batch_flow,
            off_policy=True,
            q_apply=q_apply,
            q1_params=q1_p,
            q2_params=q2_p,
            target_q1_params=tq1_p,
            target_q2_params=tq2_p,
            delay_q_target_update=args.delay_q_target_update_flow,
            delay_policy_update=args.delay_q_target_update_flow,
            lr_schedule_end=args.lr_schedule_end,
            lr_schedule_steps=args.lr_schedule_steps,
            alpha_lr=args.alpha_lr_flow,
            delay_alpha_update=args.delay_alpha_update_flow,
            noise_scale=args.noise_scale_flow,
            target_entropy_scale=args.target_entropy_scale_flow,
            adv_clip=args.adv_clip_flow,
            q_backup_clip=args.q_backup_clip_flow,
        )

    elif args.alg == 'fpmd':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_flow_ppo_net(
            init_network_key, obs_dim, act_dim,
            policy_hidden_sizes=diffusion_hidden_sizes,
            value_hidden_sizes=hidden_sizes,
            policy_activation=mish,
            value_activation=mish,
            num_steps=args.flow_steps,
        )
        q_key = jax.random.fold_in(init_network_key, 0x0F1CE)
        q = hk.without_apply_rng(hk.transform(
            lambda obs, act: QNet(hidden_sizes, mish)(obs, act)
        ))
        sample_obs = jnp.zeros((1, obs_dim))
        sample_act = jnp.zeros((1, act_dim))
        q1_key, q2_key = jax.random.split(q_key)
        q1_p = q.init(q1_key, sample_obs, sample_act)
        q2_p = q.init(q2_key, sample_obs, sample_act)
        q_apply = q.apply
        algorithm = FPMD(
            agent, params,
            lr=args.lr,
            value_lr=args.lr,
            q_lr=args.lr,
            reward_scale=args.reward_scale,
            max_ratio_weight=args.max_ratio_weight,
            gamma=args.gamma_flow,
            gae_lambda=0.95,
            value_loss_coeff=0.5,
            n_epochs=args.n_epochs_flow,
            num_minibatches=args.num_minibatches_flow,
            max_grad_norm=args.max_grad_norm_flow,
            huber_delta=10.0,
            value_clip=0.2,
            value_updates_per_batch=10,
            policy_updates_per_batch=args.policy_updates_per_batch_flow,
            off_policy=True,
            q_apply=q_apply,
            q1_params=q1_p,
            q2_params=q2_p,
            target_q1_params=q1_p,
            target_q2_params=q2_p,
            delay_q_target_update=args.delay_q_target_update_flow,
            delay_policy_update=args.delay_q_target_update_flow,
            lr_schedule_end=args.lr_schedule_end,
            lr_schedule_steps=args.lr_schedule_steps,
            alpha_lr=args.alpha_lr_flow,
            delay_alpha_update=args.delay_alpha_update_flow,
            noise_scale=args.noise_scale_flow,
            target_entropy_scale=args.target_entropy_scale_flow,
        )

    elif args.alg == 'doobpo_flow_grpo':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        (
            agent,
            params,
            ratio_net,
            ratio_params,
            q_apply,
            q1_p,
            q2_p,
            tq1_p,
            tq2_p,
        ) = create_doobpo_flow_net(
            init_network_key, obs_dim, act_dim,
            policy_hidden_sizes=diffusion_hidden_sizes,
            value_hidden_sizes=hidden_sizes,
            policy_activation=mish,
            value_activation=mish,
            num_steps=args.flow_steps,
            include_twin_q=True,
            ratio_net_type=args.ratio_net_type,
            ratio_hidden_sizes=ratio_hidden_sizes_kw,
            ratio_resnet_hidden_dim=args.ratio_resnet_hidden_dim,
            ratio_resnet_num_blocks=args.ratio_resnet_num_blocks,
        )
        algorithm = DoobPOFlowGRPO(
            agent, params, ratio_net, ratio_params,
            lr=args.lr,
            value_lr=args.lr,
            ratio_lr=args.ratio_lr,
            q_lr=args.lr,
            reward_scale=args.reward_scale,
            ppo_eps=args.ppo_eps,
            max_ratio_weight=args.max_ratio_weight,
            ratio_regularizer_lambda=args.ratio_regularizer_lambda,
            gamma=args.gamma_flow,
            gae_lambda=0.95,
            value_loss_coeff=0.5,
            n_epochs=args.n_epochs_flow,
            num_minibatches=args.num_minibatches_flow,
            max_grad_norm=args.max_grad_norm_flow,
            huber_delta=10.0,
            value_clip=0.2,
            value_updates_per_batch=10,
            ratio_updates_per_batch=args.ratio_updates_per_batch_flow,
            policy_updates_per_batch=args.policy_updates_per_batch_flow,
            off_policy=True,
            q_apply=q_apply,
            q1_params=q1_p,
            q2_params=q2_p,
            target_q1_params=tq1_p,
            target_q2_params=tq2_p,
            delay_q_target_update=args.delay_q_target_update_flow,
            delay_policy_update=args.delay_q_target_update_flow,
            lr_schedule_end=args.lr_schedule_end,
            lr_schedule_steps=args.lr_schedule_steps,
            alpha_lr=args.alpha_lr_flow,
            delay_alpha_update=args.delay_alpha_update_flow,
            noise_scale=args.noise_scale_flow,
            target_entropy_scale=args.target_entropy_scale_flow,
            num_grpo_samples=args.num_grpo_samples,
            adv_clip=args.adv_clip_flow,
            q_backup_clip=args.q_backup_clip_flow,
            grpo_batch_norm_adv=args.grpo_batch_norm_adv,
        )

    elif args.alg == 'fpo_flow':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_flow_ppo_net(
            init_network_key, obs_dim, act_dim,
            policy_hidden_sizes=diffusion_hidden_sizes,
            value_hidden_sizes=hidden_sizes,
            policy_activation=mish,
            value_activation=mish,
            num_steps=args.flow_steps,
        )
        fh = get_task_hparams(FPO_HPARAMS, args.env)
        print(
            f"FPO-Flow: per-task hparams for {args.env!r} — "
            f"lr={fh['lr']}, value_lr={fh['value_lr']}, target_kl={fh['target_kl']}"
        )
        algorithm = FPOFlow(
            agent, params,
            n_epochs=4,
            n_cfm_samples=32,
            ppo_eps=0.1,
            gamma=0.99,
            lr=fh["lr"],
            value_lr=fh["value_lr"],
            reward_scale=1.0,
            gae_lambda=0.95,
            value_loss_coeff=0.5,
            normalize_obs=True,
            discretize_t=True,
            num_minibatches=args.num_minibatches_flow,
            obs_dim=obs_dim,
            max_grad_norm=fh["max_grad_norm"],
            huber_delta=10.0,
            value_clip=0.2,
            cfm_loss_scale=16.0,
            cfm_diff_clip=1.0,
            target_kl=fh["target_kl"],
            value_updates_per_batch=10,
            policy_updates_per_batch=1,
        )

    elif args.alg == 'reinflow':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_flow_ppo_net(
            init_network_key, obs_dim, act_dim,
            policy_hidden_sizes=diffusion_hidden_sizes,
            value_hidden_sizes=hidden_sizes,
            policy_activation=mish,
            value_activation=mish,
            num_steps=args.flow_steps,
        )
        noise_key = jax.random.fold_in(init_network_key, 1)
        noise_net, noise_params = create_explore_noise_net(
            noise_key, obs_dim, act_dim,
            min_noise_std=args.min_noise_std_flow,
            max_noise_std=args.max_noise_std_flow,
        )
        on_policy_batch = args.batch_size
        total_train_iters = args.total_step // on_policy_batch
        if args.reinflow_legacy_hparams:
            rf = dict(
                n_epochs=5,
                ppo_eps=0.1,
                gamma=0.99,
                actor_lr=5e-6,
                critic_lr=1e-4,
                noise_lr=args.noise_lr_flow,
                critic_weight_decay=1e-4,
                reward_scale=1.0,
                gae_lambda=0.95,
                ent_coef=0.03,
                logprob_min=args.logprob_min_flow,
                logprob_max=args.logprob_max_flow,
                randn_clip=args.randn_clip_flow,
                clip_intermediate=args.clip_intermediate_flow,
                normalize_logprob=bool(args.normalize_logprob_flow),
                n_critic_warmup=10,
                total_train_iters=total_train_iters,
                min_noise_std=args.min_noise_std_flow,
                max_noise_std=args.max_noise_std_flow,
                reward_scale_running=False,
                num_envs=args.num_vec_envs,
                num_minibatches=args.num_minibatches_flow,
                lr_cycle_steps=100,
                lr_warmup_steps=10,
                actor_min_lr=2.5e-6,
                critic_min_lr=5e-5,
                target_kl=0.05,
                max_grad_norm=1.0,
                huber_delta=10.0,
                value_clip=0.2,
                value_updates_per_batch=10,
                policy_updates_per_batch=1,
            )
            print("ReinFlow: using --reinflow_legacy_hparams (old DoobPO defaults).")
        else:
            oh = get_task_hparams(REINFLOW_HPARAMS, args.env)
            print(
                f"ReinFlow: per-task hparams for {args.env!r} — "
                f"actor_lr={oh['actor_lr']}, critic_lr={oh['critic_lr']}, target_kl={oh['target_kl']}, "
                f"n_critic_warmup={oh['n_critic_warmup']}, reward_scale_running={oh['reward_scale_running']}"
            )
            rf = dict(
                n_epochs=5,
                ppo_eps=oh["ppo_eps"],
                gamma=0.99,
                actor_lr=oh["actor_lr"],
                critic_lr=oh["critic_lr"],
                noise_lr=oh["actor_lr"],
                critic_weight_decay=oh["critic_weight_decay"],
                reward_scale=1.0,
                gae_lambda=0.95,
                ent_coef=0.03,
                logprob_min=args.logprob_min_flow,
                logprob_max=args.logprob_max_flow,
                randn_clip=args.randn_clip_flow,
                clip_intermediate=args.clip_intermediate_flow,
                normalize_logprob=bool(args.normalize_logprob_flow),
                n_critic_warmup=oh["n_critic_warmup"],
                total_train_iters=total_train_iters,
                min_noise_std=args.min_noise_std_flow,
                max_noise_std=args.max_noise_std_flow,
                reward_scale_running=oh["reward_scale_running"],
                num_envs=args.num_vec_envs,
                num_minibatches=args.num_minibatches_flow,
                lr_cycle_steps=oh["lr_cycle_steps"],
                lr_warmup_steps=oh["lr_warmup_steps"],
                actor_min_lr=oh["actor_min_lr"],
                critic_min_lr=oh["critic_min_lr"],
                target_kl=oh["target_kl"],
                max_grad_norm=oh["max_grad_norm"],
                huber_delta=10.0,
                value_clip=0.2,
                value_updates_per_batch=10,
                policy_updates_per_batch=1,
            )
        algorithm = ReinFlow(agent, params, noise_net, noise_params, **rf)

    elif args.alg == 'pirl':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_flow_ppo_net(
            init_network_key, obs_dim, act_dim,
            policy_hidden_sizes=diffusion_hidden_sizes,
            value_hidden_sizes=hidden_sizes,
            policy_activation=mish,
            value_activation=mish,
            num_steps=args.flow_steps,
        )
        algorithm = PiRL(
            agent, params,
            n_epochs=4,
            noise_level=args.noise_level_flow,
            min_sigma=args.min_sigma_flow,
            ppo_eps=0.1,
            lr=5e-6,
            value_lr=1e-4,
            reward_scale=1.0,
            gae_lambda=0.95,
            ent_coef=0.0,
            value_loss_coeff=0.5,
            max_grad_norm=1.0,
            huber_delta=10.0,
            value_clip=0.2,
            num_minibatches=args.num_minibatches_flow,
            noise_anneal=bool(args.noise_anneal_flow),
            noise_start=args.noise_start_flow,
            noise_end=args.noise_end_flow,
            noise_anneal_steps=args.noise_anneal_steps_flow,
            ignore_last=bool(args.ignore_last_flow),
            value_updates_per_batch=10,
            policy_updates_per_batch=1,
        )

    elif args.alg == 'idem':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_diffv2_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps, 
                                          num_particles=args.num_particles, 
                                          noise_scale=args.noise_scale,
                                          beta_schedule_scale=args.beta_schedule_scale)
        algorithm = IDEM(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "qsm":
        agent, params = create_qsm_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=20, num_particles=args.num_particles)
        algorithm = QSM(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "sac":
        agent, params = create_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = SAC(agent, params, lr=args.lr)
    elif args.alg == "dsact":
        agent, params = create_dsact_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = DSACT(agent, params, lr=args.lr)
    elif args.alg == "dacer":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_dacer_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish, 
                                         num_timesteps=args.diffusion_steps)
        algorithm = DACER(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "dacer_doubleq":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_dacer_doubleq_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish, num_timesteps=args.diffusion_steps)
        algorithm = DACERDoubleQ(agent, params, lr=args.lr)
    elif args.alg == "dipo":
        diffusion_buffer = TreeBuffer.from_example(
            ObsActionPair.create_example(obs_dim, act_dim),
            args.total_step,
            int(master_rng.integers(0, 2**32 - 1)),
            remove_batch_dim=False
        )
        TreeBuffer.connect(buffer, diffusion_buffer, lambda exp: ObsActionPair(exp.obs, exp.action))

        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))

        agent, params = create_dipo_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=100)
        algorithm = DIPO(agent, params, diffusion_buffer, lr=args.lr, action_gradient_steps=30, policy_target_delay=2, action_grad_norm=0.16)
    elif args.alg == "qvpo":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_qvpo_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps,
                                          num_particles=args.num_particles,
                                          noise_scale=args.noise_scale)
        algorithm = QVPO(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update)
    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")

    if args.cluster:
        PROJECT_ROOT = Path('/n/netscratch/nali_lab_seas/Lab/haitongma/sdac_logs')

    exp_dir = PROJECT_ROOT / "logs" / args.env / (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}_{args.suffix}')
    # On-policy flow baselines (always on-policy; everything else is off-policy).
    _on_policy_algs = ("fpo_flow", "reinflow", "pirl")
    use_on_policy = args.alg in _on_policy_algs
    if use_on_policy:
        n_epochs_resolved = getattr(algorithm, '_n_epochs', args.n_epochs_flow)
        on_policy_batch = args.batch_size
        assert args.total_step % on_policy_batch == 0, "total_step must be divisible by batch_size for on-policy"
        assert args.num_vec_envs > 0, "On-policy requires vector env (--num_vec_envs > 0)"
        assert on_policy_batch % args.num_vec_envs == 0, "batch_size must be divisible by num_vec_envs"
        trainer = OnPolicyTrainer(
            env=env,
            algorithm=algorithm,
            log_path=exp_dir,
            batch_size=on_policy_batch,
            total_step=args.total_step,
            n_epochs=n_epochs_resolved,
            act_steps=act_steps,
        )
        trainer.setup(GAEExperience.create_example(obs_dim, act_dim, on_policy_batch))
    else:
        trainer = OffPolicyTrainer(
            env=env,
            algorithm=algorithm,
            buffer=buffer,
            batch_size=args.batch_size,
            start_step=args.start_step,
            total_step=args.total_step,
            sample_per_iteration=1,
            update_per_iteration=args.utd,
            evaluate_env=eval_env,
            save_policy_every=int(args.total_step / 20),
            warmup_with="random",
            log_path=exp_dir,
            update_log_n_step=1 if args.debug else 1000,
        )
        trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))

    # Save the arguments to a YAML file
    args_dict = vars(args)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)
    trainer.run(train_key)
