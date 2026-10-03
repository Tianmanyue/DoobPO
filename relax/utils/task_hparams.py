"""
Per-task hyperparameters for on-policy flow baselines (ReinFlow, FPO).

Each algorithm has three task groups whose LR scales mirror the original
ReinFlow ShortCut yaml files:

  halfcheetah  — conservative LR
  loc_medium   — moderate LR  (hopper, ant, humanoid + all others as fallback)
  walker2d     — aggressive LR (~13x halfcheetah)

Lookup: ``get_task_hparams(TABLE, env_id)`` returns a plain dict.
"""

from __future__ import annotations


def _task_key(env_id: str) -> str:
    """Map e.g. 'HalfCheetah-v4' -> 'halfcheetah'."""
    return env_id.split("-")[0].lower()


# ---------------------------------------------------------------------------
# ReinFlow  (from official cfg/gym/finetune/*/ft_ppo_shortcut_mlp.yaml)
# ---------------------------------------------------------------------------
REINFLOW_HPARAMS: dict[str, dict] = {
    "halfcheetah": dict(
        actor_lr=3.0e-5,   critic_lr=3.0e-4,
        actor_min_lr=1.5e-5, critic_min_lr=1.5e-4,
        lr_cycle_steps=100, lr_warmup_steps=10,
        n_critic_warmup=10, critic_weight_decay=1e-4,
        target_kl=1.0, ppo_eps=0.01,
        reward_scale_running=True, max_grad_norm=0.0,
    ),
    "walker2d": dict(
        actor_lr=4.0e-4,   critic_lr=4.0e-3,
        actor_min_lr=4.0e-4, critic_min_lr=4.0e-3,
        lr_cycle_steps=1000, lr_warmup_steps=100,
        n_critic_warmup=5,  critic_weight_decay=1e-5,
        target_kl=1.0, ppo_eps=0.01,
        reward_scale_running=True, max_grad_norm=0.0,
    ),
    # hopper / ant / humanoid  (and fallback for all others)
    "_default": dict(
        actor_lr=4.5e-5,   critic_lr=6.5e-4,
        actor_min_lr=2.0e-5, critic_min_lr=3.0e-4,
        lr_cycle_steps=100, lr_warmup_steps=10,
        n_critic_warmup=0,  critic_weight_decay=1e-5,
        target_kl=1.0, ppo_eps=0.01,
        reward_scale_running=True, max_grad_norm=0.0,
    ),
}

# ---------------------------------------------------------------------------
# FPO-Flow  (derived from FPO paper defaults, scaled by ReinFlow LR ratios)
#
# Base (halfcheetah):  lr=5e-6, value_lr=1e-4
# Ratios from ReinFlow:
#   loc_medium / halfcheetah:  actor ~1.5x,  critic ~2.2x
#   walker2d   / halfcheetah:  actor ~13x,   critic ~13x
# ---------------------------------------------------------------------------
FPO_HPARAMS: dict[str, dict] = {
    "halfcheetah": dict(
        lr=5.0e-6,   value_lr=1.0e-4,
        target_kl=0.05, max_grad_norm=1.0,
    ),
    "walker2d": dict(
        lr=7.0e-5,   value_lr=1.0e-3,
        target_kl=0.10, max_grad_norm=1.0,
    ),
    "_default": dict(
        lr=7.5e-6,   value_lr=2.0e-4,
        target_kl=0.05, max_grad_norm=1.0,
    ),
}


def get_task_hparams(table: dict[str, dict], env_id: str) -> dict:
    """Look up per-task hparams.  Falls back to ``table["_default"]``."""
    return table.get(_task_key(env_id), table["_default"])
