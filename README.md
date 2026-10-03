# LFGPO: Likelihood-Free Generative Policy Optimization

A unified framework for reinforcement-learning fine-tuning of **diffusion**
and **flow** policies **without evaluating exact action likelihoods**.

LFGPO decouples policy improvement into two stages:

1. **Ratio learning.** Train a lightweight ratio network `r_β(s, a)` with a
   trust-region RL objective (PPO, GRPO, or — when an analytic form is
   available — policy mirror descent / SAC).
2. **Doob's Drift Matching.** Distill the policy improvement encoded in
   `r_β` into the generative policy by minimising a *conditional* drift
   matching loss, which is gradient-equivalent to the ideal Doob's Drift
   Matching objective. For diffusion policies, the resulting tilted
   process is exactly the Doob *h*-transform of the pretrained reverse
   SDE, and equivalently the Schrödinger Bridge to the target.

This avoids the intractable likelihoods of generative policies while
supporting a broad family of policy-update rules. See the paper for the
full derivation, theoretical guarantees, and the unified-framework view of
prior methods.

> Code release accompanying the paper (NeurIPS 2026). Preprint forthcoming.
>
> **Project page:** https://tianmanyue.github.io/LFGPO

---

## Installation

**System requirements.** Linux only (the C extensions in `src/` use
`futex`, `prctl`, and Linux-specific spinlock primitives). GPU training
requires an NVIDIA driver compatible with CUDA 12 — driver `>= 525.60.13`
is the JAX-stated minimum, and we have tested with driver `590.48.01`.
The CUDA 12 toolkit and cuDNN 8.9 are bundled by the `jax[cuda12]` pip
wheels, so no separate CUDA / cuDNN system install is required.

```bash
conda create -n relax python=3.10 numpy tqdm tensorboardX matplotlib scikit-learn black snakeviz ipykernel setproctitle numba
conda activate relax

# JAX with CUDA 12
pip install --upgrade "jax[cuda12]==0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -r requirements.txt
pip install -e .
```

`pip install -e .` builds the small C extensions in `src/` (futex, spinlock,
prctl) used by the vectorised env workers.

## Quickstart

A single command runs any algorithm on any MuJoCo task:

```bash
XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=.2 \
python scripts/train_mujoco.py --alg lfgpo --env Ant-v4 --seed 100
```

Logs (TensorBoard + wandb + final eval CSV) land in
`logs/<env>/<alg>_<timestamp>_s<seed>_<suffix>/`.

Hyperparameters used in the paper are the script's argparse defaults; for
per-task tuned hyperparameters of the on-policy flow baselines (`fpo_flow`,
`reinflow`), see `relax/utils/task_hparams.py` (read automatically).

---

## Our methods (LFGPO instantiations)

LFGPO is parameterised along two axes — **generative policy class**
(diffusion / flow) and **ratio-learning rule** (PMD / PPO / GRPO). The
combinations introduced in this paper are:

| Paper name      | Generative class | Ratio update | `--alg` flag        |
| --------------- | ---------------- | ------------ | ------------------- |
| **LFGPO-PPO (DP)**  | Diffusion    | PPO learned ratio | `lfgpo`          |
| **LFGPO-GRPO (DP)** | Diffusion    | GRPO learned ratio | `lfgpo_grpo`    |
| **LFGPO-PMD (Flow)**  | Flow       | PMD (analytic, exp-advantage) | `fpmd`     |
| **LFGPO-PPO (Flow)**  | Flow       | PPO learned ratio | `lfgpo_flow`     |
| **LFGPO-GRPO (Flow)** | Flow       | GRPO learned ratio | `lfgpo_flow_grpo` |

The diffusion analytic-ratio instantiation (LFGPO-PMD applied to a
diffusion policy) is **not new**: it coincides with prior work DPMD
([Ma et al. 2025](https://arxiv.org/abs/2502.00361)), and the paper shows
DPMD arises as a special case of the LFGPO framework. We therefore list
it as a baseline below rather than as one of our methods.

## Baselines

| Paper name | Reference | `--alg` flag / source |
| ---------- | --------- | --------------------- |
| **Diffusion-policy RL** | | |
| DIPO   | [Yang et al. 2023](https://arxiv.org/abs/2305.13122)        | `dipo`   |
| DACER  | [Wang et al. 2024](https://arxiv.org/abs/2409.01400)        | `dacer`  |
| SDAC   | [Ma et al. 2025](https://arxiv.org/abs/2502.00361)          | `sdac`   |
| DPMD   | [Ma et al. 2025](https://arxiv.org/abs/2502.00361)          | `dpmd`   |
| DPPO   | [Ren et al. 2024](https://arxiv.org/abs/2409.00588)         | external — see [irom-princeton/dppo](https://github.com/irom-princeton/dppo) |
| **Flow-policy RL** | | |
| FPO     | [McAllister et al. 2025](https://arxiv.org/abs/2502.05787)  | `fpo_flow`  |
| ReinFlow | [Zhang et al. 2025](https://arxiv.org/abs/2505.22094)      | `reinflow`  |
| Flow-SDE | [Chen et al. 2025](https://arxiv.org/abs/2502.01819)       | `pirl`      |

The codebase also ships several diffusion-RL algorithms not reported in the
paper but kept for reference: `sac`, `dsact`, `dacer_doubleq`, `qsm`,
`qsmv2`, `qvpo`, `idem`. Run them the same way (`--alg <flag>`).

---

## Layout

```
relax/                       core package
├── algorithm/               algorithms (one file per --alg)
├── network/                 network factories (Diffv2, FlowPPO, …)
├── trainer/                 OffPolicyTrainer / OnPolicyTrainer + evaluator subprocess
├── buffer/                  TreeBuffer (replay)
├── env/                     vectorised env workers (futex IPC)
└── utils/                   diffusion / flow-matching schedulers, RNG, …
src/                         C extensions (futex, spinlock, prctl)
scripts/train_mujoco.py      single training entry point
```

---

## Citation

<!-- TODO: replace with the official entry once the arXiv / proceedings version is public. -->
```bibtex
@inproceedings{su2026lfgpo,
  author    = {Su, Maojiang and Zhu, Qijie and Yu, Shuyang and Xiao, Yu and Chen, Minshuo and Wang, Zhaoran and Liu, Han},
  title     = {Likelihood-Free Generative Policy Optimization},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```
