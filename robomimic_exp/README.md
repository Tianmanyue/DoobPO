# DoobPO on Robomimic

PyTorch code for the **Robomimic** manipulation experiments of the DoobPO paper. It applies DoobPO to
pretrained **diffusion** and **flow** policies and compares against DPPO and ReinFlow on the state-based
**Can**, **Square** and **Transport** tasks. The MuJoCo experiments live in the JAX code at the repository
root (`../relax`); this directory is a separate, self-contained codebase with its own environment.

This code is a fork of [ReinFlow](https://github.com/ReinFlow/ReinFlow), which builds on
[DPPO](https://github.com/irom-lab/dppo). See [`NOTICE`](NOTICE) and [`LICENSE`](LICENSE).

|                | Diffusion policy                      | Flow (rectified-flow) policy       |
| -------------- | ------------------------------------- | ---------------------------------- |
| **Our method** | **DoobPPO** (PPO-clipped ratio)       | **DoobGRPO** (GRPO group ratio)    |
| **Baseline**   | DPPO                                  | ReinFlow                           |

All experiments are **state-based** (low-dimensional observations). The metric is task **success rate**.

## Status

- **DoobPPO (diffusion) on Can and Square**: hyperparameters are tuned, and the configs
  `cfg/robomimic/finetune/{can,square}/ft_doobpo_diffusion_mlp.yaml` are the settings used in the paper
  (151 iterations on Can, 200 on Square).
- **Flow (DoobGRPO) and Transport**: the configs run end to end, but their hyperparameters were **not tuned**
  and no Robomimic results are reported for them in the paper. Treat them as a starting point.

---

## 1. Installation

```bash
conda create --name doobpo_robomimic python=3.8 -y && conda activate doobpo_robomimic
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .                       # this directory
pip install "robosuite==1.4.1" "robomimic==0.3.0" "mujoco==3.1.6" hydra-core gdown beautifulsoup4
python $(python -c "import robosuite,os;print(os.path.dirname(robosuite.__file__))")/scripts/setup_macros.py
```

> **`mujoco==3.1.6` is required** (the version DPPO pins together with `robosuite==1.4.1`). With newer
> MuJoCo versions the Robomimic contact physics differ and RL fails to learn to hold the object, so the
> success rate stays at 0. If `import mujoco_py` fails, add
> `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia`.

Set the environment variables (checkpoint, data and log directories, rendering libraries) once per shell:

```bash
source set_env.sh      # edit CKPT_ROOT and DPPO_WANDB_ENTITY inside for your machine
```

---

## 2. Reproducing the experiments

Common flags: `device=cuda:0 +sim_device=cuda:0` (`+sim_device` enables EGL offscreen rendering; omit it to
fall back to osmesa, about 3x slower). Add `wandb=null` to disable logging. Use `seed=42` (and 43, 44 for
extra seeds). Replace `can` with `square` or `transport` throughout.

### 2.1 Diffusion (DoobPPO vs DPPO)

Pretrained BC checkpoints and normalization statistics are **downloaded automatically** on the first run.

```bash
# Ours: DoobPPO
python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_doobpo_diffusion_mlp \
    device=cuda:0 +sim_device=cuda:0 seed=42
# Baseline: DPPO
python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_diffusion_mlp \
    device=cuda:0 +sim_device=cuda:0 seed=42
```

### 2.2 Flow (DoobGRPO vs ReinFlow), untuned

Flow policies need a state BC checkpoint. Pretrain it first (rectified-flow BC, about 50 epochs):

```bash
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_reflow_mlp device=cuda:0
```

This writes `.../<env>_pre_reflow_mlp_ta*_td100/<TIMESTAMP>_42/checkpoint/state_50.pt`. Copy it to
`pretrained/flow_bc/can_reflow_state50.pt` (the path the flow configs expect), or point `base_policy_path`
in `cfg/robomimic/finetune/can/{ft_doobpo_flow_mlp,ft_ppo_reflow_mlp}.yaml` at it. Then:

```bash
# Ours: DoobGRPO
python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_doobpo_flow_mlp \
    device=cuda:0 +sim_device=cuda:0 seed=42
# Baseline: ReinFlow
python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_reflow_mlp \
    device=cuda:0 +sim_device=cuda:0 seed=42
```

### 2.3 Results

Each run periodically evaluates the deterministic policy and logs **success rate** to wandb (if an entity is
set) and to a local `.pkl` file under its `logdir`. Exact numbers vary across seeds and hardware; the
training procedure itself is fully specified by the configs.

---

## 3. What DoobPO adds on top of ReinFlow

- `model/diffusion/diffusion_doobpo.py`: `RatioNet` and `DoobPODiffusion` (off-policy; twin-Q critic,
  PPO-clipped ratio network, ratio-reweighted score / drift matching). Port of `../relax/algorithm/doobpo.py`.
- `model/flow/ft_doobpo/doobpo_flow.py`: `DoobPOFlow` (off-policy; twin-Q critic, GRPO group-relative
  advantage, ratio-reweighted velocity matching). Port of `../relax/algorithm/doobpo_flow_grpo.py`.
- `agent/finetune/doobpo/`: the DoobPO training agents.
- `cfg/robomimic/finetune/{can,square,transport}/ft_doobpo_*.yaml`: DoobPO configs.

Everything else (environment wrappers, DPPO and ReinFlow agents, BC pretraining, utilities) comes from the
upstream projects and is kept as is, apart from trimming to what these experiments need.

## Acknowledgements

Built on [ReinFlow](https://github.com/ReinFlow/ReinFlow), [DPPO](https://github.com/irom-lab/dppo) and
[Robomimic](https://github.com/ARISE-Initiative/robomimic). We thank their authors.
