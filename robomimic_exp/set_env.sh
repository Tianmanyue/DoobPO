#!/usr/bin/env bash
# DoobPO-Robomimic environment setup. Source this before running any experiment:
#     conda activate doobpo_robomimic
#     source set_env.sh
#
# It sets: mujoco_py rendering libs, REINFLOW_* / DPPO_* dirs (shared ckpt dir),
# and wandb entity. Edit CKPT_ROOT / WANDB_ENTITY for your machine.

# --- mujoco_py native libs (robomimic 0.3.0 imports mujoco_py at module load) ---
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export CPATH="${CONDA_PREFIX}/include${CPATH:+:${CPATH}}"
  export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

# --- repo code dir: derive from the installed editable package so it
#     exactly matches util/dirs.py's abspath check (avoids the REINFLOW_DIR mismatch error) ---
export REINFLOW_DIR="$(python -c 'import util, os; print(os.path.dirname(util.__path__[0]))')"

# --- shared checkpoint / data / log root (downloaded ckpts + run outputs live here,
#     outside the git repos so the deliverable stays clean; download once, reuse) ---
CKPT_ROOT="${CKPT_ROOT:-${REINFLOW_DIR}/runtime}"
export DPPO_LOG_DIR="${CKPT_ROOT}/log"
export DPPO_DATA_DIR="${CKPT_ROOT}/data"
export REINFLOW_LOG_DIR="${DPPO_LOG_DIR}"
export REINFLOW_DATA_DIR="${DPPO_DATA_DIR}"
mkdir -p "${DPPO_LOG_DIR}" "${DPPO_DATA_DIR}"

# --- wandb (set to your entity, or pass wandb=null on the CLI to disable) ---
export DPPO_WANDB_ENTITY="${DPPO_WANDB_ENTITY:-dummy}"
export REINFLOW_WANDB_ENTITY="${DPPO_WANDB_ENTITY}"

echo "[set_env] REINFLOW_DIR=${REINFLOW_DIR}"
echo "[set_env] DPPO_LOG_DIR=${DPPO_LOG_DIR}"
echo "[set_env] DPPO_DATA_DIR=${DPPO_DATA_DIR}"
