#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=00:15:00
#$ -p 0
#$ -N lgm_generator_smoke
#$ -j y
#$ -o /gs/bs/tga-lab_otm/us07824-lgm-generator-smoke-j4SzPSSj/logs/gpu.log
set -euo pipefail
EXPERIMENT_DIR=/gs/bs/tga-lab_otm/us07824-lgm-generator-smoke-j4SzPSSj
REUSE_DIR=/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv
test ! -e "$EXPERIMENT_DIR/status.json"
module purge
module load cuda/12.8.0
export PATH="$REUSE_DIR/env/bin:$PATH"
export PYTHONPATH="$EXPERIMENT_DIR/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME="$REUSE_DIR/cache/torch"
export HF_HOME="$REUSE_DIR/cache/huggingface"
export XDG_CACHE_HOME="$EXPERIMENT_DIR/cache"
export MPLCONFIGDIR="$EXPERIMENT_DIR/cache/matplotlib"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XFORMERS_DISABLED=1
export WANDB_MODE=offline WANDB_DIR="$EXPERIMENT_DIR"
export WANDB_CACHE_DIR="$EXPERIMENT_DIR/cache/wandb"
export WANDB_CONFIG_DIR="$EXPERIMENT_DIR/cache/wandb-config"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="${T4TMPDIR:?This script requires an allocated TSUBAME job}"
cd "$EXPERIMENT_DIR/src"
date -Is
nvidia-smi
python -u "$EXPERIMENT_DIR/experiment/run_smoke.py"
date -Is
