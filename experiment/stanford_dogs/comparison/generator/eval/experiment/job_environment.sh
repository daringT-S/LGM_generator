# Sourced only inside an allocated job. No shell initialization files change.
REUSE_DIR=/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv
module purge
module load cuda/12.8.0
export PATH="$REUSE_DIR/env/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME="$REUSE_DIR/cache/torch"
export HF_HOME="$REUSE_DIR/cache/huggingface"
export XDG_CACHE_HOME="$EXPERIMENT_DIR/cache"
export MPLCONFIGDIR="$EXPERIMENT_DIR/cache/matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XFORMERS_DISABLED=1
export WANDB_MODE=offline
export WANDB_DIR="$EXPERIMENT_DIR"
export WANDB_CACHE_DIR="$EXPERIMENT_DIR/cache/wandb"
export WANDB_CONFIG_DIR="$EXPERIMENT_DIR/cache/wandb-config"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TMPDIR="${T4TMPDIR:?This script requires an allocated TSUBAME job}"
cd "$EXPERIMENT_DIR/src"
