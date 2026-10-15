#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -p 0
#$ -N dogs_pixel_distill
#$ -j y
#$ -o /gs/bs/tga-lab_otm/us07824-lgm-stanford-dogs-comparison-plan-q361seki/pixel/distill/logs/distill.log
set -euo pipefail
EXPERIMENT_DIR=/gs/bs/tga-lab_otm/us07824-lgm-stanford-dogs-comparison-plan-q361seki/pixel/distill
source "$EXPERIMENT_DIR/experiment/job_environment.sh"
date -Is
nvidia-smi
python -u "$EXPERIMENT_DIR/experiment/run_distill.py"
date -Is
