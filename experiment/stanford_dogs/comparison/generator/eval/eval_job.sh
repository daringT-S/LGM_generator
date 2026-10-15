#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=01:00:00
#$ -p 0
#$ -N dogs_generator_eval
#$ -j y
#$ -o /gs/bs/tga-lab_otm/us07824-lgm-stanford-dogs-comparison-plan-q361seki/generator/eval/logs/eval.log
set -euo pipefail
EXPERIMENT_DIR=/gs/bs/tga-lab_otm/us07824-lgm-stanford-dogs-comparison-plan-q361seki/generator/eval
source "$EXPERIMENT_DIR/experiment/job_environment.sh"
date -Is
nvidia-smi
python -u "$EXPERIMENT_DIR/experiment/run_eval.py"
date -Is
