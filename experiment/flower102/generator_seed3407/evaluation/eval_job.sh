#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=01:00:00
#$ -p 0
#$ -N lgm_generator_eval3
#$ -j y
#$ -o /gs/bs/tga-lab_otm/us07824-lgm-generator-eval3-3QoaNElk/logs/eval.log
set -euo pipefail
EXPERIMENT_DIR=/gs/bs/tga-lab_otm/us07824-lgm-generator-eval3-3QoaNElk
source "$EXPERIMENT_DIR/experiment/job_environment.sh"
trap 'job_exit=$?; trap - EXIT; python "$EXPERIMENT_DIR/experiment/report.py" || true; exit "$job_exit"' EXIT
date -Is
nvcc --version
python -u "$EXPERIMENT_DIR/experiment/run_eval.py"
date -Is
