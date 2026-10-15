#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -p 0
#$ -N lgm_gen_s123_distill
#$ -j y
#$ -o /gs/bs/tga-lab_otm/us07824-lgm-generator-seeds-wrpWhzXU/seed_123/distill/logs/distill.log
set -euo pipefail
EXPERIMENT_DIR=/gs/bs/tga-lab_otm/us07824-lgm-generator-seeds-wrpWhzXU/seed_123/distill
source "$EXPERIMENT_DIR/experiment/job_environment.sh"
trap 'job_exit=$?; trap - EXIT; python "$EXPERIMENT_DIR/experiment/report.py" || true; exit "$job_exit"' EXIT
date -Is
nvidia-smi
python -u "$EXPERIMENT_DIR/experiment/run_distill.py"
date -Is
