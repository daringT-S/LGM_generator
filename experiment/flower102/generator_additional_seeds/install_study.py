"""Materialize six isolated jobs without submitting them."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'study-plan.json').read_text())
assert str(ROOT)==plan['study_dir']
baseline=Path(plan['baseline_distillation_root'])
reuse=Path(plan['reuse_dir'])
shutil.copytree(baseline/'src',ROOT/'src',ignore=shutil.ignore_patterns('__pycache__'))
source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'src').rglob('*.py')}
distill=json.loads((baseline/'experiment/plan.json').read_text())
evalplan=json.loads(Path(plan['baseline_eval_root'],'experiment/plan.json').read_text())
base_env=Path(plan['baseline_eval_root'],'experiment/job_environment.sh').read_text()
for seed in plan['distillation_seeds']:
    seed_root=ROOT/f'seed_{seed}'
    for phase in ['distill','eval']:
        run=seed_root/phase
        (run/'logs').mkdir(parents=True,exist_ok=False)
        (run/'cache').mkdir(exist_ok=False)
        (run/'src').symlink_to(ROOT/'src',target_is_directory=True)
        shutil.copytree(ROOT/'templates'/phase,run/'experiment')
        if phase=='distill':
            shutil.copy2(ROOT/'seed_control.py',run/'experiment/seed_control.py')
            cfg=dict(distill['distillation'],job_tag=str(run/'distillation'),
                     run_name=f'flowers102_generator_pixel_adam_it5000_ipc1_seed{seed}')
            jobplan=dict(distill,experiment_dir=str(run),phase=phase,seed=seed,distillation=cfg,
                         cli_args=['--'+k+'='+str(v) for k,v in cfg.items()],
                         previous_pixel_artifacts=plan['preserved_artifacts'],
                         status='awaiting_user_approval')
            limit='02:00:00'
        else:
            droot=seed_root/'distill'
            dcfg=json.loads((droot/'experiment/plan.json').read_text())['distillation']
            data=droot/'distillation/flowers102/dinov2_vitb'/dcfg['run_name']/'data.pth'
            jobplan=dict(evalplan,experiment_dir=str(run),phase=phase,seed=seed,
                         distillation_root=str(droot),source_data=str(data),
                         run_name=f'generator_seed{seed}_final',
                         preserved_artifacts=plan['preserved_artifacts'],
                         status='awaiting_distillation_and_user_approval')
            jobplan.pop('data_sha256',None) # This will be measured after the new distillation, never guessed.
            limit='01:00:00'
        (run/'experiment/plan.json').write_text(json.dumps(jobplan,indent=2,ensure_ascii=False))
        shutil.copy2(ROOT/'phase_report.py',run/'experiment/report.py')
        (run/'experiment/job_environment.sh').write_text(base_env)
        job=run/f'{phase}_job.sh'
        job.write_text(f'''#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt={limit}
#$ -p 0
#$ -N lgm_gen_s{seed}_{phase}
#$ -j y
#$ -o {run}/logs/{phase}.log
set -euo pipefail
EXPERIMENT_DIR={run}
source "$EXPERIMENT_DIR/experiment/job_environment.sh"
trap 'job_exit=$?; trap - EXIT; python "$EXPERIMENT_DIR/experiment/report.py" || true; exit "$job_exit"' EXIT
date -Is
nvidia-smi
python -u "$EXPERIMENT_DIR/experiment/run_{phase}.py"
date -Is
''')
        # The template's evaluation entry script is named run_eval.py.
        subprocess.run(['bash','-n',str(job)],check=True)
        files=dict(source_files)
        for p in (run/'experiment').iterdir():
            if p.is_file():
                if p.suffix=='.py': ast.parse(p.read_text(),filename=str(p))
                files[str(p.relative_to(run))]=hashlib.sha256(p.read_bytes()).hexdigest()
        files[job.name]=hashlib.sha256(job.read_bytes()).hexdigest()
        (run/'experiment/source-manifest.json').write_text(json.dumps({'files':files},indent=2))
        subprocess.run(['python3',str(run/'experiment/report.py')],check=True)
print('Prepared six job scripts; no qsub was executed.')
