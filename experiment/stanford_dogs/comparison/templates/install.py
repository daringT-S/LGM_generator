"""Prepare isolated jobs, without calling qsub."""
import ast,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'plan.json').read_text())
assert str(ROOT)==P['proposed_remote_dir']
assert json.loads((ROOT/'data-validation.json').read_text())['status']=='passed'
baseline=Path('/gs/bs/tga-lab_otm/us07824-lgm-generator-seeds-wrpWhzXU')
old=json.loads((baseline/'study-plan.json').read_text())['preserved_artifacts']
for f in [baseline/'experiment_report.md',baseline/'study-results.json']:
 old[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for f in baseline.glob('seed_*/*/status.json'):old[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for f in baseline.glob('seed_*/distill/distillation/**/data.pth'):old[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
(ROOT/'preserved-artifacts.json').write_text(json.dumps(old,indent=2))
env=(baseline/'seed_42/distill/experiment/job_environment.sh').read_text()
source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').rglob('*.py')}
for name,h in source.items():assert hashlib.sha256((baseline/name).read_bytes()).hexdigest()==h
for mode in ['pixel','generator']:
 for phase in ['distill','eval']:
  run=ROOT/mode/phase
  (run/'logs').mkdir(parents=True,exist_ok=False);(run/'cache').mkdir()
  (run/'src').symlink_to(ROOT/'src',target_is_directory=True)
  shutil.copytree(ROOT/'templates'/(mode if phase=='distill' else 'eval'),run/'experiment')
  plan=dict(experiment_dir=str(run),study_root=str(ROOT),reuse_dir=P['reuse_dir'],source_commit='d0dbe18683ea2f0bccda7807ca9f61c73908bced',phase=phase,mode=mode,seed=3407,preserved_artifacts=old,previous_pixel_artifacts=old,checkpoint_snapshot_interval=500)
  if phase=='distill':
   cfg={k:v for k,v in P['distillation'].items() if k not in ['distill_modes','updates','seed']}
   cfg.update(distill_mode='pixel' if mode=='pixel' else 'generator_pixel',checkpoint_it=100,image_log_it=500,skip_if_exists=True,data_root=str(ROOT/'data'),job_tag=str(run/'distillation'),run_name=f'stanforddogs_{mode}_adam_it5000_ipc1_s3407')
   plan.update(distillation=cfg,distillation_seed=3407,run_name=cfg['run_name'],cli_args=['--'+k+'='+str(v) for k,v in cfg.items()])
   shutil.copy2(ROOT/'validate_distillation.py',run/'experiment/validate_distillation.py')
  else:
   droot=ROOT/mode/'distill';dcfg=json.loads((droot/'experiment/plan.json').read_text())['distillation']
   cfg={k:v for k,v in P['evaluation'].items() if k not in ['evaluation_seeds','std_ddof']}
   plan.update(evaluation=cfg,evaluation_seeds=[3407,3408,3409],distillation_root=str(droot),source_data=str(droot/'distillation/stanforddogs/dinov2_vitb'/dcfg['run_name']/'data.pth'),run_name=f'{mode}_final')
  (run/'experiment/plan.json').write_text(json.dumps(plan,indent=2))
  (run/'experiment/job_environment.sh').write_text(env)
  limit='02:00:00' if phase=='distill' else '01:00:00'
  script=run/f'{phase}_job.sh'
  script.write_text(f'''#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt={limit}
#$ -p 0
#$ -N dogs_{mode}_{phase}
#$ -j y
#$ -o {run}/logs/{phase}.log
set -euo pipefail
EXPERIMENT_DIR={run}
source "$EXPERIMENT_DIR/experiment/job_environment.sh"
date -Is
nvidia-smi
python -u "$EXPERIMENT_DIR/experiment/run_{phase}.py"
date -Is
''')
  subprocess.run(['bash','-n',str(script)],check=True)
  files=dict(source)
  for f in (run/'experiment').iterdir():
   if f.suffix=='.py':ast.parse(f.read_text())
   files[str(f.relative_to(run))]=hashlib.sha256(f.read_bytes()).hexdigest()
  files[script.name]=hashlib.sha256(script.read_bytes()).hexdigest()
  (run/'experiment/source-manifest.json').write_text(json.dumps({'files':files},indent=2))
print('4 isolated scripts ready; no jobs submitted')
