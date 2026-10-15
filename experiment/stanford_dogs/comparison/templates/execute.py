"""Run only the four approved jobs sequentially; stop on failure, never retry."""
import hashlib,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'plan.json').read_text())
PYTHON=str(Path(P['reuse_dir'])/'env/bin/python')
STATE={'status':'running','completed_phases':[]}
def emit(event,**fields):print(json.dumps(dict(event=event,**fields)),flush=True)
def save():
 t=ROOT/'execution-status.json.tmp';t.write_text(json.dumps(STATE,indent=2));t.replace(ROOT/'execution-status.json')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def checked(args,**kwargs):
 p=subprocess.run(args,capture_output=True,text=True,**kwargs)
 if p.returncode:raise RuntimeError(str(args)+'\n'+p.stdout[-2000:]+p.stderr[-4000:])
 return p.stdout
def main():
 assert json.loads((ROOT/'approval.json').read_text())['approved'] is True
 assert json.loads((ROOT/'data-validation.json').read_text())['status']=='passed'
 with (ROOT/'execution-started.json').open('x') as f:json.dump({'time':time.time()},f)
 try:
  for mode in ['pixel','generator']:
   for phase in ['distill','eval']:
    run=ROOT/mode/phase
    STATE.update(mode=mode,phase=phase,stage='preparing');save()
    plan=json.loads((run/'experiment/plan.json').read_text())
    for rel,digest in json.loads((run/'experiment/source-manifest.json').read_text())['files'].items():assert sha(run/rel)==digest,rel
    if phase=='eval':
     d=ROOT/mode/'distill';s=json.loads((d/'status.json').read_text());v=json.loads((d/'distillation-validation.json').read_text());a=json.loads((d/'job-accounting.json').read_text())
     assert s['status']=='success' and s['completed_updates']==5001 and v['status']=='passed' and v['png_count']==120
     assert a['exit_status']==a['failed']=='0'
     src=Path(plan['source_data']);assert sha(src)==s['data_sha256']
     dest=run/'input/stanforddogs/dinov2_vitb'/plan['run_name']/'data.pth'
     dest.parent.mkdir(parents=True,exist_ok=False);shutil.copy2(src,dest);dest.chmod(0o444)
     inputs={str(src):sha(src),str(dest):sha(dest),str(d/'checkpoints/step_05000_final.pth'):sha(d/'checkpoints/step_05000_final.pth')}
     if mode=='generator':inputs[str(d/'generator_final.pth')]=sha(d/'generator_final.pth')
     (run/'input-manifest.json').write_text(json.dumps(inputs,indent=2))
    assert len(list(ROOT.glob('*/*/submission.json')))<4
    with (run/'submission-attempt.json').open('x') as f:json.dump({'time':time.time()},f)
    output=checked(['qsub','-g','tga-lab_otm',f'{phase}_job.sh'],cwd=run)
    match=re.search(r'Your job (\d+)',output);assert match,output
    job=match.group(1)
    (run/'submission.json').write_text(json.dumps({'job_id':job,'output':output}))
    STATE.update(job_id=job,stage='queued_or_running');save();emit('submitted',mode=mode,phase=phase,job_id=job)
    statusfile=run/('status.json' if phase=='distill' else 'eval_status.json')
    while True:
     q=subprocess.run(['qstat','-j',job],capture_output=True,text=True)
     active=bool(re.search(r'job_number:\s*'+job+r'\b',q.stdout))
     s=json.loads(statusfile.read_text()) if statusfile.exists() else {}
     fields={k:s[k] for k in ['status','completed_updates','last_loss','gpu','current_evaluation','current_epoch','completed_evaluations','error'] if k in s}
     emit('progress',mode=mode,phase=phase,job_id=job,active=active,**fields)
     if not active:
      account=subprocess.run(['qacct','-j',job],capture_output=True,text=True)
      if account.returncode==0 and 'jobnumber' in account.stdout:break
     time.sleep(30)
    a={}
    for line in account.stdout.splitlines():
     parts=line.split(None,1)
     if len(parts)==2:a[parts[0]]=parts[1].strip()
    (run/'job-accounting.json').write_text(json.dumps(a,indent=2));(run/'logs/qacct.txt').write_text(account.stdout)
    assert a['jobnumber']==job and a['exit_status']==a['failed']=='0',a
    assert s['status']=='success',s
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(ROOT/'src'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'))
    script='validate_distillation.py' if phase=='distill' else 'validate_evaluation.py'
    STATE['stage']='cpu_validation';save();emit('validation_started',mode=mode,phase=phase)
    with (run/'logs/independent-validation.log').open('x') as log:
     p=subprocess.Popen([PYTHON,str(run/'experiment'/script)],env=env,stdout=log,stderr=subprocess.STDOUT)
     while p.poll() is None:emit('validation_running',mode=mode,phase=phase);time.sleep(20)
    assert p.returncode==0,(run/'logs/independent-validation.log').read_text()[-8000:]
    validation='distillation-validation.json' if phase=='distill' else 'evaluation-validation.json'
    assert json.loads((run/validation).read_text())['status']=='passed'
    STATE['completed_phases'].append(dict(mode=mode,phase=phase,job_id=job,exit_status=0,validation='passed'));save()
    emit('phase_complete',mode=mode,phase=phase,job_id=job)
    checked([PYTHON,str(ROOT/'report.py')],env=env)
  STATE.update(status='complete',stage='all_validated');save()
  checked([PYTHON,str(ROOT/'report.py')],env=env)
  emit('study_complete',completed_phases=STATE['completed_phases'])
 except BaseException as e:
  STATE.update(status='error',error=str(e),traceback=traceback.format_exc());save();emit('study_error',error=str(e));raise
if __name__=='__main__':main()
