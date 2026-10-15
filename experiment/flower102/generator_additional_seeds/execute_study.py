"""Run the approved six-job study sequentially; stop on any failure, never retry."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'study-plan.json').read_text())
PYTHON=Path(PLAN['reuse_dir'])/'env/bin/python'
STATE={'status':'running','completed_phases':[]}


def emit(event,**values):
    print(json.dumps({'event':event,**values},ensure_ascii=False),flush=True)


def save():
    temp=ROOT/'execution-status.json.tmp'
    temp.write_text(json.dumps(STATE,indent=2,ensure_ascii=False))
    temp.replace(ROOT/'execution-status.json')


def run_checked(command,**kwargs):
    result=subprocess.run(command,capture_output=True,text=True,**kwargs)
    if result.returncode:
        raise RuntimeError(json.dumps({'command':[str(x) for x in command],
                                       'code':result.returncode,'stdout':result.stdout[-6000:],
                                       'stderr':result.stderr[-6000:]},ensure_ascii=False))
    return result.stdout


def cpu_environment():
    return dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONNOUSERSITE='1',
                PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2',XDG_CACHE_HOME=str(ROOT/'cache'),
                MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline')


def main():
    assert json.loads((ROOT/'approval.json').read_text())['approved'] is True
    # Exclusive creation also prevents restarting an uncertain orchestration.
    with (ROOT/'execution-started.json').open('x') as handle:
        json.dump({'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},handle)
    try:
        for seed in PLAN['distillation_seeds']:
            for phase in ['distill','eval']:
                run=ROOT/f'seed_{seed}'/phase
                STATE.update(seed=seed,phase=phase,stage='submitting')
                save()
                run_checked(['python3',str(ROOT/'submit.py'),phase,str(seed)])
                submitted=json.loads((run/'submission.json').read_text())
                job=submitted['job_id']
                STATE.update(job_id=job,stage='queued_or_running')
                save()
                emit('submitted',seed=seed,phase=phase,job_id=job)
                status_file=run/('status.json' if phase=='distill' else 'eval_status.json')
                while True:
                    q=subprocess.run(['qstat','-j',job],capture_output=True,text=True)
                    active=bool(re.search(r'job_number:\s*'+re.escape(job)+r'\b',q.stdout))
                    state=json.loads(status_file.read_text()) if status_file.exists() else {}
                    progress={k:state[k] for k in ['status','completed_updates','last_loss','gpu',
                               'current_evaluation','current_epoch','completed_evaluations',
                               'peak_allocated_bytes','error'] if k in state}
                    emit('progress',seed=seed,phase=phase,job_id=job,active=active,**progress)
                    if not active:
                        account=subprocess.run(['qacct','-j',job],capture_output=True,text=True)
                        if account.returncode==0 and 'jobnumber' in account.stdout:
                            break
                    time.sleep(30)
                run_checked(['python3',str(ROOT/'record_accounting.py'),phase,str(seed)])
                accounting=json.loads((run/'job-accounting.json').read_text())
                state=json.loads(status_file.read_text()) if status_file.exists() else {}
                if accounting['exit_status']!='0' or accounting['failed']!='0' or state.get('status')!='success':
                    raise RuntimeError(f'Job {job} failed: '+json.dumps({'accounting':accounting,'state':state},ensure_ascii=False))
                STATE.update(stage='cpu_validation')
                save()
                emit('validation_started',seed=seed,phase=phase,job_id=job)
                validator='validate_results.py' if phase=='distill' else 'validate_evaluation.py'
                with (run/'logs/independent-validation.log').open('x') as log:
                    process=subprocess.Popen([str(PYTHON),str(run/'experiment'/validator)],
                                             env=cpu_environment(),stdout=log,stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        emit('validation_running',seed=seed,phase=phase,job_id=job)
                        time.sleep(20)
                    if process.returncode:
                        raise RuntimeError('Independent validation failed: '+(run/'logs/independent-validation.log').read_text()[-10000:])
                validation_file=run/('distillation-validation.json' if phase=='distill' else 'evaluation-validation.json')
                assert json.loads(validation_file.read_text())['status']=='passed'
                run_checked(['python3',str(run/'experiment/report.py')])
                STATE['completed_phases'].append({'seed':seed,'phase':phase,'job_id':job,
                                                   'exit_status':0,'validation':'passed'})
                save()
                emit('phase_complete',seed=seed,phase=phase,job_id=job)
                run_checked([str(PYTHON),str(ROOT/'aggregate.py')],env=cpu_environment())
        STATE.update(status='complete',stage='all_validated')
        save()
        emit('study_complete',completed_phases=STATE['completed_phases'])
    except BaseException as error:
        STATE.update(status='error',error=str(error),traceback=traceback.format_exc())
        save()
        emit('study_error',seed=STATE.get('seed'),phase=STATE.get('phase'),job_id=STATE.get('job_id'),error=str(error))
        raise


if __name__=='__main__':
    main()
