"""Invoke only after explicit user approval: one attempt for one phase/seed."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def check_evaluation_gate(droot,seed):
    state=json.loads((droot/'status.json').read_text())
    validation=json.loads((droot/'distillation-validation.json').read_text())
    account=json.loads((droot/'job-accounting.json').read_text())
    assert state['status']=='success' and state['completed_updates']==5001
    assert state['distillation_seed']==state['generator_init_cpu_seed']==state['fixed_noise_init_cuda_seed']==seed
    assert validation['status']=='passed' and validation['png_count']==102
    assert state['fixed_noise_unchanged'] and state['generator_parameters_updated']
    assert account['exit_status']=='0' and account['failed']=='0'
    assert str(account['jobnumber'])==str(state['job_id'])
    data=Path(state['data_path'])
    assert sha(data)==state['data_sha256']
    return state,data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['distill','eval'])
    parser.add_argument('seed',type=int,choices=[42,123,2026])
    args=parser.parse_args()
    approval=json.loads((ROOT/'approval.json').read_text()) # Absent until a user approves the concrete plan.
    assert approval['approved'] is True
    assert approval['study_plan_sha256']==sha(ROOT/'study-plan.json')
    assert approval['max_jobs']==6 and approval['max_points']==1.44
    assert json.loads((ROOT/'seed-validation.json').read_text())['status']=='passed'
    run=ROOT/f'seed_{args.seed}'/args.phase
    p=json.loads((run/'experiment/plan.json').read_text())
    manifest=json.loads((run/'experiment/source-manifest.json').read_text())['files']
    for relative,expected in manifest.items(): assert sha(run/relative)==expected,relative
    assert not (run/('status.json' if args.phase=='distill' else 'eval_status.json')).exists()
    source=None
    if args.phase=='eval':
        state,source=check_evaluation_gate(Path(p['distillation_root']),args.seed)
    # Keep the concrete plan to one active H100 job at a time.
    for submitted in ROOT.glob('seed_*/*/submission.json'):
        prior=json.loads(submitted.read_text())
        if prior.get('job_id'):
            query=subprocess.run(['qstat','-j',prior['job_id']],capture_output=True,text=True)
            assert not re.search(r'job_number:\s*'+re.escape(prior['job_id'])+r'\b',query.stdout), 'Another study job is still active'
            assert 'Following jobs do not exist' in query.stderr+query.stdout, 'Cannot confirm that the prior job has exited'
    attempt={'phase':args.phase,'seed':args.seed,
             'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'command':['qsub','-g','tga-lab_otm',args.phase+'_job.sh']}
    with (run/'submission-attempt.json').open('x') as handle: json.dump(attempt,handle,indent=2)
    if source is not None:
        destination=run/'input/flowers102/dinov2_vitb'/p['run_name']/'data.pth'
        destination.parent.mkdir(parents=True,exist_ok=False)
        with source.open('rb') as src,destination.open('xb') as dst: shutil.copyfileobj(src,dst)
        destination.chmod(0o444)
        assert sha(destination)==state['data_sha256']
        droot=Path(p['distillation_root'])
        files=[source,droot/'generator_final.pth',droot/'checkpoints/step_05000_final.pth',droot/'initialization.json',droot/'status.json']
        with (run/'input-manifest.json').open('x') as handle: json.dump({str(f):sha(f) for f in files},handle,indent=2)
    result=subprocess.run(attempt['command'],cwd=run,capture_output=True,text=True)
    record=dict(attempt,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    match=re.search(r'Your job (\d+)\b',result.stdout)
    if result.returncode==0 and match: record['job_id']=match.group(1)
    with (run/'submission.json').open('x') as handle: json.dump(record,handle,indent=2)
    print(json.dumps(record,indent=2))
    assert result.returncode==0 and match,'Submission failed or uncertain: do not retry'


if __name__=='__main__':
    main()
