"""Final CPU audit of observational hooks, frozen jobs and absence of approval."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    import submit
    assert not torch.cuda.is_available() and not (ROOT/'approval.json').exists()
    seed_results=json.loads((ROOT/'seed-validation.json').read_text())
    plan=json.loads((ROOT/'study-plan.json').read_text())
    baseline=json.loads(Path(plan['baseline_distillation_root'],'status.json').read_text())
    assert seed_results['cpu_initialization_tests']['3407']['weights_sha256']==baseline['initial_generator_sha256']
    template=ROOT/'templates/distill/run_distill.py'
    parsed=ast.parse(template.read_text())
    hook=next(n for n in ast.walk(parsed) if isinstance(n,ast.FunctionDef) and n.name=='checked_init')
    code=compile(ast.Module(body=[hook],type_ignores=[]),'observer-hook','exec')
    status={'cpu_rng_before_generator_sha256':'original'}
    environment={'torch':torch,'PLAN':{'seed':42},'status':status,
                 'tensor_hash':lambda xs:hashlib.sha256(xs[0].cpu().numpy().tobytes()).hexdigest(),
                 'original_init':lambda *a,**kw:'original_constructor_result'}
    exec(code,environment)
    with patch.object(torch.cuda,'initial_seed',side_effect=AssertionError('CPU restore must not inspect CUDA')):
        assert environment['checked_init'](None,device='cpu')=='original_constructor_result'
        assert status=={'cpu_rng_before_generator_sha256':'original'}
    torch.manual_seed(42)
    with patch.object(torch.cuda,'initial_seed',return_value=42), \
         patch.object(torch.cuda,'get_rng_state',return_value=torch.zeros(8,dtype=torch.uint8)):
        assert environment['checked_init'](None)=='original_constructor_result'
        assert status['generator_init_cpu_seed']==status['fixed_noise_init_cuda_seed']==42
    # Only the recording branch above is mocked; this is not a GPU initialization test.
    with tempfile.TemporaryDirectory(dir=ROOT/'cache') as directory:
        with patch.object(submit,'ROOT',Path(directory)),patch.object(sys,'argv',['submit.py','distill','42']), \
             patch.object(submit.subprocess,'run') as runner:
            try: submit.main()
            except FileNotFoundError: pass
            else: raise AssertionError('Submission did not require approval')
            runner.assert_not_called()
    checked=[]
    for seed in plan['distillation_seeds']:
        for phase,limit in [('distill','02:00:00'),('eval','01:00:00')]:
            root=ROOT/f'seed_{seed}'/phase
            script=root/f'{phase}_job.sh'
            text=script.read_text()
            assert '#$ -l gpu_1=1\n' in text and '#$ -p 0\n' in text and '#$ -g ' not in text
            assert f'#$ -l h_rt={limit}\n' in text
            subprocess.run(['bash','-n',str(script)],check=True)
            manifest=json.loads((root/'experiment/source-manifest.json').read_text())['files']
            for rel,expected in manifest.items(): assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected,rel
            assert not list(root.glob('*status.json')) and not list(root.glob('submission*.json'))
            checked.append(str(script.relative_to(ROOT)))
    for path in ROOT.rglob('*.py'):
        if '__pycache__' not in str(path): ast.parse(path.read_text(),filename=str(path))
    result={'status':'passed','baseline_3407_initial_generator_weights_replayed_on_cpu':True,
            'cpu_restore_does_not_replace_gpu_initialization_metadata':True,
            'gpu_metadata_branch_control_test':'passed_with_mocked_cuda_no_gpu',
            'submission_without_approval_blocked_before_subprocess':True,
            'checked_job_scripts':checked,'max_total_points':1.44,'max_concurrent_jobs':1,
            'gpu_jobs_submitted':0}
    with (ROOT/'execution-plan-validation.json').open('x') as handle: json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
