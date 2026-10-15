"""CPU-only reproducibility tests of the real Generator and experiment seed hook."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from seed_control import set_distillation_seed
from aggregate import hierarchical_summary
from submit import check_evaluation_gate


def digest(tensors):
    h=hashlib.sha256()
    for x in tensors: h.update(x.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    from config import DistillCfg,EvalCfg
    from synsets.generator import GeneratorDataset
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    study=json.loads((ROOT/'study-plan.json').read_text())
    baseline=Path(study['baseline_distillation_root'])
    oldplan=json.loads((baseline/'experiment/plan.json').read_text())
    assert oldplan['seed']==3407
    oldrunner=(baseline/'experiment/run_distill.py').read_text()
    assert 'torch.manual_seed(3407)' in oldrunner and 'module.main(cfg)' in oldrunner
    ckpt=torch.load(baseline/'checkpoints/step_05000_final.pth',map_location='cpu',weights_only=False)
    generator=torch.Generator().set_state(ckpt['random_state']['torch'])
    assert generator.initial_seed()==3407
    cpu_results={}
    for path in (ROOT/'src').rglob('*.py'):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==hashlib.sha256((baseline/'src'/path.relative_to(ROOT/'src')).read_bytes()).hexdigest()
    for seed in [3407,42,123,2026]:
        repetitions=[]
        for _ in range(2):
            set_distillation_seed(seed)
            assert torch.initial_seed()==seed
            random_values=[float(np.random.rand()),random.random()]
            # Include RNG-consuming iterator construction before the synset.
            # This is a CPU reproducibility check, not a bitwise replay of GPU/workers.
            loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(102,1)),batch_size=102,shuffle=True)
            iterator=iter(loader)
            next(iterator)
            cfg=DistillCfg(explicit_bool=True).parse_args([
                '--dataset=flowers102','--model=dinov2_vitb','--distill_mode=generator_pixel',
                '--distill_opt=adam','--ipc=1','--lr=0.002','--syn_res=256'])
            dataset=GeneratorDataset(SimpleNamespace(num_classes=102),cfg,device='cpu')
            assert dataset.noise.shape==(102,128) and not dataset.noise.requires_grad
            assert torch.equal(dataset.syn_labels,torch.arange(102))
            assert {id(p) for p in dataset.generator.parameters()}=={id(p) for g in dataset.optimizer.param_groups for p in g['params']}
            weights_before=digest(dataset.generator.parameters())
            noise_before=digest([dataset.noise])
            with torch.no_grad():
                preview=dataset.generator(dataset.noise[:1])
            assert preview.shape==(1,3,32,32) and torch.isfinite(preview).all()
            assert noise_before==digest([dataset.noise])
            repetitions.append({'weights_sha256':weights_before,'noise_sha256':noise_before,'other_rng_values':random_values})
            del iterator,loader,dataset
        assert repetitions[0]==repetitions[1]
        cpu_results[str(seed)]=repetitions[0]
    assert len({x['weights_sha256'] for x in cpu_results.values()})==4
    assert len({x['noise_sha256'] for x in cpu_results.values()})==4
    for seed in study['distillation_seeds']:
        root=ROOT/f'seed_{seed}'
        dp=json.loads((root/'distill/experiment/plan.json').read_text())
        assert dp['seed']==seed
        ignore={'job_tag','run_name'}
        assert {k:v for k,v in dp['distillation'].items() if k not in ignore}=={k:v for k,v in oldplan['distillation'].items() if k not in ignore}
        source=(root/'distill/experiment/run_distill.py').read_text()
        assert "set_distillation_seed(PLAN['seed'])" in source and 'torch.manual_seed(3407)' not in source
        assert source.index("set_distillation_seed(PLAN['seed'])")<source.index('module.main(cfg)')
        ep=json.loads((root/'eval/experiment/plan.json').read_text())
        expected=json.loads(Path(study['baseline_eval_root'],'experiment/plan.json').read_text())
        assert ep['evaluation']==expected['evaluation'] and ep['evaluation_seeds']==[3407,3408,3409]
        actual=EvalCfg(explicit_bool=True).parse_args(['--'+k+'='+str(v) for k,v in ep['evaluation'].items()])
        assert actual.num_eval==3 and actual.eval_epochs==1000
        for phase in ['distill','eval']:
            run=root/phase
            manifest=json.loads((run/'experiment/source-manifest.json').read_text())['files']
            for rel,expected_hash in manifest.items(): assert hashlib.sha256((run/rel).read_bytes()).hexdigest()==expected_hash, str(run/rel)
            assert not list(run.glob('*status.json')) and not (run/'submission-attempt.json').exists()
    fixture=hierarchical_summary({42:[0.8,0.9,1.0],123:[0.6,0.7,0.8],2026:[0.4,0.5,0.6]})
    assert math.isclose(fixture['mean_of_seed_means'],0.7)
    assert math.isclose(fixture['std_of_seed_means_ddof0'],math.sqrt(2/75))
    assert math.isclose(fixture['per_seed']['42']['std_eval_ddof0'],math.sqrt(1/150))
    assert not math.isclose(fixture['std_of_seed_means_ddof0'],math.sqrt(1/30))
    # Test the success-only evaluation gate with tiny synthetic files, without qsub.
    with tempfile.TemporaryDirectory(dir=ROOT/'cache') as directory:
        directory=Path(directory)
        data=directory/'data.pth'
        data.write_bytes(b'gate-fixture-only-not-training-data')
        valid_state={'status':'success','completed_updates':5001,'distillation_seed':42,
                     'generator_init_cpu_seed':42,'fixed_noise_init_cuda_seed':42,
                     'fixed_noise_unchanged':True,'generator_parameters_updated':True,
                     'job_id':'fixture','data_path':str(data),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
        validation={'status':'passed','png_count':102}
        account={'jobnumber':'fixture','exit_status':'0','failed':'0'}
        def save_case(state,validation,account):
            for filename,content in [('status.json',state),('distillation-validation.json',validation),('job-accounting.json',account)]:
                (directory/filename).write_text(json.dumps(content))
        save_case(valid_state,validation,account)
        check_evaluation_gate(directory,42)
        cases=[(dict(valid_state,status='error'),validation,account),
               (dict(valid_state,completed_updates=5000),validation,account),
               (dict(valid_state,distillation_seed=123),validation,account),
               (valid_state,dict(validation,png_count=101),account),
               (valid_state,validation,dict(account,exit_status='1')),
               (dict(valid_state,data_sha256='incorrect'),validation,account)]
        for state,v,a in cases:
            save_case(state,v,a)
            try: check_evaluation_gate(directory,42)
            except AssertionError: pass
            else: raise AssertionError('Evaluation gate accepted a failed/invalid distillation')
    for path,expected in study['preserved_artifacts'].items():
        h=hashlib.sha256()
        with Path(path).open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''): h.update(block)
        assert h.hexdigest()==expected,path
    result={'status':'passed','existing_distillation_seed':3407,'baseline_checkpoint_cpu_seed':3407,
            'cpu_initialization_tests':cpu_results,'same_seed_identical_weights_and_noise':True,
            'different_seeds_different_weights_and_noise':True,'config_except_seed_and_output_metadata_unchanged':True,
            'eval_seeds':[3407,3408,3409],'hierarchical_statistics_test':'passed',
            'evaluation_success_gate_test':'passed','production_sources_identical_to_3407':True,
            'cpu_gpu_initialization_bitwise_equivalence_tested':False,'gpu_jobs_submitted':0}
    with (ROOT/'seed-validation.json').open('x') as handle: json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
