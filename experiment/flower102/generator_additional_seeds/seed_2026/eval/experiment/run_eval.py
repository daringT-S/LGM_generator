"""Execute the official distillation.eval entry point once, with --num_eval=3."""
import json
from pathlib import Path
import runpy
import sys
import time
import traceback

from common import DATA, PLAN, ROOT, RUN, configure_gpu, eval_args, sha256, write_json
from save_observer import observe_saves


def main():
    status_path = ROOT / 'eval_status.json'
    assert not status_path.exists(), 'Refusing to rerun an existing evaluation attempt'
    distill = json.loads((Path(PLAN['distillation_root']) / 'status.json').read_text())
    assert distill['status'] == 'success' and distill['completed_updates'] == 5001
    assert DATA.exists() and sha256(DATA) == distill['data_sha256']
    inputs = json.loads((ROOT / 'input-manifest.json').read_text())
    for filename, expected in inputs.items():
        assert sha256(filename) == expected, 'Input changed: ' + filename
    assert not (RUN / 'eval').exists(), 'Refusing to overwrite previous evaluation files'
    status = {'status': 'running', 'completed_evaluations': [], 'config': PLAN['evaluation'],
              'data_path': str(DATA), 'data_sha256': sha256(DATA), 'std_ddof': 0}
    write_json(status_path, status, exclusive=True)
    started = time.perf_counter()
    torch = None
    original_save = None
    last_completed = 0
    try:
        status.update(configure_gpu())
        import numpy as np
        import torch as torch_module
        torch = torch_module
        snapshots = ROOT / 'evaluation-checkpoints'
        snapshots.mkdir(exist_ok=False)
        original_save = torch.save
        # Observe the official checkpoints; no model, optimizer, metric,
        # sampling or evaluation routine is replaced.
        def on_checkpoint(obj):
            nonlocal last_completed
            if {'current_epoch', 'top1_list', 'top5_list', 'fc'}.issubset(obj):
                count = len(obj['top1_list'])
                if count > last_completed:
                    status['completed_evaluations'] = [
                        {'evaluation': i + 1, 'seed': PLAN['evaluation_seeds'][i], 'top1': float(a), 'top5': float(b)}
                        for i, (a, b) in enumerate(zip(obj['top1_list'], obj['top5_list']))]
                    last_completed = count
                status['current_evaluation'] = count + 1
                status['current_epoch'] = int(obj['current_epoch'])
                write_json(status_path, status)
        torch.save = observe_saves(original_save, snapshots, on_checkpoint)
        sys.argv = ['distillation.eval', *eval_args()]
        namespace = runpy.run_module('distillation.eval', run_name='__main__')
        evaluator = namespace['evaluator']
        assert len(evaluator.top1_list) == len(evaluator.top5_list) == 3
        assert evaluator.random_seed == 3407
        status['completed_evaluations'] = [
            {'evaluation': i + 1, 'seed': PLAN['evaluation_seeds'][i], 'top1': float(a), 'top5': float(b)}
            for i, (a, b) in enumerate(zip(evaluator.top1_list, evaluator.top5_list))]
        summary = {key: float(namespace[key]) for key in ['top1_mean', 'top1_std', 'top5_mean', 'top5_std']}
        for key, values in [('top1', evaluator.top1_list), ('top5', evaluator.top5_list)]:
            assert all(np.isfinite(v) and 0 <= v <= 1 for v in values)
            assert np.isclose(summary[key+'_mean'], np.mean(values))
            assert np.isclose(summary[key+'_std'], np.std(values, ddof=0))
        saved = torch.load(RUN / 'eval/dinov2_vitb.pth', map_location='cpu', weights_only=False)
        assert saved == summary
        assert sha256(DATA) == status['data_sha256'], 'Synthetic images changed during evaluation'
        for filename, expected in inputs.items():
            assert sha256(filename) == expected, 'Input changed: ' + filename
        for filename, expected in PLAN['preserved_artifacts'].items():
            assert sha256(filename) == expected, 'Original artifact changed: ' + filename
        status.update(status='success', summary=summary, evaluator='official distillation.eval',
                      seeds=PLAN['evaluation_seeds'], classifier_reinitialized_each_repeat=True,
                      pixel_top1_reference_percent=PLAN['pixel_top1_mean_percent'],
                      top1_delta_percentage_points=100 * summary['top1_mean'] - PLAN['pixel_top1_mean_percent'])
    except BaseException as exc:
        status.update(status='error', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        if original_save is not None:
            torch.save = original_save
        status['total_seconds'] = time.perf_counter() - started
        if torch is not None and torch.cuda.is_initialized():
            status['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
            status['peak_reserved_gpu_memory_bytes'] = torch.cuda.max_memory_reserved()
        write_json(status_path, status)
        print('EVAL_STATUS', json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
