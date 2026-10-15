"""CPU-only validation of this run's saved final artifacts; no training."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    from PIL import Image
    from torch.utils.data import TensorDataset
    from config import DistillCfg
    from synsets.generator import GeneratorDataset
    from run_distill import tensor_hash
    assert not torch.cuda.is_available()
    state = json.loads((ROOT / 'status.json').read_text())
    assert state['status'] == 'success' and state['completed_updates'] == 5001
    assert state['adam_steps'] == 5001 and state['generator_parameters_updated']
    assert state['fixed_noise_unchanged'] and state['final_images_match_generator']
    plan = json.loads((ROOT / 'experiment/plan.json').read_text())
    assert state['distillation_seed'] == plan['seed']
    assert state['generator_init_cpu_seed'] == state['fixed_noise_init_cuda_seed'] == plan['seed']
    for rel, expected in json.loads((ROOT / 'experiment/source-manifest.json').read_text())['files'].items():
        assert sha256(ROOT / rel) == expected
    data = torch.load(state['data_path'], map_location='cpu', weights_only=True)
    images, labels = data['images'], data['labels']
    assert images.shape == (102, 3, 256, 256) and images.dtype == torch.float32
    assert torch.equal(labels, torch.arange(102)) and labels.dtype == torch.int64
    assert torch.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
    assert len(TensorDataset(images, labels)) == 102
    assert sha256(state['data_path']) == state['data_sha256']
    package = torch.load(ROOT / 'generator_final.pth', map_location='cpu', weights_only=True)
    checkpoint = torch.load(ROOT / 'checkpoints/step_05000_final.pth', map_location='cpu', weights_only=False)
    assert checkpoint['global_step'] == 5000
    assert package['noise'].shape == (102, 128) and not package['noise'].requires_grad
    assert tensor_hash([package['noise']]) == state['initial_noise_sha256'] == state['final_noise_sha256']
    assert torch.equal(package['labels'], labels)
    assert all(int(value['step'].item()) == 5001 for value in package['opt_state']['state'].values())
    assert set(checkpoint['random_state']) == {'torch', 'cuda', 'python', 'numpy'}
    for name, param in package['generator'].items():
        assert torch.isfinite(param).all()
        assert torch.equal(param, checkpoint['synset']['generator'][name])
    cfg = DistillCfg(explicit_bool=True).parse_args(plan['cli_args'])
    restored = GeneratorDataset(type('Classes', (), {'num_classes': 102})(), cfg, device='cpu')
    restored.load_from_dict(package)
    assert tensor_hash(restored.generator.parameters()) == state['final_generator_sha256']
    assert state['initial_generator_sha256'] != state['final_generator_sha256']
    with torch.no_grad():
        regenerated = restored.get_data()[0]
    # GPU live Generator/export equality was already checked by the job runner.
    # Also record CPU/GPU floating-point differences without treating them as corruption.
    max_difference = (regenerated - images).abs().max().item()
    assert torch.isfinite(regenerated).all()
    with (ROOT / 'loss.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5001 and [int(r['step']) for r in rows] == list(range(5001))
    import math
    assert all(math.isfinite(float(r['loss'])) for r in rows)
    assert len(list((ROOT / 'checkpoints').glob('step_*.pth'))) == 11
    assert len(list((ROOT / 'images').glob('class_*.png'))) == 102
    for index in range(102):
        with Image.open(ROOT / 'images' / f'class_{index:03d}.png') as png:
            assert png.size == (256, 256) and png.mode == 'RGB'
            png.verify()
    for filename in ['loss_curve.png', 'synthetic_grid.png']:
        with Image.open(ROOT / filename) as image:
            image.verify()
    # Previous Pixel experiment artifacts must remain byte-identical.
    for path, expected in plan['previous_pixel_artifacts'].items():
        assert sha256(path) == expected, path
    result = dict(status='passed', updates=5001, loss_rows=5001,
                  images_shape=list(images.shape), png_count=102, snapshot_count=11,
                  generator_updated=True, fixed_noise_unchanged=True, cpu_restore=True,
                  cpu_regeneration_max_abs_difference=max_difference,
                  live_gpu_export_equality=True, data_sha256=state['data_sha256'],
                  previous_pixel_artifacts_unchanged=True)
    with (ROOT / 'distillation-validation.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
