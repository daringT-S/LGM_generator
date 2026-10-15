"""Record 5,001 original LGM updates. Never submit or retry jobs."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

from seed_control import set_distillation_seed

ROOT = Path(__file__).resolve().parents[1]
REUSE = Path('/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv')
PLAN = json.loads((ROOT / 'experiment/plan.json').read_text())
RUN_NAME = PLAN['distillation']['run_name']
JOB_TAG = ROOT / 'distillation'
RUN = JOB_TAG / 'flowers102/dinov2_vitb' / RUN_NAME


def digest_file(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def tensor_hash(tensors):
    hasher = hashlib.sha256()
    for tensor in tensors:
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def write_status(status):
    temporary = ROOT / 'status.json.tmp'
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2))
    temporary.replace(ROOT / 'status.json')


def main():
    assert not RUN.exists(), 'Refusing to overwrite an existing run'
    # Exclusive creation prevents a second run of this experiment directory.
    with (ROOT / 'status.json').open('x') as handle:
        handle.write('{"status":"starting"}')
    status = {'status': 'running', 'job_id': os.environ.get('JOB_ID'), 'completed_updates': 0}
    started = time.perf_counter()
    torch = None
    step_file = None
    experiment_started = None
    try:
        assert status['job_id'], 'Run only within a scheduled GPU job'
        assert os.environ.get('WANDB_MODE') == 'offline'
        manifest = json.loads((ROOT / 'experiment/source-manifest.json').read_text())
        for relative, expected in manifest['files'].items():
            assert digest_file(ROOT / relative) == expected, relative
        sys.path.insert(0, str(ROOT / 'src'))
        import numpy as np
        import torch as torch_module
        torch = torch_module
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert 'H100' in torch.cuda.get_device_name(0)
        status['gpu'] = torch.cuda.get_device_name(0)
        status['torch'] = torch.__version__
        status['cuda'] = torch.version.cuda
        # Same second-order-compatible attention setup as the Pixel experiment.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.multiprocessing.set_sharing_strategy('file_system')
        assets = json.loads((REUSE / 'assets-manifest.json').read_text())
        assert digest_file(assets['weights']) == assets['weights_sha256']
        original_hub_load = torch.hub.load

        def cached_load(repo_or_dir, model, *args, **kwargs):
            assert repo_or_dir == 'facebookresearch/dinov2' and model == 'dinov2_vitb14'
            kwargs['source'] = 'local'
            return original_hub_load(assets['dinov2_hub_repo'], model, *args, **kwargs)

        torch.hub.load = cached_load
        from config import DistillCfg
        from synsets.generator import GeneratorDataset
        import distillation.distill as module
        set_distillation_seed(PLAN['seed'])
        status['distillation_seed'] = PLAN['seed']
        # Observe the original constructor without reseeding or changing its operations.
        original_init = GeneratorDataset.__init__
        def checked_init(dataset, *args, **kwargs):
            if torch.device(kwargs.get('device', 'cuda')).type != 'cuda':
                return original_init(dataset, *args, **kwargs)
            assert torch.initial_seed() == PLAN['seed']
            assert torch.cuda.initial_seed() == PLAN['seed']
            status['generator_init_cpu_seed'] = torch.initial_seed()
            status['fixed_noise_init_cuda_seed'] = torch.cuda.initial_seed()
            status['cpu_rng_before_generator_sha256'] = tensor_hash([torch.get_rng_state()])
            status['cuda_rng_before_noise_sha256'] = tensor_hash([torch.cuda.get_rng_state()])
            original_init(dataset, *args, **kwargs)
        GeneratorDataset.__init__ = checked_init
        plan = json.loads((ROOT / 'experiment/plan.json').read_text())
        cfg = DistillCfg(explicit_bool=True).parse_args(plan['cli_args'])
        assert cfg.distill_mode == 'generator_pixel' and cfg.distill_opt == 'adam'
        assert cfg.iterations == 5000 and cfg.ipc == 1 and type(cfg.lr) is float
        assert cfg.device_count == 1
        status['config'] = cfg.as_dict()
        (ROOT / 'checkpoints').mkdir(exist_ok=False)
        noise_hash = None
        parameters_before = None
        step_file = (ROOT / 'loss.csv').open('x', newline='', buffering=1)
        writer = csv.DictWriter(step_file, fieldnames=['step', 'loss', 'seconds', 'elapsed_seconds'])
        writer.writeheader()

        class ObservedLinearGM(module.LinearGM):
            def __init__(self, *args, **kwargs):
                nonlocal noise_hash, parameters_before
                super().__init__(*args, **kwargs)
                ds = self.distilled_dataset
                assert type(ds) is GeneratorDataset
                assert type(ds.optimizer) is torch.optim.Adam
                assert ds.noise.shape == (102, 128) and not ds.noise.requires_grad
                assert torch.equal(ds.syn_labels, torch.arange(102, device='cuda'))
                assert {id(p) for p in ds.generator.parameters()} == {
                    id(p) for group in ds.optimizer.param_groups for p in group['params']
                }
                noise_hash = tensor_hash([ds.noise])
                parameters_before = tensor_hash(ds.generator.parameters())
                status['initial_noise_sha256'] = noise_hash
                status['initial_generator_sha256'] = parameters_before
                with (ROOT / 'initialization.json').open('x') as handle:
                    json.dump({key: status[key] for key in [
                        'distillation_seed', 'generator_init_cpu_seed', 'fixed_noise_init_cuda_seed',
                        'cpu_rng_before_generator_sha256', 'cuda_rng_before_noise_sha256',
                        'initial_noise_sha256', 'initial_generator_sha256',
                    ]}, handle, indent=2)
                status['dataset_class'] = type(ds).__name__
                status['generator_parameters'] = sum(p.numel() for p in ds.generator.parameters())
                write_status(status)

            def match_gradients(self):
                tick = time.perf_counter()
                loss = super().match_gradients()
                torch.cuda.synchronize()
                ds = self.distilled_dataset
                assert math.isfinite(loss)
                assert not ds.noise.requires_grad and ds.noise.grad is None
                if self.global_step % 100 == 0 or self.global_step == cfg.iterations:
                    assert tensor_hash([ds.noise]) == noise_hash
                    for param in ds.generator.parameters():
                        assert param.grad is not None and torch.isfinite(param.grad).all()
                        assert int(ds.optimizer.state[param]['step'].item()) == self.global_step + 1
                row = {'step': self.global_step, 'loss': loss,
                       'seconds': time.perf_counter() - tick,
                       'elapsed_seconds': time.perf_counter() - experiment_started}
                writer.writerow(row)
                status.update(completed_updates=self.global_step + 1, last_step=self.global_step,
                              last_loss=loss, elapsed_seconds=row['elapsed_seconds'])
                if self.global_step == 0:
                    status['first_loss'] = loss
                if self.global_step % 100 == 0 or self.global_step == cfg.iterations:
                    status['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    write_status(status)
                    print('STEP_LOSS', json.dumps(row), flush=True)
                if self.global_step == cfg.iterations:
                    status['fixed_noise_unchanged'] = True
                    status['final_noise_sha256'] = tensor_hash([ds.noise])
                    status['final_generator_sha256'] = tensor_hash(ds.generator.parameters())
                    status['generator_parameters_updated'] = status['final_generator_sha256'] != parameters_before
                    assert status['generator_parameters_updated']
                    status['adam_steps'] = 5001
                return loss

            def save_checkpoint(self):
                super().save_checkpoint()
                if self.global_step % 500 == 0 or self.global_step == cfg.iterations:
                    suffix = '_final' if self.global_step == cfg.iterations else ''
                    target = ROOT / 'checkpoints' / f'step_{self.global_step:05d}{suffix}.pth'
                    assert not target.exists()
                    shutil.copy2(Path(self.log_dir) / 'ckpt.pth', target)

            def save_data(self):
                super().save_data()
                # Check native export against the live final Generator on the same device.
                exported = torch.load(Path(self.log_dir) / 'data.pth', map_location='cpu', weights_only=True)
                with torch.no_grad():
                    generated, labels = self.distilled_dataset.get_data()
                    torch.testing.assert_close(generated.cpu(), exported['images'], rtol=1e-5, atol=1e-6)
                    assert torch.equal(labels.cpu(), exported['labels'])
                status['final_images_match_generator'] = True

        module.LinearGM = ObservedLinearGM
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        experiment_started = time.perf_counter()
        module.main(cfg)
        torch.cuda.synchronize()
        status['experiment_seconds'] = time.perf_counter() - experiment_started
        assert status['completed_updates'] == 5001
        data_path = RUN / 'data.pth'
        data = torch.load(data_path, map_location='cpu', weights_only=True)
        images, labels = data['images'], data['labels']
        assert images.shape == (102, 3, 256, 256)
        assert torch.equal(labels, torch.arange(102))
        assert torch.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
        # Keep a CPU-portable Generator package alongside the native final checkpoint.
        checkpoint = torch.load(ROOT / 'checkpoints/step_05000_final.pth',
                                map_location='cpu', weights_only=False)
        assert checkpoint['global_step'] == 5000
        package = checkpoint['synset']
        assert tensor_hash([package['noise']]) == noise_hash
        assert torch.equal(package['labels'], labels)
        assert all(int(s['step'].item()) == 5001 for s in package['opt_state']['state'].values())
        assert not (ROOT / 'generator_final.pth').exists()
        torch.save(package, ROOT / 'generator_final.pth')
        restored = GeneratorDataset(type('Classes', (), {'num_classes': 102})(), cfg, device='cpu')
        restored.load_from_dict(package)
        assert tensor_hash(restored.generator.parameters()) == status['final_generator_sha256']
        assert not restored.noise.requires_grad
        status['checkpoint_restored_on_cpu'] = True
        from torchvision.utils import save_image
        (ROOT / 'images').mkdir(exist_ok=False)
        for label, image in enumerate(images):
            save_image(image, ROOT / 'images' / f'class_{label:03d}.png')
        save_image(images, ROOT / 'synthetic_grid.png', nrow=10)
        step_file.close()
        step_file = None
        with (ROOT / 'loss.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 5001 and [int(r['step']) for r in rows] == list(range(5001))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([int(r['step']) for r in rows], [float(r['loss']) for r in rows], linewidth=0.7)
        ax.set(xlabel='Step', ylabel='LGM loss', title='Generator: Flowers102 / DINOv2 ViT-B')
        fig.tight_layout()
        fig.savefig(ROOT / 'loss_curve.png')
        plt.close(fig)
        status.update(status='success', data_path=str(data_path), data_sha256=digest_file(data_path),
                      shape=list(images.shape), png_count=len(list((ROOT / 'images').glob('*.png'))))
    except BaseException as exc:
        status.update(status='error', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        if step_file is not None:
            step_file.close()
        status['total_seconds'] = time.perf_counter() - started
        if torch is not None and torch.cuda.is_initialized():
            status['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            status['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        write_status(status)
        print('GENERATOR_STATUS', json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
