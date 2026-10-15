import csv
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

from common import DATA, PLAN, ROOT, RUN, configure_gpu, distill_args, sha256, write_json


def main():
    status_path = ROOT / 'status.json'
    assert not status_path.exists(), 'Refusing to rerun or overwrite an existing attempt'
    assert not RUN.exists(), 'Run directory must not already exist'
    status = {'status': 'running', 'completed_updates': 0, 'config': PLAN['distillation']}
    write_json(status_path, status, exclusive=True)
    started = time.perf_counter()
    torch = None
    step_file = None
    experiment_started = None
    try:
        status.update(configure_gpu())
        import numpy as np
        import torch as torch_module
        torch = torch_module
        from config import DistillCfg
        from synsets.pixels import PixelDataset
        import distillation.distill as module
        torch.manual_seed(PLAN['distillation_seed'])
        np.random.seed(PLAN['distillation_seed'])
        random.seed(PLAN['distillation_seed'])
        sys.argv = ['distill.py', *distill_args()]
        cfg = DistillCfg(explicit_bool=True).parse_args()
        assert type(cfg.lr) is float and cfg.lr == PLAN['distillation']['lr']
        for key in ['iterations', 'ipc', 'augs_per_batch', 'num_workers', 'checkpoint_it', 'image_log_it', 'syn_res', 'real_res', 'crop_res']:
            assert type(getattr(cfg, key)) is int, key + ' must be int'
        assert cfg.device_count == 1 and cfg.ipc == 1 and cfg.iterations == 5000
        original_class = module.LinearGM
        checkpoint_root = ROOT / 'checkpoints'
        checkpoint_root.mkdir(exist_ok=False)
        step_file = (ROOT / 'loss.csv').open('x', newline='', buffering=1)
        writer = csv.DictWriter(step_file, fieldnames=['step', 'loss', 'step_seconds', 'elapsed_seconds'])
        writer.writeheader()
        class RecordingLinearGM(original_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert type(self.distilled_dataset) is PixelDataset
                assert type(self.distilled_dataset.optimizer) is torch.optim.Adam
                assert not self.distilled_dataset.optimizer.state
                status['dataset_class'] = type(self.distilled_dataset).__name__
                status['optimizer_class'] = type(self.distilled_dataset.optimizer).__name__
                status['initial_parameter_sha256'] = __import__('hashlib').sha256(self.distilled_dataset.syn_images.detach().cpu().numpy().tobytes()).hexdigest()
                print('Verified PixelDataset and Adam', flush=True)

            def match_gradients(self):
                tick = time.perf_counter()
                loss = super().match_gradients()
                torch.cuda.synchronize()
                assert math.isfinite(loss), f'Non-finite loss at {self.global_step}'
                record = {'step': int(self.global_step), 'loss': float(loss),
                          'step_seconds': time.perf_counter() - tick,
                          'elapsed_seconds': time.perf_counter() - experiment_started}
                writer.writerow(record)
                status['completed_updates'] += 1
                if self.global_step == 0: status['first_loss'] = float(loss)
                status['last_step'] = record['step']
                status['last_loss'] = record['loss']
                if self.global_step % 100 == 0 or self.global_step == cfg.iterations:
                    print('STEP_LOSS', json.dumps(record), flush=True)
                    write_json(status_path, status)
                if self.global_step == cfg.iterations:
                    parameter = self.distilled_dataset.syn_images
                    state = self.distilled_dataset.optimizer.state[parameter]
                    status['adam_step'] = int(state['step'].item())
                    status['parameters_updated'] = True
                    status['final_parameter_sha256'] = __import__('hashlib').sha256(parameter.detach().cpu().numpy().tobytes()).hexdigest()
                    assert status['adam_step'] == 5001
                    assert status['initial_parameter_sha256'] != status['final_parameter_sha256']
                return loss

            def save_checkpoint(self):
                super().save_checkpoint()
                if self.global_step % PLAN['checkpoint_snapshot_interval'] == 0 or self.global_step == cfg.iterations:
                    suffix = '_final' if self.global_step == cfg.iterations else ''
                    target = checkpoint_root / f'step_{self.global_step:05d}{suffix}.pth'
                    assert not target.exists()
                    shutil.copy2(Path(self.log_dir) / 'ckpt.pth', target)

        module.LinearGM = RecordingLinearGM
        torch.cuda.synchronize()
        experiment_started = time.perf_counter()
        module.main(cfg)
        torch.cuda.synchronize()
        status['experiment_seconds'] = time.perf_counter() - experiment_started
        assert status['completed_updates'] == 5001
        data = torch.load(DATA, map_location='cpu', weights_only=True)
        images, labels = data['images'], data['labels']
        assert tuple(images.shape) == (120, 3, 256, 256)
        assert torch.equal(labels, torch.arange(120))
        assert torch.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
        class_names = json.loads((Path(PLAN["study_root"]) / "data-validation.json").read_text())["class_names"]
        from torchvision.utils import save_image
        png_dir = ROOT / 'images'
        png_dir.mkdir(exist_ok=False)
        image_manifest = []
        for index, image in enumerate(images):
            filename = f'class_{index:03d}.png'
            save_image(image, png_dir / filename)
            image_manifest.append({'label': index, 'class_name': class_names[index], 'file': 'images/' + filename})
        save_image(images, ROOT / 'synthetic_grid.png', nrow=10)
        write_json(ROOT / 'image-manifest.json', image_manifest, exclusive=True)
        status.update(status='success', data_path=str(DATA), data_sha256=sha256(DATA),
                      images_shape=list(images.shape), data_bytes=DATA.stat().st_size)
    except BaseException as exc:
        status.update(status='error', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        if step_file is not None:
            step_file.close()
        status['total_seconds'] = time.perf_counter() - started
        if torch is not None and torch.cuda.is_initialized():
            status['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
            status['peak_reserved_gpu_memory_bytes'] = torch.cuda.max_memory_reserved()
        write_json(status_path, status)
        print('DISTILL_STATUS', json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
