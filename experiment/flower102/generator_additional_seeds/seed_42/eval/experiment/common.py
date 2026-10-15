"""Experiment bookkeeping only; LGM's numerical routines remain unchanged."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())
ROOT = Path(PLAN['experiment_dir'])
REUSE = Path(PLAN['reuse_dir'])
SRC = ROOT / 'src'
JOB_TAG = ROOT / 'input'
RUN = JOB_TAG / 'flowers102/dinov2_vitb' / PLAN['run_name']
DATA = RUN / 'data.pth'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
        temporary.replace(path)


def cli_options(settings):
    return ['--' + key + '=' + str(value) for key, value in settings.items()]


def eval_args():
    return cli_options(PLAN['evaluation']) + [
        '--data_root=' + str(REUSE / 'data'), '--job_tag=' + str(JOB_TAG),
        '--skip_if_exists=True',
    ]


def verify_sources():
    manifest = json.loads((HERE / 'source-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        assert sha256(ROOT / relative) == expected, 'Source changed: ' + relative
    return manifest


def configure_gpu():
    verify_sources()
    assert os.environ.get('WANDB_MODE') == 'offline'
    os.chdir(SRC)
    sys.path.insert(0, str(SRC))
    import torch
    import subprocess
    subprocess.run(['nvidia-smi'], check=True)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.multiprocessing.set_sharing_strategy('file_system')
    manifest = json.loads((REUSE / 'assets-manifest.json').read_text())
    assert sha256(manifest['weights']) == manifest['weights_sha256']
    original_load = torch.hub.load
    def cached_load(repo_or_dir, model, *args, **kwargs):
        assert repo_or_dir == 'facebookresearch/dinov2' and model == 'dinov2_vitb14'
        kwargs['source'] = 'local'
        return original_load(manifest['dinov2_hub_repo'], model, *args, **kwargs)
    torch.hub.load = cached_load
    torch.cuda.reset_peak_memory_stats()
    return {
        'job_id': os.environ.get('JOB_ID'), 'host': os.uname().nodename,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'cuda_build': torch.version.cuda, 'source_commit': PLAN['source_commit'],
    }
