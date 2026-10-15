"""Observe native checkpoint saves; accepts torch.save's actual `f` keyword."""
from pathlib import Path
import shutil


def observe_saves(original_save, snapshots, on_checkpoint):
    def recording_save(obj, f, *args, **kwargs):
        result = original_save(obj, f, *args, **kwargs)
        if isinstance(obj, dict) and {'current_epoch', 'top1_list', 'top5_list', 'fc'}.issubset(obj):
            count = len(obj['top1_list'])
            target = snapshots / f'evaluation_{count+1:02d}_epoch_{obj["current_epoch"]:05d}.pth'
            assert not target.exists(), 'Refusing to overwrite a checkpoint snapshot'
            shutil.copy2(Path(f), target)
            on_checkpoint(obj)
        return result
    return recording_save
