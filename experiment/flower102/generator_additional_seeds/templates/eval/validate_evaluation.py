"""CPU verification of three measured repetitions and independent initial states."""
import json

from common import DATA, PLAN, ROOT, RUN, sha256, verify_sources, write_json


def main():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    verify_sources()
    status = json.loads((ROOT/'eval_status.json').read_text())
    assert status['status'] == 'success'
    rows = status['completed_evaluations']
    assert len(rows) == 3
    assert [row['seed'] for row in rows] == [3407, 3408, 3409]
    assert sha256(DATA) == status['data_sha256']
    heads = []
    initial_seeds = []
    for index in range(3):
        path = ROOT/'evaluation-checkpoints'/f'evaluation_{index+1:02d}_epoch_00000.pth'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['current_epoch'] == 0
        assert len(checkpoint['top1_list']) == len(checkpoint['top5_list']) == index
        assert checkpoint['optimizer']['state'] == {}
        generator = torch.Generator(device='cpu')
        generator.set_state(checkpoint['random_state']['torch'])
        seed = generator.initial_seed()
        assert seed == 3407+index
        initial_seeds.append(seed)
        weight = checkpoint['fc']['linear.weight']
        assert tuple(weight.shape) == (102, 768)
        heads.append(weight)
    assert all(not torch.equal(heads[i], heads[j]) for i in range(3) for j in range(i))
    summary = torch.load(RUN/'eval/dinov2_vitb.pth', map_location='cpu', weights_only=True)
    for metric in ['top1', 'top5']:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        assert np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 1))
        assert np.isclose(summary[metric+'_mean'], np.mean(values))
        assert np.isclose(summary[metric+'_std'], np.std(values, ddof=0))
        assert np.isclose(summary[metric+'_mean'], status['summary'][metric+'_mean'])
    report = {'status': 'passed', 'job_id': status['job_id'],
              'initial_checkpoint_seeds': initial_seeds, 'independent_initial_weights': True,
              'empty_optimizer_state_each_initialization': True, 'completed_evaluations': rows,
              'summary': summary, 'std_ddof': 0, 'same_data_sha256': sha256(DATA)}
    write_json(ROOT/'evaluation-validation.json', report, exclusive=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
