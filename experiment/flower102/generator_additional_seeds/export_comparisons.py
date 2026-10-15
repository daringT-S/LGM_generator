"""Export measured accuracies and a panel of saved PNGs after validation."""
import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'study-plan.json').read_text())
    assert json.loads((root / 'execution-status.json').read_text())['status'] == 'complete'
    records = []
    for seed in plan['distillation_seeds']:
        run = root / f'seed_{seed}' / 'eval'
        assert json.loads((run / 'evaluation-validation.json').read_text())['status'] == 'passed'
        state = json.loads((run / 'eval_status.json').read_text())
        for row in state['completed_evaluations']:
            records.append(dict(distillation_seed=seed, evaluation=row['evaluation'],
                                evaluation_seed=row['seed'], top1_percent=100*row['top1'],
                                top5_percent=100*row['top5']))
    assert len(records) == 9
    with (root / 'accuracies.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    # Display the same class labels across independent distillation seeds.
    labels = [0, 25, 50, 75]
    fig, axes = plt.subplots(3, len(labels), figsize=(9, 7))
    for row, seed in enumerate(plan['distillation_seeds']):
        for col, label in enumerate(labels):
            with Image.open(root / f'seed_{seed}/distill/images/class_{label:03d}.png') as image:
                axes[row, col].imshow(image)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(f'Class {label}')
            if col == 0:
                axes[row, col].set_ylabel(f'Distillation seed {seed}')
    fig.tight_layout()
    fig.savefig(root / 'synthetic_samples_comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'accuracy_rows': len(records), 'samples': 12}))


if __name__ == '__main__':
    main()
