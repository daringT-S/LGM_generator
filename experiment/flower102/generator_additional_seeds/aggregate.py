"""Separate within-distillation evaluation variance from between-seed variance."""
import csv
import json
from pathlib import Path
import statistics


def hierarchical_summary(groups):
    if len(groups) != 3 or any(len(values) != 3 for values in groups.values()):
        raise ValueError('Require all three new distillation seeds and three evaluations each')
    means = [statistics.mean(values) for values in groups.values()]
    return {
        'per_seed': {str(seed): {'mean': statistics.mean(values), 'std_eval_ddof0': statistics.pstdev(values)}
                     for seed, values in groups.items()},
        'mean_of_seed_means': statistics.mean(means),
        'std_of_seed_means_ddof0': statistics.pstdev(means),
    }


def main():
    root = Path(__file__).resolve().parent
    plan = json.loads((root/'study-plan.json').read_text())
    rows, losses, runtimes = [], {}, {}
    groups = {'top1': {}, 'top5': {}}
    for seed in plan['distillation_seeds']:
        droot = root/f'seed_{seed}/distill'
        eroot = root/f'seed_{seed}/eval'
        dstate = json.loads((droot/'status.json').read_text()) if (droot/'status.json').exists() else {}
        estate = json.loads((eroot/'eval_status.json').read_text()) if (eroot/'eval_status.json').exists() else {}
        validation = json.loads((eroot/'evaluation-validation.json').read_text()) if (eroot/'evaluation-validation.json').exists() else {}
        records = estate.get('completed_evaluations', [])
        row = {'distillation_seed': seed, 'distillation_status': dstate.get('status', '未実施'),
               'evaluation_status': estate.get('status', '未実施'),
               'distillation_seconds': dstate.get('experiment_seconds'),
               'distillation_peak_gib': dstate['peak_allocated_bytes']/1024**3 if 'peak_allocated_bytes' in dstate else None}
        if estate.get('status')=='success' and validation.get('status')=='passed':
            assert [r['seed'] for r in records]==[3407,3408,3409]
            for metric in groups:
                values=[r[metric] for r in records]
                groups[metric][seed]=values
                row[metric+'_values']=values
                row[metric+'_mean']=statistics.mean(values)
                row[metric+'_std_eval_ddof0']=statistics.pstdev(values)
        rows.append(row)
        if (droot/'loss.csv').exists():
            with (droot/'loss.csv').open() as handle: losses[seed]=list(csv.DictReader(handle))
    result={'new_distillation_seeds':rows,'primary_new_seed_summary':{},
            'secondary_with_existing_3407':{},'ddof':0,'gpu_jobs_submitted_by_this_script':0}
    for metric, group in groups.items():
        if len(group)==3:
            result['primary_new_seed_summary'][metric]=hierarchical_summary(group)
            baseline=plan['baseline_evaluation']['summary'][metric+'_mean']
            means=[statistics.mean(v) for v in group.values()]+[baseline]
            result['secondary_with_existing_3407'][metric]={
                'seeds':[42,123,2026,3407], 'mean_of_seed_means':statistics.mean(means),
                'std_of_seed_means_ddof0':statistics.pstdev(means)}
    (root/'study-results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    lines=['# Generator蒸留シードの再現性比較','',
           'GPUジョブはユーザー承認後にのみ投入する。未取得の値は未実施とする。','',
           '| 蒸留seed | Top-1（評価3回）| Top-1 平均±評価std | Top-5（評価3回）| Top-5 平均±評価std | 蒸留秒 | GPU GiB |',
           '|---:|---|---|---|---|---:|---:|']
    for row in rows:
        columns=[]
        for metric in ['top1','top5']:
            values=row.get(metric+'_values')
            columns += [', '.join(f'{100*v:.4f}%' for v in values) if values else '未実施',
                        f"{100*row[metric+'_mean']:.4f}% ± {100*row[metric+'_std_eval_ddof0']:.4f} pt" if values else '未実施']
        lines += [f"| {row['distillation_seed']} | {' | '.join(columns)} | {row['distillation_seconds'] if row['distillation_seconds'] is not None else '未実施'} | {row['distillation_peak_gib'] if row['distillation_peak_gib'] is not None else '未実施'} |"]
    lines += ['', '評価内stdは各蒸留結果に対する3回の評価の母標準偏差（ddof=0）。',
              '蒸留seed間stdは42・123・2026の各評価平均を3点として計算する。9回の評価を独立な蒸留9回として数えない。',
              '既存3407を含む4点の集計は副次解析として別記する。', '',
              '## シード間集計','',
              '| 集計対象 | Top-1平均 | 蒸留seed間Top-1 std | Top-5平均 | 蒸留seed間Top-5 std |',
              '|---|---:|---:|---:|---:|']
    for label,key in [('新規42/123/2026（主解析）','primary_new_seed_summary'),('既存3407を含む4点（副次解析）','secondary_with_existing_3407')]:
        columns=[]
        for metric in ['top1','top5']:
            value=result[key].get(metric)
            columns += [f"{100*value['mean_of_seed_means']:.4f}%" if value else '未実施',
                        f"{100*value['std_of_seed_means_ddof0']:.4f} pt" if value else '未実施']
        lines.append('| '+label+' | '+' | '.join(columns)+' |')
    lines += ['', '## 各実験のファイル','']
    for seed in plan['distillation_seeds']:
        lines += [f'- seed {seed}: `seed_{seed}/distill/loss.csv`、`loss_curve.png`、`images/`、`generator_final.pth`、`checkpoints/`、`seed_{seed}/eval/eval_status.json`。']
    if len(groups['top1'])==3:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        for seed,records in losses.items():
            ax.plot([int(r['step']) for r in records],[float(r['loss']) for r in records],label=str(seed),alpha=0.6,linewidth=0.7)
        ax.set(xlabel='Distillation step',ylabel='LGM loss')
        ax.legend(title='Distillation seed')
        fig.tight_layout();fig.savefig(root/'loss_comparison.png');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,metric in zip(axes,['top1','top5']):
            seeds=plan['distillation_seeds']
            means=[100*statistics.mean(groups[metric][s]) for s in seeds]
            deviations=[100*statistics.pstdev(groups[metric][s]) for s in seeds]
            ax.errorbar(range(3),means,yerr=deviations,fmt='o',capsize=5,label='mean +/- evaluation std (ddof=0)')
            for i,seed in enumerate(seeds): ax.scatter([i]*3,[100*v for v in groups[metric][seed]],s=12)
            ax.axhline(100*plan['baseline_evaluation']['summary'][metric+'_mean'],color='gray',linestyle='--',label='existing seed 3407')
            ax.set(xticks=range(3),xticklabels=seeds,xlabel='Distillation seed',ylabel=metric+' accuracy (%)')
            ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(root/'accuracy_comparison.png');plt.close(fig)
        lines += ['', '![Loss比較](loss_comparison.png)', '![精度比較](accuracy_comparison.png)']
    else:
        lines += ['', '比較グラフ: 未実施。新規3組が正常終了してから作成する。']
    (root/'comparison_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
