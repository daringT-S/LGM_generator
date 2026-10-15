"""Build dataset indexes from recorded evaluations, preserving missing values."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text())
rows=[]
def add(dataset,method,seed,status,path):
 s=load(path)
 record={'dataset':dataset,'method':method,'distillation_seed':seed,'evaluation_status':status,'report':path,'ddof':0}
 for metric in ['top1','top5']:
  values=[r[metric] for r in s['completed_evaluations']]
  record[metric+'_values']=values
  record[metric+'_mean']=statistics.mean(values) if all(v is not None for v in values) else None
  record[metric+'_std']=statistics.pstdev(values) if all(v is not None for v in values) else None
 rows.append(record)
add('flowers102','pixel',3407,'partial (final save failed)','flower102/pixel_seed3407/evaluation-recovered.json')
add('flowers102','generator_pixel',3407,'success','flower102/generator_seed3407/evaluation/eval_status.json')
for seed in [42,123,2026]:add('flowers102','generator_pixel',seed,'success',f'flower102/generator_additional_seeds/seed_{seed}/eval/eval_status.json')
for mode in ['pixel','generator']:add('stanforddogs','generator_pixel' if mode=='generator' else mode,3407,'success',f'stanford_dogs/comparison/{mode}/eval/eval_status.json')
(ROOT/'results.json').write_text(json.dumps({'accuracy_unit':'fraction [0,1]','std_ddof':0,'runs':rows},indent=2))
with (ROOT/'results.csv').open('w',newline='') as f:
 fields=['dataset','method','distillation_seed','evaluation_status','top1_mean_percent','top1_std_percentage_points','top5_mean_percent','top5_std_percentage_points']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for row in rows:
  result={k:row[k] for k in fields[:4]}
  for metric in ['top1','top5']:
   result[metric+'_mean_percent']='取得失敗' if row[metric+'_mean'] is None else 100*row[metric+'_mean']
   result[metric+'_std_percentage_points']='取得失敗' if row[metric+'_std'] is None else 100*row[metric+'_std']
  w.writerow(result)
def fmt(value):return '取得失敗' if value is None else f'{100*value:.4f}'
for dataset,directory in [('flowers102','flower102'),('stanforddogs','stanford_dogs')]:
 subset=[r for r in rows if r['dataset']==dataset]
 lines=[f'# {dataset}: Pixel / Generatorの実験結果','',
 '共通条件: DINOv2 ViT-B、IPC1、Adam lr0.002、iterations5000（5001更新）、workers2、augs_per_batch1、TSUBAME H100×1、W&B offline。',
 '公式distillation.eval、評価seed3407/3408/3409、eval_epochs1000、各回で分類器を初期化。標準偏差はddof=0。', '',
 '| 方式 | 蒸留seed | Top-1平均 (%) | 評価内std (pt) | Top-5平均 (%) | 評価内std (pt) |',
 '|---|---:|---:|---:|---:|---:|']
 for r in subset:lines.append(f"| {r['method']} | {r['distillation_seed']} | {fmt(r['top1_mean'])} | {fmt(r['top1_std'])} | {fmt(r['top5_mean'])} | {fmt(r['top5_std'])} |")
 pixel=subset[0];gen=subset[1]
 lines+=['',f"蒸留seed3407同士のGenerator−PixelのTop-1差: **{100*(gen['top1_mean']-pixel['top1_mean']):+.4f}ポイント**。",'']
 if dataset=='flowers102':
  new=[r for r in subset if r['distillation_seed'] in [42,123,2026]]
  means=[r['top1_mean'] for r in new]
  lines += [f'追加蒸留seed42/123/2026のTop-1評価平均を3点として集計: 平均 **{100*statistics.mean(means):.4f}%**、蒸留seed間標準偏差 **{100*statistics.pstdev(means):.4f}ポイント**。',
   '表の評価内stdとは別の指標。9回の評価を独立な蒸留9回として扱わない。','',
   'Pixel評価9006786は3回の精度計算後、保存ラッパーのtorch.saveキーワード引数不一致で失敗。Top-1全3値はログから復元済み。Top-5の3回目と3回分集計は取得失敗。',
   '初期のlr型エラージョブ9006196はhistoryに保存。Generatorの4step GPUテストは省略されたため、CPUテスト準備記録をGPU実測として扱わない。','',
   '## 詳細','',
   '- [Pixel蒸留・評価記録](pixel_seed3407/experiment_report.md)',
   '- [Pixel復元評価](pixel_seed3407/evaluation-recovered.json)',
   '- [Generator seed3407蒸留](generator_seed3407/distillation/experiment_report.md)',
   '- [Generator seed3407評価](generator_seed3407/evaluation/experiment_report.md)',
   '- [追加3蒸留シードの比較表・Loss・画像](generator_additional_seeds/experiment_report.md)',
   '- [失敗したlr実験](history/pixel_lr_error_9006196/experiment_report.md)',
   '- [CPUテスト・小規模GPUテストの準備記録](history/generator_cpu_test_preparation/experiment_report.md)','',
   '![追加蒸留シードの精度比較](generator_additional_seeds/accuracy_comparison.png)',
   '![追加蒸留シードのLoss比較](generator_additional_seeds/loss_comparison.png)']
 else:
  state=load('stanford_dogs/comparison/execution-status.json');assert state['status']=='complete'
  lines += ['全4ジョブが正常終了し、独立CPU検証passed。',
   'Pixel蒸留9013760 → Pixel評価9013854 → Generator蒸留9013956 → Generator評価9014227。','',
   '公式train12,000枚/test8,580枚、120クラス、IPC1で合成画像120枚。bbox切り出しなし。',
   '各方式の蒸留はseed3407で1回。Dogsでは蒸留seed間の再現性検証は未実施。','',
   '## 詳細','',
   '- [実験レポート（時間・GPUメモリ・全精度・Loss・生成画像・警告）](comparison/experiment_report.md)',
   '- [実行完了・ジョブID](comparison/execution-status.json)',
   '- [データ監査](comparison/data-validation.json)',
   '- [全6回の精度CSV](comparison/accuracies.csv)','',
   '![精度比較](comparison/accuracy_comparison.png)',
   '![Loss比較](comparison/loss_comparison.png)',
   '![同じクラスの生成画像比較](comparison/synthetic_samples_comparison.png)']
 (ROOT/directory/'README.md').write_text('\n'.join(lines)+'\n')
(ROOT/'README.md').write_text('''# LGM実験結果

データセット別にPixelDataset方式とgenerator_pixel方式の結果を保存する。元のTSUBAME/WSL実験結果は変更していない。
`flower102/`は依頼に合わせたフォルダ名で、コード上のデータセット名は`flowers102`。
`stanford_dogs/`のCLI名は`stanforddogs`。

- [Flowers-102: 比較・追加3蒸留シードの再現性](flower102/README.md)
- [Stanford Dogs: Pixel / Generator比較](stanford_dogs/README.md)
- [全結果のCSV](results.csv)（精度は%、標準偏差はpercentage points）
- [全結果のJSON](results.json)（精度・標準偏差は0〜1のfraction）
- [実行ソースのスナップショット](source_snapshot/README.md)
- [アーカイブのSHA256・出所](archive-manifest.json)

各方式・各蒸留シードの評価は公式distillation.evalで独立初期化3回、seed3407/3408/3409。標準偏差はddof=0。
蒸留のiterations5000は現行コードではstep0〜5000の5001更新。
失敗・未実施・取得失敗は実測値と区別し、欠測値を推測で埋めていない。

設定、ジョブスクリプト、Loss CSV、グラフ、全クラスPNG、検証JSON、会計記録と圧縮ログを同梱。
元の保存先・data.pthやモデルのSHA256は各plan/status/manifestに記録。
モデル重み、Generatorパッケージ、data.pth、checkpoint、元データセットとキャッシュはTSUBAMEおよび元のreportsに保持し、Gitには含めない。
元のレポートに記載された.logは同じ場所の.log.gzを展開して参照できる。

過去の計画文書は作成当時の状態を保持している。最新の完了状態と集計はこの索引・データセット別README・execution-statusを参照する。
通常のsrc/や既存の作業中ファイルはこの結果整理では変更しない。
''')
(ROOT/'.gitignore').write_text('*.pth\n*.pt\n*.ckpt\n*.tar\n*.zip\n*.npy\n*.npz\n__pycache__/\ncache/\n')
print('Summaries generated from',len(rows),'measured distillations')
