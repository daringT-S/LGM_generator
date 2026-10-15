"""Report measured outputs only; no job submission or training."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text()) if p.exists() else {}
def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 execution=load(ROOT/'execution-status.json')
 lines=['# Stanford Dogs: Pixel vs Generator','',f"状態: {execution.get('status','未実施')}",'',
 'Stanford Dogs公式train12,000/test8,580、120クラス。DINOv2 ViT-B、IPC1、Adam lr0.002、蒸留seed3407、iterations5000（5001更新）、workers2、augs_per_batch1。',
 'H100×1、gpu_1=1、優先度0、tga-lab_otm、W&B offline。syn/real256、crop224、standard augmentation、noise初期化、色変換True。',
 '全画像を使用しbbox切り出しなし。ImageNet mean/std。評価は公式distillation.eval、1000epochs×3、seed3407/3408/3409、各回で新規linear head・Adam・scheduler。',
 '評価用Adam lrは公式式を維持。各方式1蒸留の比較であり、評価3回は独立な蒸留3回ではない。','',
 '## ジョブ・時間・メモリ','',
 '| 方式 | 処理 | Job ID | exit / failed | 処理秒 | 会計秒 | GPU peak GiB | 独立検証 |',
 '|---|---|---|---|---:|---:|---:|---|']
 results={};losses={};total=0
 for mode in ['pixel','generator']:
  d=ROOT/mode/'distill';e=ROOT/mode/'eval'
  for phase,run,statefile,vfile in [('distill',d,'status.json','distillation-validation.json'),('eval',e,'eval_status.json','evaluation-validation.json')]:
   s=load(run/statefile);a=load(run/'job-accounting.json');v=load(run/vfile);sub=load(run/'submission.json')
   sec=s.get('experiment_seconds' if phase=='distill' else 'total_seconds')
   mem=s.get('peak_allocated_bytes',s.get('peak_gpu_memory_bytes'))
   wall=a.get('wallclock',a.get('ru_wallclock'))
   if wall:total+=int(.2*(.7*max(float(wall),300)+.1*(7200 if phase=='distill' else 3600))/3600*10000)/10000
   lines.append(f"| {mode} | {phase} | {sub.get('job_id','未実施')} | {a.get('exit_status','未実施')} / {a.get('failed','未実施')} | {round(sec,3) if sec is not None else '未実施'} | {wall or '未実施'} | {round(mem/1024**3,3) if mem is not None else '未実施'} | {v.get('status','未実施')} |")
  ds=load(d/'status.json');es=load(e/'eval_status.json');ev=load(e/'evaluation-validation.json')
  if es.get('status')=='success' and ev.get('status')=='passed':results[mode]=es
  if (d/'loss.csv').exists():
   with (d/'loss.csv').open() as f:losses[mode]=list(csv.DictReader(f))
  if (d/'loss.csv').exists():
   fig,ax=plt.subplots();ax.plot([int(x['step']) for x in losses[mode]],[float(x['loss']) for x in losses[mode]],lw=.7)
   ax.set(xlabel='Step',ylabel='LGM loss',title=mode);fig.tight_layout();fig.savefig(d/'loss_curve.png');plt.close(fig)
 lines+=['',f'取得済みqacctからのポイント推定合計: {total:.4f}。全4本の最大0.96ポイント（ストレージを除く）。','',
 '## 分類精度','',
 '| 方式 | 評価seed | Top-1 | Top-5 |','|---|---:|---:|---:|']
 for mode in ['pixel','generator']:
  s=results.get(mode)
  if not s:lines.append(f'| {mode} | — | 未実施 | 未実施 |');continue
  for row in s['completed_evaluations']:lines.append(f"| {mode} | {row['seed']} | {100*row['top1']:.4f}% | {100*row['top5']:.4f}% |")
  for label,suffix in [('平均','mean'),('標準偏差 ddof=0','std')]:lines.append(f"| {mode} | {label} | {100*s['summary']['top1_'+suffix]:.4f}{'%' if suffix=='mean' else ' pt'} | {100*s['summary']['top5_'+suffix]:.4f}{'%' if suffix=='mean' else ' pt'} |")
 if len(results)==2:
  delta={metric:100*(results['generator']['summary'][metric+'_mean']-results['pixel']['summary'][metric+'_mean']) for metric in ['top1','top5']}
  lines+=['',f"Generator−Pixel: Top-1 {delta['top1']:+.4f}ポイント、Top-5 {delta['top5']:+.4f}ポイント。"]
  fig,axes=plt.subplots(1,2,figsize=(9,4))
  for ax,metric in zip(axes,['top1','top5']):
   means=[100*results[m]['summary'][metric+'_mean'] for m in ['pixel','generator']];std=[100*results[m]['summary'][metric+'_std'] for m in ['pixel','generator']]
   ax.errorbar([0,1],means,yerr=std,fmt='o',capsize=5)
   for i,mode in enumerate(['pixel','generator']):ax.scatter([i]*3,[100*r[metric] for r in results[mode]['completed_evaluations']],s=12)
   ax.set(xticks=[0,1],xticklabels=['Pixel','Generator'],ylabel=metric+' accuracy (%)')
  fig.tight_layout();fig.savefig(ROOT/'accuracy_comparison.png');plt.close(fig)
  lines+=['','![精度比較](accuracy_comparison.png)']
  with (ROOT/'accuracies.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=['mode','evaluation','seed','top1','top5']);writer.writeheader()
   for mode,s in results.items():
    for row in s['completed_evaluations']:writer.writerow(dict(mode=mode,**row))
 if losses:
  fig,ax=plt.subplots()
  for mode,rows in losses.items():ax.plot([int(r['step']) for r in rows],[float(r['loss']) for r in rows],label=mode,lw=.7,alpha=.7)
  ax.set(xlabel='Step',ylabel='LGM loss');ax.legend();fig.tight_layout();fig.savefig(ROOT/'loss_comparison.png');plt.close(fig)
  lines+=['','![Loss比較](loss_comparison.png)']
 lines+=['','## 保存結果','']
 for mode in ['pixel','generator']:
  s=load(ROOT/mode/'distill/status.json');v=load(ROOT/mode/'distill/distillation-validation.json')
  rows=losses.get(mode,[])
  lines+=[f"### {mode}",'',f"更新数: {s.get('completed_updates','未実施')}。初回Loss: {s.get('first_loss','未実施')}。最終Loss: {s.get('last_loss','未実施')}。",f"data.pth: `{s.get('data_path','未実施')}`。SHA256: `{s.get('data_sha256','未実施')}`。",f'画像: `{mode}/distill/images/`（120枚）。checkpoint: `{mode}/distill/checkpoints/`（11個）。CSV: `{mode}/distill/loss.csv`。']
  if mode=='generator':lines+=[f"Generator更新: {s.get('generator_parameters_updated','未実施')}、固定z不変: {s.get('fixed_noise_unchanged','未実施')}。重み・z・ラベル・Adam: `generator/distill/generator_final.pth`。",f"CPU再生成の最大画像差: {v.get('cpu_regeneration_max_abs_difference','未実施')}。評価は保存済みGPU data.pthを使用し、CPU再生成画像は使用しない。"]
  if (ROOT/mode/'distill/synthetic_grid.png').exists():lines+=['',f'![{mode}合成画像一覧]({mode}/distill/synthetic_grid.png)']
 if all((ROOT/m/'distill/images/class_000.png').exists() for m in ['pixel','generator']):
  fig,axes=plt.subplots(2,4,figsize=(9,5))
  for i,mode in enumerate(['pixel','generator']):
   for j,label in enumerate([0,30,60,90]):
    with Image.open(ROOT/mode/f'distill/images/class_{label:03d}.png') as im:axes[i,j].imshow(im)
    axes[i,j].set(xticks=[],yticks=[],title=f'{mode}: class {label}')
  fig.tight_layout();fig.savefig(ROOT/'synthetic_samples_comparison.png',dpi=150);plt.close(fig)
  lines+=['','![同じクラスの画像比較](synthetic_samples_comparison.png)']
 lines+=['','## エラー・警告','']
 if execution.get('error'):lines+=['```text',execution['error'],'```']
 import re
 for log in sorted(ROOT.glob('*/*/logs/*.log')):
  found=[]
  for line in log.read_text(errors='replace').splitlines():
   if re.search(r'Warning:|WARNING|Traceback|Error:',line):
    line=line.strip()[-1200:]
    if line not in found:found.append(line)
  if found:lines += [f'`{log.relative_to(ROOT)}`:','```text','\n'.join(found[:20]),'```']
 lines+=['','実行状態・エラー: `execution-status.json`。データ監査: `data-validation.json`。公式取得SHA256: `download-manifest.json`。',
 '既存Flowers出力は保存先に指定せず、保護対象のSHA256を実行・独立検証で照合する。取得できていない精度や実測値は未実施と記載する。']
 (ROOT/'experiment_report.md').write_text('\n'.join(lines)+'\n')
 (ROOT/'comparison-results.json').write_text(json.dumps({'status':execution.get('status'),'results':{m:s['summary'] for m,s in results.items()},'estimated_points':round(total,4),'ddof':0},indent=2))
 print(json.dumps({'status':execution.get('status'),'validated_evaluations':len(results)}))
if __name__=='__main__':main()
