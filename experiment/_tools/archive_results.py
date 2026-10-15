"""Create a reviewable results archive without modifying original reports."""
import csv,gzip,hashlib,json,re,shutil,statistics
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'experiment'
MAPPING={
 'us07824-lgm-pixel-5000-lrfix-qNrXBJZ2':'flower102/pixel_seed3407',
 'us07824-lgm-generator-5000-vE8eQLnO':'flower102/generator_seed3407/distillation',
 'us07824-lgm-generator-eval3-3QoaNElk':'flower102/generator_seed3407/evaluation',
 'us07824-lgm-generator-seeds-wrpWhzXU':'flower102/generator_additional_seeds',
 'us07824-lgm-pixel-5000-0RwHhugA':'flower102/history/pixel_lr_error_9006196',
 'us07824-lgm-generator-smoke-j4SzPSSj':'flower102/history/generator_cpu_test_preparation',
 'stanford-dogs-comparison-plan-q361seki':'stanford_dogs/comparison',
}
SKIP={'src','cache','wandb','__pycache__','.git','checkpoints','evaluation-checkpoints','data','input'}
ALLOWED={'.md','.json','.csv','.png','.py','.sh','.txt'}
entries=[]
for name,destination in MAPPING.items():
 source=REPO/'reports'/name;target=OUT/destination
 assert not target.exists(),target
 for p in sorted(source.rglob('*')):
  rel=p.relative_to(source)
  if p.is_symlink() or not p.is_file() or any(part in SKIP for part in rel.parts):continue
  if p.suffix not in ALLOWED and p.suffix!='.log':continue
  if p.name in ['local-monitor.log','execution-events.jsonl']:continue
  dest=target/rel
  if p.suffix=='.log':
   dest=dest.with_suffix('.log.gz');dest.parent.mkdir(parents=True,exist_ok=True)
   with p.open('rb') as inp,dest.open('xb') as raw:
    with gzip.GzipFile(filename=p.name,fileobj=raw,mode='wb',mtime=0) as gz:shutil.copyfileobj(inp,gz)
  else:
   dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
  entries.append({'path':str(dest.relative_to(OUT)),'source':str(p.relative_to(REPO)),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size})
 (target/'ARCHIVE.md').write_text(f'''# アーカイブの出所

元のWSL保存先: `reports/{name}/`。元ファイルは変更していない。
TSUBAMEの保存先はplan/status内の絶対パスを参照。

Markdown・集計JSON・設定・CSV・PNG・ジョブ/検証スクリプトを保存。
ログは`.log.gz`へ可逆圧縮した。元のレポートに記載された`.log`は同じ位置の`.log.gz`を展開して参照できる。
重み・data.pth・checkpoint・元データセット・キャッシュはGitに含めない。
元の記録中の「承認待ち」「未実施」は記録作成時点の状態であり、最新状態はデータセット別READMEとexecution-statusを参照。
''')
(OUT/'archive-manifest.json').write_text(json.dumps({'files':entries,'original_reports_unchanged':True,'binary_checkpoints_in_git':False},indent=2))
print(json.dumps({'files':len(entries),'bytes':sum(e['bytes'] for e in entries),'datasets':['flower102','stanford_dogs']}))
