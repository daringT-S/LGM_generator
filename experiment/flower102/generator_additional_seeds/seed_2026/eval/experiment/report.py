"""Per-job report without modifying existing experiments."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
plan=json.loads((root/'experiment/plan.json').read_text())
phase=plan['phase']
statefile=root/('status.json' if phase=='distill' else 'eval_status.json')
state=json.loads(statefile.read_text()) if statefile.exists() else {}
lines=[f"# Generator seed {plan['seed']}: {phase}",'',
       '状態: '+state.get('status','未実施・承認待ち'),'',
       '```json',json.dumps(state,indent=2,ensure_ascii=False),'```','',
       '実験設定: `experiment/plan.json`。既存の実験を上書きしない。']
if phase=='distill' and (root/'loss_curve.png').exists():
    lines+=['','![Loss](loss_curve.png)','![合成画像一覧](synthetic_grid.png)']
(root/'experiment_report.md').write_text('\n'.join(lines)+'\n')
