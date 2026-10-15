"""Post-run reporting only; leave the frozen execution scripts unchanged."""
import decimal
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    state = json.loads((ROOT/'eval_status.json').read_text())
    submitted = json.loads((ROOT/'submission.json').read_text())
    account = json.loads((ROOT/'job-accounting.json').read_text())
    validation = json.loads((ROOT/'evaluation-validation.json').read_text()) if (ROOT/'evaluation-validation.json').exists() else {}
    path = ROOT/'experiment_report.md'
    text = path.read_text()
    marker = '## 評価終了時の検証・会計記録'
    assert marker not in text, 'Already finalized; do not duplicate'
    warnings = []
    for line in (ROOT/'logs/eval.log').read_text(errors='replace').replace('\r','\n').splitlines():
        if ('Warning:' in line or 'wandb: WARNING' in line) and line not in warnings:
            warnings.append(line)
    old = '準備時の依存ライブラリ非推奨警告・CPUでのCUDA AMP無効化警告は検証ログに保持。実評価は未投入。'
    if state.get('status') == 'success' and account['exit_status']=='0':
        replacement = '実評価は正常終了。実行エラーなし。'
    else:
        replacement = '評価は正常完了していない。原因: '+state.get('error','会計記録を参照')
    if warnings:
        replacement += '\n\n```text\n'+'\n'.join(warnings[:20])+'\n```'
    text = text.replace(old, replacement)
    text = text.replace('承認後に評価ジョブ1本だけ投入する。', '承認済みの評価ジョブ1本を投入した。')
    D=decimal.Decimal
    points=D('0.2')*(D('0.7')*max(D(account['wallclock']),D(300))+D('0.1')*D(3600))/D(3600)
    points=points.quantize(D('0.0001'),rounding=decimal.ROUND_DOWN)
    lines=['',marker,'',
           f"- ジョブID: {submitted['job_id']}",
           f"- 終了コード / failed: {account['exit_status']} / {account['failed']}",
           f"- ホスト / GPU: {account['hostname']} / {state.get('gpu','取得失敗')}",
           f"- 開始〜終了（JST）: {account['start_time']} → {account['end_time']}",
           f"- Python全体時間（初期化・保存を含む）: {state.get('total_seconds','取得失敗')} 秒",
           f"- 会計wallclock: {account['wallclock']} 秒",
           f"- GPUピーク割当: {state.get('peak_gpu_memory_bytes',0)/1024**3:.3f} GiB",
           f"- GPUピーク予約: {state.get('peak_reserved_gpu_memory_bytes',0)/1024**3:.3f} GiB",
           f"- 会計時間からのポイント算定: {points}（請求確定値ではない）",
           f"- 保存済み結果の独立CPU検証: {validation.get('status','未実施')}",
           '- 初期checkpointから3つのシード、異なる分類器重み、空のAdam状態を検証した。' if validation.get('status')=='passed' else '- 初期状態の独立検証は未完了。',
           '- 公式集計ファイルと3回の実測値からの平均・標準偏差（ddof=0）の一致を確認した。' if validation.get('status')=='passed' else '- 集計の独立検証は未完了。',
           '- Generatorの重み・固定ノイズ・蒸留結果、Pixel実験のデータ・結果・ログのハッシュが不変であることを確認した。' if state.get('status')=='success' else '- 入力保護の最終検証は未完了。',
           '- 詳細: `evaluation-validation.json`、`job-accounting.json`、`logs/qacct.txt`。','',
           '## 元の蒸留結果への参照','',
           '[Generator蒸留レポート](../us07824-lgm-generator-5000-vE8eQLnO/experiment_report.md)',
           '[Generator Lossグラフ](../us07824-lgm-generator-5000-vE8eQLnO/loss_curve.png)',
           '[Generator画像一覧](../us07824-lgm-generator-5000-vE8eQLnO/synthetic_grid.png)',
           '[Pixel蒸留・評価レポート](../us07824-lgm-pixel-5000-lrfix-qNrXBJZ2/experiment_report.md)', '',
           '既存レポートは上書きせず、この評価専用レポートに実測精度と比較を記録した。追加ジョブや自動再投入は行っていない。']
    path.write_text(text+'\n'.join(lines)+'\n')
    print(path)


if __name__=='__main__':
    main()
