"""Report observed study outputs only; never submit jobs or modify training inputs."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def number(value, digits=3):
    return f'{value:.{digits}f}' if value is not None else '未実施'


def main():
    root = Path(__file__).resolve().parent
    plan = load(root / 'study-plan.json')
    execution = load(root / 'execution-status.json')
    report = root / 'experiment_report.md'
    # Retain the reviewed preparation text before replacing the live report.
    preparation = root / 'preparation_report.md'
    if not preparation.exists():
        with preparation.open('x') as handle:
            handle.write(report.read_text())
    lines = ['# Generator方式：蒸留シードの再現性実験', '',
             '更新（UTC）: ' + datetime.datetime.now(datetime.timezone.utc).isoformat(), '',
             '状態: **' + execution.get('status', '未実施') + '**。ユーザー承認に基づき最大6本を直列実行。失敗時の再投入は行わない。', '',
             'flowers102 / dinov2_vitb / generator_pixel / Adam / lr=0.002 / IPC=1 / iterations=5000（5,001更新）。',
             '蒸留seed=42・123・2026。各結果を公式`distillation.eval`で3回評価（seed=3407・3408・3409、eval_epochs=1000）。',
             'H100×1、gpu_1=1、優先度0、tga-lab_otm、W&B offline。蒸留2時間・評価1時間、全6本の最大1.44ポイント。',
             '損失・Adam・Generator構造・公式評価は既存seed3407と同一。学習対象はGeneratorのみで固定ノイズを更新しない。', '',
             '## 実行記録', '',
             '| 蒸留seed | 処理 | Job ID | 終了コード / failed | 処理秒 | 会計秒 | GPU peak allocated GiB | 検証 |',
             '|---:|---|---|---|---:|---:|---:|---|']
    total_points = 0
    for seed in plan['distillation_seeds']:
        for phase in ['distill', 'eval']:
            run = root / f'seed_{seed}' / phase
            submission = load(run / 'submission.json')
            accounting = load(run / 'job-accounting.json')
            state = load(run / ('status.json' if phase == 'distill' else 'eval_status.json'))
            validation = load(run / ('distillation-validation.json' if phase == 'distill' else 'evaluation-validation.json'))
            seconds = state.get('experiment_seconds' if phase == 'distill' else 'total_seconds')
            peak = state.get('peak_allocated_bytes' if phase == 'distill' else 'peak_gpu_memory_bytes')
            wall = accounting.get('wallclock', accounting.get('ru_wallclock'))
            if wall is not None:
                # Published priority-0 gpu_1 formula, truncated per job.
                points = int((0.2 * (0.7 * max(float(wall), 300) + 0.1 * (7200 if phase == 'distill' else 3600)) / 3600) * 10000) / 10000
                total_points += points
            lines.append(f"| {seed} | {phase} | {submission.get('job_id', '未実施')} | {accounting.get('exit_status', '未実施')} / {accounting.get('failed', '未実施')} | {number(seconds)} | {wall if wall is not None else '未実施'} | {number(peak / 1024**3 if peak is not None else None)} | {validation.get('status', '未実施')} |")
    lines += ['', f'取得済みqacctからの計算ポイント推定合計: **{total_points:.4f}**（会計確定値ではなく公式式による推定、ストレージを除く）。',
              'GPUメモリはPyTorch max_memory_allocated。処理秒とスケジューラ会計秒は測定範囲が異なる。', '',
              '## 精度とばらつき', '']
    comparison = root / 'comparison_report.md'
    lines += [comparison.read_text() if comparison.exists() else '未実施', '',
              '## 蒸留Loss・保存内容', '',
              '| 蒸留seed | 更新数 | 初回Loss | 最終Loss | 最小Loss | PNG数 | Generator更新 / ノイズ固定 |',
              '|---:|---:|---:|---:|---:|---:|---|']
    for seed in plan['distillation_seeds']:
        run = root / f'seed_{seed}' / 'distill'
        state = load(run / 'status.json')
        rows = []
        if (run / 'loss.csv').exists():
            with (run / 'loss.csv').open() as handle:
                rows = list(csv.DictReader(handle))
        lines += [f"| {seed} | {state.get('completed_updates', '未実施')} | {number(state.get('first_loss'), 6)} | {number(state.get('last_loss'), 6)} | {number(min(float(r['loss']) for r in rows) if rows else None, 6)} | {state.get('png_count', '未実施')} | {state.get('generator_parameters_updated', '未実施')} / {state.get('fixed_noise_unchanged', '未実施')} |"]
    for seed in plan['distillation_seeds']:
        relative = f'seed_{seed}/distill'
        state = load(root / relative / 'status.json')
        lines += ['', f'### 蒸留seed {seed}', '',
                  f"最終data.pth: `{state.get('data_path', '未実施')}`。SHA256: `{state.get('data_sha256', '未実施')}`。", '',
                  f'重み・固定ノイズ・ラベル・Adam: `{relative}/generator_final.pth`。最終checkpoint: `{relative}/checkpoints/step_05000_final.pth`。',
                  f'初期化記録: `{relative}/initialization.json`。途中checkpoint: `{relative}/checkpoints/step_*.pth`。',
                  f'全Loss: `{relative}/loss.csv`。PNG一覧: `{relative}/images/class_000.png`〜`class_101.png`。',
                  f'評価: `seed_{seed}/eval/eval_status.json`・`evaluation-validation.json`・`evaluation-checkpoints/`。']
        if (root / relative / 'synthetic_grid.png').exists():
            lines += ['', f'![seed {seed} 合成画像一覧]({relative}/synthetic_grid.png)']
    if (root / 'synthetic_samples_comparison.png').exists():
        lines += ['', '同じクラス（0・25・50・75）を蒸留seed間で比較:', '',
                  '![合成画像サンプル比較](synthetic_samples_comparison.png)', '',
                  '9回の実測精度をCSVでも保存: [accuracies.csv](accuracies.csv)。']
    lines += ['', '## 検証・エラーと警告', '',
              '蒸留正常終了・5001更新・Adam step・Generator更新・固定ノイズ不変・data形状[102,3,256,256]・float32値域[0,1]・ラベルint64 0〜101・PNG102枚・checkpoint11個を独立CPU検証する。',
              '評価は保存済みdata.pthを読み取り専用コピーして使用。CPUでGeneratorを再生成した画像は評価に使用しない。',
              '各評価で新しい分類器と空のAdam状態を検証し、評価前後の入力ハッシュを照合する。', '']
    if execution.get('error'):
        lines += ['実行停止理由:', '```text', execution['error'], '```']
    warnings = []
    for path in sorted(root.glob('seed_*/*/logs/*.log')):
        matches = []
        for line in path.read_text(errors='replace').splitlines():
            if re.search(r'Warning:|WARNING|Traceback|Error:', line):
                clean = line.strip()[-1500:]
                if clean not in matches:
                    matches.append(clean)
        if matches:
            warnings += [f'`{path.relative_to(root)}`:', '```text', '\n'.join(matches[:25]), '```']
    lines += warnings or ['取得済み.log内のWarning/Traceback/Errorに該当する行なし（未実施ジョブを除く）。']
    preserved = []
    for filename, expected in plan['preserved_artifacts'].items():
        h = hashlib.sha256()
        with Path(filename).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != expected:
            preserved.append(filename)
    assert not preserved, preserved
    lines += ['', f"以前のPixel/Generator実験の保護対象{len(plan['preserved_artifacts'])}ファイルはすべてSHA256一致。", '',
              '既存Generator seed3407: Top-1平均99.5609045%、Top-5平均99.8644749%。Pixel Top-1平均97.1431673%。',
              '評価内stdは同じ合成画像に対する分類器初期化のばらつき、蒸留seed間stdは合成画像を学習し直すことによる評価平均のばらつき。いずれもddof=0。',
              '3シードの結果はこの条件下の再現性を調べるものであり、全初期化や他モデルでの一致を保証しない。', '',
              '承認済み計画とCPU検証: [preparation_report.md](preparation_report.md)、`study-plan.json`、`seed-validation.json`、`execution-plan-validation.json`。',
              '実行記録: `execution-status.json`。未完了の実測値は未実施とし推定で補完しない。']
    report.write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': execution.get('status'), 'protected_artifacts_unchanged': True, 'estimated_points_from_accounting': total_points}))


if __name__ == '__main__':
    main()
