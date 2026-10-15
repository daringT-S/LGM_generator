# Generator合成画像の公式LGM評価（3回）

状態: success

## 評価条件

| 項目 | 設定 |
|---|---|
| Dataset / backbone | flowers102 / dinov2_vitb |
| 画像 | 蒸留9010498の最終data.pth、102枚、IPC1 |
| 評価回数 / epochs / seeds | 3 / 1000 / 3407・3408・3409 |
| 分類器 | 毎回新規初期化する768→102の線形層 |
| 分類器最適化 | Adam、lr=0.001×100/256=0.000390625、weight_decay=0、CosineAnnealingLR |
| 合成画像入力 | float32 [102,3,256,256]、[0,1]、ラベルint64 [0..101] |
| 合成画像augmentation | 元のAugBasic（水平反転・RandomResizedCrop224）|
| 正規化 | mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]、augmentation後に1回 |
| 分割 | 元のFlowers102 train1020 / test6149、画像IDの重複なし。valは不使用 |
| テスト前処理 | Resize256 → CenterCrop224 → ToTensor → 同じ正規化 |
| batch / workers | 100（合成画像は100+2の2バッチ）/ 2 |
| eval_it / patience / checkpoint_it | -1 / 5 / 100、各回epoch999でテスト |
| 精度 | Top-1/Top-5 micro accuracy、平均と標準偏差ddof=0 |
| GPU / group / priority | H100×1、gpu_1=1 / tga-lab_otm / 0 |
| W&B / 上限 | offline / 01:00:00 |

## 投入前検証

- 検証状態: passed
- 公式eval.py・EvalCfg・models・augmentation・データ処理はPixel実験のスナップショットとハッシュ一致。
- 評価ハイパーパラメータはPixel方式のplan.jsonと完全一致（合成画像と出力先を除く）。
- 実際の設定パーサー、102画像のshape/ラベル/有限性/値域、実データの分割と正規化をCPU検証。
- 元のEvaluatorのreset/repeat/checkpoint保存をCPUで検証。ダミーepoch/指標は制御フローテスト専用で、実験精度ではない。
- 保存ラッパーのf引数対応を修正。位置引数のcheckpoint保存と、公式ASTのtorch.save(obj=..., f=...)をCPUで実行してロード確認。

## CPU再生成差の扱い

CPUでGeneratorを再実行した画像との差0.007019848は、今回の評価入力には使わない。
公式評価は保存済み画像テンソルを直接ロードする。Generatorを生成・実行・最適化しない。
元の最終data.pthをバイト同一のまま専用入力領域へコピーし、変更前後のSHA256を検査する。
入力SHA256: `d9518a060a50fde628c1e7d89ea58fa4b021a28eb8f837ebf159a67bcda61312`。元のGenerator重み・ノイズ・チェックポイントもハッシュ保護する。

## 結果

| 評価 | Top-1 Accuracy | Top-5 Accuracy |
|---|---:|---:|
| 1回目 | 99.5772% | 99.9024% |
| 2回目 | 99.5446% | 99.8211% |
| 3回目 | 99.5609% | 99.8699% |
| 平均 | 99.5609% | 99.8645% |
| 標準偏差 | 0.0133% | 0.0334% |

Pixel Top-1基準: 97.1432%。差はGenerator平均−97.1432のパーセントポイントで報告する。
差: +2.4177ポイント
評価時間: 400.7025024869945 秒。ピークGPU割当: 2197092352 bytes。

## 保存先・実行コマンド

専用ルート: `/gs/bs/tga-lab_otm/us07824-lgm-generator-eval3-3QoaNElk`
元の最終画像: `/gs/bs/tga-lab_otm/us07824-lgm-generator-5000-vE8eQLnO/distillation/flowers102/dinov2_vitb/flowers102_generator_pixel_adam_it5000_ipc1/data.pth`
評価用の同一コピー: `/gs/bs/tga-lab_otm/us07824-lgm-generator-eval3-3QoaNElk/input/flowers102/dinov2_vitb/generator_final_9010498/data.pth`
公式集計: `/gs/bs/tga-lab_otm/us07824-lgm-generator-eval3-3QoaNElk/input/flowers102/dinov2_vitb/generator_final_9010498/eval/dinov2_vitb.pth`
個別精度・時間・GPUメモリ: `eval_status.json`。初期分類器を含むcheckpoint: `evaluation-checkpoints/`。ログ: `logs/eval.log`。

```bash
cd /gs/bs/tga-lab_otm/us07824-lgm-generator-eval3-3QoaNElk
qsub -g tga-lab_otm eval_job.sh
```

ジョブ内でrun_eval.pyから、次の公式エントリポイントと同じ引数を実行する。

```bash
python -m distillation.eval --dataset=flowers102 --model=dinov2_vitb --eval_model=dinov2_vitb --num_eval=3 --eval_epochs=1000 --eval_it=-1 --patience=5 --checkpoint_it=100 --num_workers=2 --real_batch_size=100 --real_res=256 --crop_res=224 --train_crop_mode=random --data_root=/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/data --job_tag=/gs/bs/tga-lab_otm/us07824-lgm-generator-eval3-3QoaNElk/input --skip_if_exists=True
```

## 時間・ポイント見積もり

前回Pixel評価は3回計算413.10秒・GPUピーク2.046GiB（最後の保存のみ失敗）。今回は10〜20分、GPU割当2〜4GiBを見込む。
上限1時間・最大0.16ポイント。10〜20分で終了なら約0.0433〜0.0666ポイント。
`0.2 × (0.7 × max(実行秒,300) + 0.1 × 3600) / 3600`（小数第5位以下切り捨て）。
[公式課金規則](https://www.somuka.titech.ac.jp/reiki_int/reiki_honbun/x385RG00001693.html)。ストレージ課金を含まない。

## エラー・警告

実評価は正常終了。実行エラーなし。

```text
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/env/lib/python3.11/site-packages/outdated/__init__.py:36: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/env/lib/python3.11/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:45: UserWarning: xFormers is disabled (SwiGLU)
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:29: UserWarning: xFormers is disabled (Attention)
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:35: UserWarning: xFormers is disabled (Block)
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
/gs/bs/tga-lab_otm/us07824-lgm-generator-eval3-3QoaNElk/src/distillation/eval.py:98: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
Epoch Progress:   0%|          | 0/2 [00:00<?, ?it/s][A/gs/bs/tga-lab_otm/us07824-lgm-generator-eval3-3QoaNElk/src/distillation/eval.py:148: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
```

承認済みの評価ジョブ1本を投入した。エラー時に自動再投入しない。既存のPixel結果とGenerator蒸留結果は変更しない。

## 評価終了時の検証・会計記録

- ジョブID: 9011526
- 終了コード / failed: 0 / 0
- ホスト / GPU: r18n9 / NVIDIA H100
- 開始〜終了（JST）: 10/14/2026 23:57:12.222 → 10/15/2026 00:04:07.576
- Python全体時間（初期化・保存を含む）: 400.7025024869945 秒
- 会計wallclock: 428.239 秒
- GPUピーク割当: 2.046 GiB
- GPUピーク予約: 2.547 GiB
- 会計時間からのポイント算定: 0.0366（請求確定値ではない）
- 保存済み結果の独立CPU検証: passed
- 初期checkpointから3つのシード、異なる分類器重み、空のAdam状態を検証した。
- 公式集計ファイルと3回の実測値からの平均・標準偏差（ddof=0）の一致を確認した。
- Generatorの重み・固定ノイズ・蒸留結果、Pixel実験のデータ・結果・ログのハッシュが不変であることを確認した。
- 詳細: `evaluation-validation.json`、`job-accounting.json`、`logs/qacct.txt`。

## 元の蒸留結果への参照

[Generator蒸留レポート](../us07824-lgm-generator-5000-vE8eQLnO/experiment_report.md)
[Generator Lossグラフ](../us07824-lgm-generator-5000-vE8eQLnO/loss_curve.png)
[Generator画像一覧](../us07824-lgm-generator-5000-vE8eQLnO/synthetic_grid.png)
[Pixel蒸留・評価レポート](../us07824-lgm-pixel-5000-lrfix-qNrXBJZ2/experiment_report.md)

既存レポートは上書きせず、この評価専用レポートに実測精度と比較を記録した。追加ジョブや自動再投入は行っていない。
