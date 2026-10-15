# LGM PixelDataset: Flowers-102 / DINOv2 ViT-B

蒸留: 取得失敗。評価: 未実施。
GPUジョブの投入はユーザー承認後。数値は生成済みの実測ファイルのみから集計する。

## 実験条件

| 項目 | 設定 |
|---|---|
| Dataset | flowers102: train 1,020枚、test 6,149枚 |
| Model / evaluation model | dinov2_vitb / dinov2_vitb（dinov2_vitb14、特徴768次元） |
| Method | 従来のPixelDataset、Generatorなし |
| IPC / synthetic images | 1 / 102 |
| Iterations | 5000（元実装のstep 0〜5000、5,001更新） |
| Optimizer / learning rate | Adam / 0.002（元の実装を維持） |
| Distillation seed | 3407 |
| workers / augs_per_batch | 2 / 1（前回の成功条件を維持） |
| synthetic / real / crop resolution | 256 / 256 / 224 |
| Init / augmentation / color | noise / standard / decorrelate_color=True |
| Hardware | TSUBAME 4.0 H100 × 1、gpu_1=1、CPU8コア、RAM96GB |
| Group / priority / W&B | tga-lab_otm / 0 / offline |
| Time limit | 蒸留02:00:00、評価01:00:00 |
| Source commit | 33d004b8c16d9ab8e39aec780f9edad767f6f033 |
| Source patches | distill_optのCLI宣言、W&B name指定とフォールバックのみ |
| Numeric routines | 元の損失関数・Adam・PixelDataset・公式Evaluatorを維持 |
| Attention backend | 前回同様xFormers無効、PyTorch math attention（二階微分対応） |

## 蒸留学習

- ジョブID: 9006196
- 完了更新数: 0
- 蒸留処理時間（初期化・保存を含む）: 未実施
- Python全体時間: 138.62 秒
- GPUピークメモリ: 15.349 GiB（PyTorch割当）
- Adam更新数の検証: 未実施

- ジョブ終了コード: 1
- 会計記録wallclock: 174.612 秒
- ジョブ開始〜終了: 163.974 秒

Lossの推移とグラフ: 取得失敗。step 0のAdam更新中に停止し、Loss記録は0件。

## 合成画像

PNG一覧・サンプル・最終data.pth: 未実施。前回の4ステップ画像は本実験結果に流用しない。

## 公式評価方法と分類精度

`runpy.run_module("distillation.eval", run_name="__main__")` により公式エントリポイントを実行し、`--num_eval=3` を指定する。
同一data.pthのハッシュを評価前後で確認する。Evaluator.reset()が各回の分類器・Adam・CosineAnnealingLRを新規作成する。
乱数シードはtorch/random/numpyについて3407、3408、3409。最終回終了後の追加resetは学習を伴わず、4回目の評価は行わない。
各回1,000 epochs、eval_it=-1（epoch999で1回テスト）、patience=5、checkpoint_it=100。最良値の選択方法も元実装のまま。
分類器学習バッチサイズは100（102画像のため各epochに100枚と2枚の2バッチ）。学習率は0.001×100/256=0.000390625、weight_decay=0。
固定バックボーンに対し、AugBasic（水平反転・ランダムクロップ）を用いて線形分類器を学習する。
テストはFlowers-102 test 6,149枚、Resize256・CenterCrop224・正規化。Top-1/Top-5はmicro accuracy。
平均と標準偏差は公式のnumpy.mean / numpy.stdに合わせ、ddof=0（母標準偏差）。以下は百分率。

| 評価 | Top-1 Accuracy | Top-5 Accuracy |
|---|---:|---:|
| 1回目 | 未実施 | 未実施 |
| 2回目 | 未実施 | 未実施 |
| 3回目 | 未実施 | 未実施 |
| 平均 | 未実施 | 未実施 |
| 標準偏差 | 未実施 | 未実施 |

- 評価全体時間: 未実施
- 評価GPUピークメモリ: 未実施
- 同一画像のハッシュ: 未実施

## エラー・警告

蒸留: 取得失敗 — `unsupported operand type(s) for /: 'str' and 'float'`

原因調査:

DistillCfg.lrに型注釈がなく、明示した--lr=0.002がstrとして解釈された。Adamのlr / bias_correctionでTypeErrorが発生した。

計画値の学習率は0.002だが、保存済みcfg.jsonの実際のlrは `'0.002'`（str）だった。
CPU検証: 明示CLI指定時の型 `str`、CLI省略時の型 `float`。
前回の成功したジョブ9005977は--lrを省略して既定のfloat値を使っていた。今回の事前検証は値の型を確認できていなかった。
修正候補はCLIの--lr指定を省略して既定値0.002を使うか、設定に `lr: float = 2e-3` と型注釈を加えること。学習アルゴリズムの変更は不要。
この調査では学習コードを修正しておらず、再投入・評価ジョブ投入も行っていない。

警告（ログから抽出）:

```text
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/env/lib/python3.11/site-packages/outdated/__init__.py:36: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/env/lib/python3.11/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
wandb: WARNING `resume` will be ignored since W&B syncing is set to `offline`. Starting a new run with run id flowers102_pixel_adam_it5000_ipc1.
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:45: UserWarning: xFormers is disabled (SwiGLU)
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:29: UserWarning: xFormers is disabled (Attention)
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:35: UserWarning: xFormers is disabled (Block)
/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
Distilling Images:   0%|          | 0/5001 [00:00<?, ?it/s]/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/repo/src/distillation/linear_gm.py:199: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/repo/src/distillation/linear_gm.py:237: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
```

エラー行:

```text
Traceback (most recent call last):
TypeError: unsupported operand type(s) for /: 'str' and 'float'
DISTILL_STATUS {"status": "error", "completed_updates": 0, "config": {"dataset": "flowers102", "model": "dinov2_vitb", "distill_mode": "pixel", "distill_opt": "adam", "iterations": 5000, "ipc": 1, "lr": 0.002, "num_workers": 2, "augs_per_batch": 1, "checkpoint_it": 100, "image_log_it": 500, "syn_res": 256, "real_res": 256, "crop_res": 224, "init_mode": "noise", "aug_mode": "standard", "decorrelate_color": true, "train_crop_mode": "random"}, "job_id": "9006196", "host": "r18n3", "gpu": "NVIDIA H100", "torch": "2.8.0+cu126", "cuda_build": "12.6", "source_commit": "33d004b8c16d9ab8e39aec780f9edad767f6f033", "dataset_class": "PixelDataset", "optimizer_class": "Adam", "initial_parameter_sha256": "b6fcdaa636dfe1b96b027868a68db96e09545a18a4ef6f2bf8573b43963adf8a", "error": "unsupported operand type(s) for /: 'str' and 'float'", "traceback": "Traceback (most recent call last):\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/experiment/run_distill.py\", line 90, in main\n    module.main(cfg)\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/repo/src/distillation/distill.py\", line 50, in main\n    distillation.distill()\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/repo/src/distillation/linear_gm.py\", line 110, in distill\n    loss = self.match_gradients()\n           ^^^^^^^^^^^^^^^^^^^^^^\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/experiment/run_distill.py\", line 57, in match_gradients\n    loss = super().match_gradients()\n           ^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/repo/src/distillation/linear_gm.py\", line 171, in match_gradients\n    self.distilled_dataset.optimizer.step()\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/env/lib/python3.11/site-packages/torch/optim/optimizer.py\", line 516, in wrapper\n    out = func(*args, **kwargs)\n          ^^^^^^^^^^^^^^^^^^^^^\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/env/lib/python3.11/site-packages/torch/optim/optimizer.py\", line 81, in _use_grad\n    ret = func(*args, **kwargs)\n          ^^^^^^^^^^^^^^^^^^^^^\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/env/lib/python3.11/site-packages/torch/optim/adam.py\", line 247, in step\n    adam(\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/env/lib/python3.11/site-packages/torch/optim/optimizer.py\", line 149, in maybe_fallback\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/env/lib/python3.11/site-packages/torch/optim/adam.py\", line 949, in adam\n    func(\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/env/lib/python3.11/site-packages/torch/optim/adam.py\", line 761, in _multi_tensor_adam\n    step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])\n                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/env/lib/python3.11/site-packages/torch/optim/adam.py\", line 761, in <listcomp>\n    step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])\n                                      ~~~^~~~\nTypeError: unsupported operand type(s) for /: 'str' and 'float'\n", "total_seconds": 138.61507783876732, "peak_gpu_memory_bytes": 16480911872, "peak_reserved_gpu_memory_bytes": 17410555904}
```

## 保存先

実験ルート: `/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA`

- 最終合成画像: `/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/distillation/flowers102/dinov2_vitb/flowers102_pixel_adam_it5000_ipc1/data.pth`（未生成の場合は未実施）
- 画像PNG: `images/class_000.png`〜`class_101.png`、`synthetic_grid.png`
- Loss: `loss.csv`、`loss_curve.png`
- 中間チェックポイント: `checkpoints/step_00000.pth`、`step_00500.pth`…`step_04500.pth`
- 最終チェックポイント: `checkpoints/step_05000_final.pth`
- 同一実験内の最新チェックポイント: `/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/distillation/flowers102/dinov2_vitb/flowers102_pixel_adam_it5000_ipc1/ckpt.pth`（元実装どおり100stepごと更新）
- 公式評価の集計: `/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/distillation/flowers102/dinov2_vitb/flowers102_pixel_adam_it5000_ipc1/eval/dinov2_vitb.pth`
- 評価中間状態: `evaluation-checkpoints/`（元実装が終了時に削除する最新ファイルとは別に保持）
- 記録: `distill_status.json`、`eval_status.json`、`logs/`、`experiment/plan.json`、`experiment/source-manifest.json`
- 実行環境の固定記録: `environment-lock.txt`、既存データとモデル重みを再利用
- ジョブ会計記録・終了コード: `job-accounting.json`、`logs/qacct-distill.txt`（未取得なら未実施）
- 取得失敗時の原因: `failure-diagnosis.json`（存在する場合）

## 承認対象の実行コマンド

```bash
cd /gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA
qsub -g tga-lab_otm distill_job.sh
# 蒸留の正常終了とdata.pthの検証後にのみ、承認済みなら以下を実行
qsub -g tga-lab_otm eval_job.sh
```

実際の蒸留CLI（記録用ラッパーが同じ引数でmainを呼び出す）:

```bash
python -m distillation.distill --dataset=flowers102 --model=dinov2_vitb --distill_mode=pixel --distill_opt=adam --iterations=5000 --ipc=1 --lr=0.002 --num_workers=2 --augs_per_batch=1 --checkpoint_it=100 --image_log_it=500 --syn_res=256 --real_res=256 --crop_res=224 --init_mode=noise --aug_mode=standard --decorrelate_color=True --train_crop_mode=random --data_root=/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/data --job_tag=/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/distillation --run_name=flowers102_pixel_adam_it5000_ipc1 --skip_if_exists=False
```

公式評価CLI（ラッパーが同じ引数で公式モジュールを実行する）:

```bash
python -m distillation.eval --dataset=flowers102 --model=dinov2_vitb --eval_model=dinov2_vitb --num_eval=3 --eval_epochs=1000 --eval_it=-1 --patience=5 --checkpoint_it=100 --num_workers=2 --real_batch_size=100 --real_res=256 --crop_res=224 --train_crop_mode=random --data_root=/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/data --job_tag=/gs/bs/tga-lab_otm/us07824-lgm-pixel-5000-0RwHhugA/distillation --skip_if_exists=True
```

## 実行前の見積もり（実測結果ではない）

蒸留20〜60分・GPU割当16〜24 GiB、評価3回合計15〜45分・GPU割当4〜16 GiBを仮見積もり。
4ステップの短い実測からの外挿のため不確実。15分上限は本実験に不足する可能性が高く、蒸留2時間・評価1時間を提案。
gpu_1=1・優先度0のポイント: 蒸留約0.0867〜0.1800、評価約0.0550〜0.1250、合計約0.1417〜0.3050。
各ジョブが指定上限まで動いた場合は蒸留0.32、評価0.16、合計0.48ポイント。
式: 0.2 × (0.7 × max(実行秒,300) + 0.1 × 指定秒) / 3600。
https://www.t4.cii.isct.ac.jp/en/node/281

## Generator方式との比較に残す条件

同じFlowers-102 split、同じDINOv2重み、IPC1・102画像、画像解像度、augmentation、評価1000epochsと同じ3シード・ddofを使用する。
前回の4ステップは動作確認であり本実験精度には含めない。今回の5000iterationsが十分な収束・精度を保証するとは扱わない。
Generator方式へ変更した項目・パラメータ数・蒸留時間・GPUメモリ・全Loss履歴を別途記録し、表現方式以外の差を明示する。
