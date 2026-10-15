# LGM Generator方式：実装と小規模テスト準備

GPUジョブは未投入・承認待ち。GPUのLoss、精度、実行時間、ピークメモリ、生成画像は未実施。
既存のPixel本実験・評価ログは変更していない。

## 変更内容と設計

| ファイル | 内容 |
|---|---|
| `src/synsets/generator.py` | ImageGeneratorとGeneratorDatasetを追加 |
| `src/config/distill_config.py` | distill_modeにgenerator_pixel、generator_latent_dim=128を追加 |
| `src/synsets/__init__.py` | 新方式のDataset選択分岐を追加 |
| `src/synsets/base.py` | 色変換用定数の配置先を指定可能にした。既定値は従来通りCUDA |
| `tests/test_generator.py` | 元のLGMメソッドを使用するCPU統合テストを追加 |
| `README.md` | 新しいCLIの使用法を追記 |

Generator構造:

`z(128) → Linear(8192) → (128,8,8) → Upsample+Conv+ReLU(64,16,16) → Upsample+Conv+ReLU(32,32,32) → Conv(3,32,32)`

最後にsyn_resへbilinear補間し、Pixel方式と同じ色変換とsigmoid(2x)を適用する。
RGB画像、[0,1]の値域、解像度を既存形式に合わせる。最初の実装は32×32の出力を補間するため、細部の表現力には制約がある。
Generatorは全画像で共有する。各クラスにIPC個の固定ランダムノイズを割り当て、クラス条件入力は追加しない。
ノイズは初期化時だけ生成してrequires_grad=False。ラベルも固定する。Adamのパラメータ群はGeneratorパラメータと完全一致する。

再利用するもの: BaseDistilledDatasetの色変換、共通のget_data/log_images/保存復元インターフェース、既存log_images関数、Pixelと同じ出力変換。
Pixel/Pyramid本体、LinearGMの損失・二階勾配・更新処理、公式評価コードは無変更。
画像をdetachせずget_dataで返すため、元のcreate_graph=Trueを通じてGeneratorまで勾配が流れる。
従来のPixel方式とPyramid方式を選ぶCLIも残し、既定方式もpyramidのまま。

## 保存と評価への接続

- checkpoint: 表現設定とバージョン、Generator重み、固定ノイズ、ラベル、Adam状態。共通のcheckpointにはstepと乱数状態も入る。
- 復元時: 解像度・latent_dim・IPC・クラス数・色変換等の一致、ノイズ形状、ラベルを確認する。
- data.pth: 元のLinearGM.save_dataで生成済み画像imagesとlabelsをCPUテンソルで保存する。
- PNG: クラス別102枚、一覧画像。チェックポイントとは別に保存する。
- 公式評価: Generatorではなく保存済みdata.pthを通常のTensorDatasetとして読む。評価アルゴリズムの変更は不要。

## CPUと静的検証

実行環境: TSUBAMEの既存Python 3.11 / PyTorch 2.8.0+cu126。CUDA_VISIBLE_DEVICESを空にし、GPUを使用していない。
学習確認は3クラス・小さな固定バックボーンを使うCPUテストに限定した。Flowers102+DINOv2でのGPU実行は未実施。

| 検証 | 結果 |
|---|---|
| 実際のCLIパーサー、新方式の選択、lr=float、数値設定=int | 成功 |
| RGB画像の生成、syn_res=16/32/37/256、値域・有限性、色変換あり/なし | 成功 |
| 元のLGM.match_gradients/get_real_grad/get_syn_gradによる4更新 | 成功 |
| 全Generatorパラメータへの有限かつ非ゼロの勾配、Adam更新 | 成功 |
| 固定ノイズ・ラベルの不変性、ノイズ勾配なし | 成功 |
| checkpoint保存復元と、その後の同一乱数での更新結果一致 | 成功 |
| 元のsave_dataで出力し、評価と同じimages/labels形式でロード | 成功 |
| Pixel/Pyramidの元のLGMによる更新 | 成功 |
| 未対応設定の明示的拒否 | 成功 |

unittestは7テストすべて成功（0.678秒、import時間を除く）。
初回はPyramid内の明示的なCUDA割当をCPUへ切り替えきれずテスト失敗。
テスト内の配置切り替えだけを修正し、初回ログ`logs/cpu-tests.log`と成功ログ`logs/cpu-tests-r2.log`の両方を保持した。
元のCUDA autocastはCPUテストでは自動無効化される警告が出る。依存ライブラリの非推奨警告も保持した。

Python AST構文検査、bash -n、git diff --checkは成功。
投入前検証も成功: 全ソースのハッシュ一致、実際のCLI引数、資源指定、モデル重みのハッシュ、専用出力先を確認した。
Generatorのパラメータ数は1,149,891。詳細は`preparation-validation.json`に保存した。ジョブ投入数は0。
Pixel/Pyramid/LinearGM/公式評価/W&B run_name処理のファイル内容が既存HEADと一致することを確認した。
GPU実行時の混合精度・全102画像での二階勾配・H100メモリ使用量は、承認後の小規模テストで確認する。

## 承認対象のジョブ

| 項目 | 指定 |
|---|---|
| Dataset / model | flowers102 / dinov2_vitb |
| Mode / optimizer / lr | generator_pixel / Adam / 0.002 |
| Iterations / IPC | 3（step0〜3、4更新）/ 1 |
| 固定ノイズ / seed | 102×128 / 3407 |
| workers / augs_per_batch | 2 / 1 |
| syn_res / real_res / crop_res | 256 / 256 / 224 |
| Init / augmentation / 色変換 | noise / standard / True |
| W&B | offline、既存のrun_name修正を使用 |
| グループ / 資源 | tga-lab_otm / gpu_1=1、H100×1 |
| 優先度 / 時間上限 / ジョブ数 | 0 / 00:15:00 / 1本 |

見積もり: 1〜5分、GPU割当16〜24GiB。Pixelの実測を参考にした予想でありGeneratorの実測ではない。
指定上限900秒で、5分以内に終了すれば0.0166ポイント、上限まで使った場合0.0400ポイント。

`0.2 × (0.7 × max(実行秒,300) + 0.1 × 900) / 3600`、小数第5位以下切り捨て。
[公式課金規則](https://www.somuka.titech.ac.jp/reiki_int/reiki_honbun/x385RG00001693.html)を2026-10-14に確認。ストレージ課金はこのGPUジョブ見積もりに含めない。

既存のPython環境、Flowers102データ、DINOv2モデル重みを再利用し、変更しない。
学習結果が存在するディレクトリでの再実行を拒否する。エラー時の自動再投入はしない。

専用実験ルート:

`/gs/bs/tga-lab_otm/us07824-lgm-generator-smoke-j4SzPSSj`

最終data.pthの予定保存先:

`/gs/bs/tga-lab_otm/us07824-lgm-generator-smoke-j4SzPSSj/distillation/flowers102/dinov2_vitb/flowers102_generator_pixel_adam_it3_ipc1/data.pth`

他の保存先: `checkpoints/step_00000.pth`〜`step_00003.pth`、`images/`、`synthetic_grid.png`、`loss.csv`、`loss_curve.png`、`status.json`、`logs/gpu.log`。
ソース固定: `experiment/source-manifest.json`。CLI全引数: `experiment/plan.json`。

承認後にのみ実行する投入コマンド:

```bash
cd /gs/bs/tga-lab_otm/us07824-lgm-generator-smoke-j4SzPSSj
qsub -g tga-lab_otm gpu_job.sh
```

計算ノードでは`python -u "$EXPERIMENT_DIR/experiment/run_smoke.py"`から、以下のCLIと同じ引数で元のmainを呼ぶ。

```bash
python -m distillation.distill --dataset=flowers102 --model=dinov2_vitb --distill_mode=generator_pixel --distill_opt=adam --generator_latent_dim=128 --iterations=3 --ipc=1 --lr=0.002 --num_workers=2 --augs_per_batch=1 --syn_res=256 --real_res=256 --crop_res=224 --init_mode=noise --aug_mode=standard --decorrelate_color=True --train_crop_mode=random --checkpoint_it=1 --image_log_it=3 --data_root=/gs/bs/tga-lab_otm/us07824-lgm-pixel-rEKz7ETv/data --job_tag=/gs/bs/tga-lab_otm/us07824-lgm-generator-smoke-j4SzPSSj/distillation --run_name=flowers102_generator_pixel_adam_it3_ipc1 --skip_if_exists=False
```

本テストでは分類精度の評価ジョブを投入しない。
後のPixel比較では、同じ学習条件と、同じ公式評価（1000epochs・3回・seed3407/3408/3409・ddof=0）を使用し、Top-1/Top-5、Loss、時間、メモリ、画像を記録する。
今回の4更新をPixelの5000iterationsと同等の実験として比較しない。
