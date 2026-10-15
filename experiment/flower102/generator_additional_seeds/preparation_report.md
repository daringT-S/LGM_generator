# Generator蒸留のシード再現性：承認待ちの実行計画

GPUジョブは未投入。新規蒸留3本と、各蒸留に対する評価3回をまとめた評価ジョブ3本、合計最大6本を準備した。
1本ずつ直列に実行する。失敗時の自動再投入は行わない。

## 既存実験の確認と変更範囲

既存Generator蒸留9010498のシードは **3407**。
元のrun_distill.pyのtorch/NumPy/Pythonのseed設定、plan.json、および最終checkpointのCPU乱数状態から確認した。
既存評価9011526のTop-1平均は99.5609045029%、Top-5平均は99.8644749324%。

新規蒸留シードは42・123・2026。シードは各実験の`experiment/plan.json`に置き、実行ラッパーが読む。
DistillCfgに存在しない`--seed`オプションは使用しない。
ラッパーは前回と同じ位置で`torch.manual_seed(seed)`、`np.random.seed(seed)`、`random.seed(seed)`を呼び、
元の`distillation.distill.main(cfg)`を直接呼ぶ。モジュールの`__main__`にある3407の固定設定は実行しない。

Generatorの重みはCPU上の初期化で、固定ノイズはCUDA上のrandnで生成される。
実GPUジョブでは生成直前にCPU/CUDA双方の初期シードを確認し、乱数状態と初期重み・ノイズのSHA256を`initialization.json`へ保存する。
生成直前の追加seedリセットは行わず、元のDataLoader初期化順も保つ。
学習するのはGeneratorのパラメータだけ。固定ノイズはrequires_grad=Falseで、100stepごとと最終で不変性を検査する。

損失、Adamの処理、Generator構造、Pixel/Pyramid、公式評価コードは既存3407のスナップショットと同一。
変更は実験ラッパーのシード指定、初期状態の記録、独立した保存先に限定した。
run_name/job_tagは衝突回避のためシードごとに別名にする。

## CPU検証

`seed-validation.json`と`logs/seed-validation.log`に記録。

- 既存checkpointのCPU乱数状態のseedが3407である。
- 3407・42・123・2026について、同じシードで2回初期化すると重み・固定ノイズのハッシュが一致する。
- 異なるシードでは重み・ノイズのハッシュが異なる。
- ラベルは0〜101で各1枚、ノイズは102×128で学習対象ではない。
- 新規蒸留の全設定は既存3407と一致（seedと保存先メタデータを除く）。
- 評価設定は既存評価と一致し、シード3407・3408・3409、1000epochs、3回を維持。
- 成功時だけ評価へ進む判定をテスト。失敗・更新数不足・異なるseed・画像数不足・終了コード異常・入力ハッシュ不一致を拒否する。
- シード内・シード間集計の計算を小さな合成数列でテスト。テスト用数値は実験精度に混ぜない。
- 実装srcは既存3407と全Pythonファイルのハッシュが一致する。

CPUとCUDAではノイズ生成器が異なるため、CPUテストからCPU/GPUの固定ノイズのビット一致は主張しない。
GPUの初期シードと固定ノイズのハッシュは実行時にも確認する。GPU実行のビット単位の決定性を保証する設定変更は加えない。

## 共通の学習・評価条件

| 項目 | 指定 |
|---|---|
| Dataset / model / mode | flowers102 / dinov2_vitb / generator_pixel |
| 蒸留 | Adam、lr0.002、IPC1、iterations5000（5001更新）|
| 蒸留seed | 42 / 123 / 2026 |
| workers / augs_per_batch | 2 / 1 |
| 解像度 | syn_res256、real_res256、crop_res224 |
| Generator | latent128、32×32出力→256補間、既存色変換とsigmoid(2x) |
| init / augmentation / 色変換 | noise / standard / True |
| 評価 | 公式distillation.eval、num_eval3、eval_epochs1000 |
| 評価seed | 毎回3407 / 3408 / 3409、各回で新規分類器・Adam・scheduler |
| 評価前処理 | 既存AugBasicとImageNet mean/std、test Resize256→CenterCrop224 |
| 評価split | Flowers102 test6149枚、train1020枚との重複なし |
| 評価その他 | batch100、eval_it=-1、patience5、checkpoint_it100 |
| グループ / GPU / 優先度 / W&B | tga-lab_otm / gpu_1=1（H100×1）/ 0 / offline |

## 時間・ポイント

| 種類 | 本数 | 1本の予想時間 | 1本の時間上限 | 1本の最大ポイント | 小計最大 |
|---|---:|---|---|---:|---:|
| 蒸留 | 3 | 約25〜40分 | 2時間 | 0.32 | 0.96 |
| 評価（各3回）| 3 | 約10〜20分 | 1時間 | 0.16 | 0.48 |
| 合計 | 6 | 直列で約105〜180分＋待ち時間 | 合計9 GPU時間 | — | **1.44** |

既存Generatorの実測は蒸留処理20分58秒、会計23分36秒、GPUピーク15.652GiB。
評価は全体6分41秒、会計7分8秒、GPUピーク2.046GiB。
新規のメモリは蒸留16〜24GiB・評価2〜4GiBを見込む。見積もりであり各seedの実測ではない。
上記予想時間で終了した場合、合計約0.425〜0.600ポイント。既存実測を3組に換算した参考値は約0.3948ポイント。

式は`0.2 × (0.7 × max(実行秒,300) + 0.1 × 指定秒) / 3600`（小数第5位以下切り捨て）。
[公式課金規則](https://www.somuka.titech.ac.jp/reiki_int/reiki_honbun/x385RG00001693.html)を2026-10-15に確認。
ストレージ課金は含まない。追加保存量は約2〜5GBの余裕を見込む。既存データ・モデル・Python環境を再利用し、変更しない。

## 保存先と投入順

専用ルート: `/gs/bs/tga-lab_otm/us07824-lgm-generator-seeds-wrpWhzXU/`

| 順 | 蒸留seed | 蒸留先 | 評価先 |
|---|---:|---|---|
| 1 | 42 | seed_42/distill/ | seed_42/eval/ |
| 2 | 123 | seed_123/distill/ | seed_123/eval/ |
| 3 | 2026 | seed_2026/distill/ | seed_2026/eval/ |

各蒸留の`distillation/flowers102/dinov2_vitb/flowers102_generator_pixel_adam_it5000_ipc1_seed{seed}/data.pth`に102画像・ラベルを保存する。
併せて`generator_final.pth`、`initialization.json`、PNG102枚・一覧、Loss CSV/グラフを保存する。
最新状態を100stepごとに保存し、0/500/.../4500/5000の11個を別名checkpointとして保持する。
最終checkpointには重み・ノイズ・ラベル・Adam状態・step・乱数状態を保存する。
途中再開は自動実行しない。元実装の再開stepやDataLoader位置の制約は既存と同じ。

各評価は蒸留の正常終了（qacct exit_status0/failed0）、5001更新、PNG102枚、保存内容の独立CPU検証に成功した場合だけ実行する。
保存済みdata.pthをそのまま評価専用入力にコピーし、読み取り専用にする。Generatorを再生成・再学習しない。
CPUでの再生成差は評価入力に入らない。評価前後で元データ・重み・ノイズを含むパッケージのハッシュを検査する。

実際のqsubは各phaseの作業フォルダから`qsub -g tga-lab_otm distill_job.sh`または`qsub -g tga-lab_otm eval_job.sh`。
`submit.py`に一度だけの投入記録、承認ファイル、評価前の成功条件、同時実行1本のガードを用意した。
**approval.jsonは未作成であり、承認前には投入できない。**

承認後の手順（現在は実行しない）:

```bash
cd /gs/bs/tga-lab_otm/us07824-lgm-generator-seeds-wrpWhzXU
python3 submit.py distill 42
# 蒸留終了→record_accounting.py distill 42→CPUのseed_42/distill/experiment/validate_results.pyがpassed
python3 submit.py eval 42
# 評価終了→record_accounting.py eval 42→CPUのseed_42/eval/experiment/validate_evaluation.pyがpassed
# 同じ順序で123、2026へ進む。失敗したphaseは再投入しない。
```

## 集計とグラフ

各蒸留seedに対しTop-1/Top-5の3つの実測値、平均・標準偏差（ddof=0）、蒸留時間・GPUメモリ・全Lossを表にする。
主解析は新規3シードの評価平均を3点として、平均と母標準偏差を計算する。
既存3407を含む4シードの集計は副次解析として別表にし、新規3点と混同しない。
評価9回を独立な蒸留9回と見なした標準偏差は使用しない。

`aggregate.py`は、各評価の成功と独立検証passedを確認した実測値のみを読む。
`study-results.json`、`comparison_report.md`、Loss比較、精度の評価内stdと各評価点を示すグラフを作成する。
現時点の新規結果はすべて未実施。比較グラフは実験完了後に作成する。
レポート・グラフ・ログはTSUBAMEに保存し、WSLの同名reportsディレクトリへ同期する。


## 最終監査

CPUシード検証・実行計画監査ともpassed。GPUジョブ投入数0、approval.json未作成。
既存3407の初期Generator重みハッシュをCPUでも再現し、実測初期ハッシュとの一致を確認した。
監査テスト初回は__init__の戻り値の期待値誤りにより失敗し、テストのみ修正後にpassedとなった。初回ログも保持した。
詳細: seed-validation.json、execution-plan-validation.json。新規結果表: [comparison_report.md](comparison_report.md)。
