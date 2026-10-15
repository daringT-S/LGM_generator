# Stanford Dogs: Pixel / Generator比較（承認待ち）

GPUジョブは未投入。ソースコード変更なし。データ本体は未取得。TSUBAME上の専用ディレクトリはまだ作成していない。

## 対応状況

`src/data/dataloaders/stanford_dogs.py`のStanfordDogsと`src/data/dataloaders/__init__.py`のfactoryが既に対応。
CLIの正式名は **stanforddogs**。依頼文の`stanford_dogs`は現在のfactoryで未対応の表記であり、投入コマンドには`--dataset=stanforddogs`を使用する。
新しいdatasetの追加は不要。Pixel/Pyramid/Generator/LGM損失・Adam・公式評価コードを変更しない。
`stanford_dogs`という別名を追加する必要もなく、最小変更は実験専用ラッパー・記録処理の準備だけとする。

既存ローダーのdownload()は画像・Annotationの120フォルダだけで完了判定し、.matファイルの欠落は検査しない。
このためGPU実行前の独立したデータ監査で全必要ファイルと分割を検証し、ローダーの自動取得に本実験を依存させない。
既存の共有環境・共有データディレクトリへの追記はせず、Dogsデータは今回専用のdata/に準備する。

## 分割・ラベル・前処理

[TFDS公式カタログ](https://www.tensorflow.org/datasets/catalog/stanford_dogs)記載: 120犬種、train12,000枚、test8,580枚、計20,580枚。
原配布のtrain_list.mat/test_list.matを読み、annotation_listのパスに.jpgを付加する。1始まりのlabelsから1を引き0〜119にする。
trainのみを蒸留に使い、testのみを最終精度に使う。別のランダム分割やtestによるハイパーパラメータ調整は行わない。

既存コードは画像全体のRGBを使い、bounding box切り出しは行わない。この条件をPixel/Generatorで揃える。
- 蒸留の実画像: Resize(256)→RandomCrop(256)→ToTensor、続いて元のaugmentationの224crop。
- 合成画像: 120枚、[120,3,256,256] float32、[0,1]。ラベルint64 0〜119を各1個。
- 評価test: Resize(256)→CenterCrop(224)→ToTensor。
- モデル入力直前: ImageNet mean=[.485,.456,.406]、std=[.229,.224,.225]。
- 合成画像評価: 既存AugBasic(crop_res224)と同じnormalizeを利用。

確認した2つの既存データ保存先にStanfordDogsはなかった。グループ領域全体を網羅検索したわけではない。
TSUBAMEから公式images.tarのHTTP HEADは200、Content-Length=793,579,520。HTTPSは証明書hostname不一致。
証明書検証の無効化は行わない。既存ローダーと同じ公式HTTPを使う場合も、公開チェックサムが確認できれば照合し、取得物のSHA256を記録する。
大きなデータの取得・展開はまだ実施していない。

取得後の必須監査:
1. images/annotation/listファイルの存在、JPEG読込とRGB変換、クラス120、train12,000/test8,580。
2. splitパス重複なし、train/test間の画像パス非重複、ラベル範囲0〜119と全クラス出現、train各クラス100枚。
3. classフォルダとラベル対応、画像パス存在、破損なし、前処理後のshape/値域/normalize。
4. IPC1のPixel/Generator保存形式と公式評価入力の互換性。

実データによる監査は**未実施**。ローダーのCPU模擬テストと混同しない。

## 共通条件と比較の単位

flowers102で使用したPython3.11、torch2.8.0+cu126、DINOv2ローカル重みを再利用する。
蒸留seedは両方式とも既存のデフォルト3407を採用する（新規指定がないため）。方式間で初期化に消費する乱数の数は異なり、同じ乱数列の位置まで一致する主張はしない。
蒸留自体は各方式1回。分類器評価3回を独立な蒸留3回として扱わない。Dogsで蒸留seed再現性を調べる追加ジョブは今回の4本には含めない。

Adam lr0.002、IPC1、iterations5000=5001更新、workers2、augs_per_batch1、syn/real256・crop224、standard augmentation、noise初期化、色変換True。
Generator latent128・固定z、Generatorパラメータのみ更新。Pixelは既存の画素パラメータのみ更新。
W&B offlineのrun_name既修正を再利用。数値型を実パーサーで検証し、設定スナップショット・ソースSHA256を保存する。

評価は公式distillation.eval、num_eval3、eval_epochs1000、seed3407/3408/3409。
各回でlinear classifier・Adam・schedulerを初期化。backboneを含む既存の評価手順を保持する。
評価用Adamのlrは公式式のまま（合成画像batch100の場合0.001×100/256=0.000390625）。蒸留lr0.002へ置き換えない。
120画像なので評価DataLoaderは100枚+20枚の2batch/epoch。eval_it=-1、patience5、checkpoint_it100で最終epochの精度を記録する。
各回Top-1/Top-5、3回平均と母標準偏差ddof0を保存する。

## 実行計画・資源

全ジョブ: H100×1（gpu_1=1）、tga-lab_otm、優先度0。1本ずつ実行し、失敗時に自動再投入しない。

| 順 | 処理 | 実行時間見込み | 上限 | GPU peak見込み | 最大ポイント |
|---:|---|---|---|---|---:|
| 1 | Pixel蒸留 | 30〜50分 | 2時間 | 20〜28GiB | 0.32 |
| 2 | Pixel評価3回 | 10〜20分 | 1時間 | 3〜5GiB | 0.16 |
| 3 | Generator蒸留 | 30〜50分 | 2時間 | 20〜28GiB | 0.32 |
| 4 | Generator評価3回 | 10〜20分 | 1時間 | 3〜5GiB | 0.16 |
| 合計 | 蒸留2本+評価2本 | 80〜140分+待ち時間 | 合計6GPU時間 | 同時実行1本 | **0.96** |

時間・メモリはDogsの実測ではない。FlowersのGenerator蒸留実測22〜32分、15.652GiB、評価約7分、2.046GiBを基に余裕を含めた推定。
蒸留batchはIPC×augs×クラス数なので102→120と約1.176倍。train画像数が約12倍でも固定5001更新の計算量が12倍になるわけではない。
Dogsでは実画像の読み込みとtest8,580枚、評価の2batch目20枚なども時間に影響する。
H100のメモリ内に収まる見込みだが、GPU実測は承認後に記録し、OOMなどの失敗時は停止する。

[TSUBAME公式規則](https://www.somuka.titech.ac.jp/reiki_int/reiki_honbun/x385RG00001693.html)（2026-10-15確認）:
`0.2*(0.7*max(actual_seconds,300)+0.1*requested_seconds)/3600`、ジョブごと小数第5位以下切り捨て。
上記予想時間なら合計約0.307〜0.447ポイント。最大0.96ポイント。ストレージ課金は含めない。
データ・画像・チェックポイント・ログ保存に約3〜6GBの余裕を見込む。実測して記録する。

## 保存先

承認後に作成するTSUBAME専用ルート:
`/gs/bs/tga-lab_otm/us07824-lgm-stanford-dogs-comparison-plan-q361seki/`

- `data/StanfordDogs/`: 今回の実データ・分割と取得manifest。
- `pixel/distill/`と`pixel/eval/`: Pixel専用、Generator側を探索しない。
- `generator/distill/`と`generator/eval/`: Generator専用、Pixel側を探索しない。
- 各蒸留: data.pth、PNG120枚・一覧、Loss CSV/グラフ、初期状態・設定、checkpoint11個（step0/500/.../4500/5000）、最新checkpoint100step間隔。
- Generatorのみ追加: 重み・固定z・ラベル・Adamを含むgenerator_final.pth。
- 各評価: 読み取り専用入力コピー、入力SHA256、3回各値、平均/標準偏差、classifier初期化記録、評価checkpoint、qacct・GPUピーク。
- ルート: experiment_report.md、精度比較表、Loss比較図、同じクラスの合成画像サンプル比較。

WSL側は今回のplanと同じreportsディレクトリに同期する。既存のFlowers-102ディレクトリは出力先に指定しない。
既存ファイルが見つかったら上書きせず停止する。

## 実行コマンド案（未実行）

実際には既存と同じ監視・checkpoint保存ラッパー経由で下記設定を渡し、損失・更新ルーチンを直接再実装しない。
両蒸留のseed3407はエントリポイント既定値と同じで、Generator構築前に適用される。

```bash
# Pixel蒸留（専用job_tag）
python -m distillation.distill --dataset=stanforddogs --model=dinov2_vitb \
  --distill_mode=pixel --distill_opt=adam --lr=0.002 --ipc=1 \
  --iterations=5000 --num_workers=2 --augs_per_batch=1 \
  --run_name=stanforddogs_pixel_adam_it5000_ipc1_s3407 \
  --job_tag=$STUDY/pixel/distill/output --data_root=$STUDY/data
# Generatorはdistill_mode=generator_pixel、専用run_nameとjob_tagへ変更
# 評価は方式別job_tag/input配下のdata.pthを1個だけ探索
python -m distillation.eval --dataset=stanforddogs --model=dinov2_vitb \
  --eval_model=dinov2_vitb --num_eval=3 --eval_epochs=1000 \
  --num_workers=2 --real_batch_size=100 --eval_it=-1 \
  --job_tag=$STUDY/pixel/eval/input --data_root=$STUDY/data
```

W&B/CUDA/DINOキャッシュ環境をjob scriptで指定し、ログインノードで学習しない。
`qsub -g tga-lab_otm distill_job.sh` / `qsub -g tga-lab_otm eval_job.sh`。
各蒸留のexit0、5001更新、120画像、labels0〜119、finite値、最終checkpointとdataの整合検証がpassedの場合のみ、その方式の評価へ進む。
Generatorは固定zの不変と重み更新、Pixelは画素パラメータ更新をそれぞれ確認する。
評価中は入力data/Generator重みを変更しない。CPUで再生成したGenerator画像を評価入力に差し替えない。

承認を受けるまでGPUジョブを投入しない。精度・Loss実測などの未取得値は「未実施」とする。

## 実施済みCPUローダー検証

`cpu-loader-validation.json`に記録。実際のget_dataset/StanfordDogsを呼び、ダウンロード・実ファイル読込・CUDA配置だけをモックした。
120クラス/クラス名120、1..120→0..119、train RandomCrop256/test CenterCrop224、画像float32[0,1]、ImageNet mean/stdを確認してpassed。
模擬ファイル名のtrain/test非重複も確認したが、実データの非重複の証明ではない。実データ件数・破損・split照合は未実施。
既存環境のpkg_resources非推奨警告あり。ソースファイルは変更していない。
