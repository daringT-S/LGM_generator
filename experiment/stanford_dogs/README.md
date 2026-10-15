# stanforddogs: Pixel / Generatorの実験結果

共通条件: DINOv2 ViT-B、IPC1、Adam lr0.002、iterations5000（5001更新）、workers2、augs_per_batch1、TSUBAME H100×1、W&B offline。
公式distillation.eval、評価seed3407/3408/3409、eval_epochs1000、各回で分類器を初期化。標準偏差はddof=0。

| 方式 | 蒸留seed | Top-1平均 (%) | 評価内std (pt) | Top-5平均 (%) | 評価内std (pt) |
|---|---:|---:|---:|---:|---:|
| pixel | 3407 | 51.2277 | 0.0440 | 88.6053 | 0.4522 |
| generator_pixel | 3407 | 77.1212 | 0.2480 | 96.4219 | 0.0415 |

蒸留seed3407同士のGenerator−PixelのTop-1差: **+25.8935ポイント**。

全4ジョブが正常終了し、独立CPU検証passed。
Pixel蒸留9013760 → Pixel評価9013854 → Generator蒸留9013956 → Generator評価9014227。

公式train12,000枚/test8,580枚、120クラス、IPC1で合成画像120枚。bbox切り出しなし。
各方式の蒸留はseed3407で1回。Dogsでは蒸留seed間の再現性検証は未実施。

## 詳細

- [実験レポート（時間・GPUメモリ・全精度・Loss・生成画像・警告）](comparison/experiment_report.md)
- [実行完了・ジョブID](comparison/execution-status.json)
- [データ監査](comparison/data-validation.json)
- [全6回の精度CSV](comparison/accuracies.csv)

![精度比較](comparison/accuracy_comparison.png)
![Loss比較](comparison/loss_comparison.png)
![同じクラスの生成画像比較](comparison/synthetic_samples_comparison.png)
