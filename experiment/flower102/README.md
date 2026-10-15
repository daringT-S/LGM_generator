# flowers102: Pixel / Generatorの実験結果

共通条件: DINOv2 ViT-B、IPC1、Adam lr0.002、iterations5000（5001更新）、workers2、augs_per_batch1、TSUBAME H100×1、W&B offline。
公式distillation.eval、評価seed3407/3408/3409、eval_epochs1000、各回で分類器を初期化。標準偏差はddof=0。

| 方式 | 蒸留seed | Top-1平均 (%) | 評価内std (pt) | Top-5平均 (%) | 評価内std (pt) |
|---|---:|---:|---:|---:|---:|
| pixel | 3407 | 97.1432 | 0.2696 | 取得失敗 | 取得失敗 |
| generator_pixel | 3407 | 99.5609 | 0.0133 | 99.8645 | 0.0334 |
| generator_pixel | 42 | 99.4959 | 0.0690 | 99.8211 | 0.0133 |
| generator_pixel | 123 | 99.4471 | 0.0579 | 99.9404 | 0.0077 |
| generator_pixel | 2026 | 99.3658 | 0.1087 | 99.9241 | 0.0153 |

蒸留seed3407同士のGenerator−PixelのTop-1差: **+2.4177ポイント**。

追加蒸留seed42/123/2026のTop-1評価平均を3点として集計: 平均 **99.4362%**、蒸留seed間標準偏差 **0.0537ポイント**。
表の評価内stdとは別の指標。9回の評価を独立な蒸留9回として扱わない。

Pixel評価9006786は3回の精度計算後、保存ラッパーのtorch.saveキーワード引数不一致で失敗。Top-1全3値はログから復元済み。Top-5の3回目と3回分集計は取得失敗。
初期のlr型エラージョブ9006196はhistoryに保存。Generatorの4step GPUテストは省略されたため、CPUテスト準備記録をGPU実測として扱わない。

## 詳細

- [Pixel蒸留・評価記録](pixel_seed3407/experiment_report.md)
- [Pixel復元評価](pixel_seed3407/evaluation-recovered.json)
- [Generator seed3407蒸留](generator_seed3407/distillation/experiment_report.md)
- [Generator seed3407評価](generator_seed3407/evaluation/experiment_report.md)
- [追加3蒸留シードの比較表・Loss・画像](generator_additional_seeds/experiment_report.md)
- [失敗したlr実験](history/pixel_lr_error_9006196/experiment_report.md)
- [CPUテスト・小規模GPUテストの準備記録](history/generator_cpu_test_preparation/experiment_report.md)

![追加蒸留シードの精度比較](generator_additional_seeds/accuracy_comparison.png)
![追加蒸留シードのLoss比較](generator_additional_seeds/loss_comparison.png)
