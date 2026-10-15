# Generator蒸留シードの再現性比較

GPUジョブはユーザー承認後にのみ投入する。未取得の値は未実施とする。

| 蒸留seed | Top-1（評価3回）| Top-1 平均±評価std | Top-5（評価3回）| Top-5 平均±評価std | 蒸留秒 | GPU GiB |
|---:|---|---|---|---|---:|---:|
| 42 | 99.5446%, 99.5446%, 99.3983% | 99.4959% ± 0.0690 pt | 99.8211%, 99.8048%, 99.8374% | 99.8211% ± 0.0133 pt | 1499.866647590883 | 15.652087688446045 |
| 123 | 99.3658%, 99.4959%, 99.4796% | 99.4471% ± 0.0579 pt | 99.9349%, 99.9512%, 99.9349% | 99.9404% ± 0.0077 pt | 1911.9031624952331 | 15.652087688446045 |
| 2026 | 99.2519%, 99.5121%, 99.3332% | 99.3658% ± 0.1087 pt | 99.9349%, 99.9024%, 99.9349% | 99.9241% ± 0.0153 pt | 1338.1782441479154 | 15.652087688446045 |

評価内stdは各蒸留結果に対する3回の評価の母標準偏差（ddof=0）。
蒸留seed間stdは42・123・2026の各評価平均を3点として計算する。9回の評価を独立な蒸留9回として数えない。
既存3407を含む4点の集計は副次解析として別記する。

## シード間集計

| 集計対象 | Top-1平均 | 蒸留seed間Top-1 std | Top-5平均 | 蒸留seed間Top-5 std |
|---|---:|---:|---:|---:|
| 新規42/123/2026（主解析） | 99.4362% | 0.0537 pt | 99.8952% | 0.0528 pt |
| 既存3407を含む4点（副次解析） | 99.4674% | 0.0712 pt | 99.8875% | 0.0476 pt |

## 各実験のファイル

- seed 42: `seed_42/distill/loss.csv`、`loss_curve.png`、`images/`、`generator_final.pth`、`checkpoints/`、`seed_42/eval/eval_status.json`。
- seed 123: `seed_123/distill/loss.csv`、`loss_curve.png`、`images/`、`generator_final.pth`、`checkpoints/`、`seed_123/eval/eval_status.json`。
- seed 2026: `seed_2026/distill/loss.csv`、`loss_curve.png`、`images/`、`generator_final.pth`、`checkpoints/`、`seed_2026/eval/eval_status.json`。

![Loss比較](loss_comparison.png)
![精度比較](accuracy_comparison.png)
