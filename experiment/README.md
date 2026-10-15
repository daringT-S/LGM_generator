# LGM実験結果

データセット別にPixelDataset方式とgenerator_pixel方式の結果を保存する。元のTSUBAME/WSL実験結果は変更していない。
`flower102/`は依頼に合わせたフォルダ名で、コード上のデータセット名は`flowers102`。
`stanford_dogs/`のCLI名は`stanforddogs`。

- [Flowers-102: 比較・追加3蒸留シードの再現性](flower102/README.md)
- [Stanford Dogs: Pixel / Generator比較](stanford_dogs/README.md)
- [全結果のCSV](results.csv)（精度は%、標準偏差はpercentage points）
- [全結果のJSON](results.json)（精度・標準偏差は0〜1のfraction）
- [実行ソースのスナップショット](source_snapshot/README.md)
- [アーカイブのSHA256・出所](archive-manifest.json)

各方式・各蒸留シードの評価は公式distillation.evalで独立初期化3回、seed3407/3408/3409。標準偏差はddof=0。
蒸留のiterations5000は現行コードではstep0〜5000の5001更新。
失敗・未実施・取得失敗は実測値と区別し、欠測値を推測で埋めていない。

設定、ジョブスクリプト、Loss CSV、グラフ、全クラスPNG、検証JSON、会計記録と圧縮ログを同梱。
元の保存先・data.pthやモデルのSHA256は各plan/status/manifestに記録。
モデル重み、Generatorパッケージ、data.pth、checkpoint、元データセットとキャッシュはTSUBAMEおよび元のreportsに保持し、Gitには含めない。
元のレポートに記載された.logは同じ場所の.log.gzを展開して参照できる。

過去の計画文書は作成当時の状態を保持している。最新の完了状態と集計はこの索引・データセット別README・execution-statusを参照する。
通常のsrc/や既存の作業中ファイルはこの結果整理では変更しない。
