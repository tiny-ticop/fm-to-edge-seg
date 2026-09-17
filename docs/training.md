# E001 Baseline学習

## 目的

Foundation Modelを追加する前の軽量Student基準値を取得します。この結果が、DINOv3 logit distillationやSAM 3擬似ラベルによる改善量の比較元になります。

## モデル

- Encoder: MobileNetV3-Small（ImageNet pretrained）
- Decoder: depthwise separable convolutionを使ったLite U-Net
- Parameters: 約100万
- Loss: BCE + Dice
- Input: 544×384

## ローカルsmoke test

GPU非搭載PCでは、全学習の前に2 train batchと1 validation batchだけを実行できます。

```powershell
python -m fm_to_edge_seg train `
  configs/experiment/e001_deepcrack_baseline.yaml `
  --device cpu `
  --epochs 1 `
  --num-workers 0 `
  --max-train-batches 2 `
  --max-validation-batches 1 `
  --no-pretrained
```

これは精度評価ではなく、学習経路の動作確認です。

## Colabでフル学習

`notebooks/colab_bootstrap.ipynb`を開き、DeepCrack準備セルまで実行した後に次を実行します。

```bash
python -m fm_to_edge_seg train configs/experiment/e001_deepcrack_baseline.yaml
```

`device: auto`により、CUDAが利用可能ならGPUを使います。初回はImageNet pretrained weightもダウンロードします。

## 保存される成果物

```text
artifacts/e001_deepcrack_baseline/
├── best.pt
├── last.pt
├── history.jsonl
├── run.json
├── summary.json
└── validation_predictions.png
```

- `best.pt`: validation Diceが最高だったepoch
- `last.pt`: 最終epoch
- `history.jsonl`: epochごとのtrain/validation指標
- `run.json`: config、環境、Git commit
- `summary.json`: best epochと実行時間
- `validation_predictions.png`: 元画像、赤い正解、緑の予測

## 指標

- IoU
- Dice
- Precision
- Recall
- BCE
- Dice Loss

細線ではIoU/Diceだけで評価しきれないため、後続実験でBoundary F1とclDiceを追加します。
