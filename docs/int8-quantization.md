# E007: ONNX Runtime INT8静的量子化

## 目的

E006で作ったFP32 ONNXをINT8へPost-Training Static Quantization（PTQ）し、次を比較します。

- segmentation精度
- model size
- CPU latency / FPS
- 未知条件`ood_test`の精度

CNNではactivationのscaleを推論時に毎回計算する動的量子化より、代表画像で事前計算する静的量子化が基本候補です。本PoCではONNX RuntimeのS8S8 QDQ形式を使用し、Convをweight/activationともINT8化します。

量子化は必ず高速化するとは限りません。CPUのINT8命令対応、量子化・逆量子化のoverhead、演算子対応によってはFP32より遅くなる場合もあります。最終判断はRaspberry Pi 5相当の実機測定で行います。

## 量子化条件

- 方式: Post-Training Static Quantization
- 形式: QDQ
- activation: signed INT8
- weight: signed INT8
- 対象演算: Conv
- calibration method標準: MinMax
- per-channel weight: 有効
- calibration画像: `train` splitのみ
- calibration画像のaugmentation: なし
- test / ood_test: calibrationへ使用しない

maskはcalibration計算に使いませんが、現在のDataset loaderは画像とmaskのpairを入力とするため、manifest上では両方必要です。

## 1. FP32 ONNXを準備する

[E006手順](onnx-edge.md)でFP32 ONNXを作ります。

```powershell
python -m fm_to_edge_seg export-onnx `
  configs\experiment\e001_deepcrack_baseline.yaml `
  artifacts\e001_deepcrack_baseline\best.pt `
  artifacts\e001_deepcrack_baseline\edge\student_fp32.onnx
```

## 2. 少数画像でINT8変換を確認する

最初は8枚で処理経路を確認します。

```powershell
python -m fm_to_edge_seg quantize-onnx `
  configs\experiment\e001_deepcrack_baseline.yaml `
  artifacts\e001_deepcrack_baseline\edge\student_fp32.onnx `
  artifacts\e001_deepcrack_baseline\edge\student_int8_debug.onnx `
  --split train `
  --calibration-samples 8 `
  --calibration-method minmax
```

変換時には次を自動確認・記録します。

- calibrationへ使ったsample ID
- manifest SHA-256
- FP32 / INT8 model SHA-256
- FP32 / INT8 model sizeと削減率
- 同じprobe画像に対するFP32 / INT8 logit差
- INT8 ONNXの構造検査
- CPUExecutionProviderでの実loadと推論

`probe_mean_error`は量子化によるlogit差の診断値であり、Dice低下量ではありません。合否は必ずtest全体で判断します。

## 3. 本calibrationを行う

DeepCrackではまず32枚を基準にします。

```powershell
python -m fm_to_edge_seg quantize-onnx `
  configs\experiment\e001_deepcrack_baseline.yaml `
  artifacts\e001_deepcrack_baseline\edge\student_fp32.onnx `
  artifacts\e001_deepcrack_baseline\edge\student_int8.onnx `
  --split train `
  --calibration-samples 32 `
  --calibration-method minmax
```

業務画像が32枚未満の場合は、存在するtrain画像だけが自動的に使用されます。ただし、糸色・糸質・機械オプションの偏りがある少数画像ではactivation範囲を代表できない可能性があります。枚数だけでなく条件の多様性を確認してください。

生成物:

```text
edge/
├── student_fp32.onnx
├── student_fp32.json
├── student_int8.onnx
└── student_int8.json
```

## 4. FP32とINT8の精度を同じtestで測る

FP32:

```powershell
python -m fm_to_edge_seg evaluate-onnx `
  configs\experiment\e001_deepcrack_baseline.yaml `
  artifacts\e001_deepcrack_baseline\edge\student_fp32.onnx `
  artifacts\e001_deepcrack_baseline\edge\fp32_test `
  --split test `
  --threads 1
```

INT8:

```powershell
python -m fm_to_edge_seg evaluate-onnx `
  configs\experiment\e001_deepcrack_baseline.yaml `
  artifacts\e001_deepcrack_baseline\edge\student_int8.onnx `
  artifacts\e001_deepcrack_baseline\edge\int8_test `
  --split test `
  --threads 1
```

実業務manifestに`ood_test`がある場合は、両方について`--split ood_test`も実行します。出力の`metrics.json`と`predictions.png`を比較します。

## 5. FP32とINT8の速度を同じ条件で測る

```powershell
python -m fm_to_edge_seg benchmark-onnx `
  artifacts\e001_deepcrack_baseline\edge\student_fp32.onnx `
  artifacts\e001_deepcrack_baseline\edge\benchmark_fp32_4threads.json `
  --threads 4 --warmup-runs 10 --runs 100

python -m fm_to_edge_seg benchmark-onnx `
  artifacts\e001_deepcrack_baseline\edge\student_int8.onnx `
  artifacts\e001_deepcrack_baseline\edge\benchmark_int8_4threads.json `
  --threads 4 --warmup-runs 10 --runs 100
```

同じPC・同じthread数・同じ電源設定で比較します。PCでINT8が速くても、Raspberry Piでも同じ比率になるとは限らないため、SBCで再測定します。

## 6. 最初の成功基準

PoCの暫定基準です。用途に応じて後から変更します。

- INT8 test Dice低下: FP32比で`0.01`以内
- INT8 ood_test Dice低下: FP32比で`0.01`以内
- model size: FP32より明確に小さい
- CPU mean / p95 latency: FP32以下、またはサイズ削減に見合う
- INT8 previewで細い領域の消失や背景誤検出が増えていない

25 FPSのモデル単体目安は平均40 ms以下です。最終アプリでは画像取得・前処理・後処理も含めて測定します。

## 7. 精度低下が大きい場合

一度に複数条件を変えず、次の順に比較します。

1. calibration画像を32枚から増やし、条件の多様性を改善
2. `--calibration-method percentile`を試す
3. `--calibration-method entropy`を試す
4. `--no-per-channel`を試し、CPU実装との相性を確認
5. 量子化に弱い演算・層の除外を次段階で追加
6. 必要ならQuantization-Aware Trainingを検討

calibration methodを変えたモデルは別ファイル名にして、同じ実験として上書きしません。

## 成功条件

- train画像だけでINT8 ONNXを生成できる
- FP32 / INT8を同じtest・ood_testで評価できる
- FP32 / INT8を同じCPU条件でbenchmarkできる
- 精度、サイズ、速度のtrade-offをJSONで説明できる
- Raspberry Piで採用する形式を実測から判断できる
