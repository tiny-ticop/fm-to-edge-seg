# E007 INT8 PTQ plan

## 問い

FP32 Studentのsegmentation精度と汎化性能をほぼ維持したまま、INT8でmodel sizeとRaspberry Pi CPU latencyを改善できるか。

## 比較対象

- FP32 ONNX
- INT8 QDQ / MinMax / per-channel
- 必要な場合だけPercentile、Entropy、per-tensorを追加

## 固定条件

- 同じStudent checkpoint
- 同じONNX opsetと入力解像度
- calibrationはtrain splitのみ
- test / ood_testはcalibrationへ使用しない
- benchmarkの機器、thread数、warmup、runsを固定
- mask threshold 0.5

## 比較表

| model | test Dice | OOD Dice | size MiB | mean ms | p95 ms | FPS |
|---|---:|---:|---:|---:|---:|---:|
| FP32 ONNX | | | | | | |
| INT8 MinMax | | | | | | |

## 暫定成功基準

- test / ood_test Dice低下が各0.01以内
- model sizeが明確に減少
- Raspberry Pi 5相当でmeanまたはp95 latencyが改善
- 細線や糸束境界の失敗例が明確に増えない

## 記録するもの

- Git commit、config、checkpoint SHA-256
- manifest SHA-256
- calibration sample IDとmethod
- FP32 / INT8 ONNX SHA-256
- 精度、サイズ、latency、実行環境
- prediction previewと失敗例
- 採用・不採用理由
