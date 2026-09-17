# E006 FP32 ONNX / CPU benchmark plan

## 問い

蒸留後も変わらない軽量Student本体を、精度を崩さずONNXへ変換し、Raspberry Pi 5相当CPUで実用的な速度にできるか。

## 比較対象

- E001 Baseline Student FP32
- E003 SAM 3 KD Student FP32
- E004b DINOv3 feature KD Student FP32

蒸留用headと教師モデルは推論時に含めません。同じStudent architectureなら、ONNXのparameter数と速度は原則同等になることを確認します。

## 固定する条件

- input: `1 x 3 x 384 x 544`
- ONNX opset 17
- ONNX Runtime CPUExecutionProvider
- warmup 10回、測定100回
- thread数1と4
- 同一SBC、電源、冷却条件

## 記録するもの

- checkpointとONNXの対応
- ONNX SHA-256
- PyTorch/ONNX最大・平均絶対誤差
- model sizeとparameter数
- mean / median / p90 / p95 latency
- FPS
- OS、CPU、Python、ONNX Runtime version

## 成功条件

- 最大絶対誤差`1e-4`以下
- FP32 ONNXがPCとSBCの両方で動く
- Raspberry Pi 5相当で再現可能な基準速度を取得する
- INT8量子化前の精度・速度・サイズ基準が確定する
