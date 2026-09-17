# E006: FP32 ONNX exportとCPU benchmark

## 目的

学習済みの軽量Studentを、PyTorch学習環境から切り離してCPU推論できるFP32 ONNXへ変換します。最初にFP32で次を確認します。

1. PyTorchとONNXの出力logitが一致する
2. ONNX Runtimeで最後まで推論できる
3. PCやRaspberry Piで同じ方法のCPU速度を測れる

このFP32結果を基準にして、次段階でINT8量子化による精度差・速度差・サイズ差を測ります。

## 重要な入出力仕様

ONNXモデルへ前処理は含めていません。

- input名: `image`
- input: `float32 NCHW`、ImageNet mean/stdで正規化済み
- 画像サイズ: experiment configの固定サイズ。DeepCrack標準は`544 x 384`
- batch size: 1固定（Edgeの1枚推論を優先）
- output名: `logits`
- output: `float32 N x 1 x H x W`
- binary mask: アプリ側で`sigmoid(logits) >= 0.5`

Edgeアプリでは、学習時と同じaspect ratio維持のletterboxとImageNet正規化を実装する必要があります。入力を単純なstretch resizeに変えると、PC評価と同じ精度にはなりません。

## 1. 依存関係を追加する

通常の学習環境へONNX関連だけ追加します。

```powershell
.venv\Scripts\Activate.ps1
pip install -e ".[edge]"
```

新しい業務PC環境をまとめて作る場合:

```powershell
pip install -e ".[train,dev,edge]"
```

## 2. checkpointをFP32 ONNXへ変換する

```powershell
python -m fm_to_edge_seg export-onnx `
  configs\experiment\e001_deepcrack_baseline.yaml `
  artifacts\e001_deepcrack_baseline\best.pt `
  artifacts\e001_deepcrack_baseline\edge\student_fp32.onnx
```

変換時には次を自動実行します。

- ONNX構造検査
- 同じ決定的入力をPyTorchとONNX Runtimeへ入力
- output shapeの確認
- logitの最大絶対誤差・平均絶対誤差の計算
- 最大絶対誤差が標準`1e-4`を超えた場合は失敗
- ONNX SHA-256、サイズ、parameter数、ライブラリversionの記録

生成物:

```text
edge/
├── student_fp32.onnx
└── student_fp32.json
```

## 3. 業務PCのCPUでbenchmarkする

GPU比較ではなくEdge CPUに近い実行条件を見るため、ONNX RuntimeのCPU providerを使います。最初は1 threadで測ります。

```powershell
python -m fm_to_edge_seg benchmark-onnx `
  artifacts\e001_deepcrack_baseline\edge\student_fp32.onnx `
  artifacts\e001_deepcrack_baseline\edge\benchmark_cpu_1thread.json `
  --threads 1 `
  --warmup-runs 10 `
  --runs 100
```

同じPCでthread数だけ変えて比較します。

```powershell
python -m fm_to_edge_seg benchmark-onnx `
  artifacts\e001_deepcrack_baseline\edge\student_fp32.onnx `
  artifacts\e001_deepcrack_baseline\edge\benchmark_cpu_4threads.json `
  --threads 4 `
  --warmup-runs 10 `
  --runs 100
```

記録される主な値:

- mean / median / p90 / p95 latency
- 理論上のモデル単体FPS
- thread数
- ONNX Runtime providerとversion
- CPU・OS情報
- model SHA-256、サイズ、parameter数

これは正規化済みtensorからlogitまでのモデル単体速度です。画像読込、letterbox、正規化、mask後処理は含みません。25 FPSのモデル単体目安は平均40 ms以下ですが、最終アプリでは前後処理を含むend-to-end時間を別途測ります。

## 4. Raspberry Piへ移す

会社・プロジェクトの許可された方法で、次だけをSBCへコピーします。

- `student_fp32.onnx`
- `student_fp32.json`
- このリポジトリのコード

SBC側で対応するONNX Runtimeを導入し、同じbenchmarkを実行します。

```bash
python -m pip install -e ".[edge]"

python -m fm_to_edge_seg benchmark-onnx \
  artifacts/e001_deepcrack_baseline/edge/student_fp32.onnx \
  artifacts/e001_deepcrack_baseline/edge/benchmark_rpi5_4threads.json \
  --threads 4 \
  --warmup-runs 10 \
  --runs 100
```

OS・Python・CPU architectureに対応する`onnxruntime` wheelが利用できない場合は、非公式wheelを無条件に使わず、ONNX Runtime公式の対応状況と会社の導入規程を確認します。

## 5. 比較上の注意

- 異なるモデルは同じSBC、電源設定、冷却、thread数で測る
- 初回実行をwarmupへ含め、測定値から除外する
- 発熱によるクロック低下があるため、長時間評価時は温度も記録する
- 同じSHA-256のモデルを比較したことを確認する
- FP32 ONNXの精度を通常の`evaluate`結果と確認してからINT8へ進む

## 成功条件

- `student_fp32.onnx`とmetadata JSONが生成される
- PyTorchとの最大絶対誤差が`1e-4`以下
- CPUExecutionProviderで100回の測定が完走する
- PCとRaspberry Piで同じ形式のbenchmark JSONを生成できる
- 次段階のINT8と比較するFP32基準値が残る
