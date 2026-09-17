# fm-to-edge-seg

Foundation Modelのセグメンテーション能力と汎用視覚特徴を、CPU向けの軽量binary segmentationモデルへ蒸留する研究PoCです。

個人開発ではDeepCrackを題材に、細長い領域のセグメンテーション、SAM 3による擬似ラベル・logit distillation、DINOv3によるfeature distillationを検証します。業務環境ではコードだけを再利用し、社内画像・mask・重み・実験結果は個人環境と分離します。

図を含む現在の実装状況とモデル小型化フローは [HTMLプロジェクトガイド](docs/project-status.html) で確認できます。

## 現在の段階

DeepCrackのデータ準備、軽量Baseline、SAM 3 logit KD、DINOv3 feature KD、撮像変化に対するRobustness評価まで実装済みです。実データ上の優劣は、各環境で同じsplitとseedを使って実験して判断します。

予定している最初の実験は次の通りです。

1. E000: DeepCrack 2画像による学習パイプラインのoverfit test（完了）
2. E001: DeepCrack + MobileNetV3-Small/Lite U-Net baseline（学習基盤完成）
3. E002: 共通teacher cache + soft-logit distillation（配線基盤）
4. E003: SAM系による少数ラベル・擬似ラベル実験
5. E004: DINOv3 feature distillation
6. E005: 合成撮像変化 + 実`ood_test`による汎化・Robustness評価
7. E006: FP32 ONNX export + CPU benchmark
8. E007: ONNX Runtime INT8 static PTQ + 精度・速度・サイズ比較

## 開発環境

- Python 3.10以上
- 個人PC: コード編集、テスト、CPU推論
- Google Colab: GPUを使う教師cache生成と学習
- Raspberry Pi 5相当: 最終的なCPU benchmark

### ローカルセットアップ

PowerShellの場合:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train,dev]"
python -m fm_to_edge_seg doctor
python -m unittest discover -s tests
pytest
```

仮想環境を有効化できない場合は、現在のPowerShellセッションだけ許可します。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

上記はGPU非搭載PC向けのCPU版PyTorchです。Colabではランタイムに用意されたPyTorchを利用します。

## データの置き場所

データセット、checkpoint、teacher feature、ONNXファイルはGit管理しません。詳しくは [data/README.md](data/README.md) を参照してください。

```text
data/
└── deepcrack/
    ├── images/
    ├── masks/
    └── manifest.csv
```

DeepCrackの取得・変換手順は [docs/deepcrack-setup.md](docs/deepcrack-setup.md) にあります。

Baselineのsmoke testとColab学習は [docs/training.md](docs/training.md) にあります。

RTX搭載の業務PCで画像準備から学習・評価まで行う手順は [docs/business-pc-workflow.md](docs/business-pc-workflow.md) にあります。

Teacher cacheとlogit distillationの考え方・E002の実行方法は [docs/distillation.md](docs/distillation.md) にあります。

SAM 3の導入、少数画像診断、teacher cache生成、E003実行は [docs/sam3-teacher.md](docs/sam3-teacher.md) にあります。

DINOv3 dense feature cache、PCA診断、E004a/E004b比較は [docs/dinov3-feature-distillation.md](docs/dinov3-feature-distillation.md) にあります。

SAM 3 / DINOv3の公式weight利用申請、認証、業務利用前の確認事項は [docs/foundation-model-access.md](docs/foundation-model-access.md) にあります。

明るさ・色味・ノイズ・ぼけと実`ood_test`による汎化評価は [docs/robustness-evaluation.md](docs/robustness-evaluation.md) にあります。

学習済みStudentのONNX変換、PyTorchとの出力一致検査、CPU benchmarkは [docs/onnx-edge.md](docs/onnx-edge.md) にあります。

INT8静的量子化、calibration、FP32との精度・速度・サイズ比較は [docs/int8-quantization.md](docs/int8-quantization.md) にあります。

## GitHubとVS Code

初回公開、別PCへのclone、VS CodeのSource Control操作は [docs/github-vscode-setup.md](docs/github-vscode-setup.md) にまとめています。

## ライセンス

このリポジトリ自体のライセンスはまだ選択していません。公開リポジトリにする前に決定します。外部データセットとFoundation Modelのライセンスは、それぞれ個別に確認してください。
