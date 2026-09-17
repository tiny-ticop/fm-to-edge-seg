# fm-to-edge-seg

Foundation Modelのセグメンテーション能力と汎用視覚特徴を、CPU向けの軽量binary segmentationモデルへ蒸留する研究PoCです。

個人開発ではDeepCrackを題材に、細長い領域のセグメンテーション、SAM 3による擬似ラベル、DINOv3によるlogit/feature distillationを検証します。業務環境ではコードだけを再利用し、社内画像・mask・重み・実験結果は個人環境と分離します。

## 現在の段階

DeepCrackのデータ準備とPyTorch Datasetを実装し、MobileNetV3-Small + Lite U-Netが少数画像へoverfitできるところまで確認済みです。

予定している最初の実験は次の通りです。

1. E000: DeepCrack 2画像による学習パイプラインのoverfit test（完了）
2. E001: DeepCrack + MobileNetV3-Small/Lite U-Net baseline（学習基盤完成）
3. E002: 共通teacher cache + soft-logit distillation（配線基盤）
4. E003: SAM系による少数ラベル・擬似ラベル実験
5. E004: DINOv3 feature distillation

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

## GitHubとVS Code

初回公開、別PCへのclone、VS CodeのSource Control操作は [docs/github-vscode-setup.md](docs/github-vscode-setup.md) にまとめています。

## ライセンス

このリポジトリ自体のライセンスはまだ選択していません。公開リポジトリにする前に決定します。外部データセットとFoundation Modelのライセンスは、それぞれ個別に確認してください。
