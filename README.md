# fm-to-edge-seg

Foundation Modelのセグメンテーション能力と汎用視覚特徴を、CPU向けの軽量binary segmentationモデルへ蒸留する研究PoCです。

個人開発ではDeepCrackを題材に、細長い領域のセグメンテーション、SAM 3による擬似ラベル、DINOv3によるlogit/feature distillationを検証します。業務環境ではコードだけを再利用し、社内画像・mask・重み・実験結果は個人環境と分離します。

## 現在の段階

プロジェクト基盤を構築した段階です。学習コードやモデル実装はまだ含まれていません。

予定している最初の実験は次の通りです。

1. E000: synthetic dataによる学習パイプラインのsmoke test
2. E001: DeepCrack + MobileNetV3-Small/Lite U-Net baseline
3. E002: DINOv3 teacherからのsoft-logit distillation
4. E003: DINOv3 feature distillation
5. E004: SAM 3による少数ラベル・擬似ラベル実験

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
pip install -e ".[dev]"
python -m fm_to_edge_seg doctor
python -m unittest discover -s tests
pytest
```

仮想環境を有効化できない場合は、現在のPowerShellセッションだけ許可します。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## データの置き場所

データセット、checkpoint、teacher feature、ONNXファイルはGit管理しません。詳しくは [data/README.md](data/README.md) を参照してください。

```text
data/
└── deepcrack/
    ├── images/
    ├── masks/
    └── manifest.csv
```

## GitHubとVS Code

初回公開、別PCへのclone、VS CodeのSource Control操作は [docs/github-vscode-setup.md](docs/github-vscode-setup.md) にまとめています。

## ライセンス

このリポジトリ自体のライセンスはまだ選択していません。公開リポジトリにする前に決定します。外部データセットとFoundation Modelのライセンスは、それぞれ個別に確認してください。
