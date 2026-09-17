# DeepCrackセットアップ

DeepCrackは細長い亀裂のbinary segmentation datasetです。本プロジェクトでは個人の非商用研究・教育目的に限って利用し、データ本体や学習済み重みをこのGitHubリポジトリへ再配布しません。

## 1. 公式リポジトリを取得

プロジェクトルートで実行します。`data/`以下はGit管理外です。

```powershell
git clone --depth 1 https://github.com/yhlleo/DeepCrack.git data/raw/DeepCrack-source
```

配布元: <https://github.com/yhlleo/DeepCrack>

公式リポジトリではデータセットがZIPで格納されています。展開します。

```powershell
Expand-Archive `
  -LiteralPath data/raw/DeepCrack-source/dataset/DeepCrack.zip `
  -DestinationPath data/raw/DeepCrack-extracted
```

## 2. 共通形式へ変換

公式リポジトリの`dataset`ディレクトリを入力します。

```powershell
python -m fm_to_edge_seg prepare-deepcrack `
  data/raw/DeepCrack-extracted `
  data/deepcrack `
  --val-fraction 0.2 `
  --seed 42
```

生成結果:

```text
data/deepcrack/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── masks/
│   ├── train/
│   ├── val/
│   └── test/
└── manifest.csv
```

公式trainからseed固定でvalidationを分離し、公式testはtestのまま保持します。maskは`0=背景、1=亀裂`のsingle-channel PNGへ変換します。

## 3. 検査

```powershell
python -m fm_to_edge_seg validate-manifest data/deepcrack/manifest.csv
```

`errors: 0`なら学習入力としてのファイル整合性は正常です。これはラベル内容の品質を保証するものではないため、次段で画像とmaskのoverlayも目視確認します。
