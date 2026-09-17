# 業務PCでの画像準備・学習・評価手順

## この手順で作るもの

業務画像と人手修正済みmaskから、軽量モデルを学習し、未知画像用のvalidation指標と予測画像を出します。画像、mask、checkpoint、実験結果はGitHubへ送信しません。

成功条件は次の4点です。

1. `doctor`でRTX GPUが表示される
2. 全画像・maskの検査がerrors 0になる
3. 学習後に`best.pt`が生成される
4. `metrics.json`と`predictions.png`で数値・見た目を確認できる

## 0. 情報管理上の確認

- 業務画像を個人PC、個人Google Drive、Colab、公開GitHubへ置かない
- 会社のGitHub・OSS・生成AI利用規程を先に確認する
- データはリポジトリ外（例: `D:\fm-to-edge-seg-data`）へ置く
- `data/`、`artifacts/`、`configs/data/local_*.yaml`は`.gitignore`対象だが、`git status`も毎回確認する

## 1. 初回セットアップ（Windows + VS Code）

PowerShellで作業用フォルダへ移動し、公開コードをcloneします。

```powershell
git clone https://github.com/tiny-ticop/fm-to-edge-seg.git
cd fm-to-edge-seg
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

PyTorch公式の [Start Locally](https://pytorch.org/get-started/locally/) で、Windows / Pip / Python / CUDAを選び、表示されたコマンドを実行します。CUDA Toolkitを個別に推測して入れるのではなく、PCのNVIDIA driverを更新したうえで公式selectorのstable版を使います。

続けて本プロジェクトを開発モードで入れます。

```powershell
pip install -e ".[train,dev]"
python -m fm_to_edge_seg doctor
code .
```

`doctor`の期待例は`cuda_available: True`、`gpu[0]: NVIDIA GeForce RTX 3090 Ti (...)`です。Falseなら学習を始めず、VS Code右下のPython interpreterが`.venv`か、NVIDIA driverとPyTorchがCUDA版かを確認します。

Notebookを使う場合はVS Code拡張機能「Python」と「Jupyter」を入れ、`notebooks/local_gpu_workflow.ipynb`を開いてkernelに`.venv`を選びます。CLIとNotebookは同じ処理を呼ぶため、どちらで実行しても結果形式は同じです。

## 2. アノテーション画像を準備する

リポジトリ外に次の構造を作ります。サブフォルダも使用できますが、画像とmaskの相対パスとファイル名を一致させます。拡張子は異なっても構いません。

```text
D:\fm-to-edge-seg-data\source\
├── images\
│   ├── run_001\frame_0001.jpg
│   └── run_002\frame_0002.jpg
├── masks\
│   ├── run_001\frame_0001.png
│   └── run_002\frame_0002.png
└── metadata.csv                 # 任意
```

maskの要件:

- 元画像と同じ幅・高さ
- 1 channelのPNGを推奨
- 背景は0、糸は255（または1）
- 一本の糸は見えている幅、密集した束は合意した外形内部を塗る
- 半透明、ぼかし、JPEG maskは使わない

変換コマンドは0以外をすべて糸として`0/1`へ正規化します。そのため、アノテーションツールのラベル番号やignore領域がある場合は、そのまま実行せず変換規則を追加してください。

### 少数枚で重要なmetadata.csv

連続フレームをランダム分割すると、ほぼ同じ画像が学習と評価の両方に入り、過大評価になります。同じ撮影系列、機械オプション、糸種を`group_id`でまとめます。

```csv
sample_id,group_id,machine_option,yarn_family,cycle_angle_deg
run_001__frame_0001,run_001,option_a,yarn_red,90
run_002__frame_0002,run_002,option_b,yarn_white,180
```

サブフォルダ区切りは`__`になります。より厳密に未知条件を決める場合は`split`列を追加し、全行へ`train`、`val`、`test`または`ood_test`を記入します。最低でもtrainとvalが各1枚必要です。自動分割時はtrain/valを作れる独立したgroupが2つ以上必要です。

## 3. 学習形式へ変換して検査する

```powershell
python -m fm_to_edge_seg prepare-binary-dataset `
  D:\fm-to-edge-seg-data\source `
  D:\fm-to-edge-seg-data\prepared `
  --metadata D:\fm-to-edge-seg-data\source\metadata.csv `
  --val-fraction 0.2

python -m fm_to_edge_seg validate-manifest `
  D:\fm-to-edge-seg-data\prepared\manifest.csv

python -m fm_to_edge_seg preview-dataset `
  D:\fm-to-edge-seg-data\prepared\manifest.csv `
  artifacts\business_data_preview.png `
  --split train --limit 12
```

metadataをまだ作らない場合は`--metadata ...`の行を削除できます。変換先を作り直す場合だけ`--overwrite`を追加します。検査は画像欠落、サイズ不一致、mask値異常を検出します。previewでは、赤いmaskが糸に合い、画像とずれていないことを人が確認します。

## 4. 業務PC専用configを作る

サンプルをコピーします。`local_`で始まるファイルはGit管理されません。

```powershell
Copy-Item configs\data\business_example.yaml.example configs\data\local_business.yaml
Copy-Item configs\experiment\business_baseline.yaml.example configs\experiment\local_business_baseline.yaml
```

`configs/data/local_business.yaml`の`manifest`を実際の絶対パスへ変更します。WindowsでもYAML内は`D:/...`のようなforward slashが安全です。

100枚未満の初期実験では、サンプル設定の`batch_size: 4`、`freeze_batch_norm: true`、pretrained encoder、augmentation、early stoppingを推奨します。10～20枚でも実行できますが、性能の結論ではなく「次にどの条件を追加収集するか」を判断する予備実験と扱います。

## 5. まずsmoke testを行う

```powershell
python -m fm_to_edge_seg train `
  configs\experiment\local_business_baseline.yaml `
  --epochs 1 `
  --num-workers 0 `
  --max-train-batches 2 `
  --max-validation-batches 1
```

これは配線確認であり、精度評価ではありません。`runtime: device=cuda`と表示され、`artifacts/business_baseline_v001/`にファイルが生成されれば成功です。

## 6. 本学習を行う

smoke testと同じ出力先を残したい場合は、configの`experiment_id`と`output_dir`を`business_baseline_v002`などへ変更してから実行します。

```powershell
python -m fm_to_edge_seg train configs\experiment\local_business_baseline.yaml
```

ImageNet pretrained weightは初回だけ取得されます。業務PCがインターネット非接続の場合は、会社の規程に沿ってtorchvisionの重みを事前搬入する必要があります。

## 7. 独立評価と結果確認

testを作った場合:

```powershell
python -m fm_to_edge_seg evaluate `
  configs\experiment\local_business_baseline.yaml `
  artifacts\business_baseline_v001\best.pt `
  artifacts\business_baseline_v001\test_evaluation `
  --split test
```

testがまだない場合は`--split val`にします。出力は次の通りです。

- `metrics.json`: IoU、Dice、Precision、Recall、処理時間、parameter数、checkpointサイズ
- `predictions.png`: 元画像、正解（赤）、予測（緑）の比較

処理時間には画像読み込みと前処理も含まれるため、最終的なRaspberry Pi速度とは分けて扱います。現段階の成功は、Diceだけでなく、別groupの細い糸が途切れていないこと、機械部品を糸と誤検出していないことです。

同じcheckpointへ明るさ・色味・ノイズ・ぼけを加えたときの劣化と、実際の`ood_test`を評価する方法は [E005: 汎化・Robustness評価](robustness-evaluation.md) を参照してください。

## 8. 実験を残す

コードと汎用configだけをcommitし、業務データはcommitしません。

```powershell
git status
git diff
```

業務実験は社内で許可された場所に、使用commit、config、データ版、group分割、`run.json`、`summary.json`、`metrics.json`をセットで保存します。画像枚数ではなく、追加した機械オプション・糸色・糸質・角度を記録すると次の実験を比較できます。

## よくある問題

- CUDAがFalse: CPU版torchを入れている、別Pythonを使っている、driverが古い
- pairing failed: `images`と`masks`で相対パスまたは拡張子を除く名前が違う
- size mismatch: mask出力時にアノテーションツールが縮小している
- validationが極端に良い: 連続フレームが別splitに漏れている可能性がある
- validationが不安定: 枚数が少なすぎるため、group単位の交差検証を次段階で使う
- CUDA out of memory: `batch_size`を4→2→1の順で下げる
