# E004: DINOv3 dense feature distillation

## 目的

SAM 3は対象領域のmaskを教師として与えます。一方、DINOv3は特定クラスのmaskではなく、画像の形状・質感・部品構造・領域間の類似性を表す汎用dense featureを生成します。E004ではDINOv3のpatch featureを、軽量Studentの中間特徴へ蒸留します。

```text
画像 ─ DINOv3 ViT-S/16 ─ C=384, stride=16 feature cache
                              ↓ cosine feature loss
画像 ─ MobileNetV3 Student ─ encoder_s16 ─ 1x1 projection
                              ↓
                       segmentation mask
```

1x1 projectionは学習時にStudentの48 channelを教師の384 channelへ写すためだけに使います。推論時には破棄するのでEdgeモデルのparameter数・速度は増えません。

## モデル選択

最初は`dinov3_vits16`を使用します。

- 21M parameterでDINOv3系列では小さい
- patch size 16でdense featureを取得できる
- RTX 3090 Tiクラスでcache生成を試しやすい
- 7B教師を使う前に、feature KD自体の効果を検証できる

Studentが小さいため、最も大きい教師が必ず最良とは限りません。教師の表現次元・計算量と、Studentが吸収できる容量は分けて考えます。

## 重要な前提

DINOv3の公式weightは利用申請が必要で、承認後にdownload URLが送られます。コードとweightには独自のDINOv3 Licenseが適用されるため、業務利用前に会社側で確認してください。

- [DINOv3公式リポジトリ](https://github.com/facebookresearch/dinov3)
- [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)

weight、業務画像、feature cacheはGitHubへ追加しません。

## 1. 専用環境とweightを準備する

SAM 3と同様に、通常環境と分けた`.venv-dinov3`を推奨します。

```powershell
py -3.12 -m venv .venv-dinov3
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv-dinov3\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

[PyTorch Start Locally](https://pytorch.org/get-started/locally/)からRTX PCに合うCUDA版PyTorchを導入し、本プロジェクトを入れます。

```powershell
pip install -e ".[train]"
git clone https://github.com/facebookresearch/dinov3.git external\dinov3
python -m fm_to_edge_seg doctor
```

公式サイトから利用申請し、承認後のURLで`dinov3_vits16` weightを社内で許可された場所へ保存します。URLやtokenはGitへ記録しません。

例:

```text
external/
├── dinov3/
│   └── hubconf.py
└── dinov3_weights/
    └── dinov3_vits16_pretrain_lvd1689m-xxxxxxxx.pth
```

## 2. 3枚でfeature cacheを診断する

DeepCrackのStudent入力は544×384で、どちらもpatch size 16の倍数です。

```powershell
python -m fm_to_edge_seg create-dinov3-feature-cache `
  data\deepcrack\manifest.csv `
  teacher_cache\dinov3_vits16_debug `
  --repository external\dinov3 `
  --weights external\dinov3_weights\dinov3_vits16_pretrain_lvd1689m-xxxxxxxx.pth `
  --model-name dinov3_vits16 `
  --width 544 `
  --height 384 `
  --split train `
  --max-samples 3
```

期待される1画像のfeature shapeは`384 × 24 × 34`です。float16圧縮前で約1.25 MB、float16換算で約0.63 MBです。

## 3. PCAで空間対応を確認する

```powershell
python -m fm_to_edge_seg preview-feature-cache `
  data\deepcrack\manifest.csv `
  teacher_cache\dinov3_vits16_debug `
  artifacts\dinov3_vits16_debug.png `
  --limit 3
```

左が入力画像、右が各画像内で3次元へPCA圧縮したfeatureです。色の意味や画像間の色を比較するものではありません。画像中の領域・輪郭変化とfeatureの変化位置が対応し、全面同色・ノイズ・位置ずれになっていないことを確認します。

## 4. train全件のcacheを生成する

debugとは別の出力先で`--max-samples`を外します。

```powershell
python -m fm_to_edge_seg create-dinov3-feature-cache `
  data\deepcrack\manifest.csv `
  teacher_cache\dinov3_vits16_544x384 `
  --repository external\dinov3 `
  --weights external\dinov3_weights\dinov3_vits16_pretrain_lvd1689m-xxxxxxxx.pth `
  --model-name dinov3_vits16 `
  --width 544 `
  --height 384 `
  --split train
```

cache metadataにはmodel名、weight SHA-256、DINOv3 repository commit、入力解像度、feature channel数、manifest SHA-256が保存されます。

## 5. 公平な比較実験を行う

feature cacheと90度回転は空間対応が崩れるため、E004では90度回転を無効化します。左右・上下反転は画像とcacheへ同時に適用できます。

最初に同じaugmentation条件のBaselineを実行します。

```powershell
python -m fm_to_edge_seg train configs\experiment\e004a_aligned_aug_baseline.yaml
```

次にfeature KDだけを追加します。

```powershell
python -m fm_to_edge_seg train configs\experiment\e004b_dinov3_feature_kd.yaml
```

比較の中心は`E004a vs E004b`です。E001は90度回転を使うため、E001との差にはaugmentation変更も含まれてしまいます。

## Feature Loss

Studentの`encoder_s16` featureを学習専用1x1 convolutionで384 channelへ射影し、教師とStudentをchannel方向にL2 normalizeしてcosine distanceを計算します。

```text
L_total = L_BCE + L_Dice + 0.25 × L_feature
```

letterbox paddingはLoss対象外です。教師featureとStudent featureの空間サイズが異なる場合は教師側をbilinear補間します。

## 何が確認できれば成功か

- 3画像cacheが`384×24×34`で生成される
- PCA previewに画像と対応した空間構造がある
- E004bの`train_kd`が有限で学習中に低下する
- projection headを含む学習が完走する
- Edge推論checkpointのStudent構造はE004aと同じ
- 未知groupの`ood_test`でE004bがE004aより改善する

validationだけ改善しOODが改善しない場合、DINOv3表現が対象の未知変化へ寄与していない、Student容量が不足している、または蒸留weight・接続層が不適切という仮説を検討します。
