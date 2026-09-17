# E005: 汎化・Robustness評価

## 目的

通常のtest精度だけでは、明るさ、色味、ノイズ、ぼけなどが変わったときの劣化を確認できません。E005では同じtest画像へ決定的な合成変化を加え、同じcheckpointの性能低下を測定します。

この評価で確認することは「撮像変化に対する感度」です。未知の機械オプション、背景、糸種、形状を合成変化だけで再現することはできません。実際の未知条件は、manifestで撮影系列や機器・糸種を分離した`ood_test`を用意して評価します。

## 標準条件

| 条件 | 変化 |
|---|---|
| `clean` | 変化なし |
| `brightness_dark` | 明るさ0.6倍 |
| `brightness_bright` | 明るさ1.4倍 |
| `contrast_low` | コントラスト0.6倍 |
| `color_desaturated` | 彩度0.4倍 |
| `gaussian_noise` | 標準偏差0.08のGaussian noise |
| `gaussian_blur` | radius 1.5のGaussian blur |

ノイズは画像内容、条件名、実験seedから決定するため、同じ入力では毎回同じ結果になります。maskには変化を加えません。

## 1. 少数画像で動作確認する

E001の学習後、まず3枚だけ評価します。

```powershell
python -m fm_to_edge_seg evaluate-robustness `
  configs\experiment\e001_deepcrack_baseline.yaml `
  artifacts\e001_deepcrack_baseline\best.pt `
  artifacts\e001_deepcrack_baseline\robustness_debug `
  --split test `
  --device cuda `
  --num-workers 0 `
  --max-samples 3
```

## 2. test全体を評価する

```powershell
python -m fm_to_edge_seg evaluate-robustness `
  configs\experiment\e001_deepcrack_baseline.yaml `
  artifacts\e001_deepcrack_baseline\best.pt `
  artifacts\e001_deepcrack_baseline\robustness `
  --split test `
  --device cuda
```

特定条件だけを再実行する場合:

```powershell
python -m fm_to_edge_seg evaluate-robustness `
  configs\experiment\e001_deepcrack_baseline.yaml `
  artifacts\e001_deepcrack_baseline\best.pt `
  artifacts\e001_deepcrack_baseline\robustness_noise `
  --split test `
  --device cuda `
  --conditions clean gaussian_noise gaussian_blur
```

`clean`を省略しても、劣化量の基準として自動的に追加されます。

## 3. 出力を読む

```text
robustness/
├── robustness.json
├── robustness.csv
└── previews/
    ├── clean.png
    ├── brightness_dark.png
    └── ...
```

重要な値:

- `clean_dice`: 無変化testのDice
- `mean_corrupted_dice`: 合成変化条件の平均Dice
- `mean_dice_drop`: cleanからの平均低下量。小さいほど安定
- `worst_condition`: 最もDiceが低かった条件
- `dice_drop_from_clean`: 条件ごとの低下量

推論時間には画像読み込みと合成変化処理も含まれるため、Raspberry Piの純粋なモデル速度とは分けて扱います。

## 4. 蒸留手法を公平に比較する

E001、E003、E004bそれぞれの`best.pt`へ同じcommand、同じsplit、同じ条件を使います。比較表には最低限、次を残します。

| 実験 | clean Dice | mean corrupted Dice | mean Dice drop | worst condition | OOD Dice |
|---|---:|---:|---:|---|---:|
| E001 Baseline | | | | | |
| E003 SAM 3 KD | | | | | |
| E004b DINOv3 feature KD | | | | | |

蒸留モデルのclean Diceだけが改善し、`mean_dice_drop`や実`ood_test`が改善しない場合、「Foundation Modelの汎化性能がStudentへ移った」とは判断しません。

## 5. 業務画像での`ood_test`

metadataの`split`列で、学習に含めない機械オプション・糸種・撮影系列を`ood_test`にします。その後、通常評価とrobustness評価を両方実行します。

```powershell
python -m fm_to_edge_seg evaluate `
  configs\experiment\local_business_baseline.yaml `
  artifacts\business_baseline_v001\best.pt `
  artifacts\business_baseline_v001\ood_evaluation `
  --split ood_test `
  --device cuda

python -m fm_to_edge_seg evaluate-robustness `
  configs\experiment\local_business_baseline.yaml `
  artifacts\business_baseline_v001\best.pt `
  artifacts\business_baseline_v001\ood_robustness `
  --split ood_test `
  --device cuda
```

## 成功条件

- 全条件を同じcheckpointで最後まで評価できる
- `robustness.json`と`robustness.csv`が生成される
- previewで変化の強さと予測崩れを目視できる
- E001/E003/E004bを同じ条件で比較できる
- 合成変化の結果と、実際の`ood_test`結果を混同しない
