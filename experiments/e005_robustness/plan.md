# E005 Robustness evaluation plan

## 問い

Foundation Modelから蒸留したStudentは、Baseline Studentより撮像変化と未知条件に強いか。

## 比較対象

1. E001: Baseline Student
2. E003: SAM 3 logit KD Student
3. E004a: feature KDと同じaugmentationのBaseline
4. E004b: DINOv3 feature KD Student

E004a/E004bを主要比較とし、augmentation差をDINOv3の効果へ混入させません。

## 固定する条件

- dataset manifestとsplit
- 入力解像度
- Student architecture
- threshold 0.5
- robustness conditionと強度
- 評価seed
- 評価機器

## 実験順序

1. 各checkpointを通常の`test`または`ood_test`で評価する
2. `evaluate-robustness --max-samples 3`で配線を確認する
3. 全画像で標準conditionを評価する
4. `robustness.json`、`robustness.csv`、previewを保存する
5. clean Dice、mean corrupted Dice、mean Dice drop、実OOD Diceを比較する

## 成功判定

蒸留モデルについて、同条件Baselineに対して次の両方を確認します。

- 実`ood_test`のDiceが改善する
- `mean_dice_drop`が同等以下、または`mean_corrupted_dice`が改善する

clean Diceだけの改善は、汎化性能の継承成功とは判定しません。合成変化だけの改善も、未知背景・未知糸種への汎化を証明するものではありません。

## 記録するもの

- Git commit
- configとcheckpointの対応
- manifest SHA-256またはデータ版
- GPU/CPUとpackage version
- 通常評価の`metrics.json`
- `robustness.json`と`robustness.csv`
- 代表的な成功例・失敗例
- 結論と次に追加すべき実データ条件
