# E000: DeepCrack tiny-batch overfit

## 目的

データ読み込み、Student、BCE＋Dice、逆伝播、checkpoint保存、推論モードまでの全経路が正常か確認する。

## 条件

- Dataset: DeepCrack trainの先頭2画像
- Input: 272×192、アスペクト比維持
- Student: MobileNetV3-Small + Lite U-Net
- Parameters: 1,003,281
- Initialization: random（pretrainedなし）
- Loss: BCE + Dice
- Optimizer: AdamW、weight decay 0
- Learning rate: 0.003
- Steps: 60
- Device: CPU
- BatchNorm: running statisticsを固定
- Seed: 42

## 結果

| 指標 | 初期 | 最終 |
|---|---:|---:|
| Loss | 1.6953 | 0.2654 |
| hard Dice | 0.0577 | 0.8065 |

約10秒で完了した。学習中だけでなく`model.eval()`後にもDice改善を維持したため、E000は成功と判断する。

## 発見

最初の試行では学習中Diceが0.77まで上がった一方、推論モードでは0.04へ戻った。2画像ではBatchNorm統計が不安定だったため、overfit検査ではrunning statisticsを固定した。これは正式学習でもbatch sizeが小さい場合に比較すべき設定とする。

## 次

全train/validationを使うE001 Baselineを実装し、validation Dice/IoU、checkpoint、予測可視化を記録する。
