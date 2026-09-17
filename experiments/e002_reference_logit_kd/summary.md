# E002 Reference Logit KD — smoke test

## 目的

Foundation Model接続前に、共通teacher cacheからsoft logitを読み、StudentへKnowledge Distillationを適用する学習経路を検証する。

この教師はDeepCrackの正解maskから作った参照信号であり、Foundation Modelではない。精度比較や汎化性能の結論には使用しない。

## 条件

- Dataset: DeepCrack train 240 / val 60
- Student: MobileNetV3-Small + Lite U-Net
- Teacher cache: reference ground-truth mask logits
- Loss: BCE + Dice + 0.5 × Bernoulli KL
- Temperature: 2.0
- Device: CPU
- Encoder pretrained: disabled
- Epoch: 1
- Train: 2 batchのみ
- Validation: 1 batchのみ
- Seed: 42

## 結果

```text
train_loss=2.5985
train_dice=0.0442
train_kd=1.7038
val_loss=1.6498
val_dice=0.1074
val_iou=0.0567
```

## 判定

成功。teacher cache生成、sample対応、KD Loss、学習、validation、checkpoint・履歴・予測preview保存まで完走した。

数値は2 batchのsmoke testなので性能評価には使わない。次はSAM系adapterで同じcache形式を生成し、参照教師を実教師へ交換する。
