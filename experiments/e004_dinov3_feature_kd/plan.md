# E004 DINOv3 Dense Feature Distillation

## 状態

実装完了、DINOv3実推論は未実施。公式weightの利用承認とCUDA GPUが必要。

## 仮説

DINOv3のdense featureをMobileNetV3のstride 16特徴へ蒸留すると、色・質感・背景部品が未知の条件でも、形状と領域構造を使ったsegmentationが安定する。

## 比較

- E004a: feature cacheと整合するaugmentationを使うsupervised baseline
- E004b: E004a + DINOv3 ViT-S/16 feature KD

両者でmodel、split、seed、input、supervised loss、augmentationを固定する。

## 実行前ゲート

1. DINOv3 Licenseとweight利用を確認
2. 3画像のdebug cacheを生成
3. shapeとPCA previewを確認
4. train全件cacheを生成
5. E004aを学習
6. E004bを学習
7. validationおよび未知groupの`ood_test`を比較

## 固定条件

- Teacher: DINOv3 ViT-S/16, LVD-1689M
- Feature: last normalized patch tokens
- Input: 544×384
- Student tap: `encoder_s16`, 48 channels
- Training-only projection: 48→384
- Feature loss: masked cosine distance
- Feature weight: 0.25
- 90-degree rotation: disabled
