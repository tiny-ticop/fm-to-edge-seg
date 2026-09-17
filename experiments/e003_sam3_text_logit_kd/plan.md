# E003 SAM 3 Text Teacher + Logit KD

## 状態

実装完了、SAM 3実推論は未実施。CUDA GPU、公式SAM 3環境、承認済みcheckpointが必要。

## 仮説

SAM 3のopen-vocabulary text promptで得た高confidence前景をsupervised lossへ追加すると、少数GTだけのE001より未知画像の亀裂/糸領域recallが改善する。

## 固定する条件

- E001と同じDeepCrack manifest・split・seed
- 同じMobileNetV3-Small + Lite U-Net
- 同じ入力解像度とaugmentation
- SAM 3 prompt: `crack`
- SAM 3 score threshold: 0.5
- KD confidence threshold: 0.5
- KD weight: 0.5
- temperature: 2.0

## 実行前ゲート

1. 3画像debug cacheを生成
2. teacher previewを目視
3. teacher単体指標を保存
4. 明らかな誤検出があればprompt/thresholdを別実験として調整
5. 条件確定後にtrain全件cacheを生成

## 比較

- E001: supervised baseline
- E002: cache/KD配線確認用reference teacher（性能比較対象外）
- E003: SAM 3 text teacher

最終判断はrandom validationだけでなく、未知groupから構成した`ood_test`で行う。
