# E002: Teacher cacheとlogit distillation

## 今回作っているもの

Foundation Modelの推論は重く、Studentの各epochで毎回実行すると遅く、教師モデルの変更も実験へ混入します。そこで教師出力を画像ごとに一度だけ計算し、`teacher_cache/`へ固定する境界を作ります。

```text
SAM系 segmentation teacher ─┐
                             ├─ binary logits + confidence cache ─ Student logit KD
人手mask（今回の配線試験） ─┘

DINOv3 dense features ─ feature cache ─ Student feature KD（次段階）
```

DINOv3は汎用画像特徴を出すモデルで、直接的なbinary segmentation logit教師ではありません。そのため、以前の仮称`DINOv3 logit KD`は撤回しました。SAM系はmask/logit教師、DINOv3はfeature教師として役割を分けます。

## なぜ最初は正解maskから教師cacheを作るのか

今回はFoundation Modelの性能評価ではなく、以下の配線試験です。

- sample IDと教師出力が正しく対応する
- resize、letterbox、反転、回転後も教師と画像がずれない
- confidenceの低い画素をKD対象から除外できる
- supervised lossだけのE001と、KDを加えたE002をconfigだけで切り替えられる
- 教師モデルを後で交換してもStudent学習コードを変更しない

正解mask由来教師はStudentが既に持つ正解をsoft targetへ変えただけなので、精度向上の根拠にはしません。ここで改善しても「Foundation Modelから汎化性能を継承した」とは判断できません。

## cache形式

```text
teacher_cache/reference_mask_logits/
├── metadata.json
├── index.csv
└── samples/
    └── <sample-idのSHA-256>.npz
```

各`.npz`は元画像解像度の次の配列を持ちます。

- `logits`: binary teacher logit、`float16`
- `confidence`: 0～1の信頼度、`float16`

ファイル名にsample IDを直接使わないため、パス記号や日本語名に依存しません。学習時に入力解像度へbilinear resizeし、padding領域はKD対象外にします。

## DeepCrackでの実行

最初にtrain splitの参照cacheを作ります。

```powershell
python -m fm_to_edge_seg create-reference-teacher-cache `
  data/deepcrack/manifest.csv `
  teacher_cache/reference_mask_logits
```

少量batchで接続確認します。

```powershell
python -m fm_to_edge_seg train `
  configs/experiment/e002_reference_logit_kd.yaml `
  --device cpu `
  --epochs 1 `
  --num-workers 0 `
  --max-train-batches 2 `
  --max-validation-batches 1 `
  --no-pretrained
```

ログに`train_kd`が表示され、次の成果物が生成されれば配線成功です。

```text
artifacts/e002_reference_logit_kd/
├── best.pt
├── last.pt
├── history.jsonl
├── run.json
├── summary.json
└── validation_predictions.png
```

フル学習はE001 Baselineと同じseed・split・model・augmentationを使用し、蒸留設定だけを変えます。これにより差分をKDの有無へ限定できます。

## Loss

E002の総Lossは次の構成です。

```text
L_total = L_BCE + L_Dice + 0.5 × L_KD
```

`L_KD`は温度付きBernoulli KL divergenceです。教師とStudentの確率分布が同じなら0になります。`confidence_threshold`以上かつ有効画像領域の画素だけを計算対象にします。

## 業務データでの扱い

業務画像由来cacheもGit管理しません。将来SAM系adapterを追加した後は、次のように教師だけを置き換えて同じStudent学習を行います。

1. SAM系でmask候補とconfidenceを生成
2. 人が修正または採否を判断
3. 同じcache形式へ保存
4. distillation configの`cache_root`だけ変更
5. E001と同じsplitで比較

成功判定は学習Diceではなく、未知の機械オプション・糸種・撮影系列から構成した`ood_test`での改善です。
