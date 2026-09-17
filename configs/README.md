# Configuration layout

実験コードを書き換えず、設定の組み合わせで比較するためのディレクトリです。

```text
configs/
├── data/           # データセットとmanifest
├── model/          # Student/Teacher構造
├── loss/           # supervised/distillation loss
├── distillation/   # SAM 3/DINOv3教師設定
└── experiment/     # 上記を束ねる実験設定
```

`local_*.yaml`と`company_*.yaml`はGitに含まれません。公開可能なtemplateから各環境で作成します。

`distillation`設定は教師モデルそのものではなく、事前生成したteacher cacheを参照します。詳細は [Teacher cacheとlogit distillation](../docs/distillation.md) を参照してください。
