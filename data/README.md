# Data directory

このディレクトリにはデータセット本体を置けますが、`README.md`以外はGitで無視されます。

共通の推奨構造:

```text
data/<dataset-name>/
├── images/
├── masks/
└── manifest.csv
```

maskは画像と同じ縦横サイズの8-bit single-channel PNGを正規形式とします。

- `0`: background
- `1`: foreground
- `255`: ignore

manifestの必須列:

```csv
sample_id,image_path,mask_path,split
sample_0001,images/sample_0001.jpg,masks/sample_0001.png,train
```

検査コマンド:

```powershell
python -m fm_to_edge_seg validate-manifest data/deepcrack/manifest.csv
```

DeepCrackは配布元の利用条件に従い、データ本体をこのリポジトリへ再配布しません。業務画像、派生mask、teacher cache、checkpointも個人GitHubへ追加しません。
