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

DeepCrackは配布元の利用条件に従い、データ本体をこのリポジトリへ再配布しません。業務画像、派生mask、teacher cache、checkpointも個人GitHubへ追加しません。

