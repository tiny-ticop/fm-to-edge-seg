# Experiments

Gitへ保存するのは、小さな再現情報と公開可能な結果だけです。

実験ごとに次を記録します。

- experiment ID
- Git commit
- config snapshot
- seed
- package versions
- dataset manifest hash
- metrics
- 実行時間
- 結論と次の仮説

dataset、checkpoint、teacher feature、業務画像由来の結果は保存しません。

E002では正解mask由来の参照teacherを使い、cacheとlogit KDの配線だけを検証します。この結果をFoundation Modelの改善として解釈しません。
