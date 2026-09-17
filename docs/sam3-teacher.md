# E003: SAM 3 text teacher

## 目的

SAM 3へ画像とテキストpromptを入力し、検出された複数の領域をbinary semantic maskへ統合してteacher cacheを作ります。Student学習時にはSAM 3を起動しないため、同じcacheでLossやStudent構造を何度でも比較できます。

DeepCrackでは`crack`、業務画像ではまず`thread`と`yarn`を別々に試します。promptは実験条件なので、途中で変更したcacheを同じ実験として扱いません。

## 重要な前提

2026年9月時点の公式SAM 3は848M parameterで、Python 3.12以上、PyTorch 2.7以上、CUDA 12.6以上のGPU環境を前提としています。checkpointはHugging Faceで利用申請と認証が必要です。また一般的なMIT LicenseではなくSAM Licenseなので、業務利用前に会社側で利用条件を確認してください。

SAM 3はEdgeへ載せるモデルではありません。GPU搭載PCで教師cacheを一度生成するためだけに使います。

公式情報:

- [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- [SAM 3 License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)

## 1. 専用環境を作る

SAM 3は依存関係が大きいため、通常のStudent開発用`.venv`と分けた`.venv-sam3`を推奨します。以下はPowerShellの構成例です。公式リポジトリの最新READMEに変更がある場合は公式手順を優先してください。

```powershell
py -3.12 -m venv .venv-sam3
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv-sam3\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

PyTorchは公式SAM 3 READMEまたは[PyTorch Start Locally](https://pytorch.org/get-started/locally/)に表示されるCUDA版を導入します。その後、SAM 3をGit管理対象外の`external/`へcloneします。

```powershell
git clone https://github.com/facebookresearch/sam3.git external\sam3
pip install -e external\sam3
pip install -e ".[train]"
```

公式SAM 3の依存関係がnative Windowsで導入できない場合は、同じGPU PC上のWSL2 Ubuntuを使います。Student学習とcache形式はOS非依存なので、教師生成だけWSL2で行う構成でも構いません。

## 2. checkpointへアクセスする

申請画面、Hugging Face認証、token管理、エラー時の確認方法は [SAM 3 / DINOv3 利用申請ガイド](foundation-model-access.md) にまとめています。先に同ガイドのSAM 3手順を完了してください。

1. 公式Hugging Faceページからブラウザでアクセス申請
2. 承認後、SAM 3専用環境で同じHugging Faceアカウントへ認証
3. `hf auth whoami`で使用中のアカウントを確認

```powershell
hf auth login
hf auth whoami
```

tokenをNotebook、config、Gitへ記載しないでください。デフォルトでは公式APIが認証情報を使ってcheckpointを取得します。会社側で承認されたローカルcheckpointを使う場合だけ`--checkpoint`を指定します。

## 3. 1～3枚で教師出力を診断する

最初から全画像を処理せず、少数画像でpromptと誤検出を確認します。

```powershell
python -m fm_to_edge_seg create-sam3-teacher-cache `
  data\deepcrack\manifest.csv `
  teacher_cache\sam3_text_crack_debug `
  --prompt crack `
  --split train `
  --score-threshold 0.5 `
  --max-samples 3
```

業務データではmanifestと出力先を業務PCのパスへ変更し、`--prompt thread`と`--prompt yarn`を別cacheとして比較します。

## 4. 教師maskを目視する

```powershell
python -m fm_to_edge_seg preview-teacher-cache `
  data\deepcrack\manifest.csv `
  teacher_cache\sam3_text_crack_debug `
  artifacts\sam3_text_crack_debug.png `
  --split train `
  --limit 3
```

previewは左から元画像、正解mask（赤）、SAM 3 teacher（緑）、confidenceです。確認項目:

- crack/thread全体を拾っているか
- 機械部品や背景模様を拾っていないか
- 細い領域が途中で消えていないか
- confidenceが高い誤検出がないか
- promptを変えたとき一貫して改善するか

この確認に失敗したcacheでStudentを学習してはいけません。Studentは教師の誤りも学習します。

正解maskがある画像では、prompt候補を数値でも比較します。

```powershell
python -m fm_to_edge_seg evaluate-teacher-cache `
  data\deepcrack\manifest.csv `
  teacher_cache\sam3_text_crack_debug `
  artifacts\sam3_text_crack_debug_evaluation `
  --split train `
  --confidence-threshold 0.5
```

`summary.json`に全体IoU・Dice・Precision・Recall・高confidence画素率、`per_sample.csv`に画像別結果が保存されます。`--max-samples`で作ったdebug cacheはcacheに存在する画像だけを評価し、`evaluated_samples/manifest_samples`も記録します。

## 5. train全件のcacheを生成する

debug cacheとは別の出力先にし、`--max-samples`を外します。

```powershell
python -m fm_to_edge_seg create-sam3-teacher-cache `
  data\deepcrack\manifest.csv `
  teacher_cache\sam3_text_crack `
  --prompt crack `
  --split train `
  --score-threshold 0.5
```

出力先が既にある場合は誤上書きを防いで停止します。意図的に再生成するときだけ`--overwrite`を使います。

## 6. E003を学習する

```powershell
python -m fm_to_edge_seg train configs\experiment\e003_sam3_text_logit_kd.yaml
```

E003では`confidence_threshold: 0.5`なので、標準設定ではconfidence 0.25の背景はKD対象外です。SAM 3が検出した高confidence前景だけを教師情報として加え、背景は人手GTによるBCE + Diceで学習します。

## mask統合とconfidenceの意味

SAM 3はtextに一致するinstance maskとinstance scoreを返します。本PoCでは:

1. 複数instance maskの和集合を糸/亀裂領域とする
2. 前景確率を0.99、背景確率を0.01としてpseudo-logitへ変換する
3. 前景confidenceを、その画素を覆うinstance scoreの最大値とする
4. 背景confidenceは弱い0.25とする

SAM 3のinstance scoreは厳密なpixel probabilityではありません。そのため「教師の生logit」とは呼ばず、hard maskから作ったpseudo-logitとして扱います。

## 成功条件

- SAM 3 cacheを最後まで生成できる
- previewで教師領域とconfidenceを説明できる
- 教師単体のDice・Recallと画像別の失敗例を記録できる
- E001 Baselineと同じsplit・seed・StudentでE003を学習できる
- randomなvalidationだけでなく、未知groupの`ood_test`でE001より改善する

E003で改善しない場合も重要な結果です。text promptが細線を捉えられない場合は、box/point prompt、SAM 2系のinteractive annotation、または人手修正済みpseudo-labelへ切り替えます。
