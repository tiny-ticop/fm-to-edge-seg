# SAM 3 / DINOv3 利用申請ガイド

最終確認日: **2026-09-18 (JST)**

この文書は、`fm-to-edge-seg`で教師モデルとして使うSAM 3とDINOv3について、公式checkpointへアクセスするまでの手順をまとめたものです。画面や申請項目は提供元によって変更される可能性があるため、リンク先に表示される最新の条件を最優先してください。

## 先に結論

| モデル | このプロジェクトで使うもの | 推奨する申請経路 | 承認後の取得方法 |
|---|---|---|---|
| SAM 3 | `facebook/sam3`、`sam3.pt` | Hugging Faceの[SAM 3公式ページ](https://huggingface.co/facebook/sam3) | 同じHugging Faceアカウントで認証し、自動取得または`hf download` |
| DINOv3 | `dinov3_vits16`、LVD-1689MのViT-S/16 weight | Metaの[DINOv3公式申請フォーム](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) | 承認メールに記載されたURLからローカルへ保存 |

DINOv3は、[ViT-S/16のHugging Face公式ページ](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)からモデル単位で申請することもできます。ただし、現在の本プロジェクトは公式DINOv3リポジトリとPyTorch形式のローカルweightを読む構成なので、最初はMeta公式フォーム経由が分かりやすいです。

## 申請前に確認すること

1. 個人検証なら個人アカウント、業務検証なら会社の規程で許可されたアカウントとメールアドレスを使います。
2. [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)と[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)を読みます。アクセス承認と、会社での利用・製品利用の承認は別です。
3. 業務利用では、申請者に会社を代表して条件へ同意する権限があるかを社内で確認します。不明なら上長、情報システム、法務・知財担当へ確認します。
4. Hugging Faceのゲート申請では、ユーザー名とメールアドレスがモデル提供者へ共有されます。これは公開プロフィールへの掲載とは別ですが、共有自体を避けて申請することはできません。
5. token、承認メール固有のdownload URL、weightをGitHubへ登録しません。本リポジトリでは`.env`、`external/`、`*.pt`、`*.pth`、`*.safetensors`をGit対象外にしています。

申請内容は事実に合わせて記入してください。利用目的を英語で求められた場合の記載例です。

```text
I am evaluating the model as an offline teacher for knowledge distillation
into a lightweight binary segmentation model for thin thread-like regions.
The teacher model will run only on an authorized GPU workstation, while the
distilled student is intended for CPU edge inference. I will comply with the
model license and my organization's policies.
```

個人検証の場合は`my organization's policies`を`the applicable license terms`へ変え、業務を装わないようにします。

## SAM 3の申請手順

### 1. Hugging Faceへログインする

[Hugging Face](https://huggingface.co/)でアカウントを作成し、ブラウザでログインします。後でPCから認証するときも、申請したものと同じアカウントを使います。

### 2. ブラウザからアクセスを申請する

1. [facebook/sam3](https://huggingface.co/facebook/sam3)を開きます。
2. ページ上部のアクセス条件とSAM Licenseを読みます。
3. 表示された`Agree and send request to access repo`相当のボタンを押します。
4. 追加項目が表示された場合は、氏名、所属、利用目的などを正確に記入します。
5. 送信後、ページ上の状態が承認済みになるまで待ちます。

Hugging Faceの仕様上、ゲートモデルの申請はブラウザからのみ行えます。承認が自動か手動かは提供者が決めるため、所要時間は保証できません。

### 3. 承認後にPCを認証する

SAM 3専用仮想環境を有効にして、Hugging Face CLIを用意します。

```powershell
.venv-sam3\Scripts\Activate.ps1
python -m pip install --upgrade huggingface_hub
hf auth login
hf auth whoami
```

現在の`hf auth login`は通常、ブラウザ認証用URLと短いコードを表示します。業務PCからブラウザ認証できない場合は、[Hugging FaceのAccess Tokens画面](https://huggingface.co/settings/tokens)でtokenを作成し、対話画面の`Paste an access token`を選べます。

tokenを作る場合は次の方針にします。

- 開発PC、Colab、業務PCごとに別tokenを作る
- 業務用途では、可能なら`fine-grained` tokenで`facebook/sam3`の読み取りだけを許可する
- token文字列をPowerShellのコマンド履歴、Notebook、`.env.example`、設定YAML、GitHub Issueへ貼らない
- 漏えいした疑いがあればAccess Tokens画面で直ちに削除または更新する

### 4. 実際にcheckpointへアクセスできるか確認する

次のコマンドは認証だけでなく、`sam3.pt`をHugging Faceのcacheへ実際にダウンロードします。ファイルが大きいため、空き容量と社内ネットワーク規程を確認してから実行してください。

```powershell
hf download facebook/sam3 sam3.pt
```

このプロジェクトでは、認証済みであれば次のteacher cache生成時にもSAM 3側がcheckpointを自動取得します。

```powershell
python -m fm_to_edge_seg create-sam3-teacher-cache `
  data\deepcrack\manifest.csv `
  teacher_cache\sam3_text_crack_debug `
  --prompt crack `
  --split train `
  --score-threshold 0.5 `
  --max-samples 3
```

SAM 3.1は別のモデルリポジトリ`facebook/sam3.1`です。SAM 3の承認が別リポジトリへ自動的に引き継がれるとは仮定せず、利用するときにそのページの条件とアクセス状態を確認してください。現時点の本PoCは画像教師としてSAM 3を使います。

## DINOv3の申請手順

### 方法A: Meta公式フォーム（本プロジェクトで推奨）

1. DINOv3公式READMEの`Pretrained models`を確認します。
2. [DINOv3 access request form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)を開きます。
3. 画面に表示されるDINOv3 Licenseとプライバシー条件を読みます。
4. 氏名、メールアドレス、所属、利用目的など、現在の画面で必須になっている項目を正確に入力します。
5. 利用するモデルとして、まずLVD-1689Mの`ViT-S/16 distilled`を選べる場合は選択します。フォームが一括申請方式なら、そのまま送信します。
6. 受信確認メールや承認メールを確認します。公式READMEによれば、承認後のメールにbackboneとadapterを含む利用可能なweight URL一覧が届きます。

承認メールのURLは公開せず、会社で許可された保存場所だけで扱います。公式READMEはブラウザではなく`wget`でのダウンロードを案内しています。Windowsで`wget`が使えない場合はWSLから実行するか、会社で許可されたダウンロード手段を使います。

```bash
mkdir -p external/dinov3_weights
wget -O external/dinov3_weights/dinov3_vits16_pretrain_lvd1689m.pth \
  "承認メールに記載されたViT-S/16のURL"
```

PowerShellで会社の規程上`curl.exe`が許可されている場合の例です。

```powershell
New-Item -ItemType Directory -Force external\dinov3_weights
curl.exe -L "承認メールに記載されたViT-S/16のURL" `
  -o external\dinov3_weights\dinov3_vits16_pretrain_lvd1689m.pth
Get-FileHash -Algorithm SHA256 `
  external\dinov3_weights\dinov3_vits16_pretrain_lvd1689m.pth
```

### 方法B: Hugging Faceからモデル単位で申請する

DINOv3の公式READMEにはHugging Face Transformers経由も掲載されています。ViT-S/16だけを使う場合は、[facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)へログインし、SAM 3と同様にブラウザ上で条件へ同意してアクセスを申請できます。

承認後は、同じアカウントで`hf auth login`を実行すればTransformersから取得できます。ただし、Hugging Face版のファイル形式と公式リポジトリの`torch.hub.load(..., weights=...)`へ渡す`.pth`は同一とは限りません。本PoCの現行CLIをそのまま使う場合は方法Aを選びます。方法Bへ切り替える場合は、DINOv3 adapterをTransformers対応へ変更してから実験条件として記録します。

### DINOv3の取得後に3枚だけ確認する

```powershell
git clone https://github.com/facebookresearch/dinov3.git external\dinov3

python -m fm_to_edge_seg create-dinov3-feature-cache `
  data\deepcrack\manifest.csv `
  teacher_cache\dinov3_vits16_debug `
  --repository external\dinov3 `
  --weights external\dinov3_weights\dinov3_vits16_pretrain_lvd1689m.pth `
  --model-name dinov3_vits16 `
  --width 544 `
  --height 384 `
  --split train `
  --max-samples 3
```

この処理に成功すると、weightのSHA-256、DINOv3リポジトリのcommit、入力解像度などがcache metadataへ記録されます。

## エラー時の切り分け

### SAM 3で401または認証エラー

```powershell
hf auth whoami
```

- 申請したHugging Faceアカウントと、CLIでログインしたアカウントが同じか確認する
- モデルページが承認済み表示か確認する
- token方式なら、tokenが失効していないか、対象モデルのread権限があるか確認する
- 別アカウントの認証が残っている場合は`hf auth login --force`で切り替える

### SAM 3で403

アクセスが未承認、tokenの権限不足、または組織側のtoken policyが原因になり得ます。モデルページとAccess Tokens画面を確認します。申請が保留・拒否された場合、承認可否はモデル提供者が管理しており、このプロジェクト側では解除できません。

### DINOv3のURLで403またはダウンロード失敗

- URLを途中で改行・省略していないか確認する
- ブラウザではなく公式README推奨の`wget`で再試行する
- 承認メールに更新されたURLや再取得方法が記載されていないか確認する
- 解決しなければMeta公式フォーム・公式案内へ戻り、非公式mirrorからweightを取得しない

### 会社ネットワークで取得できない

proxy、SSL inspection、外部ストレージ制限が関係する場合があります。認証回避や証明書検証の無効化は行わず、URL、必要容量、保存先、利用ライセンスを情報システム担当へ伝えて許可された方法で取得します。

## 業務利用前の記録チェックリスト

- [ ] 申請日、申請者、使用アカウントを社内記録へ残した
- [ ] 承認日と対象モデル名を記録した
- [ ] その時点のSAM License / DINOv3 Licenseを確認した
- [ ] weightの取得元URL種別、ファイル名、SHA-256を記録した
- [ ] モデルweightとtokenをGit管理対象外にした
- [ ] 業務画像、teacher cache、学習済みcheckpointを個人環境へ持ち出さない
- [ ] 蒸留済みStudentの配布・製品利用についても会社側で条件を確認した

## 参照した一次情報

- [SAM 3公式リポジトリ](https://github.com/facebookresearch/sam3)
- [SAM 3公式Hugging Face](https://huggingface.co/facebook/sam3)
- [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- [DINOv3公式リポジトリ](https://github.com/facebookresearch/dinov3)
- [DINOv3公式申請フォーム](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
- [DINOv3 ViT-S/16公式Hugging Face](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
- [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- [Hugging Face: Gated models](https://huggingface.co/docs/hub/models-gated)
- [Hugging Face: User access tokens](https://huggingface.co/docs/hub/security-tokens)
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
