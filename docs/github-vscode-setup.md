# GitHub・VS Codeセットアップ

## 1. 現在のローカルリポジトリをGitHubへ公開する

初回はprivate repositoryを推奨します。会社データを含めない場合でも、ライセンスと公開範囲を確認してからpublicへ変更できます。

### VS Codeから公開する方法

1. VS Codeでこのフォルダを開く。
2. 左側のSource Control、または`Ctrl+Shift+G`を開く。
3. GitHubアカウントへサインインする。
4. `Publish to GitHub`を選ぶ。
5. リポジトリ名を`fm-to-edge-seg`にする。
6. 最初は`Private repository`を選ぶ。
7. 公開対象ファイルを確認してpublishする。

VS CodeはGitHub repositoryの作成、`origin` remoteの追加、最初のpushをまとめて行います。

### GitHub Web UIとGitコマンドを使う方法

GitHubで空の`fm-to-edge-seg` repositoryを作ります。ローカル側に既にREADMEがあるため、GitHub側ではREADME、`.gitignore`、licenseを自動生成しません。

作成後に表示されるURLを使います。

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/fm-to-edge-seg.git
git push -u origin main
```

確認:

```powershell
git remote -v
git status
```

## 2. WindowsでのGitHub認証

HTTPSとGit Credential Managerの組み合わせを推奨します。Git for Windowsには通常Git Credential Managerが含まれ、初回push時にブラウザ認証が開きます。GitHubアカウントの通常パスワードをGitのpassword欄へ入力する方式は使いません。

VS Codeの右下またはAccountsメニューからGitHubへサインインしても構いません。基本的なclone、pull、pushには追加extensionは不要です。

## 3. 別PCへcloneする

### VS Code

1. `Ctrl+Shift+P`を押す。
2. `Git: Clone`を選ぶ。
3. GitHubから対象repositoryを選ぶか、HTTPS URLを入力する。
4. 保存先の親フォルダを選ぶ。
5. `Open`を選ぶ。
6. 自分のrepositoryであることを確認してWorkspace Trustを許可する。

### PowerShell

```powershell
git clone https://github.com/YOUR_ACCOUNT/fm-to-edge-seg.git
cd fm-to-edge-seg
code .
```

業務PCでは会社の規則に従い、GitHub接続とOSSコード利用の許可を確認します。会社画像用の設定は`configs/data/company_yarn.yaml`としてローカル作成し、Gitには追加しません。

## 4. 日常の変更確認とcommit

VS CodeのSource Controlで変更ファイルを選ぶとdiffを確認できます。

基本手順:

1. 変更内容をdiffで確認する。
2. commitへ含めるファイルの`+`を押してstageする。
3. commit messageを入力する。
4. `Commit`を押す。
5. `Sync Changes`または`Push`を押す。

同じ操作をPowerShellで行う場合:

```powershell
git status
git diff
git add README.md src tests
git diff --cached
git commit -m "feat: add project foundation"
git push
```

`git add .`の前には必ず`git status`を確認してください。画像、重み、token、業務用configが表示されていないことを確認します。

## 5. 個人PCと業務PCを同期する

作業開始時:

```powershell
git switch main
git pull --ff-only
```

作業終了時:

```powershell
git status
git add <確認済みのファイル>
git commit -m "変更内容"
git push
```

業務データに依存する変更を共有コードへ反映する場合も、画像パス、装置名、顧客名、評価画像、実測値などがcommitへ混入していないか確認します。

## 6. ブランチ運用

小さな実験でも、`main`を常に動く状態に保つためfeature branchを使用できます。

```powershell
git switch -c feat/deepcrack-loader
```

実装と確認後:

```powershell
git push -u origin feat/deepcrack-loader
```

GitHubでPull Requestを作り、diffとテスト結果を確認してから`main`へmergeします。一人開発でも実験単位の履歴が読みやすくなります。

## 7. Gitユーザー設定

まだ設定していない場合:

```powershell
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

個人PCと業務PCで異なるメールを使う場合は、repository内だけに設定できます。

```powershell
git config user.name "YOUR NAME"
git config user.email "YOUR_EMAIL_FOR_THIS_REPOSITORY"
```

