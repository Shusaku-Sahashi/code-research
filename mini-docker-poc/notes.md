# Notes - Mini Docker PoC

## 概要
Docker内部（namespaces/cgroups/レイヤー型FS）を理解するための最小コンテナランタイムを、Goで実装した。

## 構成方針

### 目的の確認
「Dockerクローンを作りたい」という要望の出発点は、Docker自体の内部構造を理解したいというもので、フル機能のDocker実装（レジストリ通信、Dockerfileビルド等）が目的ではないことを、実装前にAskUserQuestionで確認した。

- 実装言語: Go（本家Dockerと同じ言語）
- スコープ: namespaces + cgroups + レイヤーFSの概念が理解できる範囲まで
- 実際にこのサンドボックス上で動かして検証する

今回のスコープ外（次段階として着手予定）: veth+bridge+NATの実ネットワーク配線、レジストリからの実イメージ取得、Dockerfileビルド、user namespaceによるrootless化、cgroup v2対応、capability/seccomp。

### 命名規則
既存の`markdown-rag-poc/`に合わせて`mini-docker-poc/`ディレクトリ名を採用（`<name>-poc`パターン）。

## 実装メモ

### 2026-07-03

#### 環境確認
実装前にこのサンドボックス環境を確認した:
- root権限・フルcapability（`id` → uid=0, CapEff全ビット）
- `unshare --pid --mount --uts --net --fork --mount-proc` が動作することを確認
- cgroup v1 hybridで `/sys/fs/cgroup` に memory/pids/cpu/cpuacct/blkio/devices/freezer/unified がマウント済み
- OverlayFS利用可能（`/proc/filesystems` に `overlay` あり、nodev）
- `apt-get`利用可能、`busybox-static`パッケージ入手可能。`debootstrap`は未インストール
- `dockerd`/`docker` CLIは存在するがデーモン未起動 → 今回は使わず自前ランタイムをゼロから実装
- Go 1.24.7インストール済み。`golang.org/x/sys`最新版はGo 1.25以上を要求するため、Go 1.24と互換する`v0.29.0`に明示的にpinした（`go.mod`の`go 1.24`ディレクティブと合わせて、後から意図せずtoolchainが変わらないようにするため）

#### `/proc/self/exe` 再exec パターン
`run`が自分自身のバイナリを`/proc/self/exe`経由で`child`サブコマンドとして`SysProcAttr.Cloneflags`付きで再実行する。`unshare(2)`のように自分自身のnamespaceを切り替える方式だと、「自分がPID 1になる」効果は次に生成する子プロセスにしか現れないため、新しいプロセスとして新namespace群に入る必要がある。

`cgroup.procs`に書き込むPIDは、親プロセス（run）から見た`cmd.Process.Pid`（ホストのPID namespaceでのPID）を使う。子プロセス自身は自分をPID 1として認識するが、それは子の新PID namespace内での話であり、cgroup操作には無関係。

#### pivot_root の手順
1. `mount("", "/", "", MS_PRIVATE|MS_REC, "")` — mount伝播をホストから切り離す（このサンドボックスの`/`はすでにprivateだったが、`shared`なホストへの移植性のため明示的に実行）
2. `mount(rootfs, rootfs, "", MS_BIND|MS_REC, "")` — pivot_rootの前提条件（new_rootがマウントポイントである必要がある）を満たすための自己バインドマウント
3. `pivot_root(rootfs, rootfs/.old_root)`
4. `chdir("/")`
5. `unmount("/.old_root", MNT_DETACH)` — 旧ルートを完全にデタッチし、コンテナからホストのファイルシステムが一切見えないようにする

#### cgroup v1 パス
このサンドボックスがv1 hybridであることを確認したため、v1のファイル名（`memory.limit_in_bytes`, `pids.max`, `cpu.cfs_quota_us`/`cpu.cfs_period_us`）を使用。`cgroups.New()`内で各コントローラディレクトリの存在チェックを行い、v2専用ホストでは明確なエラーになるようにした（v2対応は今後の改善点）。

`memory.memsw.limit_in_bytes`（swap込みの制限）はこのサンドボックスに存在することを確認したが、存在しないカーネルもあるため`os.Stat`でガードしてから書き込む。

cgroupディレクトリの削除（`Cleanup()`）は`cmd.Wait()`完了後にのみ呼ぶ。プロセスが残っている状態で`rmdir`すると`EBUSY`になる。

#### busybox rootfs
`debootstrap`なしで、`apt-get install busybox-static` → busyboxバイナリをコピーし、sh/ls/ps/mount等のsymlinkを張るだけの最小rootfs（~1-2MB）を`scripts/build-rootfs.sh`で構築。レジストリ通信は不要。

#### OverlayFSレイヤーデモ
`lowerdir`は左側が優先度高（Docker用語でいう「後から追加されたレイヤー」が先頭に来る）。これを逆にすると期待と違う優先順位になるので注意。`layers-demo`は`run`とは独立したサブコマンドとして実装し、標準入力からEnter待ちで`merged`を手動で触ってから後始末する形にした。

## 検証記録

すべてこのサンドボックス上で実際に実行して確認（詳細な出力はREADME.mdのテスト結果セクションに記載）:

1. UTS/PID/Mount namespace分離 — hostname/ps aux/mount の出力でホストから隔離されていることを確認
2. メモリcgroup OOM Kill — `--mem 20m`超過でdmesgに`Memory cgroup out of memory: Killed process`ログを確認
3. pidscgroupフォークボム防止 — `--pids 20`上限で`fork: Resource temporarily unavailable`、ホストは無事なことを確認
4. cpu cgroup使用率制限 — `--cpus 0.5`で実際に`top`上50%に制限されることを確認
5. OverlayFSレイヤー重ね合わせ — 優先順位・layer限定ファイル・copy-upの3点を確認
6. `ps`コンテナ一覧 — 実行中コンテナが正しく表示されることを確認

### 検証中に起きたトラブルと対処
- フォークボムのテストで、bash風の関数定義構文`:(){ :|:& };:`はbusybox ashでは`bad function name`エラーになり動かなかった。busybox ash互換の`while true; do sh -c "sleep 5" & done`形式に書き換えて検証した。
- `layers-demo`の「Enterキー待ち」を`echo | ./mini-docker layers-demo`のようにパイプで済まそうとしたところ、`echo`が即座に改行を送ってしまい、検証コマンドを実行する前にアンマウント＆クリーンアップされてしまった。名前付きパイプ（`mkfifo`）を使い、ファイルディスクリプタを開いたまま任意のタイミングで改行を送る方式に変更して解決した。
- CPU quota検証中、バックグラウンドで起動した`mini-docker run`プロセス自体を`kill`しても、その子（busyboxの無限ループ）は孤児プロセスとして生き残ることに気づいた（Goの`exec.Command`はデフォルトでプロセスグループを作らないため、親を殺しても子は残る）。孤児プロセスと空のcgroupディレクトリ、状態ファイルを手動でクリーンアップした。テスト目的でプロセスをkillする際は、子プロセスも含めて後始末が必要になる点をnotesとして残す（`run`コマンド自体は正常な`cmd.Wait()`完了パスでは子プロセス終了後に自動でcleanupするため問題ない）。
