# Mini Docker PoC - レポート

## 概要

Dockerの内部構造（namespaces・cgroups・レイヤー型ファイルシステム）を、実際に手を動かして理解するための最小コンテナランタイムを実装した。

**スコープ（第1段階）**:
- Linux namespaces（UTS/PID/Mount/Network/IPC）によるプロセス分離
- cgroup v1 によるリソース制限（メモリ・CPU・プロセス数）
- OverlayFSによるDockerイメージのレイヤー重ね合わせ概念の体験

**意図的にスコープ外とした部分（次段階）**:
- veth+bridge+NATによる実ネットワーク配線（今回は namespace 分離のみ）
- レジストリからの実イメージ取得・Dockerfileビルド
- user namespaceによるrootless化、cgroup v2対応、capability/seccomp制御

「Dockerクローンを作る」ことそのものではなく、**Dockerが内部で何をしているかを体感できる教材を作ること**を目的とした。

---

## ファイル構成

```
mini-docker-poc/
├── README.md                       # このレポート
├── notes.md                        # 実装メモ
├── go.mod / go.sum                 # Goモジュール定義
├── cmd/mini-docker/main.go         # サブコマンドディスパッチのみ
├── internal/
│   ├── cli/
│   │   ├── run.go                  # `run` サブコマンド（親プロセス側：cgroup作成＋再exec）
│   │   ├── child.go                # `child` サブコマンド（新namespace内：pivot_root＋exec）
│   │   ├── ps.go                   # `ps` サブコマンド（実行中コンテナ一覧）
│   │   └── layers.go               # `layers-demo` サブコマンド
│   ├── container/                  # コンテナID生成・状態ファイル管理
│   ├── rootfs/pivot.go             # pivot_root によるルートファイルシステム切り替え
│   ├── cgroups/                    # cgroup v1 作成・リソース制限・後始末
│   └── overlay/demo.go             # OverlayFSレイヤーデモ
└── scripts/build-rootfs.sh         # busybox-static を使った最小rootfs構築スクリプト
```

---

## アーキテクチャ

### A. コンテナ起動フロー（`run` → `child` の再exec）

```
[ mini-docker run --rootfs ./rootfs --mem 100m -- /bin/sh ]
        ↓
  cgroup作成（memory/pids/cpu 各コントローラ配下に mini-docker/<id>/ を作成）
        ↓
  exec.Command("/proc/self/exe", "child", ...)
    SysProcAttr.Cloneflags = CLONE_NEWUTS|CLONE_NEWPID|CLONE_NEWNS|CLONE_NEWNET|CLONE_NEWIPC
        ↓                                          ← ここで新しいnamespace群に入った子プロセスが生まれる
  親プロセス: cmd.Process.Pid（ホストPID namespaceから見たPID）を
              cgroup.procs に書き込み、cmd.Wait() で終了を待つ
        ↓
  子プロセス（"child"サブコマンド、新namespace内。自分から見るとPID 1）
    1. sethostname()                       … UTS namespace
    2. pivot_root(rootfs, rootfs/.old_root) … Mount namespace
    3. mount("proc", "/proc", "proc")       … 新PID namespaceに対応したprocfs
    4. unix.Exec(コマンド)                   … プロセスイメージを完全に置き換えてPID 1として実行
```

なぜ2段階のプロセスに分かれているか: `unshare`のように自分自身のnamespaceを切り替える方式だと、「自分がPID 1になる」効果は次に生成する子プロセスにしか現れない。そのため実際のコンテナランタイム（runc等）と同様、`/proc/self/exe` を `Cloneflags` 付きで再実行し、**新しいプロセスとして**新namespace群に入る。

### B. cgroup v1 階層

```
/sys/fs/cgroup/
├── memory/mini-docker/<id>/
│   ├── memory.limit_in_bytes         ← --mem
│   ├── memory.memsw.limit_in_bytes   ← --mem（swap込み、存在する場合のみ）
│   └── cgroup.procs                  ← コンテナのホストPIDを書き込み
├── pids/mini-docker/<id>/
│   ├── pids.max                      ← --pids
│   └── cgroup.procs
└── cpu/mini-docker/<id>/
    ├── cpu.cfs_period_us             ← 固定 100000 (100ms)
    ├── cpu.cfs_quota_us              ← --cpus × 100000
    └── cgroup.procs
```

### C. OverlayFS レイヤースタック（`layers-demo`）

```
merged/  （コンテナから見える統合ビュー）
  ↑ 重ね合わせ（左が優先）
lowerdir = layer2-lower : layer1-lower     ← 読み取り専用のイメージレイヤー
upperdir = upper                            ← 書き込み可能なコンテナ層
workdir  = work                             ← OverlayFS内部管理用

例:
  layer1-lower/hello.txt  = "from layer1 (base image)"
  layer1-lower/shared.txt = "layer1 version"
  layer2-lower/shared.txt = "layer2 version (overwrites layer1)"   ← layer2が勝つ
  layer2-lower/app.txt    = "from layer2 (app layer)"

  merged/ への書き込み → upper/ へ copy-up される（下位レイヤーは変更されない）
```

これはDockerのoverlay2グラフドライバが、複数の読み取り専用イメージレイヤー＋1つの書き込み可能なコンテナ層を1つのファイルシステムビューに合成する仕組みそのもの。

---

## 技術選定

| 項目 | 選択 | 理由 |
|------|------|------|
| 実装言語 | Go | 本家Docker/runcと同じ言語。syscallパッケージ・`golang.org/x/sys/unix` で直接カーネルAPIを扱える |
| namespace切替 | `os/exec` の `SysProcAttr.Cloneflags` | `unshare`外部コマンドを呼ぶより、余分なfork/execを挟まずrunc等の実装に近い形で書ける |
| cgroupバージョン | v1（hybrid） | このサンドボックス環境がcgroup v1 hybridでマウントされているため。v2専用ホストでは`memory.max`等ファイル名が異なる（今後の改善点） |
| rootfs構築 | busybox-static + symlink | レジストリ通信や`debootstrap`なしで、~1-2MBの最小限のシェル環境を用意できる |
| レイヤー機構 | OverlayFS | カーネルネイティブのUnion Filesystemで、Docker本体のoverlay2ドライバと同じ機構 |
| CLIフレームワーク | 標準`flag`パッケージのみ | 学習教材としてOS内部機構に焦点を当てるため、cobra等の外部依存を避けた |

---

## 使い方

### 前提条件

```bash
go build -o mini-docker ./cmd/mini-docker
./scripts/build-rootfs.sh ./rootfs
```

### コンテナ起動

```bash
./mini-docker run --rootfs ./rootfs --hostname test-container -- /bin/sh
```

### リソース制限付き起動

```bash
./mini-docker run --rootfs ./rootfs --mem 100m --cpus 0.5 --pids 64 -- /bin/sh
```

### 実行中コンテナ一覧

```bash
./mini-docker ps
```

### レイヤー重ね合わせデモ

```bash
./mini-docker layers-demo
```

---

## テスト結果

このサンドボックス環境（root権限、cgroup v1 hybrid、OverlayFS対応、Linux 6.18.5）で実際に検証済み。

### 1. UTS / PID / Mount namespace 分離

```
$ hostname
vm
$ ./mini-docker run --rootfs ./rootfs --hostname test-container -- \
    /bin/sh -c 'hostname; ls /; echo ---; ps aux; echo ---; mount'
test-container
bin
dev
etc
proc
root
sys
tmp
---
PID   USER     COMMAND
    1 root     /bin/sh -c hostname; ls /; echo ---; ps aux; echo ---; mount
    8 root     {ps} /bin/sh -c hostname; ls /; echo ---; ps aux; echo ---; mount
---
/dev/vda on / type ext4 (rw,relatime,resuid=65534,resgid=65534)
proc on /proc type proc (rw,relatime)
```

ホスト側のhostname（`vm`）とは異なる値がコンテナ内で見え、`ps aux`はPID 1とその子プロセスのみ、`mount`はrootfsとprocのみを表示 — ホスト側の全プロセス・全マウントは一切見えないことを確認した。

### 2. メモリ cgroup による OOM Kill

```
$ ./mini-docker run --rootfs ./rootfs --mem 20m -- \
    /bin/sh -c 'a="x"; while true; do a="$a$a"; done'
(プロセスはKillされて終了)

$ dmesg | tail -3
oom-kill:constraint=CONSTRAINT_MEMCG,...,oom_memcg=/mini-docker/c6229c826b86,...
Memory cgroup out of memory: Killed process 7033 (sh) total-vm:51568kB, anon-rss:20496kB,...
```

`--mem 20m` を超えるメモリ確保を試みたプロセスが、cgroupのメモリ上限によってOOM Killされることを確認した。

### 3. pids cgroup によるフォークボム防止

```
$ ./mini-docker run --rootfs ./rootfs --pids 20 -- \
    /bin/sh -c 'i=0; while true; do sh -c "sleep 5" & i=$((i+1)); done'
/bin/sh: can't fork: Resource temporarily unavailable

$ uptime   # ホスト側は無事
08:30:46 up 8 min, load average: 0.34, 0.18, 0.09
```

`--pids 20` の上限に達した時点で `fork()` が `Resource temporarily unavailable` で失敗し、ホスト全体への影響なくコンテナ内だけで安全に制限されることを確認した（検証は必ず上限付きで実施し、無制限のフォークボムは実行していない）。

### 4. cpu cgroup による使用率制限

```
$ ./mini-docker run --rootfs ./rootfs --cpus 0.5 -- /bin/sh -c 'while true; do :; done' &
$ top -bn1 | grep sh
 8211 root  20   0    2408   1616   1548 R  50.0   0.0   0:01.62 sh
```

CPUを使い切ろうとする無限ループのプロセスが `--cpus 0.5` により実際に50%前後に制限されることを確認した。

### 5. OverlayFS レイヤーデモ

```
$ ./mini-docker layers-demo
$ cat overlay-demo/merged/shared.txt
layer2 version (overwrites layer1)      ← 優先度の高いlayer2が勝つ
$ cat overlay-demo/merged/hello.txt
from layer1 (base image)                ← layer1のみのファイルもそのまま見える
$ echo "new file content" > overlay-demo/merged/new.txt
$ ls overlay-demo/upper/
new.txt                                  ← 書き込みはupperへcopy-upされる
```

レイヤーの優先順位（後勝ち）とcopy-upの挙動を、実際のマウント・ファイル操作で確認した。

### 6. `ps` によるコンテナ一覧

```
$ ./mini-docker run --rootfs ./rootfs -- sleep 60 &
$ ./mini-docker ps
CONTAINER ID  PID   COMMAND    ROOTFS    UPTIME
458cbbdf3753  9329  sleep 60   ./rootfs  1s
```

---

## 設計上の判断

### pivot_root を選んだ理由（chrootではなく）
`chroot`はルートディレクトリを変更するだけでマウントポイントの完全な入れ替えではなく、脱出（chroot escape）の手法が知られている。`pivot_root`は現在のルートマウントを完全に新しいマウントへ切り替え、旧ルートを明示的にunmountできるため、実際のコンテナランタイムと同じ堅牢な方式を採用した。

### `Cloneflags` を選んだ理由（`unshare`外部コマンドではなく）
Goの`os/exec`は`SysProcAttr.Cloneflags`で`clone(2)`のnamespaceフラグを直接指定でき、余分なプロセス階層（`unshare`自体のプロセス）を挟まない。runc等、実際のコンテナランタイムの実装方針に近い。

### cgroup v1 を選んだ理由
このサンドボックス環境がcgroup v1 hybridでマウントされているため。v1と v2ではファイル名が異なる（例: `memory.limit_in_bytes` vs `memory.max`）ため、`cgroups.New()`内で各コントローラディレクトリの存在チェックを行い、v1前提であることを明示している。

### ネットワークnamespaceを分離のみに留めた理由
`CLONE_NEWNET`でnamespaceは分離するが、veth pair・bridge・NAT配線は実装しなかった。理由は2つ:
1. ブリッジ/iptables周りはネストしたサンドボックス環境で不確実性が高く、検証が難しい
2. 「何もしないとloしか無く外部と通信できない」こと自体が「だからDockerには`docker0`ブリッジが必要」という理解につながる、それ自体が良い教材になる

### レイヤーデモを独立コマンドにした理由
`run`のコンテナ実行と、OverlayFSによるイメージレイヤー概念は、本来は密接に関係する（Dockerの各イメージレイヤー＋コンテナ書き込み層）が、実イメージフォーマット・レジストリ通信を伴わずに「レイヤー重ね合わせの仕組みそのもの」だけを理解するため、あえて`layers-demo`として独立させた。

---

## 今後の改善点（次段階として取り組む）

- [ ] veth pair + bridge + NAT によるコンテナの実ネットワーク疎通
- [ ] レジストリからの実イメージ取得（`docker save`形式やOCIイメージのtar展開）
- [ ] Dockerfile相当のビルド機能
- [ ] user namespace（`CLONE_NEWUSER`）によるrootless動作
- [ ] cgroup v2 unified対応（自動判別してファイル名を切り替え）
- [ ] capability drop / seccompによるセキュリティ強化
- [ ] `run`にもOverlayFSを組み込み、コンテナごとに使い捨てのcopy-on-writeなrootfsを持たせる（`--ephemeral`）
- [ ] 簡易init/reaperプロセス（PID 1問題への対応。現状は対象コマンドがそのままPID 1になるため、ゾンビ回収を行わない）
- [ ] クラッシュ時に残る空のcgroupディレクトリの起動時クリーンアップ
