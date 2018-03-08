@ja:# 基本的な操作
@en:# Basic operations

@ja:## GPUオフロードの確認

@ja{
クエリがGPUで実行されるかどうかを確認するには`EXPLAIN`コマンドを使用します。
SQL処理は内部的にいくつかの要素に分解され処理されますが、PG-StromがGPUを適用して並列処理を行うのはSCAN、JOIN、GROUP BYの各ワークロードです。標準でPostgreSQLが提供している各処理の代わりに、GpuScan、GpuJoin、GpuPreAggが表示された場合、そのクエリはGPUによって処理される事となります。

以下は`EXPLAIN`コマンドの実行例です。
}
```
postgres=# EXPLAIN SELECT cat,count(*),avg(ax)
                     FROM t0 NATURAL JOIN t1 NATURAL JOIN t2
                    GROUP BY cat;
                                  QUERY PLAN
--------------------------------------------------------------------------------
 GroupAggregate  (cost=989186.82..989190.94 rows=27 width=20)
   Group Key: t0.cat
   ->  Sort  (cost=989186.82..989187.29 rows=189 width=44)
         Sort Key: t0.cat
         ->  Custom Scan (GpuPreAgg)  (cost=989175.89..989179.67 rows=189 width=44)
               Reduction: Local
               GPU Projection: cat, pgstrom.nrows(), pgstrom.nrows((ax IS NOT NULL)), pgstrom.psum(ax)
               Combined GpuJoin: enabled
               ->  Custom Scan (GpuJoin) on t0  (cost=14744.40..875804.46 rows=99996736 width=12)
                     GPU Projection: t0.cat, t1.ax
                     Outer Scan: t0  (cost=0.00..1833360.36 rows=99996736 width=12)
                     Depth 1: GpuHashJoin  (nrows 99996736...99996736)
                              HashKeys: t0.aid
                              JoinQuals: (t0.aid = t1.aid)
                              KDS-Hash (size: 10.39MB)
                     Depth 2: GpuHashJoin  (nrows 99996736...99996736)
                              HashKeys: t0.bid
                              JoinQuals: (t0.bid = t2.bid)
                              KDS-Hash (size: 10.78MB)
                     ->  Seq Scan on t1  (cost=0.00..1972.85 rows=103785 width=12)
                     ->  Seq Scan on t2  (cost=0.00..1935.00 rows=100000 width=4)
(21 rows)
```
@ja{
実行計画の中に見慣れない処理が含まれている事に気が付かれたでしょう。
CustomScanのフレームワーク上に、GpuJoinおよびGpuPreAggが実装されています。ここでGpuJoinは`t0`と`t1`、および`t2`とのJOIN処理を実行し、その結果を受け取るGpuPreAggは列`cat`によるGROUP BY処理をGPUで実行します。
}

@ja{
PostgreSQLがクエリ実行計画を構築する過程でPG-Stromはオプティマイザに介入し、SCAN、JOIN、GROUP BYの各ワークロードをGPUで実行可能である場合、そのコストを算出してPostgreSQLのオプティマイザに実行計画の候補を提示します。
推定されたコスト値がCPUで実行する他の実行計画よりも小さな値である場合、GPUを用いた代替の実行計画が採用される事になります。
}

@ja{
ワークロードをGPUで実行するためには、少なくとも演算式または関数、および使用されているデータ型がPG-Stromでサポートされている必要があります。
`int`や`float`といった数値型、`date`や`timestamp`といった日付時刻型、`text`のような文字列型がサポートされており、また、四則演算や大小比較といった数多くのビルトイン演算子がサポートされています。
詳細な一覧に関しては(リファレンス)[reference.md]を参照してください。
}

@ja:##CPU+GPUハイブリッド並列
@en:##CPU+GPU Hybrid Parallel

@ja{
PG-StromはPostgreSQLのCPU並列実行に対応しています。

PostgreSQLのCPU並列実行は、Gatherノードがいくつかのバックグラウンドワーカプロセスを起動し、各バックグラウンドワーカが"部分的に"実行したクエリの結果を後で結合する形で実装されています。
GpuJoinやGpuPreAggといったPG-Stromの処理はバックグラウンドワーカ側での実行に対応しており、個々のプロセスが互いにGPUを使用して処理を進めます。通常、GPUへデータを供給するために個々のCPUコアがバッファをセットアップするための処理速度は、GPUでのSQLワークロードの処理速度に比べてずっと遅いため、CPU並列とGPU並列をハイブリッドで利用する事で処理速度の向上が期待できます。
ただし、GPUを利用するために必要なCUDAコンテキストは各プロセスごとに作成され、CUDAコンテキストを生成するたびにある程度のGPUリソースが消費されるため、常にCPU並列度が高ければ良いという訳ではありません。
}
@ja{
以下の実行計画を見てください。
Gather以下の実行計画はバックグラウンドワーカーが実行可能なものです。1億行を保持する`t0`テーブルを4プロセスのバックグラウンドワーカとコーディネータプロセスでスキャンするため、プロセスあたり2000万行をGpuJoinおよびGpuPreAggで処理し、その結果をGatherノードで結合します。
}
```
# EXPLAIN SELECT cat,count(*),avg(ax)
            FROM t0 NATURAL JOIN t1
           GROUP by cat;
                                   QUERY PLAN
--------------------------------------------------------------------------------
 GroupAggregate  (cost=955705.47..955720.93 rows=27 width=20)
   Group Key: t0.cat
   ->  Sort  (cost=955705.47..955707.36 rows=756 width=44)
         Sort Key: t0.cat
         ->  Gather  (cost=955589.95..955669.33 rows=756 width=44)
               Workers Planned: 4
               ->  Parallel Custom Scan (GpuPreAgg)  (cost=954589.95..954593.73 rows=189 width=44)
                     Reduction: Local
                     GPU Projection: cat, pgstrom.nrows(), pgstrom.nrows((ax IS NOT NULL)), pgstrom.psum(ax)
                     Combined GpuJoin: enabled
                     ->  Parallel Custom Scan (GpuJoin) on t0  (cost=27682.82..841218.52 rows=99996736 width=12)
                           GPU Projection: t0.cat, t1.ax
                           Outer Scan: t0  (cost=0.00..1083384.84 rows=24999184 width=8)
                           Depth 1: GpuHashJoin  (nrows 24999184...99996736)
                                    HashKeys: t0.aid
                                    JoinQuals: (t0.aid = t1.aid)
                                    KDS-Hash (size: 10.39MB)
                           ->  Seq Scan on t1  (cost=0.00..1972.85 rows=103785 width=12)
(18 rows)
```

@ja:##下位プランの引き上げ
@en:##Pullup underlying plans

@ja{
PG-StromはSCAN、JOIN、GROUP BYの各処理をGPUで実行する事が可能ですが、これに対応するPostgreSQL標準の処理を単純に置き換えただけでは困った事態が発生します。
SCANが終わった後のデータをいったんホスト側のバッファに書き戻し、次にそれをJOINするために再びGPUへとコピーし、さらにGROUP BYを実行する前に再びホスト側のバッファに書き戻し・・・といった形で、CPUとGPUの間でデータのピンポンが発生してしまうのです。

これを避けるために、PG-Stromは下位プランを引き上げて一度のGPU Kernelの実行で処理してしまうというモードを持っています。
以下のパターンで下位プランの引き上げが発生する可能性があります。

- SCAN + JOIN
- SCAN + GROUP BY
- SCAN + JOIN + GROUP BY
}

@ja{
以下の実行計画は、下位プランの引き上げを全く行わないケースです。

GpuScanの実行結果をGpuJoinが受取り、さらにその実行結果をGpuPreAggが受け取って最終結果を生成する事が分かります。
}

```
# EXPLAIN SELECT cat,count(*),avg(ax)
            FROM t0 NATURAL JOIN t1
           WHERE aid < bid
           GROUP BY cat;
                              QUERY PLAN

--------------------------------------------------------------------------------
 GroupAggregate  (cost=1239991.03..1239995.15 rows=27 width=20)
   Group Key: t0.cat
   ->  Sort  (cost=1239991.03..1239991.50 rows=189 width=44)
         Sort Key: t0.cat
         ->  Custom Scan (GpuPreAgg)  (cost=1239980.10..1239983.88 rows=189 width=44)
               Reduction: Local
               GPU Projection: cat, pgstrom.nrows(), pgstrom.nrows((ax IS NOT NULL)), pgstrom.psum(ax)
               ->  Custom Scan (GpuJoin)  (cost=50776.43..1199522.96 rows=33332245 width=12)
                     GPU Projection: t0.cat, t1.ax
                     Depth 1: GpuHashJoin  (nrows 33332245...33332245)
                              HashKeys: t0.aid
                              JoinQuals: (t0.aid = t1.aid)
                              KDS-Hash (size: 10.39MB)
                     ->  Custom Scan (GpuScan) on t0  (cost=12634.49..1187710.85 rows=33332245 width=8)
                           GPU Projection: cat, aid
                           GPU Filter: (aid < bid)
                     ->  Seq Scan on t1  (cost=0.00..1972.85 rows=103785 width=12)
(18 rows)
```

@ja{
図：下位ノードの引き上げとCombined GPU kernelの模式図
}

![combined gpu kernel](./img/combined-gpu-kernel.png)

@ja{
一方、以下の実行計画は、下位ノードの引き上げを行ったものです。
}

```
# EXPLAIN ANALYZE SELECT cat,count(*),avg(ax)
                    FROM t0 NATURAL JOIN t1
                   WHERE aid < bid
                   GROUP BY cat;
                              QUERY PLAN
--------------------------------------------------------------------------------
 GroupAggregate  (cost=903669.50..903673.62 rows=27 width=20)
                 (actual time=7761.630..7761.644 rows=27 loops=1)
   Group Key: t0.cat
   ->  Sort  (cost=903669.50..903669.97 rows=189 width=44)
             (actual time=7761.621..7761.626 rows=27 loops=1)
         Sort Key: t0.cat
         Sort Method: quicksort  Memory: 28kB
         ->  Custom Scan (GpuPreAgg)  (cost=903658.57..903662.35 rows=189 width=44)
                                      (actual time=7761.531..7761.540 rows=27 loops=1)
               Reduction: Local
               GPU Projection: cat, pgstrom.nrows(), pgstrom.nrows((ax IS NOT NULL)), pgstrom.psum(ax)
               Combined GpuJoin: enabled
               ->  Custom Scan (GpuJoin) on t0  (cost=12483.41..863201.43 rows=33332245 width=12)
                                                (never executed)
                     GPU Projection: t0.cat, t1.ax
                     Outer Scan: t0  (cost=12634.49..1187710.85 rows=33332245 width=8)
                                     (actual time=59.623..5557.052 rows=100000000 loops=1)
                     Outer Scan Filter: (aid < bid)
                     Rows Removed by Outer Scan Filter: 50002874
                     Depth 1: GpuHashJoin  (plan nrows: 33332245...33332245, actual nrows: 49997126...49997126)
                              HashKeys: t0.aid
                              JoinQuals: (t0.aid = t1.aid)
                              KDS-Hash (size plan: 10.39MB, exec: 64.00MB)
                     ->  Seq Scan on t1  (cost=0.00..1972.85 rows=103785 width=12)
                                         (actual time=0.013..15.303 rows=100000 loops=1)
 Planning time: 0.506 ms
 Execution time: 8495.391 ms
(21 rows)
```
@ja{
まず、テーブル`t0`へのスキャンがGpuJoinの実行計画に埋め込まれ、GpuScanが消えている事にお気付きでしょう。
これはGpuJoinが配下のGpuScanを引き上げ、一体化したGPUカーネル関数でWHERE句の処理も行った事を意味しています。

加えて奇妙なことに、GpuJoinが(never executed)と表示されています。これはGpuPreAggが配下のGpuJoinを引き上げ、
一体化したGPUカーネル関数でJOINとGROUP BYを実行した事を意味しています。
}

@ja{
SCAN処理の引き上げは`pg_strom.pullup_outer_scan`パラメータによって制御できます。
また、JOIN処理の引き上げは`pg_strom.pullup_outer_join`パラメータによって制御できます。
いずれのパラメータもデフォルトでは`on`に設定されており、通常はこれを無効化する必要はありませんが、トラブル時の問題切り分け手段の一つとして利用する事ができます。
}


@ja:# システム管理上の注意
@en:# Notes for system administration

@ja:## ナレッジベース
@en:## Knowledge base

@ja{
PG-Stromプロジェクトのwikiサイトには、ノートと呼ばれる詳細な技術情報が公開されています。
}
@en{
We publish several articles, just called "notes", on the project wiki-site of PG-Strom.
}
(https://github.com/heterodb/pg-strom/wiki)[https://github.com/heterodb/pg-strom/wiki]

@ja:## MPSデーモンの利用
@en:## Usage of MPS daemon

@ja{
PostgreSQLのようにマルチプロセス環境でGPUを使用する場合、GPU側コンテキストスイッチの低減やデバイス管理に必要なリソースの低減を目的として、MPS(Multi-Process Service)を使用する事が一般的なソリューションです。
}

(https://docs.nvidia.com/deploy/mps/index.html)[https://docs.nvidia.com/deploy/mps/index.html]

@ja{
しかし、PG-Stromの利用シーンでは、MPSサービスの既知問題により正常に動作しないCUDA APIが存在し、以下のような限定された条件下を除いては使用すべきではありません。

- GPUを使用するPostgreSQLプロセス（CPU並列クエリにおけるバックグラウンドワーカを含む）の数が16個以下であること。
    - Volta世代のGPUでは48個以下
- gstore_fdwを使用しない事。
}

@ja{
これは`CUipcMemHandle`を用いてプロセス間でGPUデバイスメモリを共有する際に、MPSサービス下のプロセスで獲得したGPUデバイスメモリを非MPSサービス下のプロセスでオープンできない事で、GpuJoinが使用するハッシュ表をバックグラウンドワーカー間で共有できなくなるための制限事項です。

この問題は既にNVIDIAへ報告され、新しいバージョンのCUDA Toolkitにおいて修正されるとの回答を得ています。
}


@ja:# トラブルシューティング
@en:# Trouble Shooting

@ja:## 問題の切り分け
@en:## Identify the problem

@ja{
特定のワークロードを実行した際に何がしかの問題が発生する場合には、それが何に起因するものであるのかを特定するのはトラブルシューティングの第一歩です。

残念ながら、PostgreSQL開発者コミュニティと比べPG-Stromの開発者コミュニティは非常に少ない数の開発者によって支えられています。そのため、ソフトウェアの品質や実績といった観点から、まずPG-Stromが悪さをしていないか疑うのは妥当な判断です。

PG-Stromの全機能を一度に有効化/無効化するには`pg_strom.enabled`パラメータを使用する事ができます。
以下の設定を行う事でPG-Stromは無効化され、標準のPostgreSQLと全く同一の状態となります。
それでもなお問題が再現するかどうかは一つの判断材料となるでしょう。
}
```
# SET pg_strom.enabled = off;
```

@ja{
この他にも、GpuScan、GpuJoin、GpuPreAggといった特定の実行計画のみを無効化するパラメータや、SCANやJOINの引き上げを無効化するパラメータも定義されています。

これらの詳細は(リファレンス)[references.md#gpu]を参照してください。
}

@ja:## クラッシュダンプの採取
@en:## Collecting crash dump

@ja{
システムのクラッシュを引き起こすような重大なトラブルの解析にはクラッシュダンプの採取が欠かせません。
本節では、PostgreSQLとPG-Stromをクラッシュダンプを取得し、障害発生時のバックトレースを採取するための手順を説明します。
}

@ja:### LimitCORE 設定の追加
@en:### Add LimitCORE configuration

@ja{
システムのクラッシュ時にクラッシュダンプを生成させるには、PostgreSQLサーバプロセスが生成する事のできるcoreファイルのサイズを無制限に変更する必要があります。

systemdからPostgreSQLを起動する場合、これを設定するのは以下のファイルです。
}

```
/etc/systemd/system/postgresql-<version>.service.d/override.conf
```
@ja{
このファイルに以下の内容を追記します。
}
```
[Service]
LimitCORE=infinity
```
@ja{
この後でPostgreSQLサーバプロセスを再起動すると、*Max core file size*がunlimitedに設定されているはずです。

以下のように確認する事ができます。
}

```
$ cat /proc/<PID of postmaster>/limits
Limit                     Soft Limit           Hard Limit           Units
    :                         :                    :                  :
Max core file size        unlimited            unlimited            bytes
    :                         :                    :                  :
```

@ja:### debuginfoパッケージのインストール
@en:### Installation of debuginfo package

@ja{
クラッシュダンプから意味のある情報を読み取るにはシンボル情報が必要です。

これらは`-debuginfo`パッケージに格納されており、システムにインストールされているPostgreSQLおよびPG-Stromのパッケージに応じてそれぞれ追加インストールが必要です。
}

```
# yum install postgresql10-debuginfo pg_strom-PG10-debuginfo
            :
================================================================================
 Package                  Arch    Version             Repository           Size
================================================================================
Installing:
 pg_strom-PG10-debuginfo  x86_64  1.9-180301.el7      heterodb-debuginfo  766 k
 postgresql10-debuginfo   x86_64  10.3-1PGDG.rhel7    pgdg10              9.7 M

Transaction Summary
================================================================================
Install  2 Packages
            :
Installed:
  pg_strom-PG10-debuginfo.x86_64 0:1.9-180301.el7
  postgresql10-debuginfo.x86_64 0:10.3-1PGDG.rhel7

Complete!
```

@ja:### バックトレースの確認
@en:### Checking the backtrace

@ja{
クラッシュダンプの作成されるパスは、カーネルパラメータ`kernel.core_pattern`および`kernel.core_uses_pid`の値によって決まります。
通常はプロセスのカレントディレクトリに作成されますので、systemdからPostgreSQLを起動した場合はデータベースクラスタが構築される`/var/lib/pgdata`を確認してください。

`core.<PID>`ファイルが生成されているのを確認したら、gdbを用いてクラッシュに至るバックトレースを確認します。

gdbの`-c`オプションでコアファイルを、`-f`オプションでクラッシュしたプログラムを指定します。
}

```
# gdb -c /var/lib/pgdata/core.134680 -f /usr/pgsql-10/bin/postgres
GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-100.el7_4.1
       :
(gdb) bt
#0  0x00007fb942af3903 in __epoll_wait_nocancel () from /lib64/libc.so.6
#1  0x00000000006f71ae in WaitEventSetWaitBlock (nevents=1,
    occurred_events=0x7ffee51e1d70, cur_timeout=-1, set=0x2833298)
    at latch.c:1048
#2  WaitEventSetWait (set=0x2833298, timeout=timeout@entry=-1,
    occurred_events=occurred_events@entry=0x7ffee51e1d70,
    nevents=nevents@entry=1, wait_event_info=wait_event_info@entry=100663296)
    at latch.c:1000
#3  0x00000000006210fb in secure_read (port=0x2876120,
    ptr=0xcaa7e0 <PqRecvBuffer>, len=8192) at be-secure.c:166
#4  0x000000000062b6e8 in pq_recvbuf () at pqcomm.c:963
#5  0x000000000062c345 in pq_getbyte () at pqcomm.c:1006
#6  0x0000000000718682 in SocketBackend (inBuf=0x7ffee51e1ef0)
    at postgres.c:328
#7  ReadCommand (inBuf=0x7ffee51e1ef0) at postgres.c:501
#8  PostgresMain (argc=<optimized out>, argv=argv@entry=0x287bb68,
    dbname=0x28333f8 "postgres", username=<optimized out>) at postgres.c:4030
#9  0x000000000047adbc in BackendRun (port=0x2876120) at postmaster.c:4405
#10 BackendStartup (port=0x2876120) at postmaster.c:4077
#11 ServerLoop () at postmaster.c:1755
#12 0x00000000006afb7f in PostmasterMain (argc=argc@entry=3,
    argv=argv@entry=0x2831280) at postmaster.c:1363
#13 0x000000000047bbef in main (argc=3, argv=0x2831280) at main.c:228
```

@ja{
gdbの`bt`コマンドでバックトレースを確認します。
このケースでは、クライアントからのクエリを待っている状態のPostgreSQLバックエンドに`SIGSEGV`シグナルを送出してクラッシュを引き起こしたため、`WaitEventSetWait`延長上の`__epoll_wait_nocancel`でプロセスがクラッシュしている事がわかります。
}


