@ja:# 基本的な操作
@en:# Basic operations

@ja:## GPUオフロードの確認
@en:## Confirmation of GPU off-loading

@ja{
クエリがGPUで実行されるかどうかを確認するには`EXPLAIN`コマンドを使用します。
SQL処理は内部的にいくつかの要素に分解され処理されますが、PG-StromがGPUを適用して並列処理を行うのはSCAN、JOIN、GROUP BYの各ワークロードです。標準でPostgreSQLが提供している各処理の代わりに、GpuScan、GpuJoin、GpuPreAggが表示された場合、そのクエリはGPUによって処理される事となります。

以下は`EXPLAIN`コマンドの実行例です。
}

@en{
You can use `EXPLAIN` command to check whether query is executed on GPU device or not.
A query is internally split into multiple elements and executed, and PG-Strom is capable to run SCAN, JOIN and GROUP BY in parallel on GPU device. If you can find out GpuScan, GpuJoin or GpuPreAgg was displayed instead of the standard operations by PostgreSQL, it means the query is partially executed on GPU device.

Below is an example of `EXPLAIN` command output.
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
CustomScan機構を用いてGpuJoinおよびGpuPreAggが実装されています。ここでGpuJoinは`t0`と`t1`、および`t2`とのJOIN処理を実行し、その結果を受け取るGpuPreAggは列`cat`によるGROUP BY処理をGPUで実行します。
}
@en{
You can notice some unusual query execution plans.
GpuJoin and GpuPreAgg are implemented on the CustomScan mechanism. In this example, GpuJoin runs JOIN operation on `t0`, `t1` and `t1`, then GpuPreAgg which receives the result of GpuJoin runs GROUP BY operation by the `cat` column on GPU device.
}

@ja{
PostgreSQLがクエリ実行計画を構築する過程でPG-Stromはオプティマイザに介入し、SCAN、JOIN、GROUP BYの各ワークロードをGPUで実行可能である場合、そのコストを算出してPostgreSQLのオプティマイザに実行計画の候補を提示します。
推定されたコスト値がCPUで実行する他の実行計画よりも小さな値である場合、GPUを用いた代替の実行計画が採用される事になります。
}
@en{
PG-Strom interacts with the query optimizer during PostgreSQL is building a query execution plan, and it offers alternative query execution plan with estimated cost for PostgreSQL's optimizer, if any of SCAN, JOIN, or GROUP BY are executable on GPU device.
This estimated cost is better than other query execution plans that run on CPU, it chooses the alternative execution plan that shall run on GPU device.
}

@ja{
ワークロードをGPUで実行するためには、少なくとも演算式または関数、および使用されているデータ型がPG-Stromでサポートされている必要があります。
`int`や`float`といった数値型、`date`や`timestamp`といった日付時刻型、`text`のような文字列型がサポートされており、また、四則演算や大小比較といった数多くのビルトイン演算子がサポートされています。
詳細な一覧に関しては[リファレンス](references.md)を参照してください。
}
@en{
For GPU execution, it requires operators, functions and data types in use must be supported by PG-Strom.
It supports numeric types like `int` or `float`, date and time types like `date` or `timestamp`, variable length string like `text` and so on. It also supports arithmetic operations, comparison operators and many built-in operators.
See [References](references.md) for the detailed list.
}

@ja:##CPU+GPUハイブリッド並列
@en:##CPU+GPU Hybrid Parallel

@ja{
PG-StromはPostgreSQLのCPU並列実行に対応しています。

PostgreSQLのCPU並列実行は、Gatherノードがいくつかのバックグラウンドワーカプロセスを起動し、各バックグラウンドワーカが"部分的に"実行したクエリの結果を後で結合する形で実装されています。
GpuJoinやGpuPreAggといったPG-Stromの処理はバックグラウンドワーカ側での実行に対応しており、個々のプロセスが互いにGPUを使用して処理を進めます。通常、GPUへデータを供給するために個々のCPUコアがバッファをセットアップするための処理速度は、GPUでのSQLワークロードの処理速度に比べてずっと遅いため、CPU並列とGPU並列をハイブリッドで利用する事で処理速度の向上が期待できます。
ただし、GPUを利用するために必要なCUDAコンテキストは各プロセスごとに作成され、CUDAコンテキストを生成するたびにある程度のGPUリソースが消費されるため、常にCPU並列度が高ければ良いという訳ではありません。
}
@en{
PG-Strom also supports PostgreSQL's CPU parallel execution.

In the CPU parallel execution mode, Gather node launches several background worker processes, then it gathers the result of "partial" execution by individual background workers.
CustomScan execution plan provided by PG-Strom, like GpuJoin or GpuPreAgg, support execution at the background workers. They process their partial task using GPU individually. A CPU core usually needs much more time to set up buffer to supply data for GPU than execution of SQL workloads on GPU, so hybrid usage of CPU and GPU parallel can expect higher performance.
On the other hands, each process creates CUDA context that is required to communicate GPU and consumes a certain amount of GPU resources, so higher parallelism on CPU-side is not always better.
}

@ja{
以下の実行計画を見てください。
Gather以下の実行計画はバックグラウンドワーカーが実行可能なものです。1億行を保持する`t0`テーブルを4プロセスのバックグラウンドワーカとコーディネータプロセスでスキャンするため、プロセスあたり2000万行をGpuJoinおよびGpuPreAggで処理し、その結果をGatherノードで結合します。
}
@en{
Look at the query execution plan below.
Execution plan tree under the Gather is executable on background worker process. It scans `t0` table which has 100million rows using four background worker processes and the coordinator process, in other words, 20million rows are handled per process by GpuJoin and GpuPreAgg, then its results are merged at Gather node.
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
}
@en{
PG-Strom can run SCAN, JOIN and GROUP BY workloads on GPU, however, it does not work with best performance if these custom execution plan simply replace the standard operations at PostgreSQL.
An example of problematic scenario is that SCAN once writes back its result data set to the host buffer then send the same data into GPU again to execute JOIN. Once again, JOIN results are written back and send to GPU to execute GROUP BY. It causes data ping-pong between CPU and GPU.

To avoid such inefficient jobs, PG-Strom has a special mode which pulls up its sub-plan to execute a bunch of jobs in a single GPU kernel invocation. Combination of the operations blow can cause pull-up of sub-plans.
}
- SCAN + JOIN
- SCAN + GROUP BY
- SCAN + JOIN + GROUP BY

![combined gpu kernel](./img/combined-kernel-overview.png)

@ja{
以下の実行計画は、下位プランの引き上げを全く行わないケースです。

GpuScanの実行結果をGpuJoinが受取り、さらにその実行結果をGpuPreAggが受け取って最終結果を生成する事が分かります。
}
@en{
The execution plan example below never pulls up the sub-plans.

GpuJoin receives the result of GpuScan, then its results are passed to GpuPreAgg to generate the final results.
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
この場合、各実行ステージにおいてGPUとホストバッファの間でデータのピンポンが発生するため、実行効率はよくありません。
}
@en{
This example causes data ping-pong between GPU and host buffers for each execution stage, so not efficient and less performance.
}

@ja{
一方、以下の実行計画は、下位ノードの引き上げを行ったものです。
}
@en{
On the other hands, the query execution plan below pulls up sub-plans.
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

加えて奇妙なことに、`EXPLAIN ANALYZE`の結果にはGpuJoinが(never executed)と表示されています。
これはGpuPreAggが配下のGpuJoinを引き上げ、一体化したGPUカーネル関数でJOINとGROUP BYを実行した事を意味しています。
}
@en{
You may notice that SCAN on the table `t0` is embedded into GpuJoin, and GpuScan gets vanished.
It means GpuJoin pulls up the underlying GpuScan, then combined GPU kernel function is also responsible for evaluation of the supplied WHERE-clause.

In addition, here is a strange output in `EXPLAIN ANALYZE` result - it displays *(never executed)* for GpuJoin.
It means GpuJoin is never executed during the query execution, and it is right. GpuPreAgg pulls up the underlying GpuJoin, then its combined GPU kernel function runs JOIN and GROUP BY.
}

@ja{
SCAN処理の引き上げは`pg_strom.pullup_outer_scan`パラメータによって制御できます。
また、JOIN処理の引き上げは`pg_strom.pullup_outer_join`パラメータによって制御できます。
いずれのパラメータもデフォルトでは`on`に設定されており、通常はこれを無効化する必要はありませんが、トラブル時の問題切り分け手段の一つとして利用する事ができます。
}
@en{
The `pg_strom.pullup_outer_scan` parameter controls whether SCAN is pulled up, and the `pg_strom.pullup_outer_join` parameter also controls whether JOIN is pulled up.
Both parameters are configured to `on`. Usually, no need to disable them, however, you can use the parameters to identify the problems on system troubles.
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
[https://github.com/heterodb/pg-strom/wiki](https://github.com/heterodb/pg-strom/wiki)

@ja:## MPSデーモンの利用
@en:## Usage of MPS daemon

@ja{
PostgreSQLのようにマルチプロセス環境でGPUを使用する場合、GPU側コンテキストスイッチの低減やデバイス管理に必要なリソースの低減を目的として、MPS(Multi-Process Service)を使用する事が一般的なソリューションです。
}
@en{
In case when multi-process application like PostgreSQL uses GPU device, it is a well known solution to use MPS (Multi-Process Service) to reduce context switch on GPU side and resource consumption for device management.
}

[https://docs.nvidia.com/deploy/mps/index.html](https://docs.nvidia.com/deploy/mps/index.html)

@ja{
しかし、PG-Stromの利用シーンでは、MPSサービスの既知問題により正常に動作しないCUDA APIが存在し、以下のような限定された条件下を除いては使用すべきではありません。

- GPUを使用するPostgreSQLプロセス（CPU並列クエリにおけるバックグラウンドワーカを含む）の数が常に16個以下である。Volta世代のGPUの場合は48個以下である。
- gstore_fdwを使用しない事。
}
@en{
However, here is a known issue; some APIs don't work correctly user the use case of PG-Strom due to the problem of MPS daemon. So, we don't recomment to use MPS daemon except for the situation below:

- Number of PostgreSQL processes which use GPU device (including the background workers launched by CPU parallel execution) is always less than 16. If Volta generation, it is less than 48.
- gstore_fdw shall not be used.
}

@ja{
これは`CUipcMemHandle`を用いてプロセス間でGPUデバイスメモリを共有する際に、MPSサービス下のプロセスで獲得したGPUデバイスメモリを非MPSサービス下のプロセスでオープンできない事で、GpuJoinが使用するハッシュ表をバックグラウンドワーカー間で共有できなくなるための制限事項です。

この問題は既にNVIDIAへ報告し、新しいバージョンのCUDA Toolkitにおいて修正されるとの回答を得ています。
}
@en{
This known problem is, when we share GPU device memory inter processes using `CUipcMemHandle`, a device memory region acquired by the process under MPS service cannot be opened by the process which does not use MPS. This problem prevents to share the inner hash-table of GpuJoin with background workers on CPU parallel execution.

This problem is already reported to NVIDIA, then we got a consensu to fix it at the next version of CUDA Toolkit.
}

@ja:# トラブルシューティング
@en:# Trouble Shooting

@ja:## 問題の切り分け
@en:## Identify the problem

@ja{
特定のワークロードを実行した際に何がしかの問題が発生する場合には、それが何に起因するものであるのかを特定するのはトラブルシューティングの第一歩です。

残念ながら、PostgreSQL開発者コミュニティと比べPG-Stromの開発者コミュニティは非常に少ない数の開発者によって支えられています。そのため、ソフトウェアの品質や実績といった観点から、まずPG-Stromが悪さをしていないか疑うのは妥当な判断です。
}
@en{
In case when a particular workloads produce problems, it is the first step to identify which stuff may cause the problem.

Unfortunately, much smaller number of developer supports the PG-Strom development community than PostgreSQL developer's community, thus, due to the standpoint of software quality and history, it is a reasonable estimation to suspect PG-Strom first.
}
@ja{
PG-Stromの全機能を一度に有効化/無効化するには`pg_strom.enabled`パラメータを使用する事ができます。
以下の設定を行う事でPG-Stromは無効化され、標準のPostgreSQLと全く同一の状態となります。
それでもなお問題が再現するかどうかは一つの判断材料となるでしょう。
}
@en{
The `pg_strom.enabled` parameter allows to turn on/off all the functionality of PG-Strom at once.
The configuration below disables PG-Strom, thus identically performs with the standard PostgreSQL.
}
```
# SET pg_strom.enabled = off;
```

@ja{
この他にも、GpuScan、GpuJoin、GpuPreAggといった特定の実行計画のみを無効化するパラメータも定義されています。

これらの詳細は[リファレンス](references.md#gpu)を参照してください。
}
@en{
In addition, we provide parameters to disable particular execution plan like GpuScan, GpuJoin and GpuPreAgg.

See [references](references.md#gpu) for more details.
}

@ja:## クラッシュダンプの採取
@en:## Collecting crash dump

@ja{
システムのクラッシュを引き起こすような重大なトラブルの解析にはクラッシュダンプの採取が欠かせません。
本節では、PostgreSQLとPG-Stromプロセスのクラッシュダンプ(CPU側)、およびPG-StromのGPUカーネルのクラッシュダンプ(GPU側)を取得し、障害発生時のバックトレースを採取するための手段を説明します。
}
@en{
Crash dump is very helpful for analysis of serious problems which lead system crash for example.
This session introduces the way to collect crash dump of the PostgreSQL and PG-Strom process (CPU side) and PG-Strom's GPU kernel, and show the back trace on the serious problems.
}

@ja:### PostgreSQL起動時設定の追加
@en:### Add configuration on PostgreSQL startup

@ja{
プロセスのクラッシュ時にクラッシュダンプ(CPU側)を生成するには、PostgreSQLサーバプロセスが生成する事のできる core ファイルのサイズを無制限に変更する必要があります。これはPostgreSQLサーバプロセスを起動するシェル上で`ulimit -c`コマンドを実行して変更する事ができます。

GPUカーネルのエラー時にクラッシュダンプ(GPU側)を生成するには、PostgreSQLサーバプロセスが環境変数`CUDA_ENABLE_COREDUMP_ON_EXCEPTION`に`1`が設定されている必要があります。
}
@en{
For generation of crash dump (CPU-side) on process crash, you need to change the resource limitation of the operating system for size of core file  PostgreSQL server process can generate.

For generation of crash dump (GPU-size) on errors of GPU kernel, PostgreSQL server process has `CUDA_ENABLE_COREDUMP_ON_EXCEPTION`environment variable, and its value has `1`.
}
@ja{
systemdからPostgreSQLを起動する場合、`/etc/systemd/system/postgresql-<version>.service.d/`以下に設定ファイルを作成し、これらの設定を追加する事ができます。

RPMインストールの場合は、以下の内容の`pg_strom.conf`というファイルが作成されています。
}
@en{
You can put a configuration file at `/etc/systemd/system/postgresql-<version>.service.d/` when PostgreSQL is kicked by systemd.

In case of RPM installation, a configuration file `pg_strom.conf` is also installed on the directory, and contains the following initial configuration.
}
```
[Service]
LimitNOFILE=65536
LimitCORE=infinity
#Environment=CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
```
@ja{
CUDA9.1においては、通常、GPUカーネルのクラッシュダンプの生成には数分以上の時間を要し、その間、エラーを発生したPostgreSQLセッションの応答は完全に停止してしまします。
そのため、は特定クエリの実行において発生するGPUカーネルに起因するエラーの原因調査を行う場合にだけ、`CUDA_ENABLE_COREDUMP_ON_EXCEPTION`環境変数を設定する事をお勧めします。
RPMインストールにおけるデフォルト設定は、`CUDA_ENABLE_COREDUMP_ON_EXCEPTION`環境変数の行をコメントアウトしています。
}
@en{
In CUDA 9.1, it usually takes more than several minutes to generate crash dump of GPU kernel, and it entirely stops response of the PostgreSQL session which causes an error.
So, we recommend to set `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` environment variable only if you investigate errors of GPU kernels which happen on a certain query.
The default configuration on RPM installation comments out the line of `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` environment variable.
}

@ja{
PostgreSQLサーバプロセスを再起動すると、*Max core file size*がunlimitedに設定されているはずです。

以下のように確認する事ができます。
}
@en{
PostgreSQL server process should have unlimited *Max core file size* configuration, after the next restart.

You can check it as follows.
}

```
# cat /proc/<PID of postmaster>/limits
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

@ja:### CPU側バックトレースの確認
@en:### Checking the back-trace on CPU side

@ja{
クラッシュダンプの作成されるパスは、カーネルパラメータ`kernel.core_pattern`および`kernel.core_uses_pid`の値によって決まります。
通常はプロセスのカレントディレクトリに作成されますので、systemdからPostgreSQLを起動した場合はデータベースクラスタが構築される`/var/lib/pgdata`を確認してください。

`core.<PID>`ファイルが生成されているのを確認したら、`gdb`を用いてクラッシュに至るバックトレースを確認します。

`gdb`の`-c`オプションでコアファイルを、`-f`オプションでクラッシュしたプログラムを指定します。
}
@en{
The kernel parameter `kernel.core_pattern` and `kernel.core_uses_pid` determine the path where crash dump is written out.
It is usually created on the current working directory of the process, check `/var/lib/pgdata` where the database cluster is deployed, if you start PostgreSQL server using systemd.

Once `core.<PID>` file gets generated, you can check its back-trace to reach system crash using `gdb`.

`gdb` speficies the core file by `-c` option, and the crashed program by `-f` option.
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
@en{
`bt` command of `gdb` displays the backtrace.
In this case, I sent `SIGSEGV` signal to the PostgreSQL backend which is waiting for queries from the client for intentional crash, the process got crashed at `__epoll_wait_nocancel` invoked by `WaitEventSetWait`.
}


@ja:### GPU側バックトレースの確認
@en:### Checking the backtrace on GPU

@ja{
GPUカーネルのクラッシュダンプは、（`CUDA_COREDUMP_FILE`環境変数を用いて明示的に指定しなければ）PostgreSQLサーバプロセスのカレントディレクトリに生成されます。
systemdからPostgreSQLを起動した場合はデータベースクラスタが構築される`/var/lib/pgdata`を確認してください。以下の名前でGPUカーネルのクラッシュダンプが生成されています。
}
@en{
Crash dump of GPU kernel is generated on the current working directory of PostgreSQL server process, unless you don't specify the path using `CUDA_COREDUMP_FILE` environment variable explicitly.
Check `/var/lib/pgdata` where the database cluster is deployed, if systemd started PostgreSQL. Dump file will have the following naming convension.
}

`core_<timestamp>_<hostname>_<PID>.nvcudmp`

@ja{
なお、デフォルト設定ではGPUカーネルのクラッシュダンプにはシンボル情報などのデバッグ情報が含まれていません。この状態では障害解析を行う事はほとんど不可能ですので、以下の設定を行ってPG-Stromが生成するGPUプログラムにデバッグ情報を含めるようにしてください。

ただし、この設定は実行時のパフォーマンスを低下させるため、恒常的な使用は非推奨です。
トラブル解析時にだけ使用するようにしてください。
}
@en{
Note that the dump-file of GPU kernel contains no debug information like symbol information in the default configuration.
It is nearly impossible to investigate the problem, so enable inclusion of debug information for the GPU programs generated by PG-Strom, as follows.

Also note than we don't recommend to turn on the configuration for daily usage, because it makes query execution performan slow down.
Turn on only when you investigate the troubles.
}
```
nvme=# set pg_strom.debug_jit_compile_options = on;
SET
```

@ja{
生成されたGPUカーネルのクラッシュダンプを確認するには`cuda-gdb`コマンドを使用します。
}
@en{
You can check crash dump of the GPU kernel using `cuda-gdb` command.
}
```
# /usr/local/cuda/bin/cuda-gdb
NVIDIA (R) CUDA Debugger
9.1 release
Portions Copyright (C) 2007-2017 NVIDIA Corporation
        :
For help, type "help".
Type "apropos word" to search for commands related to "word".
(cuda-gdb)
```

@ja{
引数なしで`cuda-gdb`コマンドを実行し、プロンプト上で`target`コマンドを使用して先ほどのクラッシュダンプを読み込みます。
}
@en{
Run `cuda-gdb` command, then load the crash dump file above using `target` command on the prompt.
}
```
(cuda-gdb) target cudacore /var/lib/pgdata/core_1521131828_magro.heterodb.com_216238.nvcudmp
Opening GPU coredump: /var/lib/pgdata/core_1521131828_magro.heterodb.com_216238.nvcudmp
[New Thread 216240]

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7ff4dc82f930 (cuda_gpujoin.h:1159)
[Current focus set to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]
#0  0x00007ff4dc82f938 in _INTERNAL_8_pg_strom_0124cb94::gpujoin_exec_hashjoin (kcxt=0x7ff4f7fffbf8, kgjoin=0x7fe9f4800078,
    kmrels=0x7fe9f8800000, kds_src=0x7fe9f0800030, depth=3, rd_stack=0x7fe9f4806118, wr_stack=0x7fe9f480c118, l_state=0x7ff4f7fffc48,
    matched=0x7ff4f7fffc7c "") at /usr/pgsql-10/share/extension/cuda_gpujoin.h:1159
1159            while (khitem && khitem->hash != hash_value)
```

@ja{
この状態で`bt`コマンドを使用し、問題発生個所へのバックトレースを採取する事ができます。
}
@en{
You can check backtrace where the error happened on GPU kernel using `bt` command.
}
```
(cuda-gdb) bt
#0  0x00007ff4dc82f938 in _INTERNAL_8_pg_strom_0124cb94::gpujoin_exec_hashjoin (kcxt=0x7ff4f7fffbf8, kgjoin=0x7fe9f4800078,
    kmrels=0x7fe9f8800000, kds_src=0x7fe9f0800030, depth=3, rd_stack=0x7fe9f4806118, wr_stack=0x7fe9f480c118, l_state=0x7ff4f7fffc48,
    matched=0x7ff4f7fffc7c "") at /usr/pgsql-10/share/extension/cuda_gpujoin.h:1159
#1  0x00007ff4dc9428f0 in gpujoin_main<<<(30,1,1),(256,1,1)>>> (kgjoin=0x7fe9f4800078, kmrels=0x7fe9f8800000, kds_src=0x7fe9f0800030,
    kds_dst=0x7fe9e8800030, kparams_gpreagg=0x0) at /usr/pgsql-10/share/extension/cuda_gpujoin.h:1347
```
@ja{
より詳細な`cuda-gdb`コマンドの利用法は[CUDA Toolkit Documentation - CUDA-GDB](http://docs.nvidia.com/cuda/cuda-gdb/)を参照してください。
}
@en{
Please check [CUDA Toolkit Documentation - CUDA-GDB](http://docs.nvidia.com/cuda/cuda-gdb/) for more detailed usage of `cuda-gdb` command.
}
