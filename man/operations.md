@ja:#基本的な操作
@en:#Basic operations

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
詳細な一覧に関しては[リファレンス](ref_devfuncs.md)を参照してください。
}
@en{
For GPU execution, it requires operators, functions and data types in use must be supported by PG-Strom.
It supports numeric types like `int` or `float`, date and time types like `date` or `timestamp`, variable length string like `text` and so on. It also supports arithmetic operations, comparison operators and many built-in operators.
See [References](ref_devfuncs.md) for the detailed list.
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

@ja{※本節の内容は有効ではありません。最新の実装を踏まえた書き直しが必要です。}
@en{(*) This section does not follow the latest version, and need to rewrite according to the latest implementation.}

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

@ja:##GpuJoinにおけるInner Pinned Buffer
@en:##Inner Pinned Buffer of GpuJoin

@ja{
以下の実行計画を見てください。
PG-Stromがテーブルを結合する際、通常は最もサイズの大きなテーブル（この場合は`lineorder`で、OUTER表と呼びます）を非同期的に読み込みながら、他のテーブルとの結合処理および集計処理を進めます。
JOINアルゴリズムの制約上、予めそれ以外のテーブル（この場合は`date1`、`part`、`supplier`で、INNER表と呼びます）をメモリ上に読み出し、またJOINキーのハッシュ値を計算する必要があります。これらのテーブルはOUTER表ほど大きなサイズではないものの、数GBを越えるようなINNERバッファの準備は相応に重い処理となります。
}
@en{
Look at the EXPLAIN output below.
When PG-Strom joins tables, it usually reads the largest table (`lineorder` in this case; called the OUTER table) asynchronously, while performing join processing and aggregation processing with other tables. Let's proceed.
Due to the constraints of the JOIN algorithm, it is necessary to read other tables (`date1`, `part`, `supplier` in this case; called the INNER tables) into memory in advance, and also calculate the hash value of the JOIN key. Although these tables are not as large as the OUTER table, preparing an INNER buffer that exceeds several GB is a heavy process.
}

@ja{
GpuJoinは通常、PostgreSQLのAPIを通してINNER表を一行ごとに読み出し、そのハッシュ値を計算するとともに共有メモリ上のINNERバッファに書き込みます。GPU-Serviceプロセスは、このINNERバッファをGPUメモリに転送し、そこではじめてOUTER表を読み出してJOIN処理を開始する事ができるようになります。
INNER表が相応に大きくGPUで実行可能な検索条件を含む場合、以下の実行計画のように、GpuJoinの配下にGpuScanが存在するケースがあり得ます。この場合、INNER表はいったんGpuScanによってGPUで処理された後、その実行結果をCPU側に戻し、さらにINNERバッファに書き込まれた後でもう一度GPUへロードされます。ずいぶんと無駄なデータの流れが存在するように見えます。
}
@en{
GpuJoin usually reads the INNER table through the PostgreSQL API row-by-row, calculates its hash value, and writes them to the INNER buffer on the host shared memory. The GPU-Service process transfers this INNER buffer onto the GPU device memory, then we can start reading the OUTER table and processing the JOIN with inner tables.
If the INNER table is relatively large and contains search conditions that are executable on the GPU, GpuScan may exists under GpuJoin, as in the EXPLAIN output below. In this case, the INNER table is once processed on the GPU by GpuScan, the execution results are returned to the CPU, and then written to the INNER buffer before it is loaded onto the GPU again. It looks like there is quite a bit of wasted data flow.
}

```
=# explain
   select sum(lo_revenue), d_year, p_brand1
     from lineorder, date1, part, supplier
    where lo_orderdate = d_datekey
      and lo_partkey = p_partkey
      and lo_suppkey = s_suppkey
      and p_brand1 between 'MFGR#2221' and 'MFGR#2228'
      and s_region = 'ASIA'
    group by d_year, p_brand1;
                                                  QUERY PLAN
---------------------------------------------------------------------------------------------------------------
 GroupAggregate  (cost=31007186.70..31023043.21 rows=6482 width=46)
   Group Key: date1.d_year, part.p_brand1
   ->  Sort  (cost=31007186.70..31011130.57 rows=1577548 width=20)
         Sort Key: date1.d_year, part.p_brand1
         ->  Custom Scan (GpuJoin) on lineorder  (cost=275086.19..30844784.03 rows=1577548 width=20)
               GPU Projection: date1.d_year, part.p_brand1, lineorder.lo_revenue
               GPU Join Quals [1]: (part.p_partkey = lineorder.lo_partkey) ... [nrows: 5994236000 -> 7804495]
               GPU Outer Hash [1]: lineorder.lo_partkey
               GPU Inner Hash [1]: part.p_partkey
               GPU Join Quals [2]: (supplier.s_suppkey = lineorder.lo_suppkey) ... [nrows: 7804495 -> 1577548]
               GPU Outer Hash [2]: lineorder.lo_suppkey
               GPU Inner Hash [2]: supplier.s_suppkey
               GPU Join Quals [3]: (date1.d_datekey = lineorder.lo_orderdate) ... [nrows: 1577548 -> 1577548]
               GPU Outer Hash [3]: lineorder.lo_orderdate
               GPU Inner Hash [3]: date1.d_datekey
               GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on part  (cost=0.00..59258.00 rows=2604 width=14)
                     Filter: ((p_brand1 >= 'MFGR#2221'::bpchar) AND (p_brand1 <= 'MFGR#2228'::bpchar))
               ->  Custom Scan (GpuScan) on supplier  (cost=100.00..190348.83 rows=2019384 width=6)
                     GPU Projection: s_suppkey
                     GPU Pinned Buffer: enabled
                     GPU Scan Quals: (s_region = 'ASIA'::bpchar) [rows: 9990357 -> 2019384]
                     GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on date1  (cost=0.00..72.56 rows=2556 width=8)
(24 rows)
```

@ja{
このように、INNER表の読出しやINNERバッファの構築の際にCPUとGPUの間でデータのピンポンが発生する場合、***Pinned Inner Buffer***を使用するよう設定する事で、GpuJoinの実行開始リードタイムの短縮や、メモリ使用量を削減する事ができます。
上の実行計画では、`supplier`表の読出しがGpuScanにより行われる事になっており、統計情報によれば約200万行が読み出されると推定されています。その一方で、`GPU Pinned Buffer: enabled`の出力に注目してください。これは、INNER表の推定サイズが`pg_strom.pinned_inner_buffer_threshold`の設定値を越える場合、GpuScanの処理結果をそのままGPUメモリに残しておき、それを次のGpuJoinでINNERバッファの一部として利用するという機能です（必要であればハッシュ値の計算もGPUで行います）。
そのため、`supplier`表の内容はGPU-Direct SQLによってストレージからGPUへと読み出された後、CPU側に戻されたり、再度GPUへロードされたりすることなく、次のGpuJoinで利用される事になります。
}
@en{
In this way, if data ping-pong occurs between the CPU and GPU when reading the INNER table or building the INNER buffer, you can configure GPUJoin to use ***Pinned Inner Buffer***. It is possible to shorten the execution start lead time and reduce memory usage.
In the above EXPLAIN output, reading of the `supplier` table will be performed by GpuScan, and according to the statistical information, it is estimated that about 2 million rows will be read from the table. Meanwhile, notice the output of `GPU Pinned Buffer: enabled`. This is a function that if the estimated size of the INNER table exceeds the configuration value of `pg_strom.pinned_inner_buffer_threshold`, the processing result of GpuScan is retained in the GPU memory and used as part of the INNER buffer at the next GpuJoin. (If necessary, hash value calculation is also performed on the GPU).
Therefore, after the contents of the `supplier` table are read from storage to the GPU using GPU-Direct SQL, they can be used in the next GPUJoin without being returned to the CPU or loaded to the GPU again. It will be.
}

@ja{
一方でPinned Inner Bufferの使用には若干のトレードオフもあるため、デフォルトでは無効化されています。
本機能を使用する場合には、明示的に`pg_strom.pinned_inner_buffer_threshold`パラメータを設定する必要があります。

Pinned Inner Bufferを使用した場合、CPU側はINNERバッファの内容を完全には保持していません。そのため、TOAST化された可変長データをGPUで参照した場合など、CPU Fallback処理を行う事ができずエラーを発生させます。また、CPU Fallbackを利用して実装されているRIGHT/FULL OUTER JOINも同様の理由でPinned Inner Bufferと共存する事ができません。
}
@en{
However, there are some tradeoffs to using Pinned Inner Buffer, so it is disabled by default.
When using this feature, you must explicitly set the `pg_strom.pinned_inner_buffer_threshold` parameter.

The CPU side does not completely retain the contents of the INNER buffer when Pinned Inner Buffer is in use. Therefore, CPU fallback processing cannot be performed and an error will be raised. Also, RIGHT/FULL OUTER JOIN, which is implemented using CPU Fallback, cannot coexist with Pinned Inner Buffer for the same reason.
}

@ja:##ナレッジベース
@en:##Knowledge base

@ja{
PG-Stromプロジェクトのwikiサイトには、ノートと呼ばれる詳細な技術情報が公開されています。
}
@en{
We publish several articles, just called "notes", on the project wiki-site of PG-Strom.
}
[https://github.com/heterodb/pg-strom/wiki](https://github.com/heterodb/pg-strom/wiki)













