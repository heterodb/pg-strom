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
-------------------------------------------------------------------------------------------------
 HashAggregate  (cost=2924539.01..2924612.42 rows=5873 width=46)
   Group Key: date1.d_year, part.p_brand1
   ->  Custom Scan (GpuPreAgg) on lineorder  (cost=2924421.55..2924494.96 rows=5873 width=46)
         GPU Projection: pgstrom.psum(lo_revenue), d_year, p_brand1
         GPU Join Quals [1]: (lo_partkey = p_partkey) [plan: 600046000 -> 783060 ]
         GPU Outer Hash [1]: lo_partkey
         GPU Inner Hash [1]: p_partkey
         GPU Join Quals [2]: (lo_suppkey = s_suppkey) [plan: 783060 -> 157695 ]
         GPU Outer Hash [2]: lo_suppkey
         GPU Inner Hash [2]: s_suppkey
         GPU Join Quals [3]: (lo_orderdate = d_datekey) [plan: 157695 -> 157695 ]
         GPU Outer Hash [3]: lo_orderdate
         GPU Inner Hash [3]: d_datekey
         GPU Group Key: d_year, p_brand1
         Scan-Engine: GPU-Direct with 2 GPUs <0,1>
         ->  Seq Scan on part  (cost=0.00..41481.00 rows=1827 width=14)
               Filter: ((p_brand1 >= 'MFGR#2221'::bpchar) AND (p_brand1 <= 'MFGR#2228'::bpchar))
         ->  Custom Scan (GpuScan) on supplier  (cost=100.00..19001.67 rows=203767 width=6)
               GPU Projection: s_suppkey
               GPU Scan Quals: (s_region = 'ASIA'::bpchar) [plan: 1000000 -> 203767]
               Scan-Engine: GPU-Direct with 2 GPUs <0,1>
         ->  Seq Scan on date1  (cost=0.00..72.56 rows=2556 width=8)
(22 rows)
```
@ja{
実行計画の中に見慣れない処理が含まれている事に気が付かれたでしょう。
CustomScan機構を用いてGpuJoinおよびGpuPreAggが実装されています。ここでGpuJoinは`lineorder`と`date1`、`part`および`supplier`とのJOIN処理を実行し、その結果を受け取るGpuPreAggは列`d_year`と`p_brand1`によるGROUP BY処理をGPUで実行します。
}
@en{
You can notice some unusual query execution plans.
GpuJoin and GpuPreAgg are implemented on the CustomScan mechanism. In this example, GpuJoin runs JOIN operation on `lineorder`, `date1`, `part` and `supplier`, then GpuPreAgg which receives the result of GpuJoin runs GROUP BY operation by the `d_year` and `p_brand1` column on GPU device.
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
ただし、GPUを利用するためにはバックグラウンドで動作するPG-Strom GPU Serviceに接続してセッション毎の初期化が必要になりますので、常にCPU並列度が高ければ良いというわけではありません。
}
@en{
PG-Strom also supports PostgreSQL's CPU parallel execution.

PostgreSQL's CPU parallel execution is implemented by a Gather node starting several background worker processes, and later combining the results of queries that each background worker "partially" executed.
PG-Strom processes such as GpuJoin and GpuPreAgg can be executed on the background worker side, and each process uses the GPU to perform processing. Normally, the processing speed of each CPU core to set up a buffer to supply data to the GPU is much slower than the processing speed of SQL workloads on the GPU, so a hybrid of CPU parallelism and GPU parallelism can be expected to improve processing speed.


In the CPU parallel execution mode, Gather node launches several background worker processes, then it gathers the result of "partial" execution by individual background workers.
CustomScan execution plan provided by PG-Strom, like GpuJoin or GpuPreAgg, support execution at the background workers. They process their partial task using GPU individually. A CPU core usually needs much more time to set up buffer to supply data for GPU than execution of SQL workloads on GPU, so hybrid usage of CPU and GPU parallel can expect higher performance.
On the other hands, each PostgreSQL process needs to connect the PG-Strom GPU service, running in the background, and initialize the per-session stete, so a high degree of CPU parallelism is not always better.
}

@ja{
以下の実行計画を見てください。
Gather以下の実行計画はバックグラウンドワーカーが実行可能なものです。6億行を保持する`lineorder`テーブルを2プロセスのバックグラウンドワーカとコーディネータプロセスでスキャンするため、プロセスあたり約2億行をGpuPreAggで処理し、その結果をGatherおよびHashAggregateノードで結合します。
}
@en{
Look at the execution plan below.

The execution plan below Gather can be executed by background workers. The `lineorder` table, which holds 600 million rows, is scanned by two background worker processes and the coordinator process, so approximately 200 million rows per process are processed by GpuPreAgg, and the results are joined by Gather and HashAggregate nodes.
}
```
=# explain
select sum(lo_revenue), d_year, p_brand1
  from lineorder, date1, part, supplier
  where lo_orderdate = d_datekey
    and lo_partkey = p_partkey
    and lo_suppkey = s_suppkey
    and p_brand1 between
           'MFGR#2221' and 'MFGR#2228'
    and s_region = 'ASIA'
  group by d_year, p_brand1;
                                                 QUERY PLAN
-------------------------------------------------------------------------------------------------------------
 HashAggregate  (cost=1265644.05..1265717.46 rows=5873 width=46)
   Group Key: date1.d_year, part.p_brand1
   ->  Gather  (cost=1264982.11..1265600.00 rows=5873 width=46)
         Workers Planned: 2
         ->  Parallel Custom Scan (GpuPreAgg) on lineorder  (cost=1263982.11..1264012.70 rows=5873 width=46)
               GPU Projection: pgstrom.psum(lo_revenue), d_year, p_brand1
               GPU Join Quals [1]: (lo_partkey = p_partkey) [plan: 250019100 -> 326275 ]
               GPU Outer Hash [1]: lo_partkey
               GPU Inner Hash [1]: p_partkey
               GPU Join Quals [2]: (lo_suppkey = s_suppkey) [plan: 326275 -> 65706 ]
               GPU Outer Hash [2]: lo_suppkey
               GPU Inner Hash [2]: s_suppkey
               GPU Join Quals [3]: (lo_orderdate = d_datekey) [plan: 65706 -> 65706 ]
               GPU Outer Hash [3]: lo_orderdate
               GPU Inner Hash [3]: d_datekey
               GPU Group Key: d_year, p_brand1
               Scan-Engine: GPU-Direct with 2 GPUs <0,1>
               ->  Parallel Seq Scan on part  (cost=0.00..29231.00 rows=761 width=14)
                     Filter: ((p_brand1 >= 'MFGR#2221'::bpchar) AND (p_brand1 <= 'MFGR#2228'::bpchar))
               ->  Parallel Custom Scan (GpuScan) on supplier  (cost=100.00..8002.40 rows=84903 width=6)
                     GPU Projection: s_suppkey
                     GPU Scan Quals: (s_region = 'ASIA'::bpchar) [plan: 1000000 -> 84903]
                     Scan-Engine: GPU-Direct with 2 GPUs <0,1>
               ->  Parallel Seq Scan on date1  (cost=0.00..62.04 rows=1504 width=8)
(24 rows)
```

@ja:##下位プランの統合
@en:##Consolidation of sub-plans

@ja{
PG-StromはSCAN、JOIN、GROUP BY、SORTの各処理をGPUで実行する事が可能ですが、これに対応するPostgreSQL標準の処理を単純に置き換えただけでは困った事態が発生します。
SCANが終わった後のデータをいったんホスト側のバッファに書き戻し、次にそれをJOINするために再びGPUへとコピーし、さらにGROUP BYを実行する前に再びホスト側のバッファに書き戻し・・・といった形で、CPUとGPUの間でデータのピンポンが発生してしまいます。
}
@en{
PG-Strom can execute SCAN, JOIN, GROUP BY, and SORT processes on the GPU. However, if you simply replace the corresponding standard PostgreSQL processes with GPU processes, you will encounter problems.
After SCAN, the data is written back to the host buffer, then copied back to the GPU for JOIN, and then written back to the host buffer again before executing GROUP BY, resulting in data ping-pong between the CPU and GPU.
}
@ja{
CPUのメモリ上でデータ（行）を交換するのと比較して、CPUとGPUの間はPCI-Eバスで結ばれているため、どうしてもデータ転送には大きなコストが発生してしまいます。これを避けるには、SCAN、JOIN、GROUP BY、SORTといった一連のGPU対応タスクが連続して実行可能である場合には、できる限りGPUメモリ上でデータ交換を行い、CPUへデータを書き戻すのは最小限に留めるべきであるという事です。
}
@en{
Compared to exchanging data (rows) in CPU memory, the CPU and GPU are connected by a PCI-E bus, so data transfer inevitably incurs a large cost. To avoid this, when a series of GPU-compatible tasks such as SCAN, JOIN, GROUP BY, and SORT can be executed consecutively, data should be exchanged in GPU memory as much as possible and writing data back to the CPU should be minimized.
}
![combined gpu kernel](./img/combined-kernel-overview.png)

@ja{
以下の実行計画は、SCAN、JOIN、GROUP BYの複合ワークロードをPostgreSQLで実行する場合のものです。
最もサイズの大きな`lineorder`テーブルを軸に、`part`、`supplier`、`date1`の各テーブルをHashJoinを用いて結合し、最後に集計処理を行うAggregateが登場している事が分かります。
}
@en{
The following execution plan is for a mixed workload of SCAN, JOIN, and GROUP BY executed on PostgreSQL.
You can see that the `lineorder` table, which is the largest, is used as the axis to join the `part`, `supplier`, and `date1` tables using HashJoin, and finally an Aggregate is used to perform the aggregation process.
}
```
=# explain
select sum(lo_revenue), d_year, p_brand1
  from lineorder, date1, part, supplier
  where lo_orderdate = d_datekey
    and lo_partkey = p_partkey
    and lo_suppkey = s_suppkey
    and p_brand1 between
           'MFGR#2221' and 'MFGR#2228'
    and s_region = 'ASIA'
  group by d_year, p_brand1;
                                                          QUERY PLAN
-------------------------------------------------------------------------------------------------------------------------------
 Finalize HashAggregate  (cost=14892768.98..14892842.39 rows=5873 width=46)
   Group Key: date1.d_year, part.p_brand1
   ->  Gather  (cost=14891403.50..14892651.52 rows=11746 width=46)
         Workers Planned: 2
         ->  Partial HashAggregate  (cost=14890403.50..14890476.92 rows=5873 width=46)
               Group Key: date1.d_year, part.p_brand1
               ->  Hash Join  (cost=52477.64..14889910.71 rows=65706 width=20)
                     Hash Cond: (lineorder.lo_orderdate = date1.d_datekey)
                     ->  Parallel Hash Join  (cost=52373.13..14888902.74 rows=65706 width=20)
                           Hash Cond: (lineorder.lo_suppkey = supplier.s_suppkey)
                           ->  Parallel Hash Join  (cost=29240.51..14864272.81 rows=326275 width=26)
                                 Hash Cond: (lineorder.lo_partkey = part.p_partkey)
                                 ->  Parallel Seq Scan on lineorder  (cost=0.00..13896101.47 rows=250019147 width=20)
                                 ->  Parallel Hash  (cost=29231.00..29231.00 rows=761 width=14)
                                       ->  Parallel Seq Scan on part  (cost=0.00..29231.00 rows=761 width=14)
                                             Filter: ((p_brand1 >= 'MFGR#2221'::bpchar) AND (p_brand1 <= 'MFGR#2228'::bpchar))
                           ->  Parallel Hash  (cost=22071.33..22071.33 rows=84903 width=6)
                                 ->  Parallel Seq Scan on supplier  (cost=0.00..22071.33 rows=84903 width=6)
                                       Filter: (s_region = 'ASIA'::bpchar)
                     ->  Hash  (cost=72.56..72.56 rows=2556 width=8)
                           ->  Seq Scan on date1  (cost=0.00..72.56 rows=2556 width=8)
(21 rows)
```

@ja{
一方、PG-Stromを用いた場合はずいぶんと様子が異なります。
結果の射影処理を行うResultノードの除けば、全ての処理がCustom Scan (GpuPreAgg)で実行されています。
（※なお、この実行計画では、結果を最大限にシンプルにするため、CPU並列とCPU-Fallbackは無効化しています）
}
@en{
On the other hand, when using PG-Strom, the situation is quite different.
Except for the Result node which projects the results, all processing is executed by Custom Scan (GpuPreAgg).
(Note: In this execution plan, CPU parallelism and CPU-Fallback are disabled to keep the results as simple as possible.)
}
@ja{
しかしGPU-PreAggとはいえ、この処理はGROUP BYだけを行っている訳ではありません。
EXPLAINの出力に付随する各種のパラメータを読むと、このGPU-PreAggは最もサイズの大きな`lineorder`テーブルをスキャンしつつ、下位ノードで`part`、`supplier`、`date1`テーブルを読み出してこれとJOIN処理を行います。そして`d_year`と`p_brand1`によるグループ化そ行った上で、同じキーによるソート処理を行った上で、処理結果をCPUに戻しています。
}
@en{
However, even though it is GPU-PreAgg, this processing does not only perform GROUP BY.
Reading the various parameters accompanying the EXPLAIN output, we can see that this GPU-PreAgg scans the `lineorder` table, which is the largest, while reading the `part`, `supplier`, and `date1` tables at the lower nodes and performing JOIN processing with these. It then groups by `d_year` and `p_brand1`, sorts by the same keys, and returns the processing results to the CPU.
}
@ja{
PostgreSQLにおいては、複雑なクエリはある程度多くの要素に分解され、数多くの処理ステップを含む実行計画が生成される事が多くなります。
一方、PG-Stromにおいてもこれらの要素は抜け漏れなく実行されるのですが、できる限り一個のプランに統合された形になっている方が、基本的には効率の良い実行計画であると言えます。
}
@en{
In PostgreSQL, complex queries are often broken down into many elements, and an execution plan containing many processing steps is generated.
On the other hand, in PG-Strom, these elements are also executed without any omissions, but it is generally more efficient to integrate them into a single plan as much as possible.
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
     group by d_year, p_brand1
     order by d_year, p_brand1;
                                           QUERY PLAN
-------------------------------------------------------------------------------------------------
 Result  (cost=3111326.30..3111451.10 rows=5873 width=46)
   ->  Custom Scan (GpuPreAgg) on lineorder  (cost=3111326.30..3111363.01 rows=5873 width=46)
         GPU Projection: pgstrom.psum(lo_revenue), d_year, p_brand1
         GPU Join Quals [1]: (lo_partkey = p_partkey) [plan: 600046000 -> 783060 ]
         GPU Outer Hash [1]: lo_partkey
         GPU Inner Hash [1]: p_partkey
         GPU Join Quals [2]: (lo_suppkey = s_suppkey) [plan: 783060 -> 157695 ]
         GPU Outer Hash [2]: lo_suppkey
         GPU Inner Hash [2]: s_suppkey
         GPU Join Quals [3]: (lo_orderdate = d_datekey) [plan: 157695 -> 157695 ]
         GPU Outer Hash [3]: lo_orderdate
         GPU Inner Hash [3]: d_datekey
         GPU Group Key: d_year, p_brand1
         Scan-Engine: GPU-Direct with 2 GPUs <0,1>
         GPU-Sort keys: d_year, p_brand1
         ->  Seq Scan on part  (cost=0.00..41481.00 rows=1827 width=14)
               Filter: ((p_brand1 >= 'MFGR#2221'::bpchar) AND (p_brand1 <= 'MFGR#2228'::bpchar))
         ->  Custom Scan (GpuScan) on supplier  (cost=100.00..19156.92 rows=203767 width=6)
               GPU Projection: s_suppkey
               GPU Scan Quals: (s_region = 'ASIA'::bpchar) [plan: 1000000 -> 203767]
               Scan-Engine: GPU-Direct with 2 GPUs <0,1>
         ->  Seq Scan on date1  (cost=0.00..72.56 rows=2556 width=8)
(22 rows)
```

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













