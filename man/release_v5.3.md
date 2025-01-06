@ja:#PG-Strom v5.3リリース
@en:#PG-Strom v5.3 Release

<div style="text-align: right;">PG-Strom Development Team (xx-xxx-2025)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v5.3における主要な変更は点は以下の通りです。

- マルチGPUのPinned Inner Buffer
- Arrow_Fdw 仮想列
- AVG(numeric)、SUM(numeric)の精度改善
- GpuPreAggの最終マージをGPU上で実行可能に
- GPUタスクのスケジューリング改善
- 累積的なバグの修正
}

@en{
Major changes in PG-Strom v5.3 are as follows:

- Partitioned Pinned Inner Buffer
- Arrow_Fdw Virtual Columns
- AVG(numeric), SUM(numeric) accuracy improvement
- GpuPreAgg final merge on GPU device
- Improved GPU-tasks scheduling
- Cumulative bug fixes
}

@ja:##動作環境
@en:##Prerequisites

@ja{
- PostgreSQL v15以降
- CUDA Toolkit 12.2 以降
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 6.0 以降 (Pascal以降; Turing以降を推奨)
}
@en{
- PostgreSQL v15 or later
- CUDA Toolkit 12.2 or later
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 6.0 or later (Pascal at least; Turing or newer is recommended)
}

@ja:##マルチGPUのPinned Inner Buffer
@en:##Multi-GPUs Pinned Inner Buffer











@ja:##Arrow_Fdw 仮想列
@en:##Arrow_Fdw Virtual Columns

@ja{
データをバックアップする際に、そのファイル名の一部がデータの属性を示すように命名規則を設ける運用はしばしば見られます。
例えば、`my_data_2018_tokyo.arrow`というファイル名に、ここには2018年の東京エリアでのデータが格納されているといった意味合いを与える場合です。

Arrow_Fdw外部テーブルでは、`dir`オプションを用いて複数のApache ArrowファイルへのSQLアクセスが可能ですが、そのファイル名の一部になにがしかの意味を持たせている場合、それをヒントとしてArrow_Fdw外部テーブルへのアクセスを高速化する事ができます。

以下に例を示します。
}
@en{
When backing up data, it is common to set up a naming convention so that part of the file name indicates the attributes of the data.
For example, a file name like `my_data_2018_tokyo.arrow` might convey that the data stored here is from the Tokyo area in 2018.

In Arrow_Fdw foreign table, you can use `dir` option to access multiple Apache Arrow files with SQL. If the file name has some meaning, you can use it as a hint to speed up access to Arrow_Fdw foreign table.

Here is an example.
}
```
$ ls /opt/arrow/mydata
f_lineorder_1993_AIR.arrow    f_lineorder_1994_SHIP.arrow   f_lineorder_1996_MAIL.arrow
f_lineorder_1993_FOB.arrow    f_lineorder_1994_TRUCK.arrow  f_lineorder_1996_RAIL.arrow
f_lineorder_1993_MAIL.arrow   f_lineorder_1995_AIR.arrow    f_lineorder_1996_SHIP.arrow
f_lineorder_1993_RAIL.arrow   f_lineorder_1995_FOB.arrow    f_lineorder_1996_TRUCK.arrow
f_lineorder_1993_SHIP.arrow   f_lineorder_1995_MAIL.arrow   f_lineorder_1997_AIR.arrow
f_lineorder_1993_TRUCK.arrow  f_lineorder_1995_RAIL.arrow   f_lineorder_1997_FOB.arrow
f_lineorder_1994_AIR.arrow    f_lineorder_1995_SHIP.arrow   f_lineorder_1997_MAIL.arrow
f_lineorder_1994_FOB.arrow    f_lineorder_1995_TRUCK.arrow  f_lineorder_1997_RAIL.arrow
f_lineorder_1994_MAIL.arrow   f_lineorder_1996_AIR.arrow    f_lineorder_1997_SHIP.arrow
f_lineorder_1994_RAIL.arrow   f_lineorder_1996_FOB.arrow    f_lineorder_1997_TRUCK.arrow

postgres=# IMPORT FOREIGN SCHEMA f_lineorder FROM SERVER arrow_fdw INTO public
           OPTIONS (dir '/opt/arrow/mydata', pattern 'f_lineorder_@{year}_${shipmode}.arrow');
IMPORT FOREIGN SCHEMA
postgres=# \d f_lineorder
                             Foreign table "public.f_lineorder"
       Column       |     Type      | Collation | Nullable | Default |     FDW options
--------------------+---------------+-----------+----------+---------+----------------------
 lo_orderkey        | numeric       |           |          |         |
 lo_linenumber      | integer       |           |          |         |
 lo_custkey         | numeric       |           |          |         |
 lo_partkey         | integer       |           |          |         |
 lo_suppkey         | numeric       |           |          |         |
 lo_orderdate       | integer       |           |          |         |
 lo_orderpriority   | character(15) |           |          |         |
 lo_shippriority    | character(1)  |           |          |         |
 lo_quantity        | numeric       |           |          |         |
 lo_extendedprice   | numeric       |           |          |         |
 lo_ordertotalprice | numeric       |           |          |         |
 lo_discount        | numeric       |           |          |         |
 lo_revenue         | numeric       |           |          |         |
 lo_supplycost      | numeric       |           |          |         |
 lo_tax             | numeric       |           |          |         |
 lo_commit_date     | character(8)  |           |          |         |
 lo_shipmode        | character(10) |           |          |         |
 year               | bigint        |           |          |         | (virtual 'year')
 shipmode           | text          |           |          |         | (virtual 'shipmode')
Server: arrow_fdw
FDW options: (dir '/opt/arrow/mydata', pattern 'f_lineorder_@{year}_${shipmode}.arrow')
```

@ja{
このシステムでは、`/opt/arrow/mydata`ディレクトリに、SSBM(Star Schema Benchmark)のlineorderテーブルから出力したデータが`lo_ordate`の年次、および`lo_shipmode`の値ごとにまとめて保存されています。つまり、`f_lineorder_1995_RAIL.arrow`ファイルから読み出した値は、必ず`lo_orderdate`の値が19950101～19951231の範囲内に収まっているという事です。

そうすると、このファイル名の一部に埋め込まれた"年次"の値を用いれば、明らかにマッチする行が存在しないApache Arrowファイルを読み飛ばし、クエリの応答速度を高速化できるのではないかというアイデアが生まれてきます。それがArrow_Fdwの仮想列機能で、この例では列オプション`virtual`を持つ`year`列および`shipmode`列が仮想列に相当します。
}
@en{
In this system, data output from the lineorder table of SSBM (Star Schema Benchmark) is stored in the `/opt/arrow/mydata` directory by year of `lo_ordate` and value of `lo_shipmode`. In other words, the value read from the `lo_orderdate` field in the `f_lineorder_1995_RAIL.arrow` file is always contains the value larger than or equal to 19950101 and less than or equal to 19951231.

This gives rise to the idea that by using the "year" value embedded in the file name, it may be possible to skip Apache Arrow files that clearly do not have matching rows, thereby speeding up the response time of queries. This is the virtual column feature of Arrow_Fdw, and in this example, the `year` column and `shipmode` column with the column option `virtual` correspond to the virtual columns.
}
@ja{
実際にはディレクトリ`/opt/arrow/mydata`配下のApache Arrowファイルにこれらの列は存在しませんが、外部表オプション`pattern`で指定された文字列の中で`@ {year}`や`$ {shipmode}`はワイルドカードとして作用します。列オプション`virtual`で`year`や`shipmode`という名前を指定すると、指定したワイルドカードにマッチするファイル名の一部分があたかもその仮想列の値であるかのように振舞います。

例えば、`f_lineorder_1995_RAIL.arrow`に由来する行の仮想列`year`の値は`1995`となりますし、仮想列`shipmode`の値は`'RAIL'`となります。
}
@en{
Although these columns do not actually exist in the Apache Arrow files under the directory `/opt/arrow/mydata`, `@ {year}` and `${shipmode}` act as wildcards in the string specified in the foreign table option `pattern`. When `year` and `shipmode` are specified in the `virtual` column options, the part of the file name that matches the specified wildcard will behave as if it were the value of that virtual column.

For example, the value of the virtual column `year` in the row derived from `f_lineorder_1995_RAIL.arrow` will be `1995`, and the value of the virtual column `shipmode` will be `'RAIL'`.
}
@ja{
この性質を利用してクエリを最適化する事ができます。

以下の例は、この外部テーブルに対して、1993年のデータおよびいくつかの条件を付加して集計値を検索するものです。
物理的にApache Arrowファイルに存在する列`lo_orderdate`の値に範囲条件`between 19930101 and 19931231`を付加したものと比べ、仮想列`year`と`1993`を比較するよう検索条件を調整したものは、EXPLAIN ANALYZEの出力のうち`Stats-Hint:`によれば全60個のRecord-Batchのうち48個を読み飛ばし、実際には1/5の12個しか処理していない（しかし結果は同じである）事がわかります。

全てのケースにおいて利用できる最適化ではありませんが、データ配置や命名規則、ワークロードによっては利用する事のできる手軽な最適化と言えるでしょう。
}
@en{
You can take advantage of this property to optimize your queries.

The following example searches for aggregate values on this foreign table by searching data from 1993 and some additional conditions.
Compared to adding the range condition `between 19930101 and 19931231` to the value of the column `lo_orderdate` that physically exists in the Apache Arrow file, the search condition adjusted to compare the virtual columns `year` and `1993` skips 48 of the total 60 Record-Batches according to `Stats-Hint:` in the EXPLAIN ANALYZE output, and actually processes only 12 Record-Batches (1/5, but the result is the same).
}

@ja:***仮想列による最適化なし***
@en:***without virtual-column optimization***
```
=# explain analyze
   select sum(lo_extendedprice*lo_discount) as revenue
     from f_lineorder
    where lo_orderdate between 19930101 and 19931231
      and lo_discount between 1 and 3
      and lo_quantity < 25;
                                        QUERY PLAN
------------------------------------------------------------------------------------------
 Aggregate  (cost=463921.94..463921.95 rows=1 width=32) (actual time=175.826..175.828 rows=1 loops=1)
   ->  Custom Scan (GpuPreAgg) on f_lineorder  (cost=463921.92..463921.93 rows=1 width=32)	\
                                               (actual time=175.808..175.811 rows=2 loops=1)
         GPU Projection: pgstrom.psum((lo_extendedprice * lo_discount))
         GPU Scan Quals: ((lo_orderdate >= 19930101) AND (lo_orderdate <= 19931231) AND		\
                          (lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND	\
                          (lo_quantity < '25'::numeric)) [plan: 65062080 -> 542, exec: 65062081 -> 1703647]
         GPU Group Key:
         referenced: lo_orderdate, lo_quantity, lo_extendedprice, lo_discount
         file0: /opt/arrow/mydata/f_lineorder_1996_MAIL.arrow (read: 107.83MB, size: 427.16MB)
         file1: /opt/arrow/mydata/f_lineorder_1996_SHIP.arrow (read: 107.82MB, size: 427.13MB)
                                   :
         file29: /opt/arrow/mydata/f_lineorder_1993_TRUCK.arrow (read: 107.51MB, size: 425.91MB)
         GPU-Direct SQL: enabled (N=2,GPU0,1; direct=413081, ntuples=65062081)
 Planning Time: 0.769 ms
 Execution Time: 176.390 ms
(39 rows)

=# select sum(lo_extendedprice*lo_discount) as revenue
from f_lineorder
where lo_orderdate between 19930101 and 19931231
and lo_discount between 1 and 3
and lo_quantity < 25;
    revenue
---------------
 6385711057885
(1 row)
```

@ja:***仮想列による最適化あり***
@en:***with virtual-column optimization***
```
=# explain analyze
   select sum(lo_extendedprice*lo_discount) as revenue
     from f_lineorder
    where year = 1993
      and lo_discount between 1 and 3
      and lo_quantity < 25;
                                        QUERY PLAN
------------------------------------------------------------------------------------------
 Aggregate  (cost=421986.99..421987.00 rows=1 width=32) (actual time=54.624..54.625 rows=1 loops=1)
   ->  Custom Scan (GpuPreAgg) on f_lineorder  (cost=421986.97..421986.98 rows=1 width=32)	\
                                               (actual time=54.616..54.618 rows=2 loops=1)
         GPU Projection: pgstrom.psum((lo_extendedprice * lo_discount))
         GPU Scan Quals: ((year = 1993) AND (lo_discount <= '3'::numeric) AND (lo_quantity < '25'::numeric) \
                      AND (lo_discount >= '1'::numeric)) [plan: 65062080 -> 542, exec:13010375 -> 1703647]
         GPU Group Key:
         referenced: lo_quantity, lo_extendedprice, lo_discount, year
         Stats-Hint: (year = 1993)  [loaded: 12, skipped: 48]
         file0: /opt/arrow/mydata/f_lineorder_1996_MAIL.arrow (read: 99.53MB, size: 427.16MB)
         file1: /opt/arrow/mydata/f_lineorder_1996_SHIP.arrow (read: 99.52MB, size: 427.13MB)

         file29: /opt/arrow/mydata/f_lineorder_1993_TRUCK.arrow (read: 99.24MB, size: 425.91MB)
         GPU-Direct SQL: enabled (N=2,GPU0,1; direct=76245, ntuples=13010375)
 Planning Time: 0.640 ms
 Execution Time: 55.078 ms
(40 rows)

=# select sum(lo_extendedprice*lo_discount) as revenue
from f_lineorder
where year = 1993
and lo_discount between 1 and 3
and lo_quantity < 25;
    revenue
---------------
 6385711057885
(1 row)
```

@ja:##AVG(numeric)、SUM(numeric)の精度改善
@en:##AVG(numeric), SUM(numeric) accuracy improvement

@ja{
GPUのAtomic演算に関する制限（浮動小数点型の<code>atomicAdd()</code>は64bitまで対応）から、これまでnumeric型の集計処理は浮動小数点型（float8）へのキャストを行って実装されてきました。
しかしこの場合、わずか53bitしか仮数部を持たない倍精度型浮動小数点データを数百万回も加算するという事になり、計算誤差の蓄積がかなり酷いものになるという課題がありました。そのため、numeric型集計関数のGPUでの実行を抑止するというオプションをわざわざ用意したほどです。（`pg_strom.enable_numeric_aggfuncs`オプション）

v5.3では、numericデータ型のGPU内部実装である128bit固定小数点表現を利用して集計処理を行うよう改良されました。計算誤差が全く発生しなくなったわけではありませんが、実数を表現する桁数が増えた分、計算誤差はかなりマイルドになっています。
}
@en{
Due to limitations on atomic operations on GPUs (floating-point type <code>atomicAdd()</code> only supports up to 64 bits), numeric aggregation has previously been implemented with values tranformed to 64bit floating-point type (float8).
However, in this case, double-precision floating-point data, which has a mantissa of only 53 bits, is added millions of times, posing the issue of severe accumulation of calculation errors. For this reason, we have gone to the trouble of providing an option to prevent numeric aggregation functions from being executed on the GPU. (`pg_strom.enable_numeric_aggfuncs` option)

In v5.3, the calculation process has been improved to use the 128-bit fixed-point representation, which is the GPU internal implementation of the numeric data type. This does not mean that calculation errors have completely disappeared, but the calculation errors are much milder due to the increased number of digits used to represent real numbers.
}

```
### by CPU(PostgreSQL)

postgres=# select count(*), sum(x) from t;
  count   |              sum
----------+-------------------------------
 10000000 | 5000502773.174181378779819237
(1 row)

### by GPU(PG-Strom v5.2)

postgres=# select count(*), sum(x) from t;
  count   |       sum
----------+------------------
 10000000 | 5022247013.24539
(1 row)

postgres=# select count(*), sum(x) from t;
  count   |       sum
----------+------------------
 10000000 | 5011118562.96062
(1 row)

### by GPU(PG-Strom v5.3)

postgres=# select count(*), sum(x) from t;
  count   |             sum
----------+-----------------------------
 10000000 | 5000502773.1741813787793780
(1 row)

postgres=# select count(*), sum(x) from t;
  count   |             sum
----------+-----------------------------
 10000000 | 5000502773.1741813787793780
(1 row)
```

@ja:##GpuPreAggの最終マージをGPU上で実行可能に
@en:##GpuPreAgg final merge on GPU device

@ja{
集約演算を並列に処理する場合、これはPostgreSQLにおけるCPU並列処理でも同様ですが、部分的な集約処理を先に行い、その中間結果を最後にマージするという方法を取ります。例えばXの平均値AVG(X)を計算する場合、Xの出現回数とXの総和があれば平均値を計算できるわけですので、ワーカープロセス毎に"部分的な"Xの出現回数とXの総和を計算し、それを最後にシングルプロセスで集計します。

この方式は、巨大なテーブルからごく少数のカテゴリ毎の集計値を計算する、つまりカーディナリティの低い処理であれば非常に効果的に働きます。一方で、カーディナリティの低い、例えば単純な重複排除クエリのようにグループ数の多いタイプのワークロードであれば、部分的な集計処理のほかに、シングルCPUスレッドでの最終マージ処理を行うため、並列処理の効果は限定的になりがちです。
}
@en{
When processing an aggregate operation in parallel, as is the case with CPU parallel processing in PostgreSQL, partial aggregation processing is performed first, then the intermediate results are merged at the end. For example, when calculating the average value of X, the average (AVG(X)) can be calculated using the number of occurrences of X and the total sum of X, so each worker process calculates the "partial" number of occurrences of X and the total sum of X, and then aggregates them in a single process at the end.

This method works very effectively for low cardinality processing, such as calculating aggregate values for a small number of categories from a huge table. On the other hand, for the workloads with low cardinality and a large number of groups, such as simple duplicate removal queries, the effect of parallel processing tends to be limited because partial aggregation processing and final merge processing are performed in a single CPU thread.
}

```
### by CPU (PostgreSQL)

=# EXPLAIN SELECT t1.cat, count(*) cnt, sum(a)
             FROM t0 JOIN t1 ON t0.cat = t1.cat
         GROUP BY t1.cat;
                                          QUERY PLAN
----------------------------------------------------------------------------------------------
 Finalize HashAggregate  (cost=193413.59..193413.89 rows=30 width=20)
   Group Key: t1.cat
   ->  Gather  (cost=193406.84..193413.14 rows=60 width=20)
         Workers Planned: 2
         ->  Partial HashAggregate  (cost=192406.84..192407.14 rows=30 width=20)
               Group Key: t1.cat
               ->  Hash Join  (cost=1.68..161799.20 rows=4081019 width=12)
                     Hash Cond: (t0.cat = t1.cat)
                     ->  Parallel Seq Scan on t0  (cost=0.00..105362.15 rows=4166715 width=4)
                     ->  Hash  (cost=1.30..1.30 rows=30 width=12)
                           ->  Seq Scan on t1  (cost=0.00..1.30 rows=30 width=12)
(11 rows)
```

@ja{
これまでのPG-Stromでも同じ問題を抱えていました。
つまり、並列ワーカープロセスの制御下で動作するGpuPreAggは"部分的な"集計処理は行うものの、これらのCPU並列プロセスで処理された"部分的な"集計結果は、最終的にシングルスレッドのCPUで動作する最終マージ処理を挟まねばならないというものです。

以下の例では、GpuPreAggの結果を（並列ワーカープロセスを制御する）Gatherノードが受け取り、それを（シングルスレッドのCPUで動作する）HashAggregateが受け取る実行計画になっており、グループ数が多くなるとワークロードに占めるシングルCPUスレッドで処理する部分の割合が無視できないものになってきます。
}
@en{
Previous versions of PG-Strom had the same problem.
That is, although GpuPreAgg, which runs under the control of parallel worker processes, performs "partial" aggregation processing, the "partial" aggregation results processed by these CPU parallel processes must be merged in the end by a single-threaded CPU.

In the example below, the execution plan is such that the Gather node (which controls the parallel worker processes) receives the results of GpuPreAgg, and then HashAggregate (which runs on a single-threaded CPU) receives them. As the number of groups increases, the proportion of the workload processed by a single CPU thread becomes significant.
}

```
### by GPU (PG-Strom v5.2)

=# EXPLAIN SELECT t1.cat, count(*) cnt, sum(a)
             FROM t0 JOIN t1 ON t0.cat = t1.cat
         GROUP BY t1.cat;
                                           QUERY PLAN
------------------------------------------------------------------------------------------------
 HashAggregate  (cost=30100.15..30100.53 rows=30 width=20)
   Group Key: t1.cat
   ->  Gather  (cost=30096.63..30099.93 rows=30 width=44)
         Workers Planned: 2
         ->  Parallel Custom Scan (GpuPreAgg) on t0  (cost=29096.63..29096.93 rows=30 width=44)
               GPU Projection: pgstrom.nrows(), pgstrom.psum(t1.a), t1.cat
               GPU Join Quals [1]: (t0.cat = t1.cat) ... [nrows: 4166715 -> 4081019]
               GPU Outer Hash [1]: t0.cat
               GPU Inner Hash [1]: t1.cat
               GPU Group Key: t1.cat
               GPU-Direct SQL: enabled (N=2,GPU0,1)
               ->  Parallel Seq Scan on t1  (cost=0.00..1.18 rows=18 width=12)
(12 rows)
```

@ja{
GpuPreAggがGPU上で構築する集計結果が"部分的な"ものでなければ、CPU側で再度の集計処理を行う必要はありません。
その際に一つだけ問題になるのが、CPU-Fallback処理です。可変長データが外部テーブルに格納されている（TOAST可）などが原因で、一部の行をCPUで処理した場合、GPUメモリ上の結果だけでは整合性のある集計結果を出力する事ができません。

しかし、現実の集計処理においてCPU-Fallbackが発生するパターンはそれほど多くはありません。そのため、PG-Strom v5.3では、CPU-Fallback処理が無効化されている場合にはGPUデバイスメモリ上で最終マージ処理まで実行し、CPU側での集計処理を省略するモードを搭載しました。

以下の実行計画の例では、Gatherノードの配下にGpuPreAggが配置されていますが、これまでのようにそれを最終的にマージするHashAggregateは組み込まれていません。GpuPreAggが整合性のある結果を返すため、CPU側で追加の集計処理は必要ないのです。
}
@en{
If the aggregation results that GpuPreAgg constructs on the GPU are not "partial", there is no need to perform the aggregation process again on the CPU side.
The only problem that can arise in this case is CPU-Fallback processing. If some rows are processed by the CPU due to some reasons such as variable-length data being stored in an external table (TOAST is possible), it is not possible to output a consistent aggregation result using only the results in GPU memory.

However, in the real world, there are not many cases where CPU-Fallback occurs. Therefore, PG-Strom v5.3 has a mode that performs the final merge process on the GPU device memory when CPU-Fallback is disabled, and omits the CPU-side aggregation process.

In the example execution plan below, GpuPreAgg is placed under the Gather node, but HashAggregate is not included to finally merge it as in the previous example. Because GpuPreAgg returns a consistent result, no additional aggregation processing is required on the CPU side.
}

```
### by GPU (PG-Strom v5.3)

=# set pg_strom.cpu_fallback = off;
SET
=# EXPLAIN SELECT t1.cat, count(*) cnt, sum(a)
             FROM t0 JOIN t1 ON t0.cat = t1.cat
         GROUP BY t1.cat;
                                           QUERY PLAN
------------------------------------------------------------------------------------------------
 Gather  (cost=30096.63..30100.45 rows=30 width=20)
   Workers Planned: 2
   ->  Result  (cost=29096.63..29097.45 rows=30 width=20)
         ->  Parallel Custom Scan (GpuPreAgg) on t0  (cost=29096.63..29096.93 rows=30 width=44)
               GPU Projection: pgstrom.nrows(), pgstrom.psum(t1.a), t1.cat
               GPU Join Quals [1]: (t0.cat = t1.cat) ... [nrows: 4166715 -> 4081019]
               GPU Outer Hash [1]: t0.cat
               GPU Inner Hash [1]: t1.cat
               GPU Group Key: t1.cat
               GPU-Direct SQL: enabled (N=2,GPU0,1)
               ->  Parallel Seq Scan on t1  (cost=0.00..1.18 rows=18 width=12)
(11 rows)
```

@ja:##その他の新機能
@en:##Other New Features
@ja{
- GPUスケジュールの改善
    - 従来はPostgreSQLバックエンドプロセスに対して1個のGPUを固定で割当て、マルチGPUを利用にはPostgreSQLのパラレルクエリが前提となっていました。この設計はv3.x以前のアーキテクチャに由来するもので、実装をシンプルにするメリットの一方、マルチGPUのスケジューリングが難しいという課題がありました。
    - v5.3ではGPU-Serviceが全てのGPUタスクのスケジューリングを担います。マルチGPUの環境においては、PostgreSQLバックエンドから受け取ったGPUタスクの要求は、スケジュール可能なGPUのうち、その時点でキューイングされているタスク数が最も小さなGPUに割り当てられるようになりました。
    - これにより、より自然な形でマルチGPUの処理能力を利用できるようになりました。
- CUDA Stack Limit Checker
    - 再帰処理を含むGPUコードにおいて、スタック領域の使い過ぎをチェックするロジックが入りました。
    - これにより、複雑なパターンを含むLIKE句などで、予期せぬスタックの使い過ぎによるGPUカーネルのクラッシュを抑止する事が期待されます。
- GPUでのRIGHT OUTER JOIN処理
    - これまではCPU-Fallbackの機構を使用して実装されていたRIGHT OUTER JOINが、GPU上で実行されるようになりました。
- GPU-Direct SQL Decision Making Logs
    - いくつかの環境でGPU-Direct SQLが発動しない理由を探るためのヒントを得られるようになりました。
    - 環境変数 `HETERODB_EXTRA_EREPORT_LEVEL=1` をセットしてPostgreSQLを起動する事で有効化されます。
- `pgstrom.arrow_fdw_metadata_info()`でApache Arrowファイルのメタデータを参照できるようになりました。
- `column IN (X,Y,Z,...)`の演算子でArrow_FdwのMIN/MAX統計情報を参照するようになりました。
- ワークロードに対するGPU割り当てポリシーを`optimal`、`numa`、`system`の3通りから選べるようになりました。
    - `optimal`は従来通り、PCI-Eバス上でストレージとGPUの距離が最も近いもの。
    - `numa`は同一CPUの配下に接続されているストレージとGPUをペアにする事で、QPI跨ぎのデータ転送を抑止します。
    - `system`はシステムで有効なGPUを全てスケジュール可能とします。
}
@en{
- Improved GPU-tasks scheduling
    - In the previous version, one GPU was assigned to a PostgreSQL backend process, and the use of multiple GPUs was premised on the use of PostgreSQL parallel queries. This design originated from the architecture of v3.x, and while it had the advantage of simplifying the implementation, it also had the problem of making multi-GPU scheduling difficult.
    - In v5.3, the GPU-Service is responsible for all GPU task scheduling. In a multi-GPU environment, a GPU task request received from the PostgreSQL backend is assigned to the schedulable GPU with the smallest number of tasks currently queued.
    - This makes possible to utilize the processing power of multiple GPUs in a more natural way.
- CUDA Stack Limit Checker
    - A logic to check for excessive stack usage in GPU code has been added, in the recursive CUDA functions.
    - This is expected to prevent GPU kernel crashes caused by unexpected excessive stack usage, for example in LIKE clauses that include complex patterns.
- RIGHT OUTER JOIN processing on GPU
    - RIGHT OUTER JOIN, which was previously implemented using a CPU-Fallback mechanism, is now executed on the GPU.
- GPU-Direct SQL Decision Making Logs
    - We got some hints on why GPU-Direct SQL does not work in some system environments.
    - It is enabled using environment variable `HETERODB_EXTRA_EREPORT_LEVEL=1` before starting PostgreSQL server process.
- `pgstrom.arrow_fdw_metadata_info()` allows to reference the metadata of Apache Arrow files.
- `column IN (X,Y,Z,...)` operator now refers MIN/MAX statistics of Arrow_Fdw.
- GPU assignment policy is now configurable from: `optimal`, `numa`, and `system`.
    - `optimal` is the same as before, where the storage and GPU are closest on the PCI-E bus.
    - `numa` pairs storage and GPU connected under the same CPU, preventing data transfer across QPI.
    - `system` allows scheduling of all GPUs available in the system.

}

@ja:##累積的なバグの修正
@en:##Cumulative bug fixes

- [#864] arrow_fdw: metadata cache refactoring for custom key-value displaying
- [#860] bugfix: MIN() MAX() returned empty result if nitems is multiple of 2^32
- [#856] add fallback by managed-memory if raw-gpu-memory exceeds the hard limit
- [#852] wrong template deployment of move_XXXX_value() callback
- [#847] bugfix: max(float) used wrong operator
- [#844] postgis: st_crosses() should return false for identical linestring geometry
- [#831] arrow-fdw: incorrect record-batch index calculation
- [#829] bugfix: GpuScan considered inheritance-table as if it is normal table
- [#829] bugfix: pgstrom_partial_nrows() didn't return correct value if '0' is given
- [#827] bugfix: RIGHT OUTER tuple didn't check 'other_quals' in 'join_quals'
- [#825] arrowFieldTypeIsEqual didn't work for FixedSizeBinary
- [#824] pg2arrow: moved empty results check to the end of file creation.
- [#820] bugfix: CPU-fallback ExprState was not initialized correctly
- [#820] additional fix on CPU-fallback expression references
- [#819] bugfix: a bunch of rows were skipped after GPU kernel suspend/resume
- [#817] GPU-Service didn't detach worker thread's exit status.
- [#812] CPU-fallback at depth>0 didn't set ecxt_scanslot correctly.
- [#812] bugfix: pgfn_st_crosses() returned uninitialized results
- [#812] fix wrong CPU fallback at GiST-Join
- [#811] add delay to apply pg_strom.max_async_tasks
- [#809][#810] Documentation fix
- [#808] pg2arrow: put_decimal_value() handles numeric with negative weight incorrectly.
- [#805] CREATE OPERATOR makes a pseudo commutor/negator operators in the default namespace
- [#743][#838] nvcc working directory is moved to $PGDATA/.pgstrom_fatbin_build_XXXXXX
- [#xxx] pg2arrow: raise an error if numeric value is Nan, +Inf or -Inf
- [#xxx] bugfix: CPU-fallback handling of system columns
- [#xxx] bugfix: cuMemcpyPeer() caused SEGV when # of threads > 20
- [#xxx] bugfix: scan_block_count was not initialized on the DSM

