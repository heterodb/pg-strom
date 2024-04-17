@ja:#PG-Strom v5.1リリース
@en:#PG-Strom v5.1 Release

<div style="text-align: right;">PG-Strom Development Team (17-Apr-2024)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v5.1における主要な変更は点は以下の通りです。

- パーティションに対応したGpuJoin/PreAggのサポートを追加しました。
- GPUコードのビルドを起動時に実行環境で行うようになりました。
- pg2arrowが並列処理に対応しました。
- CUDA Stackのサイズを適応的に設定するようになりました。
- 累積的なバグの修正
}

@en{
Major changes in PG-Strom v5.1 are as follows:

- Added support for partition-wise GPU-Join/PreAgg.
- GPU code is now built in the execution environment at startup.
- pg2arrow now support parallel execution.
- CUDA Stack size is now set adaptically.
- Cumulative bug fixes
}

@ja:##動作環境
@en:##Prerequisites

@ja{
- PostgreSQL v15.x, v16.x
- CUDA Toolkit 12.2 以降
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 6.0 以降 (Pascal以降; Volta以降を推奨)
}
@en{
- PostgreSQL v15.x, v16.x
- CUDA Toolkit 12.2 or later
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 6.0 or later (Pascal at least; Volta or newer is recommended)
}

@ja:##パーティション対応
@en:##Partition-wise GpuJoin/GpuPreAgg

@ja{
PostgreSQLパーティションへの対応自体はPG-Strom v3.0でも行われていましたが、うまく実行計画を作成できない事が多く、実験的ステータスを脱する事のできないものでした。そこで、PG-Strom v5.1では内部の設計を根本的に見直して再度実装し、再び正式な機能として取り入れました。

以下の`lineorder`テーブルがパーティション化されており、`date1`テーブルが非パーティション化テーブルである場合、これまでは、`lineorder`配下のパーティションテーブルから読み出したデータを全て`Append`ノードによって結合された後でなければJOINする事ができませんでした。
通常、PG-StromはCPUをバイパスしてNVME-SSDからGPUへデータをロードして各種のSQL処理を行う（GPU-Direct SQL）ため、JOINに先立ってCPU側へデータを戻さねばならないというのは大きなペナルティです。
}
@en{
Support for PostgreSQL partitions itself was also included in PG-Strom v3.0, but execution plans often could not be created properly, therefore it could not be moved out of its experimental status.
Then, in PG-Strom v5.1, we fundamentally revised the internal design, re-implemented it, and incorporated it as an official feature again.

If the `lineorder` table below is partitioned and the `date1` table is a non-partitioned table, previously all the data read from the partitioned tables under `lineorder` must be joined with `date1` table after the consolidation of all the partition leafs by the `Append` node.

Usually, PG-Strom bypasses the CPU and loads data from the NVME-SSD to the GPU to perform various SQL processing (GPU-Direct SQL), so the data must be returned to the CPU before JOIN. It has been a big penalty.
}


```
ssbm=# explain (costs off)
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,date1
where lo_orderdate = d_datekey
and d_year = 1993
and lo_discount between 1 and 3
and lo_quantity < 25;
                                                                              QUERY PLAN
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Aggregate
   ->  Hash Join
         Hash Cond: (lineorder.lo_orderdate = date1.d_datekey)
         ->  Append
               ->  Custom Scan (GpuScan) on lineorder__p1992 lineorder_1
                     GPU Projection: lo_extendedprice, lo_discount, lo_orderdate
                     GPU Scan Quals: ((lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND (lo_quantity < '25'::numeric)) [rows: 91250920 -> 11911380]
                     GPU-Direct SQL: enabled (GPU-0)
               ->  Custom Scan (GpuScan) on lineorder__p1993 lineorder_2
                     GPU Projection: lo_extendedprice, lo_discount, lo_orderdate
                     GPU Scan Quals: ((lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND (lo_quantity < '25'::numeric)) [rows: 91008500 -> 11980460]
                     GPU-Direct SQL: enabled (GPU-0)
               ->  Custom Scan (GpuScan) on lineorder__p1994 lineorder_3
                     GPU Projection: lo_extendedprice, lo_discount, lo_orderdate
                     GPU Scan Quals: ((lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND (lo_quantity < '25'::numeric)) [rows: 91044060 -> 12150700]
                     GPU-Direct SQL: enabled (GPU-0)
               ->  Custom Scan (GpuScan) on lineorder__p1995 lineorder_4
                     GPU Projection: lo_extendedprice, lo_discount, lo_orderdate
                     GPU Scan Quals: ((lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND (lo_quantity < '25'::numeric)) [rows: 91011720 -> 11779920]
                     GPU-Direct SQL: enabled (GPU-0)
               ->  Custom Scan (GpuScan) on lineorder__p1996 lineorder_5
                     GPU Projection: lo_extendedprice, lo_discount, lo_orderdate
                     GPU Scan Quals: ((lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND (lo_quantity < '25'::numeric)) [rows: 91305650 -> 11942810]
                     GPU-Direct SQL: enabled (GPU-0)
               ->  Custom Scan (GpuScan) on lineorder__p1997 lineorder_6
                     GPU Projection: lo_extendedprice, lo_discount, lo_orderdate
                     GPU Scan Quals: ((lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND (lo_quantity < '25'::numeric)) [rows: 91049100 -> 12069740]
                     GPU-Direct SQL: enabled (GPU-0)
               ->  Custom Scan (GpuScan) on lineorder__p1998 lineorder_7
                     GPU Projection: lo_extendedprice, lo_discount, lo_orderdate
                     GPU Scan Quals: ((lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND (lo_quantity < '25'::numeric)) [rows: 53370560 -> 6898138]
                     GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on lineorder__p1999 lineorder_8
                     Filter: ((lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND (lo_quantity < '25'::numeric))
         ->  Hash
               ->  Seq Scan on date1
                     Filter: (d_year = 1993)
(37 rows)
```

@ja{
PG-Strom v5.1では、非パーティションテーブルをプッシュダウンし、パーティション子テーブルから読み出したデータとJOINしてから結果を返す事ができるようになりました。
場合によってはGROUP-BY処理まで済ませた上でCPU側に戻す事もでき、例えば以下の例では、総数6億件のパーティション子テーブルから検索条件を満たす7千万行を返さねばならないところ、非パーティションテーブルである`date1`とのJOINと、その次に実行する集約関数`SUM()`をパーティション子テーブルにプッシュダウンする事で、CPUでは僅か8行を処理するだけで済んでいます。

INNER側の読出しが複数回発生するというデメリットはありますが（※将来のバージョンで改修される予定です）、このような書き換えによってCPUで処理すべきデータが大幅に減少し、処理速度の改善に寄与します。
}
@en{

In PG-Strom v5.1, it is now possible to push-down JOINs with non-partitioned tables to partitioned child tables. In some cases, it is also possible to complete the GROUP-BY processing and then return much smaller results to CPU.
For example, in the example below, 70 million rows extracted from a total of 600 million rows in the partitioned child tables. By performing a JOIN with the non-partitioned table `date1` and then aggregation function `SUM()` pushed-down to the partitioned child tables, the CPU only needs to process 8 rows.

Although there is a disadvantage that reading on the INNER side occurs multiple times (* This will be fixed in a future version), this type of rewriting will significantly reduce the amount of data that must be processed by the CPU, contributing to improved processing speed. To do.
}

```
ssbm=# explain (costs off)
select sum(lo_extendedprice*lo_discount) as revenue
from lineorder,date1
where lo_orderdate = d_datekey
and d_year = 1993
and lo_discount between 1 and 3
and lo_quantity < 25;
                                               QUERY PLAN
----------------------------------------------------------------------------------------------------
 Aggregate
   ->  Append
         ->  Custom Scan (GpuPreAgg) on lineorder__p1992 lineorder_1
               GPU Projection: pgstrom.psum(((lineorder_1.lo_extendedprice * lineorder_1.lo_discount))::double precision)
               GPU Scan Quals: ((lineorder_1.lo_discount >= '1'::numeric) AND (lineorder_1.lo_discount <= '3'::numeric) AND (lineorder_1.lo_quantity < '25'::numeric)) [rows: 91250920 -> 11911380]
               GPU Join Quals [1]: (lineorder_1.lo_orderdate = date1.d_datekey) ... [nrows: 11911380 -> 1700960]
               GPU Outer Hash [1]: lineorder_1.lo_orderdate
               GPU Inner Hash [1]: date1.d_datekey
               Inner Siblings-Id: 2
               GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on date1
                     Filter: (d_year = 1993)
         ->  Custom Scan (GpuPreAgg) on lineorder__p1993 lineorder_2
               GPU Projection: pgstrom.psum(((lineorder_2.lo_extendedprice * lineorder_2.lo_discount))::double precision)
               GPU Scan Quals: ((lineorder_2.lo_discount >= '1'::numeric) AND (lineorder_2.lo_discount <= '3'::numeric) AND (lineorder_2.lo_quantity < '25'::numeric)) [rows: 91008500 -> 11980460]
               GPU Join Quals [1]: (lineorder_2.lo_orderdate = date1.d_datekey) ... [nrows: 11980460 -> 1710824]
               GPU Outer Hash [1]: lineorder_2.lo_orderdate
               GPU Inner Hash [1]: date1.d_datekey
               Inner Siblings-Id: 2
               GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on date1
                     Filter: (d_year = 1993)
         ->  Custom Scan (GpuPreAgg) on lineorder__p1994 lineorder_3
               GPU Projection: pgstrom.psum(((lineorder_3.lo_extendedprice * lineorder_3.lo_discount))::double precision)
               GPU Scan Quals: ((lineorder_3.lo_discount >= '1'::numeric) AND (lineorder_3.lo_discount <= '3'::numeric) AND (lineorder_3.lo_quantity < '25'::numeric)) [rows: 91044060 -> 12150700]
               GPU Join Quals [1]: (lineorder_3.lo_orderdate = date1.d_datekey) ... [nrows: 12150700 -> 1735135]
               GPU Outer Hash [1]: lineorder_3.lo_orderdate
               GPU Inner Hash [1]: date1.d_datekey
               Inner Siblings-Id: 2
               GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on date1
                     Filter: (d_year = 1993)
         ->  Custom Scan (GpuPreAgg) on lineorder__p1995 lineorder_4
               GPU Projection: pgstrom.psum(((lineorder_4.lo_extendedprice * lineorder_4.lo_discount))::double precision)
               GPU Scan Quals: ((lineorder_4.lo_discount >= '1'::numeric) AND (lineorder_4.lo_discount <= '3'::numeric) AND (lineorder_4.lo_quantity < '25'::numeric)) [rows: 91011720 -> 11779920]
               GPU Join Quals [1]: (lineorder_4.lo_orderdate = date1.d_datekey) ... [nrows: 11779920 -> 1682188]
               GPU Outer Hash [1]: lineorder_4.lo_orderdate
               GPU Inner Hash [1]: date1.d_datekey
               Inner Siblings-Id: 2
               GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on date1
                     Filter: (d_year = 1993)
         ->  Custom Scan (GpuPreAgg) on lineorder__p1996 lineorder_5
               GPU Projection: pgstrom.psum(((lineorder_5.lo_extendedprice * lineorder_5.lo_discount))::double precision)
               GPU Scan Quals: ((lineorder_5.lo_discount >= '1'::numeric) AND (lineorder_5.lo_discount <= '3'::numeric) AND (lineorder_5.lo_quantity < '25'::numeric)) [rows: 91305650 -> 11942810]
               GPU Join Quals [1]: (lineorder_5.lo_orderdate = date1.d_datekey) ... [nrows: 11942810 -> 1705448]
               GPU Outer Hash [1]: lineorder_5.lo_orderdate
               GPU Inner Hash [1]: date1.d_datekey
               Inner Siblings-Id: 2
               GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on date1
                     Filter: (d_year = 1993)
         ->  Custom Scan (GpuPreAgg) on lineorder__p1997 lineorder_6
               GPU Projection: pgstrom.psum(((lineorder_6.lo_extendedprice * lineorder_6.lo_discount))::double precision)
               GPU Scan Quals: ((lineorder_6.lo_discount >= '1'::numeric) AND (lineorder_6.lo_discount <= '3'::numeric) AND (lineorder_6.lo_quantity < '25'::numeric)) [rows: 91049100 -> 12069740]
               GPU Join Quals [1]: (lineorder_6.lo_orderdate = date1.d_datekey) ... [nrows: 12069740 -> 1723574]
               GPU Outer Hash [1]: lineorder_6.lo_orderdate
               GPU Inner Hash [1]: date1.d_datekey
               Inner Siblings-Id: 2
               GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on date1
                     Filter: (d_year = 1993)
         ->  Custom Scan (GpuPreAgg) on lineorder__p1998 lineorder_7
               GPU Projection: pgstrom.psum(((lineorder_7.lo_extendedprice * lineorder_7.lo_discount))::double precision)
               GPU Scan Quals: ((lineorder_7.lo_discount >= '1'::numeric) AND (lineorder_7.lo_discount <= '3'::numeric) AND (lineorder_7.lo_quantity < '25'::numeric)) [rows: 53370560 -> 6898138]
               GPU Join Quals [1]: (lineorder_7.lo_orderdate = date1.d_datekey) ... [nrows: 6898138 -> 985063]
               GPU Outer Hash [1]: lineorder_7.lo_orderdate
               GPU Inner Hash [1]: date1.d_datekey
               Inner Siblings-Id: 2
               GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on date1
                     Filter: (d_year = 1993)
         ->  Custom Scan (GpuPreAgg) on lineorder__p1999 lineorder_8
               GPU Projection: pgstrom.psum(((lineorder_8.lo_extendedprice * lineorder_8.lo_discount))::double precision)
               GPU Scan Quals: ((lineorder_8.lo_discount >= '1'::numeric) AND (lineorder_8.lo_discount <= '3'::numeric) AND (lineorder_8.lo_quantity < '25'::numeric)) [rows: 150 -> 1]
               GPU Join Quals [1]: (lineorder_8.lo_orderdate = date1.d_datekey) ... [nrows: 1 -> 1]
               GPU Outer Hash [1]: lineorder_8.lo_orderdate
               GPU Inner Hash [1]: date1.d_datekey
               Inner Siblings-Id: 2
               GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on date1
                     Filter: (d_year = 1993)
(82 rows)
```


@ja:##GPUコードの起動時ビルド
@en:##Build GPU code on startup

@ja{
以前のバージョンのPG-Stromでは、予めビルドされたGPU向けのバイナリモジュールを配布する方式をとっていました。
これはシンプルではあるのですが、PG-Strom（PostgreSQL）実行環境のCUDA ToolkitやNVIDIAドライバのバージョンの組合せによっては、GPUバイナリモジュールを認識できず実行時エラーを起こしてしまう事がありました。典型的には、RPMパッケージをビルドした環境よりも古いバージョンのCUDA ToolkitやNVIDIAドライバが実行環境にインストールされている場合です。

PG-Strom v5.1では、起動時にGPU用のソースコードやCUDA Toolkitのバージョンを確認し、差分があればGPU向けバイナリモジュールをビルドするように変更されました。この修正により、PG-Stromは実行環境にインストールされたGPUデバイス、およびCUDA Toolkit向けのGPUバイナリモジュールを利用することができるようになりました。

一部のPG-Strom用GPUモジュールにはビルドに時間がかかるものがあります。そのため、PG-StromやCUDA Toolkitのバージョンアップ後、初回の起動時にはPG-Stromの機能が利用可能となるまで数分の時間がかかる場合があります。
}
@en{
Previous versions of PG-Strom was distributing pre-built binary modules for GPUs.
Although this is simple, the GPU binary module often raised a runtime error depending on the combination of CUDA Toolkit and NVIDIA driver versions in the PG-Strom (PostgreSQL) execution environment.
Typically, this is when the execution environment has an older version of the CUDA Toolkit or NVIDIA driver installed than the environment in which the RPM package was built.

PG-Strom v5.1 has been changed to check the GPU source code and CUDA Toolkit version at startup, and build a GPU binary module if there are any difference. With this fix, PG-Strom can now utilize GPU devices and GPU binary modules for CUDA Toolkit in the execution environment.
}

@ja:##pg2arrowの並列実行
@en:##pg2arrow parallel execution

@ja{
`pg2arrow`は新たに`-n|--num-workers`オプションと`-k|--parallel-keys`オプションをサポートするようになりました。

`-n N_WORKERS`は指定した数のスレッドがそれぞれPostgreSQLに接続し、並列にクエリを実行した結果をApache Arrowファイルに書き込みます。クエリには特殊な文字列`$(N_WORKERS)`と`$(WORKER_ID)`を含む事ができ、これらはPostgreSQLにクエリを投げる際に、それぞれワーカー数とワーカー固有のID値に置き換えられます。ユーザはこれを利用して、各ワーカースレッドが読み出すタプルが互いに重複したり欠損したりしないように調整する必要があります。

もう一つの`-k|--parallel-key`オプションは、引数で与えたカンマ区切りのキー値のそれぞれに対してワーカースレッドを起動し、クエリ中の`$(PARALLEL_KEY)`をキーと置き換えた上で、これをPostgreSQLで実行した結果をApache Arrowファイルとして書き込みます。
例えば、`lineorder`テーブルがパーティション化されており、子テーブルとして、`lineorder__sun`, `lineorder__mon`, ... `lineorder__sat` が存在した場合、個々のワーカースレッドがパーティションの子テーブルをそれぞれスキャンするといった形で処理を並列化できます。
この場合、`-k`オプションは`-k sun,mon,tue,wed,thu,fri,sat`と指定し、`-c`オプションには`SELECT * FROM lineorder__$(PARALLEL_KEY)`と指定すれば、7個のワーカースレッドがそれぞれパーティションの子テーブルをスキャンする事になります。
}
@en{
`pg2arrow` now supports the new `-n|--num-workers` and `-k|--parallel-keys` options.

`-n N_WORKERS` option launches the specified number of threads to connect to PostgreSQL for each, execute queries in parallel, and write the results to the Apache Arrow file.
Queries can contain the special token `$(N_WORKERS)` and `$(WORKER_ID)`, which will be replaced by the number of workers and worker-specific ID values, respectively, when querying PostgreSQL.
It is user's responsibility to ensure the tuples read by each worker thread do not overlap or are missing.

Another `-k|--parallel-key` option starts a worker thread for each comma-separated key value given by the argument, and replaces `$(PARALLEL_KEY)` with the key in the query.
The worker thread runs this query for each, then write the result as an Apache Arrow file.

For example, if the `lineorder` table is partitioned and there are child tables `lineorder__sun`, `lineorder__mon`, ... `lineorder__sat`, each worker thread scans each child table of the partition according to the keys given by the `-k sun,mon,tue,wed,thu,fri,sat` option. This parallel key is replaced by the `$(PARALLEL_KEY)` in the query given by `-c 'SELECT * FROM lineorder__$(PARALLEL_KEY)'` option. It launches 7 worker threads which shall scan the partitioned child table for each.
}

```
$ pg2arrow -d ssbm -c 'SELECT * FROM lineorder__$(PARALLEL_KEY)' -o /path/to/f_lineorder.arrow -k=sun,mon,tue,wed,thu,fri,sat --progress
worker:1 SQL=[SELECT * FROM lineorder__sun]
worker:3 SQL=[SELECT * FROM lineorder__tue]
worker:2 SQL=[SELECT * FROM lineorder__mon]
worker:4 SQL=[SELECT * FROM lineorder__wed]
worker:5 SQL=[SELECT * FROM lineorder__thu]
   :
   :
```

