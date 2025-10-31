@ja:#PG-Strom v6.1リリース
@en:#PG-Strom v6.1 Release

<div style="text-align: right;">PG-Strom Development Team (30th-Oct-2025)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v6.1における主要な変更は点は以下の通りです。

- Apache Parquetファイルの対応
- libarrow/libparquetの利用（独自実装からの移行）
- SELECT INTO Directモード
- GPUメモリフットプリントの削減
- 累積的なバグの修正
}

@en{
Major changes in PG-Strom v6.1 are as follows:

- Support of Apache Parquet files.
- Migration to libarrow/libparquet, from own implementation.
- SELECT INTO Direct mode
- Reduction of GPU memory footprint.
- Cumulative bug fixes
}

@ja:##動作環境
@en:##Prerequisites

@ja{
- PostgreSQL v15以降
- CUDA Toolkit 13 以降
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 7.5 以降 (Turing以降)
}
@en{
- PostgreSQL v15 or later
- CUDA Toolkit 13 or later
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 7.5 or later (Turing or newer)
}

@ja:##Apache Parquetファイルの対応
@en:##Support of Apache Parquet files

@ja{
PG-Strom v6.1ではApache Parquetファイルの読み出しに対応しました。

Arrow_Fdw外部テーブルのオプションでParquetファイルを指定する事で、Arrowファイルと同様に読み出す事ができるようになりました。
複数ファイルを指定する場合、同じスキーマ定義を持つファイルであれば混在する事も可能です。

Apache Parquet形式は、Apache Arrowとよく似た特徴を持つ列指向の構造化データ形式ですが、多様な圧縮オプションを持ち、比較的低速なストレージからデータを読み出す際の帯域がボトルネックとなるケースで有用です。
現在のところ、圧縮データの展開にはCPU上で動作するlibparquetを利用しているためにParquetファイルに対してはGPU-Direct SQLを使用する事ができません。（ストライピング構成のNVME-SSDなど、PCI-Eバスの帯域をフルに活用できる高速なストレージを利用する場合は、引き続きApache Arrowの利用を推奨します）

利用方法は以下の通りです。外部テーブルのオプション`file`や`files`、`dir`でApache Parquetファイルを指定します。
}
@en{
PG-Strom v6.1 now supports reading Apache Parquet files.

Once Parquet files are specified in Arrow_Fdw foreign table options, you can read them in the same way as Arrow files.
As long as they have same schema definition, we can specify multiple files even if Apache Arrow and Parquet files are mixtured.

The Apache Parquet format is a columnar structured data format with similar characteristics to Apache Arrow, but it offers a variety of compression options, making it useful in cases where bandwidth bottlenecks arise when reading data from relatively slow storage.

Currently, GPU-Direct SQL cannot be used with Parquet files because it uses libparquet, which runs on the CPU, to decompress compressed data. (We still recommend using Apache Arrow when using high-speed storage that can fully utilize the PCI-E bus bandwidth, such as striped NVME-SSDs.)

Follow the steps below. Specify the Apache Parquet file using the foreign table options `file`, `files`, or `dir`.
}
```
postgres=# IMPORT FOREIGN SCHEMA weather FROM SERVER arrow_fdw
                            INTO public OPTIONS (file '/tmp/weather.parquet');
IMPORT FOREIGN SCHEMA

postgres=# EXPLAIN SELECT count(*), avg("MinTemp"), avg("MaxTemp") FROM weather WHERE "WindDir9am" like '%N%';
                                     QUERY PLAN
-------------------------------------------------------------------------------------
 Custom Scan (GpuPreAgg) on weather  (cost=100.23..100.25 rows=1 width=72)
   GPU Projection: pgstrom.nrows(), pgstrom.pavg("MinTemp"), pgstrom.pavg("MaxTemp")
   GPU Scan Quals: ("WindDir9am" ~~ '%N%'::text) [plan: 366 -> 366]
   GPU Group Key:
   referenced: "MinTemp", "MaxTemp", "WindDir9am"
   file0: /tmp/weather.parquet (read: 0B, size: 20.51KB)
   Scan-Engine: VFS with GPU0
(7 rows)

postgres=# SELECT count(*), avg("MinTemp"), avg("MaxTemp")
             FROM weather
            WHERE "WindDir9am" like '%N%';
 count |        avg         |        avg
-------+--------------------+--------------------
   164 | 6.6121951219512205 | 19.833536585365856
(1 row)
```

@ja{
同一のスキーマ構造を持つファイルであれば、Apache ArrowとParquetの混在も可能です。
}
@en{
Apache Arrow and Parquet files can be used together as long as the files have the same schema structure.
}
```
postgres=# IMPORT FOREIGN SCHEMA f_ssbm_part FROM SERVER arrow_fdw
                            INTO public OPTIONS (files '/tmp/ssmb_part.arrow,/tmp/ssmb_part.parquet');
IMPORT FOREIGN SCHEMA

postgres=# EXPLAIN ANALYZE
           SELECT count(*), p_color, max(p_size)
             FROM f_ssbm_part
            WHERE p_mfgr LIKE 'MFGR#%'
            GROUP BY p_color;
                                                                  QUERY PLAN
-----------------------------------------------------------------------------------------------------------------------------------------------
 Gather  (cost=4116.93..4142.39 rows=200 width=72) (actual time=173.977..174.230 rows=92 loops=1)
   Workers Planned: 2
   Workers Launched: 2
   ->  Parallel Custom Scan (GpuPreAgg) on f_ssbm_part  (cost=3116.93..3118.39 rows=200 width=72) (actual time=58.952..59.007 rows=31 loops=3)
         GPU Projection: pgstrom.nrows(), pgstrom.pmax((p_size)::double precision), p_color
         GPU Scan Quals: (p_mfgr ~~ 'MFGR#%'::text) [plan: 3600000 -> 7500, exec: 3600000 -> 3600000]
         GPU Group Key: p_color
         referenced: p_mfgr, p_color, p_size
         file0: /tmp/ssmb_part.arrow (read: 54.65MB, size: 176.95MB)
         file1: /tmp/ssmb_part.parquet (read: 0B, size: 17.62MB)
         Scan-Engine: VFS with GPU0; vfs=6996, ntuples=3600000
 Planning Time: 0.276 ms
 Execution Time: 174.388 ms
(13 rows)
```

@ja:###独自実装からlibarrow/libparquetへの移行
@en:###Migration to libarrow/libparquet, from own implementation

@ja{
PG-StromがApache Arrow形式に対応したのは2019年リリースのv2.2ですが、この頃はC言語で利用可能なライブラリの品質がまだ十分ではなく、Apache Arrow形式ファイルの読み出しを独自に実装していました。
その後、数年を経てApache Arrow形式が多くの分野で利用されるようになり、また圧縮データの展開を含むApache Parquet形式にも対応するため、本バージョン以降のPG-Stromではlibarrow/libparquetを利用して実装されています。

周辺コマンドのlibarrow/libparquet対応も順次進めており、以下のモジュールに関しては対応が完了しています。
- Arrow_Fdw (PG-Strom本体)
- pg2arrow
- arrow2csv
- tsv2arrow

以下のモジュールに関しても、順次対応を進めていく予定です。
- pcap2arrow
- vcf2arrow
- fluentd(arrow-file)プラグイン
}
@en{
PG-Strom first supported the Apache Arrow format in v2.2, released in 2019. However, at the time, the quality of the C libraries available was not yet sufficient, so we implemented our own implementation for reading Apache Arrow files.

Over the years, the Apache Arrow format has become widely used, and since PG-Strom now supports the Apache Parquet format, including decompression of compressed data, this version and subsequent versions of PG-Strom use libarrow/libparquet.

We are also gradually working on supporting libarrow/libparquet in peripheral commands, and the following modules have been fully supported:

- Arrow_Fdw (PG-Strom main body)
- pg2arrow
- arrow2csv
- tsv2arrow

We plan to gradually support the following modules:
- pcap2arrow
- vcf2arrow
- fluentd(arrow-file) plugin
}

@ja:##SELECT INTO Directモード
@en:##SELECT INTO Direct mode

@ja{
ETLなどでよく見られるワークロードですが、元となるテーブルをスキャンしその結果を別のテーブルに挿入するケースでは、GPUでのSQL処理の後でPostgreSQLでの処理をバイパスできるケースがあります。本バージョンの新機能であるSELECT INTO Directモードは、そのような場合にGPUでの処理結果をPostgreSQLでの処理をバイパスして直接ストレージに書き込む事で、大量データの書き込みを高速化します。

以下の例は、lineorderテーブル(316GB)をスキャンした結果をCREATE TABLE ASを用いて新しいテーブルへ書き込む例です。
`Gather`や`Custom Scan (GpuScan)`のactual rows=0となっていますが、`SELECT-INTO Direct`が有効で1106653ブロック(8.44GB)文を書き込んだと表示されています。
これはバグではなく、`Custom Scan (GpuScan)`の処理結果をPostgreSQLバックエンドへ返す事なくストレージに書き込んでいるため、Gatherノードを通過した結果行は0件であったという事を意味しています。
}
@en{
In the workload usually seen in ETL, where the source table is scanned and the results are inserted into another table, it is sometimes possible to bypass PostgreSQL processing after GPU SQL processing. SELECT INTO Direct mode, a new feature in this version, accelerates large-scale data writes by writing the GPU-processed results directly to storage, bypassing PostgreSQL processing.

The following example shows how to write the results of a scan of the lineorder table (316 GB) to a new table using CREATE TABLE AS.

Although `Gather` and `Custom Scan (GpuScan)` show actual rows = 0, with `SELECT-INTO Direct` enabled, it shows that 1,106,653 blocks (8.44 GB) were written.

This is not a bug; rather, because the results of `Custom Scan (GpuScan)` were written to storage with no rows returned to the PostgreSQL backend, no result rows passed through the Gather node.
}
```
postgres=# explain analyze create unlogged table lineorder_1995_ship as select * from lineorder where lo_orderdate between 19950101 and 19951231 and lo_shipmode='RAIL';
                                                                                                                               QUERY PLAN
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Gather  (cost=1100.00..9514869.83 rows=48077662 width=103) (actual time=27233.307..27236.331 rows=0 loops=1)
   Workers Planned: 2
   Workers Launched: 2
   ->  Parallel Custom Scan (GpuScan) on lineorder  (cost=100.00..4706103.63 rows=20032359 width=103) (actual time=27182.877..27182.880 rows=0 loops=3)
         GPU Projection: lo_orderkey, lo_linenumber, lo_custkey, lo_partkey, lo_suppkey, lo_orderdate, lo_orderpriority, lo_shippriority, lo_quantity, lo_extendedprice, lo_ordertotalprice, lo_discount, lo_revenue, lo_supplycost, lo_tax, lo_commit_date, lo_shipmode
         GPU Scan Quals: ((lo_orderdate >= 19950101) AND (lo_orderdate <= 19951231) AND (lo_shipmode = 'RAIL'::bpchar)) [plan: 2400012000 -> 20032360, exec: 2400012063 -> 52012672]
         Scan-Engine: GPU-Direct with 2 GPUs <0,1>; direct=41379519, ntuples=2400012063
         SELECT-INTO Direct: enabled, nblocks=1106653 (8.44GB)
 Planning Time: 0.286 ms
 Execution Time: 27247.873 ms
(10 rows)
```

@ja{
SELECT INTO Directモードは以下のような場合に発動します。

- SELECT INTOまたはCREATE TABLE ASでテーブルを新しく作成する場合
    - したがって、トランザクションはACCESS EXCLUSIVEロックを持っている事が保証されます
- テーブルがUNLOGGEDテーブルである場合
    - したがって、トランザクションログを書き込む必要はありません。
- テーブルのアクセスメソッドが`heap`（PostgreSQLの標準）である場合
- `pg_strom.cpu_fallback = off` であること。
    - TOAST化の必要な可変長データが存在する場合、PG-StromはCPU fallbackで処理するため、書き込むべきデータが全てPostgreSQL Blockにインライン格納できるものである事が保証されます。
- `pg_strom.enable_select_into_direct = on`であること。

`pg_strom.enable_select_into_direct`パラメータを使って、意図的にSELECT INTO Directモードを無効化する事もできます。
その場合、以下のように`Custom Scan (GpuScan)`の処理結果を1行ごとに`Gather`が受け取り、最終的に、PostgreSQLのテーブルアクセスメソッドを通じてディスクに書き込みが行われますが、GPU-ServiceからUNIXドメインソケットを介してPostgreSQLバックエンドに処理結果を転送したり、それを1行ごとに取り出してメモリ上でコピー・整形するための処理コストにより、処理時間が伸びています。
}
@en{
SELECT INTO Direct mode is activated in the following cases:

- When a new table is created with SELECT INTO or CREATE TABLE AS
    - Therefore, the transaction is guaranteed to have an ACCESS EXCLUSIVE lock.
- When the table is an UNLOGGED table.
    - Therefore, there is no need to write a transaction log.
- When the table's access method is `heap` (the PostgreSQL standard).
- When `pg_strom.cpu_fallback = off` is set.
    - When variable-length data that requires toasting exists, PG-Strom uses CPU fallback, ensuring that all data to be written can be stored inline in PostgreSQL blocks.
- When `pg_strom.enable_select_into_direct = on` is set.

You can also intentionally disable SELECT INTO Direct mode using the `pg_strom.enable_select_into_direct` parameter.
In this case, `Gather` receives the results of `Custom Scan (GpuScan)` row by row, as shown below, and finally writes them to disk via PostgreSQL's table access method. However, the processing time increases due to the processing costs of transferring the results from GPU-Service to the PostgreSQL backend via a UNIX domain socket, and extracting them row by row, copying and formatting them in memory.
}
```
postgres=# set pg_strom.enable_select_into_direct = off;
SET
postgres=# explain analyze create unlogged table lineorder_1995_ship_normal as select * from lineorder where lo_orderdate between 19950101 and 19951231 and lo_shipmode='RAIL';
                                                                                                                               QUERY PLAN

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Gather  (cost=1100.00..9514869.83 rows=48077662 width=103) (actual time=28.196..9550.290 rows=52012672 loops=1)
   Workers Planned: 2
   Workers Launched: 2
   ->  Parallel Custom Scan (GpuScan) on lineorder  (cost=100.00..4706103.63 rows=20032359 width=103) (actual time=30.414..10766.055 rows=17337557 loops=3)
         GPU Projection: lo_orderkey, lo_linenumber, lo_custkey, lo_partkey, lo_suppkey, lo_orderdate, lo_orderpriority, lo_shippriority, lo_quantity, lo_extendedprice, lo_ordertotalprice, lo_discount, lo_revenue, lo_supplycost, lo_tax, lo_commit_date, lo_shipmode
         GPU Scan Quals: ((lo_orderdate >= 19950101) AND (lo_orderdate <= 19951231) AND (lo_shipmode = 'RAIL'::bpchar)) [plan: 2400012000 -> 20032360, exec: 2400012063 -> 52012672]
         Scan-Engine: GPU-Direct with 2 GPUs <0,1>; direct=41379519, ntuples=2400012063
 Planning Time: 0.577 ms
 Execution Time: 63814.360 ms
(9 rows)
```

@ja:##その他の変更点
@en:##Other Changes

@ja:###GPUメモリフットプリントの削減
@en:###Reduction of GPU memory footprint
@ja{
GPU-Joinで巨大なテーブル同士のJOINを実行する際に、最もシビアな制約条件となるのは、INNER側テーブルをハッシュ表に読み込んだ際にGPUメモリに収まるかどうかという点です。
従来は、PostgreSQLのHeapTupleDataデータ構造を元にハッシュ値やポインタなどを付加したデータ構造を用いており、INNER側テーブル１行あたりペイロード以外に48バイトの領域が必要でした。通常、INNER側テーブルは必要な行だけが読み出されるため、ペイロードよりもヘッダの方がGPUメモリを消費するといった状況も珍しくありませんでした。

本バージョンでは、PostgreSQLのMinimalTupleDataデータ構造を元にして不要なデータを捨象し、また単にパディングとして未使用になっていたエリアにハッシュ値などのデータを詰め込む事で、INNER側テーブル１行あたりペイロード以外の所用データ量を24バイトに圧縮しました。
これはとりわけ大容量のテーブル同士のJOINにおいて有用で、例えば、あるテスト用クエリにおいては従来INNER側テーブルが85GBを要していたGPUメモリ消費を61GBにまで削減する事ができました。（28%削減）

ハードウェア的なGPUメモリサイズを越えない範囲に収めるというのは非常に重要で、例えば、80GBのDRAMを持つNVIDIA H100では85GBのハッシュ表をロードする事はできず、INNERテーブルを2分割してGPU-Joinを2周する必要がありましたが、これをGPUメモリの範囲に収める事で、GPU-Joinを1周するだけで巨大テーブルのJOINを完了できることを意味します。
}
@en{
When using GPU-Join to join large tables, the most severe constraint is whether the inner table will fit into GPU memory when loaded into a hash table.
Previously, we used a data structure based on PostgreSQL's HeapTupleData data structure, adding hash values ​​and pointers, which required 48 bytes of space per row in the inner table in addition to the payload. Since only the necessary rows are typically loaded from the inner table, it was not uncommon for the header to consume more GPU memory than the payload.

This version is based on PostgreSQL's MinimalTupleData data structure, and by discarding unnecessary data and filling unused area simply as padding with data such as hash values, we have compressed the amount of data required outside the payload per row in the inner table to 24 bytes.
This is particularly useful for joins between large tables; for example, in one test query, we were able to reduce GPU memory consumption from 85GB for the inner table to 61GB. (28% reduction)

Keeping the load within the hardware GPU memory size is extremely important. For example, an NVIDIA H100 with 80GB of DRAM cannot load an 85GB hash table, which would require splitting the inner table in two and running two GPU-Join passes. However, keeping this within the GPU memory range means that a join of large tables can be completed with just one GPU-Join pass.
}

@ja:###pg2arrowの`--flatten`オプション
@en:###`--flatten` option of pg2arrow
@ja{
PostgreSQLが一度に出力できる列は1600個に制限されていますが、非常にスケールの大きなデータセットをApache ArrowまたはParquet形式で出力する場合、この制限が問題となる事があります。
これまでは、主クエリの結果を元にパラメータとして副クエリを実行し、その結果をクライアント側（pg2arrow）で結合する事で1600列以上のApache Arrow形式を出力する`--inner-join`および`--outer-join`オプションを提供していました。しかし、この方法は極めて低速で、大規模なデータセットの出力にはかなりの困難を伴うものでした。

そこで、クエリが複数列の内容をパックした複合型を含む場合、それらを複合型のフィールドとして扱うのではなく、複合型を展開してそれらのサブフィールドを「列」としてApache ArrowまたはParquet形式で書き込むためのオプションが`--flatten`です。

例えば、3000個の列を持つようなデータセットをダンプしたい場合、100列ずつを複合型としてパックしてそれをクライアント側（pg2arrow）で展開するという方法をとれば、そのクエリは30個の複合型を出力しているに過ぎませんので、PostgreSQLの制限には抵触しません。

この機能により`--inner-join`および`--outer-join`オプションは不要となったため、本バージョンより削除されました。
}
@en{
PostgreSQL limits to 1600 columns that we can output at a time, but this limit can pose a problem when outputting very large datasets in Apache Arrow or Parquet format.
Previously, we provided the `--inner-join` and `--outer-join` options to output Apache Arrow format with more than 1600 columns by executing a subquery as a parameter based on the result of the main query and joining the results on the client side (pg2arrow). However, this method was extremely slow and posed considerable challenges when outputting large datasets.

Therefore, when a query includes a composite type that packs the contents of multiple columns, the `--flatten` option is used to expand the composite type and write those subfields as "columns" in Apache Arrow or Parquet format, rather than treating them as composite type fields.

For example, if you want to dump a dataset with 3000 columns, you can pack 100 columns as composite types and then extract them on the client side (pg2arrow). This query will only output 30 composite types, so it will not violate the PostgreSQL limit.

This feature makes the `--inner-join` and `--outer-join` options unnecessary, so they have been removed from this version.
}

@ja:###`pg_strom.cpu_fallback`のデフォルト値
@en:###Default setting of `pg_strom.cpu_fallback`
@ja{
`pg_strom.cpu_fallback`のデフォルト値が`notice`から`off`へ変更されました。

これは、本パラメータの無効化が、GPU-PreAggの最終マージをGPU上で実行する事、GPU-JoinにPinned-Inner-Buffer機能を有効化する事、さらにSELECT INTO Directモードを有効化するための前提条件となっているためです。
}
@en{
The default value of `pg_strom.cpu_fallback` has been changed from `notice` to `off`.

This is because disabling this parameter is a prerequisite for running the final merge of GPU-PreAgg on the GPU, enabling the Pinned-Inner-Buffer feature for GPU-Join, and enabling SELECT INTO Direct mode.
}

@ja:##累積的なバグの修正
@en:##Cumulative bug fixes

- [#948] gpupreagg: allows to embed grouping-key and aggregate-functions in expressions
- [#947] OpenSession had unintentional 1sec wait if other worker already did the final command.
- [#938] arrow_meta.cpp has wrong type cast (TimeType as TimestampType)
- [#937] parquet: prevent parquet::arrow::FileReader::ReadRowGroup() calls in concurrent multi-threading
- [#937] parquet_read: double-linked list was re-designed based on C++ template
- [#934] fix build issue related to libarrow/libparquet on RHEL8
- [#931] allows to build in CUDA 13.0
- [#921] an empty inner relation with RIGHT OUTER JOIN makes GPU kernel launch failure
- [#919] allows to build PG-Strom towards PostgreSQL v18
- [#918] incorrect GPU-Join using numeric join key
- [#916] improvement of pinned-inner-buffer memory management
- [#xxx] fix: LockHeldByMe() was changed at PG17
- [#xxx] adjust kds_nslots on partitioned inner pinned buffer
- [#xxx] pushdown of Result node just over the GPU-PreAgg node
- [#xxx] simple Result has no outer-plan
- [#xxx] nogroup aggregation might crash on mergeGpuPreAggGroupByBuffer
- [#xxx] const ColumnChunkMetaData::statistics() potentially returns NULL
- [#xxx] __dlist_foreach() may goes into infinite loop in c++ code
- [#xxx] arrow_fdw wrong 'files' option parse, infinite loop by strtok
