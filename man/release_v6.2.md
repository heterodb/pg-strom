@ja:#PG-Strom v6.2リリース
@en:#PG-Strom v6.2 Release

<div style="text-align: right;">PG-Strom Development Team (20th-Jun-2026)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v6.2における主要な変更は点は以下の通りです。

- Parquet読み出しの高速化と、Parquet Cacheの対応
- 非対称なPartition-wise Join/Group-Byの再設計
- 包括的なlibarrow / libparquet への移行
- GPU-Cache、DPUサポート、PostGISサポートの削除
- 累積的なバグの修正
}

@en{
Major changes in PG-Strom v6.2 are as follows:

- Improved Parquet read performance and support for Parquet Cache
- Redesign of asymmetric Partition-wise Join/Group-By
- Comprehensive migration to libarrow / libparquet
- Removal of GPU-Cache, DPU support, and PostGIS support
- Cumulative bug fixes
}

@ja:##動作環境
@en:##Prerequisites

@ja{
- PostgreSQL v16以降
- CUDA Toolkit 13 以降
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 7.5 以降 (Turing以降)
}
@en{
- PostgreSQL v16 or later
- CUDA Toolkit 13 or later
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 7.5 or later (Turing or newer)
}

@ja:##Parquet読み出しの高速化と、Parquet Cacheの対応
@en:##Improved Parquet read performance and support for Parquet Cache

@ja{
クエリの実行時にParquetファイルを読み出す処理が高速化されました。

Parquetファイル末尾のメタデータを使い回す事でファイルアクセスの回数を減らし、
ファイル構造の解析に要する処理を軽量化しています。

v6.1の実装では、Parquetファイルのメタデータ解析は排他的に処理されていたため、
この部分がParquetファイルの読み出しにおける主要なボトルネックの一つでした。
}
@en{}

@ja{
**Parquet Cacheの対応**

PG-Strom v6.2では、Parquetファイルの読み出しを高速化するため、新たにParquet Cache機能が追加されました。

Parquet Cacheは、Parquetファイル内の圧縮データを読み出し・解凍した後のデータを、高速ストレージ上にキャッシュするための専用キャッシュ機構です。Parquetファイルは一般に列単位で圧縮されているため、GPUで処理を行う前には、対象となるカラムデータを読み出し、圧縮を解除し、GPUで処理可能な形式へ展開する必要があります。Parquet Cacheは、この解凍後のデータをあらかじめ高速ストレージ上に保持しておくことで、以後の同一データへのアクセスにおいて、Parquetの読み出しおよび解凍処理のコストを削減します。

Parquet Cacheに格納されるデータは、Apache Arrowと同等のメモリ表現を持つ列指向データです。そのため、GPU-Direct SQL機構と組み合わせることで、NVME-SSDからGPUメモリへデータを直接転送し、CPUを介した余分なコピーや変換処理を抑制することができます。

この機能は、Parquetファイル本体をNFSやHDDなどの比較的低速なストレージに配置し、Parquet CacheをNVME-SSD（高速ストレージ）上に配置する構成で特に効果を発揮します。低速ストレージ上のParquetファイルを初回アクセス時に読み出し、必要なデータを解凍したうえで高速ストレージにキャッシュしておくことで、2回目以降のアクセスでは高速ストレージ上のキャッシュ済みデータを利用できます。

一般的なファイルシステムキャッシュや、ブロックデバイスレベルのNVMEキャッシュ機構とは異なり、Parquet CacheはParquetファイルの圧縮済みバイト列をそのままキャッシュするのではなく、圧縮データを解凍した後の、GPU処理に適した列指向データをキャッシュします。これにより、単にストレージI/Oを高速化するだけでなく、Parquet形式に固有の解凍処理やデータ展開処理を回避できる点が大きな特徴です。

また、Parquet Cacheへの書き込みは非同期かつO_DIRECTを使用して行われ、サーバのPage Cacheを経由しません。そのため、大規模なParquetデータを読み出してキャッシュを構築する場合でも、OSのPage Cacheを不要に消費したり、他のワークロードのキャッシュ効率を低下させたりすることを避けられます。これは、GPU-Direct SQLによってNVME-SSDからGPUメモリへ直接データを転送する構成では、CPU側のPage Cacheを介在させる利点はほとんどないためです。
Parquet Cacheは、このようなGPU直結型のデータ処理に適したキャッシュ機構として設計されています。
}


- ここにサンプル、グラフ
- io帯域グラフの方がよいか。

- pgstrom.parquet_cache_info()



@ja:##非対称なPartition-wise Join/Group-Byの再設計
@en:##Redesign of asymmetric Partition-wise Join/Group-By

@ja{
パーティション化されたテーブルと、非パーティションテーブルのGPU-JOINや、パーティション子要素に対するGPU-PreAggのプッシュダウンはPG-Strom v2.2で追加された機能です。
しかし、この実装アプローチには問題がありました。
一つは、パーティション子要素のそれぞれにGPU-JoinやGPU-PreAggをプッシュダウンし、最後にそれをAppendするという、実行計画の大きな改変を伴うアプローチではクエリオプティマイザの設計が非常に複雑化してしまい、ソフトウェア品質上の課題を抱えてしまう事になっていたこと。
もう一つは、GPU-Joinをパーティション子要素にプッシュダウンした場合、それと結合すべき非パーティションテーブルはGPU-Joinのそれぞれに対して読み出され、またJOINに使うハッシュ表はパーティションの数だけGPUメモリ上にロードされリソースを圧迫する事になりました。
以下の古いバージョンにおける実行計画を見てください。


}


```
# EXPLAIN SELECT cat,count(*),avg(ax)
            FROM pt NATURAL JOIN t1
           WHERE ymd > '2017-01-01'::date
           GROUP BY cat;
                                   QUERY PLAN
--------------------------------------------------------------------------------
 HashAggregate  (cost=196410.07..196412.57 rows=200 width=48)
   Group Key: pt_2017.cat
   ->  Gather  (cost=66085.69..196389.07 rows=1200 width=72)
         Workers Planned: 2
         ->  Parallel Append  (cost=65085.69..195269.07 rows=600 width=72)
               ->  Parallel Custom Scan (GpuPreAgg)  (cost=65085.69..65089.69 rows=200 width=72)
                     Reduction: Local
                     Combined GpuJoin: enabled
                     ->  Parallel Custom Scan (GpuJoin) on pt_2017  (cost=32296.64..74474.20 rows=1050772 width=40)
                           Outer Scan: pt_2017  (cost=28540.80..66891.11 rows=1050772 width=36)
                           Outer Scan Filter: (ymd > '2017-01-01'::date)
                           Depth 1: GpuHashJoin  (nrows 1050772...2521854)
                                    HashKeys: pt_2017.aid
                                    JoinQuals: (pt_2017.aid = t1.aid)
                                    KDS-Hash (size: 10.78MB)
                           ->  Seq Scan on t1  (cost=0.00..1935.00 rows=100000 width=12)
               ->  Parallel Custom Scan (GpuPreAgg)  (cost=65078.35..65082.35 rows=200 width=72)
                     Reduction: Local
                     Combined GpuJoin: enabled
                     ->  Parallel Custom Scan (GpuJoin) on pt_2018  (cost=32296.65..74465.75 rows=1050649 width=40)
                           Outer Scan: pt_2018  (cost=28540.81..66883.43 rows=1050649 width=36)
                           Outer Scan Filter: (ymd > '2017-01-01'::date)
                           Depth 1: GpuHashJoin  (nrows 1050649...2521557)
                                    HashKeys: pt_2018.aid
                                    JoinQuals: (pt_2018.aid = t1.aid)
                                    KDS-Hash (size: 10.78MB)
                           ->  Seq Scan on t1  (cost=0.00..1935.00 rows=100000 width=12)
               ->  Parallel Custom Scan (GpuPreAgg)  (cost=65093.03..65097.03 rows=200 width=72)
                     Reduction: Local
                     Combined GpuJoin: enabled
                     ->  Parallel Custom Scan (GpuJoin) on pt_2019  (cost=32296.65..74482.64 rows=1050896 width=40)
                           Outer Scan: pt_2019  (cost=28540.80..66898.79 rows=1050896 width=36)
                           Outer Scan Filter: (ymd > '2017-01-01'::date)
                           Depth 1: GpuHashJoin  (nrows 1050896...2522151)
                                    HashKeys: pt_2019.aid
                                    JoinQuals: (pt_2019.aid = t1.aid)
                                    KDS-Hash (size: 10.78MB)
                           ->  Seq Scan on t1  (cost=0.00..1935.00 rows=100000 width=12)
(38 rows)
```












@ja:##その他の修正
@en:##Other changes


@ja{
**包括的なlibarrow / libparquet への移行**

以下のコマンド、モジュールが新たにlibarrow/libparquetを使用するよう書き換えられました。これにより従来の独自実装によるArrowの実装は全て置き換えられました。

- fluent-plugin-arrow-file
- vcf2arrow
}
@en{
**Comprehensive migration to libarrow / libparquet**

The following commands and modules have been rewritten to use libarrow/libparquet. As a result, all previous custom Arrow implementations have been replaced.

- fluent-plugin-arrow-file
- vcf2arrow
}

@ja{
**利用頻度の低い機能の削除**

GPU-Cache、DPUサポート、PostGISサポート、およびパーティション数がGPU数を越える場合のPinned Inner Buffer機構は削除されました。
これらの機能は、実際にはほとんど利用されていないにも関わらず、実装に必要以上の複雑さをもたらし、ソフトウェア品質上の課題となっていました。
（加えてDPUサポートに関しては、対応するSmart-SSD製品を販売していた会社がなくなってしまいました）
}
@en{
**Removal of rarely used features**

GPU-Cache, DPU support, PostGIS support, and the Pinned Inner Buffer mechanism used when the number of partitions exceeds the number of GPUs have been removed.

These features were rarely used in practice, yet they added disproportionate implementation complexity and had become a burden on maintaining software quality.

(In the case of DPU support, the vendor of the corresponding Smart-SSD products has discontinued its business.)
}
@ja{
**pg2arrowの新オプション**

pg2arrowに--paramオプションを追加しました。これはSQLコマンド中の@(NAME)を置き換えるもので、シェルスクリプトで機械的にテーブルをダンプする際に便利です。
}
@en{
**pg2arrow new option**

The --param option has been added to pg2arrow. This option replaces @(NAME) placeholders in SQL commands, which is useful when mechanically dumping tables from shell scripts.
}

@ja:##累積的なバグの修正
@en:##Cumulative bug fixes

- [#1026] add special case for apply_scanjoin_target_to_paths
- [#1022] bugfix: expressions including CaseTestExpr was unintentionally replaced by pseudo-constant
- [#1021] bugfix: BRIN based scan generated wrong result set in parallel mode
- [#1019] fix compilation issue for PG16
- [#1013] bugfix: Segfault on pgstromBrinIndexExecReset() with NestedLoop
- [#1012] adjust EXPLAIN output for partitioned relation
- [#1011] bugfix: custom_metadata embedded in parquet/arrow schema definition
- [#1009] allows to build without pkgconf (auto switch to pkg-config)
- [#1007] bugfix: using BRIN index with to_timestamp(const value) makes infinite loop
- [#1006] bugfix: ShutdownDSM() may be called before execution end
- [#996] GPU architecture list is updated to the current available models
- [#993] Fix a minor error about to the formatting of relid
- [#991] connection workers should be terminated before DSM release
- [#990] fluentd: ts_column and tag_column properties are not assigned correctly.
- [#984] arrow_fdw: multiple virtual columns with 'virtual_metadata_split' shows only last metadata
- [#xxx] parquetReadArrowTable() didn't consider ARROW_ALIGN to calculate buffer length
- [#xxx] arrow2csv: print IP address if field contains 'pg_type' = 'inet' metadata
- [#xxx] bugfix: arrow_fdw stats_hint with VAR in (...) operator didn't work correctly
- [#xxx] wrong definition of host_mlocation
- [#xxx] bugfix: pickup_outer_referenced() didn't expect expressions in the qub-queries
- [#xxx] bugfix: readArrowFileInfo didn't close the file
- [#xxx] bugfix: allvisfrac of arrow_fdw table may have NaN
- [#xxx] bugfix: EXPLAIN on parquet may reference NULL DSM
- [#xxx] bugfix: parquet cache skip block-write
- [#xxx] bugfix: parquet cache wrongly flushed the block buffer
