@ja:#SQLオブジェクト
@en:#SQL Objects
@ja{
本章ではPG-Stromが独自に提供するSQLオブジェクトについて説明します。
}
@en{
This chapter introduces SQL objects additionally provided by PG-Strom.
}

@ja:##システム情報
@en:##System Information

`pgstrom.device_info` @ja{システムビュー} @en{System View}
@ja{
: PG-Strom用にインストールされたGPUデバイスのプロパティを表示します。
: このビューのスキーマ定義は以下の通りです。
}
@en{
: It shows properties of GPU devices installed for PG-Strom.
: Below is schema definition of the view.
}

|name      |type   |description                                       |
|:---------|:------|:-------------------------------------------------|
|gpu_id    |`int`  |@ja{GPUデバイス番号} @en{GPU device number}       |
|att_name  |`text` |@ja{デバイス属性名} @en{Attribute name}           |
|att_value |`text` |@ja{デバイス属性値} @en{Attribute value}          |
|att_desc  |`text` |@ja{デバイス属性の説明} @en{Attribute description}|

@ja{
GPUデバイスのプロパティは非常に数が多く、またCUDAドライバのバージョンによっても対応しているプロパティの数はまちまちです。
そのため、`pgstrom.device_info`システムビューでは、GPUデバイス番号(`gpu_id`)と、デバイス属性名(`att_name`)によって対象となるプロパティを特定します。

以下は`pgstrom.device_info`システムビューの出力例です。
}
@en{
There are various kind of GPU device properties, but depending on the CUDA driver version where system is running.
So, `pgstrom.device_info` system view identifies the target property by GPU device identifier (`gpu_id`) and attribute name (`att_name`).

Below is an example of `pgstrom.device_info` system view.
}


```
postgres=# select * from pgstrom.gpu_device_info limit 10;
 gpu_id |       att_name        |                att_value                 |              att_desc
--------+-----------------------+------------------------------------------+-------------------------------------
      0 | DEV_NAME              | NVIDIA A100-PCIE-40GB                    | GPU Device Name
      0 | DEV_ID                | 0                                        | GPU Device ID
      0 | DEV_UUID              | GPU-13943bfd-5b30-38f5-0473-78979c134606 | GPU Device UUID
      0 | DEV_TOTAL_MEMSZ       | 39.39GB                                  | GPU Total RAM Size
      0 | DEV_BAR1_MEMSZ        | 64.00GB                                  | GPU PCI Bar1 Size
      0 | NUMA_NODE_ID          | -1                                       | GPU NUMA Node Id
      0 | MAX_THREADS_PER_BLOCK | 1024                                     | Maximum number of threads per block
      0 | MAX_BLOCK_DIM_X       | 1024                                     | Maximum block dimension X
      0 | MAX_BLOCK_DIM_Y       | 1024                                     | Maximum block dimension Y
      0 | MAX_BLOCK_DIM_Z       | 64                                       | Maximum block dimension Z
(10 rows)
```

@ja:##Arrow_Fdw
@en:##Arrow_Fdw

`fdw_handler pgstrom.arrow_fdw_handler()`
@ja:: Arrow_FdwのFDWハンドラ関数です。通常、ユーザがこの関数を使用する必要はありません。
@en:: FDW handler function of Arrow_Fdw. Usually, users don't need to invoke this function.

`void pgstrom.arrow_fdw_validator(text[], oid)`
@ja:: Arrow_FdwのFDWオプション検証用関数です。通常、ユーザがこの関数を使用する必要はありません。
@en:: FDW options validation function of Arrow_Fdw. Usually, users don't need to invoke this function.

`event_trigger pgstrom.arrow_fdw_precheck_schema()`
@ja:: Arrowファイルのスキーマ定義をチェックするためのイベントトリガ関数です。通常、ユーザがこの関数を使用する必要はありません。
@en:: Event trigger function to validate schema definition of Arrow files. Usually, users don't need to invoke this function.

`text pgstrom.arrow_fdw_check_pattern(text, text)`
@ja:: 第一引数で与えたファイル名が、第二引数で与えたパターンにマッチするかどうかを検査します。Arrow_Fdwの`pattern`オプションを用いる場合に、ワイルドカードがどのように働くのかを検査するのに便利です。
@en:: Checks whether the file name given in the first argument matches the pattern given in the second argument. This is useful for checking how wildcards work when using the `pattern` option of Arrow_Fdw.

```
=# select pgstrom.arrow_fdw_check_pattern('data_202408_tokyo.data', 'data_@{ymd}_${region}.data');
          arrow_fdw_check_pattern
-------------------------------------------
 true {@[ymd]=[202408], $[region]=[tokyo]}
(1 row)
```
@ja{
上記の例では、ファイル名`data_202408_tokyo.data`に対して、パターン`'data_@{ymd}_${region}.data'`のうち、ワイルドカード`@{ymd}`に相当する部分が`202408`にマッチし、`${region}`に相当する部分が`tokyo`にマッチしています。
}
@en{
In the above example, for the file name `data_202408_tokyo.data`, the part of the pattern `'data_@{ymd}_${region}.data'` that corresponds to the wildcard `@{ymd}` matches `202408`, and the part that corresponds to `${region}` matches `tokyo`.
}

`record pgstrom.arrow_fdw_metadata_info(regclass)`
@ja:: Arrowファイルに埋め込まれているメタデータ（CustomMetadata）を表示します。
@en:: Displays the metadata (CustomMetadata) embedded in the supplied Arrow file.

```
postgres=# select relid, filename, field, key, substring(value, 0, 64) from pgstrom.arrow_fdw_metadata_info('f_lineorder');
    relid    |              filename               |    field     |     key     |                            substring
-------------+-------------------------------------+--------------+-------------+-----------------------------------------------------------------
 f_lineorder | /opt/arrow/f_lineorder_sorted.arrow |              | sql_command | select * from v_lineorder order by lo_orderdate
 f_lineorder | /opt/arrow/f_lineorder_sorted.arrow | lo_orderdate | min_values  | 19920101,19920102,19920103,19920104,19920105,19920107,19920108,
 f_lineorder | /opt/arrow/f_lineorder_sorted.arrow | lo_orderdate | max_values  | 19920102,19920103,19920104,19920105,19920107,19920108,19920109,
(3 rows)
```
@ja{
上記の例では、Arrow_Fdw管理下の外部テーブルである`f_lineorder`に埋め込まれているメタデータ（CustomMetadata）を、それぞれKEY=VALUE形式で出力しています。
スキーマに埋め込まれたメタデータは`field`列がNULLとなっています。そうでなければ列名が表示され、この場合、`sql_command`メタデータがスキーマに埋め込まれ、また、`min_values`および`max_values`が`lo_orderdate`列に埋め込まれている事がわかります。
}
@en{
In the above example, the metadata (CustomMetadata) embedded in the `f_lineorder` foreign table managed by Arrow_Fdw is output in KEY=VALUE form.

Metadata embedded in the schema has the `field` column set to NULL. Otherwise, the column name is displayed. In this case, you can see that the `sql_command` metadata is embedded in the schema, and the `min_values` and `max_values` are embedded in the `lo_orderdate` column.
}

`json pgstrom.arrow_fdw_metadata_stats()`
@ja:: Arrow_Fdwのメタデータキャッシュの統計情報を表示します
@en:: Displays the statistics of the Arrow_Fdw metadata-cache

@ja{
Arrow_Fdwは何らかの形で一度参照したArrowファイルのメタデータ（スキーマ定義やデータ配置など）を共有メモリ上に記憶しており、次回以降にArrow_Fdw外部テーブルを参照する際の応答性を改善する事ができます。

この関数は、共有メモリ上のメタデータキャッシュに関する統計情報を表示する事ができます。
メタデータキャッシュが不足すると、Arrow_Fdw外部テーブルの参照応答性が低下する事がありますので、その場合は`arrow_fdw.metadata_cache_size`を拡大してメタデータキャッシュを拡大する事が必要になります。
}
@en{
Arrow_Fdw somehow remembers the metadata (schema definition, data placement, etc.) of an Arrow file that has been referenced once in shared memory, and can improve responsiveness when referencing an Arrow_Fdw foreign table from the next time onwards.

This function can display statistical information about the metadata cache in shared memory.

If the metadata cache is insufficient, the reference responsiveness of Arrow_Fdw foreign tables may decrease. In that case, it is necessary to enlarge the metadata cache by increasing `arrow_fdw.metadata_cache_size`.
}

```
s=# select pgstrom.arrow_fdw_metadata_stats();

       arrow_fdw_metadata_stats

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 { "total_cache_sz" : 536870912, "total_cache_allocated" : 9568256, "total_cache_usage" : 9354072, "num_file_entries" : 2, "cache_usage_efficiency" : 0.977615, "cache_num_blocks" : 4096, "cache_active_blocks" : 73, "cache_free_blocks" : 4023, "cache_used_ratio" : 0.017822 "last_allocate_timestamp" : "2025-03-29 13:24:44.447" }
(1 row)
```
@ja{
上記の例のように統計情報はJSON形式で表示され、各フィールドの意味は以下の通りです。
}
@en{
As in the example above, the statistics are displayed in JSON format, with the meaning of each field as follows:
}
- total_cache_sz
    - @ja{メタデータキャッシュの総容量（バイト単位）}@en{The total size of the metadata cache (in bytes)}
- total_cache_allocated
    - @ja{割り当て済みのメタデータキャッシュサイズ（バイト単位）}@en{Allocated metadata cache size (in bytes)}
- total_cache_usage
    - @ja{使用済みのメタデータキャッシュサイズ（バイト単位）}@en{Used metadata cache size (in bytes)}
- num_file_entries
    - @ja{キャッシュが保持されているArrowファイルの個数}@en{Number of Arrow files that are cached}
- cache_usage_efficiency
    - @ja{割り当て済みメタデータキャッシュに占める使用済みメタデータキャッシュの比率}@en{The ratio of used metadata cache to the allocated metadata cache}
- cache_num_blocks
    - @ja{メタデータキャッシュのブロック数（128kB単位）}@en{Number of blocks in the metadata cache (in 128kB units)}
- cache_active_blocks
    - @ja{割り当て済みのメタデータキャッシュブロック数}@en{Number of allocated metadata cache blocks}
- cache_free_blocks
    - @ja{空いているメタデータキャッシュブロック数}@en{Number of free metadata cache blocks}
- cache_used_ratio
    - @ja{割り当て済みのメタデータキャッシュブロックの比率}@en{Percentage of metadata cache blocks allocated}
- last_allocate_timestamp
    - @ja{最後にメタデータキャッシュを割り当てた時のタイムスタンプ}@en{The timestamp of the last metadata cache allocation}
- last_reclaim_timestamp
    - @ja{最後にメタデータキャッシュが回収された時のタイムスタンプ}@en{The timestamp when the metadata cache was last cleared}

`void pgstrom.arrow_fdw_import_file(text, text, text = null)`
@ja{
: Apache Arrow形式ファイルをインポートし、新たに外部テーブル(foreign table)を定義します。第一引数は外部テーブルの名前、第二引数はApache Arrow形式ファイルのパス、省略可能な第三引数はスキーマ名です。
: この関数は`IMPORT FOREIGN SCHEMA`構文に似ていますが、PostgreSQLにおけるテーブルの列数制限（`MaxTupleAttributeNumber` = 1664）を越える列が定義されたApache Arrow形式ファイルをインポートできます。つまり、これに該当しない大半のユースケースでは`IMPORT FOREIGN SCHEMA`構文を利用すべきです。
: 以下の例は、`pgstrom.arrow_fdw_import_file`を用いて2000個のInt16列を持つApache Arrowファイルをインポートしたものです。`\d mytest`の実行結果より、新たに作成された外部テーブル`mytest`が2000個のフィールドを持っている事が分かります。
: PostgreSQL内部表現の都合上、全ての列を一度に読み出す事はできませんが、最後の例のように一部の列だけを参照するワークロードであれば実行可能です。
}
@en{
: This function tries to import Apache Arrow file, and defines a new foreign table. Its first argument is name of the new foreign table, the second argument is path of the Apache Arrow file, and the optional third argument is the schema name.
: This function is similar to `IMPORT FOREIGN SCHEMA` statement, but allows to import Apache Arrow files that have wider fields than the limitation of number of columns in PostgreSQL (`MaxTupleAttributeNumber` = 1664). So, we recommend to use `IMPORT FOREIGN SCHEMA` statement for most cases.
: The example below shows the steps to import an Apache Arrow file with 2000 of Int16 fields by the `pgstrom.arrow_fdw_import_file`. The result of `\d mytest` shows this foreign table has 2000 fields.
: Due to the internal data format of PostgreSQL, it is not possible to read all the columns at once, but possible to read a part of columns like the last example.
}

```
=# select pgstrom.arrow_fdw_import_file('mytest', '/tmp/wide2000.arrow');
 arrow_fdw_import_file
-----------------------

(1 row)

=# \d
            List of relations
 Schema |  Name  |     Type      | Owner
--------+--------+---------------+--------
 public | mytest | foreign table | kaigai
(1 row)


=# \d mytest
                    Foreign table "public.mytest"
  Column   |   Type   | Collation | Nullable | Default | FDW options
-----------+----------+-----------+----------+---------+-------------
 object_id | integer  |           |          |         |
 c000      | smallint |           |          |         |
 c001      | smallint |           |          |         |
 c002      | smallint |           |          |         |
 c003      | smallint |           |          |         |
   :             :          :          :          :            :
 c1997     | smallint |           |          |         |
 c1998     | smallint |           |          |         |
 c1999     | smallint |           |          |         |
Server: arrow_fdw
FDW options: (file '/tmp/wide2000.arrow')

=# select * from mytest ;
ERROR:  target lists can have at most 1664 entries

=# select c0010,c1234,c1999 from mytest limit 3;
 c0010 | c1234 | c1999
-------+-------+-------
   232 |   232 |   232
   537 |   537 |   537
   219 |   219 |   219
(3 rows)
```

@ja:##GPUキャッシュ
@en:##GPU Cache

`pgstrom.gpucache_info` @ja{システムビュー}@en{System View}
@ja:: GPUキャッシュの現在の状態を表示します。<br>このビューのスキーマ定義は以下の通りです。
@en:: It shows the current status of GPU Cache.<br>Below is schema definition of the view.


|name               |type  |description  |
|:------------------|:-----|-------------|
|`database_oid`     |`oid` |@ja{GPUキャッシュを設定したテーブルの属するデータベースのOIDです}  @en{Database OID where the table with GPU Cache belongs to.} |
|`database_name`    |`text`|@ja{GPUキャッシュを設定したテーブルの属するデータベースの名前です} @en{Database name where the table with GPU Cache belongs to.} |
|`table_oid`        |`oid` |@ja{GPUキャッシュを設定したテーブルのOIDです。必ずしも現在のデータベースとは限らない事に留意してください。} @en{Table OID that has GPU Cache. Note that it may not be in the current database.} |
|`table_name`       |`text`|@ja{GPUキャッシュを設定したテーブルの名前です。必ずしも現在のデータベースとは限らない事に留意してください。} @en{Table name that has GPU Cache. Note that it may not be in the current database.} |
|`signature`        |`int8`|@ja{GPUキャッシュの一意性を示すハッシュ値です。例えば`ALTER TABLE`の前後などでこの値が変わる場合があります。} @en{An identifier hash value of GPU Cache. It may be changed after `ALTER TABLE` for example.} |
|`phase`            |`text`|@ja{GPUキャッシュ構築の段階を示します。`not_built`, `is_empty`, `is_loading`, `is_ready`, `corrupted`のいずれかです。} @en{Phase of GPU cache construction: either `not_built`, `is_empty`, `is_loading`, `is_ready`, or `corrupted`} |
|`rowid_num_used`   |`int8`|@ja{割当て済みの行IDの数です。} @en{Number of allocated row-id} |
|`rowid_num_free`   |`int8`|@ja{未割当の行IDの数です。} @en{Number of free row-id} |
|`gpu_main_sz`      |`int8`|@ja{GPUキャッシュ上の固定長データ用の領域のサイズです。} @en{Size of the fixed-length values area on the GPU Cache.} |
|`gpu_main_nitems`  |`int8`|@ja{GPUキャッシュ上のタプル数です。} @en{Number of tuples on the GPU Cache.} |
|`gpu_extra_sz`     |`int8`|@ja{GPUキャッシュ上の可変長データ用の領域のサイズです。} @en{Size of the variable-length values area on the GPU Cache.} |
|`gpu_extra_usage`  |`int8`|@ja{GPUキャッシュ上の可変長データ領域の使用済みサイズです。} @en{Size of the used variable-length values area on the GPU Cache.} |
|`gpu_extra_dead`   |`int8`|@ja{GPUキャッシュ上の可変長データ領域の未使用サイズです。} @en{Size of the free variable-length values area on the GPU Cache.} |
|`redo_write_ts`    |`timestamptz`|@ja{REDOログバッファを最後に更新した時刻です。} @en{Last update timestamp on the REDO Log buffer} |
|`redo_write_nitems`|`int8`|@ja{REDOログバッファに書き込まれたREDOログの総数です。} @en{Total number of REDO Log entries written to the REDO Log buffer.} |
|`redo_write_pos`   |`int8`|@ja{REDOログバッファに書き込まれたREDOログの総バイト数です。} @en{Total bytes of REDO Log entries written to the REDO Log buffer.} |
|`redo_read_nitems` |`int8`|@ja{REDOログバッファから読み出し、GPUに適用されたREDOログの総数です。} @en{Total number of REDO Log entries read from REDO Log buffer, and already applied to.} |
|`redo_read_pos`    |`int8`|@ja{REDOログバッファから読み出し、GPUに適用されたREDOログの総バイト数です。} @en{Total bytes of REDO Log entries read from REDO Log buffer, and already applied to.} |
|`redo_sync_pos`    |`int8`|@ja{REDOログバッファ書き込まれたREDOログのうち、既にGPUキャッシュへの適用をバックグラウンドワーカにリクエストした位置です。REDOログバッファの残り容量が逼迫してきた際に、多数のセッションが同時に非同期のリクエストを発生させる事を避けるため、内部的に使用されます。} @en{The latest position on the REDO Log buffer, where it is already required the background worker to synchronize onto the GPU Cache. When free space of REDO Log buffer becomes tight, it is internally used to avoid flood of simultaneous asynchronized requests by many sessions.} |
|`config_options`   |`text`|@ja{GPUキャッシュのオプション文字列です。} @en{Options string of the GPU Cache} |

@ja{
以下は`pgstrom.gpucache_info`システムビューの出力例です。
}
@en{
Below is an example of `pgstrom.gpucache_info` system view.
}
```
=# select * from pgstrom.gpucache_info ;
 database_oid | database_name | table_oid |    table_name    | signature  |  phase   | rowid_num_used | rowid_num_free | gpu_main_sz | gpu_main_nitems | gpu_extra_sz | gpu_extra_usage | gpu_extra_dead |         redo_write_ts         | redo_write_nitems | redo_write_pos | redo_read_nitems | redo_read_pos |
redo_sync_pos |                                                   config_options
--------------+---------------+-----------+------------------+------------+----------+----------------+----------------+-------------+-----------------+--------------+-----------------+----------------+-------------------------------+-------------------+----------------+------------------+---------------+---------------+---------------------------------------------------------------------------------------------------------------------
       193450 | hoge          |    603029 | cache_test_table | 4529357070 | is_ready |           4000 |           6000 |      439904 |            4000 |      3200024 |          473848 |              0 | 2023-12-18 01:25:42.850193+09 |              4000 |         603368 |             4000 |        603368 |       603368 | gpu_device_id=0,max_num_rows=10000,redo_buffer_size=157286400,gpu_sync_interval=4000000,gpu_sync_threshold=10485760
(1 row)
```

`trigger pgstrom.gpucache_sync_trigger()`
: @ja{テーブル更新の際にGPUキャッシュを同期するためのトリガ関数です。詳しくは[GPUキャッシュ](gpucache.md)の章を参照してください。}
: @en{A trigger function to synchronize GPU Cache on table updates. See [GPU Cache](gpucache.md) chapter for more details.}

`bigint pgstrom.gpucache_apply_redo(regclass)`
: @ja{引数で指定されたテーブルにGPUキャッシュが設定されている場合、未適用のREDOログを強制的にGPUキャッシュに適用します。}
: @en{If the given table has GPU Cache configured, it forcibly applies the REDO log entries onto the GPU Cache.}

`bigint pgstrom.gpucache_compaction(regclass)`
: @ja{引数で指定されたテーブルにGPUキャッシュが設定されている場合、可変長データバッファを強制的にコンパクト化します。}
: @en{If the given table has GPU Cache configured, it forcibly run compaction of the variable-length data buffer.}

`bigint pgstrom.gpucache_recovery(regclass)`
: @ja{破損（corrupted）状態となったGPUキャッシュを復元しようと試みます。}
: @en{It tries to recover the corrupted GPU cache.}

<!--
@ja:##HyperLogLog 関数
@en:##HyperLogLog Functions

`bigint pg_catalog.hll_count(TYPE)`
: @ja{HyperLogLogアルゴリズムを使用してキー値のカーディナリティを推定する集約関数です。}
: @en{An aggregate function to estimate cardinarity of the key value, using HyperLogLog algorithm.}
: @ja{`TYPE`は`int1`、`int2`、`int4`、`int8`、`numeric`、`date`、`time`、`timetz`、`timestamp`、`timestamptz`、`bpchar`、`text`、または`uuid`のいずれかです。}
: @en{`TYPE` is any of `int1`, `int2`, `int4`, `int8`, `numeric`, `date`, `time`, `timetz`, `timestamp`, `timestamptz`, `bpchar`, `text`, or `uuid`.}
: @ja{PG-StromのHyperLogLog機能について、詳しくは[HyperLogLog](hll_count.md)を参照してください。}
: @en{See HyperLogLog for more details of [HyperLogLog](hll_count.md) functionality of PG-Strom.}

`bytea pg_catalog.hll_sketch(TYPE)`
: @ja{引数で与えたキー値から、HyperLogLogアルゴリズムで使用するHLL Sketchを生成し、`bytea`データとして返す集約関数です。}
: @en{An aggregate function to build HLL Sketch, used for HyperLogLog algorithm, then return as `bytea` datum.}
: @ja{`TYPE`は`int1`、`int2`、`int4`、`int8`、`numeric`、`date`、`time`、`timetz`、`timestamp`、`timestamptz`、`bpchar`、`text`、または`uuid`のいずれかです。}
: @en{`TYPE` is any of `int1`, `int2`, `int4`, `int8`, `numeric`, `date`, `time`, `timetz`, `timestamp`, `timestamptz`, `bpchar`, `text`, or `uuid`.}

`bigint pg_catalog.hll_merge(bytea)`
: @ja{HLL Sketchから、元になったキー値のカーディナリティを推定する集約関数です。引数は`hll_sketch()`関数の生成したHLL Sketchである事が期待されています。}
: @en{An aggregate function that estimate cardinarity of the key values that are the source of the supplied HLL Sketch.}

`bytea pg_catalog.hll_combine(bytea)`
: @ja{複数のHLL Sketchを結合し、その結果をまたHLL Sketchとして出力する集約関数です。例えば週次データのHLL Sketchを月次データに変換するといった利用方法を想定しています。}
: @en{An aggregate function that combines multiple HLL Sketches, then returns a consolicated HLL Sketch. It is expected to transform HLL Sketch of weekly data to monthly data, for example.}

`int4[] pg_catalog.hll_sketch_histogram(bytea)`
: @ja{引数として与えたHLL Sketchを走査し、各レジスタの値に基づくヒストグラムを作成して出力する関数です。これは集約関数ではありません。`hll_sketch()`などで出力したHLL Sketchの内容を可視化する事を目的としています。}
: @en{A function to generate a histogram based on the register values of the supplied HLL Sketch. This is not an aggregate function. It expects to visualize the contents of HLL Sketch generated by `hll_sketch()` and so on.}
-->

@ja:##テストデータ生成
@en:##Test Data Generator

`void pgstrom.random_setseed(int)`
@ja:: 乱数の系列を初期化します。
@en:: It initializes the random seed.

`bigint pgstrom.random_int(float=0.0, bigint=0, bigint=INT_MAX)`
@ja:: `bigint`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `bigint` type within the range.

`float pgstrom.random_float(float=0.0, float=0.0, float=1.0)`
@ja:: `float`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `float` type within the range.

`date pgstrom.random_date(float=0.0, date='2015-01-01', date='2025-12-31')`
@ja:: `date`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `date` type within the range.

`time pgstrom.random_time(float=0.0, time='00:00:00', time='23:59:59')`
@ja:: `time`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `time` type within the range.

`timetz pgstrom.random_timetz(float=0.0, time='00:00:00', time='23:59:59')`
@ja:: `timetz`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `timetz` type within the range.

`timestamp pgstrom.random_timestamp(float=0.0, timestamp='2015-01-01', timestamp='2025-01-01')`
@ja:: `timestamp`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `timestamp` type within the range.

`macaddr pgstrom.random_macaddr(float=0.0, macaddr='ab:cd:00:00:00', macaddr='ab:cd:ff:ff:ff:ff')`
@ja:: `macaddr`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `macaddr` type within the range.

`inet pgstrom.random_inet(float=0.0, inet='192.168.0.1/16')`
@ja:: `inet`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `inet` type within the range.

`text pgstrom.random_text(float=0.0, text='test_**')`
@ja:: `text`型のランダムデータを生成します。第二引数の'*'文字をランダムに置き換えます。
@en:: It generates random data in `text` type. The '*' characters in 2nd argument shall be replaced randomly.

`text pgstrom.random_text_len(float=0.0, int=10)`
@ja:: `text`型のランダムデータを指定文字列長の範囲内で生成します。
@en:: It generates random data in `text` type within the specified length.

`int4range pgstrom.random_int4range(float=0.0, bigint=0, bigint=INT_MAX)`
@ja:: `int4range`型のランダムデータを指定の範囲内で生成します。}
@en:: It generates random data in `int4range` type within the range.}

`int8range pgstrom.random_int8range(float=0.0, bigint=0, bigint=LONG_MAX)`
@ja:: `int8range`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `int8range` type within the range.

`tsrange pgstrom.random_tsrange(float=0.0, timestamp='2015-01-01', timestamp='2025-01-01')`
@ja:: `tsrange`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `tsrange` type within the range.

`tstzrange pgstrom.random_tstzrange(float=0.0, timestamptz='2015-01-01', timestamptz='2025-01-01')`
@ja:: `tstzrange`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `tstzrange` type within the range.

`daterange pgstrom.random_daterange(float=0.0, date='2015-01-01', date='2025-12-31')`
@ja:: `daterange`型のランダムデータを指定の範囲内で生成します。
@en:: It generates random data in `daterange` type within the range.

@ja:##その他の関数
@en:##Other Functions

`text pgstrom.githash()`
@ja:: 現在ロードされているPG-Stromモジュールの元となったソースコードリビジョンのハッシュ値を表示します。この値は、障害時にソフトウェアのリビジョンを特定するのに有用です。
@en:: It displays the hash value of the source code revision from the currently loaded PG-Strom module is based. This value is useful in determining the software revision in the event of a failure.

```
postgres=# select pgstrom.githash();
                 githash
------------------------------------------
 103984be24cafd1e7ce6330a050960d97675c196
```

`text pgstrom.license_query()`
@ja:: ロードされていれば、現在ロードされている商用サブスクリプションを表示します。
@en:: It displays the active commercial subscription, if loaded.

```
=# select pgstrom.license_query();
                                                                               license_query
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 { "version" : 2, "serial_nr" : "HDB-TRIAL", "issued_at" : "2020-11-24", "expired_at" : "2025-12-31", "gpus" : [ { "uuid" : "GPU-8ba149db-53d8-c5f3-0f55-97ce8cfadb28" } ]}
(1 row)
```

