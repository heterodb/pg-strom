@ja:#GPUキャッシュ
@en:#GPU Cache

@ja:##概要
@en:##Overview

@ja{
GPUにはホストシステムのRAMとは独立なデバイスメモリが搭載されており、GPUで計算するにはPCI-Eバスなどを通じて、一旦、ホストシステムやストレージデバイスからデータをGPUデバイスメモリ側へ転送する必要があります。
PG-StromがSQLクエリをGPUで処理する場合も例外ではなく、内部的には、PostgreSQLテーブルから読み出したレコードをGPUへと転送し、その上でGPUでSQLの各種処理を実行します。
しかしこれらの処理には、多かれ少なかれテーブルの読み出しやデータの転送に時間を要します（多くの場合、GPUでの処理よりも遥かに長い時間の！）。
}
@en{
GPU has a device memory that is independent of the RAM in the host system, and in order to calculate on the GPU, data must be transferred from the host system or storage device to the GPU device memory once through the PCI-E bus.
The same is true when PG-Strom processes SQL queries on the GPU. Internally, the records read from the PostgreSQL table are transferred to the GPU, and then various SQL operations are executed on the GPU.
However, these processes take time to read the tables and transfer the data. (In many cases, much longer than the processing on the GPU!)
}

@ja{
GPUキャッシュ（GPU Cache）とは、GPUデバイスメモリ上に予め領域を確保しておき、そこにPostgreSQLテーブルの複製を保持しておく機能です。
テーブルへの更新はトリガを通じてGPUキャッシュへ非同期に反映されるため、トランザクション性能を大きく損なうことなく、検索/分析系のSQLは常に最新のデータに対して実行されます。
検索/分析系のSQLを実行する際には既にGPU上にデータがロードされているため、改めてテーブルからレコードを読み出したり、PCI-Eバスを介してデータを転送したりする必要がありません。

GPUキャッシュは、頻繁に更新されるデータに対して、応答時間を重視する検索/分析系のクエリを実行するワークロードを意図しています。
例えば、直近に収集されたデータを対象に、数十～数百ミリ秒といった応答時間の要件で検索を行うようなケースです。
Apache ArrowやParquetといったファイル形式を経由する場合には、ファイルの生成やメタデータの読み出しに余計な時間を要しますが、GPUキャッシュであれば、PostgreSQLテーブルにINSERTするだけで、そのデータを検索/分析クエリから参照する事ができます。
}
@en{
GPU Cache is a function that reserves an area on the GPU device memory in advance and keeps a copy of the PostgreSQL table there.
Updates on the table are applied to the GPU Cache asynchronously via a trigger, so search/analysis SQL always runs on the latest data without significant impact on the transaction performance.
When search/analysis SQL is executed, the data is already loaded on the GPU, so there is no need to read the records from the table again or to transfer the data over the PCI-E bus.

GPU Cache is intended for workloads that run latency-sensitive search/analysis queries on frequently updated data.
For example, a search on the recently collected data with a response time requirement of tens to hundreds of milliseconds.
Using file formats such as Apache Arrow or Parquet needs extra time to generate the files and to read their metadata, however, GPU Cache allows search/analysis queries to refer the data just after INSERT into the PostgreSQL table.
}

@ja{
!!! Note
    PG-Strom v6.2のGPUキャッシュは、従来のGPUキャッシュを全面的に再実装したものです。
    設定方法や状態確認のための関数が従来のバージョンから変わっていますので、本章末尾の「従来のバージョンからの変更点」も併せて参照してください。
}
@en{
!!! Note
    GPU Cache in PG-Strom v6.2 is an entire re-implementation of the former GPU Cache.
    The configuration and the functions to check its status are changed from the former versions, so please also see [Changes from the former versions](#changes-from-the-former-versions).
}

@ja:##アーキテクチャ
@en:##Architecture

@ja{
GPUキャッシュでは、並列度の高い更新系ワークロードに対応することと、検索/分析クエリが常に最新のデータを参照するという2つの要件をクリアする必要があります。

多くのシステムではCPUとGPUはPCI-Eバスを介して接続され、その通信には相応のレイテンシが発生します。
そのため、GPUキャッシュの対象テーブルが1行更新されるたびにGPUキャッシュを同期していては、トランザクション性能に大きな影響を与えてしまいます。

GPUキャッシュを設定したテーブルに対してテーブルの更新を伴うSQLコマンド（INSERT、UPDATE、DELETE）を実行すると、AFTER ROWトリガによって、更新内容が更新ログとしてGPU Serviceへ送られます。
GPU Serviceの内部では、GPUごとにGPUキャッシュ専用のワーカが動作しており、受け取った更新ログをログバッファに蓄積します。
この処理はGPUへの呼び出しを伴わないため、トランザクション性能への影響はほとんどありません。

ログバッファに蓄積された更新ログは、ログバッファが一杯になった時や、検索/分析クエリがGPUキャッシュを参照する時、`pgstrom.gpucache_apply_redo()`関数を実行した時に、まとめてGPUへ転送され、GPUの数千プロセッサコアが並列に更新ログをGPUキャッシュに適用します。
}
@en{
GPU Caches needs to satisfy two requirements: highly parallel update-based workloads and search/analytical queries on constantly up-to-date data.

In many systems, the CPU and GPU are connected via the PCI-E bus, and there is a reasonable delay in their communication. Therefore, synchronizing GPU Cache every time a row is updated in the target table will significantly degrade the transaction performance.

When a SQL command (INSERT, UPDATE, DELETE) that updates the table with GPU Cache is executed, the AFTER ROW trigger sends the updated contents to the GPU Service as update logs.
In the GPU Service, a dedicated worker for GPU Cache runs for each GPU, and it accumulates the received update logs in the log buffer.
Since this process does not involve any GPU call, it has little impact on transaction performance.

The update logs accumulated in the log buffer are transferred to the GPU in batches when the log buffer becomes full, when a search/analysis query refers to the GPU Cache, or when the `pgstrom.gpucache_apply_redo()` function is executed, then thousands of processor cores on the GPU apply them onto the GPU Cache in parallel.
}

@ja{
検索/分析クエリでGPUキャッシュの対象テーブルを参照する際には、テーブルからデータを読み出してGPUにロードするのではなく、既にGPUデバイスメモリ上に割当て済みのGPUキャッシュを参照します。これに先立って、未適用の更新ログが存在する場合、これらは全て、検索/分析クエリの実行前にGPUキャッシュへ適用されます。
また、GPUキャッシュ上の各行はトランザクションの可視性制御に必要な情報を保持しているため、検索/分析クエリの結果は、直接テーブルを参照した場合と同じ結果となり、問い合わせの一貫性は常に保持されています。
}
@en{
Search/analysis queries against the target table of GPU Cache do not load the table data, but refer the GPU Cache already allocated on the GPU device memory.
If there are any unapplied update logs at that time, they will all be applied to the GPU Cache prior to the execution of the search/analysis query.
In addition, each row on the GPU Cache keeps the information needed for transaction visibility control, so the results of a search/analysis query will be the same as if it were referring to the table directly, and the query will always be consistent.
}

@ja:###マルチGPU
@en:###Multi-GPUs

@ja{
GPUキャッシュは、1個のテーブルを複数のGPUにまたがってキャッシュする事ができます。
この場合、テーブルの各行は、行の物理位置（ctid）のハッシュ値によって、いずれか1個のGPUに振り分けられます。つまり、各GPUはテーブルの一部を保持し、全てのGPUを合わせるとテーブル全体となります。
検索/分析クエリは、GPUキャッシュを保持する全てのGPUで並列に実行され、その結果が統合されます。

これにより、1個のGPUのデバイスメモリには収まらない大きさのテーブルであっても、GPUキャッシュを設定する事ができます。
}
@en{
GPU Cache can cache a table across multiple GPUs.
In this case, each row of the table is distributed to one of the GPUs according to the hash value of its physical location (ctid). That is, each GPU keeps a part of the table, and all the GPUs together keep the entire table.
Search/analysis queries are executed on all the GPUs that keep the GPU Cache in parallel, then their results are merged.

This allows to configure GPU Cache even on a table that is too large to fit into the device memory of a single GPU.
}

@ja:###GPUメモリの割当て
@en:###GPU memory allocation

@ja{
GPUキャッシュの領域はCUDAのUnified Memory（Managed Memory）として確保されます。
トリガの`cache_size`オプションで指定した大きさは、各GPU上に予約される仮想的なアドレス空間の大きさであり、物理的なGPUデバイスメモリは、GPUキャッシュに実際にデータが格納されるにつれて消費されます。
仮想的な大きさと物理的な消費量は、後述の`pgstrom.gpucache_info()`関数で確認する事ができます。
}
@en{
The area of GPU Cache is allocated as CUDA's Unified Memory (Managed Memory).
The size specified by the `cache_size` option of the trigger is the size of the virtual address space reserved on each GPU, and the physical GPU device memory is consumed as the data is actually stored in the GPU Cache.
You can check the virtual size and the physical consumption using the `pgstrom.gpucache_info()` function described later.
}

@ja:###初期ロード
@en:###Initial loading

@ja{
GPUキャッシュを設定した直後や、PostgreSQLの再起動後には、GPUキャッシュはまだGPU上に構築されていません（`not_built`状態）。
この状態のテーブルを検索/分析クエリが最初に参照するか、テーブルが最初に更新されると、そのPostgreSQLバックエンドプロセスがテーブルをフルスキャンしてGPUキャッシュを構築します（初期ロード）。
初期ロードの実行中に同じテーブルを参照しようとした他のセッションは、初期ロードの完了を待ちます。

初期ロードはテーブルの大きさに応じて相応の時間を要する処理ですので、応答時間を重視するワークロードでは、運用開始前に一度テーブルを参照し、GPUキャッシュを構築しておく事をお勧めします。
}
@en{
Just after the configuration of GPU Cache, or after the restart of PostgreSQL, the GPU Cache is not built on the GPU yet (`not_built` phase).
When a search/analysis query first refers the table in this phase, or when the table is first updated, that PostgreSQL backend process scans the entire table and builds the GPU Cache (initial loading).
Other sessions that try to refer the same table during the initial loading wait for its completion.

Since the initial loading takes time according to the table size, we recommend to refer the table once before starting the operation to build the GPU Cache, if your workload is latency-sensitive.
}

@ja:##設定
@en:##Configuration

@ja{
GPUキャッシュを有効にするには、対象となるテーブルに対して
`pgstrom.gpucache_sync_trigger()`関数を実行するAFTER INSERT OR UPDATE OR DELETEの行トリガを設定します。
1個のテーブルに設定できるこのトリガは1個だけです。

レプリケーションのスレーブ側でGPUキャッシュを使用する場合、このトリガの発行モードが`ALWAYS`である事が必要です。

以下の例は、テーブル `dpoints` に対してGPUキャッシュを設定する例です。
}
@en{
To enable GPU Cache, configure a trigger that executes `pgstrom.gpucache_sync_trigger()` function
on AFTER INSERT OR UPDATE OR DELETE for each row.
Only one such trigger can be configured on a table.

If GPU Cache is used on the replication slave, the invocation mode of this trigger must be `ALWAYS`.

Below is an example to configure GPU Cache on the `dpoints` table.
}

```
=# create trigger row_sync after insert or update or delete on dpoints for row
                  execute function pgstrom.gpucache_sync_trigger();
=# alter table dpoints enable always trigger row_sync;
```

@ja:###GPUキャッシュのカスタマイズ
@en:###GPU Cache Customize

@ja{
GPUキャッシュの行トリガに引数として KEY=VALUE 形式のオプション文字列を与える事で、GPUキャッシュをカスタマイズする事ができます。
複数のオプションを指定する場合は、カンマで区切って並べます。

例えば、以下のGPUキャッシュは、GPU0とGPU1（`gpumask=3`）にそれぞれ4GBの領域を確保して作成しています。
}
@en{
You can customize GPU Cache by specifying an optional string in the form of KEY=VALUE as an argument to the row trigger of GPU Cache.
To specify multiple options, separate them by commas.

The following SQL statement is an example of creating a GPU Cache with 4GB of area on each of GPU0 and GPU1 (`gpumask=3`).
}

```
=# create trigger row_sync after insert or update or delete on dpoints for row
   execute function pgstrom.gpucache_sync_trigger('gpumask=3,cache_size=4g');
```

@ja{
行トリガの引数に与える事のできるオプションは以下の通りです。

`gpumask=MASK` (default: 全てのGPU)
:   GPUキャッシュを確保する対象のGPUを、10進数のビットマスクで指定します。例えば、GPU0だけであれば`1`、GPU1だけであれば`2`、GPU0とGPU1であれば`3`を指定します。
:   複数のGPUを指定した場合、テーブルの各行はこれらのGPUに分散して保持されます。

`cache_size=SIZE` (default: 512m)
:   GPUごとに確保するGPUキャッシュの大きさを指定します。単位として、k、m、gを指定できます。単位を省略した場合はバイト単位と見なします。最小値は64MBです。
:   GPUキャッシュには、コミット前の更新行や、更新/削除された古いバージョンの行も保持されるため、ある程度の余裕を持って指定する必要があります。
}
@en{
The options that can be given to the argument of the row trigger are shown below.

`gpumask=MASK` (default: all the GPUs)
:   Specify the target GPUs to allocate GPU Cache, as a bitmask in decimal. For example, `1` for GPU0 only, `2` for GPU1 only, and `3` for both of GPU0 and GPU1.
:   If multiple GPUs are specified, each row of the table is distributed to these GPUs.

`cache_size=SIZE` (default: 512m)
:   Specify the size of GPU Cache to be allocated for each GPU. You can use k, m and g as the unit. If the unit is omitted, the value is considered as bytes. The minimum value is 64MB.
:   GPU Cache also keeps the updated rows prior to commit and the old version of updated/deleted rows, so it should be specified with some margin.
}

@ja:###テーブル定義の変更
@en:###Changes of the table definition

@ja{
`ALTER TABLE`によるテーブル定義の変更、`CREATE OR REPLACE TRIGGER`や`ALTER TRIGGER`によるGPUキャッシュのトリガの変更（オプションの変更を含む）、`TRUNCATE`を実行すると、そのトランザクションのコミット時に既存のGPUキャッシュは無効となり、GPU上の領域はGPUキャッシュのワーカによって解放されます。
新しい定義に基づくGPUキャッシュは、次にテーブルが参照/更新された時に、改めて初期ロードによって構築されます。

`DROP TABLE`や、GPUキャッシュのトリガに対する`DROP TRIGGER`を実行した場合も、同様にコミット時にGPUキャッシュは無効となり、その領域は解放されます。
}
@en{
When you run `ALTER TABLE` to change the table definition, `CREATE OR REPLACE TRIGGER` or `ALTER TRIGGER` to change the trigger of GPU Cache (including its options), or `TRUNCATE`, the existing GPU Cache becomes invalid on the commit of the transaction, and its area on the GPU is released by the worker of GPU Cache.
The GPU Cache based on the new definition is built by the initial loading again, when the table is referred or updated next time.

When you run `DROP TABLE`, or `DROP TRIGGER` on the trigger of GPU Cache, the GPU Cache also becomes invalid on the commit, and its area is released.
}

@ja:###GPUキャッシュのオプション
@en:###GPU Cache Options

@ja{
GPUキャッシュに関連して、以下のPostgreSQL設定パラメータが定義されています。
}
@en{
Below are GPU Cache related PostgreSQL configuration parameters.
}

@ja{
`pg_strom.enable_gpucache`　（default: on）
:   GPUキャッシュが利用可能である場合、検索/分析系のクエリでGPUキャッシュを使用するかどうかを制御します。
:   この値が off になっていると、GPUキャッシュが存在していてもこれを無視し、テーブルから都度データを読み出そうとします。
:   なお、本設定はトリガによる更新ログの送出には影響しません。

`pg_strom.gpucache_max_relation_entries`　（default: 1024）
:   GPUキャッシュを設定できるテーブル数の上限です。通常、既定値から変更する必要はありません。
:   本パラメータはPostgreSQLの起動時にのみ設定できます。
}
@en{
`pg_strom.enable_gpucache` (default: on)
:   This option controls whether search/analytical queries will use GPU Cache or not.
:   If this value is off, the data will be read from the table each time, ignoring GPU Cache even if it is available.
:   Note that this setting has no effect on sending update logs by triggers.

`pg_strom.gpucache_max_relation_entries` (default: 1024)
:   The maximum number of tables that GPU Cache can be configured. Usually, no need to change from the default.
:   This parameter can be set only at the PostgreSQL startup.
}

@ja:##運用
@en:##Operations

@ja:###GPUキャッシュの利用を確認する
@en:###Confirm GPU Cache usage

@ja{
GPUキャッシュの参照は透過的に行われます。ユーザはキャッシュの有無を意識する必要はなく、PG-Stromが自動的に判定して処理を切り替えます。
GPUキャッシュを利用できるテーブルでは、テーブルからデータを読み出すためのI/Oコストが発生しないものとして実行計画が作成されます。

以下のクエリ実行計画は、GPUキャッシュの設定されたテーブル dpoints への参照を含むものです。
「GPU-Cache」フィールドに、このテーブルのGPUキャッシュの基本的な情報が表示されており、このクエリでは dpoints テーブルを読み出すのではなく、GPUキャッシュを参照してクエリを実行する事がわかります。

`cache-sz`に表示されているのはGPUごとのGPUキャッシュの大きさ、`gpumask`に表示されているのはGPUキャッシュを保持しているGPUのビットマスク（16進数）です。
}
@en{
GPU Cache is referred to transparently. The user does not need to be aware of the presence or absence of GPU Cache, and PG-Strom will automatically determine and switch the process.
For the table where GPU Cache is available, the query plan is built assuming no I/O cost to read the data from the table.

The following is the query plan for a query that refers to the table "dpoints" which has GPU Cache set.
The "GPU-Cache" field shows the basic information about GPU Cache of this table. We can see that the query is executed with referring to GPU Cache and not the "dpoints" table.

`cache-sz` shows the size of GPU Cache for each GPU, and `gpumask` shows the bitmask (in hexadecimal) of the GPUs that keep the GPU Cache.
}

```
=# explain select count(*) from dpoints where x < 50;
                                   QUERY PLAN
--------------------------------------------------------------------------------
 Aggregate  (cost=334.01..334.02 rows=1 width=8)
   ->  Custom Scan (GpuPreAgg) on dpoints  (cost=333.99..334.00 rows=1 width=8)
         GPU Projection: pgstrom.nrows()
         GPU Scan Quals: (x < '50'::double precision) [plan: 1000000 -> 497550]
         GPU Group Key:
         GPU-Cache: cache-sz: 1024MB, gpumask: 00000003
         Scan-Engine: GPU-Direct with 2 GPUs <0,1>
(7 rows)
```

@ja:###GPUキャッシュの状態を確認する
@en:###Check status of GPU Cache

@ja{
GPUキャッシュの現在の状態を確認するには`pgstrom.gpucache_info()`関数を使用します。
引数にテーブルを指定すると、そのテーブルのGPUキャッシュの状態をJSON形式で返します。GPUキャッシュが設定されていないテーブルの場合は NULL を返します。
引数を省略すると、全てのGPUキャッシュの状態と、その合計値をJSON形式で返します。
}
@en{
Use the `pgstrom.gpucache_info()` function to check the current state of GPU Cache.
If a table is given as argument, it returns the status of GPU Cache of the table in JSON format. It returns NULL if the table has no GPU Cache configured.
If the argument is omitted, it returns the status of all the GPU Caches and their total in JSON format.
}

```
=# select pgstrom.gpucache_info('dpoints');
{ "database_oid" : 190428, "table_oid" : 219029, "table_sig" : 2291764617,
  "table_name" : "dpoints", "gpu_mask" : "00000003", "cache_sz" : 1073741824,
  "phase" : "ready",
  "gpus" : [ { "gpu-id" : 0, "nitems" : 500432, "usage" : 36031104,
               "dead_nitems" : 0, "dead_space" : 0,
               "virtual_mem_sz" : 1073741824, "physical_mem_sz" : 147409352 },
             { "gpu-id" : 1, "nitems" : 499568, "usage" : 35968896,
               "dead_nitems" : 0, "dead_space" : 0,
               "virtual_mem_sz" : 1073741824, "physical_mem_sz" : 147340232 } ] }
```

@ja{
各フィールドの意味は以下の通りです。

- `database_oid`
    - GPUキャッシュを設定したテーブルの属するデータベースのOIDです。
- `table_oid`
    - GPUキャッシュを設定したテーブルのOIDです。必ずしも現在のデータベースとは限らない事に留意してください。
- `table_sig`
    - GPUキャッシュの一意性を示すハッシュ値です。例えば`ALTER TABLE`の前後などでこの値が変わる場合があります。
- `table_name`
    - GPUキャッシュを設定したテーブルの名前です。現在のデータベースに属するテーブルの場合のみ表示されます。
- `gpu_mask`
    - GPUキャッシュを保持しているGPUのビットマスク（16進数）です。
- `cache_sz`
    - GPUごとのGPUキャッシュの大きさ（バイト）です。
- `phase`
    - GPUキャッシュの状態で、以下のいずれかです。
        - `not_built` ... まだGPUキャッシュが構築されていない
        - `loading` ... 初期ロードの実行中
        - `ready` ... GPUキャッシュが利用可能
        - `corrupted` ... GPUキャッシュが破損しており、利用できない
- `gpus`
    - GPUごとのGPUキャッシュの状態です。`phase`が`corrupted`の場合は表示されません。
        - `gpu-id` ... GPUのデバイス番号
        - `nitems` ... GPUキャッシュ上の行数
        - `usage` ... GPUキャッシュ上の行が消費している領域の大きさ（バイト）
        - `dead_nitems` ... 更新/削除によって不要となった行の数
        - `dead_space` ... 更新/削除によって不要となった行が消費している領域の大きさ（バイト）
        - `virtual_mem_sz` ... GPUキャッシュとして予約された仮想的な領域の大きさ（バイト）
        - `physical_mem_sz` ... GPUキャッシュが実際に消費している物理的な領域の大きさ（バイト）

引数を省略した場合は、各テーブルのGPUキャッシュの状態が`caches`配列に格納され、併せてGPUキャッシュの数（`num_caches`）と、全GPUキャッシュの仮想的な領域の合計（`total_virtual_sz`）、物理的な領域の合計（`total_physical_sz`）が返されます。

なお、`gpus`に表示される統計情報は、GPUキャッシュのワーカが定期的に更新するものであり、必ずしも最新の値を反映しているとは限りません。
}
@en{
The meaning of each field is as follows:

- `database_oid`
    - The OID of the database to which the table with GPU Cache enabled exists.
- `table_oid`
    - The OID of the table with GPU Cache enabled. Note that the database this table exists in is not necessarily the database you are connected to.
- `table_sig`
    - A hash value indicating the uniqueness of GPU Cache. This value may change, for example, before and after executing `ALTER TABLE`.
- `table_name`
    - The name of the table with GPU Cache enabled. It is shown only if the table belongs to the current database.
- `gpu_mask`
    - The bitmask (in hexadecimal) of the GPUs that keep the GPU Cache.
- `cache_sz`
    - The size of GPU Cache for each GPU in bytes.
- `phase`
    - The phase of GPU Cache; one of the following.
        - `not_built` ... GPU Cache is not built yet
        - `loading` ... initial loading is in progress
        - `ready` ... GPU Cache is available
        - `corrupted` ... GPU Cache is corrupted and not available
- `gpus`
    - The status of GPU Cache for each GPU. It is not shown if `phase` is `corrupted`.
        - `gpu-id` ... device number of the GPU
        - `nitems` ... number of rows on the GPU Cache
        - `usage` ... size of the area consumed by the rows on the GPU Cache in bytes
        - `dead_nitems` ... number of rows that became unnecessary by update/delete
        - `dead_space` ... size of the area consumed by the rows that became unnecessary by update/delete in bytes
        - `virtual_mem_sz` ... size of the virtual area reserved for the GPU Cache in bytes
        - `physical_mem_sz` ... size of the physical area actually consumed by the GPU Cache in bytes

If the argument is omitted, the status of GPU Cache of each table is stored in the `caches` array, with the number of GPU Caches (`num_caches`), the total size of the virtual area (`total_virtual_sz`) and the total size of the physical area (`total_physical_sz`) of all the GPU Caches.

Note that the statistics shown in `gpus` are updated periodically by the worker of GPU Cache, so they do not always reflect the latest values.
}

@ja:###コンパクション
@en:###Compaction

@ja{
UPDATEやDELETEによって不要となった古いバージョンの行は、トランザクションのコミット後もGPUキャッシュ上の領域を消費し続けます。
GPUキャッシュの空き領域が不足して更新ログを適用できなくなると、GPUキャッシュは自動的にコンパクション（不要な行を取り除き、領域を詰め直す処理）を実行します。

`pgstrom.gpucache_compaction(regclass)`関数を使用すると、指定したテーブルのGPUキャッシュに対して、明示的にコンパクションを要求する事ができます。
}
@en{
The old version of rows that became unnecessary by UPDATE or DELETE keep consuming the area on the GPU Cache even after the commit of the transaction.
When the free area of the GPU Cache becomes insufficient to apply the update logs, the GPU Cache automatically runs compaction (a process to remove unnecessary rows and to pack the area).

The `pgstrom.gpucache_compaction(regclass)` function explicitly requests compaction of the GPU Cache of the specified table.
}

@ja:###GPUキャッシュの破損と復元
@en:###GPU Cache corruption and recovery

@ja{
コンパクションを行ってもGPUキャッシュの空き領域が不足する場合など、何らかの理由でGPUキャッシュに更新ログを適用できなかった場合、GPUキャッシュは破損（corrupted）状態に移行し、GPUキャッシュの領域は解放されます。

一度GPUキャッシュが破損すると、これを復元するまでは、検索/分析系のクエリでGPUキャッシュを参照する事はなくなり、テーブルから都度データを読み出すようになります。
（運悪く、検索/分析系のクエリが実行を開始した後にGPUキャッシュが破損した場合、そのクエリはエラーを返す事があります。）
また、初期ロードの途中でGPUキャッシュが破損した場合、初期ロードを実行していたSQLコマンドはエラーを返します。

GPUキャッシュを破損状態から復元するのは `pgstrom.gpucache_recovery(regclass)` 関数です。
更新ログを適用できなかった原因を取り除いた上でこの関数を実行すると、GPUキャッシュは`not_built`状態に戻り、次にテーブルが参照/更新された時に、改めて初期ロードによって構築されます。

例えば、`cache_size`で指定した領域に収まらない行数を挿入しようとした場合であれば、テーブルから一部の行を削除した後で`pgstrom.gpucache_recovery()`関数を実行します。
なお、トリガの定義を変更して`cache_size`を拡大した場合は、テーブル定義の変更と同様に新しいGPUキャッシュが構築されるため、`pgstrom.gpucache_recovery()`関数を実行する必要はありません。
}
@en{
If and when update logs could not be applied on the GPU cache by some reasons, like lack of free area even after compaction, GPU cache moves to the "corrupted" state, and its area is released.

Once GPU cache gets corrupted, search/analysis SQL does not reference the GPU cache and reads the data from the table each time, until it is recovered.
(If GPU cache gets corrupted after beginning of a search/analysis SQL unfortunately, this query may raise an error.)
In addition, if GPU cache gets corrupted during the initial loading, the SQL command that runs the initial loading raises an error.

The `pgstrom.gpucache_recovery(regclass)` function recovers the GPU cache from the corrupted state.
If you run this function after removal of the cause where update logs could not be applied, the GPU cache goes back to the `not_built` state, then it is built by the initial loading again when the table is referred or updated next time.

For example, if GPU cache gets corrupted because you tried to insert more rows than the area specified by `cache_size`, you delete a part of rows from the table, then run `pgstrom.gpucache_recovery()` function.
Note that if you reconfigure the trigger with expanded `cache_size`, a new GPU cache is built like the changes of the table definition, so you don't need to run `pgstrom.gpucache_recovery()` function.
}

@ja:###未適用の更新ログの適用
@en:###Apply the pending update logs

@ja{
未適用の更新ログは、検索/分析クエリがGPUキャッシュを参照する際に自動的に適用されるため、通常、ユーザが明示的に操作する必要はありません。
`pgstrom.gpucache_apply_redo()`関数を使用すると、全てのGPUキャッシュに対して、未適用の更新ログを適用するよう要求する事ができます。
}
@en{
The pending update logs are automatically applied when a search/analysis query refers to the GPU Cache, so usually, users don't need to operate them explicitly.
The `pgstrom.gpucache_apply_redo()` function requests to apply the pending update logs on all the GPU Caches.
}

@ja:##従来のバージョンからの変更点
@en:##Changes from the former versions

@ja{
PG-Strom v6.2のGPUキャッシュは、従来のバージョンから以下の点が変更されています。

- 1個のテーブルを複数のGPUにまたがってキャッシュできるようになりました。
- トリガのオプションは`gpumask`と`cache_size`の2つになりました。
    - `gpu_device_id`は、`gpumask`に置き換えられました。
    - `max_num_rows`は廃止され、GPUキャッシュの大きさは`cache_size`で指定するようになりました。
    - `redo_buffer_size`、`gpu_sync_interval`、`gpu_sync_threshold`は廃止されました。
- `pgstrom.gpucache_info`ビューは廃止され、JSON形式で状態を返す`pgstrom.gpucache_info()`関数に置き換えられました。
- `pgstrom.gpucache_apply_redo(regclass)`関数は、引数を取らない`pgstrom.gpucache_apply_redo()`関数に置き換えられました。
- 設定パラメータ`pg_strom.gpucache_auto_preload`は廃止されました。
- EXPLAINの出力は、`GPU Cache`フィールドから`GPU-Cache`フィールドに変更されました。
}
@en{
GPU Cache in PG-Strom v6.2 has the following changes from the former versions.

- A table can be cached across multiple GPUs.
- The options of the trigger are now `gpumask` and `cache_size`.
    - `gpu_device_id` is replaced by `gpumask`.
    - `max_num_rows` is removed, and the size of GPU Cache is specified by `cache_size`.
    - `redo_buffer_size`, `gpu_sync_interval` and `gpu_sync_threshold` are removed.
- `pgstrom.gpucache_info` view is removed, and replaced by `pgstrom.gpucache_info()` function that returns the status in JSON format.
- `pgstrom.gpucache_apply_redo(regclass)` function is replaced by `pgstrom.gpucache_apply_redo()` function that takes no arguments.
- The configuration parameter `pg_strom.gpucache_auto_preload` is removed.
- The EXPLAIN output is changed from `GPU Cache` field to `GPU-Cache` field.
}
