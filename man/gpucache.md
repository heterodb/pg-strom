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

比較的データサイズの小さな（～10GB程度）データセットで、更新頻度が高いうえに、しばしばリアルタイムのデータに対して検索/分析系のSQLを実行するというパターンを意図しています。

後述するログベースの同期メカニズムにより、並列度の高いトランザクショナルなワークロードを妨げることなくGPUキャッシュを最新の状態に保つ事が可能です。
その一方で、検索/分析系のSQLを実行する際には既にGPU上にデータがロードされているため、改めてテーブルからレコードを読み出したり、PCI-Eバスを介してデータを転送したりする事なく、SQLワークロードを実行する事ができるようになります。
}
@en{
GPU Cache is a function that reserves an area on the GPU device memory in advance and keeps a copy of the PostgreSQL table there.
This can be used to execute search/analysis SQL in real time for data that is relatively small(~10GB) and is frequently updated.
The log-based synchronization mechanism described below allows GPU Cache to be kept up-to-date without interfering with highly parallel and transactional workloads.
Nevertheless, you can process search/analytical SQL workloads on data already loaded on GPU Cache without reading the records from the table again or transferring the data over the PCI-E bus.
}

![GPU Cache Usage](./img/gpucache_usage.png)

@ja{
GPUキャッシュの典型的な利用シーンとしては、自動車や携帯電話といったモバイルデバイスの位置情報（現在位置）を時々刻々収集し、[GPU版PostGIS](../postgis/)などを用いて他のデータと突き合わせるといったケースが考えられます。
多数のデバイスから送出される位置情報の更新は極めて更新ヘビーなワークロードですが、一方で、最新の位置情報に基づいて検索/分析クエリを実行する必要もあるため、これら更新データを遅滞なくGPU側へ適用する必要があります。
データサイズには制約がありますが、GPUキャッシュは高頻度の更新と、高性能な検索/分析クエリの実行を両立する一つのオプションです。
}
@en{
A typical use case of GPU Cache is to join location data, such as the current position of a mobile device like a car or a cell phone, collected in real time with other data using [GPU-PostGIS](../postgis/).
The workload of updating location information sent out by many devices is extremely heavy. However, it also needs to be applied on the GPU side without delay in order to perform search/analysis queries based on the latest location information.
Although the size is limited, GPU Cache is one option to achieve both high frequency updates and high-performance search/analysis query execution.
}

@ja:##アーキテクチャ
@en:##Architecture

@ja{
GPUキャッシュでは、並列度の高い更新系ワークロードに対応することと、検索/分析クエリが常に最新のデータを参照するという2つの要件をクリアする必要があります。

多くのシステムではCPUとGPUはPCI-Eバスを介して接続され、その通信には相応のレイテンシが発生します。
そのため、GPUキャッシュの対象テーブルが1行更新されるたびにGPUキャッシュを同期していては、トランザクション性能に大きな影響を与えてしまいます。

GPUキャッシュを作成すると、GPUデバイスメモリ上にキャッシュ用のメモリ領域を確保するだけでなく、ホスト側共有メモリ上にREDOログバッファを作成します。
テーブルの更新を伴うSQLコマンド（INSERT、UPDATE、DELETE）を実行すると、AFTER ROWトリガによって更新内容がREDOログバッファにコピーされますが、この処理はGPUへの呼び出しを伴わない、CPUとRAMだけで完結する処理ですので、トランザクション性能への影響はほとんどありません。
}
@en{
GPU Caches needs to satisfy two requirements: highly parallel update-based workloads and search/analytical queries on constantly up-to-date data.
In many systems, the CPU and GPU are connected via the PCI-E bus, and there is a reasonable delay in their communication. Therefore, synchronizing GPU Cache every time a row is updated in the target table will significantly degrade the transaction performance.
Using GPU Cache allocates a "REDO Log Buffer" on the shared memory on the host side in addition to the area on the memory of the GPU.
When a SQL command (INSERT, UPDATE, DELETE) is executed to update a table, the updated contents are copied to the REDO Log Buffer by the AFTER ROW trigger. Since this process can be completed by CPU and RAM alone without any GPU call, it has little impact on transaction performance.
}

![GPU Cache Architecture](./img/gpucache_arch.png)

@ja{
REDOログバッファに未適用のREDOログが一定量たまるか、最後の書き込みから一定時間が経過すると、バックグラウンドワーカープロセス（GPU memory keeper）によって未適用のREDOログはGPUへロードされ、更新差分をGPUキャッシュに適用します。
この時、REDOログはまとめてGPUに転送され、さらにGPUの数千プロセッサコアが並列にREDOログを適用するため、通常は処理遅延が問題となる事はありません。
}

@en{
When a certain amount of unapplied REDO Log Entries accumulate in the REDO Log Buffer, or a certain amount of time has passed since the last write, it is loaded by a background worker process (GPU memory keeper) and applied to GPU Cache.
At this time, REDO Log Entries are transferred to the GPU in batches and processed in parallel by thousands of processor cores on the GPU, so delays caused by this process are rarely a problem.
}

@ja{
検索/分析クエリでGPUキャッシュの対象テーブルを参照する際には、テーブルからデータを読み出してGPUにロードするのではなく、既にGPUデバイスメモリ上に割当て済みのGPUキャッシュをマッピングして利用します。これに先立って、クエリの実行開始時点で未適用のREDOログが存在する場合、これらは全て、検索/分析クエリの実行前にGPUキャッシュへ適用されます。
そのため、検索/分析クエリが対象のGPUキャッシュをスキャンした結果は、直接テーブルを参照した場合と同じ結果を返す事となり、問い合わせの一貫性は常に保持されています。
}

@en{
Search/analysis queries against the target table in GPU Cache do not load the table data, but use the data mapped from GPU Cache pre-allocated on the GPU device memory.
If there are any unapplied REDO Logs at the start of the search/analysis query, they will all be applied to GPU Cache. 
This means that the results of a search/analysis query scanning the target GPU Cache will return the same results as if it were referring to the table directly, and the query will always be consistent.
}

@ja:##設定
@en:##Configuration

@ja{
GPUキャッシュを有効にするには、対象となるテーブルに対して
`pgstrom.gpucache_sync_trigger()`関数を実行するAFTER INSERT OR UPDATE OR DELETEの行トリガを設定します。

レプリケーションのスレーブ側でGPUキャッシュを使用する場合、このトリガの発行モードが`ALWAYS`である事が必要です。

以下の例は、テーブル `dpoints` に対してGPUキャッシュを設定する例です。
}
@en{
To enable GPU Cache, configure a trigger that executes `pgstrom.gpucache_sync_trigger()` function
on AFTER INSERT OR UPDATE OR DELETE for each row.

If GPU Cache is used on the replication slave, the invocation mode of this trigger must be `ALWAYS`.

Below is an example to configure GPU Cache on the `dpoints` table.
}

```
=# create trigger row_sync after insert or update or delete on dpoints_even for row
                  execute function pgstrom.gpucache_sync_trigger();
=# alter table dpoints_even enable always trigger row_sync;
```

@ja{
!!! Note
    PostgreSQL v12.x 以前のバージョンにおける追加設定
    
    PostgreSQL v12および以前のバージョンでGPUキャッシュを利用する場合、上記のトリガに加えて、
    `pgstrom.gpucache_sync_trigger()`関数を実行するBEFORE TRUNCATEの構文トリガの設定が必要です。
    
    レプリケーションのスレーブ側でGPUキャッシュを実行する場合、同様に、このトリガの発行モードが
    `ALWAYS`である事が必要です。
    
    PostgreSQL v13ではObject Access Hookが拡張され、拡張モジュールはトリガ設定なしで
    TRUNCATEの実行を捕捉できるようになりました。
    しかしそれ以前のバージョンでは、TRUNCATEを捕捉してGPUキャッシュの一貫性を保つには、
    BEFORE TRUNCATEの構文トリガが必要です。
}
@en{
!!! Note
    Additional configuration at PostgreSQL v12 or prior.
    
    In case when GPU Cache is used at PostgreSQL v12 or prior, you need to configure
    an additional BEFORE TRUNCATE statement trigger that executes `pgstrom.gpucache_sync_trigger()` function.
    If you want to use the GPU Cache on the replication slave, 

    If you use GPU Cache at the PostgreSQL v12 or prior, in a similar way, invocation mode of this trigger must have `ALWAYS`.
    
    PostgreSQL v13 enhanced its object-access-hook mechanism, so allows extension modules
    to capture execution of TRUNCATE without triggers configuration.
    On the other hand, the prior version still needs the BEFORE TRUNCATE statement trigger to keep consistency
    of GPU Cache by capture of TRUNCATE.
}

@ja{
以下は、PostgreSQL v12以前でGPUキャッシュを`dpoints`テーブルに設定する例です。
}
@en{
Below is an example to configure GPU Cache on the `dpoints` table at PostgreSQL v12 or prior.
}
    
```
=# create trigger row_sync after insert or update or delete on dpoints_even for row
                  execute function pgstrom.gpucache_sync_trigger();
=# create trigger stmt_sync before truncate on dpoints_even for statement
                  execute function pgstrom.gpucache_sync_trigger();
=# alter table dpoints_even enable always trigger row_sync;
=# alter table dpoints_even enable always trigger stmt_sync;
```


@ja:###GPUキャッシュのカスタマイズ
@en:###GPU Cache Customize

@ja{
GPUキャッシュの行トリガに引数として KEY=VALUE 形式のオプション文字列を与える事で、GPUキャッシュをカスタマイズする事ができます。
構文トリガの方ではありませんのでご注意ください。

例えば、以下のGPUキャッシュは行数の最大値が250万行、REDOログバッファのサイズを100MBとして作成しています。
}
@en{
You can customize GPU Cache by specifying an optional string in the form of KEY=VALUE as an argument to GPU Cache line trigger. Please note that where you should specify is not to the syntax trigger.
The following SQL statement is an example of creating a GPU Cache whose maximum row count is 2.5 million rows and the size of the REDO Log Buffer is 100MB.
}

```
=# create trigger row_sync after insert or update or delete on dpoints_even for row
   execute function pgstrom.gpucache_sync_trigger('max_num_rows=2500000,redo_buffer_size=100m');
```

@ja{
行トリガの引数に与える事のできるオプションは以下の通りです。

`gpu_device_id=GPU_ID` (default: 0)
:   GPUキャッシュを確保する対象のGPUデバイスIDを指定します。

`max_num_rows=NROWS`  (default: 10485760)
:   GPUキャッシュ上に確保できる行数を指定します。
:   PostgreSQLテーブルと同様に、GPUキャッシュでも可視性制御のためにコミット前の更新行を保持する必要があるため、ある程度の余裕を持って`max_num_rows`を指定する必要があります。なお、更新/削除された古いバージョンの行は、トランザクションのコミット後に解放されます。

`redo_buffer_size=SIZE`　（default: 160m）
:   REDOログバッファのサイズを指定します。単位として、k、m、gを指定できる。

`gpu_sync_interval=SECONDS`　（default: 5）
:   REDOログバッファへの最後の書き込みから SECONDS 秒経過すると、たとえ更新行数が少なくとも、REDOログをGPU側へ反映します。

`gpu_sync_threshold=SIZE`　（default: `redo_buffer_size`の25%）
:   REDOログバッファの書き込みのうち、未反映分の大きさが SIZE バイトに達すると、GPU側にREDOログを反映します。
:   単位としてk、m、gを指定できる。
}

@en{
The options that can be given to the argument of the line trigger are shown below.

`gpu_device_id=GPU_ID` (default: 0)
:   Specify the target GPU device ID to allocate GPU Cache.

`max_num_rows=NROWS` (default: 10485760)
:   Specify the number of rows that can be allocated on GPU Cache.
:   Just as with PostgreSQL tables, GPU Cache needs to retain updated rows prior to commit for visibility control, so `max_num_rows` should be specified with some margin. Note that the old version of the updated/deleted row will be released after the transaction is committed.

`redo_buffer_size=SIZE` (default: 160m)
:   Specify the size of REDO Log Buffer. You can use k, m and g as the unit.

`gpu_sync_interval=SECONDS` (default: 5)
:   If the specified time has passed since the last write to the REDO Log Buffer, REDO Log will be applied to the GPU, regardless of the number of rows updated.

`gpu_sync_threshold=SIZE` (default: 25% of `redo_buffer_size`)
:   When the unapplied REDO Log in the REDO Log Buffer reaches SIZE bytes, it is applied to the GPU side.
:   You can use k, m and g as the unit.
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
:   なお、本設定はトリガによるREDOログバッファへの追記には影響しません。
}
@en{
`pg_strom.enable_gpucache` (default: on)
:   This option controls whether search/analytical queries will use GPU Cache or not.
:   If this value is off, the data will be read from the table each time, ignoring GPU Cache even if it is available.
:   Note that this setting has no effect on REDO Log Buffer appending by triggers.
}
@ja{
`pg_strom.gpucache_auto_preload`　（default: NULL）
:   PostgreSQLの起動時/再起動時に、本設定パラメータで指定されたテーブルのGPUキャッシュを予め構築しておきます。
:   書式は `DATABASE_NAME.SCHEMA_NAME.TABLE_NAME` で、複数個のテーブルを指定する場合はこれをカンマ区切りで並べます。
:   まだGPUデバイス上にGPUキャッシュが構築されていない場合、対象テーブルをフルスキャンしてGPU側へ転送するのは、最初にGPUキャッシュにアクセスしようと試みたPostgreSQLバックエンドプロセスの役割です。これは通常、相応の時間を要する処理ですが、このオプションにロードすべきテーブルを記載しておくことで、検索/分析クエリの初回実行時に長々と待たされる事を抑止できます。
:   なお、この設定パラメータに '*' と指定した場合、GPUキャッシュを持つ全てのテーブルの内容を順にGPUへロードしようと試みます。この時、バックグラウンドワーカは全てのデータベースに順にアクセスしていく事となりますが、postmasterに再起動を促すため終了コード 1 を返します。
:   以下のように、サーバの起動ログに「GPUCache Startup Preloader」が終了コード 1 で終了したと出力されますが、これは異常ではありません
}
@en{
`pg_strom.gpucache_auto_preload` (default: NULL)
:   When PostgreSQL is started/restarted, GPU Cache for the table specified by this parameter will be built in advance.
:   The value should be in the format: `DATABASE_NAME.SCHEMA_NAME.TABLE_NAME`. To specify multiple tables, separate them by commas.
:   If GPU Cache is not built, the PostgreSQL backend process that first attempts to access GPU Cache will scan the entire target table and transfer it to the GPU. This process usually takes a considerable amount of time. However, by specifying the tables that should be loaded in this option, you can avoid waiting too long the first time you run a search/analysis query.
:   If this parameter is set to '*', it will attempt to load the contents of all tables with GPU Cache into the GPU in order. At this time, the background worker will access all the databases in order, and will return exit code 1 to prompt the postmaster to restart.
:   The server startup log will show that the "GPUCache Startup Preloader" exited with exit code 1 as follows, but this is not abnormal.
}

```
 LOG:  database system is ready to accept connections
 LOG:  background worker "GPUCache Startup Preloader" (PID 856418) exited with exit code 1
 LOG:  background worker "GPUCache Startup Preloader" (PID 856427) exited with exit code 1
 LOG:  create GpuCacheSharedState dpoints:164c95f71
 LOG:  gpucache: AllocMemory dpoints:164c95f71 (main_sz=772505600, extra_sz=0)
 LOG:  gpucache: auto preload 'public.dpoints' (DB: postgres)
 LOG:  create GpuCacheSharedState mytest:1773a589b
 LOG:  gpucache: auto preload 'public.mytest' (DB: postgres)
 LOG:  gpucache: AllocMemory mytest:1773a589b (main_sz=675028992, extra_sz=0)

```

@ja:##運用
@en:##Operations

@ja:###GPUキャッシュの利用を確認する
@en:###Confirm GPU Cache usage

@ja{
GPUキャッシュの参照は透過的に行われます。ユーザはキャッシュの有無を意識する必要はなく、PG-Stromが自動的に判定して処理を切り替えます。

以下のクエリ実行計画は、GPUキャッシュの設定されたテーブル dpoints への参照を含むものです。下から3行目の「GPU Cache」フィールドに、このテーブルのGPUキャッシュの基本的な情報が表示されており、このクエリでは dpoints テーブルを読み出すのではなく、GPUキャッシュを参照してクエリを実行する事がわかります。

なお、`max_num_rows`に表示されているのはGPUキャッシュの保持できる最大の行数、`main`に表示されているのはGPUキャッシュの固定長フィールド用の領域の大きさ、`extra`に表示されているのは可変長データ用の領域の大きさです。
}
@en{
GPU Cache is referred to transparently. The user does not need to be aware of the presence or absence of GPU Cache, and PG-Strom will automatically determine and switch the process.

The following is the query plan for a query that refers to the table "dpoints" which has GPU Cache set.
The 3rd row from the bottom, in the "GPU Cache" field, shows the basic information about GPU Cache of this table. We can see that the query is executed with referring to GPU Cache and not the "dpoints" table.

Note that the meaning of each item is as follows: `max_num_rows` indicates the maximum number of rows that GPU Cache can hold; `main` indicates the size of the area in GPU Cache for fixed-length fields; `extra` indicates the size of the area for variable-length data. 
}

```
=# explain
   select pref, city, count(*)
     from giscity g, dpoints d
    where pref = 'Tokyo'
      and st_contains(g.geom,st_makepoint(d.x, d.y))
    group by pref, city;
                                               QUERY PLAN
--------------------------------------------------------------------------------------------------------
 HashAggregate  (cost=5638809.75..5638859.99 rows=5024 width=29)
   Group Key: g.pref, g.city
   ->  Custom Scan (GpuPreAgg)  (cost=5638696.71..5638759.51 rows=5024 width=29)
         Reduction: Local
         Combined GpuJoin: enabled
         GPU Preference: GPU0 (NVIDIA Tesla V100-PCIE-16GB)
         ->  Custom Scan (GpuJoin) on dpoints d  (cost=631923.57..5606933.23 rows=50821573 width=21)
               Outer Scan: dpoints d  (cost=0.00..141628.18 rows=7999618 width=16)
               Depth 1: GpuGiSTJoin(nrows 7999618...50821573)
                        HeapSize: 3251.36KB
                        IndexFilter: (g.geom ~ st_makepoint(d.x, d.y)) on giscity_geom_idx
                        JoinQuals: st_contains(g.geom, st_makepoint(d.x, d.y))
               GPU Preference: GPU0 (NVIDIA Tesla V100-PCIE-16GB)
               GPU Cache: NVIDIA Tesla V100-PCIE-16GB [max_num_rows: 12000000, main: 772.51M, extra: 0]
               ->  Seq Scan on giscity g  (cost=0.00..8929.24 rows=6353 width=1883)
                     Filter: ((pref)::text = 'Tokyo'::text)
(16 rows)

```

@ja:###GPUキャッシュの状態を確認する
@en:###Check status of GPU Cache

@ja{
GPUキャッシュの現在の状態を確認するには`pgstrom.gpucache_info`ビューを使用します。
}
@en{
Use the `pgstrom.gpucache_info` view to check the current state of GPU Cache.
}

```
=# select * from pgstrom.gpucache_info ;
 database_oid | database_name | table_oid | table_name | signature  | refcnt | corrupted | gpu_main_sz | gpu_extra_sz |       redo_write_ts        | redo_write_nitems | redo_write_pos | redo_read_nitems | redo_read_pos | redo_sync_pos |  config_options
--------------+---------------+-----------+------------+------------+--------+-----------+-------------+--------------+----------------------------+-------------------+----------------+------------------+---------------+---------------+------------------------------------------------------------------------------------------------------------------------
        12728 | postgres      |     25244 | mytest     | 6295279771 |      3 | f         |   675028992 |            0 | 2021-05-14 03:00:18.623503 |            500000 |       36000000 |           500000 |      36000000 |      36000000 | gpu_device_id=0,max_num_rows=10485760,redo_buffer_size=167772160,gpu_sync_interval=5000000,gpu_sync_threshold=41943040
        12728 | postgres      |     25262 | dpoints    | 5985886065 |      3 | f         |   772505600 |            0 | 2021-05-14 03:00:18.524627 |           8000000 |      576000192 |          8000000 |     576000192 |     576000192 | gpu_device_id=0,max_num_rows=12000000,redo_buffer_size=167772160,gpu_sync_interval=5000000,gpu_sync_threshold=41943040
(2 rows)
```

@ja{
このビューで表示されるGPUキャッシュの状態は、その時点で初期ロードが終わっており、GPUデバイスメモリ上に領域が確保されているものだけである事に留意してください。
つまり、トリガ関数が設定されているが初期ロードが終わっていない（まだ誰もアクセスしていない）場合、潜在的に確保されうるGPUキャッシュはまだ`pgstrom.gpucache_info`には現れません。
}
@en{
Note that `pgstrom.gpucache_info` will only show the status of GPU Caches that have been initially loaded and have space allocated on the GPU device memory at that time. In other words, if the trigger function is set but not yet initially loaded (no one has accessed it yet), the potentially allocated GPU Cache will not be shown yet.
}

@ja{
各フィールドの意味は以下の通りです。

- `database_oid`
    - GPUキャッシュを設定したテーブルの属するデータベースのOIDです
- `database_name`
    - GPUキャッシュを設定したテーブルの属するデータベースの名前です
- `table_oid`
    - GPUキャッシュを設定したテーブルのOIDです。必ずしも現在のデータベースとは限らない事に留意してください。
- `table_name`
    - GPUキャッシュを設定したテーブルの名前です。必ずしも現在のデータベースとは限らない事に留意してください。
- `signature`
    - GPUキャッシュの一意性を示すハッシュ値です。例えば`ALTER TABLE`の前後などでこの値が変わる場合があります。
- `refcnt`
    - GPUキャッシュの参照カウンタです。これは必ずしも最新の値を反映しているとは限りません。
- `corrupted`
    - GPUキャッシュの内容が破損しているかどうかを示します。
- `gpu_main_sz`
    - GPUキャッシュ上に確保された固定長データ用の領域のサイズです。
- `gpu_extra_sz`
    - GPUキャッシュ上に確保された可変長データ用の領域のサイズです。
- `redo_write_ts`
    - REDOログバッファを最後に更新した時刻です。
- `redo_write_nitems`
    - REDOログバッファに書き込まれたREDOログの総数です。
- `redo_write_pos`
    - REDOログバッファに書き込まれたREDOログの総バイト数です。
- `redo_read_nitems`
    - REDOログバッファから読み出し、GPUに適用されたREDOログの総数です。
- `redo_read_pos`
    - REDOログバッファから読み出し、GPUに適用されたREDOログの総バイト数です。
- `redo_sync_pos`
    - REDOログバッファに書き込まれたREDOログのうち、既にGPUキャッシュへの適用をバックグラウンドワーカにリクエストした位置です。
    - REDOログバッファの残り容量が逼迫してきた際に、多数のセッションが同時に非同期のリクエストを発生させる事を避けるため、内部的に使用されます。
- `config_options`
    - GPUキャッシュのオプション文字列です。
}

@en{
The meaning of each field is as follows:

- `database_oid`
    - The OID of the database to which the table with GPU Cache enabled exists.
- `database_name`
    - The name of the database to which the table with GPU Cache enabled exists.
- `table_oid`
    - The OID of the table with GPU Cache enabled. Note that the database this table exists in is not necessarily the database you are connected to.
- `table_name`
    - The name of the table with GPU Cache enabled. Note that the database this table exists in is not necessarily the database you are connected to.
- `signature`
    - A hash value indicating the uniqueness of GPU Cache. This value may change, for example, before and after executing `ALTER TABLE`.
- `refcnt`
    - Reference counter of the GPU Cache. It does not always reflect the latest value.
- `corrupted`
    - Shows whether the GPU Cache is corrupted.
- `gpu_main_sz`
    - The size of the area reserved in GPU Cache for fixed-length data.
- `gpu_extra_sz`
    - The size of the area reserved in GPU Cache for variable-length data.
- `redo_write_ts`
    - The time when the REDO Log Buffer was last updated.
- `redo_write_nitems`
    - The total number of REDO Logs in the REDO Log Buffer.
- `redo_write_pos`
    - The total size (in bytes) of the REDO Logs in the REDO Log Buffer.
- `redo_read_nitems`
    - The total number of REDO Logs read from the REDO Log Buffer and applied to the GPU.
- `redo_read_pos`
    - The total size (in bytes) of REDO Logs read from the REDO Log Buffer and applied to the GPU.
- `redo_sync_pos`
    - The position of the REDO Log which is scheduled to be applied to GPU Cache by the background worker on the REDO Log Buffer.
    - This is used internally to avoid a situation where many sessions generate asynchronous requests at the same time when the remaining REDO Log Buffer is running out.
- `config_options`
    - The optional string to customize GPU Cache.
}


@ja:###GPUキャッシュの破損と復元
@en:###GPU Cache corruption and recovery

@ja{
GPUキャッシュに`max_num_rows`で指定した以上の行数を挿入しようとしたり、可変長データのバッファ長が肥大化しすぎたり、
といった理由でGPUキャッシュにREDOログを適用できなかった場合、GPUキャッシュは破損（corrupted）状態に移行します。

一度GPUキャッシュが破損すると、これを手動で復旧するまでは、検索/分析系のクエリでGPUキャッシュを参照する事はなくなり、
また、テーブルの更新に際してもREDOログの記録を行わなくなります。
（運悪く、検索/分析系のクエリが実行を開始した後にGPUキャッシュが破損した場合、そのクエリはエラーを返す事があります。）

GPUバッファを破損状態から復元するのは `pgstrom.gpucache_recovery(regclass)` 関数です。
REDOログを適用できなかった原因を取り除いた上でこの関数を実行すると、再度、GPUキャッシュの初期ロードを行い、元の状態への
復旧を試みます。

例えば、`max_num_rows` で指定した以上の行数を挿入しようとした場合であれば、トリガの定義を変更して `max_num_rows` 設定を
拡大するか、テーブルから一部の行を削除した後で、`pgstrom.gpucache_recovery()`関数を実行するという事になります。
}
@en{
If and when REDO logs could not be applied on the GPU cache by some reasons, like insertion of more rows than the `max_num_rows` configuration, or too much consumption of variable-length data buffer, GPU cache moves to the "corrupted" state.

Once GPU cache gets corrupted, search/analysis SQL does not reference the GPU cache, and table updates stops writing REDO log.
(If GPU cache gets corrupted after beginning of a search/analysis SQL unfortunately, this query may raise an error.)

The `pgstrom.gpucache_recovery(regclass)` function recovers the GPU cache from the corrupted state.
If you run this function after removal of the cause where REDO logs could not be applied, it runs initial-loading of the GPU cache again, then tries to recover the GPU cache.

For example, if GPU cache gets corrupted because you tried to insert more rows than the `max_num_rows`, you reconfigure the trigger with expanded `max_num_rows` configuration or you delete a part of rows from the table, then runs `pgstrom.gpucache_recovery()` function.
}
