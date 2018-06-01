@ja:#インメモリ列キャッシュ
@en:#In-memory Columnar Cache

@ja:##概要
@en:##Overview

@ja{
PG-Stromはプロセッサへ高速にデータを供給するためのストレージ関連機能をもう一つ持っています。

インメモリ列キャッシュは、対象テーブルのデータブロックを読み出し、PostgreSQL標準のデータ形式である行データから集計・解析ワークロードに適した列データ形式へと変換し、メモリ上にキャッシュする機能です。

SSD-to-GPUダイレクトSQL実行とは異なり、この機能を利用するには特別なハードウェアは必要ありません。しかし一方で、現在もなおRAMの容量はSSDよりも小さく、目安としてはシステムRAMサイズの60%～75%程度の「大規模でないデータセット」を取り扱うのに向いた機能です。
}
@en{
PG-Strom has one another feature related to storage to supply processors data stream.

In-memory columnar cache reads data blocks of the target table, convert the row-format of PostgreSQL to columnar format which is suitable for summary and analytics, and cache them on memory.

This feature requires no special hardware like SSD-to-GPU Direct SQL Execution, on the other hands, RAM capacity is still smaller than SSD, so this feature is suitable to handle "not a large scale data set" up to 60%-75% of the system RAM size.
}

@ja{
本機能は「列ストア」ではありません。すなわち、列データに変換しキャッシュされた内容は例えばPostgreSQLサーバプロセスを再起動すれば消えてしまいます。また、キャッシュされた領域を更新するような`UPDATE`文を実行すると、PG-Stromは当該キャッシュを消去します。
これは、列データ形式は本質的に更新ワークロードに弱い事を踏まえた上での設計です。つまり、行ストアの更新に対して整合性を保ったまま列ストアを更新しようとすると、書き込み性能の大幅な劣化は不可避です。一方で、単純に更新されたブロックを含む列キャッシュを消去（invalidation）するだけであれば、ほとんど処理コストはかかりません。
PG-Stromは行データであっても列データであっても、起動するGPUプログラムを変更するだけで対応可能です。すなわち、列キャッシュが消去され、通常通りPostgreSQLのshared bufferからデータを読み出さざるを得ない状況であっても柔軟に対応する事ができるのです。
}
@en{
This feature is not "a columnar store". It means cached and converted data blocks are flashed once PostgreSQL server process has restarted for example. When any cached rows get updated, PG-Strom invalidates the columnar cache block which contains the updated rows.
This design on the basis that columnar format is vulnerable to updating workloads. If we try to update columnar-store with keeping consistency towards update of row-store, huge degradation of write performance is not avoidable. On the other hands, it is lightweight operation to invalidate the columnar cache block which contains the updated row.
PG-Strom can switch GPU kernels to be invoked for row- or columnar-format according to format of the loading data blocks. So, it works flexibly, even if a columnar cache block gets invalidated thus PG-Strom has to load data blocks from the shared buffer of PostgreSQL.
}

![overview of in-memory columnar cache](./img/ccache-overview.png)


@ja:##初期設定
@en:##System Setup

@ja:###列キャッシュの格納先
@en:###Location of the columnar cache

@ja{
`pg_strom.ccache_base_dir`パラメータによって列キャッシュの格納先を指定する事ができます。デフォルト値は`/dev/shm`で、これは一般的なLinxディストリビューションにおいて`tmpfs`が配置されているパスであり、この配下に作成されたファイルは二次記憶装置のバッキングストアを持たない揮発性のデータとなります。

このパラメータを変更する事で、例えばNVMe-SSD等、より大容量かつリーズナブルに高速なストレージ領域をバッキングストアとする列キャッシュを構築する事ができます。ただし、列キャッシュの更新はたとえ一行であってもその前後の領域を含むチャンク全体（128MB単位）の無効化を引き起こす事は留意してください。I/Oを伴う読み書きが頻発するような状況になると、意図しない性能劣化を招く可能性があります。
}
@en{
The `pg_strom.ccache_base_dir` parameter allows to specify the path to store the columnar cache. The default is `/dev/shm` where general Linux distribution mounts `tmpfs` filesystem, so files under the directory are "volatile", with no backing store.

Custom configuration of the parameter enables to construct columnar cache on larger and reasonably fast storage, like NVMe-SSD, as backing store. However, note that update of the cached rows invalidates whole of the chunk (128MB) which contains the updated rows. It may lead unexpected performance degradation, if workloads have frequent read / write involving I/O operations.
}
@ja:###列キャッシュビルダの設定
@en:###Columnar Cache Builder Configuration

@ja{
PG-Stromは一つまたは複数のバックグラウンドワーカーを使用して、インメモリ列キャッシュを非同期かつ自動的に構築する事ができます。この処理を行うバックグラウンドワーカーを列キャッシュビルダーと呼びます。

列キャッシュビルダーは、ユーザのSQLを処理するセッションの動作とは非同期に、指定されたデータベース内のテーブルのうち列キャッシュを構築すべき対象をラウンドロビンでスキャンし、これを列データへと変換した上でキャッシュします。

一度列キャッシュが構築されると、他の全てのバックエンドからこれを参照する事ができます。一般的なディスクキャッシュのメカニズムとは異なり、列キャッシュが構築されていない領域へのアクセスであっても、列キャッシュをオンデマンドで作成する事はありません。この場合は、通常のPostgreSQLのストレージシステムを通して行データを参照する事となります。
}
@en{
PG-Strom can build in-memory columnar cache automatically and asynchronously using one or multiple background workers. These background workers are called columnar cache builder.

Columnar cache builder scans the target tables to construct columnar cache in the specified database, by round-robin, then converts to columnar format and keep it on the cache. It is an asynchronous job from the backend process which handles user's SQL.

Once a columnar cache is built, any other backend process can reference them. PG-Strom never construct columnar cache on demand, unlike usual disk cache mechanism, even if it is access to the area where columnar cache is not built yet. In this case, PG-Strom loads row-data through the normal storage system of PostgreSQL.
}
@ja{
列キャッシュビルダの数は起動時に決まっていますので、これを増やすには後述の`pg_strom.ccache_num_builders`パラメータを設定し、PostgreSQLの再起動が必要です。
また、列キャッシュビルダは特定のデータベースに紐付けられますので、複数のデータベースで列キャッシュを使用する場合には、少なくともデータベース数以上の列キャッシュビルダが存在している事が必要です。

列キャッシュビルダを紐づけるデータベースを指定するには、`pg_strom.ccache_databases`パラメータを指定します。
このパラメータの指定には特権ユーザ権限が必要ですが、PostgreSQLの実行中にも変更する事が可能です。（もちろん、`postgresql.conf`に記載して起動時に設定する事も可能です。）

データベース名をカンマ区切りで指定すると、列キャッシュビルダが順番に指定したデータベースに関連付けられていきます。例えば、列キャッシュビルダが5プロセス存在し、`postgres,my_test,benchmark`という3つのデータベースを`pg_strom.ccache_databases`に指定した場合、`postgres`および`my_test`データベースには2プロセスの、`benchmark`データベースには1プロセスの列キャッシュビルダが割り当てられる事になります。
}
@en{
The number of columnar cache builders are fixed on the startup, so you need to setup `pg_strom.ccache_num_builders` parameters then restart PostgreSQL to increase the number of workers.

The `pg_strom.ccache_databases` parameter configures the databases associated with columnar cache builders.
It requires superuser privilege to setup, and is updatable on PostgreSQL running. (Of course, it is possible to assign by `postgresql.conf` configuration on startup.)

Once a comma separated list of database names are assigned, columnar cache builders are associated to the specified databases in rotation. For example, if 5 columnar cache builders are running then 3 databases (`postgres,my_test,benchmark`) are assigned on the `pg_strom.ccache_databases`, 2 columnar cache builders are assigned on the `postgres` and `my_test` database for each, and 1 columnar cache builder is assigned on the `benchmark` database.
}


@ja:###対象テーブルの設定
@en:###Source Table Configuration

@ja{
DB管理者は列キャッシュに格納すべきテーブルを予め指定する必要があります。

SQL関数`pgstrom_ccache_enabled(regclass)`は、引数で指定したテーブルを列キャッシュの構築対象に加えます。
逆に、SQL関数`pgstrom_ccache_disabled(regclass)`は、引数で指定したテーブルの列キャッシュの構築対象から外します。

内部的には、これらの操作は対象テーブルに対して更新時のキャッシュ無効化を行うトリガ関数の設定として実装されています。
つまり、キャッシュを無効化する手段を持たないテーブルに対しては列キャッシュを作成しないという事です。
}
@en{
DBA needs to specify the target tables to build columnar cache.

A SQL function `pgstrom_ccache_enabled(regclass)` adds the supplied table as target to build columnar cache.
Other way round, a SQL function `pgstrom_ccache_disabled(regclass)` drops the supplied table from the target to build.

Internally, it is implemented as a special trigger function which invalidate columnar cache on write to the target tables.
It means we don't build columnar cache on the tables which have no way to invalidate columnar cache.
}

```
postgres=# select pgstrom_ccache_enabled('t0');
 pgstrom_ccache_enabled
------------------------
 enabled
(1 row)
```

@ja:##運用
@en:##Operations

@ja:###列キャッシュの状態を確認する
@en:###Check status of columnar cache

@ja{
列キャッシュの状態を確認するには`pgstrom.ccache_info`システムビューを使用します。

チャンク単位で、テーブル、ブロック番号やキャッシュの作成時刻、最終アクセス時刻などを参照する事ができます。
}
@en{
`pgstrom.ccache_info` provides the status of the current columnar cache.

You can check the table, block number, cache creation time and last access time per chunk.
}

```
contrib_regression_pg_strom=# SELECT * FROM pgstrom.ccache_info ;
 database_id | table_id | block_nr | nitems  |  length   |             ctime             |             atime
-------------+----------+----------+---------+-----------+-------------------------------+-------------------------------
       13323 | 25887    |   622592 | 1966080 | 121897472 | 2018-02-18 14:31:30.898389+09 | 2018-02-18 14:38:43.711287+09
       13323 | 25887    |   425984 | 1966080 | 121897472 | 2018-02-18 14:28:39.356952+09 | 2018-02-18 14:38:43.514788+09
       13323 | 25887    |    98304 | 1966080 | 121897472 | 2018-02-18 14:28:01.542261+09 | 2018-02-18 14:38:42.930281+09
         :       :             :         :          :                :                               :
       13323 | 25887    |    16384 | 1963079 | 121711472 | 2018-02-18 14:28:00.647021+09 | 2018-02-18 14:38:42.909112+09
       13323 | 25887    |   737280 | 1966080 | 121897472 | 2018-02-18 14:34:32.249899+09 | 2018-02-18 14:38:43.882029+09
       13323 | 25887    |   770048 | 1966080 | 121897472 | 2018-02-18 14:28:57.321121+09 | 2018-02-18 14:38:43.90157+09
(50 rows)
```


@ja:###列キャッシュの利用を確認する
@en:###Check usage of columnar cache

@ja{
あるクエリが列キャッシュを使用する可能性があるかどうか、`EXPLAIN`コマンドを使用して確認する事ができます。

以下のクエリは、テーブル`t0`と`t1`をジョインしますが、`t0`に対するスキャンを含む`Custom Scan (GpuJoin)`に`CCache: enabled`と表示されています。
これは、`t0`に対するスキャンの際に列キャッシュを使用する可能性がある事を示しています。ただし、実際に使われるかどうかはクエリが実行されるまで分かりません。並行する更新処理の影響で、列キャッシュが破棄される可能性もあるからです。
}
@en{
You can check whether a particular query may reference columnar cache, or not, using `EXPLAIN` command.

The query below joins the table `t0` and `t1`, and the `Custom Scan (GpuJoin)` which contains scan on the `t0` shows `CCache: enabled`.
It means columnar cache may be referenced at the scan on `t0`, however, it is not certain whether it is actually referenced until query execution. Columnar cache may be invalidated by the concurrent updates.
}
```
postgres=# EXPLAIN SELECT id,ax FROM t0 NATURAL JOIN t1 WHERE aid < 1000;

                                  QUERY PLAN
-------------------------------------------------------------------------------
 Custom Scan (GpuJoin) on t0  (cost=12398.65..858048.45 rows=1029348 width=12)
   GPU Projection: t0.id, t1.ax
   Outer Scan: t0  (cost=10277.55..864623.44 rows=1029348 width=8)
   Outer Scan Filter: (aid < 1000)
   Depth 1: GpuHashJoin  (nrows 1029348...1029348)
            HashKeys: t0.aid
            JoinQuals: (t0.aid = t1.aid)
            KDS-Hash (size: 10.78MB)
   CCache: enabled
   ->  Seq Scan on t1  (cost=0.00..1935.00 rows=100000 width=12)
(10 rows)
```

@ja{
`EXPLAIN ANALYZE`コマンドを使用すると、クエリが実際に列キャッシュを何回参照したのかを知る事ができます。

先ほどのクエリを実行すると、`t0`に対するスキャンを含む`Custom Scan (GpuJoin)`に`CCache Hits: 50`と表示されています。
これは、列キャッシュへの参照が50回行われた事を示しています。列キャッシュのチャンクサイズは128MBですので、合計で6.4GB分のストレージアクセスが列キャッシュにより代替された事となります。
}
@en{
`EXPLAIN ANALYZE` command tells how many times columnar cache is referenced during the query execution.

After the execution of this query, `Custom Scan (GpuJoin)` which contains scan on `t0` shows `CCache Hits: 50`.
It means that columnar cache is referenced 50 times. Because the chunk size of columnar cache is 128MB, storage access is replaced to the columnar cache by 6.4GB.
}
```
postgres=# EXPLAIN ANALYZE SELECT id,ax FROM t0 NATURAL JOIN t1 WHERE aid < 1000;

                                    QUERY PLAN

-------------------------------------------------------------------------------------------
 Custom Scan (GpuJoin) on t0  (cost=12398.65..858048.45 rows=1029348 width=12)
                              (actual time=91.766..723.549 rows=1000224 loops=1)
   GPU Projection: t0.id, t1.ax
   Outer Scan: t0  (cost=10277.55..864623.44 rows=1029348 width=8)
                   (actual time=7.129..398.270 rows=100000000 loops=1)
   Outer Scan Filter: (aid < 1000)
   Rows Removed by Outer Scan Filter: 98999776
   Depth 1: GpuHashJoin  (plan nrows: 1029348...1029348, actual nrows: 1000224...1000224)
            HashKeys: t0.aid
            JoinQuals: (t0.aid = t1.aid)
            KDS-Hash (size plan: 10.78MB, exec: 64.00MB)
   CCache Hits: 50
   ->  Seq Scan on t1  (cost=0.00..1935.00 rows=100000 width=12)
                       (actual time=0.011..13.542 rows=100000 loops=1)
 Planning time: 23.390 ms
 Execution time: 1409.073 ms
(13 rows)
```

@ja:### `DROP DATABASE`コマンドに関する注意事項
@en:### Attension for `DROP DATABASE` command

@ja{
列キャッシュビルダを使用して非同期に列キャッシュを構築する場合、内部的にはバックグラウンドワーカープロセスが指定されたデータベースに接続し続ける事になります。
`DROP DATABASE`コマンドを使用してデータベースを削除する時、PostgreSQLは当該データベースに接続しているセッションが存在するかどうかをチェックします。この時、ユーザセッションが一つも存在していないにも関わらず、列キャッシュビルダがデータベースへの接続を保持し続ける事で`DROP DATABASE`コマンドが失敗してしまいます。
}
@en{
When columnar cache builder constructs columnar cache asynchronously, background worker process has internally connected to the specified database.
When `DROP DATABASE` command tries to drop a database, PostgreSQL checks whether any session connects to the database. At that time, even if no user session connects to the database, `DROP DATABASE` will fail by columnar cache builder which keeps connection to the database.
}
@ja{
これを避けるには、`DROP DATABASE`コマンドの実行前に、`pg_strom.ccache_databases`パラメータから当該データベースを除外してください。列キャッシュビルダは直ちに再起動し、新しい設定に基づいてデータベースへの接続を試みます。
}
@en{
Please remove the database name from the `pg_strom.ccache_databases` parameter prior to execution of `DROP DATABASE` command.
Columnar cache builder will restart soon, then tries to connect databases according to the new configuration.
}
