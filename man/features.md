@ja{
本章ではPG-Stromの持つ先進機能について説明します。
}
@en{
This chapter introduces advanced features of PG-Strom.
}

@ja:#[SSD-to-GPUダイレクトSQL実行](#hoge)
@en:#SSD-to-GPU Direct SQL Execution

@ja:##概要
@en:##Overview

@ja{
SQLワークロードを高速に処理するには、プロセッサが効率よく処理を行うのと同様に、ストレージやメモリからプロセッサへ高速にデータを供給する事が重要です。処理すべきデータがプロセッサに届いていなければ、プロセッサは手持ち無沙汰になってしまいます。

SSD-to-GPUダイレクトSQL実行機能は、PCIeバスに直結する事で高速なI/O処理を実現するNVMe-SSDと、同じPCIeバス上に接続されたGPUをダイレクトに接続し、ハードウェア限界に近い速度でデータをプロセッサに供給する事でSQLワークロードを高速に処理するための機能です。
}

@ja{
通常、ストレージブロック上に格納されたPostgreSQLデータブロックは、PCIeバスを通していったんCPU/RAMへとロードされます。その後、クエリ実行計画にしたがってWHERE句によるフィルタリングやJOIN/GROUP BYといった処理を行うわけですが、集計系ワークロードの特性上、入力するデータ件数より出力するデータ件数の方がはるかに少ない件数となります。例えば数十億行を読み出した結果をGROUP BYで集約した結果が高々数百行という事も珍しくありません。

言い換えれば、我々はゴミデータを運ぶためにPCIeバス上の帯域を消費しているとも言えますが、CPUがレコードの中身を調べるまでは、その要不要を判断できないため、一般的な実装ではこれは不可避と言えます。
}

![SSD2GPU Direct SQL Execution Overview](./img/ssd2gpu-overview.png)

@ja{
SSD-to-GPUダイレクトSQL実行機能は、ストレージから読み出すデータの流れを変え、データブロックをCPU/RAMへとロードする前にGPUへ転送してSQLワークロードを処理する事でデータ件数を劇的に減らすための機能です。いわば、GPUをストレージとCPU/RAMの間に位置するプリプロセッサーとして利用する事で、CPUの負荷を下げ、結果としてI/O処理の高速化を実現しようとするアプローチです。
}

@ja{
本機能は内部的にNVIDIAのGPUDirect RDMAを使用しています。これはカスタムLinux kernel moduleを利用する事で、GPUデバイスメモリと他のPCIeデバイスの間でP2Pのデータ転送を可能にする基盤技術です。
そのため、本機能を利用するには、PostgreSQLの拡張モジュールであるPG-Stromだけではなく、Linux kernelの拡張モジュールであるNVMe-Stromドライバが必要です。

また、本機能が対応しているのはNVMe仕様のSSDのみです。SASやSATAといったインターフェースで接続されたSSDはサポートされていません。今までに動作実績のあるNVMe-SSDについては [002: HW Validation List](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List#nvme-ssd-validation-list) が参考になるでしょう。
}

@ja:##初期設定
@en:##System Setup

@ja:###ドライバのインストール
@en:###Driver Installation

@ja{
SSD-to-GPUダイレクトSQL実行機能を利用するには`nvme-strom`パッケージが必要です。このパッケージはNVMe-SSDとGPU間のpeer-to-peer DMAを仲介するLinux kernel moduleを含んでおり、[HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/)から入手可能です。

既に`heterodb-swdc`パッケージをインストールしている場合、`yum`コマンドによるインストールも可能です。
}

```
$ sudo yum install nvme-strom
Loaded plugins: fastestmirror
      :
Resolving Dependencies
--> Running transaction check
---> Package nvme-strom.x86_64 0:0.6-1.el7 will be installed
--> Finished Dependency Resolution

Dependencies Resolved

================================================================================
 Package             Arch            Version            Repository         Size
================================================================================
Installing:
 nvme-strom          x86_64          0.6-1.el7          heterodb           58 k

Transaction Summary
================================================================================
Install  1 Package

Total download size: 58 k
Installed size: 217 k
Is this ok [y/d/N]: y
Downloading packages:
Package nvme-strom-0.6-1.el7.x86_64.rpm is not signed0 B/s |    0 B   --:-- ETA
nvme-strom-0.6-1.el7.x86_64.rpm                            |  58 kB   00:00


Package nvme-strom-0.6-1.el7.x86_64.rpm is not signed
```

@ja{
`nvme-strom`パッケージのインストールが完了すると、以下のように`lsmod`コマンドで`nvme_strom`モジュールが出力されます。
}

```
$ lsmod | grep nvme
nvme_strom             12625  0
nvme                   27722  4
nvme_core              52964  9 nvme
```

@ja:###テーブルスペースの設計
@en:###Designing Tablespace

@ja{
SSD-to-GPUダイレクトSQL実行は以下の条件で使用されます。
- スキャン対象のテーブルがNVMe-SSDで構成された区画に配置されている。
    - `/dev/nvmeXXXX`ブロックデバイス、または`/dev/nvmeXXXX`ブロックデバイスのみから構成されたmd-raid0区画が対象です。
    - md-raid0を用いたストライピング読出しに関しては、HeteroDB社の提供するサブスクリプションが必要です。
- テーブルサイズが`pg_strom.nvme_strom_threshold`よりも大きい事。
    - この設定値は任意に変更可能ですが、デフォルト値は本体搭載物理メモリに`shared_buffers`の設定値の1/3を加えた大きさです。
}

@ja{
テーブルをNVMe-SSDで構成された区画に配置するには、データベースクラスタ全体をNVMe-SSDボリュームに格納する以外にも、PostgreSQLのテーブルスペース機能を用いて特定のテーブルや特定のデータベースのみをNVMe-SSDボリュームに配置する事ができます。

例えば `/opt/nvme` にNVMe-SSDボリュームがマウントされている場合、以下のようにテーブルスペースを作成する事ができます。
}

```
CREATE TABLESPACE my_nvme LOCATION '/opt/nvme';
```

@ja{
このテーブルスペース上にテーブルを作成するには、`CREATE TABLE`構文で以下のように指定します。
}

```
CREATE TABLE my_table ()
```

@ja{
あるいは、データベースのデフォルトテーブルスペースを変更するには、`ALTER DATABASE`構文で以下のように指定します。
この場合、既存テーブルの配置されたテーブルスペースは変更されない事に留意してください。
}
```
ALTER DATABASE my_database SET TABLESPACE my_nvme;
```

@ja:##運用
@en:##Operations

@ja:###GUCパラメータによる制御
@en:###Controls using GUC parameters

@ja{
SSD-to-GPUダイレクトSQL実行に関連するGUCパラメータは2つあります。

一つは`pg_strom.nvme_strom_enabled`で、SSD-to-GPUダイレクト機能の有効/無効を単純にon/offします。
本パラメータが`off`になっていると、テーブルのサイズや物理配置とは無関係にSSD-to-GPUダイレクトSQL実行は使用されません。デフォルト値は`on`です。
}

{
もう一つのパラメータは`pg_strom.nvme_strom_threshold`で、SSD-to-GPUダイレクトSQL実行が使われるべき最小のテーブルサイズを指定します。

テーブルの物理配置がNVMe-SSD区画（または、NVMe-SSDのみで構成されたmd-raid0区画）上に存在し、かつ、テーブルのサイズが本パラメータの指定値よりも大きな場合、PG-StromはSSD-to-GPUダイレクトSQL実行を選択します。
本パラメータのデフォルト値は、システムの物理メモリサイズと`shared_buffers`パラメータの指定値の1/3です。つまり、初期設定では間違いなくオンメモリで処理しきれないサイズのテーブルに対してだけSSD-to-GPUダイレクトSQL実行を行うよう調整されています。

これは、一回の読み出しであればSSD-to-GPUダイレクトSQL実行に優位性があったとしても、オンメモリ処理ができる程度のテーブルに対しては、二回目以降のディスクキャッシュ利用を考慮すると、必ずしも優位とは言えないという仮定に立っているという事です。

ワークロードの特性によっては必ずしもこの設定が正しいとは限りません。
}

@ja:###SSD-to-GPUダイレクトSQL実行の利用を確認する
@en:###Ensure usage of SSD-to-GPU Direct SQL Execution

@ja{
`EXPLAIN`コマンドを実行すると、当該クエリでSSD-to-GPUダイレクトSQL実行が利用されるのかどうかを確認する事ができます。

以下のクエリの例では、`Custom Scan (GpuJoin)`による`lineorder`テーブルに対するスキャンに`NVMe-Strom: enabled`との表示が出ています。この場合、`lineorder`テーブルからの読出しにはSSD-to-GPUダイレクトSQL実行が利用されます。
}

```
# explain (costs off)
select sum(lo_revenue), d_year, p_brand1
from lineorder, date1, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 'MFGR#12'
and s_region = 'AMERICA'
  group by d_year, p_brand1
  order by d_year, p_brand1;
                                          QUERY PLAN
----------------------------------------------------------------------------------------------
 GroupAggregate
   Group Key: date1.d_year, part.p_brand1
   ->  Sort
         Sort Key: date1.d_year, part.p_brand1
         ->  Custom Scan (GpuPreAgg)
               Reduction: Local
               GPU Projection: pgstrom.psum((lo_revenue)::double precision), d_year, p_brand1
               Combined GpuJoin: enabled
               ->  Custom Scan (GpuJoin) on lineorder
                     GPU Projection: date1.d_year, part.p_brand1, lineorder.lo_revenue
                     Outer Scan: lineorder
                     Depth 1: GpuHashJoin  (nrows 2406009600...97764190)
                              HashKeys: lineorder.lo_partkey
                              JoinQuals: (lineorder.lo_partkey = part.p_partkey)
                              KDS-Hash (size: 10.67MB)
                     Depth 2: GpuHashJoin  (nrows 97764190...18544060)
                              HashKeys: lineorder.lo_suppkey
                              JoinQuals: (lineorder.lo_suppkey = supplier.s_suppkey)
                              KDS-Hash (size: 131.59MB)
                     Depth 3: GpuHashJoin  (nrows 18544060...18544060)
                              HashKeys: lineorder.lo_orderdate
                              JoinQuals: (lineorder.lo_orderdate = date1.d_datekey)
                              KDS-Hash (size: 461.89KB)
                     NVMe-Strom: enabled
                     ->  Custom Scan (GpuScan) on part
                           GPU Projection: p_brand1, p_partkey
                           GPU Filter: (p_category = 'MFGR#12'::bpchar)
                     ->  Custom Scan (GpuScan) on supplier
                           GPU Projection: s_suppkey
                           GPU Filter: (s_region = 'AMERICA'::bpchar)
                     ->  Seq Scan on date1
(31 rows)
```


@ja:###Visibility Mapに関する注意事項
@en:###Attension for visibility map

@ja{
現在のところ、PG-StromのGPU側処理では行単位のMVCC可視性チェックを行う事ができません。これは、可視性チェックを行うために必要なデータ構造がホスト側だけに存在するためですが、ストレージ上のブロックを直接GPUに転送する場合、少々厄介な問題が生じます。

NVMe-SSDにP2P DMAを要求する時点では、ストレージブロックの内容はまだCPU/RAMへと読み出されていないため、具体的にどの行が可視であるのか、どの行が不可視であるのかを判別する事ができません。これは、PostgreSQLがレコードをストレージへ書き出す際にMVCC関連の属性と共に書き込んでいるためで、似たような問題がIndexOnlyScanを実装する際に表面化しました。

これに対処するため、PostgreSQLはVisibility Mapと呼ばれるインフラを持っています。これは、あるデータブロック中に存在するレコードが全てのトランザクションから可視である事が明らかであれば、該当するビットを立てる事で、データブロックを読むことなく当該ブロックにMVCC不可視なレコードが存在するか否かを判定する事を可能とするものです。

SSD-to-GPUダイレクトSQL実行はこのインフラを利用しています。つまり、Visibility Mapがセットされており、MVCC可視性チェックに意味のないブロックのみを選択してSSD-to-GPUのP2P DMAを実行するのです。

Visibility MapはVACUUMのタイミングで作成されるため、以下のように明示的にVACUUMを実行する事で強制的にVisibility Mapを構築する事ができます。
}
```
VACUUM ANALYZE linerorder;
```


@ja:#インメモリ列キャッシュ
@en:#In-memory Columnar Cache

@ja:##概要
@en:##Overview

@ja{
PG-Stromはプロセッサへ高速にデータを供給するためのストレージ関連機能をもう一つ持っています。

インメモリ列キャッシュは、対象テーブルのデータブロックを読み出し、PostgreSQL標準のデータ形式である行データから集計・解析ワークロードに適した列データ形式へと変換し、メモリ上にキャッシュする機能です。

SSD-to-GPUダイレクトSQL実行とは異なり、この機能を利用するには特別なハードウェアは必要ありません。しかし一方で、現在もなおRAMの容量はSSDに比べると小さく、目安としてはシステムRAMサイズの60%～75%程度の「大規模でないデータセット」を取り扱うのに向いた機能です。
}

@ja{
本機能は「列ストア」ではありません。すなわち、列データに変換しキャッシュされた内容はPostgreSQLサーバプロセスを再起動すれば消えてしまいます。また、キャッシュされた領域を更新するような`UPDATE`文を実行すると、PG-Stromは当該キャッシュを消去します。
これは、列データ形式は本質的に更新ワークロードに弱い事を踏まえた上での設計です。つまり、行ストアの更新に対して整合性を保ったまま列ストアを更新しようとすると、書き込み性能の大幅な劣化は不可避です。一方で、単純に更新されたブロックを含む列キャッシュを消去（invalidation）するだけであれば、ほとんど処理コストはかかりません。
PG-Stromは行データであっても列データであっても、起動するGPUプログラムを変更するだけで対応可能です。すなわち、列キャッシュが消去され、通常通りPostgreSQLのshared bufferからデータを読み出さざるを得ない状況であっても柔軟に対応する事ができるのです。
}

@ja:##初期設定
@en:##System Setup

@ja:###列キャッシュの格納先
@en:###Location of the columnar cache

@ja{
`pg_strom.ccache_base_dir`パラメータによって列キャッシュの格納先を指定する事ができます。デフォルト値は`/dev/shm`で、これは一般的なLinxディストリビューションにおいて`tmpfs`が配置されているパスであり、この配下に作成されたファイルは二次記憶装置のバッキングストアを持たない揮発性のデータとなります。

このパラメータを変更する事で、例えばNVMe-SSD等、より大容量かつリーズナブルに高速なストレージ領域をバッキングストアとする列キャッシュを構築する事ができます。ただし、列キャッシュの更新はたとえ一行であってもその前後の領域を含むチャンク全体（128MB単位）の無効化を引き起こす事は留意してください。I/Oを伴う読み書きが頻発するような状況になると、意図しない性能劣化を招く可能性があります。
}

@ja:###列キャッシュビルダの設定
@en:###Columnar Cache Builder Configuration

@ja{
PG-Stromは一つまたは複数のバックグラウンドワーカーを使用して、インメモリ列キャッシュを非同期かつ自動的に構築する事ができます。この処理を行うバックグラウンドワーカーを列キャッシュビルダーと呼びます。

列キャッシュビルダーは、ユーザのSQLを処理するセッションの動作とは非同期に、指定されたデータベース内のテーブルのうち列キャッシュを構築すべき対象をラウンドロビンでスキャンし、これを列データへと変換した上でキャッシュします。

一度列キャッシュが構築されると、他の全てのバックエンドからこれを参照する事ができます。一般的なディスクキャッシュのメカニズムとは異なり、列キャッシュが構築されていない領域へのアクセスであっても、列キャッシュをオンデマンドで作成する事はありません。この場合は、通常のPostgreSQLのストレージシステムを通して行データを参照する事となります。

列キャッシュビルダの数は起動時に決まっていますので、これを増やすには後述の`pg_strom.ccache_num_builders`パラメータを設定し、PostgreSQLの再起動が必要です。
また、列キャッシュビルダは特定のデータベースに紐付けられますので、複数のデータベースで列キャッシュを使用する場合には、少なくともデータベース数以上の列キャッシュビルダが存在している事が必要です。

列キャッシュビルダを紐づけるデータベースを指定するには、`pg_strom.ccache_databases`パラメータを指定します。
このパラメータの指定には特権ユーザ権限が必要ですが、PostgreSQLの実行中にも変更する事が可能です。（もちろん、`postgresql.conf`に記載して起動時に設定する事も可能です。）

データベース名をカンマ区切りで指定すると、列キャッシュビルダが順番に指定したデータベースに関連付けられていきます。例えば、列キャッシュビルダが5プロセス存在し、`postgres,my_test,benchmark`という3つのデータベースを`pg_strom.ccache_databases`に指定した場合、`postgres`および`my_test`データベースには2プロセスの、`benchmark`データベースには1プロセスの列キャッシュビルダが割り当てられる事になります。
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

これを避けるには、`DROP DATABASE`コマンドの実行前に、`pg_strom.ccache_databases`パラメータから当該データベースを除外してください。列キャッシュビルダは直ちに再起動し、新しい設定に基づいてデータベースへの接続を試みます。
}

@ja:#GPUメモリストア(gstore_fdw)
@en:#GPU Memory Store(gstore_fdw)

@ja:##概要
@en:##Overview

@ja{
通常、PG-StromはGPUデバイスメモリを一時的にだけ利用します。クエリの実行中に必要なだけのデバイスメモリを割り当て、その領域にデータを転送してSQLワークロードを実行するためにGPUカーネルを実行します。GPUカーネルの実行が完了すると、当該領域は速やかに開放され、他のワークロードでまた利用する事が可能となります。

これは複数セッションの並行実行やGPUデバイスメモリよりも巨大なテーブルのスキャンを可能にするための設計ですが、状況によっては必ずしも適切ではない場合もあります。

典型的な例は、それほど巨大ではなくGPUデバイスメモリに載る程度の大きさのデータに対して、繰り返し様々な条件で計算を行うといった利用シーンです。これは機械学習やパターンマッチ、類似度サーチといったワークロードが該当します。
}

@ja{
現在のGPUにとって、数GB程度のデータをオンメモリで処理する事はそれほど難しい処理ではありませんが、PL/CUDA関数の呼び出しの度にGPUへロードすべきデータをCPUで加工し、これをGPUへ転送するのはコストのかかる処理です。

加えて、PostgreSQLの可変長データには1GBのサイズ上限があるため、これをPL/CUDA関数の引数として与える場合、データサイズ自体は十分にGPUデバイスメモリに載るものであってもデータ形式には一定の制約が存在する事になります。
}

@ja{
GPUメモリストア(gstore_fdw)は、あらかじめGPUデバイスメモリを確保しデータをロードしておくための機能です。
これにより、PL/CUDA関数の呼び出しの度に引数をセットアップしたりデータを転送する必要がなくなるほか、GPUデバイスメモリの容量が許す限りデータを確保する事ができますので、可変長データの1GBサイズ制限も無くなります。

gstore_fdwはその名の通り、PostgreSQLの外部データラッパ（Foreign Data Wrapper）を使用して実装されています。
gstore_fdwの制御する外部テーブル（Foreign Table）に対して`INSERT`、`UPDATE`、`DELETE`の各コマンドを実行する事で、GPUデバイスメモリ上のデータ構造を更新する事ができます。また、同様に`SELECT`文を用いてデータを読み出す事ができます。

外部テーブルを通してGPUデバイスメモリに格納されたデータは、PL/CUDA関数から参照する事ができます。
現在のところ、SQLから透過的に生成されたGPUプログラムは当該GPUデバイスメモリ領域を参照する事はできませんが、将来のバージョンにおいて改良が予定されています。
}

@ja:##初期設定
@en:##Setup

@ja{
通常、外部テーブルを作成するには以下の3ステップが必要です。
- `CREATE FOREIGN DATA WRAPPER`コマンドにより外部データラッパを定義する
- `CREATE SERVER`コマンドにより外部サーバを定義する
- `CREATE FOREIGN TABLE`コマンドにより外部テーブルを定義する

このうち、最初の2ステップは`CREATE EXTENSION pg_strom`コマンドの実行に含まれており、個別に実行が必要なのは最後の`CREATE FOREIGN TABLE`のみです。
}

```
CREATE FOREIGN TABLE ft (
    id int,
    x0 real,
    x1 real,
    x2 real,
    x3 real,
    x4 real,
    x5 real,
    x6 real,
    x7 real,
    x8 real,
    x9 real
) SERVER gstore_fdw OPTIONS (pinning '0', format 'pgstrom');
```

@ja{
`CREATE FOREIGN TABLE`コマンドを使用して外部テーブルを作成する際、いくつかのオプションを指定する必要があります。

`SERVER gstore_fdw`は必須です。外部テーブルがgstore_fdwによって制御されることを指定しています。

`OPTIONS`句では`pinning`および`format`オプションを指定します。

`pinning`オプションは、外部テーブルがデバイスメモリを割り当てるGPUのデバイス番号を指定します。このオプションは必須です。

`format`オプションは、外部テーブルがデバイスメモリ上にデータを書き込む際の内部データ形式を指定します。純粋にSQLを用いてデータ入出力を行い場合、ユーザが内部データ形式を意識する必要はありませんが、PL/CUDA関数をプログラミングしたり、IPCハンドルを用いて外部プログラムとGPUデバイスメモリの連携を取る場合には考慮が必要です。
}




@ja:##運用
@en:##Operations


データのロード

トランザクショナルではあるが、、、

データ容量の確認

preserved memoryの確認

注意事項




@ja:###内部データ構造（pgstromフォーマット）
@en:###Internal Data Format (pgstrom format)










@ja:##関連機能
@en:##Related Features

ラージオブジェクト













