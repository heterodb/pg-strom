@ja{
本章ではPG-Stromの持つ先進機能について説明します。
}
@en{
This chapter introduces advanced features of PG-Strom.
}

@ja:#SSD-to-GPUダイレクトSQL実行
@en:#SSD-to-GPU Direct SQL Execution

@ja:##概要
@en:##Overview

@ja{
SQLワークロードを高速に処理するには、プロセッサが効率よく処理を行うのと同様に、ストレージやメモリからプロセッサへ高速にデータを供給する事が重要です。処理すべきデータがプロセッサに届いていなければ、プロセッサは手持ち無沙汰になってしまいます。

SSD-to-GPUダイレクトSQL実行機能は、PCIeバスに直結する事で高速なI/O処理を実現するNVMe-SSDと、同じPCIeバス上に接続されたGPUをダイレクトに接続し、ハードウェア限界に近い速度でデータをプロセッサに供給する事でSQLワークロードを高速に処理するための機能です。
}
@en{
For the fast execution of SQL workloads, it needs to provide processors rapid data stream from storage or memory, in addition to processor's execution efficiency. Processor will run idle if data stream would not be delivered.

SSD-to-GPU Direct SQL Execution directly connects NVMe-SSD which enables high-speed I/O processing by direct attach to the PCIe bus and GPU device that is also attached on the same PCIe bus, and runs SQL workloads very high speed by supplying data stream close to the wired speed of the hardware.
}

@ja{
通常、ストレージ上に格納されたPostgreSQLデータブロックは、PCIeバスを通していったんCPU/RAMへとロードされます。その後、クエリ実行計画にしたがってWHERE句によるフィルタリングやJOIN/GROUP BYといった処理を行うわけですが、集計系ワークロードの特性上、入力するデータ件数より出力するデータ件数の方がはるかに少ない件数となります。例えば数十億行を読み出した結果をGROUP BYで集約した結果が高々数百行という事も珍しくありません。

言い換えれば、我々はゴミデータを運ぶためにPCIeバス上の帯域を消費しているとも言えますが、CPUがレコードの中身を調べるまでは、その要不要を判断できないため、一般的な実装ではこれは不可避と言えます。
}
@en{
Usually, PostgreSQL data blocks on the storage shall be once loaded to CPU/RAM through the PCIe bus, then, PostgreSQL runs WHERE-clause for filtering or JOIN/GROUP BY according to the query execution plan. Due to the characteristics of analytic workloads, the amount of result data set is much smaller than the source data set. For example, it is not rare case to read billions rows but output just hundreds rows after the aggregation operations with GROUP BY.

In the other words, we consume bandwidth of the PCIe bus to move junk data, however, we cannot determine whether rows are necessary or not prior to the evaluation by SQL workloads on CPU. So, it is not avoidable restriction in usual implementation.
}


![SSD2GPU Direct SQL Execution Overview](./img/ssd2gpu-overview.png)

@ja{
SSD-to-GPUダイレクトSQL実行はデータの流れを変え、ストレージ上のデータブロックをPCIeバス上のP2P DMAを用いてGPUに直接転送し、GPUでSQLワークロードを処理する事でCPUが処理すべきレコード数を減らすための機能です。いわば、ストレージとCPU/RAMの間に位置してSQLを処理するためのプリプロセッサとしてGPUを活用し、結果としてI/O処理を高速化するためのアプローチです。
}
@en{
SSD-to-GPU Direct SQL Execution changes the flow to read blocks from the storage sequentially. It directly loads data blocks to GPU using peer-to-peer DMA over PCIe bus, then runs SQL workloads on GPU device to reduce number of rows to be processed by CPU. In other words, it utilizes GPU as a pre-processor of SQL which locates in the middle of the storage and CPU/RAM for reduction of CPU's load, then tries to accelerate I/O processing in the results.
}

@ja{
本機能は内部的にNVIDIAのGPUDirect RDMAを使用しています。これはカスタムLinux kernel moduleを利用する事で、GPUデバイスメモリと他のPCIeデバイスの間でP2Pのデータ転送を可能にする基盤技術です。
そのため、本機能を利用するには、PostgreSQLの拡張モジュールであるPG-Stromだけではなく、Linux kernelの拡張モジュールであるNVMe-Stromドライバが必要です。

また、本機能が対応しているのはNVMe仕様のSSDのみです。SASやSATAといったインターフェースで接続されたSSDはサポートされていません。今までに動作実績のあるNVMe-SSDについては [002: HW Validation List](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List#nvme-ssd-validation-list) が参考になるでしょう。
}
@en{
This feature internally uses NVIDIA GPUDirect RDMA. It allows peer-to-peer data transfer over PCIe bus between GPU device memory and third parth device by coordination using a custom Linux kernel module.
So, this feature requires NVMe-Strom driver which is a Linux kernel module in addition to PG-Strom which is a PostgreSQL extension module.

Also note that this feature supports only NVMe-SSD. It does not support SAS or SATA SSD.
We have tested several NVMe-SSD models. You can refer [002: HW Validation List](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List#nvme-ssd-validation-list) for your information.
}

@ja:##初期設定
@en:##System Setup

@ja:###ドライバのインストール
@en:###Driver Installation

@ja{
SSD-to-GPUダイレクトSQL実行機能を利用するには`nvme_strom`パッケージが必要です。このパッケージはNVMe-SSDとGPU間のP2P DMAを仲介するLinux kernel moduleを含んでおり、[HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/)から入手可能です。

既に`heterodb-swdc`パッケージをインストールしている場合、`yum`コマンドによるインストールも可能です。
}
@en{
`nvme_strom` package is required to activate SSD-to-GPU Direct SQL Execution. This package contains a custom Linux kernel module which intermediates P2P DMA from NVME-SSD to GPU. You can obtain the package from the [HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/).

If `heterodb-swdc` package is already installed, you can install the package by `yum` command.
}

```
$ sudo yum install nvme_strom
            :
================================================================================
 Package             Arch            Version            Repository         Size
================================================================================
Installing:
 nvme_strom          x86_64          0.8-1.el7          heterodb          178 k

Transaction Summary
================================================================================
Install  1 Package
            :
DKMS: install completed.
  Verifying  : nvme_strom-0.8-1.el7.x86_64                                  1/1

Installed:
  nvme_strom.x86_64 0:0.8-1.el7

Complete!
```

@ja{
`nvme_strom`パッケージのインストールが完了すると、以下のように`lsmod`コマンドで`nvme_strom`モジュールが出力されます。
}
@en{
Once `nvme_strom` package gets installed, you can see `nvme_strom` module using `lsmod` command below.
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
SSD-to-GPUダイレクトSQL実行は以下の条件で発動します。

- スキャン対象のテーブルがNVMe-SSDで構成された区画に配置されている。
    - `/dev/nvmeXXXX`ブロックデバイス、または`/dev/nvmeXXXX`ブロックデバイスのみから構成されたmd-raid0区画が対象です。
- テーブルサイズが`pg_strom.nvme_strom_threshold`よりも大きい事。
    - この設定値は任意に変更可能ですが、デフォルト値は本体搭載物理メモリに`shared_buffers`の設定値の1/3を加えた大きさです。
}
@en{
SSD-to-GPU Direct SQL Execution shall be invoked in the following case.

- The target table to be scanned locates on the partition being consist of NVMe-SSD.
    - `/dev/nvmeXXXX` block device, or md-raid0 volume which consists of NVMe-SSDs only.
- The target table size is larger than `pg_strom.nvme_strom_threshold`.
    - You can adjust this configuration. Its default is physical RAM size of the system plus 1/3 of `shared_buffers` configuration.
}

@ja{
!!! Note
    md-raid0を用いて複数のNVMe-SSD区画からストライピング読出しを行うには、HeteroDB社の提供するエンタープライズサブスクリプションの適用が必要です。
}
@en{
!!! Note
    Striped read from multiple NVMe-SSD using md-raid0 requires the enterprise subscription provided by HeteroDB,Inc.
}

@ja{
テーブルをNVMe-SSDで構成された区画に配置するには、データベースクラスタ全体をNVMe-SSDボリュームに格納する以外にも、PostgreSQLのテーブルスペース機能を用いて特定のテーブルや特定のデータベースのみをNVMe-SSDボリュームに配置する事ができます。
}
@en{
In order to deploy the tables on the partition consists of NVMe-SSD, you can use the tablespace function of PostgreSQL to specify particular tables or databases to place them on NVMe-SSD volume, in addition to construction of the entire database cluster on the NVMe-SSD volume.
}
@ja{
例えば `/opt/nvme` にNVMe-SSDボリュームがマウントされている場合、以下のようにテーブルスペースを作成する事ができます。
PostgreSQLのサーバプロセスの権限で当該ディレクトリ配下のファイルを読み書きできるようパーミッションが設定されている必要がある事に留意してください。
}
@en{
For example, you can create a new tablespace below, if NVMe-SSD is mounted at `/opt/nvme`.
}
```
CREATE TABLESPACE my_nvme LOCATION '/opt/nvme';
```

@ja{
このテーブルスペース上にテーブルを作成するには、`CREATE TABLE`構文で以下のように指定します。
}
@en{
In order to create a new table on the tablespace, specify the `TABLESPACE` option at the `CREATE TABLE` command below.
}

```
CREATE TABLE my_table (...) TABLESPACE my_nvme;
```

@ja{
あるいは、データベースのデフォルトテーブルスペースを変更するには、`ALTER DATABASE`構文で以下のように指定します。
この場合、既存テーブルの配置されたテーブルスペースは変更されない事に留意してください。
}
@en{
Or, use `ALTER DATABASE` command as follows, to change the default tablespace of the database.
Note that tablespace of the existing tables are not changed in thie case.
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
}
@en{
There are two GPU parameters related to SSD-to-GPU Direct SQL Execution.
}
@ja{
一つは`pg_strom.nvme_strom_enabled`で、SSD-to-GPUダイレクト機能の有効/無効を単純にon/offします。
本パラメータが`off`になっていると、テーブルのサイズや物理配置とは無関係にSSD-to-GPUダイレクトSQL実行は使用されません。デフォルト値は`on`です。
}
@en{
The first is `pg_strom.nvme_strom_enabled` that simply turn on/off the function of SSD-to-GPU Direct SQL Execution.
If `off`, SSD-to-GPU Direct SQL Execution should not be used regardless of the table size or physical location. Default is `on`.
}
@ja{
もう一つのパラメータは`pg_strom.nvme_strom_threshold`で、SSD-to-GPUダイレクトSQL実行が使われるべき最小のテーブルサイズを指定します。

テーブルの物理配置がNVMe-SSD区画（または、NVMe-SSDのみで構成されたmd-raid0区画）上に存在し、かつ、テーブルのサイズが本パラメータの指定値よりも大きな場合、PG-StromはSSD-to-GPUダイレクトSQL実行を選択します。
本パラメータのデフォルト値は、システムの物理メモリサイズと`shared_buffers`パラメータの指定値の1/3です。つまり、初期設定では間違いなくオンメモリで処理しきれないサイズのテーブルに対してだけSSD-to-GPUダイレクトSQL実行を行うよう調整されています。

これは、一回の読み出しであればSSD-to-GPUダイレクトSQL実行に優位性があったとしても、オンメモリ処理ができる程度のテーブルに対しては、二回目以降のディスクキャッシュ利用を考慮すると、必ずしも優位とは言えないという仮定に立っているという事です。

ワークロードの特性によっては必ずしもこの設定が正しいとは限りません。
}
@en{
The other one is `pg_strom.nvme_strom_threshold` which specifies the least table size to invoke SSD-to-GPU Direct SQL Execution.

PG-Strom will choose SSD-to-GPU Direct SQL Execution when target table is located on NVMe-SSD volume (or md-raid0 volume which consists of NVMe-SSD only), and the table size is larger than this parameter.
Its default is sum of the physical memory size and 1/3 of the `shared_buffers`. It means default configuration invokes SSD-to-GPU Direct SQL Execution only for the tables where we certainly cannot process them on memory.

Even if SSD-to-GPU Direct SQL Execution has advantages on a single table scan workload, usage of disk cache may work better on the second or later trial for the tables which are available to load onto the main memory.

On course, this assumption is not always right depending on the workload charasteristics.
}

@ja:###SSD-to-GPUダイレクトSQL実行の利用を確認する
@en:###Ensure usage of SSD-to-GPU Direct SQL Execution

@ja{
`EXPLAIN`コマンドを実行すると、当該クエリでSSD-to-GPUダイレクトSQL実行が利用されるのかどうかを確認する事ができます。

以下のクエリの例では、`Custom Scan (GpuJoin)`による`lineorder`テーブルに対するスキャンに`NVMe-Strom: enabled`との表示が出ています。この場合、`lineorder`テーブルからの読出しにはSSD-to-GPUダイレクトSQL実行が利用されます。
}
@en{
`EXPLAIN` command allows to ensure whether SSD-to-GPU Direct SQL Execution shall be used in the target query, or not.

In the example below, a scan on the `lineorder` table by `Custom Scan (GpuJoin)` shows `NVMe-Strom: enabled`. In this case, SSD-to-GPU Direct SQL Execution shall be used to read from the `lineorder` table.
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
}
@en{
Right now, GPU routines of PG-Strom cannot run MVCC visibility checks per row, because only host code has a special data structure for visibility checks. It also leads a problem.
}
@ja{
NVMe-SSDにP2P DMAを要求する時点では、ストレージブロックの内容はまだCPU/RAMへと読み出されていないため、具体的にどの行が可視であるのか、どの行が不可視であるのかを判別する事ができません。これは、PostgreSQLがレコードをストレージへ書き出す際にMVCC関連の属性と共に書き込んでいるためで、似たような問題がIndexOnlyScanを実装する際に表面化しました。

これに対処するため、PostgreSQLはVisibility Mapと呼ばれるインフラを持っています。これは、あるデータブロック中に存在するレコードが全てのトランザクションから可視である事が明らかであれば、該当するビットを立てる事で、データブロックを読むことなく当該ブロックにMVCC不可視なレコードが存在するか否かを判定する事を可能とするものです。

SSD-to-GPUダイレクトSQL実行はこのインフラを利用しています。つまり、Visibility Mapがセットされており、"all-visible"であるブロックだけがSSD-to-GPU P2P DMAで読み出すようリクエストが送出されます。
}
@en{
We cannot know which row is visible, or invisible at the time when PG-Strom requires P2P DMA for NVMe-SSD, because contents of the storage blocks are not yet loaded to CPU/RAM, and MVCC related attributes are written with individual records. PostgreSQL had similar problem when it supports IndexOnlyScan.

To address the problem, PostgreSQL has an infrastructure of visibility map which is a bunch of flags to indicate whether any records in a particular data block are visible from all the transactions. If associated bit is set, we can know the associated block has no invisible records without reading the block itself.

SSD-to-GPU Direct SQL Execution utilizes this infrastructure. It checks the visibility map first, then only "all-visible" blocks are required to read with SSD-to-GPU P2P DMA.
}
@ja{
Visibility MapはVACUUMのタイミングで作成されるため、以下のように明示的にVACUUMを実行する事で強制的にVisibility Mapを構築する事ができます。
}
@en{
VACUUM constructs visibility map, so you can enforce PostgreSQL to construct visibility map by explicit launch of VACUUM command.
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

@ja:#GPUメモリストア(gstore_fdw)
@en:#GPU Memory Store(gstore_fdw)

@ja:##概要
@en:##Overview

@ja{
通常、PG-StromはGPUデバイスメモリを一時的にだけ利用します。クエリの実行中に必要なだけのデバイスメモリを割り当て、その領域にデータを転送してSQLワークロードを実行するためにGPUカーネルを実行します。GPUカーネルの実行が完了すると、当該領域は速やかに開放され、他のワークロードでまた利用する事が可能となります。

これは複数セッションの並行実行やGPUデバイスメモリよりも巨大なテーブルのスキャンを可能にするための設計ですが、状況によっては必ずしも適切ではない場合もあります。

典型的な例は、それほど巨大ではなくGPUデバイスメモリに載る程度の大きさのデータに対して、繰り返し様々な条件で計算を行うといった利用シーンです。これは機械学習やパターンマッチ、類似度サーチといったワークロードが該当します。
S}
@en{
Usually, PG-Strom uses GPU device memory for temporary purpose only. It allocates a certain amount of device memory needed for query execution, then transfers data blocks and launch GPU kernel to process SQL workloads. Once GPU kernel gets finished, these device memory regison shall be released soon, to re-allocate unused device memory for other workloads.

This design allows concurrent multiple session or scan workloads on the tables larger than GPU device memory. It may not be optimal depending on circumstances.

A typical example is, repeated calculation under various conditions for data with a scale large enough to fit in the GPU device memory, not so large. This applies to workloads such as machine-learning, pattern matching or similarity search.
}

@ja{
現在のGPUにとって、数GB程度のデータをオンメモリで処理する事はそれほど難しい処理ではありませんが、PL/CUDA関数の呼び出しの度にGPUへロードすべきデータをCPUで加工し、これをGPUへ転送するのはコストのかかる処理です。

加えて、PostgreSQLの可変長データには1GBのサイズ上限があるため、これをPL/CUDA関数の引数として与える場合、データサイズ自体は十分にGPUデバイスメモリに載るものであってもデータ形式には一定の制約が存在する事になります。
}
@en{
For modern GPUs, it is not so difficult to process a few gigabytes data on memory at most, but it is a costly process to setup data to be loaded onto GPU device memory and transfer them.

In addition, since variable length data in PostgreSQL has size limitation up to 1GB, it restricts the data format when it is givrn as an argument of PL/CUDA function, even if the data size itself is sufficient in the GPU device memory.
}

@ja{
GPUメモリストア(gstore_fdw)は、あらかじめGPUデバイスメモリを確保しデータをロードしておくための機能です。
これにより、PL/CUDA関数の呼び出しの度に引数をセットアップしたりデータを転送する必要がなくなるほか、GPUデバイスメモリの容量が許す限りデータを確保する事ができますので、可変長データの1GBサイズ制限も無くなります。

gstore_fdwはその名の通り、PostgreSQLの外部データラッパ（Foreign Data Wrapper）を使用して実装されています。
gstore_fdwの制御する外部テーブル（Foreign Table）に対して`INSERT`、`UPDATE`、`DELETE`の各コマンドを実行する事で、GPUデバイスメモリ上のデータ構造を更新する事ができます。また、同様に`SELECT`文を用いてデータを読み出す事ができます。

外部テーブルを通してGPUデバイスメモリに格納されたデータは、PL/CUDA関数から参照する事ができます。
現在のところ、SQLから透過的に生成されたGPUプログラムは当該GPUデバイスメモリ領域を参照する事はできませんが、将来のバージョンにおいて改良が予定されています。
}

@en{
GPU memory store (gstore_fdw) is a feature to preserve GPU device memory and to load data to the memory preliminary.
It makes unnecessary to setup arguments and load for each invocation of PL/CUDA function, and eliminates 1GB limitation of variable length data because it allows GPU device memory allocation up to the capacity.

As literal, gstore_fdw is implemented using foreign-data-wrapper of PostgreSQL.
You can modify the data structure on GPU device memory using `INSERT`, `UPDATE` or `DELETE` commands on the foreign table managed by gstore_fdw. In the similar way, you can also read the data using `SELECT` command.

PL/CUDA function can reference the data stored onto GPU device memory through the foreign table.
Right now, GPU programs which is transparently generated from SQL statement cannot reference this device memory region, however, we plan to enhance the feature in the future release.
}

![GPU memory store](./img/gstore_fdw-overview.png)

@ja:##初期設定
@en:##Setup

@ja{
通常、外部テーブルを作成するには以下の3ステップが必要です。

- `CREATE FOREIGN DATA WRAPPER`コマンドにより外部データラッパを定義する
- `CREATE SERVER`コマンドにより外部サーバを定義する
- `CREATE FOREIGN TABLE`コマンドにより外部テーブルを定義する

このうち、最初の2ステップは`CREATE EXTENSION pg_strom`コマンドの実行に含まれており、個別に実行が必要なのは最後の`CREATE FOREIGN TABLE`のみです。
}

@en{
Usually it takes the 3 steps below to create a foreign table.

- Define a foreign-data-wrapper using `CREATE FOREIGN DATA WRAPPER` command
- Define a foreign server using `CREATE SERVER` command
- Define a foreign table using `CREATE FOREIGN TABLE` command

The first 2 steps above are included in the `CREATE EXTENSION pg_strom` command. All you need to run individually is `CREATE FOREIGN TABLE` command last.
}

```
CREATE FOREIGN TABLE ft (
    id int,
    signature smallint[] OPTIONS (compression 'pglz')
)
SERVER gstore_fdw OPTIONS(pinning '0', format 'pgstrom');
```

@ja{
`CREATE FOREIGN TABLE`コマンドを使用して外部テーブルを作成する際、いくつかのオプションを指定することができます。

`SERVER gstore_fdw`は必須です。外部テーブルがgstore_fdwによって制御されることを指定しています。

`OPTIONS`句では以下のオプションがサポートされています。
}

@en{
You can specify some options on creation of foreign table using `CREATE FOREIGN TABLE` command.

`SERVER gstore_fdw` is a mandatory option. It indicates the new foreign table is managed by gstore_fdw.

The options below are supported in the `OPTIONS` clause.
}

@ja{
|名前|対象  |説明       |
|:--:|:----:|:----------|
|`pinning`|テーブル|デバイスメモリを確保するGPUのデバイス番号を指定します。|
|`format`|テーブル|GPUデバイスメモリ上の内部データ形式を指定します。デフォルトは`pgstrom`です。|
|`compression`|カラム|可変長データを圧縮して保持するかどうかを指定します。デフォストは非圧縮です。|
}
@en{
|name|target|description|
|:--:|:----:|:----------|
|`pinning`|table|Specifies device number of the GPU where device memory is preserved.|
|`format`|table|Specifies the internal data format on GPU device memory. Default is `pgstrom`|
|`compression`|column|Specifies whether variable length data is compressed, or not. Default is uncompressed.|
}

@ja{
`format`オプションで選択可能なパラメータは、現在のところ`pgstrom`のみです。これは、PG-Stromがインメモリ列キャッシュの内部フォーマットとして使用しているものと同一です。
純粋にSQLを用いてデータの入出力を行うだけであればユーザが内部データ形式を意識する必要はありませんが、PL/CUDA関数をプログラミングしたり、IPCハンドルを用いて外部プログラムとGPUデバイスメモリを共有する場合には考慮が必要です。
}
@en{
Right now, only `pgstrom` is supported for `format` option. It is identical data format with what PG-Strom uses for in-memory columnar cache.
In most cases, no need to pay attention to internal data format on writing / reading GPU data store using SQL. On the other hands, you need to consider when you program PL/CUDA function or share the GPU device memory with external applications using IPC handle.
}
@ja{
`compression`オプションで選択可能なパラメータは、現在のところ`plgz`のみです。これは、PostgreSQLが可変長データを圧縮する際に用いているものと同一の形式で、PL/CUDA関数からはGPU内関数`pglz_decompress()`を呼び出す事で展開が可能です。圧縮アルゴリズムの特性上、例えばデータの大半が0であるような疎行列を表現する際に有用です。
}
@en{
Right now, only `pglz` is supported for `compression` option. This compression logic adopts an identical data format and algorithm used by PostgreSQL to compress variable length data larger than its threshold.
It can be decompressed by GPU internal function `pglz_decompress()` from PL/CUDA function. Due to the characteristics of the compression algorithm, it is valuable to represent sparse matrix that is mostly zero.
}

@ja:##運用
@en:##Operations

@ja:###データのロード
@en:###Loading data

@ja{
通常のテーブルと同様にINSERT、UPDATE、DELETEによって外部テーブルの背後に存在するGPUデバイスメモリを更新する事ができます。

ただし、gstore_fdwはこれらコマンドの実行開始時に`SHARE UPDATE EXCLUSIVE`ロックを獲得する事に注意してください。これはある時点において１トランザクションのみがgstore_fdw外部テーブルを更新できることを意味します。
この制約は、PL/CUDA関数からgstore_fdw外部テーブルを参照するときに個々のレコード単位で可視性チェックを行う必要がないという特性を得るためのトレードオフです。
}
@en{
Like normal tables, you can write GPU device memory on behalf of the foreign table using `INSERT`, `UPDATE` and `DELETE` command.

Note that gstore_fdw acquires `SHARE UPDATE EXCLUSIVE` lock on the beginning of these commands. It means only single transaction can update the gstore_fdw foreign table at a certain point.
It is a trade-off. We don't need to check visibility per record when PL/CUDA function references gstore_fdw foreign table.
}

@ja{
また、gstore_fdw外部テーブルに書き込まれた内容は、通常のテーブルと同様にトランザクションがコミットされるまでは他のセッションからは不可視です。
この特性は、トランザクションの原子性を担保するには重要な性質ですが、古いバージョンを参照する可能性のある全てのトランザクションがコミットまたはアボートするまでの間は、古いバージョンのgstore_fdw外部テーブルの内容をGPUデバイスメモリに保持しておかねばならない事を意味します。

そのため、通常のテーブルと同様にINSERT、UPDATE、DELETEが可能であるとはいえ、数行を更新してトランザクションをコミットするという事を繰り返すのは避けるべきです。基本的には大量行のINSERTによるバルクロードを行うべきです。
}
@en{
Any contents written to the gstore_fdw foreign table is not visible to other sessions until transaction getting committed, like regular tables.
This is a significant feature to ensure atomicity of transaction, however, it also means the older revision of gstore_fdw foreign table contents must be kept on the GPU device memory until any concurrent transaction which may reference the older revision gets committed or aborted.

So, even though you can run `INSERT`, `UPDATE` or `DELETE` commands as if it is regular tables, you should avoidto update several rows then commit transaction many times. Basically, `INSERT` of massive rows at once (bulk loading) is recommended.
}

@ja{
通常のテーブルとは異なり、gstore_fdwに記録された内容は揮発性です。つまり、システムの電源断やPostgreSQLの再起動によってgstore_fdw外部テーブルの内容は容易に失われてしまいます。したがって、gstore_fdw外部テーブルにロードするデータは、他のデータソースから容易に復元可能な形にしておくべきです。
}
@en{
Unlike regular tables, contents of the gstore_fdw foreign table is vollatile. So, it is very easy to loose contents of the gstore_fdw foreign table by power-down or PostgreSQL restart. So, what we load onto gstore_fdw foreign table should be reconstructable by other data source.
}

@ja:###デバイスメモリ消費量の確認
@en:###Checking the memory consumption

@ja{
gstore_fdwによって消費されるデバイスメモリのサイズを確認するには`pgstrom.gstore_fdw_chunk_info`システムビューを参照します。


ああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああ
}
@en{


}

```
postgres=# select * from pgstrom.gstore_fdw_chunk_info ;
 database_oid | table_oid | revision | xmin | xmax | pinning | format  |  rawsize  |  nitems
--------------+-----------+----------+------+------+---------+---------+-----------+----------
        13806 |     26800 |        3 |    2 |    0 |       0 | pgstrom | 660000496 | 15000000
        13806 |     26797 |        2 |    2 |    0 |       0 | pgstrom | 440000496 | 10000000
(2 rows)
```

@ja{
`nvidia-smi`コマンドを
}

@en{

}

```
$ nvidia-smi
Wed Apr  4 15:11:50 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.30                 Driver Version: 390.30                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P40           Off  | 00000000:02:00.0 Off |                    0 |
| N/A   39C    P0    52W / 250W |   1221MiB / 22919MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      6885      C   ...bgworker: PG-Strom GPU memory keeper     1211MiB |
+-----------------------------------------------------------------------------+
```


@ja:###内部データ形式
@en:###Internal Data Format

@ja{
gstore_fdwがGPUデバイスメモリ上にデータを保持する際の内部データ形式の詳細はノートを参照してください。

- `pgstrom`フォーマットの詳細
    - ここにノートのリンク
}

@en{
See the notes for details of the internal data format when gstore_fdw write on GPU device memory.

- Detail of the `pgstrom` format
    - Here is a link to the note
}

@ja:##関連機能
@en:##Related Features

@ja{
CUDAには`cuIpcGetMemHandle()`および`cuIpcOpenMemHandle()`というAPIが用意されています。前者を用いてアプリケーションプログラムが確保したGPUデバイスメモリのユニークな識別子を取得し、後者を用いて別のアプリケーションプログラムから同一のGPUデバイスメモリを参照する事が可能となります。言い換えれば、ホストシステムにおける共有メモリのような仕組みを備えています。

このユニークな識別子は`CUipcMemHandle`型のオブジェクトで、内部的には単純な64バイトのバイナリデータです。
本節では`CUipcMemHandle`識別子を利用して、PostgreSQLと外部プログラムの間でGPUを介したデータ交換を行うための関数について説明します。
}
@en{
CUDA provides special APIs `cuIpcGetMemHandle()` and `cuIpcOpenMemHandle()`.
The first allows to get a unique identifier of GPU device memory allocated by applications. The other one allows to reference a shared GPU device memory region from other applications. In the other words, it supports something like a shared memory on the host system.

This unique identifier is `CUipcMemHandle` object; which is simple binary data in 64bytes.
This session introduces SQL functions which exchange GPU device memory with other applications using `CUipcMemHandle` identifier.
}


### gstore_export_ipchandle(reggstore)

@ja{
本関数は、gstore_fdw制御下の外部テーブルがGPU上に確保しているデバイスメモリの`CUipcMemHandle`識別子を取得し、bytea型のバイナリデータとして出力します。
外部テーブルが空でGPU上にデバイスメモリを確保していなければNULLを返します。

- 第1引数(*ftable_oid*): 外部テーブルのOID。`reggstore`型なので、外部テーブル名を文字列で指定する事もできる。
- 戻り値: `CUipcMemHandle`識別子のbytea型表現。

}
@en{
This function gets `CUipcMemHandle` identifier of the GPU device memory which is preserved by gstore_fdw foreign table, then returns as a binary data in `bytea` type.
If foreign table is empty and has no GPU device memory, it returns NULL.

- 1st arg(*ftable_oid*): OID of the foreign table. Because it is `reggstore` type, you can specify the foreign table by name string.
- result: `CUipcMemHandle` identifier in the bytea type.
}

```
# select gstore_export_ipchandle('ft');
                                                      gstore_export_ipchandle

------------------------------------------------------------------------------------------------------------------------------------
 \xe057880100000000de3a000000000000904e7909000000000000800900000000000000000000000000020000000000005c000000000000001200d0c10101005c
(1 row)
```

### lo_import_gpu(int, bytea, bigint, bigint, oid=0)

@ja{
本関数は、外部アプリケーションがGPU上に確保したデバイスメモリ領域をPostgreSQL側で一時的にオープンし、当該領域の内容を読み出してPostgreSQLラージオブジェクトとして書き出します。
第5引数で指定したラージオブジェクトが既に存在する場合、ラージオブジェクトはGPUデバイスメモリから読み出した内容で置き換えられます。ただし所有者・パーミッション設定は保持されます。これ以外の場合は、新たにラージオブジェクトを作成し、GPUデバイスメモリから読み出した内容を書き込みます。
}
@en{
This function temporary opens the GPU device memory region acquired by external applications, then read this region and writes out as a largeobject of PostgreSQL.
If largeobject already exists, its contents is replaced by the data read from the GPU device memory. It keeps owner and permission configuration. Elsewhere, it creates a new largeobject, then write out the data which is read from GPU device memory.
}
@ja{
- 第1引数(*device_nr*): デバイスメモリを確保したGPUデバイス番号
- 第2引数(*ipc_mhandle*): `CUipcMemHandle`識別子のbytea型表現。
- 第3引数(*offset*): 読出し開始位置のデバイスメモリ領域先頭からのオフセット
- 第4引数(*length*): バイト単位での読出しサイズ
- 第5引数(*loid*): 書き込むラージオブジェクトのOID。省略した場合 0 が指定されたものと見なす。
- 戻り値: 書き込んだラージオブジェクトのOID
}
@en{
- 1st arg(*device_nr*): GPU device number where device memory is acquired
- 2nd arg(*ipc_mhandle*): `CUipcMemHandle` identifier in bytea type
- 3rd(*offset*): offset of the head position to read, from the GPU device memory region.
- 4th(*length*): size to read in bytes
- 5th(*loid*): OID of the largeobject to be written. 0 is assumed, if no valid value is supplied.
- result: OID of the written largeobject
}

### lo_export_gpu(oid, int, bytea, bigint, bigint)

@ja{
本関数は、外部アプリケーションがGPU上に確保したデバイスメモリ領域をPostgreSQL側で一時的にオープンし、当該領域へPostgreSQLラージオブジェクトの内容を書き出します。
ラージオブジェクトのサイズが指定された書き込みサイズよりも小さい場合、残りの領域は 0 でクリアされます。
}
@en{}
@ja{
- 第1引数(*loid*): 読み出すラージオブジェクトのOID
- 第2引数(*device_nr*): デバイスメモリを確保したGPUデバイス番号
- 第3引数(*ipc_mhandle*): `CUipcMemHandle`識別子のbytea型表現。
- 第4引数(*offset*): 書き込み開始位置のデバイスメモリ領域先頭からのオフセット
- 第5引数(*length*): バイト単位での書き込みサイズ
- 戻り値: 実際に書き込んだバイト数。指定されたラージオブジェクトの大きさが*length*よりも小さな場合、*length*よりも小さな値を返す事がある。
}
@en{
- 1st arg(*loid*): OID of the largeobject to be read
- 2nd arg(*device_nr*): GPU device number where device memory is acquired
- 3rd arg(*ipc_mhandle*): `CUipcMemHandle` identifier in bytea type
- 4th arg(*offset*): offset of the head position to write, from the GPU device memory region.
- 5th arg(*length*): size to write in bytes
- result: Length of bytes actually written. If length of the largeobject is less then *length*, it may return the value less than *length*.
}


