@ja:<h1>SSD-to-GPUダイレクトSQL実行</h1>
@en:<h1>SSD-to-GPU Direct SQL Execution</h1>

@ja:#概要
@en:#Overview

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

@ja:#初期設定
@en:#System Setup

@ja:##ドライバのインストール
@en:##Driver Installation

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

@ja:##テーブルスペースの設計
@en:##Designing Tablespace

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

@ja:##GPUとNVME-SSD間の距離
@en:##Distance between GPU and NVME-SSD

@ja{
サーバの選定とGPUおよびNVME-SSDの搭載にあたり、デバイスの持つ性能を最大限に引き出すには、デバイス間の距離を意識したコンフィグが必要です。

SSD-to-GPUダイレクトSQL機能がその基盤として使用している[NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)は、P2P DMAを実行するには互いのデバイスが同じPCIe root complexの配下に接続されている事を要求しています。つまり、デュアルCPUシステムでNVME-SSDがCPU1に、GPUがCPU2に接続されており、P2P DMAがCPU間のQPIを横切るよう構成する事はできません。

また、性能の観点からはCPU内蔵のPCIeコントローラよりも、専用のPCIeスイッチを介して互いのデバイスを接続する方が推奨されています。
}
@en{
On selection of server hardware and installation of GPU and NVME-SSD, hardware configuration needs to pay attention to the distance between devices, to pull out maximum performance of the device.

[NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/), basis of the SSD-to-GPU Direct SQL mechanism, requires both of the edge devices of P2P DMA are connected on the same PCIe root complex. In the other words, unable to configure the P2P DMA traverses QPI between CPUs when NVME-SSD is attached on CPU1 and GPU is attached on CPU2 at dual socket system.

From standpoint of the performance, it is recommended to use dedicated PCIe-switch to connect both of the devices more than the PCIe controller built in CPU.
}

@ja{
以下の写真はHPC向けサーバのマザーボードで、8本のPCIe x16スロットがPCIeスイッチを介して互いに対となるスロットと接続されています。また、写真の左側のスロットはCPU1に、右側のスロットはCPU2に接続されています。

例えば、SSD-2上に構築されたテーブルをSSD-to-GPUダイレクトSQLを用いてスキャンする場合、最適なGPUの選択はGPU-2でしょう。またGPU-1を使用する事も可能ですが、GPUDirect RDMAの制約から、GPU-3とGPU-4の使用は避けねばなりません。
}
@en{
The photo below is a motherboard of HPC server. It has 8 of PCIe x16 slots, and each pair is linked to the other over the PCIe switch. The slots in the left-side of the photo are connected to CPU1, and right-side are connected to CPU2.

When a table on SSD-2 is scanned using SSD-to-GPU Direct SQL, the optimal GPU choice is GPU-2, and it may be able to use GPU1. However, we have to avoid to choose GPU-3 and GPU-4 due to the restriction of GPUDirect RDMA.
}

![Motherboard of HPC Server](./img/pcie-hpc-server.png)

@ja{
PG-Stromは起動時にシステムのPCIeバストポロジ情報を取得し、GPUとNVME-SSD間の論理的な距離を算出します。
これは以下のように起動時のログに記録されており、例えば`/dev/nvme2`をスキャンする時はGPU1といった具合に、各NVME-SSDごとに最も距離の近いGPUを優先して使用するようになります。
}
@en{
PG-Strom calculate logical distances on any pairs of GPU and NVME-SSD using PCIe bus topology information of the system on startup time.
It is displayed at the start up log. Each NVME-SSD determines the preferable GPU based on the distance, for example, `GPU1` shall be used on scan of the `/dev/nvme2`.
}

```
$ pg_ctl restart
     :
LOG:  GPU<->SSD Distance Matrix
LOG:             GPU0     GPU1     GPU2
LOG:      nvme0  (   3)      7       7
LOG:      nvme5      7       7   (   3)
LOG:      nvme4      7       7   (   3)
LOG:      nvme2      7   (   3)      7
LOG:      nvme1  (   3)      7       7
LOG:      nvme3      7   (   3)      7
     :
```

@ja{
通常は自動設定で問題ありません。
ただ、NVME-over-Fabric(RDMA)を使用する場合はPCIeバス上のnvmeデバイスの位置を取得できないため、手動でNVME-SSDとGPUの位置関係を設定する必要があります。

例えば`nvme1`には`gpu2`を、`nvme2`と`nvme3`には`gpu1`を割り当てる場合、以下の設定を`postgresql.conf`へ記述します。この手動設定は、自動設定よりも優先する事に留意してください。
}
@en{
Usually automatic configuration works well.
In case when NVME-over-Fabric(RDMA) is used, unable to identify the location of nvme device on the PCIe-bus, so you need to configure the logical distance between NVME-SSD and GPU manually.

The example below shows the configuration of `gpu2` for `nvme1`, and `gpu1` for `nvme2` and `nvme3`.
It shall be added to `postgresql.conf`. Please note than manual configuration takes priority than the automatic configuration.
}
```
pg_strom.nvme_distance_map = nvme1:gpu2, nvme2:gpu1, nvme3:gpu1
```


@ja:#運用
@en:#Operations

@ja:##GUCパラメータによる制御
@en:##Controls using GUC parameters

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


@ja:##Visibility Mapに関する注意事項
@en:##Attension for visibility map

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

