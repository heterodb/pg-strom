@ja:#GPUダイレクトSQL実行
@en:#GPU Direct SQL Execution

@ja:##概要
@en:##Overview

@ja{
SQLワークロードを高速に処理するには、プロセッサが効率よく処理を行うのと同様に、ストレージやメモリからプロセッサへ高速にデータを供給する事が重要です。処理すべきデータがプロセッサに届いていなければ、プロセッサは手持ち無沙汰になってしまいます。

GPUダイレクトSQL実行機能は、PCIeバスに直結する事で高速なI/O処理を実現するNVMe-SSDと、同じPCIeバス上に接続されたGPUをダイレクトに接続し、ハードウェア限界に近い速度でデータをプロセッサに供給する事でSQLワークロードを高速に処理するための機能です。
}
@en{
For the fast execution of SQL workloads, it needs to provide processors rapid data stream from storage or memory, in addition to processor's execution efficiency. Processor will run idle if data stream would not be delivered.

GPUDirect SQL Execution directly connects NVMe-SSD which enables high-speed I/O processing by direct attach to the PCIe bus and GPU device that is also attached on the same PCIe bus, and runs SQL workloads very high speed by supplying data stream close to the wired speed of the hardware.
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
GPUダイレクトSQL実行はデータの流れを変え、ストレージ上のデータブロックをPCIeバス上のP2P DMAを用いてGPUに直接転送し、GPUでSQLワークロードを処理する事でCPUが処理すべきレコード数を減らすための機能です。いわば、ストレージとCPU/RAMの間に位置してSQLを処理するためのプリプロセッサとしてGPUを活用し、結果としてI/O処理を高速化するためのアプローチです。
}
@en{
GPU Direct SQL Execution changes the flow to read blocks from the storage sequentially. It directly loads data blocks to GPU using peer-to-peer DMA over PCIe bus, then runs SQL workloads on GPU device to reduce number of rows to be processed by CPU. In other words, it utilizes GPU as a pre-processor of SQL which locates in the middle of the storage and CPU/RAM for reduction of CPU's load, then tries to accelerate I/O processing in the results.
}

@ja{
本機能は、内部的にNVIDIA GPUDirect Storageモジュール（`nvidia-fs`）を使用して、GPUデバイスメモリとNVMEストレージとの間でP2Pのデータ転送を行います。
したがって、本機能を利用するには、PostgreSQLの拡張モジュールであるPG-Stromだけではなく、上記のLinux kernelモジュールが必要です。

また、本機能が対応しているのはNVME仕様のSSDや、NVME-oFで接続されたリモートデバイスのみです。
SASやSATAといったインターフェースで接続された旧式のストレージには対応していません。
今までに動作実績のあるNVME-SSDについては [002: HW Validation List](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List#nvme-ssd-validation-list) が参考になるでしょう。
}
@en{
This feature internally uses the NVIDIA GPUDirect Storage module (`nvidia-fs`) to coordinate P2P data transfer from NVME storage to GPU device memory.
So, this feature requires this Linux kernel module, in addition to PG-Strom as an extension of PostgreSQL.

Also note that this feature supports only NVME-SSD or NVME-oF remove devices.
It does not support legacy storages like SAS or SATA-SSD.
We have tested several NVMD-SSD models. You can refer [002: HW Validation List](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List#nvme-ssd-validation-list) for your information.
}

@ja:##初期設定
@en:##System Setup

@ja:###ドライバのインストール
@en:###Driver Installation

@ja{
以前のPG-Stromでは、GPUダイレクトSQLの利用にはHeteroDB社の開発した独自のLinux kernelドライバが必要でしたが、v3.0以降ではNVIDIAの提供するGPUDirect Storageを利用するように設計を変更しています。GPUDirect Storage用のLinux kernelドライバ（`nvidia-fs`）はCUDA Toolkitのインストールプロセスに統合され、本マニュアルの「インストール」の章に記載の手順でシステムをセットアップした場合、特に追加の設定は必要ではありません。

必要なLinux kernelドライバがインストールされているかどうか、`modinfo`コマンドや`lsmod`コマンドを利用して確認する事ができます。
}
@en{
The previous version of PG-Strom required its original Linux kernel module developed by HeteroDB for GPU-Direct SQL support, however, the version 3.0 revised the software design to use GPUDirect Storage provided by NVIDIA, as a part of CUDA Toolkit. The Linux kernel module for GPUDirect Storage (`nvidia-fs`) is integrated into the CUDA Toolkit installation process and requires no additional configuration if you have set up your system as described in the Installation chapter of this manual. 
You can check whether the required Linux kernel drivers are installed using the `modinfo` command or `lsmod` command.
}


```
$ modinfo nvidia-fs
filename:       /lib/modules/5.14.0-427.18.1.el9_4.x86_64/extra/nvidia-fs.ko.xz
description:    NVIDIA GPUDirect Storage
license:        GPL v2
version:        2.20.5
rhelversion:    9.4
srcversion:     096A726CAEC0A059E24049E
depends:
retpoline:      Y
name:           nvidia_fs
vermagic:       5.14.0-427.18.1.el9_4.x86_64 SMP preempt mod_unload modversions
sig_id:         PKCS#7
signer:         DKMS module signing key
sig_key:        18:B4:AE:27:B8:7D:74:4F:C2:27:68:2A:EB:E0:6A:F0:84:B2:94:EE
sig_hashalgo:   sha512
   :              :

$ lsmod | grep nvidia
nvidia_fs             323584  32
nvidia_uvm           6877184  4
nvidia               8822784  43 nvidia_uvm,nvidia_fs
drm                   741376  2 drm_kms_helper,nvidia
```

@ja:###テーブルスペースの設計
@en:###Designing Tablespace

@ja{
GPUダイレクトSQL実行は以下の条件で発動します。

- スキャン対象のテーブルがNVMe-SSDで構成された区画に配置されている。
    - `/dev/nvmeXXXX`ブロックデバイス、または`/dev/nvmeXXXX`ブロックデバイスのみから構成されたmd-raid0区画が対象です。
- テーブルサイズが`pg_strom.gpudirect_threshold`よりも大きい事。
    - この設定値は任意に変更可能ですが、デフォルト値は本体搭載物理メモリに`shared_buffers`の設定値の1/3を加えた大きさです。
}
@en{
GPU Direct SQL Execution shall be invoked in the following case.

- The target table to be scanned locates on the partition being consist of NVMe-SSD.
    - `/dev/nvmeXXXX` block device, or md-raid0 volume which consists of NVMe-SSDs only.
- The target table size is larger than `pg_strom.gpudirect_threshold`.
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

@ja:###GPUとNVME-SSD間の距離
@en:###Distance between GPU and NVME-SSD

@ja{
サーバの選定とGPUおよびNVME-SSDの搭載にあたり、デバイスの持つ性能を最大限に引き出すには、デバイス間の距離を意識したコンフィグが必要です。

GPUダイレクトSQL機能がその基盤として使用している[NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)は、P2P DMAを実行するには互いのデバイスが同じPCIe root complexの配下に接続されている事を要求しています。つまり、デュアルCPUシステムでNVME-SSDがCPU1に、GPUがCPU2に接続されており、P2P DMAがCPU間のQPIを横切るよう構成する事はできません。

また、性能の観点からはCPU内蔵のPCIeコントローラよりも、専用のPCIeスイッチを介して互いのデバイスを接続する方が推奨されています。
}
@en{
On selection of server hardware and installation of GPU and NVME-SSD, hardware configuration needs to pay attention to the distance between devices, to pull out maximum performance of the device.

[NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/), basis of the GPU Direct SQL mechanism, requires both of the edge devices of P2P DMA are connected on the same PCIe root complex. In the other words, unable to configure the P2P DMA traverses QPI between CPUs when NVME-SSD is attached on CPU1 and GPU is attached on CPU2 at dual socket system.

From standpoint of the performance, it is recommended to use dedicated PCIe-switch to connect both of the devices more than the PCIe controller built in CPU.
}

@ja{
以下の写真はHPC向けサーバのマザーボードで、8本のPCIe x16スロットがPCIeスイッチを介して互いに対となるスロットと接続されています。また、写真の左側のスロットはCPU1に、右側のスロットはCPU2に接続されています。

例えば、SSD-2上に構築されたテーブルをGPUダイレクトSQLを用いてスキャンする場合、最適なGPUの選択はGPU-2でしょう。またGPU-1を使用する事も可能ですが、GPUDirect RDMAの制約から、GPU-3とGPU-4の使用は避けねばなりません。
}
@en{
The photo below is a motherboard of HPC server. It has 8 of PCIe x16 slots, and each pair is linked to the other over the PCIe switch. The slots in the left-side of the photo are connected to CPU1, and right-side are connected to CPU2.

When a table on SSD-2 is scanned using GPU Direct SQL, the optimal GPU choice is GPU-2, and it may be able to use GPU1. However, we have to avoid to choose GPU-3 and GPU-4 due to the restriction of GPUDirect RDMA.
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
LOG:  PG-Strom: GPU0 NVIDIA A100-PCIE-40GB (108 SMs; 1410MHz, L2 40960kB), RAM 39.50GB (5120bits, 1.16GHz), PCI-E Bar1 64GB, CC 8.0
LOG:  [0000:41:00:0] GPU0 (NVIDIA A100-PCIE-40GB; GPU-13943bfd-5b30-38f5-0473-78>
LOG:  [0000:81:00:0] nvme0 (NGD-IN2500-080T4-C) --> GPU0 [dist=9]
LOG:  [0000:82:00:0] nvme2 (INTEL SSDPF2KX038TZ) --> GPU0 [dist=9]
LOG:  [0000:c2:00:0] nvme3 (INTEL SSDPF2KX038TZ) --> GPU0 [dist=9]
LOG:  [0000:c6:00:0] nvme5 (Corsair MP600 CORE) --> GPU0 [dist=9]
LOG:  [0000:c3:00:0] nvme4 (INTEL SSDPF2KX038TZ) --> GPU0 [dist=9]
LOG:  [0000:c1:00:0] nvme1 (INTEL SSDPF2KX038TZ) --> GPU0 [dist=9]
LOG:  [0000:c4:00:0] nvme6 (NGD-IN2500-080T4-C) --> GPU0 [dist=9]
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
pg_strom.nvme_distance_map = 'nvme1=gpu2,nvme2=gpu1,nvme3=gpu1'
```

@ja{
ローカルのNVME-SSDデバイス以外、例えば100Gbイーサネットで接続されたストレージサーバからGPU-Direct SQLを実行する場合など、PCI-Eバス上の距離の概念が当てはまらない場合は、ストレージがマウントされたディレクトリと、そこに関連付けるGPUを指定する事もできます。
以下は設定例です。
}
@en{
If the concept of distance on the PCI-E bus is not suitable, such as when running GPU-Direct SQL from a storage server connected via 100Gb Ethernet, other than a local NVME-SSD device, you can specify the directory where the storage is mounted, and the preferable GPU devices to be associated with.
Below is a setting example.
}
```
pg_strom.nvme_distance_map = '/mnt/0=gpu0,/mnt/1=gpu1'
```

@ja:###GUCパラメータによる制御
@en:###Controls using GUC parameters

@ja{
GPUダイレクトSQL実行に関連するGUCパラメータは2つあります。
}
@en{
There are two GPU parameters related to GPU Direct SQL Execution.
}
@ja{
一つは`pg_strom.gpudirect_enabled`で、GPUダイレクト機能の有効/無効を単純にon/offします。
本パラメータが`off`になっていると、テーブルのサイズや物理配置とは無関係にGPUダイレクトSQL実行は使用されません。デフォルト値は`on`です。
}
@en{
The first is `pg_strom.gpudirect_enabled` that simply turn on/off the function of GPU Direct SQL Execution.
If `off`, GPU Direct SQL Execution should not be used regardless of the table size or physical location. Default is `on`.
}
@ja{
もう一つのパラメータは`pg_strom.gpudirect_threshold`で、GPUダイレクトSQL実行が使われるべき最小のテーブルサイズを指定します。

テーブルの物理配置がNVME-SSD区画（または、NVME-SSDのみで構成されたmd-raid0区画）上に存在し、かつ、テーブルのサイズが本パラメータの指定値よりも大きな場合、PG-StromはGPUダイレクトSQL実行を選択します。
本パラメータのデフォルト値は`2GB`です。つまり、明らかに小さなテーブルに対してはGPUダイレクトSQLではなく、PostgreSQLのバッファから読み出す事を優先します。

これは、一回の読み出しであればGPUダイレクトSQL実行に優位性があったとしても、オンメモリ処理ができる程度のテーブルに対しては、二回目以降のディスクキャッシュ利用を考慮すると、必ずしも優位とは言えないという仮定に立っているという事です。

ワークロードの特性によっては必ずしもこの設定が正しいとは限りません。
}
@en{
The other one is `pg_strom.gpudirect_threshold` which specifies the least table size to invoke GPU Direct SQL Execution.

PG-Strom will choose GPU Direct SQL Execution when target table is located on NVME-SSD volume (or md-raid0 volume which consists of NVME-SSD only), and the table size is larger than this parameter.
Its default configuration is `2GB`. In other words, for obviously small tables, priority is given to reading from PostgreSQL's buffer rather than GPU-Direct SQL.

Even if GPU Direct SQL Execution has advantages on a single table scan workload, usage of disk cache may work better on the second or later trial for the tables which are available to load onto the main memory.

On course, this assumption is not always right depending on the workload charasteristics.
}

@ja:###GPUダイレクトSQL実行の利用を確認する
@en:###Ensure usage of GPU Direct SQL Execution

@ja{
`EXPLAIN`コマンドを実行すると、当該クエリでGPUダイレクトSQL実行が利用されるのかどうかを確認する事ができます。

以下のクエリの例では、`Custom Scan (GpuJoin)`による`lineorder`テーブルに対するスキャンに`NVMe-Strom: enabled`との表示が出ています。この場合、`lineorder`テーブルからの読出しにはGPUダイレクトSQL実行が利用されます。
}
@en{
`EXPLAIN` command allows to ensure whether GPU Direct SQL Execution shall be used in the target query, or not.

In the example below, a scan on the `lineorder` table by `Custom Scan (GpuJoin)` shows `NVMe-Strom: enabled`. In this case, GPU Direct SQL Execution shall be used to read from the `lineorder` table.
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

GPUダイレクトSQL実行はこのインフラを利用しています。つまり、Visibility Mapがセットされており、"all-visible"であるブロックだけがP2P DMAで読み出すようリクエストが送出されます。
}
@en{
We cannot know which row is visible, or invisible at the time when PG-Strom requires P2P DMA for NVMe-SSD, because contents of the storage blocks are not yet loaded to CPU/RAM, and MVCC related attributes are written with individual records. PostgreSQL had similar problem when it supports IndexOnlyScan.

To address the problem, PostgreSQL has an infrastructure of visibility map which is a bunch of flags to indicate whether any records in a particular data block are visible from all the transactions. If associated bit is set, we can know the associated block has no invisible records without reading the block itself.

GPU Direct SQL Execution utilizes this infrastructure. It checks the visibility map first, then only "all-visible" blocks are required to read with P2P DMA.
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

