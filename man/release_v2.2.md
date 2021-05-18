@ja:#PG-Strom v2.2リリース
@en:#PG-Strom v2.2 Release

<div style="text-align: right;">PG-Strom Development Team (1-May-2019)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v2.2における主要な機能強化は以下の通りです。

- テーブルパーティションへの対応
- Arrow_Fdwによる列指向ストアのサポート
- ビルド済みGPUバイナリへの対応
- Jsonbデータ型の対応
- 可変長データ型を返すGPU関数の対応
- GPUメモリストア（Gstore_Fdw）のソート対応
- NVMEoFへの対応（実験的機能）
}
@en{
Major enhancement in PG-Strom v2.2 includes:

- Table partitioning support
- Columnar store support with Arrow_Fdw
- Pre-built GPU binary support
- Enables to implement GPU functions that returns variable length data
- GpuSort support on GPU memory store (Gstore_Fdw)
- NVME-oF support (Experimental)
}

@ja:##動作環境
@en:##Prerequisites

@ja{
- PostgreSQL v9.6, v10, v11
- CUDA Toolkit 10.1
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 6.0 以降 (Pascal以降)
}
@en{
- PostgreSQL v9.6, v10, v11
- CUDA Toolkit 10.1
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 6.0 or later (Pascal or Volta)
}

@ja:##新機能
@en:##New Features

@ja{
- テーブルパーティションへの対応
    - マルチGPU構成の場合、パーティションを構成する子テーブルの物理的なGPUとの距離に応じて最適なGPUを選択するようになりました。NVME-oF環境などPCIeバスの構成だけでは最適距離を判断できない場合は、DB管理者は`pg_strom.nvme_distance_map`パラメータを用いて対応関係を設定する事ができます。
    - 非パーティションテーブルとのJOIN時、パーティション子テーブルと非パーティションテーブルとのJOINを行った後、各子テーブルの処理結果を結合するような実行計画を生成できるようになりました。本機能は Asymmetric Partition-wise JOIN という名称でPostgreSQL v13の本体機能へと提案されています。
- Arrow_Fdwによる列指向ストアのサポート
    - 外部テーブル経由でApache Arrow形式ファイルの読み出しに対応するようになりました。
    - SSD-to-GPU Direct SQLを用いたApache Arrowの読み出しとSQL実行にも対応しています。
- ビルド済みGPUバイナリへの対応
    - SQLからGPUバイナリコードを生成する際、従来は動的に変更する要素のない関数群（ライブラリ関数に酷似）も含めてCUDA Cのソースコードを生成し、それをNVRTC(NVIDIA Run-Time Compiler)を用いてビルドしていました。しかし、一部の複雑な関数の影響でビルド時間が極端に長くなるという問題がありました。
    - v2.2において、静的な関数群は事前にビルドされ、SQLから動的に生成する部分のみを実行時にコンパイルするように変更されました。これにより、GPUバイナリの生成時間が大幅に減少する事となりました。
- JSONBデータ型の対応
    - GPU側でJSONBオブジェクトの子要素を参照し、`numeric`や`text`値として条件句などで利用できるようになった。
- 可変長データ型を返すGPU関数の対応
    - `textcat`など可変長データ型を返すSQL関数をGPU側で実装する事ができるようになった。
- GPUメモリストア（Gstore_Fdw）のソート対応
    - PL/CUDAのデータソースとして利用する以外に、GPUメモリストアからデータを読み出して実行するSQLをGPUで実行する事ができるようになりました。
    - 対応しているワークロードはGpuScanおよびGpuSortの２種類で、JOINおよびGROUP BYにはまだ対応していません。
- リグレッションテストの追加
    - 簡易なテストのため、リグレッションテストを追加しました。
- NVME-oFへの対応（実験的機能）
    - NVME-over-Fabricを用いてマウントされたリモートのNVMEディスクからのSSD-to-GPU Direct SQLに対応しました。ただし、Red Hat Enterprise Linux 7.x / CentOS 7.xでは`nvme_rdma`ドライバの入れ替えが必要となり、現在のところ実験的機能という形になっています。
}
@en{
- Table partitioning support
    - If multi-GPUs configuration, an optimal GPU shall be chosen according to the physical distance between GPU and child tables that construct a partition. If PG-Strom cannot identify the distance from PCIe-bus topology, like NVME-oF configuration, DBA can configure the relation of GPU and NVME-SSD using `pg_strom.nvme_distance_map`.
    - When we join a partitioned table with non-partition tables, this version can produce a query execution plan that preliminary joins the non-partitioned table with partition child tables for each, and gather the results from child tables. This feature is proposed to PostgreSQL v13 core, as Asymmetric Partition-wise JOIN.
- Columnar store support with Arrow_Fdw
    - It supports to read external Apache Arrow files using foreign table.
    - It also supports SSD-to-GPU Direct SQL on Apache Arrow files.
- Pre-built GPU binary support
    - When GPU binary code is generated from SQL, the older version wrote out eitire CUDA C source code, including static portions like libraries, then NVRTC(NVIDIA Run-Time Compiker) built them on the fly. However, a part of complicated function consumed much longer compilation time.
    - v2.2 preliminary builds static functions preliminary, and only dynamic portion from SQL are built dynamically. It reduces the time for GPU binary generation.
- JSONB data type support
    - This version allows to reference elements of JSONB object, and to utilize them as `numeric` or `test`.
- Enables to implement GPU functions that returns variable length data
    - This version allows to implement SQL functions that returns variable-length data, like `textcat`, on GPU devices.
- GpuSort support on GPU memory store (Gstore_Fdw)
    - This version allows to read data from GPU memory store for SQL workloads execution, not only PL/CUDA.
- Addition of regression test
    - Several simple regression tests are added.
- NVME-oF support (Experimental)
    - It supports SSD-to-GPU Direct SQL from remote SSD disks which are mounted using NVME-over-Fabric. Please note that it is an experimental feature, and it needs to replace the `nvme_rdma` kernel module on Red Hat Enterprise Linux 7.x / CentOS 7.x.
}

@ja:##将来廃止予定の機能
@en:##Features to be deprecated

@ja{
- PostgreSQL v9.6サポート
    - PostgreSQL v9.6のCustomScan APIには、動的共有メモリ(DSM)の正しいハンドリングに必要な幾つかのAPIが欠けており、実行時統計情報の採取などが不可能でした。
    - また、内部的に式表現(Expression)を保持するための方法にも変更が加えられている事から、少なくない箇所で `#if ... #endif` ブロックが必要となり、コードの保守性を損なっていました。
    - これらの問題により、PostgreSQL v9.6サポートは本バージョンが最後となります。PG-StromをPostgreSQL v9.6でお使いの場合は、早期にPostgreSQL v11へと移行される事をお勧めします。
}
@en{
- PostgreSQL v9.6 support
    - CustomScan API in PostgreSQL v9.6 lacks a few APIs to handle dynamic shared memory (DSM), so it is unable to collect run-time statistics.
    - It also changes the way to keep expression objects internally, therefore, we had to put `#if ... #endif` blocks at no little points. It has damaged to code maintainability.
    - Due to the problems, this is the last version to support PostgreSQL v9.6. If you applied PG-Strom on PostgreSQL v9.6, let us recommend to move PostgreSQL v11 as soon as possible.
}
@ja{
- Gstore_Fdw外部テーブルのpgstromフォーマット
    - GPUメモリストア上のデータ形式は、元々PL/CUDAのデータソースとして利用するために設計された独自の列形式で、可変長データやnumericデータ型の表現はPostgreSQLのものをそのまま利用していました。
    - その後、GPU上でのデータ交換用共通形式として、Apache Arrow形式を元にしたNVIDIA RAPIDS(cuDF)が公開され、多くの機械学習ソフトウェアやPythonでのソフトウェアスタックなど対応が強化されつつあります。
    - 今後、PG-StromはGstore_Fdwの内部データ形式をcuDFと共通のフォーマットに変更し、これら機械学習ソフトウェアとの相互運用性を改善します。コードの保守性を高くするため、従来の独自データ形式は廃止となります。
}
@en{
- The `pgstrom` format of Gstore_Fdw foreign table
    - The internal data format on GPU memory store (Gstore_Fdw) is originally designed for data source of PL/CUDA procedures. It is our own format, and used PostgreSQL's data representations as is, like variable-length data, numeric, and so on.
    - After that, NVIDIA released RAPIDS(cuDF), based on Apache Arrow, for data exchange on GPU, then its adoption becomes wider on machine-learning application and Python software stack.
    - PG-Strom will switch its internal data format of Gstore_Fdw, to improve interoperability with these machine-learning software, then existing data format shall be deprecated.
}


@ja:##廃止された機能
@en:##Dropped Features

@ja{
- インメモリ列キャッシュ
    - ユースケースを分析した結果、多くのケースではArrow_Fdwで十分に代替可能なワークロードである事が分かりました。重複機能であるため、インメモリ列キャッシュは削除されました。
}
@en{
- In-memory columnar cache
    - As results of use-case analysis, we concluded Arrow_Fdw can replace this feature in most cases. Due to feature duplication, we dropped the in-memory columnar cache.
}
