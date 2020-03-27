@ja:#PG-Strom v2.3リリース
@en:#PG-Strom v2.3 Release

<div style="text-align: right;">PG-Strom Development Team (1-Apr-2020)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v2.3における主要な機能強化は以下の通りです。
- GpuJoinのInnerバッファの構築がCPU並列に対応しました。
- Arrow_FdwがINSERT/TRUNCATEに対応しました。
- pg2arrowコマンドが追記モードに対応しました。
- mysql2arrowコマンドが追加されました。
}

@en{
Major changes in PG-Strom v2.3 includes:
- GpuJoin supports parallel construction of inner buffer
- Arrow_Fdw now becomes writable; supports INSERT/TRUNCATE.
- pg2arrow command supports 'append' mode.
- mysql2arrow command was added.
}

@ja:##動作環境
@en:##Prerequisites

@ja{
- PostgreSQL v10, v11, v12
- CUDA Toolkit 10.1 以降
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 6.0 以降 (Pascal or Volta)
}
@en{
- PostgreSQL v10, v11, v12
- CUDA Toolkit 10.1 or later
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 6.0 or later (Pascal or Volta)
}

@ja:##新機能
@en:##New Features

@ja{
- GpuJoinのInnerバッファの構築がCPU並列に対応
    - 従来はGpuJoinのInner側バッファの構築はバックエンドプロセスのみが行っていました。この制約により、パーティション化されたテーブルの並列スキャンが極端に遅延するという問題がありました。
    - 本バージョンでの機能強化により、バックエンド、ワーカープロセスのどちらでもInner側バッファを構築する事が可能となりました。パーティション化テーブルをスキャンする場合でも、各パーティション子テーブルに割り当てられたプロセスが直ちにGpuJoin処理を開始する事ができるようになります。
- Partition-wise Asymmetric GpuJoinの再設計
    - 全体的なデザインの再設計を行い、適切な局面において多段GpuJoinが選択されやすくなるよう改良を行いました。
- Arrow_FdwがINSERT/TRUNCATEに対応しました。
    - Arrow_Fdw外部テーブルに対して、`INSERT`によるバルクロードと、`pgstrom.arrow_fdw_truncate`によるTRUNCATE処理を行う事が可能となりました。
- CuPy連携とデータフレームのGPUエクスポート
    - Arrow_Fdw外部テーブルの内容をGPUデバイスメモリ上のデータフレームにロードし、これをCuPyの`cupy.ndarray`オブジェクトとしてPythonスクリプトから参照する事が可能となりました。
- pg2arrowコマンドが追記モードに対応しました。
    - `pg2arrow`コマンドに`--append`オプションが追加され、既存のApache Arrowに対して追記を行う事が可能となりました。
    - また同時に、`SELECT * FROM table`の別名表記として`-t table`オプションが追加されました。
- mysql2arrowコマンドを追加されました。
    - PostgreSQLではなく、MySQLに接続してクエリを実行し、その結果をApache Arrowファイルとして保存する`mysql2arrow`コマンドを追加しました。
    - 列挙型のデータも通常のUtf8型として保存する（DictionaryBatchを使用しない）以外は、`pg2arrow`と同等の機能を持っています。
- リグレッションテストを追加しました
    - PostgreSQLのリグレッションテストフレームワークに合わせて、幾つかの基本的なテストケースを追加しています。
}
@en{
- GpuJoin supports parallel construction of inner buffer
    - The older version construct inner buffer of GpuJoin by the backend process only. This restriction leads a problem; parallel scan of partitioned table delays extremely.
    - This version allows both of the backend and worker processes to construct inner buffer. In case when we scan a partitioned table, any processes that is assigned to a particular child table can start GpuJoin operations immediately.
- Refactoring of the partition-wise asymmetric GpuJoin
    - By the refactoring of the partition-wise asymmetric GpuJoin, optimizer becomes to prefer multi-level GpuJoin in case when it offers cheaper execution cost.
- Arrow_Fdw becomes writable; INSERT/TRUNCATE supported
    - Arrow_Fdw foreign table allows bulk-loading by `INSERT` and data elimination by `pgstrom.arrow_fdw_truncate`.
- pg2arrow command supports 'append' mode.
    - We added `--append` option for `pg2arrow` command. As literal, it appends query results on existing Apache Arrow file.
    - Also, `-t table` option was added as an alias of `SELECT * FROM table`.
- mysql2arrow command was added.
    - We added `mysql2arrow` command that connects to MySQL server, not PostgreSQL, and write out SQL query results as Apache Arrow files.
    - It has equivalent functionality to `pg2arrow` except for enum data type. `mysql2arrow` saves enum values as flat Utf8 values without DictionaryBatch chunks.
- Regression test was added
    - Several test cases were added according to the PostgreSQL regression test framework.
}

@ja:##修正された主な不具合
@en:##Significant bug fixes

@ja{
- GPUデバイス関数/型のキャッシュ無効化のロジックを改善
    - ALTERコマンドの実行時、全てのGPUデバイス関数/型のメタ情報キャッシュを無効化していましたが、実際に無効化の必要のあるエントリのみをクリアするよう修正を行いました。
- GROUP BYで同じ列を偶数回指定した際に極端なパフォーマンスの低下を修正
    - GROUP BYのキー値が複数ある時に、GpuPreAggはハッシュ値をXORで結合していました。そのため、同じ列を偶数回指定した場合には常にハッシュインデックスが0になるという問題がありました。適当なランダム化処理を加える事でハッシュ値が分散するよう修正しています。
- 潜在的なGpuScan無限ループの問題を修正
    - SSD2GPU Direct SQLの使用時、変数の未初期化によりGpuScanが無限ループに陥る可能性がありました。
- 潜在的なGpuJoinのGPUカーネルクラッシュ
    - 3個以上のテーブルを結合するGpuJoinで、変数の未初期化によりGPUカーネルのクラッシュを引き起こす可能性がありました。
}
@en{
- Revised cache invalidation logic for GPU device functions / types
    - The older version had invalidated all the metadata cache entries of GPU device functions / type on execution of ALTER command. It was revised to invalidate the entries that are actually updated.
- Revised extreme performance degradation if GROUP BY has same grouping key twice or even number times.
    - GpuPreAgg combined hash values of grouping key of GROUP BY using XOR. So, if case when same column appeared even number time, it always leads 0 for hash-index problematically. Now we add a randomization for better hash distribution.
- Potential infinite loop on GpuScan
    - By uninitialized values, GpuScan potentially goes to infinite loop when SSD2GPU Direct SQL is available.
- Potential GPU kernel crash on GpuJoin
    - By uninitialized values, GpuJoin potentially makes GPU kernel crash when 3 or more tables are joined.
}


@ja:##廃止された機能
@en:##Deprecated Features

@ja{
- PostgreSQL v9.6サポート
    - PostgreSQL v9.6のCustomScan APIには、動的共有メモリ(DSM)の正しいハンドリングに必要な幾つかのAPIが欠けており、v10以降と共通のコードを保守する上で障害となっていました。これらの問題から、本バージョンでは PostgreSQL v9.6 はサポート外となります。
- PL/CUDA
    - ユースケースを分析した結果、独自のプログラミング環境よりも、Python言語などユーザの使い慣れた言語環境の方が望ましい事が分かりました。
    - 今後は、Arrow_FdwのGPUエクスポート機能とPL/Python経由でのCuPy呼出しを併用する事で、In-database機械学習/統計解析の代替手段となります。
- Gstore_Fdw
    - 本機能は、書き込み可能Arrow_FdwとGPUエクスポート機能により代替されました。
- Largeobject～GPU間エクスポート/インポート
    - ユースケースを分析した結果、本機能は不要と判断しました。
}

@en{
- PostgreSQL v9.6 Support
    - CustomScan API in PostgreSQL v9.6 lacks a few APIs to handle dynamic shared memory (DSM). It has been a problem to handle a common code for v10 or later. To avoid the problem, we dropped PostgreSQL v9.6 support in this version.
- PL/CUDA
    - According to the usecase analytics, users prefer familiar programming language environment like Python, rather than own special environment.
    - A combination of Arrow_Fdw's GPU export functionality and CuPy invocation at PL/Python is a successor of PL/CUDA, for in-database machine-learning / statistical analytics.
- Gstore_Fdw
    - This feature is replaced by the writable Arrow_Fdw and its GPU export functionality.
- Largeobject export to/import from GPU
    - According to the usecase analytics, we determined this feature is not needed.
}

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


@ja:#PG-Strom v2.0リリース
@en:#PG-Strom v2.0 Release

<div style="text-align: right;">PG-Strom Development Team (17-Apr-2018)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v2.0における主要な機能強化は以下の通りです。

- GPUを管理する内部インフラストラクチャの全体的な再設計と安定化
- CPU+GPUハイブリッド並列実行
- SSD-to-GPUダイレクトSQL実行
- インメモリ列指向キャッシュ
- GPUメモリストア(store_fdw)
- GpuJoinとGpuPreAggの再設計に伴う高速化
- GpuPreAgg+GpuJoin+GpuScan 密結合GPUカーネル

新機能のサマリはこちらからダウンロードできます：[PG-Strom v2.0 Technical Brief](./blob/20180417_PGStrom_v2.0_TechBrief.pdf).
}

@en{
Major enhancement in PG-Strom v2.0 includes:

- Overall redesign of the internal infrastructure to manage GPU and stabilization
- CPU+GPU hybrid parallel execution
- SSD-to-GPU Direct SQL Execution
- In-memory columnar cache
- GPU memory store (gstore_fdw)
- Redesign of GpuJoin and GpuPreAgg and speed-up
- GpuPreAgg + GpuJoin + GpuScan combined GPU kernel

You can download the summary of new features from: [PG-Strom v2.0 Technical Brief](./blob/20180417_PGStrom_v2.0_TechBrief.pdf).
}

@ja:##動作環境
@en:##Prerequisites

@ja{
- PostgreSQL v9.6, v10
- CUDA Toolkit 9.1
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 6.0 以降 (Pascal以降)
}
@en{
- PostgreSQL v9.6, v10
- CUDA Toolkit 9.1
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 6.0 or later (Pascal or Volta)
}

@ja:##新機能
@en:##New Features

@ja{
- GPUを管理する内部インフラストラクチャの全体的な再設計と安定化
    - PostgreSQLバックエンドプロセスは同時に１個のGPUだけを利用するようになりました。マルチGPUを利用する場合はPostgreSQLのCPU並列との併用が前提になりますが、CPUスレッドがGPUへデータを供給するスループットはGPUの処理能力よりもずっと低いため、通常、これは問題とはなりません。ソフトウェア設計のシンプル化を優先しました。
    - Pascal世代以降のGPUで採用されたGPUデバイスメモリのデマンドページングをほぼ全面的に採用するようになりました。SQLワークロードの多くは実際に実行してみるまで必要な結果バッファの大きさが分からないため、これまでは必要以上にバッファを獲得し、またメモリ不足時には再実行を行っていましたが、これらは同時実行プロセスの利用可能なリソースを制限し、また複雑な例外ロジックはバグの温床でした。GPUデバイスメモリのデマンドページングを利用する事で、設計のシンプル化を行いました。
    - CUDAの非同期インターフェースの利用を止めました。GPUデバイスメモリのデマンドページングを利用すると、DMA転送のための非同期API（`cuMemCpyHtoD`など）は同期的に振舞うようになるため、GPUカーネルの多重度が低下してしまいます。代わりにPG-Strom自身がワーカースレッドを管理し、これらのワーカースレッドがそれぞれ同期APIを呼び出すよう設計変更を行いました。副産物として、非同期コールバック（`cuStreamAddCallback`）を利用する必要がなくなったので、MPSを利用する事が可能となりました。
}
@en{
- Entire re-design and stabilization of the internal infrastructure to manage GPU device.
    - PostgreSQL backend process simultaneously uses only one GPU at most. In case of multi-GPUs installation, it assumes combination use with CPU parallel execution of PostgreSQL. Usually, it is not a matter because throughput of CPU to provide data to GPU is much narrower than capability of GPU processors. We prioritized simpleness of the software architecture.
    - We began to utilize the demand paging feature of GPU device memory supported at the GPU models since Pascal generation. In most of SQL workloads, we cannot know exact size of the required result buffer prior to its execution, therefore, we had allocated more buffer than estimated buffer length, and retried piece of the workloads if estimated buffer size is not sufficient actually. This design restricts available resources of GPU which can be potentially used for other concurrent processes, and complicated error-retry logic was a nightmare for software quality. The demand paging feature allows to eliminate and simplify these stuffs.
    - We stop to use CUDA asynchronous interface. Use of the demand paging feature on GPU device memory makes asynchronous APIs for DMA (like `cuMemCpyHtoD`) perform synchronously, then it reduces concurrency and usage ratio of GPU kernels. Instead of the CUDA asynchronous APIs, PG-Strom manages its own worker threads which call synchronous APIs for each. As a by-product, we also could eliminate asynchronous callbacks (`cuStreamAddCallback`), it allows to use MPS daemon which has a restriction at this API.
}

@ja{
- CPU+GPUハイブリッド並列実行
    - PostgreSQL v9.6で新たにサポートされたCPU並列実行に対応しました。
    - PG-Stromの提供するGpuScan、GpuJoinおよびGpuPreAggの各ロジックは複数のPostgreSQLバックグラウンドワーカープロセスにより並列に実行する事が可能です。
    - PostgreSQL v9.6ではCPU並列実行の際に`EXPLAIN ANALYZE`で取得するPG-Strom独自の統計情報が正しくありません。これは、CustomScanインターフェースAPIで`ShutdownCustomScan`が提供されていなかったため、DSM（動的共有メモリ）の解放前にコーディネータプロセスがワーカープロセスの情報を回収する手段が無かったためです。
}
@en{
- CPU+GPU Hybrid Parallel Execution
    - CPU parallel execution at PostgreSQL v9.6 is newly supported.
    - CustomScan logic of GpuScan, GpuJoin and GpuPreAgg provided by PG-Strom are executable on multiple background worker processes of PostgreSQL in parallel.
    - Limitation: PG-Strom's own statistics displayed at `EXPLAIN ANALYZE` if CPU parallel execution. Because PostgreSQL v9.6 does not provide `ShutdownCustomScan` callback of the CustomScan interface, coordinator process has no way to reclaim information of worker processes prior to the release of DSM (Dynamic Shared Memory) segment.
}
@ja{
- SSD-to-GPUダイレクトSQL実行
    - Linuxカーネルモジュール `nvme_strom` を用いる事で、NVMe規格に対応したSSD上のPostgreSQLデータブロックを、CPU/RAMを介さずダイレクトにGPUデバイスメモリへ転送する事が可能となりました。システムRAMに載り切らない大きさのデータを処理する場合であっても、本機能によりPG-Stromの適用が現実的な選択肢となる事でしょう。
    - ブロックデバイス層やファイルシステムを経由しないためハードウェア限界に近い高スループットを引き出す事が可能で、かつ、GPUでSQLワークロードを処理するためCPUの処理すべきデータ量を減らす事ができます。このような特性の組み合わせにより、一般的には計算ワークロードのアクセラレータとして認識されているGPUを、I/Oワークロードの高速化に適用する事に成功しました。
}
@en{
- SSD-to-GPU Direct SQL Execution
    - By cooperation with the `nvme_strom` Linux kernel module, it enables to load PostgreSQL's data blocks on NVMe-SSD to GPU device memory directly, bypassing the CPU and host buffer. This feature enables to apply PG-Strom on the area which have to process large data set more than system RAM size.
    - It allows to pull out pretty high throughput close to the hardware limitation because its data stream skips block-device or filesystem layer. Then, GPU runs SQL workloads that usually reduce the amount of data to be processed by CPU. The chemical reaction of these characteristics enables to redefine GPU's role as accelerator of I/O workloads also, not only computing intensive workloads.
}
@ja{
- インメモリ列指向キャッシュ
    - RAMサイズに載る程度の大きさのデータに対しては、よりGPUでの処理に適した列データ形式に変形してキャッシュする事が可能になりました。テーブルのスキャンに際して、列指向キャッシュが存在する場合にはPostgreSQLの共有バッファよりもこちらを優先して参照します。
    - インメモリ列指向キャッシュは同期的、または非同期的にバックグラウンドで構築する事が可能です。
    - 初期のPG-Stromで似たような機能が存在していた事を覚えておられるかもしれません。v2.0で新たに実装された列指向キャッシュは、キャッシュされた行が更新されると、当該行を含むキャッシュブロックを消去（invalidation）します。行ストアの更新に合わせて列キャッシュ側の更新を行うという事は行わないため、更新ワークロードに対するパフォーマンスの低下は限定的です。
}
@en{
- In-memory Columnar Cache
    - For middle size data-set loadable onto the system RAM, it allows to cache data-blocks in column format which is more suitable for GPU computing. If cached data-blocks are found during table scan, PG-Strom prefers to reference the columnar cache more than shared buffer of PostgreSQL.
    - In-memory columnar cache can be built synchronously, or asynchronously by the background workers.
    - You may remember very early revision of PG-Strom had similar feature. In case when a cached tuple gets updated, the latest in-memory columnar cache which we newly implemented in v2.0 invalidates the cache block which includes the updated tuples. It never updates the columnar cache according to the updates of row-store, so performance degradation is quite limited.
}
@ja{
- GPUメモリストア(gstore_fdw)
    - GPU上に確保したデバイスメモリ領域に対して、外部テーブル（Foreign Table）のインターフェースを利用してSQLのSELECT/INSERT/UPDATE/DELETEにより読み書きを行う機能です。
    - 内部データ形式は `pgstrom` 型のみがサポートされています。これは、PG-Stromのバッファ形式`KDS_FORMAT_COLUMN`タイプと同一の形式でデータを保持するものです。可変長データを保存する場合、LZ方式によるデータ圧縮を行う事も可能です。
    - v2.0の時点では、GPUメモリストアはPL/CUDA関数のデータソースとしてだけ利用する事が可能です。
}
@en{
- GPU Memory Store (gstore_fdw)
    - It enables to write to / read from preserved GPU device memory region by SELECT/INSERT/UPDATE/DELETE in SQL-level, using foreign table interface.
    - In v2.0, only `pgstrom` internal data format is supported. It saves written data using PG-Strom's buffer format of `KDS_FORMAT_COLUMN`. It can compress variable length data using LZ algorithm.
    - In v2.0, GPU memory store can be used as data source of PL/CUDA user defined function.
}
@ja{
- GpuJoinとGpuPreAggの再設計に伴う高速化
    - 従来、GpuJoinとGpuPreAggで内部的に使用していたDynamic Parallelismの利用をやめ、処理ロジック全体の見直しを行いました。これは、GPUサブカーネルの起動後、その完了を単に待っているだけのGPUカーネルが実行スロットを占有し、GPUの使用率が上がらないという問題があったためです。
    - この再設計に伴う副産物として、GpuJoinのサスペンド/レジューム機能が実装されました。原理上、SQLのJOIN処理は入力した行数よりも出力する行数の方が増えてしまう事がありますが、処理結果を書き込むバッファの残りサイズが不足した時点でGpuJoinをサスペンドし、新しい結果バッファを割り当ててレジュームするように修正されました。これにより、結果バッファのサイズ推定が簡略化されたほか、実行時のバッファ不足による再実行の必要がなくなりました。
}
@en{
- Redesign and performance improvement of GpuJoin and GpuPreAgg
    - Stop using Dynamic Parallelism which we internally used in GpuJoin and GpuPreAgg, and revised entire logic of these operations. Old design had a problem of less GPU usage ratio because a GPU kernel which launches GPU sub-kernel and just waits for its completion occupied GPU's execution slot.
    - A coproduct of this redesign is suspend/resume of GpuJoin. In principle, JOIN operation of SQL may generate larger number of rows than number of input rows, but preliminary not predictive. The new design allows to suspend GPU kernel once buffer available space gets lacked, then resume with new result buffer. It simplifies size estimation logic of the result buffer, and eliminates GPU kernel retry by lack of buffer on run-time.
}
@ja{
- GpuPreAgg+GpuJoin+GpuScan 密結合GPUカーネル
    - GPUで実行可能なSCAN、JOIN、GROUP BYが連続しているとき、対応するGpuScan、GpuJoin、GpuPreAggに相当する処理を一回のGPUカーネル呼び出しで実行する事が可能になりました。これは、GpuJoinの結果バッファをそのままGpuPreAggの入力バッファとして扱うなど、CPUとGPUの間のデータ交換を最小限に抑えるためのアプローチです。
    - この機能は特に、SSD-to-GPUダイレクトSQL実行と組み合わせて使用すると効果的です。
}
@en{
- GpuPreAgg+GpuJoin+GpuScan combined GPU kernel
    - In case when GPU executable SCAN, JOIN and GROUP BY are serially cascaded, a single GPU kernel invocation runs a series of tasks equivalent to the GpuScan, GpuJoin and GpuPreAgg. This is an approach to minimize data exchange between CPU and GPU. For example, result buffer of GpuJoin is used as input buffer of GpuPreAgg.
    - This feature is especially valuable if combined with SSD-to-GPU Direct SQL Execution.
}
@ja{
- 新しいデータ型の対応
    - `uuid`型に対応しました。
    - ネットワークアドレス型（`inet`、`cidr`、および`macaddr`）に対応しました。
    - 範囲型（`int4range`、`int8range`、`tsrange`、`tstzrange`、`daterange`）に対応しました。
    - 半精度浮動小数点型（`float2`）に対応しました。半精度浮動小数点型に関連するCPU側の実装はPG-Stromの独自実装によるものです。
- 新しい演算子/関数の対応
    - 日付時刻型に対する`EXTRACT(field FROM timestamp)`演算子に対応しました。
}
@ja{
- New data type support
    - `uuid` type
    - Network address types (`inet`, `cidr` and `macaddr`)
    - Range data types (`int4range`, `int8range`, `tsrange`, `tstzrange`, `daterange`)
    - Half-precision floating point type (`float2`). Its CPU side are also implemented by PG-Strom itself, not PostgreSQL's built-in feature.
- New operators / functions
    - `EXTRACT(field FROM timestamp)` operator on the date and time types
}
@ja{
- PL/CUDA関連の強化
    - `#plcuda_include`の拡張により、`text`型を返すSQL関数を指定できるようになりました。引数の値によって挿入するコードを変える事ができるため、単に外部定義関数を読み込むだけでなく、動的にいくつものGPUカーネルのバリエーションを作り出すことも可能です。
    - PL/CUDA関数の引数に`reggstore`型を指定した場合、GPUカーネル関数へは対応するGPUメモリストアのポインタが渡されます。OID値が渡されるわけではない事に留意してください。
}
@en{
- PL/CUDA Enhancement
    - `#plcuda_include` is enhanced to specify SQL function which returns `text` type. It can change the code block to inject according to the argument, so it also allows to generate multiple GPU kernel variations, not only inclusion of externally defined functions.
    - If PL/CUDA takes `reggstore` type argument, GPU kernel function receives pointer of the GPU memory store. Note that it does not pass the OID value.
}
@ja{
- その他の機能強化
    - `lo_import_gpu`および`lo_export_gpu`関数により、外部アプリケーションの確保したGPUメモリの内容を直接PostgreSQLのラージオブジェクトに記録したり、逆にラージオブジェクトの内容をGPUメモリに書き出す事が可能です。

}
@en{
- Other Enhancement
    - `lo_import_gpu` and `lo_export_gpu` functions allows to import contents of the GPU device memory acquired by external applications directly, or export contents of the largeobject to the GPU device memory.
}
@ja{
- パッケージング
    - PostgreSQL Global Development Groupの配布するPostgreSQLパッケージに適合するよう、RPMパッケージ化を行いました。
    - 全てのソフトウェア物件はHeteroDB SWDC(Software Distribution Center)よりダウンロードが可能です。
}
@en{
- Packaging
    - Add RPM packages to follow the PostgreSQL packages distributed by PostgreSQL Global Development Group.
    - All the software packages are available at HeteroDB SWDC(Software Distribution Center) and downloadable.
}
@ja{
- ドキュメント
    - PG-Stromドキュメントをmarkdownとmkdocsを用いて全面的に書き直しました。従来のHTMLを用いたアプローチに比べ、よりメンテナンスが容易で新機能の開発に合わせたドキュメントの拡充が可能となります。
}
@en{
- Document
    - PG-Strom documentation was entirely rewritten using markdown and mkdocs. It makes documentation maintenance easier than the previous HTML based approach, so expects timely updates according to the development of new features.
}
@ja{
- テスト
    - PostgreSQLのリグレッションテストフレームワークを使用して、PG-Stromのリグレッションテストを作成しました。
}
@en{
- Test
    - Regression test for PG-Strom was built on top of the regression test framework of PostgreSQL.
}

@ja:##廃止された機能
@en:##Dropped features

@ja{
- PostgreSQL v9.5サポート
    - PostgreSQL v9.6ではCPU並列クエリの提供に伴い、オプティマイザ/エグゼキュータ共に大きな修正が加えられました。これらと密接に連携する拡張モジュールにとって最もインパクトの大きな変更は『upper planner path-ification』と呼ばれるインターフェースの強化で、集約演算やソートなどの実行計画もコストベースで複数の異なる方法を比較して最適なものを選択できるようになりました。
    - これはGpuPreAggを実装するためにフックを利用して実行計画を書き換えていた従来の方法とは根本的に異なり、より合理的かつ信頼できる方法でGPUを用いた集約演算を挿入する事が可能となり、バグの温床であった実行計画の書き換えロジックを捨てる事が可能になりました。
    - 同時に、CustomScanインターフェースにもCPU並列に対応するためのAPIが拡張され、これらに対応するためにPostgreSQL v9.5サポートは廃止されました。
}
@en{
- PostgreSQL v9.5 Support
    - PostgreSQL v9.6 had big changes in both of the optimizer and executor to support CPU parallel query execution. The biggest change for extension modules that interact them is an enhancement of the interface called "upper planner path-ification". It allows to choose an optimal execution-plan from the multiple candidates based on the estimated cost, even if it is aggregation or sorting.
    - It is fundamentally different from the older way where we rewrote query execution plan to inject GpuPreAgg using the hooks. It allows to inject GpuPreAgg node in more reasonable and reliable way, and we could drop complicated (and buggy) logic to rewrite query execution plan once constructed.
    - CustomScan interface is also enhanced to support CPU parallel execution. Due to the reason, we dropped PostgreSQL v9.5 support to follow these new enhancement.
}
@ja{
- GpuSort機能
    - GpuSort機能は性能上のメリットが得られないため廃止されました。
    - ソートはGPUの得意とするワークロードの一つです。しかし、GPUデバイスメモリの大きさを越えるサイズのデータをソートする場合、複数のチャンクに分割して部分ソートを行い、後でCPU側でこれを結合して最終結果を出力する必要があります。
    - 結合フェーズの処理を軽くするには、GPUでソートすべきチャンクのサイズを大きく必要がありますが、一方でチャンクサイズが大きくなるとソート処理を開始するためのリードタイムが長くなり、PG-Stromの特長の一つである非同期処理によるデータ転送レイテンシの隠ぺいが効かなくなるというトレードオフがあります。
    - これらの問題に対処するのは困難、少なくとも時期尚早であると判断し、GpuSort機能は廃止されました。
}
@en{
- GpuSort feature
    - We dropped GpuSort because we have little advantages in the performance.
    - Sorting is one of the GPU suitable workloads. However, in case when we try to sort data blocks larger than GPU device memory, we have to split the data blocks into multiple chunks, then partially sort them and merge them by CPU to generate final results.
    - Larger chunk size is better to reduce the load to merge multiple chunks by CPU, on the other hands, larger chunk size takes larger lead time to launch GPU kernel to sort. It means here is a trade-off; which disallows asynchronous processing by PG-Strom to make data transfer latency invisible.
    - It is hard to solve the problem, or too early to solve the problem, we dropped GpuSort feature once.
}

