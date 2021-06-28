@ja:#PG-Strom v3.0リリース
@en:#PG-Strom v3.0 Release

<div style="text-align: right;">PG-Strom Development Team (29-Jun-2021)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v3.0における主要な変更は点は以下の通りです。

- NVIDIA GPUDirect Storage (cuFile) に対応しました。
- いくつかのPostGIS関数がGPUで実行可能となりました。
- GiSTインデックスを使用したGpuJoinに対応しました。
- 新たにGPUキャッシュ機能が実装されました。
- ユーザ定義のGPUデータ型/関数/演算子に対応しました。(実験的)
- ソフトウェアライセンスをGPLv2からPostgreSQLライセンスへと切り替えました。
}

@en{
Major changes in PG-Strom v3.0 are as follows:

- NVIDIA GPUDirect Storage (cuFile) is now supported.
- Several PostGIS functions are executable on GPUs.
- GpuJoin using GiST index is now supported.
- GPU Cache mechanism is newly implemented.
- User-defined GPU data types/functions/operators are experimentally supported.
- Software license was switched from GPLv2 to PostgreSQL license.
}


@ja:##動作環境
@en:##Prerequisites

@ja{
- PostgreSQL v11, v12, v13
- CUDA Toolkit 11.2 以降
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 6.0 以降 (Pascal以降)
}
@en{
- PostgreSQL v11, v12, v13
- CUDA Toolkit 11.2 or later
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 6.0 or later (Pascal or newer)
}

##NVIDIA GPUDirect Storage

@ja{
[GPUダイレクトSQL](../ssd2gpu)用のドライバとして、従来の nvme_strom カーネルモジュールに加えて、
NVIDIAが開発を進めている[GPUDirect Storage](https://developer.nvidia.com/blog/gpudirect-storage/)にも対応しました。
}
@en{
[GPUDirect Storage](https://developer.nvidia.com/blog/gpudirect-storage/), has been developed by NVIDIA, is now supported as a driver for [GPU Direct SQL](../ssd2gpu), in addition to the existing nvme_strom kernel module.
}
![GPUDirect Storage](./img/release_3_0a.png)

@ja{
どちらのドライバも概ね同等の機能、性能を有していますが、GPUDirect Storageの対応により、従来からのローカルNVME-SSDに加えて、NVME-oF(NVME over Fabrics)デバイスやSDS(Software Defined Storage)デバイス、およびその上に構築された共有ファイルシステムからのGPUダイレクトSQLにも対応する事となり、より大規模で柔軟なストレージ構成を取る事が可能になります。
}
@en{
Both of drivers have almost equivalent functionalities and performance, but supports of GPUDirect Storage enables P2P direct read from NVME-oF (NVME over Fabrics) devices, SDS (Software Defined Storage) devices and shared filesystem built on these devices. Therefore, it offers larger and more flexible storage configuration.
}

@ja{
GPUDirect SQLは、PostgreSQL標準のHeapテーブルとApache Arrowファイルの読み出しに利用する事ができ、いずれの場合においても、テーブルスキャンがボトルネックとなるようなワークロードにおいて顕著な高速化が期待できます。

以下の測定結果は、GPU 1台とNVME-SSD 4台を用いて、SSBM(Star Schema Benchmark)ワークロードをGPUDirect SQLをGPUDirect Storageドライバの下で実行したものですが、PostgreSQL heapとApache Arrowのいずれのケースにおいても、単位時間あたりのデータ処理件数はPostgreSQLに比べ大幅に改善している事が分かります。
}
@en{
We can use GPUDirect SQL to scan PostgreSQL's heap table and Apache Arrow files. It can expect significant performance improvement for the workloads where table scans are the major bottleneck, in either driver cases.

The performance measurement below is by SSBM (Star Schema Benchmark) using 1xGPU and 4xNVME-SSDs under the GPUDirect Storage driver. It shows number of rows processed per unit time is significantly improved regardless of the storage system; either PostgreSQL heap or Apache Arrow.
}
![Star Schema Benchmark Results](./img/release_3_0b.png)

@ja{
また、クエリを実行中のNVME-SSDからの読み出しスループットを比較してみると、ファイルシステムを介した読出し（PostgreSQL Heap Storage）に比べ、GPUDirect Storageを使用した場合にはハードウェア限界に近い性能値を引き出せている事が分かります。
}
@en{
In comparison of the read throughput from NVME-SSD drives during the query execution, it shows the table scan by GPUDirect Storage pulls out almost optimal performance close to the hardware limitation, much faster than the scan by filesystem (PostgreSQL Heap Storage).
}

@ja:##GPU版PostGISとGiSTインデックス
@en:##GPU-PostGIS and GiST-index

@ja{
いくつかのPostGIS関数にGPU版を実装しました。
条件句でこれらのPostGIS関数が使用されている場合、PG-StromはGPU側でこれを実行するようGPUプログラムを自動生成します。

GPU版PostGISの主たるターゲットは、携帯電話や自動車といった移動体デバイスの最新の位置情報（Real-time Location Data）と、市区町村や学区の境界といった領域（Area Definition Data）との間で行われる突合処理です。
}
@en{
We have implemented GPU versions of several PostGIS functions. 
When these PostGIS functions are used in qualifier clauses (like, WHERE-clause), PG-Strom will automatically generate a GPU program to execute it on the GPU.

The main target of GPU version of PostGIS is the workload to check the real-time location data of mobile devices, like smartphones or vehicles, against the area definition data like boundary of municipality or school districts.
}

![PostGIS Overview](./img/release_3_0c.png)

@ja{
例えば、一定のエリア内に存在する携帯電話に広告を配信したい時、一定のエリア内に存在する自動車に渋滞情報を配信したい時など、位置をキーとして該当するデバイスを検索する処理に効果を発揮します。

以下の例は、東京近郊エリアを包含する矩形領域内にランダムな1600万個の点データを作成し、市区町村ごとにその領域内に含まれる点の数をカウントするという処理の応答時間を計測したものです。
通常のPostGISとGiSTインデックスの組み合わせでは160秒以上を要した処理が、GPU版PostGISとGiSTインデックスの組み合わせにおいては、僅か0.830秒で応答しています。
}
@en{
For example, when you want to deliver an advertisement to smartphonws in a particular area, or when you want to deliver traffic jam information to cara in a particular area, it is effective in the process of searching for the corresponding device using the position as a key.

In the following example, it creates 16 million random points data in a rectangular area that includes the Tokyo region, then count up number of the points contained in the cities in Tokyo for each.
The vanilla PostGIS and GiST index took more than 160sec, on the other hand, GPU-version of PostGIS and GiST index responded in 0.830 sec.
}

![GPU PostGIS and GiST Index](./img/release_3_0d.png)

@ja:##GPUキャッシュ
@en:##GPU Cache

@ja{
GPUデバイスメモリ上に予め領域を確保しておき、対象となるテーブルの複製を保持しておく機能です。
比較的小規模のデータサイズ（～10GB程度）で、更新頻度の高いデータに対して分析/検索系のクエリを効率よく実行するために設計されました。
分析/検索系クエリの実行時には、GPU上のキャッシュを参照する事で、テーブルからデータを読み出す事なくGPUでSQLワークロードを処理します。

これは典型的には、数百万デバイスのリアルタイムデータをGPU上に保持しておき、タイムスタンプや位置情報の更新が高頻度で発生するといったワークロードです。
}
@en{
GPU Cache mechanism can store a copy of the target table in a pre-allocated area on the GPU device memory.
It was designed for efficient execution of analytical/search queries on frequently updated data with relatively small data size (~10GB).

The GPU can process SQL workloads by referring to GPU Cache instead of loading data from tables when executing analytical/search queries.

This is typically a workload that keeps real-time data from millions of devices on the GPU and frequently updates timestamps and location information.
}

![GPU Cache Overview](./img/release_3_0e.png)

@ja{
GPUキャッシュが設定されたテーブルを更新すると、その更新履歴をオンメモリのREDOログバッファに格納し、それを一定間隔か、または分析/検索系ワークロードの実行前にGPUキャッシュ側に反映します。
この仕組みにより、高頻度での更新と、GPUキャッシュとの整合性とを両立しています。
}
@en{
When the table with GPU cache is updated, the update history is stored in the on-memory redo log buffer, then applied to the GPU cache at a regular intervals or before executing the analysis / search workload.
By this mechanism, it achieved both of frequent updates and consistency of GPU cache.
}

@ja:##ユーザ定義のGPUデータ型/関数
@en:##User-defined GPU datatype/functions

@ja{
ユーザ定義のGPUデータ型/関数を追加するためのAPIを新たに提供します。
これにより、PG-Strom自体には手を加えることなく、ニッチな用途のデータ型やそれを処理するためのSQL関数をユーザが独自に定義、実装する事が可能となりました。

!!! Notice
    本APIは実験的ステータスであり、将来のバージョンで予告なく変更される可能性があります。
    また、本APIの利用にはPG-Strom内部構造を十分に理解している事が前提ですので、ドキュメントは提供されていません。
}

@en{
A new API is provided to add user-defined GPU data types/functions. This allows users to define and implement their own niche data types and SQL functions to process them, without modifying PG-Strom itself.

!!! Notice
    This API is still under the experimental state, so its specifications may be changed without notifications.
    Also note that we assume the users of this API well understand PG-Strom internal, so no documentations are provided right now.
}

@ja:##PostgreSQLライセンスの採用
@en:##PostgreSQL License Adoption

@ja{
PG-Strom v3.0以降ではPostgreSQLライセンスを採用します。

歴史的な経緯により、これまでのPG-StromではGPLv2を採用していましたが、PG-Stromコア機能や周辺ツールと組み合わせたソリューション開発にライセンス体系が障害になるとの声を複数いただいていました。
}

@en{
PG-Strom v3.0 or later adopt the PostgreSQL License.

The earlier version of PG-Strom has used GPLv2 due to the historical background, however, we recognized several concerns that license mismatch prevents joint solution development using PG-Strom core features and comprehensive tools.
}

@ja:##その他の変更
@en:##Other updates

@ja{
- 独自に `int1` (8bit整数) データ型、および関連する演算子に対応しました。
- `pg2arrow`に`--inner-join`および`--outer-join`オプションを追加しました。PostgreSQLの列数制限を越えた数の列を持つApache Arrowファイルを生成できるようになります。
- マルチGPU環境では、GPUごとに専用のGPU Memory Keeperバックグラウンドワーカーが立ち上がるようになりました。
- PostgreSQL v13.x に対応しました。
- CUDA 11.2 および Ampere世代のGPUに対応しました。
- ScaleFlux社のComputational Storage製品CSD2000シリーズでのGPUDirect SQLに対応しました（cuFileドライバのみ）
- 雑多なバグの修正
}
@en{
- Unique int1 (8-bit integer) data type and related operators are now supported.
- `--inner-join` and `--outer-join` options are now available for `pg2arrow`. Apache Arrow files having more columns than the limit of PostgreSQL can now be generated.
- In a multi-GPU environment, the GPU Memory Keeper background worker will now be launched for each GPU.
- PostgreSQL v13.x is now supported.
- CUDA 11.2 and Ampere generation GPUs are now supported.
- GPUDirect SQL now supports ScaleFlux's Computational Storage CSD2000 series (only cuFile driver).
- Miscellaneous bug fixes
}

@ja:##廃止された機能
@en:##Deprecated Features

@ja{
- PostgreSQL v10.x 系列のサポートが廃止されました。
- Pythonスクリプトとのデータ連携機能（PyStrom）が廃止されました。
}
@en{
- Support for PostgreSQL v10.x has been discontinued.
- The feature to link data with Python scripts (PyStrom) has been discontinued.
}
