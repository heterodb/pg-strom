@ja:# リファレンス
@en:# References

@ja:## データ型
@en:## Supported Data Types

@ja{
PG-Stromは以下のデータ型をGPUで利用する事ができます。
}
@en{
PG-Strom support the following data types for use on GPU device.
}

@ja{
**標準の数値データ型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`smallint`        |`cl_short`        |2 bytes |    |
|`integer`         |`cl_int`          |4 bytes |    |
|`bigint`          |`cl_long`         |8 bytes |    |
|`real`            |`cl_float`        |4 bytes |    |
|`float`           |`cl_double`       |8 bytes |    |
|`numeric`         |`cl_ulong`        |可変長  |64bitの内部形式にマップ|
}
@en{
**Built-in numeric types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`smallint`        |`cl_short`        |2 bytes |    |
|`integer`         |`cl_int`          |4 bytes |    |
|`bigint`          |`cl_long`         |8 bytes |    |
|`real`            |`cl_float`        |4 bytes |    |
|`float`           |`cl_double`       |8 bytes |    |
|`numeric`         |`cl_ulong`        |variable length|mapped to 64bit internal format|
}

@ja{
!!! Note
    GPUが`numeric`型のデータを処理する際、実装上の理由からこれを64bitの内部表現に変換して処理します。
    これら内部表現への/からの変換は透過的に行われますが、例えば、桁数の大きな`numeric`型のデータは表現する事ができないため、PG-StromはCPU側でのフォールバック処理を試みます。したがって、桁数の大きな`numeric`型のデータをGPUに与えると却って実行速度が低下してしまう事になります。
    これを避けるには、GUCパラメータ`pg_strom.enable_numeric_type`を使用して`numeric`データ型を含む演算式をGPUで実行しないように設定します。
}
@en{
!!! Note
    When GPU processes values in `numeric` data type, it is converted to an internal 64bit format because of implementation reason.
    It is transparently converted to/from the internal format, on the other hands, PG-Strom cannot convert `numaric` datum with large number of digits, so tries to fallback operations by CPU. Therefore, it may lead slowdown if `numeric` data with large number of digits are supplied to GPU device.
    To avoid the problem, turn off the GUC option `pg_strom.enable_numeric_type` not to run operational expression including `numeric` data types on GPU devices.
}



@ja{
**標準の日付時刻型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`date`            |`DateADT`         |4 bytes |    |
|`time`            |`TimeADT`         |8 bytes |    |
|`timetz`          |`TimeTzADT`       |12 bytes|    |
|`timestamp`       |`Timestamp`       |8 bytes |    |
|`timestamptz`     |`TimestampTz`     |8 bytes |    |
|`interval`        |`Interval`        |16 bytes|time interval|
}

@en{
**Built-in date and time types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`date`            |`DateADT`         |4 bytes |    |
|`time`            |`TimeADT`         |8 bytes |    |
|`timetz`          |`TimeTzADT`       |12 bytes|    |
|`timestamp`       |`Timestamp`       |8 bytes |    |
|`timestamptz`     |`TimestampTz`     |8 bytes |    |
|`interval`        |`Interval`        |16 bytes|    |
}

@ja{
**標準の可変長データ型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`bpchar`          |`varlena *`       |可変長  |    |
|`varchar`         |`varlena *`       |可変長  |    |
|`bytea`           |`varlena *`       |可変長  |    |
|`text`            |`varlena *`       |可変長  |    |
}

@en{
**Built-in variable length types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`bpchar`          |`varlena *`       |variable length|
|`varchar`         |`varlena *`       |variable length|
|`bytea`           |`varlena *`       |variable length|
|`text`            |`varlena *`       |variable length|
}

@ja{
**標準の雑多なデータ型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`boolean`         |`cl_bool`         |1 byte  |    |
|`money`           |`cl_long`         |8 bytes |    |
|`uuid`            |`pg_uuid`         |16 bytes|    |
|`macaddr`         |`macaddr`         |6 bytes |    |
|`inet`            |`inet_struct`     |7 bytes or 19 bytes||
|`cidr`            |`inet_struct`     |7 bytes or 19 bytes||


}
@en{
**Built-in miscellaneous types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`boolean`         |`cl_bool`         |1 byte  |    |
|`money`           |`cl_long`         |8 bytes |    |
|`uuid`            |`pg_uuid`         |16 bytes|    |
|`macaddr`         |`macaddr`         |6 bytes |    |
|`inet`            |`inet_struct`     |7 bytes or 19 bytes||
|`cidr`            |`inet_struct`     |7 bytes or 19 bytes||
}

@ja{
**標準の範囲型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`int4range`       |`__int4range`     |14 bytes|    |
|`int8range`       |`__int8range`     |22 bytes|    |
|`tsrange`         |`__tsrange`       |22 bytes|    |
|`tstzrange`       |`__tstzrange`     |22 bytes|    |
|`daterange`       |`__daterange`     |14 bytes|    |
}
@en{
**Built-in range data types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`int4range`       |`__int4range`     |14 bytes|    |
|`int8range`       |`__int8range`     |22 bytes|    |
|`tsrange`         |`__tsrange`       |22 bytes|    |
|`tstzrange`       |`__tstzrange`     |22 bytes|    |
|`daterange`       |`__daterange`     |14 bytes|    |
}

@ja{
**PG-Stromが追加で提供するデータ型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`float2`          |`half_t`          |2 bytes |半精度浮動小数点数|

}
@en{
**Extra Types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`float2`          |`half_t`          |2 bytes |Half precision data type|
}



@ja:##関数と演算子
@en:## Functions and Operators



@ja:### デバイス情報関数
@en:### Device information functions


@ja:## システムビュー
@en:## System View

@ja{
PG-Stromは内部状態をユーザやアプリケーションに出力するためのシステムビューをいくつか提供しています。
これらのシステムビューは将来のバージョンで情報が追加される可能性があります。そのため、アプリケーションから`SELECT * FROM ...`によってこれらシステムビューを参照する事は避けてください。
}
@en{
PG-Strom provides several system view to export its internal state for users or applications.
The future version may add extra fields here. So, it is not recommended to reference these information schemas using `SELECT * FROM ...`.
}

### pgstrom.device_info
@ja{
`pgstrom.device_info`システムビューは、PG-Stromが認識しているGPUのデバイス属性値を出力します。
GPUはモデルごとにコア数やメモリ容量、最大スレッド数などのスペックが異なっており、PL/CUDA関数などで直接GPUのプログラミングを行う場合には、これらの情報を元にソフトウェアを最適化する必要があります。

|名前        |データ型  |説明|
|:-----------|:---------|:---|
|device_nr   |`int`     |GPUデバイス番号 |
|aindex      |`int`     |属性インデックス|
|attribute   |`text`    |デバイス属性名  |
|value       |`text`    |デバイス属性値  |
}
@en{
`pgstrom.device_into` system view exports device attributes of the GPUs recognized by PG-Strom.
GPU has different specification for each model, like number of cores, capacity of global memory, maximum number of threads and etc, user's software should be optimized according to the information if you try raw GPU programming with PL/CUDA functions.

|Name        |Data Type |Description|
|:-----------|:---------|:----------|
|device_nr   |`int`     |GPU device number |
|aindex      |`int`     |Attribute index |
|attribute   |`text`    |Attribute name |
|value       |`text`    |Value of the attribute |
}

### pgstrom.device_preserved_meminfo
@ja{
`pgstrom.device_preserved_meminfo`システムビューは、複数のPostgreSQLバックエンドプロセスから共有するために予め確保済みのGPUデバイスメモリ領域の情報を出力します。
現在のところ、gstore_fdwのみが本機能を使用しています。

|名前        |データ型  |説明|
|:-----------|:---------|:---|
|device_nr   |`int`     |GPUデバイス番号
|handle      |`bytea`   |確保済みGPUデバイスメモリのIPCハンドラ
|owner       |`regrole` |確保済みGPUデバイスメモリの作成者
|length      |`bigint`  |確保済みGPUデバイスメモリのバイト単位の長さ
|ctime       |`timestamp with time zone`|確保済みGPUデバイスメモリの作成時刻
}
@en{
`pgstrom.device_preserved_meminfo` system view exports information of the preserved device memory; which can be shared multiple PostgreSQL backend.
Right now, only gstore_fdw uses this feature.

|Name        |Data Type |Description|
|:-----------|:---------|:----------|
|device_nr   |`int`     |GPU device number
|handle      |`bytea`   |IPC handle of the preserved device memory
|owner       |`regrole` |Owner of the preserved device memory
|length      |`bigint`  |Length of the preserved device memory in bytes
|ctime       |`timestamp with time zone`|Timestamp when the preserved device memory is created

}

### pgstrom.ccache_info
@ja{
`pgstrom.ccache_info`システムビューは、列指向キャッシュの各チャンク（128MB単位）の情報を出力します。

|名前        |データ型  |説明|
|:-----------|:---------|:---|
|database_id |`oid`     |データベースID |
|table_id    |`regclass`|テーブルID |
|block_nr    |`int`     |チャンクの先頭ブロック番号 |
|nitems      |`bigint`  |チャンクに含まれる行数 |
|length      |`bigint`  |キャッシュされたチャンクのサイズ |
|ctime       |`timestamp with time zone`|チャンク作成時刻|
|atime       |`timestamp with time zone`|チャンク最終アクセス時刻|
}
@en{
`pgstrom.ccache_info` system view exports attribute of the columnar-cache chunks (128MB unit for each).

|Name        |Data Type |Description|
|:-----------|:---------|:---|
|database_id |`oid`     |Database Id |
|table_id    |`regclass`|Table Id |
|block_nr    |`int`     |Head block-number of the chunk |
|nitems      |`bigint`  |Number of rows in the chunk |
|length      |`bigint`  |Raw size of the cached chunk |
|ctime       |`timestamp with time zone`|Timestamp of the chunk creation |
|atime       |`timestamp with time zone`|Timestamp of the least access to the chunk |
}


### pgstrom.ccache_builder_info
@ja{
`pgstrom.ccache_builder_info`システムビューは、列指向キャッシュの非同期ビルダープロセスの情報を出力します。

|名前        |データ型  |説明|
|:-----------|:---------|:---|
|builder_id  |`int`     |列指向キャッシュ非同期ビルダーのID |
|state       |`text`    |ビルダープロセスの状態。（`shutdown`: 停止中、`startup`: 起動中、`loading`: 列指向キャッシュの構築中、`sleep`: 一時停止中）|
|database_id |`oid`     |ビルダープロセスが割り当てられているデータベースID |
|table_id    |`regclass`|`state`が`loading`の場合、読出し中のテーブルID |
|block_nr    |`int`     |`state`が`loading`の場合、読出し中のブロック番号 |
}
@en{
`pgstrom.ccache_builder_info` system view exports information of asynchronous builder process of columnar cache.

|Name        |Data Type  |Description|
|:-----------|:----------|:---|
|builder_id  |`int`      |Asynchronous builder Id of columnar cache |
|state       |`text`     |State of the builder process (`shutdown`, `startup`, `loading` or `sleep`) |
|database_id |`oid`      |Database Id where builder process is assigned on |
|table_id    |`regclass` |Table Id where the builder process is scanning on, if `state` is `loading`. |
|block_nr    |`int`      |Block number where the builder process is scanning on, if `state` is `loading`. |
}

@ja:## GUCパラメータ
@en:## GUC Parameters

@ja{
本節ではPG-Stromの提供する設定パラメータについて説明します。
}
@en{
This session introduces PG-Strom's configuration parameters.
}

@ja:### 特定機能の有効化/無効化
@en:### Enables/disables a particular feature

@ja{
|パラメータ名                   |型    |初期値|説明       |
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.enabled`             |`bool`|`on` |PG-Strom機能全体を一括して有効化/無効化する。|
|`pg_strom.enable_gpuscan`      |`bool`|`on` |GpuScanによるスキャンを有効化/無効化する。|
|`pg_strom.enable_gpuhashjoin`  |`bool`|`on` |HashJoinによるGpuJoinを有効化/無効化する。|
|`pg_strom.enable_gpunestloop`  |`bool`|`on` |NestLoopによるGpuJoinを有効化/無効化する。|
|`pg_strom.enable_gpupreagg`    |`bool`|`on` |GpuPreAggによる集約処理を有効化/無効化する。|
|`pg_strom.pullup_outer_scan`   |`bool`|`on` |GpuPreAgg/GpuJoin直下の実行計画が全件スキャンである場合に、上位ノードでスキャン処理も行い、CPU/RAM⇔GPU間のデータ転送を省略するかどうかを制御する。|
|`pg_strom.pullup_outer_join`   |`bool`|`on` |GpuPreAgg直下がGpuJoinである場合に、JOIN処理を上位の実行計画に引き上げ、CPU⇔GPU間のデータ転送を省略するかどうかを制御する。|
|`pg_strom.enable_numeric_type` |`bool`|`on` |GPUで`numeric`データ型を含む演算式を処理するかどうかを制御する。|
|`pg_strom.cpu_fallback`        |`bool`|`off`|GPUプログラムが"CPU再実行"エラーを返したときに、実際にCPUでの再実行を試みるかどうかを制御する。|
|`pg_strom.nvme_strom_enabled`  |`bool`|`on` |SSD-to-GPUダイレクトSQL実行機能を有効化/無効化する。|
|`pg_strom.nvme_strom_threshold`|`int` |自動 |SSD-to-GPUダイレクトSQL実行機能を発動させるテーブルサイズの閾値を設定する。|
}

@en{
|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.enabled`             |`bool`|`on` |Enables/disables entire PG-Strom features at once|
|`pg_strom.enable_gpuscan`      |`bool`|`on` |Enables/disables GpuScan|
|`pg_strom.enable_gpuhashjoin`  |`bool`|`on` |Enables/disables GpuJoin by HashJoin|
|`pg_strom.enable_gpunestloop`  |`bool`|`on` |Enables/disables GpuJoin by NestLoop|
|`pg_strom.enable_gpupreagg`    |`bool`|`on` |Enables/disables GpuPreAgg|
|`pg_strom.pullup_outer_scan`   |`bool`|`on` |Enables/disables to pull up full-table scan if it is just below GpuPreAgg/GpuJoin, to reduce data transfer between CPU/RAM and GPU.|
|`pg_strom.pullup_outer_join`   |`bool`|`on` |Enables/disables to pull up tables-join if GpuJoin is just below GpuPreAgg, to reduce data transfer between CPU/RAM and GPU.|
|`pg_strom.enable_numeric_type` |`bool`|`on` |Enables/disables support of `numeric` data type in arithmetic expression on GPU device|
|`pg_strom.cpu_fallback`        |`bool`|`off`|Controls whether it actually run CPU fallback operations, if GPU program returned "CPU ReCheck Error"|
|`pg_strom.nvme_strom_enabled`  |`bool`|`on` |Enables/disables the feature of SSD-to-GPU Direct SQL Execution|
|`pg_strom.nvme_strom_threshold`|`int` |自動 |Controls the table-size threshold to invoke the feature of SSD-to-GPU Direct SQL Execution|
}

@ja:### オプティマイザに関する設定
@en:### Optimizer Configuration

@ja{
|パラメータ名                   |型    |初期値|説明       |
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.chunk_size`          |`int` |65534kB|PG-Stromが1回のGPUカーネル呼び出しで処理するデータブロックの大きさです。かつては変更可能でしたが、ほとんど意味がないため、現在では約64MBに固定されています。|
|`pg_strom.gpu_setup_cost`      |`real`|4000  |GPUデバイスの初期化に要するコストとして使用する値。|
|`pg_strom.gpu_dma_cost`        |`real`|10    |チャンク(64MB)あたりのDMA転送に要するコストとして使用する値。|
|`pg_strom.gpu_operator_cost`   |`real`|0.00015|GPUの演算式あたりの処理コストとして使用する値。`cpu_operator_cost`よりも大きな値を設定してしまうと、いかなるサイズのテーブルに対してもPG-Stromが選択されることはなくなる。|
}
@en{
|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.chunk_size`          |`int` |65534kB|Size of the data blocks processed by a single GPU kernel invocation. It was configurable, but makes less sense, so fixed to about 64MB in the current version.|
|`pg_strom.gpu_setup_cost`      |`real`|4000  |Cost value for initialization of GPU device|
|`pg_strom.gpu_dma_cost`        |`real`|10    |Cost value for DMA transfer over PCIe bus per data-chunk (64MB)|
|`pg_strom.gpu_operator_cost`   |`real`|0.00015|Cost value to process an expression formula on GPU. If larger value than `cpu_operator_cost` is configured, no chance to choose PG-Strom towards any size of tables|
}


@ja:### エグゼキュータに関する設定
@en:### Executor Configuration
@ja{
|パラメータ名                       |型    |初期値|説明       |
|:----------------------------------|:----:|:----:|:----------|
|`pg_strom.global_max_async_tasks`  |`int` |160 |PG-StromがGPU実行キューに投入する事ができる非同期タスクのシステム全体での最大値。
|`pg_strom.local_max_async_tasks`   |`int` |8   |PG-StromがGPU実行キューに投入する事ができる非同期タスクのプロセス毎の最大値。CPUパラレル処理と併用する場合、この上限値は個々のバックグラウンドワーカー毎に適用されます。したがって、バッチジョブ全体では`pg_strom.local_max_async_tasks`よりも多くの非同期タスクが実行されることになります。
|`pg_strom.max_number_of_gpucontext`|`int` |自動|GPUデバイスを抽象化した内部データ構造 GpuContext の数を指定します。通常、初期値を変更する必要はありません。
}
@en{
|Parameter                         |Type  |Default|Description|
|:---------------------------------|:----:|:----:|:----------|
|`pg_strom.global_max_async_tasks` |`int` |160   |Number of asynchronous taks PG-Strom can throw into GPU's execution queue in the whole system.|
|`pg_strom.local_max_async_tasks`  |`int` |8     |Number of asynchronous taks PG-Strom can throw into GPU's execution queue per process. If CPU parallel is used in combination, this limitation shall be applied for each background worker. So, more than `pg_strom.local_max_async_tasks` asynchronous tasks are executed in parallel on the entire batch job.|
|`pg_strom.max_number_of_gpucontext`|`int`|auto  |Specifies the number of internal data structure `GpuContext` to abstract GPU device. Usually, no need to expand the initial value.|
}


@ja:### 列指向キャッシュ関連の設定
@en:### Columnar Cache Configuration

@ja{
|パラメータ名                  |型      |初期値    |説明       |
|:-----------------------------|:------:|:---------|:----------|
|`pg_strom.ccache_base_dir`    |`string`|`'/dev/shm'`|列指向キャッシュを保持するファイルシステム上のパスを指定します。通常、`tmpfs`がマウントされている`/dev/shm`を変更する必要はありません。|
|`pg_strom.ccache_databases`   |`string`|`''`        |列指向キャッシュの非同期ビルドを行う対象データベースをカンマ区切りで指定します。`pgstrom_ccache_prewarm()`によるマニュアルでのキャッシュビルドには影響しません。|
|`pg_strom.ccache_num_builders`|`int`   |`2`       |列指向キャッシュの非同期ビルドを行うワーカープロセス数を指定します。少なくとも`pg_strom.ccache_databases`で設定するデータベースの数以上にワーカーが必要です。|
|`pg_strom.ccache_log_output`  |`bool`  |`false`   |列指向キャッシュの非同期ビルダーがログメッセージを出力するかどうかを制御します。|
|`pg_strom.ccache_total_size`  |`int`   |自動      |列指向キャッシュの上限を kB 単位で指定します。区画サイズの75%またはシステムの物理メモリの66%のいずれか小さな方がデフォルト値です。|
}
@en{
|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.ccache_base_dir`    |`string`|`'/dev/shm'`|Specifies the directory path to store columnar cache data files. Usually, no need to change from `/dev/shm` where `tmpfs` is mounted at.|
|`pg_strom.ccache_databases`   |`string`|`''`    |Specified the target databases for asynchronous columnar cache build, in comma separated list. It does not affect to the manual cache build by `pgstrom_ccache_prewarm()`.|
|`pg_strom.ccache_num_builders`|`int`   |`2`     |Specified the number of worker processes for asynchronous columnar cache build. It needs to be larger than or equeal to the number of databases in `pg_strom.ccache_databases`.|
|`pg_strom.ccache_log_output`  |`bool`  |`false` |Controls whether columnar cache builder prints log messages, or not|
|`pg_strom.ccache_total_size`  |`int`   |auto    |Upper limit of the columnar cache in kB. Default is the smaller in 75% of volume size or 66% of system physical memory.|
}

@ja:### gstore_fdw関連の設定
@en:### gstore_fdw Configuration

@ja{
|パラメータ名                   |型      |初期値    |説明       |
|:------------------------------|:------:|:---------|:----------|
|`pg_strom.gstore_max_relations`|`int`   |100       |gstore_fdwを用いた外部表数の上限です。パラメータの更新には再起動が必要です。|
}
@en{
|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.gstore_max_relations`|`int`   |100       |Upper limit of the number of foreign tables with gstore_fdw. It needs restart to update the parameter.|
}

@ja:### GPUプログラムの生成とビルドに関連する設定
@en:### Configuration of GPU code generation and build
@ja{
|パラメータ名                   |型      |初期値  |説明       |
|:------------------------------|:------:|:-------|:----------|
|`pg_strom.program_cache_size`  |`int`   |`256MB` |ビルド済みのGPUプログラムをキャッシュしておくための共有メモリ領域のサイズです。パラメータの更新には再起動が必要です。|
|`pg_strom.debug_jit_compile_options`|`bool`|`off`|GPUプログラムのJITコンパイル時に、デバッグオプション（行番号とシンボル情報）を含めるかどうかを指定します。GPUコアダンプ等を用いた複雑なバグの解析に有用ですが、性能のデグレードを引き起こすため、通常は使用すべきでありません。|
|`pg_strom.debug_kernel_source` |`bool`  |`off`    |このオプションが`on`の場合、`EXPLAIN VERBOSE`コマンドで自動生成されたGPUプログラムを書き出したファイルパスを出力します。|
}
@en{
|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.program_cache_size`  |`int`   |`256MB` |Amount of the shared memory size to cache GPU programs already built. It needs restart to update the parameter.|
|`pg_strom.debug_jit_compile_options`|`bool`|`off`|Controls to include debug option (line-numbers and symbol information) on JIT compile of GPU programs. It is valuable for complicated bug analysis using GPU core dump, however, should not be enabled on daily use because of performance degradation.|
|`pg_strom.debug_kernel_source` |`bool`  |`off`   |If enables, `EXPLAIN VERBOSE` command also prints out file paths of GPU programs written out.|
}


@ja:### GPUデバイスに関連する設定
@en:### GPU Device Configuration
@ja{
|パラメータ名                   |型      |初期値 |説明       |
|:------------------------------|:------:|:------|:----------|
|`pg_strom.cuda_visible_devices`|`string`|`''`   |PostgreSQLの起動時に特定のGPUデバイスだけを認識させてい場合は、カンマ区切りでGPUデバイス番号を記述します。これは環境変数`CUDA_VISIBLE_DEVICES`を設定するのと同等です。|
|`pg_strom.gpu_memory_segment_size`|`int`|`512MB`|PG-StromがGPUメモリをアロケーションする際に、1回のCUDA API呼び出しで獲得するGPUデバイスメモリのサイズを指定します。この値が大きいとAPI呼び出しのオーバーヘッドは減らせますが、デバイスメモリのロスは大きくなります。
|`pg_strom.max_num_preserved_gpu_memory`|`int`|2048|確保済みGPUデバイスメモリのセグメント数の上限を指定します。通常は初期値を変更する必要はありません。|
}
@en{
|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.cuda_visible_devices`|`string`|`''`   |List of GPU device numbers in comma separated, if you want to recognize particular GPUs on PostgreSQL startup. It is equivalent to the environment variable `CUDAVISIBLE_DEVICES`|
|`pg_strom.gpu_memory_segment_size`|`int`|`512MB`|Specifies the amount of device memory to be allocated per CUDA API call. Larger configuration will reduce the overhead of API calls, but not efficient usage of device memory.|
|`pg_strom.max_num_preserved_gpu_memory`|`int`|2048|Upper limit of the number of preserved GPU device memory segment. Usually, don't need to change from the default value.|
}





