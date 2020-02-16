@ja{
<h1>GUCパラメータ</h1>

本節ではPG-Stromの提供する設定パラメータについて説明します。
}
@en{
<h1>GUC Parameters</h1>

This session introduces PG-Strom's configuration parameters.
}

@ja{
# 特定機能の有効化/無効化

|パラメータ名                   |型    |初期値|説明       |
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.enabled`             |`bool`|`on` |PG-Strom機能全体を一括して有効化/無効化する。|
|`pg_strom.enable_gpuscan`      |`bool`|`on` |GpuScanによるスキャンを有効化/無効化する。|
|`pg_strom.enable_gpuhashjoin`  |`bool`|`on` |HashJoinによるGpuJoinを有効化/無効化する。|
|`pg_strom.enable_gpunestloop`  |`bool`|`on` |NestLoopによるGpuJoinを有効化/無効化する。|
|`pg_strom.enable_gpupreagg`    |`bool`|`on` |GpuPreAggによる集約処理を有効化/無効化する。|
|`pg_strom.enable_brin`         |`bool`|`on` |BRINインデックスを使ったテーブルスキャンを有効化/無効化する。|
|`pg_strom.enable_partitionwise_gpujoin`|`bool`|`on`|GpuJoinを各パーティションの要素へプッシュダウンするかどうかを制御する。PostgreSQL v10以降でのみ対応。|
|`pg_strom.enable_partitionwise_gpupreagg`|`bool`|`on`|GpuPreAggを各パーティションの要素へプッシュダウンするかどうかを制御する。PostgreSQL v10以降でのみ対応。|
|`pg_strom.pullup_outer_scan`   |`bool`|`on` |GpuPreAgg/GpuJoin直下の実行計画が全件スキャンである場合に、上位ノードでスキャン処理も行い、CPU/RAM⇔GPU間のデータ転送を省略するかどうかを制御する。|
|`pg_strom.pullup_outer_join`   |`bool`|`on` |GpuPreAgg直下がGpuJoinである場合に、JOIN処理を上位の実行計画に引き上げ、CPU⇔GPU間のデータ転送を省略するかどうかを制御する。|
|`pg_strom.enable_numeric_aggfuncs` |`bool`|`on` |`numeric`データ型を引数に取る集約演算をGPUで処理するかどうかを制御する。|
|`pg_strom.cpu_fallback`        |`bool`|`off`|GPUプログラムが"CPU再実行"エラーを返したときに、実際にCPUでの再実行を試みるかどうかを制御する。|
|`pg_strom.regression_test_mode`|`bool`|`off`|GPUモデル名など、実行環境に依存して表示が変わる可能性のある`EXPLAIN`コマンドの出力を抑制します。これはリグレッションテストにおける偽陽性を防ぐための設定で、通常は利用者が操作する必要はありません。|
}

@en{
# Enables/disables a particular feature

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.enabled`             |`bool`|`on` |Enables/disables entire PG-Strom features at once|
|`pg_strom.enable_gpuscan`      |`bool`|`on` |Enables/disables GpuScan|
|`pg_strom.enable_gpuhashjoin`  |`bool`|`on` |Enables/disables GpuJoin by HashJoin|
|`pg_strom.enable_gpunestloop`  |`bool`|`on` |Enables/disables GpuJoin by NestLoop|
|`pg_strom.enable_gpupreagg`    |`bool`|`on` |Enables/disables GpuPreAgg|
|`pg_strom.enable_brin`         |`bool`|`on` |Enables/disables BRIN index support on tables scan|
|`pg_strom.enable_partitionwise_gpujoin`|`bool`|`on`|Enables/disables whether GpuJoin is pushed down to the partition children. Available only PostgreSQL v10 or later.|
|`pg_strom.enable_partitionwise_gpupreagg`|`bool`|`on`|Enables/disables whether GpuPreAgg is pushed down to the partition children. Available only PostgreSQL v10 or later.|
|`pg_strom.pullup_outer_scan`   |`bool`|`on` |Enables/disables to pull up full-table scan if it is just below GpuPreAgg/GpuJoin, to reduce data transfer between CPU/RAM and GPU.|
|`pg_strom.pullup_outer_join`   |`bool`|`on` |Enables/disables to pull up tables-join if GpuJoin is just below GpuPreAgg, to reduce data transfer between CPU/RAM and GPU.|
|`pg_strom.enable_numeric_aggfuncs` |`bool`|`on` |Enables/disables support of aggregate function that takes `numeric` data type.|
|`pg_strom.cpu_fallback`        |`bool`|`off`|Controls whether it actually run CPU fallback operations, if GPU program returned "CPU ReCheck Error"|
|`pg_strom.regression_test_mode`|`bool`|`off`|It disables some `EXPLAIN` command output that depends on software execution platform, like GPU model name. It avoid "false-positive" on the regression test, so use usually don't tough this configuration.|
}

@ja{
#オプティマイザに関する設定

|パラメータ名                   |型    |初期値|説明       |
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.chunk_size`          |`int` |65534kB|PG-Stromが1回のGPUカーネル呼び出しで処理するデータブロックの大きさです。かつては変更可能でしたが、ほとんど意味がないため、現在では約64MBに固定されています。|
|`pg_strom.gpu_setup_cost`      |`real`|4000  |GPUデバイスの初期化に要するコストとして使用する値。|
|`pg_strom.gpu_dma_cost`        |`real`|10    |チャンク(64MB)あたりのDMA転送に要するコストとして使用する値。|
|`pg_strom.gpu_operator_cost`   |`real`|0.00015|GPUの演算式あたりの処理コストとして使用する値。`cpu_operator_cost`よりも大きな値を設定してしまうと、いかなるサイズのテーブルに対してもPG-Stromが選択されることはなくなる。|
}
@en{
#Optimizer Configuration

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.chunk_size`          |`int` |65534kB|Size of the data blocks processed by a single GPU kernel invocation. It was configurable, but makes less sense, so fixed to about 64MB in the current version.|
|`pg_strom.gpu_setup_cost`      |`real`|4000  |Cost value for initialization of GPU device|
|`pg_strom.gpu_dma_cost`        |`real`|10    |Cost value for DMA transfer over PCIe bus per data-chunk (64MB)|
|`pg_strom.gpu_operator_cost`   |`real`|0.00015|Cost value to process an expression formula on GPU. If larger value than `cpu_operator_cost` is configured, no chance to choose PG-Strom towards any size of tables|
}

@ja{
#エグゼキュータに関する設定

|パラメータ名                       |型    |初期値|説明       |
|:----------------------------------|:----:|:----:|:----------|
|`pg_strom.global_max_async_tasks`  |`int` |160 |PG-StromがGPU実行キューに投入する事ができる非同期タスクのシステム全体での最大値。
|`pg_strom.local_max_async_tasks`   |`int` |8   |PG-StromがGPU実行キューに投入する事ができる非同期タスクのプロセス毎の最大値。CPUパラレル処理と併用する場合、この上限値は個々のバックグラウンドワーカー毎に適用されます。したがって、バッチジョブ全体では`pg_strom.local_max_async_tasks`よりも多くの非同期タスクが実行されることになります。
|`pg_strom.max_number_of_gpucontext`|`int` |自動|GPUデバイスを抽象化した内部データ構造 GpuContext の数を指定します。通常、初期値を変更する必要はありません。
}
@en{
#Executor Configuration

|Parameter                         |Type  |Default|Description|
|:---------------------------------|:----:|:-----:|:----------|
|`pg_strom.global_max_async_tasks` |`int` |160   |Number of asynchronous taks PG-Strom can throw into GPU's execution queue in the whole system.|
|`pg_strom.local_max_async_tasks`  |`int` |8     |Number of asynchronous taks PG-Strom can throw into GPU's execution queue per process. If CPU parallel is used in combination, this limitation shall be applied for each background worker. So, more than `pg_strom.local_max_async_tasks` asynchronous tasks are executed in parallel on the entire batch job.|
|`pg_strom.max_number_of_gpucontext`|`int`|auto  |Specifies the number of internal data structure `GpuContext` to abstract GPU device. Usually, no need to expand the initial value.|
}

@ja{
# SSD-to-GPUダイレクト関連の設定
|パラメータ名                   |型      |初期値|説明       |
|:------------------------------|:------:|:-----|:----------|
|`pg_strom.nvme_strom_enabled`  |`bool`  |`on`  |SSD-to-GPUダイレクトSQL機能を有効化/無効化する。|
|`pg_strom.nvme_strom_threshold`|`int`   |自動  |SSD-to-GPUダイレクトSQL機能を発動させるテーブルサイズの閾値を設定する。|
|`pg_strom.nvme_distance_map`   |`string`|`NULL`|NVME-SSDに近いGPUを手動で設定します。通常はsysfsから取得したPCIeバストポロジ情報による自動設定で問題ありません。|
}
@en{
# SSD-to-GPU Direct Configuration
|Parameter                      |Type    |Default|Description|
|:------------------------------|:------:|:-----:|:----------|
|`pg_strom.nvme_strom_enabled`  |`bool`  |`on`   |Enables/disables SSD-to-GPU Direct SQL mechanism|
|`pg_strom.nvme_strom_threshold`|`int`   |auto   |Controls the table-size threshold to invoke SSD-to-GPU Direct SQL mechanism|
|`pg_strom.nvme_distance_map`   |`string`|`NULL` |Manually configures the closest GPU for each NVME-SSD. Usually, it is configured automatically according to the PCIe bus topology information by sysfs.|
}

@ja{
#Arrow_Fdw関連の設定
|パラメータ名                    |型      |初期値    |説明       |
|:-------------------------------|:------:|:---------|:----------|
|`arrow_fdw.enabled`             |`bool`  |`on`      |推定コスト値を調整し、Arrow_Fdwの有効/無効を切り替えます。ただし、GpuScanが利用できない場合には、Arrow_FdwによるForeign ScanだけがArrowファイルをスキャンできるという事に留意してください。|
|`arrow_fdw.metadata_cache_size` |`int`   |128MB     |Arrowファイルのメタ情報をキャッシュする共有メモリ領域のサイズを指定します。<br>パラメータの更新には再起動が必要です。|
|`arrow_fdw.record_batch_size`   |`int`   |256MB     |Arrow_Fdw外部テーブルへ書き込む際の RecordBatch の大きさの閾値です。`INSERT`コマンドが完了していなくとも、Arrow_Fdwは総書き込みサイズがこの値を越えるとバッファの内容をApache Arrowファイルへと書き出します。|
}
@en{
#Arrow_Fdw Configuration
|Parameter                       |Type  |Default|Description|
|:-------------------------------|:----:|:-----:|:----------|
|`arrow_fdw.enabled`             |`bool`|`on`   |By adjustment of estimated cost value, it turns on/off Arrow_Fdw. Note that only Foreign Scan (Arrow_Fdw) can scan on Arrow files, if GpuScan is not capable to run on.|
|`arrow_fdw.metadata_cache_size` |`int` |128MB  |Size of shared memory to cache metadata of Arrow files.<br>It needs to restart to update the parameter.|
|`arrow_fdw.record_batch_size`   |`int` |256MB  |Threshold of RecordBatch when Arrow_Fdw foreign table is written. When total amount of the buffer size exceeds this configuration, Arrow_Fdw writes out the buffer to Apache Arrow file, even if `INSERT` command is not completed yet.
}

@ja{
#GPUプログラムの生成とビルドに関連する設定

|パラメータ名                   |型      |初期値  |説明       |
|:------------------------------|:------:|:-------|:----------|
|`pg_strom.program_cache_size`  |`int`   |`256MB` |ビルド済みのGPUプログラムをキャッシュしておくための共有メモリ領域のサイズです。パラメータの更新には再起動が必要です。|
|`pg_strom.num_program_builders`|`int`|`2`|GPUプログラムを非同期ビルドするためのバックグラウンドプロセスの数を指定します。パラメータの更新には再起動が必要です。|
|`pg_strom.debug_jit_compile_options`|`bool`|`off`|GPUプログラムのJITコンパイル時に、デバッグオプション（行番号とシンボル情報）を含めるかどうかを指定します。GPUコアダンプ等を用いた複雑なバグの解析に有用ですが、性能のデグレードを引き起こすため、通常は使用すべきでありません。|
|`pg_strom.debug_kernel_source` |`bool`  |`off`    |このオプションが`on`の場合、`EXPLAIN VERBOSE`コマンドで自動生成されたGPUプログラムを書き出したファイルパスを出力します。|
}
@en{
#Configuration of GPU code generation and build

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.program_cache_size`  |`int` |`256MB` |Amount of the shared memory size to cache GPU programs already built. It needs restart to update the parameter.|
|`pg_strom.num_program_builders`|`int`|`2`|Number of background workers to build GPU programs asynchronously. It needs restart to update the parameter.|
|`pg_strom.debug_jit_compile_options`|`bool`|`off`|Controls to include debug option (line-numbers and symbol information) on JIT compile of GPU programs. It is valuable for complicated bug analysis using GPU core dump, however, should not be enabled on daily use because of performance degradation.|
|`pg_strom.debug_kernel_source` |`bool`  |`off`   |If enables, `EXPLAIN VERBOSE` command also prints out file paths of GPU programs written out.|
}

@ja{
#GPUデバイスに関連する設定

|パラメータ名                   |型      |初期値 |説明       |
|:------------------------------|:------:|:------|:----------|
|`pg_strom.cuda_visible_devices`|`string`|`''`   |PostgreSQLの起動時に特定のGPUデバイスだけを認識させてい場合は、カンマ区切りでGPUデバイス番号を記述します。これは環境変数`CUDA_VISIBLE_DEVICES`を設定するのと同等です。|
|`pg_strom.gpu_memory_segment_size`|`int`|`512MB`|PG-StromがGPUメモリをアロケーションする際に、1回のCUDA API呼び出しで獲得するGPUデバイスメモリのサイズを指定します。この値が大きいとAPI呼び出しのオーバーヘッドは減らせますが、デバイスメモリのロスは大きくなります。
|`pg_strom.max_num_preserved_gpu_memory`|`int`|2048|確保済みGPUデバイスメモリのセグメント数の上限を指定します。通常は初期値を変更する必要はありません。|
}
@en{
#GPU Device Configuration

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:-----:|:----------|
|`pg_strom.cuda_visible_devices`|`string`|`''`   |List of GPU device numbers in comma separated, if you want to recognize particular GPUs on PostgreSQL startup. It is equivalent to the environment variable `CUDAVISIBLE_DEVICES`|
|`pg_strom.gpu_memory_segment_size`|`int`|`512MB`|Specifies the amount of device memory to be allocated per CUDA API call. Larger configuration will reduce the overhead of API calls, but not efficient usage of device memory.|
|`pg_strom.max_num_preserved_gpu_memory`|`int`|2048|Upper limit of the number of preserved GPU device memory segment. Usually, don't need to change from the default value.|
}

@ja{
#システム共有メモリに関連する設定
|パラメータ名                   |型    |初期値 |説明       |
|:------------------------------|:----:|:------|:----------|
|shmbuf.segment_size            |`int` |`256MB`|           |
|shmbuf.num_logical_segments    |`int` |自動   |デフォルトの論理セグメントサイズはシステム搭載物理メモリの2倍の大きさです。|

}
@en{
#System Shared Memory Configuration
|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:-----:|:----------|
|shmbuf.segment_size            |`int` |`256MB`|
|shmbuf.num_logical_segments    |`int` |auto   |Default logical segment size is double size of system physical memory size.|
}


