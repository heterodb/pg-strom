@ja{
#GUCパラメータ

本節ではPG-Stromの提供する設定パラメータについて説明します。
}
@en{
#GUC Parameters

This session introduces PG-Strom's configuration parameters.
}

@ja{
## 特定機能の有効化/無効化

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
## Enables/disables a particular feature

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
## オプティマイザに関する設定

|パラメータ名                   |型    |初期値|説明       |
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.chunk_size`          |`int` |65534kB|PG-Stromが1回のGPUカーネル呼び出しで処理するデータブロックの大きさです。かつては変更可能でしたが、ほとんど意味がないため、現在では約64MBに固定されています。|
|`pg_strom.gpu_setup_cost`      |`real`|4000  |GPUデバイスの初期化に要するコストとして使用する値。|
|`pg_strom.gpu_dma_cost`        |`real`|10    |チャンク(64MB)あたりのDMA転送に要するコストとして使用する値。|
|`pg_strom.gpu_operator_cost`   |`real`|0.00015|GPUの演算式あたりの処理コストとして使用する値。`cpu_operator_cost`よりも大きな値を設定してしまうと、いかなるサイズのテーブルに対してもPG-Stromが選択されることはなくなる。|
}
@en{
## Optimizer Configuration

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.chunk_size`          |`int` |65534kB|Size of the data blocks processed by a single GPU kernel invocation. It was configurable, but makes less sense, so fixed to about 64MB in the current version.|
|`pg_strom.gpu_setup_cost`      |`real`|4000  |Cost value for initialization of GPU device|
|`pg_strom.gpu_dma_cost`        |`real`|10    |Cost value for DMA transfer over PCIe bus per data-chunk (64MB)|
|`pg_strom.gpu_operator_cost`   |`real`|0.00015|Cost value to process an expression formula on GPU. If larger value than `cpu_operator_cost` is configured, no chance to choose PG-Strom towards any size of tables|
}

@ja{
## エグゼキュータに関する設定

|パラメータ名                 |型    |初期値|説明       |
|:----------------------------|:----:|:----:|:----------|
|`pg_strom.max_async_tasks`   |`int` |`5`   |PG-StromがGPU実行キューに投入する事ができる非同期タスクのプロセス毎の最大値。CPUパラレル処理と併用する場合、この上限値は個々のバックグラウンドワーカー毎に適用されます。したがって、バッチジョブ全体では`pg_strom.max_async_tasks`よりも多くの非同期タスクが実行されることになります。|
|`pg_strom.reuse_cuda_context`|`bool`|`off` |クエリの実行に伴って作成したCUDAコンテキストを、次回のクエリ実行時に再利用します。通常、CUDAコンテキストの作成には100～200ms程度を要するため、応答速度の改善が期待できる一方、一部のGPUデバイスメモリを占有し続けるというデメリットもあります。そのため、ベンチマーク等の用途を除いては使用すべきではありません。<br>また、CPUパラレルを利用する場合には効果はありません。|
}
@en{
##Executor Configuration

|Parameter                    |Type  |Default|Description|
|:----------------------------|:----:|:-----:|:----------|
|`pg_strom.max_async_tasks`   |`int` |`5`    |Number of asynchronous taks PG-Strom can throw into GPU's execution queue per process. If CPU parallel is used in combination, this limitation shall be applied for each background worker. So, more than `pg_strom.max_async_tasks` asynchronous tasks are executed in parallel on the entire batch job.|
|`pg_strom.reuse_cuda_context`|`bool`|`off`  |If `on`, it tries to reuse CUDA context, constructed according to the previous query execution, on the next query execution. Usually, construction of CUDA context takes 100-200ms, it may improve queries response time, on the other hands, it continue to occupy a part of GPU device memory on the down-side. So, we don't recommend to enable this parameter expect for benchmarking and so on.<br>Also, this configuration makes no sense if query uses CPU parallel execution.|
}

@ja{
##GPUダイレクトSQL関連の設定
|パラメータ名                   |型      |初期値|説明       |
|:------------------------------|:------:|:-----|:----------|
|`pg_strom.gpudirect_driver`    |`text`  |自動  |GPUダイレクトSQLのドライバソフトウェア名を示す読み取り専用パラメータです。|
|`pg_strom.gpudirect_enabled`   |`bool`  |`on`  |GPUダイレクトSQL機能を有効化/無効化する。|
|`pg_strom.gpudirect_threshold` |`int`   |自動  |GPUダイレクトSQL機能を発動させるテーブルサイズの閾値を設定する。|
|`pg_strom.cufile_io_unitsz`    |`int`   |`16MB`|cuFile APIを使用してデータを読み出す際のI/Oサイズを指定する。通常は変更の必要はありません。|
|`pg_strom.nvme_distance_map`   |`string`|`NULL`|NVME-SSDに最も近いGPUを手動で設定します。書式は`<nvmeX>:<gpuX>[,...]`で、NVME-SSDとGPUのペアをカンマ区切りの文字列で記述します。（例：`nvme0:gpu0,nvme1:gpu0`）<br>ローカルNVME-SSDに対しては多くの場合自動設定で十分ですが、NVME-oFデバイスを使用する場合は手動で近傍のGPUを指定する必要があります。|
}
@en{
##GPUDirect SQL Configuration
|Parameter                     |Type    |Default|Description|
|:-----------------------------|:------:|:-----:|:----------|
|`pg_strom.gpudirect_driver`   |`text`  |auto   |It shows the driver software name of GPUDirect SQL (read-only).|
|`pg_strom.gpudirect_enabled`  |`bool`  |`on`   |Enables/disables GPUDirect SQL feature.|
|`pg_strom.gpudirect_threshold`|`int`   |auto   |Controls the table-size threshold to invoke GPUDirect SQL feature.|
|`pg_strom.cufile_io_unitsz`   |`int`   |`16MB` |Unit size of read-i/o when PG-Strom uses cuFile API. No need to change from the default setting for most cases. It is available only if PG-Strom was built with `WITH_CUFILE=1`.|
|`pg_strom.nvme_distance_map`  |`string`|`NULL` |It manually configures the closest GPU for particular NVME devices. Its format string is `<nvmeX>:<gpuX>[,...]`; comma separated list of NVME-GPU pairs. (Examle: `nvme0:gpu0,nvme1:gpu0`)<br>Automatic configuration is often sufficient for local NVME-SSD drives, on the other hands, you need to configure the closest GPU manually, if NVME-oF devices are in use.|
}

@ja{
##Arrow_Fdw関連の設定
|パラメータ名                    |型      |初期値    |説明       |
|:-------------------------------|:------:|:---------|:----------|
|`arrow_fdw.enabled`             |`bool`  |`on`      |推定コスト値を調整し、Arrow_Fdwの有効/無効を切り替えます。ただし、GpuScanが利用できない場合には、Arrow_FdwによるForeign ScanだけがArrowファイルをスキャンできるという事に留意してください。|
|`arrow_fdw.metadata_cache_size` |`int`   |128MB     |Arrowファイルのメタ情報をキャッシュする共有メモリ領域の大きさを指定します。共有メモリの消費量がこのサイズを越えると、古いメタ情報から順に解放されます。|
|`arrow_fdw.record_batch_size`   |`int`   |256MB     |Arrow_Fdw外部テーブルへ書き込む際の RecordBatch の大きさの閾値です。`INSERT`コマンドが完了していなくとも、Arrow_Fdwは総書き込みサイズがこの値を越えるとバッファの内容をApache Arrowファイルへと書き出します。|
}
@en{
##Arrow_Fdw Configuration
|Parameter                       |Type  |Default|Description|
|:-------------------------------|:----:|:-----:|:----------|
|`arrow_fdw.enabled`             |`bool`|`on`   |By adjustment of estimated cost value, it turns on/off Arrow_Fdw. Note that only Foreign Scan (Arrow_Fdw) can scan on Arrow files, if GpuScan is not capable to run on.|
|`arrow_fdw.metadata_cache_size` |`int` |128MB  |Size of shared memory to cache metadata of Arrow files.<br>Once consumption of the shared memory exceeds this value, the older metadata shall be released based on LRU.|
|`arrow_fdw.record_batch_size`   |`int` |256MB  |Threshold of RecordBatch when Arrow_Fdw foreign table is written. When total amount of the buffer size exceeds this configuration, Arrow_Fdw writes out the buffer to Apache Arrow file, even if `INSERT` command is not completed yet.
}

@ja{
##Gstore_Fdw関連の設定
|パラメータ名                 |型    |初期値|説明       |
|:----------------------------|:----:|:-----|:----------|
|`gstore_fdw.enabled`         |`bool`|`on`  |推定コスト値を調整し、Gstore_Fdwの有効/無効を切り替えます。ただし、GpuScanが利用できない場合には、Gstore_FdwによるForeign ScanだけがGPUメモリストアをスキャンできるという事に留意してください。|
|`gstore_fdw.auto_preload`    |`bool`|`on`  |PostgreSQLの起動後、Base fileに保存されたGPUメモリストアの内容をGPUに自動的にロードするかどうかを制御します。自動ロードを行わない場合、最初にGPUメモリストアを参照したタイミングでGPUへのロードが行われ、結果としてクエリ応答速度が低下する可能性があります。|
|`gstore_fdw.default_base_dir`|`text`|`NULL`|`base_file`が明示的に指定されなかった場合に、Baseファイルを自動生成するディレクトリを指定します。指定のない場合、現在のデータベースのデフォルトテーブルスペース上に作成されます。|
|`gstore_fdw.default_redo_dir`|`text`|`NULL`|`redo_log_file`が明示的に指定されなかった場合に、RedoLogファイルを自動生成するディレクトリを指定します。指定のない場合、現在のデータベースのデフォルトテーブルスペース上に作成されます。|
}
@en{
##Gstore_Fdw Configuration
|Parameter                    |Type  |Default|Description|
|:----------------------------|:----:|:-----:|:----------|
|`gstore_fdw.enabled`         |`bool`|`on`   |By adjustment of estimated cost value, it turns on/off Gstore_Fdw. Note that only Foreign Scan (Gstore_Fdw) can scan on GPU memory store, if GpuScan is not capable to run on.|
|`gstore_fdw.auto_preload`    |`bool`|`on`   |Controls whether the GPU memory store shall be pre-loaded to GPU devices next to the PostgreSQL startup. If not pre-loaded, GPU memory store shall be loaded on the demand by someone's reference. It may lead slow-down of query response time on the first call.|
|`gstore_fdw.default_base_dir`|`text`|`NULL` |Directory to create base files, if `base_file` was not specified in the foreign-table options. In the default, it shall be created on the default tablespace of the current database.|
|`gstore_fdw.default_redo_dir`|`text`|`NULL` |Directory to create redo-log files, if `redo_log_file` was not specified in the foreign-table options. In the default, it shall be created on the default tablespace of the current database.|
}

@ja{
##GPUプログラムの生成とビルドに関連する設定

|パラメータ名                   |型      |初期値  |説明       |
|:------------------------------|:------:|:-------|:----------|
|`pg_strom.program_cache_size`  |`int`   |`256MB` |ビルド済みのGPUプログラムをキャッシュしておくための共有メモリ領域のサイズです。パラメータの更新には再起動が必要です。|
|`pg_strom.num_program_builders`|`int`|`2`|GPUプログラムを非同期ビルドするためのバックグラウンドプロセスの数を指定します。パラメータの更新には再起動が必要です。|
|`pg_strom.debug_jit_compile_options`|`bool`|`off`|GPUプログラムのJITコンパイル時に、デバッグオプション（行番号とシンボル情報）を含めるかどうかを指定します。GPUコアダンプ等を用いた複雑なバグの解析に有用ですが、性能のデグレードを引き起こすため、通常は使用すべきでありません。|
|`pg_strom.extra_kernel_stack_size`|`int`|`0`|GPUカーネルの実行時にスレッド毎に割り当てるスタックの大きさをバイト単位で指定します。通常は初期値を変更するの必要はありません。|
}
@en{
##Configuration of GPU code generation and build

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.program_cache_size`  |`int` |`256MB` |Amount of the shared memory size to cache GPU programs already built. It needs restart to update the parameter.|
|`pg_strom.num_program_builders`|`int`|`2`|Number of background workers to build GPU programs asynchronously. It needs restart to update the parameter.|
|`pg_strom.debug_jit_compile_options`|`bool`|`off`|Controls to include debug option (line-numbers and symbol information) on JIT compile of GPU programs. It is valuable for complicated bug analysis using GPU core dump, however, should not be enabled on daily use because of performance degradation.|
|`pg_strom.extra_kernel_stack_size`|`int`|`0`|Extra size of stack, in bytes, for each GPU kernel thread to be allocated on execution. Usually, no need to change from the default value.|
}

@ja{
##GPUデバイスに関連する設定

|パラメータ名                   |型      |初期値 |説明       |
|:------------------------------|:------:|:------|:----------|
|`pg_strom.cuda_visible_devices`|`string`|`''`   |PostgreSQLの起動時に特定のGPUデバイスだけを認識させてい場合は、カンマ区切りでGPUデバイス番号を記述します。これは環境変数`CUDA_VISIBLE_DEVICES`を設定するのと同等です。|
|`pg_strom.gpu_memory_segment_size`|`int`|`512MB`|PG-StromがGPUメモリをアロケーションする際に、1回のCUDA API呼び出しで獲得するGPUデバイスメモリのサイズを指定します。この値が大きいとAPI呼び出しのオーバーヘッドは減らせますが、デバイスメモリのロスは大きくなります。
}
@en{
##GPU Device Configuration

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:-----:|:----------|
|`pg_strom.cuda_visible_devices`|`string`|`''`   |List of GPU device numbers in comma separated, if you want to recognize particular GPUs on PostgreSQL startup. It is equivalent to the environment variable `CUDAVISIBLE_DEVICES`|
|`pg_strom.gpu_memory_segment_size`|`int`|`512MB`|Specifies the amount of device memory to be allocated per CUDA API call. Larger configuration will reduce the overhead of API calls, but not efficient usage of device memory.|
}

@ja{
##システム共有メモリに関連する設定
|パラメータ名                   |型    |初期値 |説明       |
|:------------------------------|:----:|:------|:----------|
|shmbuf.segment_size            |`int` |`256MB`|           |
|shmbuf.num_logical_segments    |`int` |自動   |デフォルトの論理セグメントサイズはシステム搭載物理メモリの2倍の大きさです。|

}
@en{
##System Shared Memory Configuration
|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:-----:|:----------|
|shmbuf.segment_size            |`int` |`256MB`|
|shmbuf.num_logical_segments    |`int` |auto   |Default logical segment size is double size of system physical memory size.|
}


