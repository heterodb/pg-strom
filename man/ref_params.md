@ja{
#GUCパラメータ

本節ではPG-Stromの提供する設定パラメータについて説明します。
}
@en{
#GUC Parameters

This session introduces PG-Strom's configuration parameters.
}

@ja{
## 機能単位の有効化/無効化

`pg_strom.enabled` [型: `bool` / 初期値: `on]`
:   PG-Strom機能全体を一括して有効化/無効化する。

`pg_strom.enable_gpuscan` [型: `bool` / 初期値: `on]`
:   GpuScanによるスキャンを有効化/無効化する。

`pg_strom.enable_gpuhashjoin` [型: `bool` / 初期値: `on]`
:   GpuHashJoinによるJOINを有効化/無効化する。

`pg_strom.enable_gpunestloop` [型: `bool` / 初期値: `on]`
:   GpuNestLoopによるJOINを有効化/無効化する。

`pg_strom.enable_gpupreagg` [型: `bool` / 初期値: `on]`
:   GpuPreAggによる集約処理を有効化/無効化する。

`pg_strom.enable_brin` [型: `bool` / 初期値: `on]`
:   BRINインデックスを使ったテーブルスキャンを有効化/無効化する。

`pg_strom.enable_gpucache` [型: `bool` / 初期値: `on]`
:   PostgreSQLテーブルの代わりにGPUキャッシュを参照するかどうかを制御する。
:   なお、この設定値を`off`にしてもトリガ関数は引き続きREDOログバッファを更新し続けます。

`pg_strom.enable_partitionwise_gpujoin` [型: `bool` / 初期値: `on]`
:   GpuJoinを各パーティションの要素へプッシュダウンするかどうかを制御する。

`pg_strom.enable_partitionwise_gpupreagg` [型: `bool` / 初期値: `on]`
:   GpuPreAggを各パーティションの要素へプッシュダウンするかどうかを制御する。

`pg_strom.pullup_outer_scan` [型: `bool` / 初期値: `on]`
:   GpuPreAgg/GpuJoin直下の実行計画が全件スキャンである場合に、上位ノードでスキャン処理も行い、CPU/RAM⇔GPU間のデータ転送を省略するかどうかを制御する。

`pg_strom.pullup_outer_join` [型: `bool` / 初期値: `on]`
:   GpuPreAgg直下がGpuJoinである場合に、JOIN処理を上位の実行計画に引き上げ、CPU⇔GPU間のデータ転送を省略するかどうかを制御する。

`pg_strom.enable_numeric_aggfuncs` [型: `bool` / 初期値: `on]`
:   `numeric`データ型を引数に取る集約演算をGPUで処理するかどうかを制御する。
:   GPUでの集約演算において`numeric`データ型は倍精度浮動小数点数にマッピングされるため、計算誤差にセンシティブな用途の場合は、この設定値を `off` にしてCPUで集約演算を実行し、計算誤差の発生を抑えることができます。

`pg_strom.cpu_fallback` [型: `bool` / 初期値: `off]`
:   GPUプログラムが"CPU再実行"エラーを返したときに、実際にCPUでの再実行を試みるかどうかを制御する。

`pg_strom.regression_test_mode` [型: `bool` / 初期値: `off]`
:   GPUモデル名など、実行環境に依存して表示が変わる可能性のある`EXPLAIN`コマンドの出力を抑制します。これはリグレッションテストにおける偽陽性を防ぐための設定で、通常は利用者が操作する必要はありません。
}

@en{
## Enables/disables a particular feature

`pg_strom.enabled` [type: `bool` / default: `on]`
:   Enables/disables entire PG-Strom features at once

`pg_strom.enable_gpuscan` [type: `bool` / default: `on]`
:   Enables/disables GpuScan

`pg_strom.enable_gpuhashjoin` [type: `bool` / default: `on]`
:   Enables/disables JOIN by GpuHashJoin

`pg_strom.enable_gpunestloop` [type: `bool` / default: `on]`
:   Enables/disables JOIN by GpuNestLoop

`pg_strom.enable_gpupreagg` [type: `bool` / default: `on]`
:   Enables/disables GpuPreAgg

`pg_strom.enable_brin` [type: `bool` / default: `on]`
:   Enables/disables BRIN index support on tables scan

`pg_strom.enable_gpucache` [type: `bool` / default: `on]`
:   Controls whether GPU Cache is referenced, instead of PostgreSQL tables, if any
:   Note that GPU Cache trigger functions continue to update the REDO Log buffer, even if this parameter is turned off.

`pg_strom.enable_partitionwise_gpujoin` [type: `bool` / default: `on]`
:   Enables/disables whether GpuJoin is pushed down to the partition children.

`pg_strom.enable_partitionwise_gpupreagg` [type: `bool` / default: `on]`
:   Enables/disables whether GpuPreAgg is pushed down to the partition children.

`pg_strom.pullup_outer_scan` [type: `bool` / default: `on]`
:   Enables/disables to pull up full-table scan if it is just below GpuPreAgg/GpuJoin, to reduce data transfer between CPU/RAM and GPU.

`pg_strom.pullup_outer_join` [type: `bool` / default: `on]`
:   Enables/disables to pull up tables-join if GpuJoin is just below GpuPreAgg, to reduce data transfer between CPU/RAM and GPU.

`pg_strom.enable_numeric_aggfuncs` [type: `bool` / default: `on]`
:   Enables/disables support of aggregate function that takes `numeric` data type.
:   Note that aggregated function at GPU mapps `numeric` data type to double precision floating point values. So, if you are sensitive to calculation errors, you can turn off this configuration to suppress the calculation errors by the operations on CPU.

`pg_strom.cpu_fallback` [type: `bool` / default: `off]`
:   Controls whether it actually run CPU fallback operations, if GPU program returned "CPU ReCheck Error"

`pg_strom.regression_test_mode` [type: `bool` / default: `off]`
:   It disables some `EXPLAIN` command output that depends on software execution platform, like GPU model name. It avoid "false-positive" on the regression test, so use usually don't tough this configuration.
}

@ja{
## オプティマイザに関する設定

`pg_strom.chunk_size` [型: `int` / 初期値: `65534kB`]
:   PG-Stromが1回のGPUカーネル呼び出しで処理するデータブロックの大きさです。かつては変更可能でしたが、ほとんど意味がないため、現在では約64MBに固定されています。

`pg_strom.gpu_setup_cost` [型: `real` / 初期値: `4000`]
:   GPUデバイスの初期化に要するコストとして使用する値。

`pg_strom.gpu_dma_cost` [型: `real` / 初期値: `10`]
:   チャンク(`pg_strom.chunk_size` = 約64MB)あたりのDMA転送に要するコストとして使用する値。

`pg_strom.gpu_operator_cost` [型: `real` / 初期値: `0.00015`]
:   GPUの演算式あたりの処理コストとして使用する値。`cpu_operator_cost`よりも大きな値を設定してしまうと、いかなるサイズのテーブルに対してもPG-Stromが選択されることはなくなる。
}
@en{
## Optimizer Configuration

`pg_strom.chunk_size` [type: `int` / default: `65534kB`]
:   Size of the data blocks processed by a single GPU kernel invocation. It was configurable, but makes less sense, so fixed to about 64MB in the current version.

`pg_strom.gpu_setup_cost` [type: `real` / default: `4000`]
:   Cost value for initialization of GPU device

`pg_strom.gpu_dma_cost` [type: `real` / default: `10`]
:   Cost value for DMA transfer over PCIe bus per data-chunk (`pg_strom.chunk_size` = 64MB)

`pg_strom.gpu_operator_cost` [type: `real` / default: `0.00015`]
:   Cost value to process an expression formula on GPU. If larger value than `cpu_operator_cost` is configured, no chance to choose PG-Strom towards any size of tables
}

@ja{
## エグゼキュータに関する設定

`pg_strom.max_async_tasks` [型: `int` / 初期値: `5`]
:   PG-StromがGPU実行キューに投入する事ができる非同期タスクのプロセス毎の最大値。
:   CPUパラレル処理と併用する場合、この上限値は個々のバックグラウンドワーカー毎に適用されます。したがって、バッチジョブ全体では`pg_strom.max_async_tasks`よりも多くの非同期タスクが実行されることになります。

`pg_strom.reuse_cuda_context` [型: `bool` / 初期値: `off`]
:   クエリの実行に伴って作成したCUDAコンテキストを、次回のクエリ実行時に再利用します。
:   通常、CUDAコンテキストの作成には100～200ms程度を要するため、応答速度の改善が期待できる一方、一部のGPUデバイスメモリを占有し続けるというデメリットもあります。そのため、ベンチマーク等の用途を除いては使用すべきではありません。
:   また、CPUパラレルを利用する場合、ワーカープロセスでは必ずCUDAコンテキストを作成する事になりますので、効果は期待できません。
}
@en{
##Executor Configuration

`pg_strom.max_async_tasks` [type: `int` / default: `5`]
:   Max number of asynchronous taks PG-Strom can throw into GPU's execution queue per process.
:   If CPU parallel is used in combination, this limitation shall be applied for each background worker. So, more than `pg_strom.max_async_tasks` asynchronous tasks are executed in parallel on the entire batch job.

`pg_strom.reuse_cuda_context` [type: `bool` / default: `off`]
:   If `on`, it tries to reuse CUDA context on the next query execution, already constructed according to the previous query execution.
:   Usually, construction of CUDA context takes 100-200ms, it may improve queries response time, on the other hands, it continue to occupy a part of GPU device memory on the down-side. So, we don't recommend to enable this parameter expect for benchmarking and so on.
:   Also, this configuration makes no sense if query uses CPU parallel execution, because the worker processes shall always construct new CUDA context for each.
}

@ja{
##GPUダイレクトSQLの設定

`pg_strom.gpudirect_driver` [型: `text`]
:   GPUダイレクトSQLのドライバソフトウェア名を示す読み取り専用パラメータです。
:   `nvidia cufile`または`heterodb nvme-strom`のどちらかです。

`pg_strom.gpudirect_enabled` [型: `bool` / 初期値: `on`]
:   GPUダイレクトSQL機能を有効化/無効化する。

`pg_strom.gpudirect_threshold` [型: `int` / 初期値: 自動]
:   GPUダイレクトSQL機能を発動させるテーブルサイズの閾値を設定する。
:   初期値は自動設定で、システムの物理メモリと`shared_buffers`設定値から計算した閾値を設定します。

`pg_strom.cufile_io_unitsz` [型: `int` / 初期値: `16MB`]
:   cuFile APIを使用してデータを読み出す際のI/Oサイズを指定する。通常は変更の必要はありません。
:   `nvidia cufile`ドライバを使用する場合のみ有効です。

`pg_strom.nvme_distance_map` [型: `text` / 初期値: `null`]
:   NVMEデバイスやNFS区画など、ストレージ区画ごとに最も近傍のGPUを手動で設定します。
:   書式は `{(<gpuX>|<nvmeX>|<sfdvX>|</path/to/nfsmount>),...}[,{...}]`で、GPUとその近傍に位置するNVMEデバイスなどストレージの識別子を `{ ... }` で囲まれたグループに記述します。
:   （例: `{gpu0,nvme1,nvme2,/opt/nfsmount},{gpu1,nvme0}`
:   
:   - `<gpuX>`はデバイス番号Xを持つGPUです。
:   - `<nvmeX>`はローカルのNVME-SSDまたはリモートのNVME-oFデバイスを意味します。
:   - `<sfdvX>`はScaleFlux社製CSDドライブ用の専用デバイスを意味します。
:   - `/path/to/nfsmount`はNFS-over-RDMAを用いてマウントしたNFS区画のマウントポイントです。
:   
:   ローカルのNVME-SSDに対しては多くの場合自動設定で十分ですが、NVME-oFデバイスやNFS-over-RDMAを使用する場合、機械的に近傍のGPUを特定する事ができないため、手動で近傍のGPUを指定する必要があります。
}
@en{
##GPUDirect SQL Configuration

`pg_strom.gpudirect_driver` [type: `text`]
:   It shows the driver software name of GPUDirect SQL (read-only).
:   Either `nvidia cufile` or `heterodb nvme-strom`

`pg_strom.gpudirect_enabled` [type: `bool` / default: `on`]
:   Enables/disables GPUDirect SQL feature.

`pg_strom.gpudirect_threshold` [type: `int` / default: auto]
:   Controls the table-size threshold to invoke GPUDirect SQL feature.
:   The default is auto configuration; a threshold calculated by the system physical memory size and `shared_buffers` configuration.

`pg_strom.cufile_io_unitsz` [type: `int` / default: `16MB`]
:   Unit size of read-i/o when PG-Strom uses cuFile API. No need to change from the default setting for most cases.
:   It is only available when `nvidia cufile` driver is used.

`pg_strom.nvme_distance_map` [type: `text` / default: `null`]
:   It manually configures the closest GPU for particular storage evices, like NVME-SSD or NFS volumes.
:   Its format string is `{(<gpuX>|<nvmeX>|<sfdvX>|</path/to/nfsmount>),...}[,{...}]`. It puts identifiers of GPU and NVME devices within `{ ... }` block to group these devices.
:   (example: `{gpu0,nvme1,nvme2,/opt/nfsmount},{gpu1,nvme0}`
:   
:   - `<gpuX>` means a GPU with device identifier X.
:   - `<nvmeX>` means a local NVME-SSD or a remote NVME-oF device.
:   - `<sfdvX>` means a special device of CSD drives by ScaleFlux,Inc.
:   - `/path/to/nfsmount` means a mount point by NFS volume with NFS-over-RDMA.
:   
:   Automatic configuration is often sufficient for local NVME-SSD drives, however, you should manually configure the closest GPU for NVME-oF or NFS-over-RDMA volumes.
}

@ja{
##Arrow_Fdw関連の設定

`arrow_fdw.enabled` [型: `bool` / 初期値: `on`]
:   推定コスト値を調整し、Arrow_Fdwの有効/無効を切り替えます。ただし、GpuScanが利用できない場合には、Arrow_FdwによるForeign ScanだけがArrowファイルをスキャンできるという事に留意してください。

`arrow_fdw.metadata_cache_size` [型: `int` / 初期値: `128MB`]
:   Arrowファイルのメタ情報をキャッシュする共有メモリ領域の大きさを指定します。共有メモリの消費量がこのサイズを越えると、古いメタ情報から順に解放されます。

`arrow_fdw.record_batch_size` [型: `int` / 初期値: `256MB`]
:   Arrow_Fdw外部テーブルへ書き込む際の RecordBatch の大きさの閾値です。`INSERT`コマンドが完了していなくとも、Arrow_Fdwは総書き込みサイズがこの値を越えるとバッファの内容をApache Arrowファイルへと書き出します。
}
@en{
##Arrow_Fdw Configuration

`arrow_fdw.enabled` [type: `bool` / default: `on`]
:   By adjustment of estimated cost value, it turns on/off Arrow_Fdw. Note that only Foreign Scan (Arrow_Fdw) can scan on Arrow files, if GpuScan is not capable to run on.

`arrow_fdw.metadata_cache_size` [type: `int` / default: `128MB`]
:   Size of shared memory to cache metadata of Arrow files.
:   Once consumption of the shared memory exceeds this value, the older metadata shall be released based on LRU.

`arrow_fdw.record_batch_size` [type: `int` / default: `256MB`]
:   Threshold of RecordBatch when Arrow_Fdw foreign table is written. When total amount of the buffer size exceeds this configuration, Arrow_Fdw writes out the buffer to Apache Arrow file, even if `INSERT` command is not completed yet.
}

@ja{
##GPUキャッシュの設定
`pg_strom.enable_gpucache` [型: `bool` / 初期値: `on`]
:   検索/分析系のクエリでGPUキャッシュを使用するかどうかを制御します。
:   なお、本設定はトリガによるREDOログバッファへの追記には影響しません。

`pg_strom.gpucache_auto_preload` [型: `text` / 初期値: `null`]
:   PostgreSQLの起動直後にGPUキャッシュをロードすべきテーブル名を指定します。
:   書式は `DATABASE_NAME.SCHEMA_NAME.TABLE_NAME` で、複数個のテーブルを指定する場合はこれをカンマ区切りで並べます。
:   GPUキャッシュの初回ロードは相応に時間のかかる処理ですが、事前に初回ロードを済ませておく事で、検索/分析クエリの初回実行時に応答速度が遅延するのを避けることができます。
:   なお、本パラメータを '*' に設定すると、GPUキャッシュを持つ全てのテーブルの内容を順にGPUへロードしようと試みます。
}
@en{
##GPU Cache configuration
`pg_strom.enable_gpucache` [type: `bool` / default: `on`]
:   Controls whether search/analytic query tries to use GPU Cache.
:   Note that this parameter does not affect to any writes on the REDO Log buffer by the trigger.

`pg_strom.gpucache_auto_preload` [type: `text` / default: `null`]
:   It specifies the table names to be loaded onto GPU Cache just after PostgreSQL startup.
:   Its format is `DATABASE_NAME.SCHEMA_NAME.TABLE_NAME`, and separated by comma if multiple tables are preloaded.
:   Initial-loading of GPU Cache usually takes a lot of time. So, preloading enables to avoid delay of response time of search/analytic queries on the first time.
:   If this parameter is '*', PG-Strom tries to load all the configured tables onto GPU Cache sequentially.
}

@ja{
##HyperLogLogの設定
`pg_strom.hll_registers_bits` [型: `int` / 初期値: `9`]
:    HyperLogLogで使用する HLL Sketch の幅を指定します。
:    実行時に`2^pg_strom.hll_registers_bits`個のレジスタを割当て、ハッシュ値の下位`pg_strom.hll_registers_bits`ビットをレジスタのセレクタとして使用します。設定可能な値は4～15の範囲内です。
:    PG-StromのHyperLogLog機能について、詳しくは[HyperLogLog](../hll_count/)を参照してください。
}

@en{
##HyperLogLog configuration
`pg_strom.hll_registers_bits` [type: `int` / default: `9`]
:    It specifies the width of HLL Sketch used for HyperLogLog.
:    PG-Strom allocates `2^pg_strom.hll_registers_bits` registers for HLL Sketch, then uses the latest `pg_strom.hll_registers_bits` bits of hash-values as register selector. It must be configured between 4 and 15.
:    See [HyperLogLog](../hll_count/) for more details of HyperLogLog functionality of PG-Strom.
}

@ja{
##GPUコードの生成、およびJITコンパイルの設定

`pg_strom.program_cache_size` [型: `int` / 初期値: `256MB`]
:   ビルド済みのGPUプログラムをキャッシュしておくための共有メモリ領域のサイズです。パラメータの更新には再起動が必要です。

`pg_strom.num_program_builders` [型: `int` / 初期値: `2`]
:   GPUプログラムを非同期ビルドするためのバックグラウンドプロセスの数を指定します。パラメータの更新には再起動が必要です。

`pg_strom.debug_jit_compile_options` [型: `bool` / 初期値: `off`]
:   GPUプログラムのJITコンパイル時に、デバッグオプション（行番号とシンボル情報）を含めるかどうかを指定します。GPUコアダンプ等を用いた複雑なバグの解析に有用ですが、性能のデグレードを引き起こすため、通常は使用すべきでありません。

`pg_strom.extra_kernel_stack_size` [型: `int` / 初期値: `0`]
:   GPUカーネルの実行時にスレッド毎に追加的に割り当てるスタックの大きさをバイト単位で指定します。通常は初期値を変更する必要はありません。
}
@en{
##Configuration of GPU code generation and build

`pg_strom.program_cache_size` [type: `int` / default: `256MB`]
:   Amount of the shared memory size to cache GPU programs already built. It needs restart to update the parameter.

`pg_strom.num_program_builders` [type: `int` / default: `2`]
:   Number of background workers to build GPU programs asynchronously. It needs restart to update the parameter.

`pg_strom.debug_jit_compile_options` [type: `bool` / default: `off`]
:   Controls to include debug option (line-numbers and symbol information) on JIT compile of GPU programs.
:   It is valuable for complicated bug analysis using GPU core dump, however, should not be enabled on daily use because of performance degradation.

`pg_strom.extra_kernel_stack_size` [type: `int` / default: `0`]
:   Extra size of stack, in bytes, for each GPU kernel thread to be allocated on execution. Usually, no need to change from the default value.
}

@ja{
##GPUデバイスに関連する設定

`pg_strom.cuda_visible_devices` [型: `text` / 初期値: `null`]
:   PostgreSQLの起動時に特定のGPUデバイスだけを認識させてい場合は、カンマ区切りでGPUデバイス番号を記述します。
:   これは環境変数`CUDA_VISIBLE_DEVICES`を設定するのと同等です。

`pg_strom.gpu_memory_segment_size` [型: `int` / 初期値: `512MB`]
:   PG-StromがGPUメモリをアロケーションする際に、1回のCUDA API呼び出しで獲得するGPUデバイスメモリのサイズを指定します。
:   この値が大きいとAPI呼び出しのオーバーヘッドは減らせますが、デバイスメモリのロスは大きくなります。
}
@en{
##GPU Device Configuration

`pg_strom.cuda_visible_devices` [type: `text` / default: `null`]
:   List of GPU device numbers in comma separated, if you want to recognize particular GPUs on PostgreSQL startup.
:   It is equivalent to the environment variable `CUDAVISIBLE_DEVICES`

`pg_strom.gpu_memory_segment_size` [type: `int` / default: `512MB`]
:   Specifies the amount of device memory to be allocated per CUDA API call.
:   Larger configuration will reduce the overhead of API calls, but not efficient usage of device memory.
}

@ja{
##PG-Strom共有メモリに関連する設定

`shmbuf.segment_size` [型: `int` / 初期値: `256MB`]
:   ポータブルな仮想アドレスを持つ共有メモリセグメントの単位長を指定します。
:   通常は初期値を変更する必要はありませんが、GPUキャッシュのREDOログバッファに`256MB`以上の大きさを指定する場合には、本パラメータも併せて拡大する必要があります。
:   本パラメータの設定値は2のべき乗だけが許されます。

`shmbuf.num_logical_segments` [型: `int` / 初期値: 自動]
:   ポータブルな仮想アドレスを持つ共有メモリのセグメント数を指定します。
:   PG-Stromは起動時に(`shmbuf.segment_size` x `shmbuf.num_logical_segments`)バイトの領域をPROT_NONE属性でmmap(2)し、その後、シグナルハンドラを利用してオンデマンドの割当てを行います。
:   デフォルトの論理セグメントサイズは自動設定で、システム搭載物理メモリの2倍の大きさです。
}
@en{
##PG-Strom shared memory configuration

`shmbuf.segment_size` [type: `int` / default: `256MB`]
:   It configures the unit length of the shared memory segment that has portable virtual addresses.
:   Usually, it does not need to change the default value, except for the case when GPU Cache uses REDO Log buffer larger than `256MB`. In this case, you need to enlarge this parameter also.
:   This parameter allows only power of 2.

`shmbuf.num_logical_segments` [type: `int` / default: auto]
:   It configures the number of the shared memory segment that has portable virtual addresses.
:   On the system startup, PG-Strom reserves (`shmbuf.segment_size` x `shmbuf.num_logical_segments`) bytes of virtual address space using mmap(2) with PROT_NONE, then, signal handler allocates physical memory on the demand.
:   The default configuration is auto; that is almost twice of the physical memory size installed on the system.
}


