@ja{
#GUCパラメータ

本節ではPG-Stromの提供する設定パラメータについて説明します。
}
@en{
#GUC Parameters

This session introduces PG-Strom's configuration parameters.
}

@ja:## 機能単位の有効化/無効化
@en:## Enables/disables a particular feature

@ja{
`pg_strom.enabled` [型: `bool` / 初期値: `on]`
:   PG-Strom機能全体を一括して有効化/無効化する。
}
@en{
`pg_strom.enabled` [type: `bool` / default: `on]`
:   Enables/disables entire PG-Strom features at once
}

@ja{
`pg_strom.enable_gpuscan` [型: `bool` / 初期値: `on]`
:   GpuScanによるスキャンを有効化/無効化する。
}
@en{
`pg_strom.enable_gpuscan` [type: `bool` / default: `on]`
:   Enables/disables GpuScan
}

@ja{
`pg_strom.enable_gpuhashjoin` [型: `bool` / 初期値: `on]`
:   GpuHashJoinによるJOINを有効化/無効化する。
}
@en{
`pg_strom.enable_gpuhashjoin` [type: `bool` / default: `on]`
:   Enables/disables JOIN by GpuHashJoin
}

@ja{
`pg_strom.enable_gpugistindex` [型: `bool` / 初期値: `on]`
:   GpuGiSTIndexによるJOINを有効化/無効化する。
}
@en{
`pg_strom.enable_gpugistindex` [type: `bool` / default: `on]`
:   Enables/disables JOIN by GpuGiSTIndex
}

@ja{
`pg_strom.enable_gpujoin` [型: `bool` / 初期値: `on]`
:   GpuJoinによるJOINを一括で有効化/無効化する。（GpuHashJoinとGpuGiSTIndexを含む）
}
@en{
`pg_strom.enable_gpujoin` [type: `bool` / default: `on]`
:   Enables/disables entire GpuJoin features (including GpuHashJoin and GpuGiSTIndex)
}

@ja{
`pg_strom.enable_gpupreagg` [型: `bool` / 初期値: `on]`
:   GpuPreAggによる集約処理を有効化/無効化する。
}
@en{
`pg_strom.enable_gpupreagg` [type: `bool` / default: `on]`
:   Enables/disables GpuPreAgg
}

@ja{
`pg_strom.enable_numeric_aggfuncs` [型: `bool` / 初期値: `on]`
:   `numeric`データ型を引数に取る集約演算をGPUで処理するかどうかを制御する。
:   GPUでの集約演算において`numeric`データ型は倍精度浮動小数点数にマッピングされるため、計算誤差にセンシティブな用途の場合は、この設定値を `off` にしてCPUで集約演算を実行し、計算誤差の発生を抑えることができます。
}
@en{
`pg_strom.enable_numeric_aggfuncs` [type: `bool` / default: `on]`
:   Enables/disables support of aggregate function that takes `numeric` data type.
:   Note that aggregated function at GPU mapps `numeric` data type to double precision floating point values. So, if you are sensitive to calculation errors, you can turn off this configuration to suppress the calculation errors by the operations on CPU.
}

@ja{
`pg_strom.enable_brin` [型: `bool` / 初期値: `on]`
:   BRINインデックスを使ったテーブルスキャンを有効化/無効化する。
}
@en{
`pg_strom.enable_brin` [type: `bool` / default: `on]`
:   Enables/disables BRIN index support on tables scan
}

@ja{
`pg_strom.cpu_fallback` [型: `enum` / 初期値: `notice`]
:   GPUプログラムが"CPU再実行"エラーを返したときに、実際にCPUでの再実行を試みるかどうかを制御する。
:   `notice` ... メッセージを出力した上でCPUでの再実行を行う
:   `on`, `true` ... メッセージを出力せずCPUでの再実行を行う
:   `off`, `false` ... エラーを発生させCPUでの再実行を行わない
}
@en{
`pg_strom.cpu_fallback` [type: `enum` / default: `notice`]
:   Controls whether it actually run CPU fallback operations, if GPU program returned "CPU ReCheck Error"
:   `notice` ... Runs CPU fallback operations with notice message
:   `on`, `true` ... Runs CPU fallback operations with no message output
:   `off`, `false` ... Disabled CPU fallback operations with an error
}
@ja{
`pg_strom.regression_test_mode` [型: `bool` / 初期値: `off]`
:   GPUモデル名など、実行環境に依存して表示が変わる可能性のある`EXPLAIN`コマンドの出力を抑制します。これはリグレッションテストにおける偽陽性を防ぐための設定で、通常は利用者が操作する必要はありません。
}
@en{
`pg_strom.regression_test_mode` [type: `bool` / default: `off]`
:   It disables some `EXPLAIN` command output that depends on software execution platform, like GPU model name. It avoid "false-positive" on the regression test, so use usually don't tough this configuration.
}

@ja:## オプティマイザに関する設定
@en:## Optimizer Configuration

@ja{
`pg_strom.gpu_setup_cost` [型: `real` / 初期値: `100 * DEFAULT_SEQ_PAGE_COST`]
:   GPUデバイスの初期化に要するコストとして使用する値。
}
@en{
`pg_strom.gpu_setup_cost` [type: `real` / default: `100 * DEFAULT_SEQ_PAGE_COST`]
:   Cost value for initialization of GPU device
}

@ja{
`pg_strom.gpu_tuple_cost` [型: `real` / 初期値: `DEFAULT_CPU_TUPLE_COST`]
:   GPUへ送出する／受け取るタプル一個あたりのコストとして使用する値。
}
@en{
`pg_strom.gpu_tuple_cost` [type: `real` / default: `DEFAULT_CPU_TUPLE_COST`]
:   Cost value to send tuples to, or receive tuples from GPU for each.
}

@ja{
`pg_strom.gpu_operator_cost` [型: `real` / 初期値: `DEFAULT_CPU_OPERATOR_COST / 16`]
:   GPUの演算式あたりの処理コストとして使用する値。`cpu_operator_cost`よりも大きな値を設定してしまうと、いかなるサイズのテーブルに対してもPG-Stromが選択されることはなくなる。
}
@en{
`pg_strom.gpu_operator_cost` [type: `real` / default: `DEFAULT_CPU_OPERATOR_COST / 16`]
:   Cost value to process an expression formula on GPU. If larger value than `cpu_operator_cost` is configured, no chance to choose PG-Strom towards any size of tables
}

@ja{
`pg_strom.enable_partitionwise_gpujoin` [型: `bool` / 初期値: `on]`
:   GpuJoinを各パーティションの要素へプッシュダウンするかどうかを制御する。
}
@en{
`pg_strom.enable_partitionwise_gpujoin` [type: `bool` / default: `on]`
:   Enables/disables whether GpuJoin is pushed down to the partition children.
}
@ja{
`pg_strom.enable_partitionwise_gpupreagg` [型: `bool` / 初期値: `on]`
:   GpuPreAggを各パーティションの要素へプッシュダウンするかどうかを制御する。
}
@en{
`pg_strom.enable_partitionwise_gpupreagg` [type: `bool` / default: `on]`
:   Enables/disables whether GpuPreAgg is pushed down to the partition children.
}

@ja{
`pg_strom.pinned_inner_buffer_threshold` [型: `int` / 初期値: `0`]
:    GpuJoinのINNER表がGpuScanまたはGpuJoinである場合、処理結果の推定サイズがこの設定値よりも大きければ、結果をいったんCPUに戻すことなく、そのままGPU側に保持した上で、続くGpuJoinのINNERバッファとして使用する。
:    設定値が`0`の場合、本機能は無効となる。
}
@en{
`pg_strom.pinned_inner_buffer_threshold` [type: `int` / 初期値: `0`]
:    If the INNER table of GpuJoin is either GpuScan or GpuJoin, and the estimated size of its processing result is larger than this configured value, the result is retained on the GPU device without being returned to the CPU, and then reused as a part of the INNER buffer of the subsequent GpuJoin.
:    If the configured value is `0`, this function will be disabled.
}

@ja:## エグゼキュータに関する設定
@en:## Executor Configuration

@ja{
`pg_strom.max_async_tasks` [型: `int` / 初期値: `12`]
:   PG-StromがGPU実行キューに投入する事ができる非同期タスクのGPUデバイス毎の最大値で、GPU Serviceのワーカースレッド数でもあります。
}
@en{
`pg_strom.max_async_tasks` [type: `int` / default: `12`]
:   Max number of asynchronous taks PG-Strom can submit to the GPU execution queue, and is also the number of GPU Service worker threads.
}

@ja:## GPUダイレクトSQLの設定
@en:## GPUDirect SQL Configuration

@ja{
`pg_strom.gpudirect_driver` [型: `text`]
:   GPUダイレクトSQLのドライバソフトウェア名を示すパラメータです。
:   `cufile`、`nvme-strom`、もしくは`vfs`のどれかです。
}
@en{
`pg_strom.gpudirect_driver` [type: `text`]
:   It shows the driver software name of GPUDirect SQL (read-only).
:   Either `cufile`, `nvme-strom` or `vfs`
}

@ja{
`pg_strom.gpudirect_enabled` [型: `bool` / 初期値: `on`]
:   GPUダイレクトSQL機能を有効化/無効化する。
}
@en{
`pg_strom.gpudirect_enabled` [type: `bool` / default: `on`]
:   Enables/disables GPUDirect SQL feature.
}

@ja{
`pg_strom.gpu_direct_seq_page_cost` [型: `real` / 初期値: `DEFAULT_SEQ_PAGE_COST / 4`]
:   オプティマイザが実行プランのコストを計算する際に、GPU-Direct SQLを用いてテーブルをスキャンする場合のコストとして`seq_page_cost`の代わりに使用される値。
}
@en{
`pg_strom.gpu_direct_seq_page_cost` [type: `real` / default: `DEFAULT_SEQ_PAGE_COST / 4`]
:   The cost of scanning a table using GPU-Direct SQL, instead of the `seq_page_cost`, when the optimizer calculates the cost of an execution plan.
}

@ja{
`pg_strom.gpudirect_threshold` [型: `int` / 初期値: 自動]
:   GPUダイレクトSQL機能を発動させるテーブルサイズの閾値を設定する。
:   初期値は自動設定で、システムの物理メモリと`shared_buffers`設定値から計算した閾値を設定します。
}
@en{
`pg_strom.gpudirect_threshold` [type: `int` / default: auto]
:   Controls the table-size threshold to invoke GPUDirect SQL feature.
:   The default is auto configuration; a threshold calculated by the system physical memory size and `shared_buffers` configuration.
}

@ja{
`pg_strom.manual_optimal_gpus` [型: `text` / 初期値: なし]
:   NVMEデバイスやNFS区画など、ストレージ区画ごとに最も近傍と判定されるGPUを手動で設定します。
:   書式は `{<nvmeX>|/path/to/tablespace}=gpuX[:gpuX...]`で、NVMEデバイスまたはテーブルスペースのパスと、その近傍であるGPU（複数可）を記述します。カンマで区切って複数の設定を記述する事も可能です。
:   例: `pg_strom.manual_optimal_gpus = 'nvme1=gpu0,nvme2=gpu1,/mnt/nfsroot=gpu0'`
:   - `<gpuX>`はデバイス番号Xを持つGPUです。
:   - `<nvmeX>`はローカルのNVME-SSDまたはリモートのNVME-oFデバイスを意味します。
:   - `/path/to/tablespace`は、テーブルスペースに紐づいたディレクトリのフルパスです。
:   ローカルのNVME-SSDに対しては多くの場合自動設定で十分ですが、NVME-oFデバイスやNFS-over-RDMAを使用する場合、機械的に近傍のGPUを特定する事ができないため、手動で近傍のGPUを指定する必要があります。
}
@en{
`pg_strom.manual_optimal_gpus` [type: `text` / default: none]
:   It manually configures the closest GPU for the target storage volumn, like NVME device or NFS volume.
:   Its format string is: `{<nvmeX>|/path/to/tablespace}=gpuX[:gpuX...]`. It describes relationship between the closest GPU and NVME device or tablespace directory path. It accepts multiple configurations separated by comma character.
:   Example: `pg_strom.manual_optimal_gpus = 'nvme1=gpu0,nvme2=gpu1,/mnt/nfsroot=gpu0'`
:   - `<gpuX>` means a GPU with device identifier X.
:   - `<nvmeX>` means a local NVME-SSD or a remote NVME-oF device.
:   - `/path/to/tablespace` means full-path of the tablespace directory.
:   Automatic configuration is often sufficient for local NVME-SSD drives, however, you should manually configure the closest GPU for NVME-oF or NFS-over-RDMA volumes.
}

@ja:## Arrow_Fdw関連の設定
@en:## Arrow_Fdw Configuration

@ja{
`arrow_fdw.enabled` [型: `bool` / 初期値: `on`]
:   推定コスト値を調整し、Arrow_Fdwの有効/無効を切り替えます。ただし、GpuScanが利用できない場合には、Arrow_FdwによるForeign ScanだけがArrowファイルをスキャンできるという事に留意してください。
}
@en{
`arrow_fdw.enabled` [type: `bool` / default: `on`]
:   By adjustment of estimated cost value, it turns on/off Arrow_Fdw. Note that only Foreign Scan (Arrow_Fdw) can scan on Arrow files, if GpuScan is not capable to run on.
}

@ja{
`arrow_fdw.stats_hint_enabled` [型: `bool` / 初期値: `on`]
:   Arrowファイルがmin/max統計情報を持っており、それを用いて不必要なrecord-batchを読み飛ばすかどうかを制御します。
}
@en{
`arrow_fdw.stats_hint_enabled` [type: `bool` / default: `on`]
:   When Arrow file has min/max statistics, this parameter controls whether unnecessary record-batches shall be skipped, or not.
}

@ja{
`arrow_fdw.metadata_cache_size` [型: `int` / 初期値: `512MB`]
:   Arrowファイルのメタ情報をキャッシュする共有メモリ領域の大きさを指定します。共有メモリの消費量がこのサイズを越えると、古いメタ情報から順に解放されます。
}
@en{
`arrow_fdw.metadata_cache_size` [type: `int` / default: `512MB`]
:   Size of shared memory to cache metadata of Arrow files.
:   Once consumption of the shared memory exceeds this value, the older metadata shall be released based on LRU.
}

@ja:##GPUキャッシュの設定
@en:##GPU Cache configuration
@ja{
`pg_strom.enable_gpucache` [型: `bool` / 初期値: `on`]
:   検索/分析系のクエリでGPUキャッシュを使用するかどうかを制御します。
:   なお、この設定値を`off`にしてもトリガ関数は引き続きREDOログバッファを更新し続けます。
}
@en{
`pg_strom.enable_gpucache` [type: `bool` / default: `on`]
:   Controls whether search/analytic query tries to use GPU Cache.
:   Note that GPU Cache trigger functions continue to update the REDO Log buffer, even if this parameter is turned off.
}
@ja{
`pg_strom.gpucache_auto_preload` [型: `text` / 初期値: `null`]
:   PostgreSQLの起動直後にGPUキャッシュをロードすべきテーブル名を指定します。
:   書式は `DATABASE_NAME.SCHEMA_NAME.TABLE_NAME` で、複数個のテーブルを指定する場合はこれをカンマ区切りで並べます。
:   GPUキャッシュの初回ロードは相応に時間のかかる処理ですが、事前に初回ロードを済ませておく事で、検索/分析クエリの初回実行時に応答速度が遅延するのを避けることができます。
:   なお、本パラメータを '*' に設定すると、GPUキャッシュを持つ全てのテーブルの内容を順にGPUへロードしようと試みます。
}
@en{
`pg_strom.gpucache_auto_preload` [type: `text` / default: `null`]
:   It specifies the table names to be loaded onto GPU Cache just after PostgreSQL startup.
:   Its format is `DATABASE_NAME.SCHEMA_NAME.TABLE_NAME`, and separated by comma if multiple tables are preloaded.
:   Initial-loading of GPU Cache usually takes a lot of time. So, preloading enables to avoid delay of response time of search/analytic queries on the first time.
:   If this parameter is '*', PG-Strom tries to load all the configured tables onto GPU Cache sequentially.
}

<!--
@ja:##HyperLogLogの設定
@en:##HyperLogLog configuration
@ja{
`pg_strom.hll_registers_bits` [型: `int` / 初期値: `9`]
:    HyperLogLogで使用する HLL Sketch の幅を指定します。
:    実行時に`2^pg_strom.hll_registers_bits`個のレジスタを割当て、ハッシュ値の下位`pg_strom.hll_registers_bits`ビットをレジスタのセレクタとして使用します。設定可能な値は4～15の範囲内です。
:    PG-StromのHyperLogLog機能について、詳しくは[HyperLogLog](hll_count.md)を参照してください。
}
@en{
`pg_strom.hll_registers_bits` [type: `int` / default: `9`]
:    It specifies the width of HLL Sketch used for HyperLogLog.
:    PG-Strom allocates `2^pg_strom.hll_registers_bits` registers for HLL Sketch, then uses the latest `pg_strom.hll_registers_bits` bits of hash-values as register selector. It must be configured between 4 and 15.
:    See [HyperLogLog](hll_count.md) for more details of HyperLogLog functionality of PG-Strom.
}
-->


@ja:## GPUデバイスに関連する設定
@en:## GPU Device Configuration

@ja{
`pg_strom.gpu_mempool_segment_sz` [型: `int` / 初期値: `1GB`]
:   GPU Serviceがメモリプール用にGPUメモリを確保する際のセグメントサイズです。
:   GPUデバイスメモリの割当ては比較的ヘビーな処理であるため、メモリプールを使用してメモリを使い回す事が推奨されています。
}
@en{
`pg_strom.gpu_mempool_segment_sz` [type: `int` / default: `1GB`]
:   The segment size when GPU Service allocates GPU device memory for the memory pool.
:   GPU device memory allocation is a relatively heavy process, so it is recommended to use memory pools to reuse memory.
}

@ja{
`pg_strom.gpu_mempool_max_ratio` [型: `real` / 初期値: `50%`]
:   GPUデバイスメモリのメモリプール用に使用する事のできるデバイスメモリの割合を指定します。
:   メモリプールによる過剰なGPUデバイスメモリの消費を抑制し、ワーキングメモリを十分に確保する事が目的です。
}
@en{
`pg_strom.gpu_mempool_max_ratio` [type: `real` / default: `50%`]
:   It specifies the percentage of device memory that can be used for the GPU device memory memory pool.
:   It works to suppress excessive GPU device memory consumption by the memory pool and ensure sufficient working memory.
}

@ja{
`pg_strom.gpu_mempool_min_ratio` [型: `real` / 初期値: `5%`]
:   メモリプールに確保したGPUデバイスメモリのうち、利用終了後も解放せずに確保したままにしておくデバイスメモリの割合を指定します。
:   最小限度のメモリプールを保持しておくことにより、次のクエリを速やかに実行する事ができます。
}
@en{
`pg_strom.gpu_mempool_min_ratio` [type: `real` / default: `5%`]
:   It specify the percentage of GPU device memory that is preserved as the memory pool segment, and remained even after memory usage.
:   By maintaining a minimum memory pool, the next query can be executed quickly.
}

@ja{
`pg_strom.gpu_mempool_release_delay` [型: `int` / 初期値: `5000`]
:   GPU Serviceは、あるメモリプール上のセグメントが空になっても、これを直ちに開放しません。そのセグメントが最後に利用されてから、本パラメータで指定された時間（ミリ秒単位）を経過すると、これを開放してシステムに返却します。
:   一定の遅延を挟む事で、GPUデバイスメモリの割当/解放の頻度を減らす事ができます。
}
@en{
`pg_strom.gpu_mempool_release_delay` [type: `int` / default: `5000`]
:   GPU Service does not release a segment of a memory pool immediately, even if it becomes empty. When the time specified by this parameter (in milliseconds) has elapsed since the segment was last used, it is released and returned to the system.
:   By inserting a certain delay, you can reduce the frequency of GPU device memory allocation/release.
}

@ja{
`pg_strom.gpuserv_debug_output` [型: `bool` / 初期値: `false`]
:   GPU Serviceのデバッグメッセージ出力を有効化/無効化します。このメッセージはデバッグにおいて有効である場合がありますが、通常は初期値のまま変更しないで下さい。
}
@en{
`pg_strom.gpuserv_debug_output` [type: `bool` / default: `false`]
:   Enable/disable GPU Service debug message output. This message may be useful for debugging, but normally you should not change it from the default value.
}

@ja{
`pg_strom.cuda_visible_devices` [型: `text` / 初期値: `null`]
:   PostgreSQLの起動時に特定のGPUデバイスだけを認識させてい場合は、カンマ区切りでGPUデバイス番号を記述します。
:   これは環境変数`CUDA_VISIBLE_DEVICES`を設定するのと同等です。
}
@en{
`pg_strom.cuda_visible_devices` [type: `text` / default: `null`]
:   List of GPU device numbers in comma separated, if you want to recognize particular GPUs on PostgreSQL startup.
:   It is equivalent to the environment variable `CUDAVISIBLE_DEVICES`
}

<!--
@ja:## DPU関連設定
@en:## DPU related configurations

@ja{
`pg_strom.dpu_endpoint_list` [型: `text` / 初期値: なし]
:   
}
@ja{
pg_strom.dpu_endpoint_default_port [型: `int` / 初期値: 6543]
}
@ja{
`pg_strom.enable_dpuscan` [型: `bool` / 初期値: `on`]
:   DpuScanによるスキャンを有効化/無効化する。
}
@ja{
`pg_strom.enable_dpujoin` [型: `bool` / 初期値: `on`]
:   DpuJoinによるJOINを一括で有効化/無効化する。（DpuHashJoinとDpuGiSTIndexを含む）
}
@ja{
`pg_strom.enable_dpuhashjoin` [型: `bool` / 初期値: `on`]
:   DpuHashJoinによるJOINを有効化/無効化する。
}
@ja{
`pg_strom.enable_dpugistindex` [型: `bool` / 初期値: `on`]
:   DpuGiSTIndexによるJOINを有効化/無効化する。
}
@ja{
`pg_strom.enable_dpupreagg` [型: `bool` / 初期値: `on`]
:   DpuPreAggによる集約処理を有効化/無効化する。
}
@ja{
`pg_strom.dpu_setup_cost` [型: `real` / 初期値: `100 * DEFAULT_SEQ_PAGE_COST`]
:   DPUデバイスの初期化に要するコストとして使用する値。
}
@ja{
`pg_strom.dpu_operator_cost` [型: `real` / 初期値: `1.2 * DEFAULT_CPU_OPERATOR_COST`]
:   DPUの演算式あたりの処理コストとして使用する値。
}
@ja{
`pg_strom.dpu_seq_page_cost` [型: `real` / 初期値: `DEFAULT_SEQ_PAGE_COST / 4`]
:   DPUデバイスが自身に紐づいたストレージからブロックを読み出すためのコスト
}
@ja{
`pg_strom.dpu_tuple_cost` [型: `real` / 初期値: `DEFAULT_CPU_TUPLE_COST`]
:   DPUへ送出する／受け取るタプル一個あたりのコストとして使用する値。
}
@ja{
`pg_strom.dpu_handle_cached_pages` [型: `bool` / 初期値: `off`]
:   PostgreSQL側の共有バッファに載っており、更新がストレージ側にまだ反映されていないブロックを、わざわざDPU側に送出してDPU側で処理させるかどうかを制御します。
:   通常、DPUの処理パフォーマンスはCPUよりも劣る上、さらにデータ転送のロスを含めるとCPUで処理する方が賢明です。
}
@ja{
`pg_strom.enable_partitionwise_dpupreagg` [型: `bool` / 初期値: `on`]
}
@ja{
`pg_strom.enable_partitionwise_dpupreagg` [型: `bool` / 初期値: `off`]
}

-->


