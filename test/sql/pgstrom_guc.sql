--
-- validation of the default GUC paremeter setting
--
\t
SHOW pg_strom.enabled;
SHOW pg_strom.enable_gpuscan;
SHOW pg_strom.enable_gpuhashjoin;
SHOW pg_strom.enable_gpugistindex;
SHOW pg_strom.enable_gpujoin;
SHOW pg_strom.enable_gpupreagg;
SHOW pg_strom.enable_numeric_aggfuncs;
SHOW pg_strom.enable_brin;
SHOW pg_strom.cpu_fallback;
SHOW pg_strom.gpu_setup_cost;
SHOW pg_strom.gpu_tuple_cost;
SHOW pg_strom.gpu_operator_cost;
SHOW pg_strom.max_async_tasks;
SHOW pg_strom.gpudirect_enabled;
SHOW pg_strom.gpu_direct_seq_page_cost;
SHOW arrow_fdw.enabled;
SHOW arrow_fdw.stats_hint_enabled;
SHOW arrow_fdw.metadata_cache_size;
SHOW pg_strom.enable_gpucache;
SHOW pg_strom.gpucache_auto_preload;
SHOW pg_strom.gpu_mempool_segment_sz;
SHOW pg_strom.gpu_mempool_max_ratio;
SHOW pg_strom.gpu_mempool_min_ratio;
SHOW pg_strom.gpu_mempool_release_delay;
SHOW pg_strom.gpuserv_debug_output;