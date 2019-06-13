--
-- validation of the default GUC paremeter setting
--
SHOW pg_strom.enabled;
SHOW pg_strom.enable_gpuscan;
SHOW pg_strom.enable_gpuhashjoin;
SHOW pg_strom.enable_gpunestloop;
SHOW pg_strom.enable_gpupreagg;
SHOW pg_strom.enable_numeric_aggfuncs;
SHOW pg_strom.cpu_fallback;
SHOW pg_strom.gpu_setup_cost;
SHOW pg_strom.gpu_operator_cost;
SHOW pg_strom.gpu_dma_cost;
SHOW pg_strom.pullup_outer_scan;
SHOW pg_strom.pullup_outer_join;
