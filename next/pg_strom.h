/*
 * pg_strom.h
 *
 * Header file of pg_strom module
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef PG_STROM_H
#define PG_STROM_H

#include "postgres.h"
#if PG_VERSION_NUM < 140000
#error Base PostgreSQL version must be v14 or later
#endif
#define PG_MAJOR_VERSION		(PG_VERSION_NUM / 100)
#define PG_MINOR_VERSION		(PG_VERSION_NUM % 100)

#include "access/brin.h"
#include "access/heapam.h"
#include "access/genam.h"
#include "access/relscan.h"
#include "access/syncscan.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/visibilitymap.h"
#include "access/xact.h"
#include "catalog/binary_upgrade.h"
#include "catalog/dependency.h"
#include "catalog/indexing.h"
#include "catalog/objectaccess.h"
#include "catalog/pg_am.h"
#include "catalog/pg_cast.h"
#include "catalog/pg_depend.h"
#include "catalog/pg_extension.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_statistic.h"
#include "catalog/pg_tablespace_d.h"
#include "catalog/pg_type.h"
#include "commands/defrem.h"
#include "commands/extension.h"
#include "commands/tablespace.h"
#include "commands/typecmds.h"
#include "common/hashfn.h"
#include "common/int.h"
#include "executor/nodeSubplan.h"
#include "funcapi.h"
#include "libpq/pqformat.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "nodes/pathnodes.h"
#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/planner.h"
#include "optimizer/restrictinfo.h"
#include "postmaster/bgworker.h"
#include "postmaster/postmaster.h"
#include "storage/bufmgr.h"
#include "storage/buf_internals.h"
#include "storage/ipc.h"
#include "storage/fd.h"
#include "storage/latch.h"
#include "storage/pmsignal.h"
#include "storage/shmem.h"
#include "storage/smgr.h"
#include "utils/builtins.h"
#include "utils/cash.h"
#include "utils/date.h"
#include "utils/datetime.h"
#include "utils/float.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "utils/inet.h"
#include "utils/inval.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/pg_locale.h"
#include "utils/rangetypes.h"
#include "utils/regproc.h"
#include "utils/rel.h"
#include "utils/resowner.h"
#include "utils/ruleutils.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"
#include "utils/syscache.h"
#include "utils/timestamp.h"
#include "utils/tuplestore.h"
#include "utils/typcache.h"
#include "utils/uuid.h"
#include "utils/wait_event.h"
#include <assert.h>
#define CUDA_API_PER_THREAD_DEFAULT_STREAM		1
#include <cuda.h>
#include <float.h>
#include <limits.h>
#include <pthread.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>
#include "xpu_common.h"
#include "pg_utils.h"
#include "heterodb_extra.h"

/* ------------------------------------------------
 *
 * Global Type Definitions
 *
 * ------------------------------------------------
 */
typedef struct GpuDevAttributes
{
	int32		NUMA_NODE_ID;
	int32		DEV_ID;
	char		DEV_NAME[256];
	char		DEV_UUID[sizeof(CUuuid)];
	size_t		DEV_TOTAL_MEMSZ;
	size_t		DEV_BAR1_MEMSZ;
	bool		DEV_SUPPORT_GPUDIRECTSQL;
#define DEV_ATTR(LABEL,a,b,c)	\
	int32		LABEL;
#include "gpu_devattrs.h"
#undef DEV_ATTR
} GpuDevAttributes;

extern GpuDevAttributes *gpuDevAttrs;
extern int		numGpuDevAttrs;
#define GPUKERNEL_MAX_SM_MULTIPLICITY	4

/*
 * devtype/devfunc/devcast definitions
 */
struct devtype_info;
struct devfunc_info;
struct devcast_info;

typedef uint32_t (*devtype_hashfunc_f)(bool isnull, Datum value);

typedef struct devtype_info
{
	uint32_t	hash;
	TypeOpCode	type_code;
	Oid			type_oid;
	uint64_t	type_flags;
	int16		type_length;
	int16		type_align;
	bool		type_byval;
	bool		type_is_negative;
	const char *type_name;
	const char *type_extension;
	int			type_sizeof;	/* sizeof(xpu_NAME_t) */
	devtype_hashfunc_f type_hashfunc;
	/* oid of type related functions */
	Oid			type_eqfunc;
	Oid			type_cmpfunc;
	/* alias type, if any */
	struct devtype_info *type_alias;
	/* element type of array, if type is array */
	struct devtype_info *type_element;
	/* attribute of sub-fields, if type is composite */
	int			comp_nfields;
	struct devtype_info *comp_subtypes[1];
} devtype_info;

typedef struct devfunc_info
{
	dlist_node	chain;
	uint32_t	hash;
	FuncOpCode	func_code;
	const char *func_extension;
	const char *func_name;
	Oid			func_oid;
	struct devtype_info *func_rettype;
	uint64_t	func_flags;
	int			func_cost;
	bool		func_is_negative;
	int			func_nargs;
	struct devtype_info *func_argtypes[1];
} devfunc_info;

typedef struct XpuConnection	XpuConnection;
typedef struct GpuCacheState	GpuCacheState;
typedef struct GpuDirectState	GpuDirectState;
typedef struct DpuStorageEntry	DpuStorageEntry;
typedef struct ArrowFdwState	ArrowFdwState;
typedef struct BrinIndexState	BrinIndexState;

typedef struct
{
	/* statistics */
	pg_atomic_uint64	ntuples_valid;
	pg_atomic_uint64	ntuples_dropped;
	/* for arrow_fdw */
	pg_atomic_uint32	af_rbatch_index;
	pg_atomic_uint32	af_rbatch_nload;	/* # of loaded record-batches */
	pg_atomic_uint32	af_rbatch_nskip;	/* # of skipped record-batches */
	/* for gpu-cache */
	pg_atomic_uint32	gc_fetch_count;
	/* common block-based table scan descriptor */
	ParallelBlockTableScanDescData bpscan;
} pgstromSharedState;

struct pgstromTaskState
{
	CustomScanState		css;
	XpuConnection	   *conn;
	pgstromSharedState *ps_state;
	GpuCacheState	   *gc_state;
	GpuDirectState	   *gd_state;
	ArrowFdwState	   *af_state;
	BrinIndexState	   *br_state;
	DpuStorageEntry	   *ds_entry;
	/* current chunk (already processed by the device) */
	XpuCommand		   *curr_resp;
	HeapTupleData		curr_htup;
	kern_data_store	   *curr_kds;
	int					curr_chunk;
	int64_t				curr_index;
	bool				scan_done;
	bool				final_done;
	/* base relation scan, if any */
	TupleTableSlot	   *base_slot;
	ExprState		   *base_quals;	/* equivalent to device quals */
	ProjectionInfo	   *base_proj;	/* base --> custom_tlist projection */
	Tuplestorestate	   *fallback_store; /* tuples processed by CPU-fallback */
	/* request command buffer (+ status for table scan) */
	TBMIterateResult   *curr_tbm;
	Buffer				curr_vm_buffer;		/* for visibility-map */
	BlockNumber			curr_block_num;		/* for KDS_FORMAT_BLOCK */
	BlockNumber			curr_block_tail;	/* for KDS_FORMAT_BLOCK */
	StringInfoData		xcmd_buf;
	/* callbacks */
	TupleTableSlot	 *(*cb_next_tuple)(struct pgstromTaskState *pts);
	XpuCommand		 *(*cb_next_chunk)(struct pgstromTaskState *pts,
									   struct iovec *xcmd_iov, int *xcmd_iovcnt);
	XpuCommand		 *(*cb_final_chunk)(struct pgstromTaskState *pts,
										struct iovec *xcmd_iov, int *xcmd_iovcnt);
	void			  (*cb_cpu_fallback)(struct pgstromTaskState *pts,
										 HeapTuple htuple);
};
typedef struct pgstromTaskState		pgstromTaskState;

/*
 * Global variables
 */
extern long		PAGE_SIZE;
extern long		PAGE_MASK;
extern int		PAGE_SHIFT;
extern long		PHYS_PAGES;
extern long		PAGES_PER_BLOCK;	/* (BLCKSZ / PAGE_SIZE) */
#define PAGE_ALIGN(x)			TYPEALIGN(PAGE_SIZE,(x))
#define PGSTROM_CHUNK_SIZE		((size_t)(65534UL << 10))

/*
 * extra.c
 */
extern void		pgstrom_init_extra(void);
extern bool		heterodbValidateDevice(int gpu_device_id,
									   const char *gpu_device_name,
									   const char *gpu_device_uuid);
extern int		gpuDirectInitDriver(void);
extern bool		gpuDirectFileDescOpen(GPUDirectFileDesc *gds_fdesc,
									  File pg_fdesc);
extern bool		gpuDirectFileDescOpenByPath(GPUDirectFileDesc *gds_fdesc,
											const char *pathname);
extern void		gpuDirectFileDescClose(const GPUDirectFileDesc *gds_fdesc);
extern CUresult	gpuDirectMapGpuMemory(CUdeviceptr m_segment,
									  size_t m_segment_sz,
									  unsigned long *p_iomap_handle);
extern CUresult	gpuDirectUnmapGpuMemory(CUdeviceptr m_segment,
										unsigned long iomap_handle);
extern bool		gpuDirectFileReadIOV(const GPUDirectFileDesc *gds_fdesc,
									 CUdeviceptr m_segment,
									 unsigned long iomap_handle,
									 off_t m_offset,
									 strom_io_vector *iovec);
extern void		extraSysfsSetupDistanceMap(const char *manual_config);
extern Bitmapset *extraSysfsLookupOptimalGpus(int fdesc);
extern ssize_t	extraSysfsPrintNvmeInfo(int index, char *buffer, ssize_t buffer_sz);

/*
 * codegen.c
 */
extern devtype_info *pgstrom_devtype_lookup(Oid type_oid);
extern devfunc_info *pgstrom_devfunc_lookup(Oid func_oid,
											List *func_args,
											Oid func_collid);
extern void		pgstrom_build_xpucode(bytea **p_xpucode,
									  Expr *expr,
									  List *input_rels_tlist,
									  uint32_t *p_extra_flags,
									  uint32_t *p_extra_bufsz,
									  uint32_t *p_kvars_nslots,
									  List **p_used_params);
extern void		pgstrom_build_projection(bytea **p_xpucode_proj,
										 List *tlist_dev,
										 List *input_rels_tlist,
										 uint32_t *p_extra_flags,
										 uint32_t *p_extra_bufsz,
										 uint32_t *p_kvars_nslots,
										 List **p_used_params);
extern bool		pgstrom_gpu_expression(Expr *expr,
									   List *input_rels_tlist,
									   int *p_devcost);
extern bool		pgstrom_dpu_expression(Expr *expr,
									   List *input_rels_tlist,
									   int *p_devcost);

extern void		pgstrom_explain_xpucode(StringInfo buf,
										bytea *xpu_code,
										const CustomScanState *css,
										ExplainState *es,
										List *ancestors);
extern char	   *pgstrom_xpucode_to_string(bytea *xpu_code);
extern void		pgstrom_init_codegen(void);

/*
 * brin.c
 */
extern IndexOptInfo *pgstromTryFindBrinIndex(PlannerInfo *root,
											 RelOptInfo *baserel,
											 List **p_indexConds,
											 List **p_indexQuals,
											 int64_t *p_indexNBlocks);
extern Cost		cost_brin_bitmap_build(PlannerInfo *root,
									   RelOptInfo *baserel,
									   IndexOptInfo *indexOpt,
									   List *indexQuals);

extern void		pgstromBrinIndexExecBegin(pgstromTaskState *pts,
										  Oid index_oid,
										  List *index_conds,
										  List *index_quals);
extern bool		pgstromBrinIndexNextChunk(pgstromTaskState *pts);
extern TBMIterateResult *pgstromBrinIndexNextBlock(pgstromTaskState *pts);
extern void		pgstromBrinIndexExecEnd(pgstromTaskState *pts);
extern void		pgstromBrinIndexExecReset(pgstromTaskState *pts);
extern Size		pgstromBrinIndexEstimateDSM(pgstromTaskState *pts);
extern Size		pgstromBrinIndexInitDSM(pgstromTaskState *pts, char *dsm_addr);
extern void		pgstromBrinIndexReInitDSM(pgstromTaskState *pts);
extern Size		pgstromBrinIndexAttachDSM(pgstromTaskState *pts, char *dsm_addr);
extern void		pgstromBrinIndexShutdownDSM(pgstromTaskState *pts);
extern void		pgstrom_init_brin(void);

/*
 * relscan.c
 */
extern Bitmapset *pickup_outer_referenced(PlannerInfo *root,
										  RelOptInfo *base_rel,
										  Bitmapset *referenced);
extern size_t	estimate_kern_data_store(TupleDesc tupdesc);
extern size_t	setup_kern_data_store(kern_data_store *kds,
									  TupleDesc tupdesc,
									  size_t length,
									  char format);
extern XpuCommand *pgstromRelScanChunkDirect(pgstromTaskState *pts,
											 struct iovec *xcmd_iov,
											 int *xcmd_iovcnt);
extern XpuCommand *pgstromRelScanChunkNormal(pgstromTaskState *pts,
											 struct iovec *xcmd_iov,
											 int *xcmd_iovcnt);
extern Size		pgstromSharedStateEstimateDSM(pgstromTaskState *pts);
extern void		pgstromSharedSteteCreate(pgstromTaskState *pts);
extern void		pgstromSharedStateInitDSM(pgstromTaskState *pts, char *dsm_addr);
extern void		pgstromSharedStateReInitDSM(pgstromTaskState *pts);
extern void		pgstromSharedStateAttachDSM(pgstromTaskState *pts, char *dsm_addr);
extern void		pgstromSharedStateShutdownDSM(pgstromTaskState *pts);
extern void		pgstrom_init_relscan(void);

/*
 * executor.c
 */
extern void		__xpuClientOpenSession(pgstromTaskState *pts,
									   const XpuCommand *session,
									   pgsocket sockfd,
									   const char *devname);
extern int
xpuConnectReceiveCommands(pgsocket sockfd,
						  void *(*alloc_f)(void *priv, size_t sz),
						  void  (*attach_f)(void *priv, XpuCommand *xcmd),
						  void *priv,
						  const char *error_label);
extern void		xpuClientCloseSession(XpuConnection *conn);
extern void		xpuClientSendCommand(XpuConnection *conn, const XpuCommand *xcmd);
extern void		xpuClientPutResponse(XpuCommand *xcmd);

extern const XpuCommand *
pgstromBuildSessionInfo(PlanState *ps,
						List *used_params,
						uint32_t num_cached_kvars,
						uint32_t kcxt_extra_bufsz,
						const bytea *xpucode_scan_quals,
						const bytea *xpucode_scan_projs);
extern void		pgstromExecInitTaskState(pgstromTaskState *pts,
										 List *outer_dev_quals);
extern TupleTableSlot  *pgstromExecTaskState(pgstromTaskState *pts);
extern void		pgstromExecEndTaskState(pgstromTaskState *pts);
extern void		pgstromExecResetTaskState(pgstromTaskState *pts);
extern void		pgstrom_init_executor(void);

/*
 * gpu_device.c
 */
extern double	pgstrom_gpu_setup_cost;		/* GUC */
extern double	pgstrom_gpu_dma_cost;		/* GUC */
extern double	pgstrom_gpu_operator_cost;	/* GUC */
extern double	pgstrom_gpu_direct_seq_page_cost; /* GUC */
extern void		gpuClientOpenSession(pgstromTaskState *pts,
									 const Bitmapset *gpuset,
									 const XpuCommand *session);
extern CUresult	gpuOptimalBlockSize(int *p_grid_sz,
									int *p_block_sz,
									unsigned int *p_shmem_sz,
									CUfunction kern_function,
									size_t dynamic_shmem_per_block,
									size_t dynamic_shmem_per_warp,
									size_t dynamic_shmem_per_thread);
extern bool		pgstrom_init_gpu_device(void);

/*
 * gpu_service.c
 */
struct gpuClient
{
	struct gpuContext *gcontext;/* per-device status */
	dlist_node		chain;		/* gcontext->client_list */
	CUmodule		cuda_module;/* preload cuda binary */
	kern_session_info *session;	/* per session info (on cuda managed memory) */
	pg_atomic_uint32 refcnt;	/* odd number, if error status */
	pthread_mutex_t	mutex;		/* mutex to write the socket */
	int				sockfd;		/* connection to PG backend */
	pthread_t		worker;		/* receiver thread */
};
typedef struct gpuClient	gpuClient;

extern int		pgstrom_max_async_gpu_tasks;	/* GUC */
extern bool		pgstrom_load_gpu_debug_module;	/* GUC */
extern const char *cuStrError(CUresult rc);
extern void		__gpuClientELogRaw(gpuClient *gclient,
								   kern_errorbuf *errorbuf);
extern void		__gpuClientELog(gpuClient *gclient,
								int errcode,
								const char *filename, int lineno,
								const char *funcname,
								const char *fmt, ...);
#define gpuClientELog(gclient,fmt,...)						\
	__gpuClientELog((gclient), ERRCODE_DEVICE_INTERNAL,		\
					__FILE__, __LINE__, __FUNCTION__,		\
					(fmt), ##__VA_ARGS__)
#define gpuClientFatal(gclient,fmt,...)						\
	__gpuClientELog((gclient), ERRCODE_DEVICE_FATAL,		\
					__FILE__, __LINE__, __FUNCTION__,		\
					(fmt), ##__VA_ARGS__)

extern __thread int			CU_DINDEX_PER_THREAD;
extern __thread CUdevice	CU_DEVICE_PER_THREAD;
extern __thread CUcontext	CU_CONTEXT_PER_THREAD;
extern __thread CUevent		CU_EVENT_PER_THREAD;

typedef struct
{
	CUdeviceptr	base;
	size_t		offset;
	size_t		length;
	unsigned long iomap_handle;		/* for old nvme_strom kmod */
} gpuMemChunk;

extern const gpuMemChunk *gpuMemAlloc(size_t bytesize);
extern void		gpuMemFree(const gpuMemChunk *chunk);
extern const gpuMemChunk *gpuservLoadKdsBlock(gpuClient *gclient,
											  kern_data_store *kds,
											  const char *pathname,
											  strom_io_vector *kds_iovec);
extern bool		gpuServiceGoingTerminate(void);
extern void		gpuClientWriteBack(gpuClient *gclient,
								   XpuCommand *resp,
								   size_t resp_sz,
								   int kds_nitems,
								   kern_data_store **kds_array);
extern void		pgstrom_init_gpu_service(void);

/*
 * gpu_direct.c
 */
extern const Bitmapset *baseRelCanUseGpuDirect(PlannerInfo *root,
											   RelOptInfo *baserel);
extern void		pgstromGpuDirectExecBegin(pgstromTaskState *pts,
										  const Bitmapset *gpuset);
extern const Bitmapset *pgstromGpuDirectDevices(pgstromTaskState *pts);
extern void		pgstromGpuDirectExecEnd(pgstromTaskState *pts);
extern void		pgstrom_init_gpu_direct(void);



/*
 * gpu_cache.c
 */





/*
 * gpu_scan.c
 */
typedef struct
{
	const Bitmapset *gpu_cache_devs; /* device for GpuCache, if any */
	const Bitmapset *gpu_direct_devs; /* device for GPU-Direct SQL, if any */
	bytea	   *kern_quals;		/* device qualifiers */
	bytea	   *kern_projs;		/* device projection */
	uint32_t	extra_flags;
	uint32_t	extra_bufsz;
	uint32_t	kvars_nslots;
	const Bitmapset *outer_refs; /* referenced columns */
	List	   *used_params;	/* Param list in use */
	List	   *dev_quals;		/* Device qualifiers */
	Oid			index_oid;		/* OID of BRIN-index, if any */
	List	   *index_conds;	/* BRIN-index key conditions */
	List	   *index_quals;	/* Original BRIN-index qualifier*/
} GpuScanInfo;
extern void		form_gpuscan_info(CustomScan *cscan, GpuScanInfo *gs_info);
extern void		deform_gpuscan_info(GpuScanInfo *gs_info, CustomScan *cscan);

extern CustomScan *PlanXpuScanPathCommon(PlannerInfo *root,
										 RelOptInfo  *baserel,
										 CustomPath  *best_path,
										 List        *tlist,
										 List        *clauses,
										 GpuScanInfo *gs_info,
										 const CustomScanMethods *methods);
extern void		ExecFallbackCpuScan(pgstromTaskState *pts, HeapTuple tuple);
extern void		gpuservHandleGpuScanExec(gpuClient *gclient, XpuCommand *xcmd);
extern void		pgstrom_init_gpu_scan(void);


/*
 * apache arrow related stuff
 */





/*
 * dpu_device.c
 */
extern double	pgstrom_dpu_setup_cost;
extern double	pgstrom_dpu_operator_cost;
extern double	pgstrom_dpu_seq_page_cost;
extern double	pgstrom_dpu_tuple_cost;
extern bool		pgstrom_dpu_handle_cached_pages;

extern DpuStorageEntry *GetOptimalDpuForTablespace(Oid tablespace_oid);
extern DpuStorageEntry *GetOptimalDpuForRelation(Relation relation);
extern void		DpuClientOpenSession(pgstromTaskState *pts,
									 const XpuCommand *session);
extern bool		pgstrom_init_dpu_device(void);

/*
 * dpu_scan.c
 */
typedef GpuScanInfo		DpuScanInfo;
#define form_dpuscan_info(a,b)		form_gpuscan_info((a),(b))
#define deform_dpuscan_info(a,b)	deform_gpuscan_info((a),(b))

extern void		pgstrom_init_dpu_scan(void);

/*
 * misc.c
 */
extern Node	   *fixup_varnode_to_origin(Node *node, List *cscan_tlist);
extern int		__appendBinaryStringInfo(StringInfo buf,
										 const void *data, int datalen);
extern int		__appendZeroStringInfo(StringInfo buf, int nbytes);
extern char	   *get_type_name(Oid type_oid, bool missing_ok);
extern Oid		get_relation_am(Oid rel_oid, bool missing_ok);
extern List	   *bms_to_pglist(const Bitmapset *bms);
extern Bitmapset *bms_from_pglist(List *pglist);
extern ssize_t	__readFile(int fdesc, void *buffer, size_t nbytes);
extern ssize_t	__preadFile(int fdesc, void *buffer, size_t nbytes, off_t f_pos);
extern ssize_t	__writeFile(int fdesc, const void *buffer, size_t nbytes);
extern ssize_t	__pwriteFile(int fdesc, const void *buffer, size_t nbytes, off_t f_pos);

extern void	   *__mmapFile(void *addr, size_t length,
						   int prot, int flags, int fdesc, off_t offset);
extern bool		__munmapFile(void *mmap_addr);
extern void	   *__mremapFile(void *mmap_addr, size_t new_size);
extern void	   *__mmapShmem(size_t length);
extern Path	   *pgstrom_copy_pathnode(const Path *pathnode);

/*
 * main.c
 */
extern bool		pgstrom_enabled;
extern bool		pgstrom_cpu_fallback_enabled;
extern bool		pgstrom_regression_test_mode;
extern int		pgstrom_max_async_tasks;
extern const CustomPath *custom_path_find_cheapest(PlannerInfo *root,
												   RelOptInfo *rel,
												   bool outer_parallel,
												   bool inner_parallel,
												   const char *custom_name);
extern bool		custom_path_remember(PlannerInfo *root,
									 RelOptInfo *rel,
									 bool outer_parallel,
									 bool inner_parallel,
									 const CustomPath *cpath);
extern void		_PG_init(void);

#endif	/* PG_STROM_H */
