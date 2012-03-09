/*
 * pg_strom.h
 *
 * Header file of pg_strom module
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#ifndef PG_STROM_H
#define PG_STROM_H
#include "commands/explain.h"
#include "storage/shmem.h"
#include "foreign/fdwapi.h"
#include "utils/relcache.h"
#include <pthread.h>
#include <semaphore.h>

/*
 * Schema of shadow tables
 */
#define PGSTROM_CHUNK_SIZE		(BLCKSZ / 2)

#define PGSTROM_SCHEMA_NAME		"pg_strom"

#define Natts_pg_strom_rmap			3
#define Anum_pg_strom_rmap_rowid	1
#define Anum_pg_strom_rmap_nitems	2
#define Anum_pg_strom_rmap_rowmap	3

#define Natts_pg_strom_cs			4
#define Anum_pg_strom_cs_rowid		1
#define	Anum_pg_strom_cs_nitems		2
#define Anum_pg_strom_cs_isnull		3
#define Anum_pg_strom_cs_values		4

/*
 * Data Structures
 */
typedef struct ShmsegList {
	struct ShmsegList  *prev;
	struct ShmsegList  *next;
} ShmsegList; 

typedef struct {
	ShmsegList		qhead;
	pthread_mutex_t	qlock;
	sem_t			qsem;
} ShmsegQueue;

#define CHUNKBUF_STATUS_FREE		1
#define CHUNKBUF_STATUS_EXEC		2
#define CHUNKBUF_STATUS_READY		3
#define CHUNKBUF_STATUS_ERROR		4

typedef struct {
	ShmsegList		chain;		/* dual-linked list to be chained */
	ShmsegQueue	   *recv_cmdq;	/* reference to the response queue */
	Const		   *gpu_cmds;	/* command sequence of OpenCL */
	Const		   *cpu_cmds;	/* command sequence of OpenMP */
	int				status;		/* status of this chunk-buffer */
	int				nattrs;		/* number of columns */
	int64			rowid;		/* the first row-id of this chunk */
	int				nitems;		/* number of rows */
	Size			dma_length;	/* length to be translated with DMA */
	bits8		   *cs_rowmap;	/* rowmap of the column store */
	int			   *cs_isnull;	/* offsets from the cs_rowmap, or 0 */
	int			   *cs_values;	/* offsets from the cs_rowmap, or 0 */
} ChunkBuffer;

#define chunk_cs_isnull(chunk, attno)			\
	((bits8 *)((chunk)->cs_rowmap + (chunk)->cs_isnull[(attno) - 1]))
#define chunk_cs_values(chunk, attno)			\
	((char *)((chunk)->cs_rowmap + (chunk)->cs_values[(attno) - 1]))

typedef struct {
	Oid			type_oid;
	bool		type_x2regs;	/* type uses dual 32bit registers? */
	bool		type_fp64;		/* type needs double */
	uint32		type_varref;	/* cmd code to reference type variable */
	uint32		type_conref;	/* cmd code to reference type constant */
} GpuTypeInfo;

typedef struct {
	Oid			func_oid;
	uint32		func_cmd;		/* cmd code of this function */
	uint16		func_nargs;		/* number of arguments */
	Oid			func_rettype;		/* return type of function */
	Oid			func_argtypes[0];	/* argument types of function */
} GpuFuncInfo;

/*
 * shmseg.c
 */
#define container_of(ptr, type, field)						\
	((void *)((uintptr_t)(ptr) - offsetof(type, field)))

#define pgstrom_shmseg_list_foreach(item, field, list)					\
	for (item = container_of((list)->next, typeof(*item), field);		\
		 &item->field != (list);										\
		 item = container_of(item->field.next, typeof(*item), field))

#define pgstrom_shmseg_list_foreach_safe(item, temp, field, list)		\
	for (item = container_of((list)->next, typeof(*item), field),		\
		 temp = container_of(item->field.next, typeof(*item), field);	\
		 &item->field != (list);										\
		 item = temp,													\
		 temp = container_of(temp->field.next, typeof(*temp), field))

extern void	pgstrom_shmseg_list_init(ShmsegList *list);
extern bool	pgstrom_shmseg_list_empty(ShmsegList *list);
extern void pgstrom_shmseg_list_delete(ShmsegList *list);
extern void pgstrom_shmseg_list_add(ShmsegList *list, ShmsegList *item);

extern bool	pgstrom_shmqueue_init(ShmsegQueue *shmq);
extern void	pgstrom_shmqueue_destroy(ShmsegQueue *shmq);
extern int	pgstrom_shmqueue_nitems(ShmsegQueue *shmq);
extern void	pgstrom_shmqueue_enqueue(ShmsegQueue *shmq, ShmsegList *item);
extern ShmsegList *pgstrom_shmqueue_dequeue(ShmsegQueue *shmq);
extern ShmsegList *pgstrom_shmqueue_trydequeue(ShmsegQueue *shmq);

extern void	   *pgstrom_shmseg_alloc(Size size);
extern void		pgstrom_shmseg_free(void *ptr);
extern void		pgstrom_shmseg_init(void);

/*
 * opencl_catalog.c
 */
extern GpuTypeInfo  *pgstrom_gpu_type_lookup(Oid typeOid);
extern GpuFuncInfo  *pgstrom_gpu_func_lookup(Oid funcOid);
extern int	pgstrom_gpu_command_string(Oid ftableOid, int cmds[],
									   char *buf, size_t buflen);

/*
 * opencl_serv.c
 */
extern void	pgstrom_opencl_init(void);
extern void	pgstrom_opencl_startup(void *shmptr, Size shmsize);
extern void pgstrom_opencl_enqueue_chunk(ChunkBuffer *chunk);
extern int	pgstrom_opencl_num_devices(void);
extern bool	pgstrom_opencl_fp64_supported(void);

/*
 * openmp_serv.c
 */
extern void pgstrom_openmp_enqueue_chunk(ChunkBuffer *chunk);

/*
 * plan.c
 */
extern FdwPlan *pgstrom_plan_foreign_scan(Oid ftableOid,
										  PlannerInfo *root,
										  RelOptInfo *baserel);
extern void		pgstrom_explain_foreign_scan(ForeignScanState *fss,
											 ExplainState *es);

/*
 * exec.c
 */
extern void pgstrom_begin_foreign_scan(ForeignScanState *fss, int eflags);
extern TupleTableSlot *pgstrom_iterate_foreign_scan(ForeignScanState *fss);
extern void pgstrom_rescan_foreign_scan(ForeignScanState *fss);
extern void pgstrom_end_foreign_scan(ForeignScanState *fss);
extern void pgstrom_executor_init(void);

/*
 * utilcmds.c
 */
extern Relation pgstrom_open_rowid_map(Relation base, LOCKMODE lockmode);
extern Relation pgstrom_open_cs_table(Relation base, AttrNumber attno,
									  LOCKMODE lockmode);
extern Relation pgstrom_open_cs_index(Relation base, AttrNumber attno,
									  LOCKMODE lockmode);
extern RangeVar *pgstrom_lookup_sequence(Relation base);
extern void		pgstrom_utilcmds_init(void);

/*
 * blkload.c
 */
extern Datum pgstrom_data_load(PG_FUNCTION_ARGS);
extern Datum pgstrom_data_clear(PG_FUNCTION_ARGS);
extern Datum pgstrom_data_compaction(PG_FUNCTION_ARGS);

/*
 * main.c
 */
extern FdwRoutine PgStromFdwHandlerData;
extern Datum pgstrom_fdw_handler(PG_FUNCTION_ARGS);




#if 0
#include "access/tuptoaster.h"
#include "catalog/indexing.h"
#include "commands/explain.h"
#include "fmgr.h"
#include "foreign/fdwapi.h"
#include "utils/memutils.h"
#include <cuda.h>

#define PGSTROM_CHUNK_SIZE		(MaximumBytesPerTuple(1) * BITS_PER_BYTE)

#define PGSTROM_SCHEMA_NAME		"pg_strom"
#define Natts_pg_strom			4
#define Anum_pg_strom_rowid		1
#define Anum_pg_strom_nitems	2
#define Anum_pg_strom_isnull	3
#define Anum_pg_strom_values	4

/*
 * utilcmds.c
 */
extern Relation pgstrom_open_shadow_table(Relation base_rel,
										  AttrNumber attnum,
										  LOCKMODE lockmode);
extern Relation pgstrom_open_shadow_index(Relation base_rel,
										  AttrNumber attnum,
										  LOCKMODE lockmode);
extern RangeVar *pgstrom_lookup_shadow_sequence(Relation base_rel);

extern void	pgstrom_utilcmds_init(void);

/*
 * blkload.c
 */
extern Datum pgstrom_data_load(PG_FUNCTION_ARGS);
extern Datum pgstrom_data_clear(PG_FUNCTION_ARGS);
extern Datum pgstrom_data_compaction(PG_FUNCTION_ARGS);

/*
 * plan.c
 */
extern FdwPlan *pgstrom_plan_foreign_scan(Oid foreignTblOid,
										  PlannerInfo *root,
										  RelOptInfo *baserel);
extern void		pgstrom_explain_foreign_scan(ForeignScanState *node,
											 ExplainState *es);

/*
 * scan.c
 */
extern void	pgstrom_begin_foreign_scan(ForeignScanState *fss, int eflags);
extern TupleTableSlot *pgstrom_iterate_foreign_scan(ForeignScanState *fss);
extern void	pgstrom_rescan_foreign_scan(ForeignScanState *fss);
extern void	pgstrom_end_foreign_scan(ForeignScanState *fss);
extern List *pgstrom_scan_debug_info(List *debug_info_list);
extern void pgstrom_scan_init(void);

/*
 * devinfo.c
 */
typedef struct {
	Oid		type_oid;
	char   *type_ident;
	char   *type_source;
	char   *type_varref;
	char   *type_conref;
	uint32	type_flags;			/* set of DEVINFO_FLAGS_* */
} PgStromDevTypeInfo;

typedef struct {
	Oid		func_oid;
	char	func_kind;			/* see the comments in devinfo.c */
	char   *func_ident;			/* identifier of device function */
	char   *func_source;		/* definition of device function, if needed */
	uint32	func_flags;			/* set of DEVINFO_FLAGS_* */
	int16	func_nargs;			/* copy from pg_proc.pronargs */
	Oid		func_argtypes[0];	/* copy from pg_proc.proargtypes */
} PgStromDevFuncInfo;

#define DEVINFO_FLAGS_DOUBLE_FP				0x0001
#define DEVINFO_FLAGS_INC_MATHFUNC_H		0x0002

extern PgStromDevTypeInfo *pgstrom_devtype_lookup(Oid type_oid);
extern PgStromDevFuncInfo *pgstrom_devfunc_lookup(Oid func_oid);

/*
 * PgStromDeviceInfo
 *
 * Properties of GPU device. Every fields are initialized at server starting
 * up time; except for device and context. CUDA context shall be switched
 * via pgstrom_set_device_context().
 */
typedef struct {
	CUdevice	device;
	CUcontext	context;
	char		dev_name[256];
	int			dev_major;
	int			dev_minor;
	int			dev_proc_nums;
	int			dev_proc_warp_sz;
	int			dev_proc_clock;
	size_t		dev_global_mem_sz;
	int			dev_global_mem_width;
	int			dev_global_mem_clock;
	int			dev_shared_mem_sz;
	int			dev_l2_cache_sz;
	int			dev_const_mem_sz;
	int			dev_max_block_dim_x;
	int			dev_max_block_dim_y;
	int			dev_max_block_dim_z;
	int			dev_max_grid_dim_x;
	int			dev_max_grid_dim_y;
	int			dev_max_grid_dim_z;
	int			dev_max_threads_per_proc;
	int			dev_max_regs_per_block;
	int			dev_integrated;
	int			dev_unified_addr;
	int			dev_can_map_hostmem;
	int			dev_concurrent_kernel;
	int			dev_concurrent_memcpy;
	int			dev_pci_busid;
	int			dev_pci_deviceid;
} PgStromDeviceInfo;

extern const PgStromDeviceInfo *pgstrom_get_device_info(int dev_index);
extern void pgstrom_set_device_context(int dev_index);
extern void pgstrom_devinfo_init(void);
extern const char *cuda_error_to_string(CUresult result);

/*
 * devsched.c
 */
extern void pgstrom_set_device_context(int dev_index);
extern int	pgstrom_get_num_devices(void);

extern Datum pgstrom_device_info(PG_FUNCTION_ARGS);
extern void pgstrom_devsched_init(void);

/*
 * nvcc.c
 */
extern void *pgstrom_nvcc_kernel_build(const char *kernel_source);
extern void pgstrom_nvcc_init(void);

/*
 * pg_strom.c
 */
extern FdwRoutine pgstromFdwHandlerData;
extern Datum pgstrom_fdw_handler(PG_FUNCTION_ARGS);
extern Datum pgstrom_fdw_validator(PG_FUNCTION_ARGS);
#endif

#endif	/* PG_STROM_H */
