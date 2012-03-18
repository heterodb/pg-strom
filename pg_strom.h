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
#define PGSTROM_CHUNK_SIZE		(BITS_PER_BYTE * BLCKSZ / 2)

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
	int		   *gpu_cmds;		/* command sequence of GPU */
	int		   *cpu_cmds;		/* command sequence of CPU */
	int			gpu_cmds_len;	/* length of GPU command sequence */
	int			cpu_cmds_len;	/* length of CPU command sequence */
	int			status;			/* CHUNKBUF_STATUS_* */
	int			nattrs;			/* number of columns in this chunk */
	int64		rowid;			/* base rowid of this chunk */
	int			nitems;			/* nitems of this chunk */
	size_t		dma_length;		/* length to be copied using DMA */
	char	   *dma_buffer;		/* base address to be copied using DMA */
	char	   *cs_rowmap;		/* rowmap of this chunk */
	int		   *cs_isnull;		/* offset from cs_rowmap, or 0 */
	int		   *cs_values;		/* offset from cs_rowmap, or 0 */
	bool		pf_enabled;		/* Is exec profiling enabled? */
	uint64		pf_async_memcpy;/* time to async memcpy */
	uint64		pf_async_kernel;/* time to async kernel exec */
	uint64		pf_queue_wait;	/* time to chunks in queue */
	struct timeval pf_timeval;	/* timestamp when chunk is queued */
	char		error_msg[256];	/* error message if CHUNKBUF_STATUS_ERROR */
} ChunkBuffer;

#define chunk_cs_isnull(chunk, attno)			\
	((bits8 *)((chunk)->cs_rowmap + (chunk)->cs_isnull[(attno) - 1]))
#define chunk_cs_values(chunk, attno)			\
	((char *)((chunk)->cs_rowmap + (chunk)->cs_values[(attno) - 1]))

#define TIMEVAL_ELAPSED(tv1,tv2)					\
	(((tv2)->tv_sec  - (tv1)->tv_sec) * 1000000 +	\
	 ((tv2)->tv_usec - (tv1)->tv_usec))

typedef struct {
	Oid			type_oid;
	const char *type_explain;	/* symbol in EXPLAIN statement */
	bool		type_x2regs;	/* type uses dual 32bit registers? */
	bool		type_fp64;		/* type needs double */
	uint32		type_varref;	/* cmd code to reference type variable */
	uint32		type_conref;	/* cmd code to reference type constant */
} GpuTypeInfo;

typedef struct {
	Oid			func_oid;
	const char *func_explain;	/* symbol in EXPLAIN statement */
	uint32		func_cmd;		/* cmd code of this function */
	uint16		func_nargs;		/* number of arguments */
	Oid			func_rettype;		/* return type of function */
	Oid			func_argtypes[0];	/* argument types of function */
} GpuFuncInfo;

typedef GpuTypeInfo	CpuTypeInfo;
typedef GpuFuncInfo	CpuFuncInfo;

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
 * cuda_serv.c
 */
extern void pgstrom_gpu_init(void);
extern void pgstrom_gpu_startup(void *shmptr, Size shmsize);
extern void pgstrom_gpu_enqueue_chunk(ChunkBuffer *chunk);
extern int	pgstrom_gpu_num_devices(void);

extern GpuTypeInfo *pgstrom_gpu_type_lookup(Oid typeOid);
extern GpuFuncInfo *pgstrom_gpu_func_lookup(Oid funcOid);
extern int	pgstrom_gpu_command_string(Oid ftableOid, int cmds[],
									   char *buf, size_t buflen);
extern Datum	pgstrom_gpu_info(PG_FUNCTION_ARGS);

/*
 * openmp_serv.c
 */
extern void pgstrom_cpu_init(void);
extern void pgstrom_cpu_startup(void *shmptr, Size shmsize);
extern void pgstrom_cpu_enqueue_chunk(ChunkBuffer *chunk);
extern CpuTypeInfo *pgstrom_cpu_type_lookup(Oid typeOid);
extern CpuFuncInfo *pgstrom_cpu_func_lookup(Oid funcOid);
extern int	pgstrom_cpu_command_string(Oid ftableOid, int cmds[],
									   char *buf, size_t buflen);

/*
 * plan.c
 */
extern void	pgstrom_get_foreign_rel_size(PlannerInfo *root,
										 RelOptInfo *baserel,
										 Oid foreigntableid);
extern void	pgstrom_get_foreign_paths(PlannerInfo *root,
									  RelOptInfo *baserel,
									  Oid foreigntableid);
extern ForeignScan *pgstrom_get_foreign_plan(PlannerInfo *root,
											 RelOptInfo *baserel,
											 Oid foreigntableid,
											 ForeignPath *best_path,
											 List *tlist,
											 List *scan_clauses);
extern void	pgstrom_explain_foreign_scan(ForeignScanState *fss,
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
extern RangeVar *pgstrom_lookup_sequence(Oid base_relid);
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

#endif	/* PG_STROM_H */
