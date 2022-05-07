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

#include "access/genam.h"
#include "access/table.h"
#include "catalog/binary_upgrade.h"
#include "catalog/dependency.h"
#include "catalog/indexing.h"
#include "catalog/objectaccess.h"
#include "catalog/pg_depend.h"
#include "catalog/pg_extension.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "commands/extension.h"
#include "commands/typecmds.h"
#include "common/hashfn.h"
#include "common/int.h"
#include "funcapi.h"
#include "libpq/pqformat.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "postmaster/postmaster.h"
#include "storage/ipc.h"
#include "storage/fd.h"
#include "storage/shmem.h"
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
#include "utils/syscache.h"
#include "utils/timestamp.h"
#include "utils/typcache.h"
#include "utils/uuid.h"
#include <assert.h>
#include <cuda.h>
#include <limits.h>
#include <sys/mman.h>
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
	char		DEV_BRAND[16];
	char		DEV_UUID[48];
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
#define DEVKERN__NVIDIA_GPU			0x0001U		/* CUDA-based GPU */
#define DEVKERN__NVIDIA_DPU			0x0002U		/* BlueField-X DPU */
#define DEVKERN__ARMv8_SPU			0x0004U		/* ARMv8-based SPU */
#define DEVKERN__ANY				0x0007U		/* Runnable on xPU */
#define DEVFUNC__LOCALE_AWARE		0x0100U		/* Device function is locale aware,
												 * thus, available only if "C" or
												 * no locale configuration */
struct devtype_info;
struct devfunc_info;
struct devcast_info;

typedef uint32_t (*devtype_hashfunc_f)(bool isnull, Datum value);

typedef struct devtype_info
{
	dlist_node	chain;
	uint32_t	hash;
	TypeOpCode	type_code;
	Oid			type_oid;
	uint32		type_flags;
	int16		type_length;
	int16		type_align;
	bool		type_byval;
	bool		type_is_negative;
	const char *type_name;
	const char *type_extension;
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
	uint32_t	func_flags;
	bool		func_is_negative;
	int			func_nargs;
	struct devtype_info *func_argtypes[1];
} devfunc_info;








/*
 * Global variables
 */
extern long		PAGE_SIZE;
extern long		PAGE_MASK;
extern int		PAGE_SHIFT;
extern long		PHYS_PAGES;
#define PAGE_ALIGN(x)	TYPEALIGN(PAGE_SIZE,(x))

/*
 * extra.c
 */
extern void		pgstrom_init_extra(void);
extern int		gpuDirectInitDriver(void);
extern void		gpuDirectFileDescOpen(GPUDirectFileDesc *gds_fdesc,
									  File pg_fdesc);
extern void		gpuDirectFileDescOpenByPath(GPUDirectFileDesc *gds_fdesc,
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
 * shmbuf.c
 */
extern void	   *shmbufAlloc(size_t sz);
extern void	   *shmbufAllocZero(size_t sz);
extern void		shmbufFree(void *addr);
extern void		pgstrom_init_shmbuf(void);

/*
 * codegen.c
 */
extern devtype_info *pgstrom_devtype_lookup(Oid type_oid);
extern devfunc_info *pgstrom_devfunc_lookup(Oid func_oid,
											List *func_args,
											Oid func_collid);
extern Const   *pgstrom_codegen_expression(Expr *expr,
										   List **p_used_params,
										   List **p_used_vars,
										   uint32_t *p_extra_flags,
										   uint32_t *p_extra_bufsz,
										   int num_rels,
										   List **rel_tlist);
extern bool		pgstrom_gpu_expression(Expr *expr);
extern void		pgstrom_init_codegen(void);

/*
 * gpu_device.c
 */
extern bool		pgstrom_gpudirect_enabled(void);
extern Size		pgstrom_gpudirect_threshold(void);
extern CUresult	gpuOptimalBlockSize(int *p_grid_sz,
									int *p_block_sz,
									CUfunction kern_function,
									CUdevice cuda_device,
									size_t dyn_shmem_per_block,
									size_t dyn_shmem_per_thread);
extern bool		pgstrom_init_gpu_device(void);

/*
 * apache arrow related stuff
 */


/*
 * misc.c
 */
extern char	   *get_type_name(Oid type_oid, bool missing_ok);
extern List	   *bms_to_pglist(const Bitmapset *bms);
extern Bitmapset *bms_from_pglist(List *pglist);
extern ssize_t	__readFile(int fdesc, void *buffer, size_t nbytes);
extern ssize_t	__preadFile(int fdesc, void *buffer, size_t nbytes, off_t f_pos);
extern ssize_t	__writeFile(int fdesc, const void *buffer, size_t nbytes);
extern ssize_t	__pwriteFile(int fdesc, const void *buffer, size_t nbytes, off_t f_pos);
extern void	   *__mmapFile(void *addr, size_t length,
						   int prot, int flags, int fdesc, off_t offset);
extern int		__munmapFile(void *mmap_addr);
extern void	   *__mremapFile(void *mmap_addr, size_t new_size);

/*
 * main.c
 */
extern void		_PG_init(void);


#endif	/* PG_STROM_H */
