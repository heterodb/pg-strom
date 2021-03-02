/*
 * cuda_codegen.h
 *
 * Definitions related to GPU code generator; that can be also used by
 * user's extra module.
 * ---
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef CUDA_CODEGEN_H
#define CUDA_CODEGEN_H
#include "postgres.h"
#include "catalog/pg_proc.h"
#include "lib/ilist.h"
#include "lib/stringinfo.h"
#include "nodes/pathnodes.h"
#include "nodes/primnodes.h"
#include "utils/typcache.h"

/*
 * Type declarations for code generator
 */
#define DEVKERNEL_NEEDS_GPUSCAN			0x00000001	/* GpuScan */
#define DEVKERNEL_NEEDS_GPUJOIN			0x00000002	/* GpuJoin */
#define DEVKERNEL_NEEDS_GPUPREAGG		0x00000004	/* GpuPreAgg */
#define DEVKERNEL_NEEDS_GPUSORT			0x00000008	/* GpuSort */

#define DEVKERNEL_NEEDS_PRIMITIVE		0x00000100
#define DEVKERNEL_NEEDS_TIMELIB			0x00000200
#define DEVKERNEL_NEEDS_TEXTLIB			0x00000400
#define DEVKERNEL_NEEDS_JSONLIB			0x00000800
#define DEVKERNEL_NEEDS_MISCLIB			0x00001000
#define DEVKERNEL_NEEDS_RANGETYPE		0x00002000
#define DEVKERNEL_NEEDS_POSTGIS			0x00004000

#define DEVKERNEL_NEEDS_USERS_EXTRA1	0x01000000
#define DEVKERNEL_NEEDS_USERS_EXTRA2	0x02000000
#define DEVKERNEL_NEEDS_USERS_EXTRA3	0x04000000
#define DEVKERNEL_NEEDS_USERS_EXTRA4	0x08000000
#define DEVKERNEL_NEEDS_USERS_EXTRA5	0x10000000
#define DEVKERNEL_NEEDS_USERS_EXTRA6	0x21000000
#define DEVKERNEL_NEEDS_USERS_EXTRA7	0x40000000
#define DEVKERNEL_USERS_EXTRA_MASK		0x7f000000

#define DEVKERNEL_BUILD_DEBUG_INFO		0x80000000

struct devtype_info;
struct devfunc_info;
struct devcast_info;
struct codegen_context;

typedef uint32 (*devtype_hashfunc_type)(struct devtype_info *dtype,
										Datum datum);

typedef struct devtype_info {
	dlist_node	chain;
	uint32		hashvalue;
	Oid			type_oid;
	uint32		type_flags;
	int16		type_length;
	int16		type_align;
	bool		type_byval;
	bool		type_is_negative;
	const char *type_name;	/* name of device type; same of SQL's type */
	/* oid of type related functions */
	Oid			type_eqfunc;	/* function to check equality */
	Oid			type_cmpfunc;	/* function to compare two values */
	/* constant initializer cstring, if any */
	const char *max_const;
	const char *min_const;
	const char *zero_const;
	/*
	 * required size for extra buffer, if device type has special
	 * internal representation, or device type needs working buffer
	 * on device-side projection.
	 */
	int			extra_sz;
	/* type specific hash-function; to be compatible to device code */
	devtype_hashfunc_type hash_func;
	/* element type of array, if type is array */
	struct devtype_info *type_element;
	/* properties of sub-fields, if type is composite */
	int			comp_nfields;
	struct devtype_info *comp_subtypes[FLEXIBLE_ARRAY_MEMBER];
} devtype_info;

/*
 * Per-function callback to estimate maximum expected length of
 * the function result. -1, if cannot estimate it.
 * If device function may consume per-thread varlena buffer, it
 * should expand context->varlena_bufsz.
 */
typedef int (*devfunc_result_sz_type)(struct codegen_context *context,
									  struct devfunc_info *dfunc,
									  Expr **args, int *args_width);
typedef struct devfunc_info {
	dlist_node	chain;
	uint32		hashvalue;
	Oid			func_oid;		/* OID of the SQL function */
	Oid			func_collid;	/* OID of collation, if collation aware */
	bool		func_is_negative;	/* True, if not supported by GPU */
	bool		func_is_strict;		/* True, if NULL strict function */
	/* fields below are valid only if func_is_negative is false */
	int32		func_flags;		/* Extra flags of this function */
	List	   *func_args;		/* argument types by devtype_info */
	devtype_info *func_rettype;	/* result type by devtype_info */
	const char *func_sqlname;	/* name of the function in SQL side */
	const char *func_devname;	/* name of the function in device side */
	Cost		func_devcost;	/* relative cost to run function on GPU */
	devfunc_result_sz_type devfunc_result_sz; /* result width estimator */
} devfunc_info;

/*
 * Callback on CoerceViaIO (type cast using in/out handler).
 * In some special cases, device code can handle this class of type cast.
 */
typedef int (*devcast_coerceviaio_callback_f)(struct codegen_context *context,
											  struct devcast_info *dcast,
											  CoerceViaIO *node);
typedef struct devcast_info {
	dlist_node		chain;
	uint32			hashvalue;
	devtype_info   *src_type;
	devtype_info   *dst_type;
	char			castmethod;	/* one of COERCION_METHOD_* */
	devcast_coerceviaio_callback_f dcast_coerceviaio_callback;
} devcast_info;

/*
 * codegen.c
 */
#ifdef __PGSTROM_MODULE__

typedef struct codegen_context {
	StringInfoData	str;
	StringInfoData	decl_temp;	/* declarations of temporary variables */
	int				decl_count;	/* # of temporary variabes in decl */
	PlannerInfo *root;		//not necessary?
	RelOptInfo	*baserel;	/* scope of Var-node, if any */
	List	   *used_params;/* list of Const/Param in use */
	List	   *used_vars;	/* list of Var in use */
	Bitmapset  *param_refs;	/* referenced parameters */
	const char *var_label;	/* prefix of var reference, if exist */
	const char *kds_label;	/* label to reference kds, if exist */
	List	   *pseudo_tlist;	/* pseudo tlist expression, if any */
	int			extra_flags;	/* external libraries to be included */
	int			varlena_bufsz;	/* required size of temporary varlena buffer */
	int			devcost;	/* relative device cost */
} codegen_context;

#endif /* __PGSTROM_MODULE__ */

extern devtype_info *pgstrom_devtype_lookup(Oid type_oid);

/*
 * main.c
 */
#define PGSTROM_USERS_EXTRA_MAGIC_V1	(0x20210227U)

typedef struct
{
	uint32		magic;			/* PGSTROM_USERS_EXTRA_MAGIC_V1 */
	uint32		pg_version;		/* PG_VERSION built for */
	uint32		extra_flags;
	const char *extra_name;
	devtype_info *(*lookup_extra_devtype)(MemoryContext memcxt,
										  TypeCacheEntry *tcache);
	devfunc_info *(*lookup_extra_devfunc)(MemoryContext memcxt,
										  Oid func_oid,
										  Oid func_rettype,
										  oidvector *func_argtypes,
										  Oid func_collid);
	devcast_info *(*lookup_extra_devcast)(MemoryContext memcxt,
										  Oid src_type_oid, Oid dst_type_oid);
} pgstromUsersExtraDescriptor;

extern uint32	pgstrom_register_users_extra(const pgstromUsersExtraDescriptor *desc);

#endif	/* CUDA_CODEGEN_H */
