/*
 * cuda_codegen.h
 *
 * Definitions related to GPU code generator; that can be also used by
 * user's extra module.
 * ---
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef CUDA_CODEGEN_H
#define CUDA_CODEGEN_H
#include "postgres.h"
#include "catalog/pg_proc.h"
#include "lib/ilist.h"
#include "lib/stringinfo.h"
#if PG_VERSION_NUM >= 120000
#include "nodes/pathnodes.h"
#endif
#include "nodes/primnodes.h"
#include "utils/typcache.h"
#include "arrow_defs.h"

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

typedef uint32 (*devtype_hashfunc_type)(struct devtype_info *dtype, Datum datum);

typedef struct devtype_info {
	dlist_node	chain;
	uint32		hashvalue;
	const char *type_extension;	/* Extension that provides this type, if any */
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
	const char *func_extension;	/* Extension that provides this function, if any */
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
											  StringInfo body,
											  struct devcast_info *dcast,
											  CoerceViaIO *node);
typedef struct devcast_info {
	dlist_node		chain;
	uint32			hashvalue;
	devtype_info   *src_type;
	devtype_info   *dst_type;
	bool			cast_is_negative;
	bool			has_domain_checks;
	/*
	 * COERCION_METHOD_INOUT -> callback is not null
	 * COERCION_METHOD_BINARY -> callback is null
	 */
	devcast_coerceviaio_callback_f dcast_coerceviaio_callback;
} devcast_info;

/*
 * devindex_info - handler information of device GiST index
 */
typedef struct devindex_info {
	dlist_node		chain;
	uint32			hashvalue;
	const char	   *oper_extension;
	Oid				opcode;
	Oid				opfamily;
	int16			opstrategy;
	const char	   *index_kind;		/* only "gist" is available now */
	const char	   *index_fname;	/* device index handler name */
	devtype_info   *ivar_dtype;		/* device type of index'ed value */
	devtype_info   *iarg_dtype;		/* device type of index argument for search */
	bool			index_is_negative;
} devindex_info;

/*
 * codegen.c
 */
extern devtype_info *pgstrom_devtype_lookup(Oid type_oid);

/*
 * main.c
 */
#define PGSTROM_USERS_EXTRA_MAGIC_V1	(0x20210227U)

struct kern_data_store;
struct kern_colmeta;

/*
 * pgstromUsersExtraDescriptor
 */
typedef struct
{
	uint32		magic;			/* PGSTROM_USERS_EXTRA_MAGIC_V1 */
	uint32		pg_version;		/* PG_VERSION built for */
	uint32		extra_flags;
	/*
	 * extra_name is an identifier of the user's extra module.
	 * - "<extra_name>.h" shall be included by the code generator.
	 * - "<extra_name>.fatbin" shall be linked by the JIT linker.
	 */
	const char *extra_name;

	/*
	 * lookup_extra_devtype() can tell PG-Strom whether the supplied data
	 * type is device supported by the user's extra module.
	 * If supported, extra module set up properties of the given devtype_info,
	 * then returns true.
	 * Elsewhere, returns false.
	 */
	bool	(*lookup_extra_devtype)(const char *type_ident,
									devtype_info *dtype);

	/*
	 * lookup_extra_devfunc() can tell PG-Strom whether the supplied function
	 * is device-supported by the user's extra module.
	 * Note that data type of the arguments are not always identical to the
	 * definition of functions, if argument type has binary compatible types.
	 * The devfunc_info should be built to satisfy dfunc_rettype and
	 * dfunc_argtypes, not function's declaration at proc_form.
	 * If no supported function by the extra module, return NULL.
	 */
	bool	(*lookup_extra_devfunc)(const char *func_ident,
									devfunc_info *dfunc);

	/*
	 * lookup_extra_devcast() can tell PG-Strom whether the supplied cast
	 * from the source to the destination is device-supported by the user's
	 * extra module.
	 * If no supported cast by the extra module, return NULL.
	 */
	bool	(*lookup_extra_devcast)(const char *src_type_ident,
									const char *dst_type_ident,
									devcast_info *dcast);

	/*
	 * lookup_extra_devindex can tell PG-Strom whether the supplied operator
	 * has device supported index handler by the user's extra module.
	 */
	bool	(*lookup_extra_devindex)(const char *oper_ident,
									 devindex_info *dindex);

	/*
	 * arrow_lookup_pgtype() can tell PG-Strom a PostgreSQL type that shall
	 * assign on the supplied ArrowField. It can reference 'hint_oid' that
	 * is the 'pg_type' field metadata.
	 * If no relevant type by the extra module, return InvalidOid.
	 */
	Oid		  (*arrow_lookup_pgtype)(ArrowField *field,
									 Oid hint_oid,	/* pg_type metadata */
									 int32 *p_type_mod);

	/*
	 * arrow_datum_ref() shall reference an arrow value in the data type
	 * that is mapped to a particular PG type by the arrow_lookup_pgtype().
	 * This is CPU part, thus, its device code also needs to have own
	 * handlers to support device types. (Typically, declared at "EXTRA_NAME.h")
	 */
	bool	  (*arrow_datum_ref)(struct kern_data_store *kds,
								 struct kern_colmeta *cmeta,
								 size_t index,
								 Datum *p_value,
								 bool *p_isnull);
} pgstromUsersExtraDescriptor;

extern uint32	pgstrom_register_users_extra(const pgstromUsersExtraDescriptor *desc);

#endif	/* CUDA_CODEGEN_H */
