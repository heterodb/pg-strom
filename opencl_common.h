/*
 * opencl_common.h
 *
 * A common header for OpenCL device code
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef OPENCL_COMMON_H
#define OPENCL_COMMON_H

/*
 * OpenCL background server always adds -DOPENCL_DEVICE_CODE on kernel build,
 * but not for the host code, so this #if ... #endif block is available only
 * OpenCL device code.
 */
#ifdef OPENCL_DEVICE_CODE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/* NULL definition */
#define NULL	((void *) 0UL)

/* basic type definitions */
typedef bool		cl_bool;
typedef char		cl_char;
typedef uchar		cl_uchar;
typedef short		cl_short;
typedef ushort		cl_ushort;
typedef int			cl_int;
typedef uint		cl_uint;
typedef long		cl_long;
typedef ulong		cl_ulong;
typedef float		cl_float;
typedef double		cl_double;

/* varlena related stuff */
typedef struct {
	int		vl_len;
	char	vl_dat[1];
} varlena;
#define VARHDRSZ			((int) sizeof(cl_int))
#define VARDATA(PTR)		(((varlena *)(PTR))->vl_dat)
#define VARSIZE(PTR)		(((varlena *)(PTR))->vl_len)
#define VARSIZE_EXHDR(PTR)	(VARSIZE(PTR) - VARHDRSZ)

/*
 * row-format related stuff
 */
typedef struct {
	struct {
		cl_ushort	bi_hi;
		cl_ushort	bi_lo;
	} ip_blkid;
	cl_ushort		ip_posid;
} ItemPointerData;

typedef struct {
	union {
		struct {
			cl_uint	t_xmin;		/* inserting xact ID */
			cl_uint	t_xmax;		/* deleting or locking xact ID */
			union {
				cl_uint	t_cid;	/* inserting or deleting command ID, or both */
				cl_uint	t_xvac;	/* old-style VACUUM FULL xact ID */
			} t_field3;
		} t_heap;
		struct {
			cl_uint	datum_len_;	/* varlena header (do not touch directly!) */
			cl_uint	datum_typmod;	/* -1, or identifier of a record type */
			cl_uint	datum_typeid;	/* composite type OID, or RECORDOID */
		} t_datum;
	} t_choice;

	ItemPointerData	t_ctid;			/* current TID of this or newer tuple */

	cl_ushort		t_infomask2;	/* number of attributes + various flags */
	cl_ushort		t_infomask;		/* various flag bits, see below */
	cl_uchar		t_hoff;			/* sizeof header incl. bitmap, padding */
	/* ^ - 23 bytes - ^ */
	cl_uchar		t_bits[1];		/* bitmap of NULLs -- VARIABLE LENGTH */
} HeapTupleHeaderData;

/*
 * information stored in t_infomask:
 */
#define HEAP_HASNULL            0x0001  /* has null attribute(s) */
#define HEAP_HASVARWIDTH        0x0002  /* has variable-width attribute(s) */
#define HEAP_HASEXTERNAL        0x0004  /* has external stored attribute(s) */
#define HEAP_HASOID             0x0008  /* has an object-id field */
#define HEAP_XMAX_KEYSHR_LOCK   0x0010  /* xmax is a key-shared locker */
#define HEAP_COMBOCID           0x0020  /* t_cid is a combo cid */
#define HEAP_XMAX_EXCL_LOCK     0x0040  /* xmax is exclusive locker */
#define HEAP_XMAX_LOCK_ONLY     0x0080  /* xmax, if valid, is only a locker */





/* template for native types */
#define STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)	\
	typedef struct {								\
		BASE	value;								\
		bool	isnull;								\
	} pg_##NAME##_t

#define STROMCL_VARLENA_DATATYPE_TEMPLACE(NAME)		\
	STROMCL_SIMPLE_TYPE_TEMPLATE(NAME, __global varlena *)

#define STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)	\
	static pg_##NAME##_t pg_##NAME##_vref(...)		\
	{												\
													\
													\
													\
													\
													\
													\
													\
													\
													\
													\
													\
													\
													\
													\
	}

#define STROMCL_VARLENA_VARREF_TEMPLATE(NAME)	\
	static pg_##NAME##_t pg_##NAME##_vref(...)	\
	{											\
												\
												\
												\
	}

#define STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)	\
	static pg_##NAME##_t pg_##NAME##_pref(...)		\
	{												\
													\
													\
	}

#define STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)	\
	static pg_##NAME##_t pg_##NAME##_pref(...)	\
	{											\
												\
												\
												\
	}

#define STROMCL_SIMPLE_TYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)

#define STROMCL_VRALENA_TYPE_TEMPLATE(NAME)		\
	STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)		\
	STROMCL_VARLENA_VARREF_TEMPLATE(NAME)		\
	STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)

/* Built-in types */
STROMCL_SIMPLE_TYPE_TEMPLATE(bool, bool)

/*
 * Functions for BooleanTest
 */
static inline pg_bool_t
pg_bool_is_true(pg_bool_t result)
{
	result.value = (!result.isnull && result.value);
	result.isnull = false;
	return result;
}

static inline pg_bool_t
pg_bool_is_not_true(pg_bool_t result)
{
	result.value = (result.isnull || !result.value);
	result.isnull = false;
	return result;
}

static inline pg_bool_t
pg_bool_is_false(pg_bool_t result)
{
	result.value = (!result.isnull && !result.value);
	result.isnull = false;
	return result;
}

static inline pg_bool_t
pg_bool_is_not_false(pg_bool_t result)
{
	result.value = (result.isnull || result.value);
	result.isnull = false;
	return result;
}

static inline pg_bool_t
pg_bool_is_unknown(pg_bool_t result)
{
	result.value = result.isnull;
	result.isnull = false;
	return result;
}

static inline pg_bool_t
pg_bool_is_not_unknown(pg_bool_t result)
{
	result.value = !result.isnull;
	result.isnull = false;
	return result;
}

/*
 * Functions for BoolOp (EXPR_AND and EXPR_OR shall be constructed on demand)
 */
static inline pg_bool_t
pg_boolop_not(pg_bool_t result)
{
	result.value = !result.value;
	/* if null is given, result is also null */
	return arg;
}
#else	/* OPENCL_DEVICE_CODE */
#include "access/htup_details.h"
#include "storage/itemptr.h"
#endif



/*
 * Simplified FormData_pg_attribute; that shall be referenced in OpenCL
 * device to understand the data format given by host.
 */
typedef struct {
	cl_uint		atttypid;
	cl_short	attlen;
	cl_short	attnum;
	cl_int		attndims;
	cl_int		attcacheoff;
	cl_int		atttypmod;
	cl_bool		attnotnull;
	cl_bool		attbyval;
	cl_char		attalign;
} simple_pg_attribute;

/*
 *
 *
 *
 *
 *
 *
 *
 */
typedef struct {
	cl_uint				t_len;
	ItemPointerData		t_self;
	HeapTupleHeaderData	t_data;
} rs_tuple;

#endif	/* OPENCL_COMMON_H */
