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
 * OpenCL intermediator always adds -DOPENCL_DEVICE_CODE on kernel build,
 * but not for the host code, so this #if ... #endif block is available
 * only OpenCL device code.
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

/* misc definition */
#define FLEXIBLE_ARRAY_MEMBER

#else	/* OPENCL_DEVICE_CODE */
#include "access/htup_details.h"
#include "storage/itemptr.h"
#endif

/*
 * Message class identifiers
 */
#define StromMsg_ParamBuf		2001
#define StromMsg_RowStore		2002
#define StromMsg_ColumnStore	2003
#defube StromMsg_ToastBuf		2004
#define StromMsg_GpuScan		3001

typedef struct {
	cl_uint			type;		/* one of StromMsg_* */
	cl_uint			length;		/* total length of this message */
} MessageTag;

/*
 * kern_parambuf
 *
 * Const and Parameter buffer. It stores constant values during a particular
 * scan, so it may make sense if it is obvious length of kern_parambuf is
 * less than constant memory (NOTE: not implemented yet).
 */
typedef struct {
	MessageTag		mtag;	/* StromMsg_ParamBuf */
	cl_uint			refcnt;	/* !HOST ONLY! reference counter */
	cl_uint			nparams;/* number of parameters */
	cl_uint			params[FLEXIBLE_ARRAY_MEMBER];	/* offset of params */
} kern_parambuf;


/*
 * Data type definitions for row oriented data format
 * ---------------------------------------------------
 */
#ifdef OPENCL_DEVICE_CODE
/*
 * we need to e-define HeapTupleData and HeapTupleHeaderData and
 * t_infomask related stuff
 */
typedef struct {
	struct {
		cl_ushort	bi_hi;
		cl_ushort	bi_lo;
	} ip_blkid;
	cl_ushort		ip_posid;
} ItemPointerData;

typedef struct HeapTupleData {
	cl_uint			t_len;
	ItemPointerData	t_self;
	cl_uint			t_tableOid;
	HOSTPTRUINT		t_data;		/* !HOSTONLY! pointer to htup on the host */
} HeapTupleData;

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

#define att_isnull(ATT, BITS) (!((BITS)[(ATT) >> 3] & (1 << ((ATT) & 0x07))))

/*
 * information stored in t_infomask:
 */
#define HEAP_HASNULL			0x0001	/* has null attribute(s) */
#define HEAP_HASVARWIDTH		0x0002	/* has variable-width attribute(s) */
#define HEAP_HASEXTERNAL		0x0004	/* has external stored attribute(s) */
#define HEAP_HASOID				0x0008	/* has an object-id field */
#define HEAP_XMAX_KEYSHR_LOCK	0x0010	/* xmax is a key-shared locker */
#define HEAP_COMBOCID			0x0020	/* t_cid is a combo cid */
#define HEAP_XMAX_EXCL_LOCK		0x0040	/* xmax is exclusive locker */
#define HEAP_XMAX_LOCK_ONLY		0x0080	/* xmax, if valid, is only a locker */

#endif

/*
 * kern_colmeta
 *
 * It stores metadata of columns being on row-store because tuple with NULL
 * values does not have always constant 
 */
typedef struct {
	cl_bool			attbyval;	/* attbyval */
	cl_char			attaligh;	/* alignment */
	cl_ushort		attlen;		/* length of attribute */
	cl_uint			attofs;		/* cached offset from the head */
} kern_colmeta;

/*
 * kern_row_store
 *
 * It stores records in row-format.
 *
 * +-----------------+ o--+
 * | MessageTag      |    | The array of tuple offset begins from colmeta[N].
 * +-----------------+    | It points a particular variable length region
 * | nrows           |    | from the tail.
 * +-----------------+    | 
 * | ncols           |    |
 * +-----------------+    |
 * | usage           |    |
 * +-----------------+    |
 * | colmeta[0]      |    |
 * | colmeta[1]      |    |
 * |    :            |    |
 * | colmeta[N-1]    |    |
 * +-----------------+ <--+
 * | tuples[0]       |
 * | tuples[1]       |
 * | tuples[2] o----------+ offset from the head of this row-store
 * |    :            |    |
 * | tuples[nrows-1] |    |
 * +-----------------+    |
 * |      :          |    |
 * | free area       |    |
 * |      :          |    |
 * +-----------------+ <------ current usage of this row-store
 * | (N-1)th rs_tuple|    |
 * +-----------------+    |
 * |      :          |    |
 * |      :          |    |
 * +-----------------+ <--+
 * | 2nd rs_tuple    |
 * +-----------------+
 * | 1st rs_tuple    |
 * +-----------------+
 * |      :          |
 * | 0th rs_tuple    |
 * |      :          |
 * +-----------------+
 */
typedef struct {
    MessageTag      mtag;   /* StromMsg_RowStore */
	cl_uint			nrows;	/* number of rows in this store */
	cl_uint			ncols;	/* number of columns in the source relation */
	cl_uint			usage;	/* usage of this store */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER];	/* metadata of columns */
} kern_row_store;

/*
 * rs_tuple
 *
 * HeapTuple representation in row-store. Even though most of metadata in
 * the HeapTupleData / HeapTupleHeaderData are not used in device kernel,
 * we put them together because it enables to avoid tuple re-construction
 * on the host side that has limited computing power.
 */
typedef struct {
	HeapTupleData		htup;
	HeapTupleHeaderData	data;
} rs_tuple;

/*
 * Data type definitions for column oriented data format
 * ---------------------------------------------------
 */

/*
 * kern_column_store
 *
 * It stores arrays in column-format
 */
typedef struct {
	MessageTag		mtag;   /* StromMsg_ColumnStore */
	cl_uint			nrows;  /* number of records in this store */
	cl_uint			ncols;	/* number of columns in this store */
	cl_uint			nullmap;/* offset of null array */
	cl_uint			coldir[FLEXIBLE_ARRAY_MEMBER]; /* offset of column array */
} kern_column_store;

/*
 * kern_toastbuf
 *
 * The kernel toast buffer has number of columns and per-column directory
 * in its header region. The per-column directory points the starting offset
 * from the head of kern_toastbuf.
 *
 * +--------------+
 * | ncols        | number of columns in this buffer
 * +--------------+
 * | coldir[0]    |
 * | coldir[1]  o-------+ In case when a varlena reference (offset=120) of
 * |   :          |     | column-1, it has to reference coldir[1] to get
 * | coldir[N-1]  |     | offset of per-column varlena buffer.
 * +--------------+     | Then, it adds per-datum offset to reach the
 * |   :          |     | address of variable.
 * |   :          |     |
 * +--------------+  <--+
 * |   :          |  )
 * +--------------+  +120
 * |'Hello!'      |
 * +--------------+
 * |   :          |
 * |   :          |
 * +--------------+
 */
typedef struct {
	MessageTag		mtag;
	dlist_node		chain;	/* !HOST ONLY! linked to column store */
	cl_uint			ncols;
	cl_uint			coldir[FLEXIBLE_ARRAY_MEMBER];
} kern_toastbuf;

#ifdef OPENCL_DEVICE_CODE













/* template for native types */
#define STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)	\
	typedef struct {								\
		BASE	value;								\
		bool	isnull;								\
	} pg_##NAME##_t

#define STROMCL_VARLENA_DATATYPE_TEMPLACE(NAME)		\
	STROMCL_SIMPLE_TYPE_TEMPLATE(NAME, __global varlena *)

#define STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)		\
	static pg_##NAME##_t								\
	pg_##NAME##_vref_rs(__private cl_uint attidx,		\
						__private cl_uint attofs,		\
						__private cl_uint rowidx,		\
						__global kern_row_store *krs)	\
	{													\
	rs_tuple *tuple;									\
	pg_##NAME##_t result;								\
														\
	if (rowidx >= krs->nrows)							\
	{													\
		result.isnull = true;							\
		result.value = (BASE) 0;						\
	}													\
	else												\
	{													\
	rs_tuple   *tuple = krs->tuples[rowidx];			\
														\
	if (HeapTupleNoNulls(tup))							\
	{													\
	cache_off

	}
	else
	{
		


}
return result;




		tuple = krs->tuples[attidx];					\
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
	}													\
														\
	static pg_##NAME##_t								\
	pg_##NAME##_vref_cs(__private int ...)				\
	{													\
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

/*
 * Functions to reference row-format data
 */
static inline __global rs_tuple *
rs_get_tuple(__global kern_row_store *krs,
			 __private cl_uint rowidx)
{
	__global cl_uint   *tup_offset;

	tup_offset = (cl_uint *)&krs->colmeta[krs->ncols];

	return (rs_tuple *)((char *)krs + tup_offset[rowidx]);
}


static inline __global void *
rs_tuple_getattr_common(__global kern_row_store *krs,
						__global rs_tuple *rs_tup,
						__private cl_int attnum)
{
	__global void *result;

	if (rs_tup->data.t_infomask & HEAP_HASNULL == 0)
	{
		/* No NULLs here, attofs can be used */
		result = ((char *)&rs_tup->data +
				  rs_tup->data.t_hoff +
				  krs->colmeta[attnum-1].attofs);
	}
	else
	{
		cl_char	   *tp;

		if (att_isnull(attnum-1, rs_tup->data.t_bits))
			return NULL;

		See nocachegetattr here!





	}
	return result;
}

static inline cl_char
rs_tuple_getattr_1(__global kern_row_store *krs,
				   __private cl_uint attidx,
				   __private cl_uint rowidx)
{
	__global cl_char   *address
		= rs_tuple_getattr_common(krs, attidx, rowidx);
	return *address;
}

static inline cl_short
rs_tuple_getattr_2(__global kern_row_store *krs,
				   __private cl_uint attidx,
				   __private cl_uint rowidx)
{
	__global cl_short  *address
		= rs_tuple_getattr_common(krs, attidx, rowidx);
	return *address;
}

static inline cl_int
rs_tuple_getattr_4(__global kern_row_store *krs,
				   __private cl_uint attidx,
				   __private cl_uint rowidx)
{
	__global cl_int  *address
		= rs_tuple_getattr_common(krs, attidx, rowidx);
	return *address;
}

static inline cl_long
rs_tuple_getattr_8(__global kern_row_store *krs,
				   __private cl_uint attidx,
				   __private cl_uint rowidx)
{
	__global cl_long  *address
		= rs_tuple_getattr_common(krs, attidx, rowidx);
	return *address;
}







#endif	/* OPENCL_COMMON_H */
