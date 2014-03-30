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

/* Misc definitions */
#define FLEXIBLE_ARRAY_MEMBER

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

/*
 * Alignment macros
 */
#define TYPEALIGN(ALIGNVAL,LEN)	\
	(((uintptr_t) (LEN) + ((ALIGNVAL) - 1)) & ~((uintptr_t) ((ALIGNVAL) - 1)))

/*
 * Simplified varlena support.
 *
 * Unlike host code, device code cannot touch external and/or compressed
 * toast datum. All the format device code can understand is usual
 * in-memory form; 4-bytes length is put on the head and contents follows.
 * So, it is a responsibility of host code to decompress the toast values
 * if device code may access compressed varlena.
 * In case when device code touches unsupported format, calculation result
 * shall be postponed to calculate on the host side.
 *
 * Note that it is harmless to have external and/or compressed toast datam
 * unless it is NOT referenced in the device code. It can understand the
 * length of these values, unlike contents.
 */
typedef struct {
	int		vl_len;
	char	vl_dat[1];
} varlena;
#define VARHDRSZ			((int) sizeof(cl_int))
#define VARDATA(PTR)		(((varlena *)(PTR))->vl_dat)
#define VARSIZE(PTR)		(((varlena *)(PTR))->vl_len)
#define VARSIZE_EXHDR(PTR)	(VARSIZE(PTR) - VARHDRSZ)


#define VARATT_IS_4B(PTR) \
	((((varattrib_1b *) (PTR))->va_header & 0x01) == 0x00)
#define VARATT_IS_4B_U(PTR) \
	((((varattrib_1b *) (PTR))->va_header & 0x03) == 0x00)
#define VARATT_IS_4B_C(PTR) \
	((((varattrib_1b *) (PTR))->va_header & 0x03) == 0x02)
#define VARATT_IS_1B(PTR) \
	((((varattrib_1b *) (PTR))->va_header & 0x01) == 0x01)
#define VARATT_IS_1B_E(PTR) \
	((((varattrib_1b *) (PTR))->va_header) == 0x01)
#define VARATT_NOT_PAD_BYTE(PTR) \
	(*((uint8 *) (PTR)) != 0)

#define VARSIZE_ANY(PTR)							\
	(VARATT_IS_1B_E(PTR) ? VARSIZE_EXTERNAL(PTR) :	\
	 (VARATT_IS_1B(PTR) ? VARSIZE_1B(PTR) :			\
	  VARSIZE_4B(PTR)))

#else	/* OPENCL_DEVICE_CODE */
#include "access/htup_details.h"
#include "storage/itemptr.h"
#endif

/*
 * Error information on device kernel
 *
 * The kern_error shall be allocated on __local memory, then written back to
 * the global memory at end of the kernel execution.
 * The caller on host has to assign local memory with length of
 * sizeof(cl_int) * 2 * get_local_size(0) on its invocation
 * on its invocation.
 */
#define StromError_Success				0	/* OK */
#define StromError_RowFiltered			1	/* Row-clause was false */
#define StromError_RowReCheck			2	/* To be checked on the host */
#define StromError_DivisionByZero		100	/* Division by zero */

#define StromErrorIsStmtLevel(errcode)	((errcode) >= 100)

#ifdef OPENCL_DEVICE_CODE
static __local cl_int  *kern_local_error;
static __local cl_int  *kern_local_error_work;

/*
 * NOTE: all the kernel function has to call at the begining
 */
static inline void
kern_init_error(__local cl_int *karg_local_buffer)
{
	kern_local_error = karg_local_buffer;
	kern_local_error_work = kern_local_error + get_local_size(0);
	kern_local_error[get_local_id(0)] = StromError_Success;
}

/*
 * It sets an error code unless no statement level error code is already
 * set. Also, RowReCheck has higher priority than RowFiltered because
 * RowReCheck implies device cannot run the given expression completely.
 * (Usually, due to compressed or external varlena datum)
 */
static inline void
kern_set_error(cl_int errcode)
{
	cl_int	oldcode = kern_local_error[get_local_id(0)];

	if (StromErrorIsStmtLevel(errcode))
	{
		if (!StromErrorIsStmtLevel(oldcode))
			error_code[get_local_id(0)] = errcode;
	}
	else if (errcode > oldcode)
		error_code[get_local_id(0)] = errcode;
}

/*
 * Get an error code to be returned in statement level
 */
static cl_int
kern_get_stmt_error(void)
{
	cl_uint		wkgrp_sz = get_local_size(0);
	cl_uint		wkgrp_id = get_local_id(0);
	cl_uint		i = 0;

	kern_local_error_work[wkgrp_id] = kern_local_error[wkgrp_id];
	while (wkgrp_sz != 0)
	{
		if (wkgrp_id & ((1<<(i+1))-1) == 0)
		{
			cl_int	errcode1 = kern_local_error_work[wkgrp_id];
			cl_int	errcode2 = kern_local_error_work[wkgrp_id + (1<<i)];

			if (!StromErrorIsStmtLevel(errcode1) &&
				StromErrorIsStmtLevel(errcode2))
				kern_local_error_work[wkgrp_id] = errcode2;
		}
		wkgrp_sz >>= 1;
		i++;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	return kern_local_error_work[0];
}

/*
 * Write back row-indexes that has no error condition or to be re-checked.
 */
static cl_int
kern_writeback_row_results(cl_uint offset, __global cl_int *results)
{
	cl_uint		wkgrp_sz = get_local_size(0);
	cl_uint		wkgrp_id = get_local_id(0);
	cl_int		i = 0;

	kern_local_error_work[wkgrp_id]
		= (kern_local_error[wkgrp_id] == StromError_Success ||
		   kern_local_error[wkgrp_id] == StromError_RowReCheck ? 1 : 0);
	/*
	 * NOTE: At the begining, kern_local_error_work has either 1 or 0
	 * according to the row-level error code. This logic tries to count
	 * number of elements with 1,
	 * example)
	 * X[0] - 1 -> 1 (X[0])      -> 1 (X[0])   -> 1 (X[0])   -> 1 *
	 * X[1] - 0 -> 1 (X[0]+X[1]) -> 1 (X[0-1]) -> 1 (X[0-1]) -> 1
	 * X[2] - 0 -> 0 (X[2])      -> 1 (X[0-2]) -> 1 (X[0-2]) -> 1
	 * X[3] - 1 -> 1 (X[2]+X[3]) -> 2 (X[0-3]) -> 2 (X[0-3]) -> 2 *
	 * X[4] - 0 -> 0 (X[4])      -> 0 (X[4])   -> 2 (X[0-4]) -> 2
	 * X[5] - 0 -> 0 (X[4]+X[5]) -> 0 (X[4-5]) -> 2 (X[0-5]) -> 2
	 * X[6] - 1 -> 1 (X[6])      -> 1 (X[4-6]) -> 3 (X[0-6]) -> 3 *
	 * X[7] - 1 -> 2 (X[6]+X[7]) -> 2 (X[4-7]) -> 4 (X[0-7]) -> 4 *
	 * X[8] - 0 -> 0 (X[8])      -> 0 (X[7])   -> 0 (X[7])   -> 4
	 * X[9] - 1 -> 1 (X[8]+X[9]) -> 1 (X[7-8]) -> 1 (X7-8])  -> 5 *
	 */
	while (wkgrp_sz != 0)
	{
		if (wkgrp_id & (1 << i) != 0)
		{
			cl_int	i_source = (wkgrp_id & ~(1 << i)) | ((1 << i) - 1);

			kern_local_error_work[wkgrp_id] += kern_local_error_work[i_source];
		}
		wkgrp_sz >>= 1;
		i++;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/*
	 * After the loop, kern_local_error_work[wkgrp_id] should be an index
	 * to be written back. Let's put its row-index
	 */
	if (kern_local_error[wkgrp_id] == StromError_Success ||
		kern_local_error[wkgrp_id] == StromError_RowReCheck)
	{
		i = kern_local_error_work[wkgrp_id];

		results[i-1] = offset + wkgrp_id;
	}
	return kern_local_error_work[get_local_size(0) - 1];
}

/*
 * In-kernel mutex mechanism
 *
 */
typedef cl_int		kern_lock_t;

static inline void
kern_lock(volatile __global kern_lock_t *lock)
{
	while (atomic_cmpxchg(lock, 0, 1) != 0);
}

static inline void
kern_unlock(__global volatile kern_lock *lock)
{
	atomic_and(lock, 0);
}

#endif





/*
 * Message class identifiers
 */
#define StromMsg_ParamBuf		2001
#define StromMsg_RowStore		2002
#define StromMsg_ColumnStore	2003
#define StromMsg_ToastBuf		2004
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
	cl_uint			poffset[FLEXIBLE_ARRAY_MEMBER];	/* offset of params */
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
	/* zero, if column has no NULLs */
	cl_bool			atthasnull;
	/* alignment; 1,2,4 or 8, not characters in pg_attribute */
	cl_char			attaligh;
	/* length of attribute */
	cl_short		attlen;
	/*
	 * It has double meaning according to the store type.
	 * In case of row-store, it holds an offset from the head of tuple if
	 * it has no NULLs and no variable length field.
	 * In case of column-store, it holds an offset of the column array from
	 * head of the column-store itself.
	 */
	cl_int			attofs;
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
 * +-----------------+
 * | MessageTag      |
 * +-----------------+
 * | nrows (=N)      |
 * +-----------------+
 * | ncols (=M)      |
 * +-----------------+
 * | colmeta[0]      |
 * | colmeta[1]   o-------+ colmeta[j].attofs points an offset of the column-
 * |    :            |    | array in this store.
 * | colmeta[M-1]    |    |
 * +-----------------+    | (char *)(kcs) + colmeta[j].attofs points is
 * |   <padding>     |    | the address of column array.
 * +-----------------+    |
 * | column array    |    |
 * | for column-0    |    |
 * +-----------------+ <--+
 * | +---------------|
 * | | Nulls map     | If colmeta[j].atthasnull is TRUE, a bitmap shall be
 * | |               | put in front of the column array. Its length is aligned
 * | +---------------| to KERN_COLSTORE_ALIGN.
 * | | array of      |
 * | | column-1      |
 * | |               |
 * +-+---------------+
 * |      :          |
 * |      :          |
 * +-----------------+
 * | column array    |
 * | for column-(M-1)|
 * +-----------------+
 */
#define KERN_COLSTORE_ALIGN		64

typedef struct {
	MessageTag		mtag;   /* StromMsg_ColumnStore */
	cl_uint			nrows;  /* number of records in this store */
	cl_uint			ncols;	/* number of columns in this store */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER]; /* metadata of columns */
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
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME, __global varlena *)

#define STROMCL_SIMPLE_VARREF_RS_TEMPLATE(NAME,BASE)				\
	static pg_##NAME##_t											\
	pg_##NAME##_vref_rs(__global kern_row_store *krs,				\
						cl_uint attnum,								\
						cl_uint rowidx)								\
	{																\
		pg_##NAME##_t result;										\
		__global rs_tuple *tup = rs_get_tuple(krs,rowidx);			\
																	\
		if (!tup)													\
		{															\
			result.isnull = true;									\
		}															\
		else														\
		{															\
			BASE   *ptr = rs_tuple_getattr(krs,tup,attnum);			\
																	\
			result.isnull = false;									\
			result.value = *ptr;									\
		}															\
		return result;												\
	}

#define STROMCL_SIMPLE_VARREF_CS_TEMPLATE(NAME,BASE)				\
	static pg_##NAME##_t											\
	pg_##NAME##_vref_cs(__global kern_column_store *kcs,			\
						cl_uint attnum,								\
						cl_uint rowidx)								\
	{																\
		pg_##NAME##_t result;										\
		__global BASE *addr = cs_getattr(kcs,attnum,rowidx);		\
																	\
		if (!addr)													\
			result.isnull = true;									\
		else														\
		{															\
			result.isnull = false;									\
			result.value = *addr;									\
		}															\
		return result;												\
	}

#define STROMCL_VARLENA_VARREF_RS_TEMPLATE(NAME)					\
	static pg_##NAME##_t											\
	pg_##NAME##_vref_rs(__global kern_row_store *krs,				\
						cl_uint attnum,								\
						cl_uint rowidx)								\
	{																\
		pg_##NAME##_t result;										\
		__global rs_tuple *tup = rs_get_tuple(krs,rowidx);			\
																	\
		if (!tup)													\
		{															\
			result.isnull = true;									\
		}															\
		else														\
		{															\
			result.isnull = false;									\
			result.value =  rs_tuple_getattr(krs,tup,attnum);		\
		}															\
		return result;												\
	}

#define STROMCL_VARLENA_VARREF_CS_TEMPLATE(NAME)
	static pg_##NAME##_t											\
	pg_##NAME##_vref_cs(__global kern_column_store *kcs,			\
						__global kern_toastbuf *toast,				\
						cl_uint attnum,								\
						cl_uint rowidx)								\
	{																\
		pg_##NAME##_t result;										 \
		__global cl_uint *vl_offset	= cs_getattr(kcs,attnum,rowidx); \
																	 \
		if (!vl_offset)												\
			result.isnull = true;									\
		else														\
		{															\
			result.isnull = false;									\
			result.value = (varlena *)((char *)toast +				\
									   toast->coldir[attnum-1] +	\
									   *vl_offset);					\
		}															\
		return result;												\
	}

#define STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)			\
	static pg_##NAME##_t									\
	pg_##NAME##_param(__global kern_parambuf *kparam,		\
					  cl_uint param_id)						\
	{														\
		pg_##NAME##_t result;								\
		__global BASE *addr;								\
															\
		if (param_id < kparam->nparam &&					\
			kparam->poffset[param_id] > 0)					\
		{													\
			result.value									\
				= *((BASE *)((char *)kparam +				\
							 kparam->poffset[param_id]));	\
			result.isnull = false;							\
		}													\
		else												\
			result.isnull = true;							\
															\
		return result;										\
	}

#define STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)				\
	static pg_##NAME##_t									\
	pg_##NAME##_param(__global kern_parambuf *kparam,		\
					  cl_uint param_id)						\
	{														\
		pg_##NAME##_t result;								\
		__global varlena *addr;								\
															\
		if (param_id < kparam->nparam &&					\
			kparam->poffset[param_id] > 0)					\
		{													\
			result.value = (varlena *)						\
				((char *)kparam + kparam->poffset[param_id]);\
			result.isnull = false;							\
		}													\
		else												\
			result.isnull = true;							\
															\
		return result;										\
	}

#define STROMCL_SIMPLE_TYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_VARREF_RS_TEMPLATE(NAME,BASE)	\
	STROMCL_SIMPLE_VARREF_CS_TEMPLATE(NAME,BASE)	\
	STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)

#define STROMCL_VRALENA_TYPE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_VARREF_RS_TEMPLATE(NAME)		\
	STROMCL_VARLENA_VARREF_CS_TEMPLATE(NAME)		\
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
	if (rowidx < krs->nrows)
	{
		__global cl_uint   *tuple_offset_array
			= (cl_uint *)&krs->colmeta[krs->ncols];

		return (rs_tuple *)((char *)krs + tuple_offset_array[rowidx]);
	}
	return NULL;
}

static __global void *
rs_tuple_getattr(__global kern_row_store *krs,
				 __global rs_tuple *rs_tup,
				 __private cl_int attnum)
{
	__global void *result;

	/*
	 * This logic is simplified nocachegetattr(). It uses attribute offset
	 * cache if tuple contains no NULL and offset cache is available.
	 * Elsewhere, it falls slow path regardless of attribute AFTER/BEFORE
	 * null or variable-length field.
	 */
	if (rs_tup->data.t_infomask & HEAP_HASNULL == 0 &&
		krs->colmeta[attnum-1].attofs >= 0)
	{
		result = ((char *)&rs_tup->data +
				  rs_tup->data.t_hoff +
				  krs->colmeta[attnum-1].attofs);
	}
	else if (att_isnull(attnum-1, rs_tup->data.t_bits))
		result = NULL;
	else
	{
		/*
		 * Undesirable case, we need to walk on the tuple from the first
		 * attribute, if tuple has either NULL or variable length field.
		 */
		__global kern_colmeta *colmeta;
		cl_uint			i, off = 0;
		cl_char		   *tp = (cl_char *)&rs_tup->data + rs_tup->data.t_hoff;

		attnum--;
		for (i=0; ; i++)
		{
			if (rs_tup->data.t_infomask & HEAP_HASNULL != 0 &&
				att_isnull(i, rs_tup->data.t_bits))
				continue;

			colmeta = &krs->colmeta[i];
			if (colmeta->attlen < 0)
			{
				if (VARATT_NOT_PAD_BYTE(result + off))
					off = TYPEALIGN(colmeta->attalign, off);
			}
			else
				off = TYPEALIGN(colmeta->attalign, off);

			if (i == attnum)
				break;
			/*
			 * Add length of this field
			 * (att_addlength_pointer in host code)
			 */
			off += (colmeta->attlen < 0 ?
					VARSIZE_ANY(result + off) :
					colmeta->attlen);
		}
		result = tp + off;
	}
	return result;
}

static __global void *
cs_getattr(__global kern_column_store *kcs,
		   cl_uint attnum,
		   cl_uint rowidx)
{
	__global char  *result;
	cl_uint			offset;

	if (attnum > kcs->ncols || rowidx >= kcs->nrows)
		return NULL;

	offset = kcs->colmeta[attnum - 1].attofs;
	if (kcs->colmeta[attnum - 1].atthasnull)
	{
		__global cl_char *nullmap = (char *)kcs + offset;

		if ((nullmap[rowidx >> 3] & (1 << (rowidx & 0x07))) != 0)
			return NULL;

		offset += TYPEALIGN(KERN_COLSTORE_ALIGN, kcs->nrows >> 3);
	}

	if (kcs->colmeta[attnum - 1].attlen < 0)
		offset += sizeof(cl_uint) * rowidx;
	else
		offset += kcs->colmeta[attnum - 1].attlen * rowidx;

	return (void *)((char *)kcs + offset);
}
#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_COMMON_H */
