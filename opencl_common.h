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
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

/* NULL definition */
#define NULL	((__global void *) 0UL)

/* Misc definitions */
#define FLEXIBLE_ARRAY_MEMBER
#define offsetof(TYPE, FIELD)   ((uintptr_t) &((TYPE *)0)->FIELD)
#define lengthof(ARRAY)			(sizeof(ARRAY) / sizeof((ARRAY)[0]))
#define BITS_PER_BYTE			8

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
#if HOSTPTRLEN == 8
typedef cl_ulong	hostptr_t;
#elif HOSTPTRLEN == 4
typedef cl_uint		hostptr_t;
#else
#error unexpected host pointer length
#endif	/* HOSTPTRLEN */

#define INT64CONST(x)	((cl_long) x##L)
#define UINT64CONST(x)	((cl_ulong) x##UL)

/*
 * Alignment macros
 */
#define TYPEALIGN(ALIGNVAL,LEN)	\
	(((uintptr_t) (LEN) + ((ALIGNVAL) - 1)) & ~((uintptr_t) ((ALIGNVAL) - 1)))
#define INTALIGN(LEN)			TYPEALIGN(sizeof(cl_int), (LEN))
#define INTALIGN_DOWN(LEN)		TYPEALIGN_DOWN(sizeof(cl_int), (LEN))
#define LONGALIGN(LEN)          TYPEALIGN(sizeof(cl_long), (LEN))
#define LONGALIGN_DOWN(LEN)     TYPEALIGN_DOWN(sizeof(cl_long), (LEN))

#else	/* OPENCL_DEVICE_CODE */
#include "access/htup_details.h"
#include "storage/itemptr.h"
#define __global	/* address space qualifier is noise on host */
#define __local		/* address space qualifier is noise on host */
#define __private	/* address space qualifier is noise on host */
#endif

/*
 * For kernel coding simplification, PG-Strom expects size of workgroup is
 * multiplexer of 32. It allows to consolidate calculation results by each
 * core into a 32bit register.
 * NOTE: kernel code assumes this unit-size is 32 (also 2^N), so it is not
 * sufficient to update this definition, if different unit size is applied.
 */
#define		PGSTROM_WORKGROUP_UNITSZ	32

/*
 * Error code definition
 */
#define StromError_Success				0	/* OK */
#define StromError_RowFiltered			1	/* Row-clause was false */
#define StromError_RowReCheck			2	/* To be checked on the host */
#define StromError_ServerNotReady		100	/* OpenCL server is not ready */
#define StromError_BadRequestMessage	101	/* Bad request message */
#define StromError_OpenCLInternal		102	/* OpenCL internal error */
#define StromError_OutOfSharedMemory	105	/* out of shared memory */
#define StromError_OutOfMemory			106	/* out of host memory */
#define StromError_DivisionByZero		200	/* Division by zero */
#define StromError_DataStoreCorruption	300	/* Row/Column Store Corrupted */
#define StromError_DataStoreNoSpace		301	/* No Space in Row/Column Store */
#define StromError_DataStoreOutOfRange	302	/* Out of range in Data Store */
#define StromError_DataStoreReCheck		303	/* Row/Column Store be rechecked */

/* significant error; that abort transaction on the host code */
#define StromErrorIsSignificant(errcode)	((errcode) >= 100 || (errcode) < 0)

#ifdef OPENCL_DEVICE_CODE
/*
 * It sets an error code unless no significant error code is already set.
 * Also, RowReCheck has higher priority than RowFiltered because RowReCheck
 * implies device cannot run the given expression completely.
 * (Usually, due to compressed or external varlena datum)
 */
static inline void
STROM_SET_ERROR(__private cl_int *p_error, cl_int errcode)
{
	cl_int	oldcode = *p_error;

	if (StromErrorIsSignificant(errcode))
	{
		if (!StromErrorIsSignificant(oldcode))
			*p_error = errcode;
	}
	else if (errcode > oldcode)
		*p_error = errcode;
}

/*
 * Data type definitions for row oriented data format
 * ---------------------------------------------------
 */

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
	hostptr_t		t_data;		/* !HOSTONLY! pointer to htup on the host */
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
 * alignment for pg-strom
 */
#define STROMALIGN_LEN			16
#define STROMALIGN(LEN)			TYPEALIGN(STROMALIGN_LEN,LEN)
#define STROMALIGN_DOWN(LEN)	TYPEALIGN_DOWN(STROMALIGN_LEN,LEN)

/*
 * kern_colmeta
 *
 * It stores metadata of columns being on row-store because tuple with NULL
 * values does not have always constant 
 */
typedef struct {
	/* true, if column never has NULL (thus, no nullmap required) */
	cl_char			attnotnull;
	/* alignment; 1,2,4 or 8, not characters in pg_attribute */
	cl_char			attalign;
	/* length of attribute */
	cl_short		attlen;
	/* offset to null-map and column-array from the head of column-store */
	cl_uint			cs_ofs;
} kern_colmeta;

/*
 * kern_row_store
 *
 * It stores records in row-format.
 *
 * +-----------------+
 * | length          |
 * +-----------------+ o--+
 * | ncols (= M)     |    | The array of tuple offset begins from colmeta[N].
 * +-----------------+    | It points a particular variable length region 
 * | nrows (= N)     |    | from the tail.
 * +-----------------+    |
 * | colmeta[0]      |    |
 * | colmeta[1]      |    |
 * |    :            |    |
 * | colmeta[M-1]    |    |
 * +-----------------+ <--+
 * | tuples[0]       |
 * | tuples[1]       |
 * | tuples[2] o----------+ offset from the head of this row-store
 * |    :            |    |
 * | tuples[N-1]     |    |
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
	cl_uint			length;	/* length of this kernel row_store */
	cl_uint			ncols;	/* number of columns in the source relation */
	cl_uint			nrows;	/* number of rows in this store */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER];	/* metadata of columns */
} kern_row_store;

static inline __global cl_uint *
kern_rowstore_get_offset(__global kern_row_store *krs)
{
	return ((__global cl_uint *)(&krs->colmeta[krs->ncols]));
}

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

static inline __global rs_tuple *
kern_rowstore_get_tuple(__global kern_row_store *krs, cl_uint krs_index)
{
	__global cl_uint   *p_offset;

	if (krs_index >= krs->nrows)
		return NULL;
	p_offset = kern_rowstore_get_offset(krs);
	if (p_offset[krs_index] == 0)
		return NULL;
	return (__global rs_tuple *)((uintptr_t)krs + p_offset[krs_index]);
}

/*
 * Data type definitions for column oriented data format
 * ---------------------------------------------------
 */

/*
 * kern_column_store
 *
 * It stores arrays in column-format
 * +-----------------+
 * | length          |
 * +-----------------+
 * | ncols (=M)      |
 * +-----------------+
 * | nrows (=N)      |
 * +-----------------+
 * | colmeta[0]      |
 * | colmeta[1]   o-------+ colmeta[j].cs_ofs points an offset of the column-
 * |    :            |    | array in this store.
 * | colmeta[M-1]    |    |
 * +-----------------+    | (char *)(kcs) + colmeta[j].cs_ofs points is
 * | column array    |    | the address of column array.
 * | for column-0    |    |
 * +-----------------+ <--+
 * | +---------------|
 * | | Nulls map     | If colmeta[j].atthasnull is TRUE, a bitmap shall be
 * | |               | put in front of the column array. Its length is aligned
 * | +---------------| to STROMALIGN_LEN
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
typedef struct {
	cl_uint			length;	/* length of this kernel column-store */
	cl_uint			ncols;	/* number of columns in this store */
	cl_uint			nrows;  /* number of records in this store */
	cl_uint			nrooms;	/* max number of records can be stored */
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
 * | magic        | magic number; a value that should not be a length of
 * +--------------+ other buffer object (like, row- or column-store)
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
#define TOASTBUF_MAGIC		0xffffffff	/* should not be length of buffer */

typedef struct {
	cl_uint			length;	/* = TOASTBUF_MAGIC, if coldir should be added */
	union {
		cl_uint		ncols;	/* number of coldir entries, if exists */
		cl_uint		usage;	/* usage counter of this toastbuf */
	};
	cl_uint			coldir[FLEXIBLE_ARRAY_MEMBER];
} kern_toastbuf;

/*
 * kern_get_datum
 *
 * Reference to a particular datum on the supplied column store.
 * It returns NULL, If it is a null-value in context of SQL. Elsewhere,
 * it returns a pointer towards global memory.
 */
static inline __global void *
kern_get_datum(__global kern_column_store *kcs,
			   cl_uint colidx,
			   cl_uint rowidx)
{
	kern_colmeta colmeta;
	cl_uint		offset;

	if (colidx >= kcs->ncols || rowidx >= kcs->nrows)
		return NULL;

	colmeta = kcs->colmeta[colidx];
	offset = colmeta.cs_ofs;
	if (!colmeta.attnotnull)
	{
		if (att_isnull(rowidx, (__global char *)kcs + offset))
			return NULL;
		offset += STROMALIGN((kcs->nrooms + 7) >> 3);
	}
	if (colmeta.attlen > 0)
		offset += colmeta.attlen * rowidx;
	else
		offset += sizeof(cl_uint) * rowidx;

	return (__global void *)((__global char *)kcs + offset);
}

/*
 * kern_parambuf
 *
 * Const and Parameter buffer. It stores constant values during a particular
 * scan, so it may make sense if it is obvious length of kern_parambuf is
 * less than constant memory (NOTE: not implemented yet).
 */
typedef struct {
	cl_uint		length;		/* total length of parambuf */
	cl_uint		nparams;	/* number of parameters */
	cl_uint		poffset[FLEXIBLE_ARRAY_MEMBER];	/* offset of params */
} kern_parambuf;

static inline __global void *
kparam_get_value(kern_parambuf *kparams, cl_uint pindex)
{
	if (pindex >= kparams->nparams)
		return NULL;
	if (kparams->poffset[pindex] == 0)
		return NULL;
	return (__global char *)kparams + kparams->poffset[pindex];
}

/*
 * kern_resultbuf
 *
 * Output buffer to write back calculation results on a parciular chunk.
 * 'errcode' informs a significant error that shall raise an error on
 * host side and abort transactions. 'results' informs row-level status.
 */
typedef struct {
	cl_uint		nrooms;		/* max number of results rooms */
	cl_uint		nitems;		/* number of results being written */
	cl_int		errcode;	/* chunk-level error */
	cl_int		results[FLEXIBLE_ARRAY_MEMBER];
} kern_resultbuf;

#ifdef OPENCL_DEVICE_CODE
/*
 * PostgreSQL Data Type support in OpenCL kernel
 *
 * A stream of data sequencea are moved to OpenCL kernel, according to
 * the above row-/column-store definition. The device code being generated
 * by PG-Strom deals with each data item using the following data type;
 *
 * typedef struct
 * {
 *     BASE    value;
 *     bool    isnull;
 * } pg_##NAME##_t
 *
 * PostgreSQL has three different data classes:
 *  - fixed-length referenced by value
 *  - fixed-length referenced by pointer
 *  - variable-length value
 *
 * Right now, we support the two except for fixed-length referenced by
 * pointer (because these are relatively minor data type than others).
 * BASE reflects the data type in PostgreSQL; may be an integer, a float
 * or something others, however, all the variable-length value has same
 * BASE type; that is an offset of associated toast buffer, to reference
 * varlena structure on the global memory.
 */

/*
 * Template of variable classes: fixed-length referenced by value
 * ---------------------------------------------------------------
 */
#define STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)			\
	typedef struct {										\
		BASE	value;										\
		bool	isnull;										\
	} pg_##NAME##_t;

#define STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)			\
	static pg_##NAME##_t									\
	pg_##NAME##_vref(__global kern_column_store *kcs,		\
					 __private int *p_errcode,				\
					 cl_uint colidx,						\
					 cl_uint rowidx)						\
	{														\
		pg_##NAME##_t result;								\
		__global BASE *addr									\
			= kern_get_datum(kcs,colidx,rowidx);			\
															\
		if (!addr)											\
			result.isnull = true;							\
		else												\
		{													\
			result.isnull = false;							\
			result.value = *addr;							\
		}													\
		return result;										\
	}

#define STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)			\
	static pg_##NAME##_t									\
	pg_##NAME##_param(__global kern_parambuf *kparam,		\
					  __private int *p_errcode,				\
					  cl_uint param_id)						\
	{														\
		pg_##NAME##_t result;								\
		__global BASE *addr;								\
															\
		if (param_id < kparam->nparams &&					\
			kparam->poffset[param_id] > 0)					\
		{													\
			result.value = *((__global BASE *)				\
							 ((uintptr_t)kparam +			\
							  kparam->poffset[param_id]));	\
			result.isnull = false;							\
		}													\
		else												\
			result.isnull = true;							\
															\
		return result;										\
	}

#define STROMCL_SIMPLE_NULLTEST_TEMPLATE(NAME)				\
	static pg_bool_t										\
	pgfn_##NAME##_isnull(__private int *p_errcode,			\
						 pg_##NAME##_t arg)					\
	{														\
		pg_bool_t result;									\
															\
		result.isnull = false;								\
		result.value = arg.isnull;							\
		return result;										\
	}														\
															\
	static pg_bool_t										\
	pgfn_##NAME##_isnotnull(__private int *p_errcode,		\
							pg_##NAME##_t arg)				\
	{														\
		pg_bool_t result;									\
															\
		result.isnull = false;								\
		result.value = !arg.isnull;							\
		return result;										\
	}

#define STROMCL_SIMPLE_TYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_NULLTEST_TEMPLATE(NAME)

/*
 * declaration of some built-in data types
 */

/* pg_bool_t */
#ifndef PG_BOOL_TYPE_DEFINED
#define PG_BOOL_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(bool, bool)
#endif

/* pg_int4_t */
#ifndef PG_INT4_TYPE_DEFINED
#define PG_INT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int4, cl_int)
#endif

/*
 * Template of variable classes: variable-length variables
 * ---------------------------------------------------------------
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
	cl_int		vl_len;
	cl_char		vl_dat[1];
} varlena;

#define VARHDRSZ			((int) sizeof(cl_int))
#define VARDATA(PTR)		VARDATA_4B(PTR)
#define VARSIZE(PTR)		VARSIZE_4B(PTR)
#define VARSIZE_EXHDR(PTR)	(VARSIZE(PTR) - VARHDRSZ)

typedef union
{
	struct						/* Normal varlena (4-byte length) */
	{
		cl_uint		va_header;
		cl_char		va_data[1];
    }		va_4byte;
	struct						/* Compressed-in-line format */
	{
		cl_uint		va_header;
		cl_uint		va_rawsize;	/* Original data size (excludes header) */
		cl_char		va_data[1];	/* Compressed data */
	}		va_compressed;
} varattrib_4b;

typedef struct
{
	cl_uchar	va_header;
	cl_char		va_data[1];		/* Data begins here */
} varattrib_1b;

/* inline portion of a short varlena pointing to an external resource */
typedef struct
{
	cl_uchar    va_header;		/* Always 0x80 or 0x01 */
	cl_uchar	va_tag;			/* Type of datum */
	cl_char		va_data[1];		/* Data (of the type indicated by va_tag) */
} varattrib_1b_e;

typedef enum vartag_external
{
	VARTAG_INDIRECT = 1,
	VARTAG_ONDISK = 18
} vartag_external;

#define VARHDRSZ_SHORT			offsetof(varattrib_1b, va_data)
#define VARATT_SHORT_MAX		0x7F

typedef struct varatt_external
{
	cl_int		va_rawsize;		/* Original data size (includes header) */
	cl_int		va_extsize;		/* External saved size (doesn't) */
	cl_int		va_valueid;		/* Unique ID of value within TOAST table */
	cl_int		va_toastrelid;	/* RelID of TOAST table containing it */
} varatt_external;

typedef struct varatt_indirect
{
	hostptr_t	pointer;	/* Host pointer to in-memory varlena */
} varatt_indirect;

#define VARTAG_SIZE(tag) \
	((tag) == VARTAG_INDIRECT ? sizeof(varatt_indirect) :	\
	 (tag) == VARTAG_ONDISK ? sizeof(varatt_external) :		\
	 0 /* should not happen */)

#define VARHDRSZ_EXTERNAL		offsetof(varattrib_1b_e, va_data)
#define VARTAG_EXTERNAL(PTR)	VARTAG_1B_E(PTR)
#define VARSIZE_EXTERNAL(PTR)	\
	(VARHDRSZ_EXTERNAL + VARTAG_SIZE(VARTAG_EXTERNAL(PTR)))

#define VARATT_IS_4B(PTR) \
	((((__global varattrib_1b *) (PTR))->va_header & 0x01) == 0x00)
#define VARATT_IS_4B_U(PTR) \
	((((__global varattrib_1b *) (PTR))->va_header & 0x03) == 0x00)
#define VARATT_IS_4B_C(PTR) \
	((((__global varattrib_1b *) (PTR))->va_header & 0x03) == 0x02)
#define VARATT_IS_1B(PTR) \
	((((__global varattrib_1b *) (PTR))->va_header & 0x01) == 0x01)
#define VARATT_IS_1B_E(PTR) \
	((((__global varattrib_1b *) (PTR))->va_header) == 0x01)
#define VARATT_IS_COMPRESSED(PTR)		VARATT_IS_4B_C(PTR)
#define VARATT_IS_EXTERNAL(PTR)			VARATT_IS_1B_E(PTR)
#define VARATT_NOT_PAD_BYTE(PTR) \
	(*((__global cl_uchar *) (PTR)) != 0)

#define VARSIZE_4B(PTR) \
	((((__global varattrib_4b *) (PTR))->va_4byte.va_header >> 2) & 0x3FFFFFFF)
#define VARSIZE_1B(PTR) \
	((((__global varattrib_1b *) (PTR))->va_header >> 1) & 0x7F)
#define VARTAG_1B_E(PTR) \
	(((__global varattrib_1b_e *) (PTR))->va_tag)

#define VARSIZE_ANY_EXHDR(PTR) \
	(VARATT_IS_1B_E(PTR) ? VARSIZE_EXTERNAL(PTR)-VARHDRSZ_EXTERNAL : \
	 (VARATT_IS_1B(PTR) ? VARSIZE_1B(PTR)-VARHDRSZ_SHORT :			 \
	  VARSIZE_4B(PTR)-VARHDRSZ))

#define VARSIZE_ANY(PTR)							\
	(VARATT_IS_1B_E(PTR) ? VARSIZE_EXTERNAL(PTR) :	\
	 (VARATT_IS_1B(PTR) ? VARSIZE_1B(PTR) :			\
	  VARSIZE_4B(PTR)))

#define VARDATA_4B(PTR)	(((__global varattrib_4b *) (PTR))->va_4byte.va_data)
#define VARDATA_1B(PTR)	(((__global varattrib_1b *) (PTR))->va_data)
#define VARDATA_ANY(PTR) \
	(VARATT_IS_1B(PTR) ? VARDATA_1B(PTR) : VARDATA_4B(PTR))
/*
 * functions to reference variable length variables
 */
STROMCL_SIMPLE_DATATYPE_TEMPLATE(varlena, __global varlena *)

static inline pg_varlena_t
pg_varlena_vref(__global kern_column_store *kcs,
				__global kern_toastbuf *toast,
				__private int *p_errcode,
				cl_uint colidx,
				cl_uint rowidx)
{
	pg_varlena_t result;
	__global cl_uint *p_offset = kern_get_datum(kcs,colidx,rowidx);

	if (!p_offset)
		result.isnull = true;
	else
	{
		cl_uint	offset = *p_offset;
		__global varlena *val;
		if (toast->length == TOASTBUF_MAGIC)
			offset += toast->coldir[colidx];
		val = ((__global varlena *)((__global char *)toast + offset));
		if (VARATT_IS_4B_U(val) || VARATT_IS_1B(val))
		{
			result.isnull = false;
			result.value = val;
		}
		else
		{
			result.isnull = true;
			STROM_SET_ERROR(p_errcode, StromError_RowReCheck);
		}
	}
	return result;
}

static inline pg_varlena_t
pg_varlena_param(__global kern_parambuf *kparam,
				 __private int *p_errcode,
				 cl_uint param_id)
{
	pg_varlena_t	result;
	__global varlena *addr;

	if (param_id < kparam->nparams &&
		kparam->poffset[param_id] > 0)
	{
		__global varlena *val = (__global varlena *)
			((__global char *)kparam + kparam->poffset[param_id]);
		if (VARATT_IS_4B_U(val) || VARATT_IS_1B(val))
		{
			result.value = val;
			result.isnull = false;
		}
		else
		{
			result.isnull = true;
			STROM_SET_ERROR(p_errcode, StromError_RowReCheck);
		}
	}
	else
		result.isnull = true;

	return result;
}

STROMCL_SIMPLE_NULLTEST_TEMPLATE(varlena)

#define STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)						\
	typedef pg_varlena_t	pg_##NAME##_t;

#define STROMCL_VARLENA_VARREF_TEMPLATE(NAME)						\
	static inline pg_##NAME##_t										\
	pg_##NAME##_vref(__global kern_column_store *kcs,				\
					 __global kern_toastbuf *toast,					\
					 __private int *p_errcode,						\
					 cl_uint colidx,								\
					 cl_uint rowidx)								\
	{																\
		return pg_varlena_vref(kcs,toast,p_errcode,colidx,rowidx);	\
	}

#define STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)						\
	static pg_##NAME##_t											\
	pg_##NAME##_param(__global kern_parambuf *kparam,				\
					  __private int *p_errcode,						\
					  cl_uint param_id)								\
	{																\
		return pg_varlena_param(kparam,p_errcode,param_id);			\
	}

#define STROMCL_VARLENA_NULLTEST_TEMPLATE(NAME)						\
	static pg_bool_t												\
	pgfn_##NAME##_isnull(__private int *p_errcode,					\
					   pg_##NAME##_t arg)							\
	{																\
		return pgfn_varlena_isnull(p_errcode, arg);					\
	}																\
	static pg_bool_t												\
	pgfn_##NAME##_isnotnull(__private int *p_errcode,				\
						  pg_##NAME##_t arg)						\
	{																\
		return pgfn_varlena_isnotnull(p_errcode, arg);				\
	}

#define STROMCL_VARLENA_TYPE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_VARREF_TEMPLATE(NAME)			\
	STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)			\
	STROMCL_VARLENA_NULLTEST_TEMPLATE(NAME)

/*
 * pg_bytea_t is also a built-in data type
 */
#ifndef PG_BYTEA_TYPE_DEFINED
#define PG_BYTEA_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(bytea)
#endif

/* ------------------------------------------------------------------
 *
 * Declaration of utility functions
 *
 * ------------------------------------------------------------------ */

/*
 * memcpy implementation for OpenCL kernel usage
 */
#if 0
static __global void *
memcpy(__global void *__dst, __global const void *__src, size_t len)
{
	__global char		*dst = __dst;
	__global const char	*src = __src;

	/* an obvious case that we don't need to take a copy */
	if (dst == src || len == 0)
		return __dst;
	/* just a charm */
	prefetch(src, len);

	while (len-- > 0)
		*dst++ = *src++;
	return __dst;
}
#else
static __global void *
memcpy(__global void *__dst, __global const void *__src, size_t len)
{
	size_t		alignMask	= sizeof(cl_uint) - 1;
	cl_ulong	buf8		= 0x0;
	cl_int		bufCnt		= 0;

	__global const cl_uchar	*srcChar;
	__global const cl_uint	*srcInt;
	__global cl_uchar		*dstChar;
	__global cl_uint		*dstInt;

	size_t	srcAddr, srcFirstChars, srcIntLen, srcRest;
	size_t	dstAddr, dstFirstChars, dstIntLen, dstRest;
	int		i, j;

	srcAddr			= (size_t)__src;
	srcFirstChars	= ((sizeof(cl_uint) - (srcAddr & alignMask)) & alignMask);
	srcFirstChars	= min(srcFirstChars, len);
	srcIntLen		= (len - srcFirstChars) / sizeof(cl_uint);
	srcRest			= (len - srcFirstChars) & alignMask;

	dstAddr			= (size_t)__dst;
	dstFirstChars	= ((sizeof(cl_uint) - (dstAddr & alignMask)) & alignMask);
	dstFirstChars	= min(dstFirstChars, len);
	dstIntLen		= (len - dstFirstChars) / sizeof(cl_uint);
	dstRest			= (len - dstFirstChars) & alignMask;


	/* load the first 0-3 charactors */
	srcChar = (__global const cl_uchar *)__src;
	for(i=0; i<srcFirstChars; i++) {
		buf8 = buf8 | (srcChar[i] << (BITS_PER_BYTE * bufCnt));
		bufCnt ++;
	}
	srcInt = (__global const cl_uint *)&srcChar[srcFirstChars];

	if(0 < srcIntLen) {
		/* load the first 1 word(4 bytes) */
		buf8 = buf8 | ((cl_ulong)srcInt[0] << (bufCnt * BITS_PER_BYTE));
		bufCnt += sizeof(cl_uint);

		/* store the first 0-3 charactors */
		dstChar = (__global cl_uchar *)__dst;
		for(j=0; j<dstFirstChars; j++) {
			dstChar[j] = (cl_uchar)buf8;
			buf8 >>= BITS_PER_BYTE;
			bufCnt --;
		}

		/* store 32bit, if buffered data is larger than 32bit */
		dstInt = (__global cl_uint *)&dstChar[dstFirstChars];
		j	   = 0;
		if(sizeof(cl_uint) <= bufCnt) {
			dstInt[j++] = (cl_uint)buf8;
			buf8 >>= (BITS_PER_BYTE * sizeof(cl_uint));
			bufCnt -= sizeof(cl_uint);
		}

		/* copy body */
		for(i=1; i<srcIntLen; i++) {
			buf8 = buf8 | ((cl_ulong)srcInt[i] << (bufCnt * BITS_PER_BYTE));
			dstInt[j++] = (cl_uint)buf8;
			buf8 >>= (BITS_PER_BYTE * sizeof(cl_uint));
		}
	}

	/* load the last 0-3 charactors */
	srcChar = (__global const cl_uchar *)&srcInt[srcIntLen];
	for(i=0; i<srcRest; i++) {
		buf8 = buf8 | ((cl_ulong)srcChar[i] << (BITS_PER_BYTE * bufCnt));
		bufCnt ++;
	}

	if(0 == srcIntLen) {
	    /* store the first 0-3 charactors */
		dstChar = (__global cl_uchar *)__dst;
		for(j=0; j<dstFirstChars; j++) {
			dstChar[j] = (cl_uchar)buf8;
			buf8 >>= BITS_PER_BYTE;
			bufCnt --;
		}
		dstInt = (__global cl_uint *)&dstChar[dstFirstChars];
		j = 0;
	}

	/* store rest charactors */
	if(sizeof(cl_uint) <= bufCnt) {
		dstInt[j++] = (cl_uint)buf8;
		buf8 >>= (BITS_PER_BYTE * sizeof(cl_uint));
		bufCnt -= sizeof(cl_uint);
	}
	dstChar = (__global cl_uchar *)&dstInt[j];
	for(j=0; j<dstRest; j++) {
		dstChar[j] = (cl_uchar)buf8;
		buf8 >>= BITS_PER_BYTE;
		bufCnt --;
	}

	return __dst;
}
#endif

/*
 * arithmetic_stairlike_add
 *
 * A utility routine to calculate sum of values when we have N items and 
 * want to know sum of items[i=0...k] (k < N) for each k, using reduction
 * algorithm on local memory (so, it takes log2(N) + 1 steps)
 *
 * The 'my_value' argument is a value to be set on the items[get_local_id(0)].
 * Then, these are calculate as follows:
 *
 *           init   1st      2nd         3rd         4th
 *           state  step     step        step        step
 * items[0] = X0 -> X0    -> X0       -> X0       -> X0
 * items[1] = X1 -> X0+X1 -> X0+X1    -> X0+X1    -> X0+X1
 * items[2] = X2 -> X2    -> X0+...X2 -> X0+...X2 -> X0+...X2
 * items[3] = X3 -> X2+X3 -> X0+...X3 -> X0+...X3 -> X0+...X3
 * items[4] = X4 -> X4    -> X4       -> X0+...X4 -> X0+...X4
 * items[5] = X5 -> X4+X5 -> X4+X5    -> X0+...X5 -> X0+...X5
 * items[6] = X6 -> X6    -> X4+...X6 -> X0+...X6 -> X0+...X6
 * items[7] = X7 -> X6+X7 -> X4+...X7 -> X0+...X7 -> X0+...X7
 * items[8] = X8 -> X8    -> X8       -> X8       -> X0+...X8
 * items[9] = X9 -> X8+X9 -> X8+9     -> X8+9     -> X0+...X9
 *
 * In Nth step, we split the array into 2^N blocks. In 1st step, a unit
 * containt odd and even indexed items, and this logic adds the last value
 * of the earlier half onto each item of later half. In 2nd step, you can
 * also see the last item of the earlier half (item[1] or item[5]) shall
 * be added to each item of later half (item[2] and item[3], or item[6]
 * and item[7]). Then, iterate this step until 2^(# of steps) less than N.
 *
 * Note that supplied items[] must have at least sizeof(cl_uint) *
 * get_local_size(0), and its contents shall be destroyed.
 * Also note that this function internally use barrier(), so unable to
 * use within if-blocks.
 */
static cl_uint
arithmetic_stairlike_add(cl_uint my_value, __local cl_uint *items,
						 __private cl_uint *total_sum)
{
	size_t		wkgrp_sz = get_local_size(0);
	cl_int		i, j;

	/* set initial value */
	items[get_local_id(0)] = my_value;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (i=1; wkgrp_sz > 0; i++, wkgrp_sz >>= 1)
	{
		/* index of last item in the earlier half of each 2^i unit */
		j = (get_local_id(0) & ~((1 << i) - 1)) | ((1 << (i-1)) - 1);

		/* add item[j] if it is later half in the 2^i unit */
		if ((get_local_id(0) & (1 << (i - 1))) != 0)
			items[get_local_id(0)] += items[j];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (total_sum)
		*total_sum = items[get_local_size(0) - 1];
	return items[get_local_id(0)] - my_value;
}

/*
 * kern_varlena_to_toast
 *
 * A common routine to copy the supplied varlena variable into another
 * column-store and toast buffer.
 * We assume the supplied toast buffer is valid and flat (that means
 * this toast buffer does not have coldir), then usage of toast-buffer
 * shall be increased according to the usage.
 * ktoast->usage is one of the hottest point for atomic operation, so
 * we try to assign all the needed region at once, thus, this routine
 * internally takes reduction arithmetic, but unavailable to call
 * under if-block.
 * It returns an offset value from the head of supplied ktoast buffer,
 * or 0 if NULL or any other errors.
 */
static cl_uint
kern_varlena_to_toast(__private int *errcode,
					  __global kern_toastbuf *ktoast,
					  __global varlena *vl_datum,
					  __local void *workbuf)
{
	__local cl_uint	base;
	cl_uint			vl_ofs;
	cl_uint			vl_len;
	cl_uint			toast_len;

	if (vl_datum)
		vl_len = VARSIZE_ANY(vl_datum);
	else
		vl_len = 0;	/* null value */

	/*
	 * To avoid storm of atomic_add on the ktoast_dst->usage,
	 * we once calculate required total length of toast buffer
	 * on local memory, then increment ktoast_dst->usage using
	 * atomic_add. Then, if still toast buffer is available, 
	 * we will copy the varlena variable.
	 */
	vl_ofs = arithmetic_stairlike_add(INTALIGN(vl_len),
									  workbuf, &toast_len);
	if (get_local_id(0) == 0)
	{
		/*
		 * to avoid overflow, we check current toast usage
		 * prior to atomic_add, and skip and set error if
		 * toast buffer obviously has no space any more.
		 */
		if (ktoast->usage + toast_len > ktoast->length)
			base = 0;
		else
		{
			base = atomic_add(&ktoast->usage, toast_len);
			if (base + toast_len > ktoast->length)
				base = 0;	/* it's overflow! */
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (!vl_datum)
		return 0;
	if (base == 0)
	{
		/*
		 * unable to set non-NULL value on the ktoast-buffer
		 * being overflowed.
		 */
		STROM_SET_ERROR(errcode, StromError_DataStoreNoSpace);
		return 0;
	}
	vl_ofs += base;

	memcpy((__global char *)ktoast + vl_ofs, vl_datum, INTALIGN(vl_len));

	return vl_ofs;
}

/*
 * kern_column_to_column
 *
 * A common routine to copy a record in column store into another column-store
 * It copies a record specified by index_src in the source column-store into
 * the location pointed by index_dst in the destination column-store.
 *
 * We assume the destination column-store must:
 * - have same number of same typed columns, with same not-null restriction
 * - have enough rooms to store the record.
 *  (usually, kcs_src and kcs_dst have same number of rooms to store values)
 * - have enough capacity to store variable-length values.
 *
 * If unable to copy a record into one, it set an error code to inform the
 * host code the problem, to recover it.
 */
static void
kern_column_to_column(__private int *errcode,
					  cl_uint kcs_ncols,
					  __global kern_column_store *kcs_dst,
					  __global kern_toastbuf *ktoast_dst,
					  cl_uint index_dst,
					  __global kern_column_store *kcs_src,
					  __global kern_toastbuf *ktoast_src,
					  cl_uint index_src,
					  __local void *workbuf)
{
	kern_colmeta	scmeta;
	kern_colmeta	dcmeta;
	cl_int			i, j;

	/*
	 * number of columns to be moved has to be larger than or equal with
	 * number of columns being constructed on the column store
	 */
	if (kcs_dst->ncols < kcs_ncols || kcs_src->ncols < kcs_ncols)
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
		return;
	}

	/*
	 * If we adopt local-memory based reduction operation later,
	 * the check below also has to be revised, because barrier()
	 * synchronization requires to run all the threads critical
	 * section.
	 */
	if (index_src >= kcs_src->nrooms || index_dst >= kcs_dst->nrooms)
		return;

	for (i=0; i < kcs_ncols; i++)
	{
		kern_colmeta	scmeta = kcs_src->colmeta[i];
	    kern_colmeta	dcmeta = kcs_dst->colmeta[i];
		__global char  *saddr;
		__global char  *daddr;
		cl_bool			isnull;

		saddr = (__global char *)kcs_src + scmeta.cs_ofs;
		daddr = (__global char *)kcs_dst + dcmeta.cs_ofs;

		if (scmeta.attnotnull)
			isnull = false;
		else
		{
			isnull = att_isnull(index_src, saddr);
			saddr += STROMALIGN((kcs_src->nrooms +
								 BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		}
		/* XXX - we may need to adjust alignment */
		saddr += (scmeta.attlen > 0
				  ? scmeta.attlen
				  : sizeof(cl_uint)) * index_src;

		/*
		 * set NULL bit, if needed.
		 *
		 * XXX - I'm concern it takes atomic operation on DRAM, however,
		 * we have no way to avoid because here is no predication about
		 * dindex to be supplied.
		 */
		if (!dcmeta.attnotnull)
		{
			__global cl_uint   *nullmap = (((__global cl_uint *)daddr) +
										   (index_dst / (sizeof(cl_uint) *
														 BITS_PER_BYTE)));
			cl_uint				shift = (index_dst & (sizeof(cl_uint) *
													  BITS_PER_BYTE - 1));
			if (!isnull)
				atomic_or(nullmap, 1 << shift);
			else
				atomic_and(nullmap, ~(1 << shift));

			/* adjust offset for nullmap */
			daddr += STROMALIGN((kcs_dst->nrooms +
								 BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		}
		else if (isnull)
		{
			/* cannot set a NULL on column with attnotnull attribute */
			STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
			return;
		}

		/* XXX - we may need to adjust alignment */
		daddr += (dcmeta.attlen > 0
				  ? dcmeta.attlen
				  : sizeof(cl_uint)) * index_dst;
		/*
		 * Right now, we have only 8, 4, 2 and 1 bytes width for
		 * fixed length variables
		 */
		if (dcmeta.attlen == sizeof(cl_ulong))
			*((__global cl_ulong *)daddr) = *((__global cl_ulong *)saddr);
		else if (dcmeta.attlen == sizeof(cl_uint))
			*((__global cl_uint *)daddr) = *((__global cl_uint *)saddr);
		else if (dcmeta.attlen == sizeof(cl_ushort))
			*((__global cl_ushort *)daddr) = *((__global cl_ushort *)saddr);
		else if (dcmeta.attlen == sizeof(cl_uchar))
			*((__global cl_uchar *)daddr) = *((__global cl_uchar *)saddr);
		else if (dcmeta.attlen > 0)
			memcpy(daddr, saddr, dcmeta.attlen);
		else
		{
			__global varlena   *vl_datum;
			cl_uint				vl_offset = *((__global cl_uint *)saddr);

			if (ktoast_src->length == TOASTBUF_MAGIC)
				vl_offset += ktoast_src->coldir[i];
			vl_datum = (__global varlena *)
				((__global char *)ktoast_src + vl_offset);
			*((__global cl_uint *)daddr) = kern_varlena_to_toast(errcode,
																 ktoast_dst,
																 vl_datum,
																 workbuf);
		}
	}
}

/*
 * Common function to translate a row-store into column-store.
 *
 * The kern_row_to_column() translates records on the supplied row-store
 * into columns array on the supplied column-store. 
 * Data format using HeapTuple takes unignorable cost if we try to calculate
 * offset of the referenced value from head of the tuple for each reference.
 * Our column-oriented format does not take such a limitation, so our logic
 * prefers column-store anyway.
 * You need to pay attention three prerequisites of this function.
 * 1. The caller must set up header portion of kern_column_store prior to
 *    kernel invocation. This function assumes each kern_colmeta entry has
 *    appropriate offset from the head.
 * 2. The caller must allocate sizeof(cl_uint) * get_local_size(0) bytes
 *    of local memory, and gives it as the last argument. Null-bitmap takes
 *    bit-operation depending on the neighbor thread.
 * 3. Local work-group size has to be multiplexer of 32.
 */
static void
kern_row_to_column(__private cl_int *errcode,
				   __global cl_char *attreferenced,
				   __global kern_row_store *krs,
				   size_t krs_index,
				   __global kern_column_store *kcs,
				   __global kern_toastbuf *ktoast,
				   size_t kcs_offset,
				   size_t kcs_nitems,
				   __local cl_uint *workbuf)
{
	__global rs_tuple  *rs_tup;
	size_t		kcs_index = kcs_offset + get_local_id(0);
	cl_uint		ncols = krs->ncols;
	cl_uint		offset;
	cl_uint		i, j, k;

	/* fetch a rs_tuple on the row-store */
	if (get_local_id(0) < kcs_nitems)
		rs_tup = kern_rowstore_get_tuple(krs, krs_index);
	else
		rs_tup = NULL;
	offset = (rs_tup != NULL ? rs_tup->data.t_hoff : 0);

	for (i=0, j=0; i < ncols; i++)
	{
		kern_colmeta	rcmeta = krs->colmeta[i];
		kern_colmeta	ccmeta = kcs->colmeta[j];
		__global char  *dest;
		__global char  *src;
		cl_char			is_referenced;

		if (!rs_tup || ((rs_tup->data.t_infomask & HEAP_HASNULL) != 0 &&
						att_isnull(i, rs_tup->data.t_bits)))
		{
			src = NULL;		/* means, this value is NULL */
		}
		else
		{
			if (rcmeta.attlen > 0)
				offset = TYPEALIGN(rcmeta.attalign, offset);
			else if (!VARATT_NOT_PAD_BYTE((uintptr_t)&rs_tup->data + offset))
				offset = TYPEALIGN(rcmeta.attalign, offset);
			src = ((__global char *)&rs_tup->data) + offset;

			/*
			 * Increment offset, according to the logic of
			 * att_addlength_pointer but no cstring support in kernel
			 */
			offset += (rcmeta.attlen > 0 ?
					   rcmeta.attlen :
					   VARSIZE_ANY(src));
		}
		/* Move to the next column unless it is not actually referenced. */
		is_referenced = attreferenced[i];
		if (!is_referenced)
			continue;

		/*
		 * calculation of destination address on the column-store,
		 * if thread is responsible to a particular entry.
		 */
		if (get_local_id(0) < kcs_nitems)
		{
			dest = ((__global char *)kcs) + ccmeta.cs_ofs;
			if (!ccmeta.attnotnull)
				dest += STROMALIGN((kcs->nrooms + 7) >> 3);

			dest += kcs_index * (ccmeta.attlen > 0
								 ? ccmeta.attlen
								 : sizeof(cl_uint));
		}
		else
			dest = NULL;	/* means, it is out of range in kcs */

		/*
		 * Copy a datum in rs_tuple into an item of column-array
		 * in kern_column_store. In case of variable length-field,
		 * column-array have only offset on the associated toast
		 * buffer. Here are two options. If caller does not use
		 * dedicated toast buffer, we deal with the source row-
		 * store as a buffer to store variable-length field.
		 * Elsewhere, the toast body shall be copied.
		 */
		if (ccmeta.attlen == sizeof(cl_char))
		{
			if (dest)
				*((__global cl_char *)dest)
					= (!src ? 0 : *((__global cl_char *)src));
		}
		else if (ccmeta.attlen == sizeof(cl_short))
		{
			if (dest)
				*((__global cl_short *)dest)
					= (!src ? 0 : *((__global cl_short *)src));
		}
		else if (ccmeta.attlen == sizeof(cl_int))
		{
			if (dest)
				*((__global cl_int *)dest)
					= (!src ? 0 : *((__global cl_int *)src));
		}
		else if (ccmeta.attlen == sizeof(cl_long))
		{
			if (dest)
				*((__global cl_long *)dest)
					= (!src ? 0 : *((__global cl_long *)src));
		}
		else if (ccmeta.attlen > 0)
		{
			if (dest && src)
				memcpy(dest, src, ccmeta.attlen);
		}
		else if (!ktoast)
		{
			/*
			 * In case of variable-length field that uses row-store to
			 * put body of the variable, all we need to do is putting
			 * an offset of the value.
			 */
			if (dest)
				*((__global cl_uint *)dest)
					= (!src ? 0 : (cl_uint)((uintptr_t)src -
											(uintptr_t)krs));
		}
		else
		{
			/*
			 * In case when we copy the variable-length field into
			 * the supplied kern_toastbuf, let's use the routeine below.
			 * This routine has to be called by all the threads in
			 * a workgroup because it takes reduction steps internally,
			 * to avoid expensive atomic operation on ktoast->usage counter
			 * (that tends to be the hottest  point of atomic operation).
			 */
			cl_uint		vl_ofs
				= kern_varlena_to_toast(errcode,
										ktoast,
										(__global varlena *)src,
										workbuf);
			if (dest != NULL)
				*((__global cl_uint *)dest) = vl_ofs;
		}

		/*
		 * Calculation of nullmap if this column is the target to be moved.
		 * Because it takes per bit operation using interaction with neighbor
		 * work-item, we use local working memory for reduction.
		 */
		if (!ccmeta.attnotnull)
		{
			cl_uint		wmask = sizeof(cl_uint) * BITS_PER_BYTE - 1;
			cl_uint		shift = (kcs_offset & wmask);

			/* reduction to calculate nullmap with 32bit width */
			workbuf[get_local_id(0)]
				= (!src ? 0 : (1 << (get_local_id(0) & wmask)));
			barrier(CLK_LOCAL_MEM_FENCE);
			for (k=2; k <= sizeof(cl_uint) * BITS_PER_BYTE; k <<= 1)
			{
				if ((get_local_id(0) & (k-1)) == 0)
					workbuf[get_local_id(0)]
						|= workbuf[get_local_id(0) + (k>>1)];
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			/* responsible thread put calculated nullmap */
			if (get_local_id(0) < kcs_nitems &&
				(shift > 0
				 ? (get_local_id(0) == 0 ||
					(get_local_id(0) & wmask) == 32 - shift)
				 : (get_local_id(0) & wmask) == 0))
			{
				__global cl_uint *p_nullmap;
				cl_uint		bitmap;
				cl_uint		mask;
				cl_uint		bi = get_local_id(0) & ~wmask;
				cl_uint		width = (kcs_nitems - bi);

				bitmap = workbuf[bi];

				p_nullmap = (__global cl_uint *)((__global char *)kcs +
												 ccmeta.cs_ofs);
				p_nullmap += (kcs_offset + get_local_id(0)) >> 5;

				if (shift > 0 && get_local_id(0) == 0)
				{
					/* special treatment if unaligned head */
					if (width < sizeof(cl_uint) * BITS_PER_BYTE - shift)
						mask = ((1 << width) - 1) << shift;
					else
						mask = ~((1 << shift) - 1);
					atomic_and(p_nullmap, ~mask);
					atomic_or(p_nullmap, (bitmap << shift) & mask);
				}
				else
				{
					if (shift > 0)
						width -= sizeof(cl_uint) * BITS_PER_BYTE - shift;

					bitmap >>= (sizeof(cl_uint) * BITS_PER_BYTE) - shift;
					if (width > shift)
						bitmap |= workbuf[bi + (sizeof(cl_uint) *
												BITS_PER_BYTE)] << shift;
					if (width < sizeof(cl_uint) * BITS_PER_BYTE)
					{
						mask = (1 << width) - 1;
						atomic_and(p_nullmap, ~mask);
						atomic_or(p_nullmap, bitmap & mask);
					}
					else
					{
						*p_nullmap = bitmap;
					}
				}
			}
		}
		/*
		 * NOTE: end of reference marker - if attreferenced[i] has negative
		 * value, neither zero nor positive, it means no more columns are
		 * referenced in this row. so, we can break the loop earlier.
		 * Also note that gpuXXX implementation may append columns for
		 * internal usage, like rindex of sort, thus j != kcs->ncols is
		 * not a good condition to break the loop.
		 */
		if (is_referenced < 0)
			break;

		j++;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}
#endif /* OPENCL_DEVICE_CODE */
#ifdef OPENCL_DEVICE_CODE
/*
 * kern_writeback_error_status
 *
 * It set thread-local error code on the variable on global memory.
 */
static void
kern_writeback_error_status(__global cl_int *error_status,
							int own_errcode,
							__local void *workmem)
{
	__local cl_int *error_temp = workmem;
	size_t		wkgrp_sz;
	size_t		mask;
	size_t		buddy;
	cl_int		errcode_0;
	cl_int		errcode_1;
	cl_int		i;

	error_temp[get_local_id(0)] = own_errcode;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (i=1, wkgrp_sz = get_local_size(0);
		 wkgrp_sz > 0;
		 i++, wkgrp_sz >>= 1)
	{
		mask = (1 << i) - 1;

		if ((get_local_id(0) & mask) == 0)
		{
			buddy = get_local_id(0) + (1 << (i - 1));

			errcode_0 = error_temp[get_local_id(0)];
			errcode_1 = (buddy < get_local_size(0)
						 ? error_temp[buddy]
						 : StromError_Success);
			if (!StromErrorIsSignificant(errcode_0) &&
				StromErrorIsSignificant(errcode_1))
				error_temp[get_local_id(0)] = errcode_1;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/*
	 * It writes back a statement level error, unless no other workgroup
	 * put a significant error status.
	 * This atomic operation set an error code, if it is still
	 * StromError_Success.
	 */
	errcode_0 = error_temp[0];
	if (get_local_id(0) == 0 && StromErrorIsSignificant(errcode_0))
	{
		atomic_cmpxchg(error_status, StromError_Success, errcode_0);
	}
}

/* ------------------------------------------------------------
 *
 * Declarations of common built-in functions
 *
 * ------------------------------------------------------------
 */

/*
 * Functions for BooleanTest
 */
static inline pg_bool_t
pgfn_bool_is_true(__private cl_int *errcode, pg_bool_t result)
{
	result.value = (!result.isnull && result.value);
	result.isnull = false;
	return result;
}

static inline pg_bool_t
pgfn_bool_is_not_true(__private cl_int *errcode, pg_bool_t result)
{
	result.value = (result.isnull || !result.value);
	result.isnull = false;
	return result;
}

static inline pg_bool_t
pgfn_bool_is_false(__private cl_int *errcode, pg_bool_t result)
{
	result.value = (!result.isnull && !result.value);
	result.isnull = false;
	return result;
}

static inline pg_bool_t
pgfn_bool_is_not_false(__private cl_int *errcode, pg_bool_t result)
{
	result.value = (result.isnull || result.value);
	result.isnull = false;
	return result;
}

static inline pg_bool_t
pgfn_bool_is_unknown(__private cl_int *errcode, pg_bool_t result)
{
	result.value = result.isnull;
	result.isnull = false;
	return result;
}

static inline pg_bool_t
pgfn_bool_is_not_unknown(__private cl_int *errcode, pg_bool_t result)
{
	result.value = !result.isnull;
	result.isnull = false;
	return result;
}

/*
 * Functions for BoolOp (EXPR_AND and EXPR_OR shall be constructed on demand)
 */
static inline pg_bool_t
pgfn_boolop_not(__private cl_int *errcode, pg_bool_t result)
{
	result.value = !result.value;
	/* if null is given, result is also null */
	return result;
}

#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_COMMON_H */
