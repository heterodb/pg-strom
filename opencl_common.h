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
#define NULL	((__global void *) 0UL)

/* Misc definitions */
#define FLEXIBLE_ARRAY_MEMBER
#define offsetof(TYPE, FIELD)   ((uintptr_t) &((TYPE *)0)->FIELD)
#define lengthof(ARRAY)			(sizeof(ARRAY) / sizeof((ARRAY)[0]))

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
	cl_int		vl_len;
	cl_char		vl_dat[1];
} varlena;
#define VARHDRSZ			((int) sizeof(cl_int))
#define VARDATA(PTR)		(((__global varlena *)(PTR))->vl_dat)
#define VARSIZE(PTR)		(((__global varlena *)(PTR))->vl_len)
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
	(((__global varattrib_4b *) (PTR))->va_4byte.va_header & 0x3FFFFFFF)
#define VARSIZE_1B(PTR) \
	(((__global varattrib_1b *) (PTR))->va_header & 0x7F)
#define VARTAG_1B_E(PTR) \
	(((__global varattrib_1b_e *) (PTR))->va_tag)

#define VARSIZE_ANY(PTR)							\
	(VARATT_IS_1B_E(PTR) ? VARSIZE_EXTERNAL(PTR) :	\
	 (VARATT_IS_1B(PTR) ? VARSIZE_1B(PTR) :			\
	  VARSIZE_4B(PTR)))

#define VARDATA_4B(PTR)		(((varattrib_4b *) (PTR))->va_4byte.va_data)
#define VARDATA_1B(PTR)		(((varattrib_1b *) (PTR))->va_data)
#define VARDATA_ANY(PTR) \
	(VARATT_IS_1B(PTR) ? VARDATA_1B(PTR) : VARDATA_4B(PTR))

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
#define StromError_DivisionByZero		200	/* Division by zero */

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
#endif	/* OPENCL_DEVICE_CODE */

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

/*
 * kern_resultbuf
 *
 * Output buffer to write back calculation results on a parciular chunk.
 * 'errcode' informs a significant error that shall raise an error on
 * host side and abort transactions. 'results' informs row-level status.
 *
 * if 'debug_usage' is not initialized to zero, it means this result-
 * buffer does not have debug buffer.
 */
#define KERN_DEBUG_UNAVAILABLE	0xffffffff
typedef struct {
	cl_uint		nrooms;		/* max number of results rooms */
	cl_uint		nitems;		/* number of results being written */
	cl_uint		debug_nums;	/* number of debug messages */
	cl_uint		debug_usage;/* current usage of debug buffer */
	cl_int		errcode;	/* chunk-level error */
	cl_int		results[FLEXIBLE_ARRAY_MEMBER];
} kern_resultbuf;

/*
 * kern_debug
 *
 * When pg_strom.kernel_debug is enabled, KERNEL_DEBUG_BUFSIZE bytes of
 * debug buffer is allocated on the behind of kern_resultbuf.
 * Usually, it shall be written back to the host with kernel execution
 * results, and will be dumped to the console.
 */
#define KERNEL_DEBUG_BUFSIZE	(4 * 1024 * 1024)	/* 4MB */

typedef struct {
	cl_uint		length;		/* length of this entry; 4-bytes aligned */
	cl_uint		global_ofs;
	cl_uint		global_sz;
	cl_uint		global_id;
	cl_uint		local_sz;
	cl_uint		local_id;
	cl_char		v_class;
	union {
		cl_ulong	v_int;
		cl_double	v_fp;
	} value;
	cl_char		label[FLEXIBLE_ARRAY_MEMBER];
} kern_debug;

#ifdef OPENCL_DEVICE_CODE

#ifdef PGSTROM_KERNEL_DEBUG
static void
pg_kern_debug_int(__global kern_resultbuf *kresult,
				  __constant char *label, size_t label_sz,
				  cl_ulong value, size_t value_sz)
{
	__global kern_debug *kdebug;
	cl_uint		offset;
	cl_uint		length = offsetof(kern_debug, label) + label_sz;
	cl_uint		i;

	if (!kresult)
		return;

	length = TYPEALIGN(sizeof(cl_uint), length);
	offset = atomic_add(&kresult->debug_usage, length);
	if (offset + length >= KERNEL_DEBUG_BUFSIZE)
		return;

	kdebug = (__global kern_debug *)
		((uintptr_t)&kresult->results[kresult->nrooms] + offset);
	kdebug->length = length;                                        \
	kdebug->global_ofs = get_global_offset(0);
	kdebug->global_sz = get_global_size(0);
	kdebug->global_id = get_global_id(0);
	kdebug->local_sz = get_local_size(0);
	kdebug->local_id = get_local_id(0);
	kdebug->v_class = (value_sz == sizeof(cl_char) ? 'c' :
					   (value_sz == sizeof(cl_short) ? 's' :
						(value_sz == sizeof(cl_int) ? 'i' : 'l')));
	kdebug->value.v_int = value;
	for (i=0; i < label_sz; i++)
		kdebug->label[i] = label[i];
	atomic_add(&kresult->debug_nums, 1);
}

static void
pg_kern_debug_fp(__global kern_resultbuf *kresult,
				 __constant char *label, size_t label_sz,
				 cl_double value, size_t value_sz)
{
	__global kern_debug *kdebug;
	cl_uint		offset;
	cl_uint		length = offsetof(kern_debug, label) + label_sz;
	cl_uint		i;

	if (!kresult)
		return;

	length = TYPEALIGN(sizeof(cl_uint), length);
	offset = atomic_add(&kresult->debug_usage, length);
	if (offset + length >= KERNEL_DEBUG_BUFSIZE)
		return;

	kdebug = (__global kern_debug *)
		((uintptr_t)&kresult->results[kresult->nrooms] + offset);
	kdebug->length = length;                                        \
	kdebug->global_ofs = get_global_offset(0);
	kdebug->global_sz = get_global_size(0);
	kdebug->global_id = get_global_id(0);
	kdebug->local_sz = get_local_size(0);
	kdebug->local_id = get_local_id(0);
	kdebug->v_class = (value_sz == sizeof(cl_float) ? 'f' : 'd');
	kdebug->value.v_fp = value;
	for (i=0; i < label_sz; i++)
		kdebug->label[i] = label[i];
	atomic_add(&kresult->debug_nums, 1);
}
__global kern_resultbuf *pgstrom_kresult_buffer = NULL;

#define KDEBUG_INT(label, value)	\
	pg_kern_debug_int(pgstrom_kresult_buffer, \
					  (label), sizeof(label), (value), sizeof(value))
#define KDEBUG_FP(label, value)	\
	pg_kern_debug_fp(pgstrom_kresult_buffer, \
					 (label), sizeof(label), (value), sizeof(value))
#define KDEBUG_INIT(kresult)	\
	do { pgstrom_kresult_buffer = (kresult); } while (0)
#else
#define KDEBUG_INT(label, value)	do {} while(0)
#define KDEBUG_FP(label, value)		do {} while(0)
#define KDEBUG_INIT(kresult)		do {} while(0)
#endif /* PGSTROM_KERNEL_DEBUG */

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
#define KERN_COLMETA_ATTNOTNULL			0x01
#define KERN_COLMETA_ATTREFERENCED		0x02
typedef struct {
	/* set of KERN_COLMETA_* flags */
	cl_uchar		flags;
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
kern_rowstore_get_tuple(__global kern_row_store *krs, cl_uint rindex)
{
	__global cl_uint   *p_offset;

	if (rindex >= krs->nrows)
		return NULL;
	p_offset = kern_rowstore_get_offset(krs);
	if (p_offset[rindex] == 0)
		return NULL;
	return (__global rs_tuple *)((uintptr_t)krs + p_offset[rindex]);
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
#define TOASTBUF_MAGIC		0xffffffff	/* should not be length of buffers */

typedef struct {
	cl_uint			magic;	/* = TOASTBUF_MAGIC */
	cl_uint			ncols;
	cl_uint			coldir[FLEXIBLE_ARRAY_MEMBER];
} kern_toastbuf;

#ifdef OPENCL_DEVICE_CODE

/* template for native types */
#define STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)	\
	typedef struct {								\
		BASE	value;								\
		bool	isnull;								\
	} pg_##NAME##_t;

#define STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)		\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,__global varlena *)

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

#define STROMCL_VARLENA_VARREF_TEMPLATE(NAME)				\
	static pg_##NAME##_t									\
	pg_##NAME##_vref(__global kern_column_store *kcs,		\
					 __global kern_toastbuf *toast,			\
					 __private int *p_errcode,				\
					 cl_uint colidx,						\
					 cl_uint rowidx)						\
	{														\
		pg_##NAME##_t result;								\
		__global cl_uint *p_offset							\
			= kern_get_datum(kcs,colidx,rowidx);			\
															\
		if (!p_offset)										\
			result.isnull = true;							\
		else												\
		{													\
			cl_uint	offset = *p_offset;						\
			__global varlena *val;							\
															\
			if (toast->magic == TOASTBUF_MAGIC)				\
				offset += toast->coldir[colidx];			\
			val = ((__global varlena *)						\
				   ((char *)toast + offset));				\
			if (VARATT_IS_4B_U(val) || VARATT_IS_1B(val))	\
			{												\
				result.isnull = false;						\
				result.value = val;							\
			}												\
			else											\
			{												\
				result.isnull = true;						\
				STROM_SET_ERROR(p_errcode,					\
								StromError_RowReCheck);		\
			}												\
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

#define STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)				\
	static pg_##NAME##_t									\
	pg_##NAME##_param(__global kern_parambuf *kparam,		\
					  __private int *p_errcode,				\
					  cl_uint param_id)						\
	{														\
		pg_##NAME##_t result;								\
		__global varlena *addr;								\
															\
		if (param_id < kparam->nparams &&					\
			kparam->poffset[param_id] > 0)					\
		{													\
			__global varlena *val =	(__global varlena *)	\
				((char *)kparam + kparam->poffset[param_id]); \
			if (VARATT_IS_4B_U(val) || VARATT_IS_1B(val))	\
			{												\
				result.value = val;							\
				result.isnull = false;						\
			}												\
			else											\
			{												\
				result.isnull = true;						\
				STROM_SET_ERROR(p_errcode,                  \
								StromError_RowReCheck);     \
			}												\
		}													\
		else												\
			result.isnull = true;							\
															\
		return result;										\
	}

#define STROMCL_SIMPLE_TYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)

#define STROMCL_VRALENA_TYPE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_VARREF_TEMPLATE(NAME)			\
	STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)


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
kern_row_to_column(__global kern_row_store *krs,
				   __global kern_column_store *kcs,
				   __local cl_char *workbuf)
{
	__global rs_tuple  *rs_tup;
	size_t		global_id = get_global_id(0);
	size_t		local_id = get_local_id(0);
	cl_uint		ncols = krs->ncols;
	cl_uint		offset;
	cl_uint		i, j;

	/* fetch a rs_tuple on the row-store */
	rs_tup = kern_rowstore_get_tuple(krs, global_id);
	offset = (rs_tup != NULL ? rs_tup->data.t_hoff : 0);

	for (i=0, j=0; i < ncols; i++)
	{
		__global kern_colmeta  *rcmeta = &krs->colmeta[i];
		__global kern_colmeta  *ccmeta;
		cl_bool	isnull;

		if (!rs_tup || ((rs_tup->data.t_infomask & HEAP_HASNULL) != 0 &&
						att_isnull(i, rs_tup->data.t_bits)))
			isnull = true;
		else
		{
			__global char  *src;

			isnull = false;

			if (rcmeta->attlen > 0)
				offset = TYPEALIGN(rcmeta->attalign, offset);
			else if (!VARATT_NOT_PAD_BYTE((uintptr_t)&rs_tup->data + offset))
				offset = TYPEALIGN(rcmeta->attalign, offset);
			src = ((__global char *)&rs_tup->data) + offset;

			/*
			 * Increment offset, according to the logic of
			 * att_addlength_pointer but no cstring support in kernel
			 */
			offset += (rcmeta->attlen > 0 ?
					   rcmeta->attlen :
					   VARSIZE_ANY(src));

			if ((rcmeta->flags & KERN_COLMETA_ATTREFERENCED) != 0)
			{
				__global char  *dest;

				ccmeta = &kcs->colmeta[j];

				dest = ((__global char *)kcs) + ccmeta->cs_ofs;

				/* adjust destination address by nullmap, if needed */
				if ((ccmeta->flags & KERN_COLMETA_ATTNOTNULL) == 0)
					dest += STROMALIGN((kcs->nrows + 7) >> 3);
				/* adjust destination address for exact slot on column-array */
				dest += get_global_id(0) * (ccmeta->attlen > 0
											? ccmeta->attlen
											: sizeof(cl_uint));
				/*
				 * Copy a datum from a field of rs_tuple into column-array
				 * of kern_column_store. In case of variable length-field,
				 * column-array will have offset to body of the variable
				 * length field in the toast buffer. The source row-store
				 * will also perform as a toast buffer after the translation.
				 *
				 * NOTE: Also note that we assume fixed length variable has
				 * 1, 2, 4, 8 or 16-bytes length. Elsewhere, it should be
				 * a variable length field.
				 */
				switch (ccmeta->attlen)
				{
					case 1:
						*((__global cl_char *)dest)
							= *((__global cl_char *)src);
						break;
					case 2:
						*((__global cl_short *)dest)
							= *((__global cl_short *)src);
						break;
					case 4:
						*((__global cl_int *)dest)
							= *((__global cl_int *)src);
						break;
					case 8:
						*((__global cl_long *)dest)
							= *((__global cl_long *)src);
						break;
					case 16:
						*((__global cl_long *)dest)
							= *((__global cl_long *)src);
						*(((__global cl_long *)dest) + 1)
							= *(((__global cl_long *)src) + 1);
						break;
					default:
						*((__global cl_uint *)dest)
							= (cl_uint)((uintptr_t)src -
										(uintptr_t)krs);
						break;
				}
			}
		}
		/*
		 * Calculation of nullmap if this column is the target to be moved.
		 * Because it takes per bit operation using interaction with neighbor
		 * work-item, we use local working memory for reduction.
		 */
		if ((rcmeta->flags & KERN_COLMETA_ATTREFERENCED) != 0)
		{
			ccmeta = &kcs->colmeta[j];

			if ((ccmeta->flags & KERN_COLMETA_ATTNOTNULL) == 0)
			{
				workbuf[local_id]
					= (!isnull ? (1 << (local_id & 0x1f)) : 0);
				barrier(CLK_LOCAL_MEM_FENCE);
				if ((local_id & 0x01) == 0)
					workbuf[local_id] |= workbuf[local_id + 1];
				barrier(CLK_LOCAL_MEM_FENCE);
				if ((local_id & 0x03) == 0)
					workbuf[local_id] |= workbuf[local_id + 2];
				barrier(CLK_LOCAL_MEM_FENCE);
				if ((local_id & 0x07) == 0)
					workbuf[local_id] |= workbuf[local_id + 4];
				barrier(CLK_LOCAL_MEM_FENCE);
				if ((local_id & 0x0f) == 0)
					workbuf[local_id] |= workbuf[local_id + 8];
				barrier(CLK_LOCAL_MEM_FENCE);
				if ((local_id & 0x1f) == 0)
					workbuf[local_id] |= workbuf[local_id + 16];
				barrier(CLK_LOCAL_MEM_FENCE);

				/* put a nullmap */
				if ((local_id & 0x1f) == 0 && global_id < krs->nrows)
				{
					__global cl_uint *p_nullmap
						= (__global cl_uint *)((uintptr_t)kcs +
											   ccmeta->cs_ofs);

					p_nullmap[global_id >> 5] = workbuf[local_id];
				}
			}
			j++;
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}

/*
 * kern_get_datum
 *
 * Reference to a particular datum on the supplied column store.
 * It returns NULL, If it is a null-value in context of SQL. Elsewhere,
 * it returns a pointer towards global memory.
 */
static __global void *
kern_get_datum(__global kern_column_store *kcs,
			   cl_uint colidx,
			   cl_uint rowidx)
{
	__global kern_colmeta *colmeta;
	cl_uint		offset;

	if (colidx >= kcs->ncols || rowidx >= kcs->nrows)
		return NULL;

	colmeta = &kcs->colmeta[colidx];
	offset = colmeta->cs_ofs;
	if ((colmeta->flags & KERN_COLMETA_ATTNOTNULL) == 0)
	{
		if (att_isnull(rowidx, (__global char *)kcs + offset))
			return NULL;
		offset += STROMALIGN((kcs->nrows + 7) >> 3);
	}

	if (colmeta->attlen > 0)
		offset += colmeta->attlen * rowidx;
	else
		offset += sizeof(cl_uint) * rowidx;

	return (__global void *)((uintptr_t)kcs + offset);
}

/* ------------------------------------------------------------
 *
 * Declarations of common built-in types and functions
 *
 * ------------------------------------------------------------
 */

/*
 * pg_bool_t is built-in data type
 */
#ifndef PG_BOOL_TYPE_DEFINED
#define PG_BOOL_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(bool, bool)
#endif

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
