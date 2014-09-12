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
#define __constant	/* address space qualifier is noise on host */
#endif

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
#define bitmaplen(NATTS) (((int)(NATTS) + BITS_PER_BYTE - 1) / BITS_PER_BYTE)

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

/*
 * information stored in t_infomask2:
 */
#define HEAP_NATTS_MASK			0x07FF	/* 11 bits for number of attributes */
#define HEAP_KEYS_UPDATED		0x2000	/* tuple was updated and key cols
										 * modified, or tuple deleted */
#define HEAP_HOT_UPDATED		0x4000	/* tuple was HOT-updated */
#define HEAP_ONLY_TUPLE			0x8000	/* this is heap-only tuple */
#define HEAP2_XACT_MASK			0xE000	/* visibility-related bits */

#endif

/*
 * alignment for pg-strom
 */
#define STROMALIGN_LEN			16
#define STROMALIGN(LEN)			TYPEALIGN(STROMALIGN_LEN,LEN)
#define STROMALIGN_DOWN(LEN)	TYPEALIGN_DOWN(STROMALIGN_LEN,LEN)

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
	struct {
		cl_char		attnotnull;	/* true, if always not null */
		cl_char		attalign;	/* type alignment */
		cl_short	attlen;		/* length of type */
	} colmeta[FLEXIBLE_ARRAY_MEMBER];	/* simplified kern_colmeta */
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
 * kern_data_store
 *
 * It stores row- and column-oriented values in the kernel space.
 *
 * +----------------------------------+
 * | length                           |
 * +----------------------------------+
 * | ncols                            |
 * +----------------------------------+
 * | nitems                           |
 * +----------------------------------+
 * | nrooms                           |
 * +----------------------------------+
 * | is_column                        |
 * +----------------------------------+
 * | __padding__[7]                   |
 * +----------------------------------+
 * | colmeta[0]                       |
 * | colmeta[1]                       |
 * |   :                              |
 * | colmeta[M-1]                     |
 * +----------------+-----------------+ <--- aligned by STROMALIGN()
 * | <row-format>   | <column-format> |
 * +----------------+-----------------+
 * | rowitems[0]    | column-data of  |
 * | rowitems[1]    | the 1st column  |
 * | rowitems[2]    | +---------------+
 * |    :           | | null bitmap   |
 * |    :           | +---------------+
 * | rowitems[N-1]  | | values array  |
 * |                | | of the 1st    |
 * +----------------+ | column        |
 * |    :           +-+---------------+ <--- cs_offset points starting
 * | alignment to   | column-data of  |      offset of the column-data
 * | BLCKSZ         | the 2nd column  |
 * +----------------+ +---------------+
 * | blocks[0]      | | null bitmap   |
 * | PageHeaderData | +---------------+
 * |      :         | | values array  |
 * | pd_linep[]     | | of the 2nd    |
 * |      :         | | column        |
 * +----------------+-+---------------+
 * | blocks[1]      |       :         |
 * | PageHeaderData |       :         |
 * |      :         |       :         |
 * | pd_linep[]     |       :         |
 * |      :         |       :         |
 * +----------------+-----------------+
 * |      :         | column-data of  |
 * |      :         | the last column |
 * +----------------+ +---------------+
 * | blocks[1]      | | null bitmap   |
 * | PageHeaderData | +---------------+
 * |      :         | | values array  |
 * | pd_linep[]     | | of the last   |
 * |      :         | | column        |
 * +----------------+-+---------------+
 */
typedef struct {
	/* true, if column never has NULL (thus, no nullmap required) */
	cl_char			attnotnull;
	/* alignment; 1,2,4 or 8, not characters in pg_attribute */
	cl_char			attalign;
	/* length of attribute */
	cl_short		attlen;
	union {
		/* 0 is valid value for neither rs_attnum (for row-store) nor
		 * cs_offset (for column-store).
		 */
		cl_uint		attvalid;
		/* If row-store, it is attribute number within row-store.
		 * (Note that attribute number is always positive, whole-
		 * -row references and system columns are not supported.)
		 */
		cl_uint		rs_attnum;
		/* If column-store, offset to nullmap and column-array from the
		 * head of kern_data_store.
		 */
		cl_uint		cs_offset;
	};
} kern_colmeta;

/*
 * kern_rowitem packs an index of block and tuple item. 
 * block_id points a particular block in the block array of this data-store.
 * item_id points a particular tuple item within the block.
 */
typedef struct {
	cl_ushort		block_id;
	cl_ushort		item_id;
} kern_rowitem;

typedef struct {
	cl_uint			length;	/* length of this kernel data store */
	cl_uint			ncols;	/* number of columns in this store */
	cl_uint			nitems; /* number of rows in this store */
	cl_uint			nrooms;	/* number of available rows in this store */
	cl_uint			nblocks;/* number of blocks in this store, if row-store.
							 * Elsewhere, always 0. */
	cl_char			is_column;/* if true, data store is column-format */
	cl_char			__padding__[3];
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER]; /* metadata of columns */
} kern_data_store;

#define KERN_DATA_STORE_ROWBLOCKS(kds)					\
	((__global PageHeader)								\
	 ((__global cl_char *)(kds) +						\
	  ((STROMALIGN(offsetof(kern_data_store,			\
							colmeta[(kds)->ncols])) +	\
		STROMALIGN(sizeof(kern_rowitem) *				\
				   (kds->nitems)) + BLCKSZ - 1) & ~(BLCKSZ - 1))))

#define KERN_DATA_STORE_ROWITEMS(kds)				\
	((__global kern_rowitem *)						\
	 ((__global cl_char *)(kds) +					\
	  STROMALIGN(offsetof(kern_data_store,			\
						  colmeta[(kds)->ncols]))))

/*
 * kern_toastbuf
 *
 * A varlena datum is represented as an offset from the head of toast-buffer,
 * and data contents are actually stored within this toast-buffer.
 */
typedef struct {
#ifndef OPENCL_DEVICE_CODE
	dlist_node		dnode;	/* host only; used to chain multiple toastbuf */
#else
	cl_char			__padding__[2 * HOSTPTRLEN];
#endif
	cl_uint			length;	/* length of toast-buffer */
	cl_uint			usage;	/* usage of toast-buffer */
	cl_char			data[FLEXIBLE_ARRAY_MEMBER];
} kern_toastbuf;

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
kparam_get_value(__global kern_parambuf *kparams, cl_uint pindex)
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
	cl_uint		nrels;		/* number of relations to be appeared */
	cl_uint		nrooms;		/* max number of results rooms */
	cl_uint		nitems;		/* number of results being written */
	cl_int		errcode;	/* chunk-level error */
	cl_char		has_rechecks;
	cl_char		all_visible;
	cl_char		__padding__[2];
	cl_int		results[FLEXIBLE_ARRAY_MEMBER];
} kern_resultbuf;

/*
 * kern_row_map
 *
 * It informs kernel code which rows are valid, and which ones are not, if
 * kern_data_store contains mixed 
 */
typedef struct {
	cl_int		nvalids;	/* # of valid rows. -1 means all visible */
	cl_int		rindex[FLEXIBLE_ARRAY_MEMBER];
} kern_row_map;

#if 0

typedef struct {
	/* same as kern_row_map but works for column references */

} kern_column_map;

#endif

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

/* forward declaration of access interface to kern_data_store */
static __global void *kern_get_datum(__global kern_data_store *kds,
									 __global kern_toastbuf *ktoast,
									 cl_uint colidx, cl_uint rowidx);

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
	pg_##NAME##_vref(__global kern_data_store *kds,			\
					 __global kern_toastbuf *ktoast,		\
					 __private int *errcode,				\
					 cl_uint colidx,						\
					 cl_uint rowidx)						\
	{														\
		pg_##NAME##_t result;								\
		__global BASE *addr									\
			= kern_get_datum(kds,ktoast,colidx,rowidx);		\
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
	pg_##NAME##_param(__global kern_parambuf *kparams,		\
					  __private int *errcode,				\
					  cl_uint param_id)						\
	{														\
		pg_##NAME##_t result;								\
		__global BASE *addr;								\
															\
		if (param_id < kparams->nparams &&					\
			kparams->poffset[param_id] > 0)					\
		{													\
			result.value = *((__global BASE *)				\
							 ((uintptr_t)kparams +			\
							  kparams->poffset[param_id]));	\
			result.isnull = false;							\
		}													\
		else												\
			result.isnull = true;							\
															\
		return result;										\
	}

#define STROMCL_SIMPLE_NULLTEST_TEMPLATE(NAME)				\
	static pg_bool_t										\
	pgfn_##NAME##_isnull(__private int *errcode,			\
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
	pgfn_##NAME##_isnotnull(__private int *errcode,			\
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
 * kern_get_datum
 *
 * Reference to a particular datum on the supplied kernel data store.
 * It returns NULL, if it is a really null-value in context of SQL,
 * or in case when out of range with error code
 */
static inline __global void *
kern_get_datum_rs(__global kern_data_store *kds,
				  cl_uint colidx, cl_uint rowidx)
{
	__global kern_row_store	*krs;
	__global rs_tuple		*rs_tup;
	cl_uint		i, ncols;
	cl_uint		offset;

	offset = STROMALIGN(offsetof(kern_data_store, colmeta[kds->ncols]));
	krs = (__global kern_row_store *)((__global char *)kds + offset);
	rs_tup = kern_rowstore_get_tuple(krs, rowidx);
	if (!rs_tup)
		return NULL;

	offset = rs_tup->data.t_hoff;
	ncols = (rs_tup->data.t_infomask2 & HEAP_NATTS_MASK);
	/* a shortcut if colidx is obviously out of range */
	if (colidx >= ncols)
		return NULL;

	for (i=0; i < ncols; i++)
	{
		if ((rs_tup->data.t_infomask & HEAP_HASNULL) != 0 &&
			att_isnull(i, rs_tup->data.t_bits))
		{
			if (i == colidx)
				return NULL;
		}
		else
		{
			__global char  *addr;

			if (krs->colmeta[i].attlen > 0)
				offset = TYPEALIGN(krs->colmeta[i].attalign, offset);
			else if (!VARATT_NOT_PAD_BYTE((uintptr_t)&rs_tup->data + offset))
				offset = TYPEALIGN(krs->colmeta[i].attalign, offset);

			addr = ((__global char *)&rs_tup->data) + offset;
			if (i == colidx)
				return addr;

			offset += (krs->colmeta[i].attlen > 0
					   ? krs->colmeta[i].attlen
					   : VARSIZE_ANY(addr));
		}
	}
	return NULL;
}

static __global void *
kern_get_datum_cs(__global kern_data_store *kds,
				  __global kern_toastbuf *ktoast,
				  cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta	cmeta = kds->colmeta[colidx];
	cl_uint			offset;
	__global void  *addr;

	/* why is this column referenced? */
	if (!cmeta.attvalid)
		return NULL;

	offset = cmeta.cs_offset;

	/* elsewhere, reference to the column-array, straight-forward */
	if (!cmeta.attnotnull)
	{
		if (att_isnull(rowidx, (__global char *)kds + offset))
			return NULL;
		offset += STROMALIGN(bitmaplen(kds->nrooms));
	}
	if (cmeta.attlen > 0)
	{
		offset += cmeta.attlen * rowidx;
		addr = (__global void *)((__global char *)kds + offset);
	}
	else
	{
		cl_uint	vl_ofs;

		offset += sizeof(cl_uint) * rowidx;
		vl_ofs = *((__global cl_uint *)((__global char *)kds + offset));
		if (ktoast->length == TOASTBUF_MAGIC)
			vl_ofs += ktoast->coldir[colidx];
		addr = (__global void *)((__global char *)ktoast + vl_ofs);
	}
	return addr;
}

static inline __global void *
kern_get_datum(__global kern_data_store *kds,
			   __global kern_toastbuf *ktoast,
			   cl_uint colidx, cl_uint rowidx)
{
	/* is it out of range? */
	if (colidx >= kds->ncols || rowidx >= kds->nitems)
		return NULL;

	if (!kds->column_form)
		return kern_get_datum_rs(kds,colidx,rowidx);
	else
		return kern_get_datum_cs(kds, ktoast, colidx, rowidx);
}

/*
 * functions to reference variable length variables
 */
STROMCL_SIMPLE_DATATYPE_TEMPLATE(varlena, __global varlena *)

static inline pg_varlena_t
pg_varlena_vref(__global kern_data_store *kds,
				__global kern_toastbuf *ktoast,
				__private int *errcode,
				cl_uint colidx,
				cl_uint rowidx)
{
	pg_varlena_t result;
	__global varlena *vl_val = kern_get_datum(kds,ktoast,colidx,rowidx);

	if (!vl_val)
		result.isnull = true;
	else
	{
		if (VARATT_IS_4B_U(vl_val) || VARATT_IS_1B(vl_val))
		{
			result.isnull = false;
			result.value = vl_val;
		}
		else
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_RowReCheck);
		}
	}
	return result;
}

static inline pg_varlena_t
pg_varlena_param(__global kern_parambuf *kparams,
				 __private int *errcode,
				 cl_uint param_id)
{
	pg_varlena_t	result;
	__global varlena *addr;

	if (param_id < kparams->nparams &&
		kparams->poffset[param_id] > 0)
	{
		__global varlena *val = (__global varlena *)
			((__global char *)kparams + kparams->poffset[param_id]);
		if (VARATT_IS_4B_U(val) || VARATT_IS_1B(val))
		{
			result.value = val;
			result.isnull = false;
		}
		else
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_RowReCheck);
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
	pg_##NAME##_vref(__global kern_data_store *kds,					\
					 __global kern_toastbuf *ktoast,				\
					 __private int *errcode,						\
					 cl_uint colidx,								\
					 cl_uint rowidx)								\
	{																\
		return pg_varlena_vref(kds,ktoast,errcode,colidx,rowidx);	\
	}

#define STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)						\
	static pg_##NAME##_t											\
	pg_##NAME##_param(__global kern_parambuf *kparams,				\
					  __private int *errcode,						\
					  cl_uint param_id)								\
	{																\
		return pg_varlena_param(kparams,errcode,param_id);			\
	}

#define STROMCL_VARLENA_NULLTEST_TEMPLATE(NAME)						\
	static pg_bool_t												\
	pgfn_##NAME##_isnull(__private int *errcode,					\
					   pg_##NAME##_t arg)							\
	{																\
		return pgfn_varlena_isnull(errcode, arg);					\
	}																\
	static pg_bool_t												\
	pgfn_##NAME##_isnotnull(__private int *errcode,					\
						  pg_##NAME##_t arg)						\
	{																\
		return pgfn_varlena_isnotnull(errcode, arg);				\
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

/* A utility function to evaluate pg_bool_t value as if built-in
 * bool variable.
 */
static inline bool
EVAL(pg_bool_t arg)
{
	if (!arg.isnull && arg.value != 0)
		return true;
	return false;
}

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
