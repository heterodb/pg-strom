/*
 * opencl_common.h
 *
 * A common header for OpenCL device code
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
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
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable 

/* NULL definition */
#ifndef NULL
#define NULL	((__global void *) 0UL)
#endif

/* Misc definitions */
#define FLEXIBLE_ARRAY_MEMBER	0
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
typedef cl_ulong	Datum;
#elif HOSTPTRLEN == 4
typedef cl_uint		hostptr_t;
typedef cl_uint		Datum;
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
#define TYPEALIGN_DOWN(ALIGNVAL,LEN) \
	(((uintptr_t) (LEN)) & ~((uintptr_t) ((ALIGNVAL) - 1)))
#define INTALIGN(LEN)			TYPEALIGN(sizeof(cl_int), (LEN))
#define INTALIGN_DOWN(LEN)		TYPEALIGN_DOWN(sizeof(cl_int), (LEN))
#define LONGALIGN(LEN)          TYPEALIGN(sizeof(cl_long), (LEN))
#define LONGALIGN_DOWN(LEN)     TYPEALIGN_DOWN(sizeof(cl_long), (LEN))
#define MAXALIGN(LEN)			TYPEALIGN(MAXIMUM_ALIGNOF, (LEN))
#define MAXALIGN_DOWN(LEN)		TYPEALIGN_DOWN(MAXIMUM_ALIGNOF, (LEN))

/*
 * MEMO: We takes dynamic local memory using cl_ulong data-type because of
 * alignment problem. The nvidia's driver adjust alignment of local memory
 * according to the data type; 1byte for cl_char, 4bytes for cl_uint and
 * so on. Unexpectedly, void * pointer has 1byte alignment even if it is
 * expected to be casted another data types.
 * A pragma option __attribute__((aligned)) didn't work at least driver
 * version 340.xx. So, we declared the local_workmem as cl_ulong * pointer
 * as a workaround.
 */
#define KERN_DYNAMIC_LOCAL_WORKMEM_ARG			\
	__local cl_ulong *__pgstrom_local_workmem
#define LOCAL_WORKMEM		(__local void *)(__pgstrom_local_workmem)

#else	/* OPENCL_DEVICE_CODE */
#include "access/htup_details.h"
#include "storage/itemptr.h"
#define __global	/* address space qualifier is noise on host */
#define __local		/* address space qualifier is noise on host */
#define __private	/* address space qualifier is noise on host */
#define __constant	/* address space qualifier is noise on host */
typedef uintptr_t	hostptr_t;
#endif

/*
 * Error code definition
 */
#define StromError_Success				0	/* OK */
#define StromError_RowFiltered			1	/* Row-clause was false */
#define StromError_CpuReCheck			2	/* To be re-checked by CPU */
#define StromError_ServerNotReady		100	/* OpenCL server is not ready */
#define StromError_BadRequestMessage	101	/* Bad request message */
#define StromError_OpenCLInternal		102	/* OpenCL internal error */
#define StromError_OutOfSharedMemory	105	/* out of shared memory */
#define StromError_OutOfMemory			106	/* out of host memory */
#define StromError_DataStoreCorruption	300	/* Row/Column Store Corrupted */
#define StromError_DataStoreNoSpace		301	/* No Space in Row/Column Store */
#define StromError_DataStoreOutOfRange	302	/* Out of range in Data Store */
#define StromError_DataStoreReCheck		303	/* Row/Column Store be rechecked */
#define StromError_SanityCheckViolation	999	/* SanityCheckViolation */

/* significant error; that abort transaction on the host code */
#define StromErrorIsSignificant(errcode)	((errcode) >= 100 || (errcode) < 0)

#ifdef OPENCL_DEVICE_CODE
/*
 * It sets an error code unless no significant error code is already set.
 * Also, CpuReCheck has higher priority than RowFiltered because CpuReCheck
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
 * Local definition of PageHeaderData and related
 */
typedef cl_ushort LocationIndex;

typedef struct
{
	cl_uint		xlogid;		/* high bits */
	cl_uint		xrecoff;	/* low bits */
} PageXLogRecPtr;

typedef cl_uint	TransactionId;
/*
 * NOTE: ItemIdData is defined using bit-fields in PostgreSQL, however,
 * OpenCL does not support bit-fields. So, we deal with ItemIdData as
 * a 32bit width integer variable, even though lower 15bits are lp_off,
 * higher 15bits are lp_len, and rest of 2bits are lp_flags.
 *
 * FIXME: Host should tell how to handle bitfield layout
 */
typedef cl_uint	ItemIdData;
typedef __global ItemIdData ItemId;

#define ItemIdGetLength(itemId)		(((itemId) >> ITEMID_LENGTH_SHIFT) & 0x7fff)
#define ItemIdGetOffset(itemId)		(((itemId) >> ITEMID_OFFSET_SHIFT) & 0x7fff)
#define ItemIdGetFlags(itemId)		(((itemId) >> ITEMID_FLAGS_SHIFT)  & 0x0003)

typedef struct PageHeaderData
{
    /* XXX LSN is member of *any* block, not only page-organized ones */
    PageXLogRecPtr	pd_lsn;		/* LSN: next byte after last byte of xlog
								 * record for last change to this page */
    cl_ushort		pd_checksum;	/* checksum */
    cl_ushort		pd_flags;		/* flag bits, see below */
    LocationIndex	pd_lower;		/* offset to start of free space */
    LocationIndex	pd_upper;		/* offset to end of free space */
    LocationIndex	pd_special;		/* offset to start of special space */
    cl_ushort		pd_pagesize_version;
	TransactionId	pd_prune_xid;	/* oldest prunable XID, or zero if none */
	ItemIdData		pd_linp[1];		/* beginning of line pointer array */
} PageHeaderData;

typedef __global PageHeaderData *PageHeader;

#define SizeOfPageHeaderData (offsetof(PageHeaderData, pd_linp))

#define PageGetMaxOffsetNumber(page) \
	(((PageHeader) (page))->pd_lower <= SizeOfPageHeaderData ? 0 :	\
	 ((((PageHeader) (page))->pd_lower - SizeOfPageHeaderData)		\
	  / sizeof(ItemIdData)))




/*
 * We need to re-define HeapTupleHeaderData and t_infomask related stuff
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
 * kern_data_store
 *
 * It stores row- and column-oriented values in the kernel space.
 *
 * +---------------------------------------------------+
 * | length                                            |
 * +---------------------------------------------------+
 * | ncols                                             |
 * +---------------------------------------------------+
 * | nitems                                            |
 * +---------------------------------------------------+
 * | nrooms                                            |
 * +---------------------------------------------------+
 * | format                                            |
 * +---------------------------------------------------+
 * | colmeta[0]                                        | aligned to
 * | colmeta[1]                                        | STROMALIGN()
 * |   :                                               |    |
 * | colmeta[M-1]                                      |    V
 * +----------------+-----------------+----------------+-------
 * | <row-format>   | <row-flat-form> |<tupslot-format>|
 * +----------------+-----------------+----------------+
 * | blkitems[0]    | rowitems[0]     | values/isnull  |
 * | blkitems[1]    | rowitems[1]     | pair of the    |
 * |    :           | rowitems[2]     | 1st row        |
 * | blkitems[N-1]  |    :            | +--------------+
 * +----------------+ rowitems[N-2]   | values[0]      |
 * | rowitems[0]    | rowitems[N-1]   |   :            |
 * | rowitems[1]    +-----------------+ values[M-1]    |
 * | rowitems[2]    |                 | +--------------+
 * |    :           +-----------------+ | isnull[0]    |
 * | rowitems[K-1]  | HeapTupleData   | |  :           |
 * +----------------+  tuple[N-1]     | | isnull[M-1]  |
 * | alignment to ..|  <contents>     +-+--------------+
 * | BLCKSZ.........+-----------------+ values/isnull  |
 * +----------------+       :         | pair of the    |
 * | blocks[0]      |       :         | 2nd row        |
 * | PageHeaderData | Row-flat form   | +--------------+
 * |      :         | consumes buffer | | values[0]    |
 * | pd_linep[]     |                 | |  :           |
 * |      :         |                 | | values[M-1]  |
 * +----------------+       ^         | +--------------+
 * | blocks[1]      |       :         | | isnull[0]    |
 * | PageHeaderData |       :         | |   :          |
 * |      :         |       :         | | isnull[M-1]  |
 * | pd_linep[]     |       :         +-+--------------+
 * |      :         |       :         |     :          |
 * +----------------+       :         +----------------+
 * |      :         +-----------------+ values/isnull  |
 * |      :         | HeapTupleData   | pair of the    |
 * +----------------+  tuple[1]       | Nth row        |
 * | blocks[N-1]    |  <contents>     | +--------------+
 * | PageHeaderData +-----------------+ | values[0]    |
 * |      :         | HeapTupleData   | |   :          |
 * | pd_linep[]     |  tuple[0]       | | values[M-1]  |
 * |      :         |  <contents>     | |   :          |
 * +----------------+-----------------+-+--------------+
 */
typedef struct {
	/* true, if column is held by value. Elsewhere, a reference */
	cl_char			attbyval;
	/* alignment; 1,2,4 or 8, not characters in pg_attribute */
	cl_char			attalign;
	/* length of attribute */
	cl_short		attlen;
	/* attribute number */
	cl_short		attnum;
	/* offset of attribute location, if deterministic */
	cl_short		attcacheoff;
} kern_colmeta;

/*
 * kern_rowitem packs an index of block and tuple item. 
 * block_id points a particular block in the block array of this data-store.
 * item_id points a particular tuple item within the block.
 */
typedef union {
	struct {
		cl_ushort	blk_index;		/* if ROW format */
		cl_ushort	item_offset;	/* if ROW format */
	};
	cl_uint			htup_offset;	/* if FLAT_ROW format */
} kern_rowitem;

typedef struct {
#ifdef OPENCL_DEVICE_CODE
	cl_int			buffer;
	hostptr_t		page;
#else
	Buffer			buffer;
	Page			page;
#endif
} kern_blkitem;

#define KDS_FORMAT_ROW			1
#define KDS_FORMAT_ROW_FLAT		2
#define KDS_FORMAT_TUPSLOT		3

typedef struct {
	hostptr_t		hostptr;	/* address of kds on the host */
	cl_uint			length;		/* length of this data-store */
	cl_uint			usage;		/* usage of this data-store */
	cl_uint			ncols;		/* number of columns in this store */
	cl_uint			nitems; 	/* number of rows in this store */
	cl_uint			nrooms;		/* number of available rows in this store */
	cl_uint			nblocks;	/* number of blocks in this store */
	cl_uint			maxblocks;	/* max available blocks in this store */
	cl_char			format;		/* one of KDS_FORMAT_* above */
	cl_char			tdhasoid;	/* copy of TupleDesc.tdhasoid */
	cl_uint			tdtypeid;	/* copy of TupleDesc.tdtypeid */
	cl_int			tdtypmod;	/* copy of TupleDesc.tdtypmod */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER]; /* metadata of columns */
} kern_data_store;

/* access macro for row-format */
#define KERN_DATA_STORE_BLKITEM(kds,blk_index)				\
	(((__global kern_blkitem *)								\
	  ((__global cl_char *)(kds) +							\
	   STROMALIGN(offsetof(kern_data_store,					\
						   colmeta[(kds)->ncols]))))		\
	 + (blk_index))
#define KERN_DATA_STORE_ROWITEM(kds,row_index)					\
	(((__global kern_rowitem *)									\
	  ((__global cl_char *)(kds) +								\
	   STROMALIGN(offsetof(kern_data_store,						\
						   colmeta[(kds)->ncols])) +			\
	   STROMALIGN(sizeof(kern_blkitem) * (kds)->maxblocks)))	\
	 + (row_index))

#define KERN_DATA_STORE_ROWBLOCK(kds,blk_index)							\
	((__global PageHeader)												\
	 ((__global cl_char *)(kds) +										\
	  (TYPEALIGN(BLCKSZ,												\
				 STROMALIGN(offsetof(kern_data_store,					\
									 colmeta[(kds)->ncols])) +			\
				 STROMALIGN(sizeof(kern_blkitem) * (kds)->maxblocks) +	\
				 STROMALIGN(sizeof(kern_rowitem) * (kds)->nitems))		\
	   + BLCKSZ * (blk_index))))

/* access macro for tuple-slot format */
#define KERN_DATA_STORE_VALUES(kds,row_index)				\
	((__global Datum *)										\
	 (((__global cl_char *)(kds) +							\
	   STROMALIGN(offsetof(kern_data_store,					\
						   colmeta[(kds)->ncols]))) +		\
	  LONGALIGN((sizeof(Datum) +							\
				 sizeof(cl_char)) *	(kds)->ncols) * (row_index)))

#define KERN_DATA_STORE_ISNULL(kds,row_index)				\
	((__global cl_char *)									\
	 (KERN_DATA_STORE_VALUES((kds),(row_index)) + (kds)->ncols))

/* length of kern_data_store */
#define KERN_DATA_STORE_LENGTH(kds)										\
	((kds)->format == KDS_FORMAT_ROW ?									\
	 ((uintptr_t)KERN_DATA_STORE_ROWBLOCK((kds), (kds)->nblocks) -		\
	  (uintptr_t)(kds)) :												\
	 STROMALIGN((kds)->length))

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
									 __global kern_data_store *ktoast,
									 cl_uint colidx, cl_uint rowidx);
/* forward declaration of writer interface to kern_data_store */
static __global Datum *pg_common_vstore(__global kern_data_store *kds,
										__global kern_data_store *ktoast,
										__private int *errcode,
										cl_uint colidx, cl_uint rowidx,
										cl_bool isnull);

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
					 __global kern_data_store *ktoast,		\
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

#define STROMCL_SIMPLE_VARSTORE_TEMPLATE(NAME,BASE)			\
	static void												\
	pg_##NAME##_vstore(__global kern_data_store *kds,		\
					   __global kern_data_store *ktoast,	\
					   __private int *errcode,				\
					   cl_uint colidx,						\
					   cl_uint rowidx,						\
					   pg_##NAME##_t datum)					\
	{														\
		__global Datum *daddr;								\
		union {												\
			BASE		v_base;								\
			Datum		v_datum;							\
		} temp;												\
		daddr = pg_common_vstore(kds, ktoast, errcode,		\
								 colidx, rowidx,			\
								 datum.isnull);				\
		if (daddr)											\
		{													\
			temp.v_datum = 0;								\
			temp.v_base = datum.value;						\
			*daddr = temp.v_datum;							\
		}													\
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
							 ((__global char *)kparams +	\
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

/*
 * Macros to calculate CRC32 value.
 * (logic was copied from pg_crc32.c)
 */
#define INIT_CRC32C(crc)		((crc) = 0xFFFFFFFF)
#define FIN_CRC32C(crc)			((crc) ^= 0xFFFFFFFF)
#define EQ_CRC32C(crc1,crc2)	((crc1) == (crc2))

#define STROMCL_SIMPLE_COMP_CRC32_TEMPLATE(NAME,BASE)		   \
	static inline cl_uint									   \
	pg_##NAME##_comp_crc32(__local cl_uint *crc32_table,       \
						   cl_uint hash, pg_##NAME##_t datum)  \
	{														   \
		cl_uint         __len = sizeof(BASE);				   \
		cl_uint         __index;							   \
		union {												   \
			BASE        as_base;							   \
			cl_uint     as_int;								   \
			cl_ulong    as_long;							   \
		} __data;											   \
															   \
		if (!datum.isnull)									   \
		{													   \
			__data.as_base = datum.value;					   \
			while (__len-- > 0)								   \
			{												   \
				__index = (hash ^ __data.as_int) & 0xff;	   \
				hash = crc32_table[__index] ^ ((hash) >> 8);   \
				__data.as_long = (__data.as_long >> 8);		   \
			}												   \
		}													   \
		return hash;										   \
	}

#define STROMCL_SIMPLE_TYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_VARSTORE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_NULLTEST_TEMPLATE(NAME)			\
	STROMCL_SIMPLE_COMP_CRC32_TEMPLATE(NAME,BASE)

/* pg_bool_t */
#ifndef PG_BOOL_TYPE_DEFINED
#define PG_BOOL_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(bool, cl_bool)
#endif

/* pg_int2_t */
#ifndef PG_INT2_TYPE_DEFINED
#define PG_INT2_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int2, cl_short)
#endif

/* pg_int4_t */
#ifndef PG_INT4_TYPE_DEFINED
#define PG_INT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int4, cl_int)
#endif

/* pg_int8_t */
#ifndef PG_INT8_TYPE_DEFINED
#define PG_INT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int8, cl_long)
#endif

/* pg_float4_t */
#ifndef PG_FLOAT4_TYPE_DEFINED
#define PG_FLOAT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float4, cl_float)
#endif

/* pg_float8_t */
#ifndef PG_FLOAT8_TYPE_DEFINED
#define PG_FLOAT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float8, cl_double)
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

#define SET_VARSIZE(PTR, len)						\
	(((__global varattrib_4b *)						\
	  (PTR))->va_4byte.va_header = (((cl_uint) (len)) << 2))

/*
 * kern_get_datum
 *
 * Reference to a particular datum on the supplied kernel data store.
 * It returns NULL, if it is a really null-value in context of SQL,
 * or in case when out of range with error code
 *
 * NOTE: We are paranoia for validation of the data being fetched from
 * the kern_data_store in row-format because we may see a phantom page
 * if the source transaction that required this kernel execution was
 * aborted during execution.
 * Once a transaction gets aborted, shared buffers being pinned are
 * released, even if DMA send request on the buffers are already
 * enqueued. In this case, the calculation result shall be discarded,
 * so no need to worry about correctness of the calculation, however,
 * needs to be care about address of the variables being referenced.
 */
static inline __global void *
kern_get_datum_tuple(__global kern_colmeta *colmeta,
					 __global HeapTupleHeaderData *htup,
					 cl_uint colidx)
{
	cl_bool		heap_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
	cl_uint		offset = htup->t_hoff;
	cl_uint		i, ncols = (htup->t_infomask2 & HEAP_NATTS_MASK);

	/* shortcut if colidx is obviously out of range */
	if (colidx >= ncols)
		return NULL;
	/* shortcut if tuple contains no NULL values */
	if (!heap_hasnull)
	{
		kern_colmeta	cmeta = colmeta[colidx];

		if (cmeta.attcacheoff >= 0)
			return (__global char *)htup + cmeta.attcacheoff;
	}
	/* regular path that walks on heap-tuple from the head */
	for (i=0; i < ncols; i++)
	{
		if (heap_hasnull && att_isnull(i, htup->t_bits))
		{
			if (i == colidx)
				return NULL;
		}
		else
		{
			__global char  *addr;
			kern_colmeta	cmeta = colmeta[i];

			if (cmeta.attlen > 0)
				offset = TYPEALIGN(cmeta.attalign, offset);
			else if (!VARATT_NOT_PAD_BYTE((__global char *)htup + offset))
				offset = TYPEALIGN(cmeta.attalign, offset);
			/* TODO: overrun checks here */
			addr = ((__global char *) htup + offset);
			if (i == colidx)
				return addr;
			offset += (cmeta.attlen > 0
					   ? cmeta.attlen
					   : VARSIZE_ANY(addr));
		}
	}
	return NULL;
}

static inline __global HeapTupleHeaderData *
kern_get_tuple_rs(__global kern_data_store *kds, cl_uint rowidx,
				  __private cl_uint *p_blk_index)
{
	__global kern_rowitem *kritem;
	PageHeader	page;
	cl_ushort	blk_index;
	cl_ushort	item_offset;
	cl_ushort	item_max;
	ItemIdData	item_id;

	if (rowidx >= kds->nitems)
		return NULL;	/* likely a BUG */

	kritem = KERN_DATA_STORE_ROWITEM(kds, rowidx);
	blk_index = kritem->blk_index;
	item_offset = kritem->item_offset;
	if (blk_index >= kds->nblocks)
		return NULL;	/* likely a BUG */

	page = KERN_DATA_STORE_ROWBLOCK(kds, blk_index);
	item_max = PageGetMaxOffsetNumber(page);
	if (offsetof(PageHeaderData, pd_linp[item_max + 1]) >= BLCKSZ ||
		item_offset == 0 || item_offset > item_max)
		return NULL;	/* likely a BUG */

	item_id = page->pd_linp[item_offset - 1];
	if (ItemIdGetOffset(item_id) +
		offsetof(HeapTupleHeaderData, t_bits) >= BLCKSZ)
		return NULL;	/* likely a BUG */

	/* also set additional properties */
	*p_blk_index = blk_index;

	return (__global HeapTupleHeaderData *)
		((__global char *)page + ItemIdGetOffset(item_id));
}

static inline __global void *
kern_get_datum_rs(__global kern_data_store *kds,
				  cl_uint colidx, cl_uint rowidx)
{
	__global HeapTupleHeaderData *htup;
	cl_uint			blkidx;

	if (colidx >= kds->ncols)
		return NULL;	/* likely a BUG */
	htup = kern_get_tuple_rs(kds, rowidx, &blkidx);
	if (!htup)
		return NULL;
	return kern_get_datum_tuple(kds->colmeta, htup, colidx);
}

static inline __global HeapTupleHeaderData *
kern_get_tuple_rsflat(__global kern_data_store *kds, cl_uint rowidx)
{
	__global kern_rowitem *kritem;

	if (rowidx >= kds->nitems)
		return NULL;	/* likely a BUG */

	kritem = KERN_DATA_STORE_ROWITEM(kds, rowidx);
	/* simple sanity check */
	if (kritem->htup_offset >= kds->length)
		return NULL;
	return (__global HeapTupleHeaderData *)
		((__global char *)kds + kritem->htup_offset);
}

static inline __global void *
kern_get_datum_rsflat(__global kern_data_store *kds,
					  cl_uint colidx, cl_uint rowidx)
{
	__global HeapTupleHeaderData *htup;

	if (colidx >= kds->ncols)
		return NULL;	/* likely a BUG */
	htup = kern_get_tuple_rsflat(kds, rowidx);
	if (!htup)
		return NULL;
	return kern_get_datum_tuple(kds->colmeta, htup, colidx);
}

static __global void *
kern_get_datum_tupslot(__global kern_data_store *kds,
					   __global kern_data_store *ktoast,
					   cl_uint colidx, cl_uint rowidx)
{
	__global Datum	   *values = KERN_DATA_STORE_VALUES(kds,rowidx);
	__global cl_char   *isnull = KERN_DATA_STORE_ISNULL(kds,rowidx);
	kern_colmeta		cmeta = kds->colmeta[colidx];

	if (isnull[colidx])
		return NULL;
	if (cmeta.attlen > 0)
		return values + colidx;
	return (__global char *)(&ktoast->hostptr) + values[colidx];
}

static inline __global void *
kern_get_datum(__global kern_data_store *kds,
			   __global kern_data_store *ktoast,
			   cl_uint colidx, cl_uint rowidx)
{
	/* is it out of range? */
	if (colidx >= kds->ncols || rowidx >= kds->nitems)
		return NULL;
	if (kds->format == KDS_FORMAT_ROW)
		return kern_get_datum_rs(kds, colidx, rowidx);
	if (kds->format == KDS_FORMAT_ROW_FLAT)
		return kern_get_datum_rsflat(kds, colidx, rowidx);
	if (kds->format == KDS_FORMAT_TUPSLOT)
		return kern_get_datum_tupslot(kds,ktoast,colidx,rowidx);
	/* TODO: put StromError_DataStoreCorruption error here */
	return NULL;
}

/*
 * common function to store a value on tuple-slot format
 */
static __global Datum *
pg_common_vstore(__global kern_data_store *kds,
				 __global kern_data_store *ktoast,
				 __private int *errcode,
				 cl_uint colidx, cl_uint rowidx,
				 cl_bool isnull)
{
	kern_colmeta		cmeta;
	__global Datum	   *slot_values;
	__global cl_char   *slot_isnull;
	/*
	 * Only tuple-slot is acceptable destination format.
	 * Only row- and row-flat are acceptable source format.
	 */
	if (kds->format != KDS_FORMAT_TUPSLOT ||
		(ktoast->format != KDS_FORMAT_ROW &&
		 ktoast->format != KDS_FORMAT_ROW_FLAT))
	{
		STROM_SET_ERROR(errcode, StromError_SanityCheckViolation);
		return NULL;
	}
	/* out of range? */
	if (colidx >= kds->ncols || rowidx >= kds->nrooms)
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreOutOfRange);
		return NULL;
	}
	slot_values = KERN_DATA_STORE_VALUES(kds, rowidx);
	slot_isnull = KERN_DATA_STORE_ISNULL(kds, rowidx);

	slot_isnull[colidx] = (cl_char)(isnull ? 1 : 0);

	return slot_values + colidx;
}

/*
 * kern_fixup_data_store
 *
 * pg_xxx_vstore() interface stores varlena datum on kern_data_store with
 * KDS_FORMAT_TUP_SLOT format using device address space. Because tup-slot
 * format intends to store host accessable representation, we need to fix-
 * up pointers in the tuple store.
 * In case of any other format, we don't need to modify the data.
 */
static void
pg_fixup_tupslot_varlena(__private int *errcode,
						 __global kern_data_store *kds,
						 __global kern_data_store *ktoast,
						 cl_uint colidx, cl_uint rowidx)
{
	__global Datum	   *values;
	__global cl_char   *isnull;
	kern_colmeta		cmeta;

	/* only tuple-slot format needs to fixup pointer values */
	if (kds->format != KDS_FORMAT_TUPSLOT)
		return;
	/* out of range? */
	if (rowidx >= kds->nitems || colidx >= kds->ncols)
		return;

	/* fixed length variable? */
	cmeta = kds->colmeta[colidx];
	if (cmeta.attlen > 0)
		return;

	values = KERN_DATA_STORE_VALUES(kds, rowidx);
	isnull = KERN_DATA_STORE_ISNULL(kds, rowidx);
	/* no need to care about NULL values */
	if (isnull[colidx])
		return;
	/* OK, non-null varlena datum has to be fixed up for host address space */
	if (ktoast->format == KDS_FORMAT_ROW)
	{
		hostptr_t	offset = values[colidx];
		hostptr_t	baseline
			= ((__global char *)KERN_DATA_STORE_ROWBLOCK(ktoast,0) -
			   (__global char *)ktoast);

		if (offset >= baseline &&
			offset - baseline < ktoast->nblocks * BLCKSZ)
		{
			cl_uint		blkidx = (offset - baseline) / BLCKSZ;
			hostptr_t	itemptr = (offset - baseline) % BLCKSZ;
			__global kern_blkitem *bitem
				= KERN_DATA_STORE_BLKITEM(ktoast, blkidx);

			values[colidx] = (Datum)(bitem->page + itemptr);
		}
		else
		{
			isnull[colidx] = (cl_char) 1;
			values[colidx] = 0;
			STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
		}
	}
	else if (ktoast->format == KDS_FORMAT_ROW_FLAT)
	{
		hostptr_t	offset = values[colidx];

		if (offset < ktoast->length)
		{
			values[colidx] = (Datum)((hostptr_t)ktoast->hostptr +
									 (hostptr_t)offset);
		}
		else
		{
			isnull[colidx] = (cl_char) 1;
			values[colidx] = 0;
			STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
		}
	}
	else
		STROM_SET_ERROR(errcode, StromError_SanityCheckViolation);
}

#if 0
static inline void
pg_dump_data_store(__global kern_data_store *kds, __constant const char *label)
{
	cl_uint		i;

	printf("gid=%zu: kds(%s) {length=%u usage=%u ncols=%u nitems=%u nrooms=%u "
		   "nblocks=%u maxblocks=%u format=%d}\n",
		   get_global_id(0), label,
		   kds->length, kds->usage, kds->ncols,
		   kds->nitems, kds->nrooms,
		   kds->nblocks, kds->maxblocks, kds->format);
	for (i=0; i < kds->ncols; i++)
		printf("gid=%zu: kds(%s) colmeta[%d] "
			   "{attnotnull=%d attalign=%d attlen=%d}\n",
			   get_global_id(0), label, i,
			   kds->colmeta[i].attnotnull,
			   kds->colmeta[i].attalign,
			   kds->colmeta[i].attlen);
}
#endif
/*
 * functions to reference variable length variables
 */
STROMCL_SIMPLE_DATATYPE_TEMPLATE(varlena, __global varlena *)

static inline pg_varlena_t
pg_varlena_vref(__global kern_data_store *kds,
				__global kern_data_store *ktoast,
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

static inline void
pg_varlena_vstore(__global kern_data_store *kds,
				  __global kern_data_store *ktoast,
				  __private int *errcode,
				  cl_uint colidx,
				  cl_uint rowidx,
				  pg_varlena_t datum)
{
	__global Datum *daddr
		= pg_common_vstore(kds, ktoast, errcode,
						   colidx, rowidx, datum.isnull);
	if (daddr)
	{
		*daddr = (Datum)((__global char *)datum.value -
						 (__global char *)&ktoast->hostptr);
	}
	/*
	 * NOTE: pg_fixup_tupslot_varlena() shall be called, prior to
	 * the write-back of kern_data_store.
	 */
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	else
		result.isnull = true;

	return result;
}

STROMCL_SIMPLE_NULLTEST_TEMPLATE(varlena)

static inline cl_uint
pg_varlena_comp_crc32(__local cl_uint *crc32_table,
					  cl_uint hash, pg_varlena_t datum)
{
	if (!datum.isnull)
	{
		__global const cl_char *__data = VARDATA_ANY(datum.value);
		cl_uint		__len = VARSIZE_ANY_EXHDR(datum.value);
		cl_uint		__index;
		while (__len-- > 0)
		{
			__index = (hash ^ *__data++) & 0xff;
			hash = crc32_table[__index] ^ (hash >> 8);
		}
	}
	return hash;
}

#define STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)			\
	typedef pg_varlena_t	pg_##NAME##_t;

#define STROMCL_VARLENA_VARREF_TEMPLATE(NAME)			\
	static pg_##NAME##_t								\
	pg_##NAME##_vref(__global kern_data_store *kds,		\
					 __global kern_data_store *ktoast,	\
					 __private int *errcode,			\
					 cl_uint colidx,					\
					 cl_uint rowidx)					\
	{													\
		return pg_varlena_vref(kds,ktoast,errcode,		\
							   colidx,rowidx);			\
	}

#define STROMCL_VARLENA_VARSTORE_TEMPLATE(NAME)				\
	static void												\
	pg_##NAME##_vstore(__global kern_data_store *kds,		\
					   __global kern_data_store *ktoast,	\
					   __private int *errcode,				\
					   cl_uint colidx,						\
					   cl_uint rowidx,						\
					   pg_##NAME##_t datum)					\
	{														\
		return pg_varlena_vstore(kds,ktoast,errcode,		\
								 colidx,rowidx,datum);		\
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

#define STROMCL_VARLENA_COMP_CRC32_TEMPLATE(NAME)					\
	static inline cl_uint											\
	pg_##NAME##_comp_crc32(__local cl_uint *crc32_table,			\
						   cl_uint hash, pg_##NAME##_t datum)		\
	{																\
		return pg_varlena_comp_crc32(crc32_table, hash, datum);		\
	}

#define STROMCL_VARLENA_TYPE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_VARREF_TEMPLATE(NAME)			\
	STROMCL_VARLENA_VARSTORE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)			\
	STROMCL_VARLENA_NULLTEST_TEMPLATE(NAME)			\
	STROMCL_VARLENA_COMP_CRC32_TEMPLATE(NAME)

/* pg_bytea_t */
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
static __global void *
memset(__global void *s, int c, size_t n)
{
	__global char  *ptr = s;

	while (n-- > 0)
		*ptr++ = c;
	return s;
}

#if 1
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
 * kern_writeback_error_status
 *
 * It set thread local error code on the status variable on the global
 * memory, if status code is still StromError_Success and any of thread
 * has a particular error code.
 * NOTE: It does not look at significance of the error, so caller has
 * to clear its error code if it is a minor one.
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
			if (errcode_0 == StromError_Success &&
				errcode_1 != StromError_Success)
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
	if (get_local_id(0) == 0)
		atomic_cmpxchg(error_status, StromError_Success, errcode_0);
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
 * macros for general binary compare functions
 */
#define devfunc_int_comp(x,y)					\
	((x) < (y) ? -1 : ((x) > (y) ? 1 : 0))

#define devfunc_float_comp(x,y)					\
	(isnan(x)									\
	 ? (isnan(y)								\
		? 0		/* NAN = NAM */					\
		: 1)	/* NAN > non-NAN */				\
	 : (isnan(y)								\
		? -1	/* non-NAN < NAN */				\
		: devfunc_int_comp((x),(y))))

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
