/*
 * xpu_common.h
 *
 * Common header portion for xPU(GPU/DPU/SPU) device code.
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_COMMON_H
#define XPU_COMMON_H
#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include "postgres_ext.h"
#include "pg_config.h"
#include "pg_config_manual.h"

/* Definition of several primitive types */
typedef __int128	int128_t;
#if defined(__CUDACC__)
#include <cuda_fp16.h>
typedef __half		float2_t;
#elif defined(HAVE_FLOAT2)
typedef _Float16	float2_t;
#else
typedef uint16_t	float2_t;
#endif
typedef float		float4_t;
typedef double		float8_t;

/*
 * Functions with qualifiers
 */
#ifdef __CUDACC__
#define INLINE_FUNCTION(RET_TYPE)				\
	__device__ __forceinline__					\
	static RET_TYPE __attribute__ ((unused))
#define STATIC_FUNCTION(RET_TYPE)		__device__ static RET_TYPE
#define PUBLIC_FUNCTION(RET_TYPE)		__device__ RET_TYPE
#define KERNEL_FUNCTION(RET_TYPE)		extern "C" __global__ RET_TYPE
#define EXTERN_DATA						extern __device__
#define PUBLIC_DATA						__device__
#define STATIC_DATA						static __device__
#else
#define INLINE_FUNCTION(RET_TYPE)		static inline RET_TYPE
#define STATIC_FUNCTION(RET_TYPE)		static RET_TYPE
#define PUBLIC_FUNCTION(RET_TYPE)		RET_TYPE
#define KERNEL_FUNCTION(RET_TYPE)		RET_TYPE
#define EXTERN_DATA						extern
#define PUBLIC_DATA
#define STATIC_DATE						static
#endif	/* __CUDACC__ */

/*
 * Several fundamental data types and macros
 */
#ifndef POSTGRES_H
#define Assert(cond)		assert(cond)
#define Max(a,b)			((a) > (b) ? (a) : (b))
#define Min(a,b)			((a) < (b) ? (a) : (b))
typedef uint64_t			Datum;

#define PointerGetDatum(X)	((Datum)(X))
#define DatumGetPointer(X)	((char *)(X))
#define TYPEALIGN(ALIGNVAL,LEN)         \
	(((uint64_t)(LEN) + ((ALIGNVAL) - 1)) & ~((uint64_t)((ALIGNVAL) - 1)))
#define TYPEALIGN_DOWN(ALIGNVAL,LEN)                        \
	(((uint64_t) (LEN)) & ~((uint64_t) ((ALIGNVAL) - 1)))

#define MAXIMUM_ALIGNOF		8
#define MAXALIGN(LEN)		TYPEALIGN(MAXIMUM_ALIGNOF,LEN)
#define MAXALIGN_DOWN(LEN)	TYPEALIGN_DOWN(MAXIMUM_ALIGNOF,LEN)
#endif	/* POSTGRES_H */
#define MAXIMUM_ALIGNOF_SHIFT 3

#ifndef __FILE_NAME__
INLINE_FUNCTION(const char *)
__runtime_file_name(const char *path)
{
	const char *s;

	for (s = path; *s != '\0'; s++)
	{
		if (*s == '/')
			path = s + 1;
	}
	return path;
}
#define __FILE_NAME__	__runtime_file_name(__FILE__)
#endif

#ifdef __CUDACC__
template <typename T>
INLINE_FUNCTION(T)
__Fetch(const T *ptr)
{
	T	temp;

	memcpy(&temp, ptr, sizeof(T));

	return temp;
}
#else
#define __Fetch(PTR)		(*(PTR))
#endif

/*
 * Thread index at CUDA C
 */
#ifdef __CUDACC__
#define get_group_id()			(blockIdx.x)
#define get_num_groups()		(gridDim.x)
#define get_local_id()			(threadIdx.x)
#define get_local_size()		(blockDim.x)
#define get_global_id()			(threadIdx.x + blockIdx.x * blockDim.x)
#define get_global_size()		(blockDim.x * gridDim.x)
#define get_global_base()		(blockIdx.x * blockDim.x)
#define get_warp_id()			(threadIdx.x / warpSize)
#define get_lane_id()			(threadIdx.x & (warpSize-1))
#endif

/*
 * Error status
 */
#ifndef MAKE_SQLSTATE
#define PGSIXBIT(ch)			(((ch) - '0') & 0x3F)
#define MAKE_SQLSTATE(ch1,ch2,ch3,ch4,ch5)				\
	(PGSIXBIT(ch1) +  (PGSIXBIT(ch2) << 6) +			\
	 (PGSIXBIT(ch3) << 12) + (PGSIXBIT(ch4) << 18) +	\
	 (PGSIXBIT(ch5) << 24))
#endif  /* MAKE_SQLSTATE */
#include "utils/errcodes.h"
#define ERRCODE_FLAGS_CPU_FALLBACK			(1U<<30)
#define ERRCODE_STROM_SUCCESS				0
#define ERRCODE_STROM_DATASTORE_NOSPACE		MAKE_SQLSTATE('H','D','B','0','4')
#define ERRCODE_STROM_WRONG_CODE_GENERATION	MAKE_SQLSTATE('H','D','B','0','5')
#define ERRCODE_STROM_DATA_CORRUPTION		MAKE_SQLSTATE('H','D','B','0','7')
#define ERRCODE_STROM_VARLENA_UNSUPPORTED	MAKE_SQLSTATE('H','D','B','0','8')
#define ERRCODE_STROM_RECURSION_TOO_DEEP	MAKE_SQLSTATE('H','D','B','0','9')

#define KERN_ERRORBUF_FILENAME_LEN		24
#define KERN_ERRORBUF_FUNCNAME_LEN		64
#define KERN_ERRORBUF_MESSAGE_LEN		200
typedef struct {
	int32_t		errcode;	/* one of the ERRCODE_* */
	int32_t		lineno;
	char		filename[KERN_ERRORBUF_FILENAME_LEN];
	char		funcname[KERN_ERRORBUF_FUNCNAME_LEN];
	char		message[KERN_ERRORBUF_MESSAGE_LEN];
} kern_errorbuf;

/*
 * kern_context - a set of run-time information
 */
typedef struct
{
	int				errcode;
	const char	   *error_filename;
	uint32_t		error_lineno;
	const char	   *error_funcname;
	const char	   *error_message;
	struct kernSessionInfo *session;
	/* current slot of kernel variable references */
	const struct kern_colmeta **kvars_cmeta;
	const void	  **kvars_addr;
	int			   *kvars_len;
	uint32_t		kvars_nslots;
	/* variable length buffer */
	char		   *vlpos;
	char		   *vlend;
	char			vlbuf[1];
} kern_context;

#define INIT_KERNEL_CONTEXT(KCXT,SESSION)								\
	do {																\
		uint32_t	__nslots = (SESSION)->kvars_nslots;					\
		uint32_t	__sz = (offsetof(kern_context, vlbuf) +				\
							MAXALIGN((SESSION)->kcxt_extra_bufsz));		\
		KCXT = alloca(__sz);											\
		memset(KCXT, 0, __sz);											\
		KCXT->session = (SESSION);										\
		if (__nslots > 0)												\
		{																\
			KCXT->kvars_nslots = __nslots;								\
			KCXT->kvars_cmeta = alloca(sizeof(void *) * __nslots);		\
			KCXT->kvars_addr = alloca(sizeof(void *) * __nslots);		\
			KCXT->kvars_len  = alloca(sizeof(int) * __nslots);			\
			memset(KCXT->kvars_cmeta, 0, sizeof(void *) * __nslots);	\
			memset(KCXT->kvars_addr, 0, sizeof(void *) * __nslots);		\
			memset(KCXT->kvars_len, -1, sizeof(int) * __nslots);		\
		}																\
		KCXT->vlpos = KCXT->vlbuf;										\
		KCXT->vlend = KCXT->vlbuf + (SESSION)->kcxt_extra_bufsz;		\
	} while(0)

INLINE_FUNCTION(void *)
kcxt_alloc(kern_context *kcxt, size_t len)
{
	char   *pos = (char *)MAXALIGN(kcxt->vlpos);

	if (pos >= kcxt->vlbuf && pos + len <= kcxt->vlend)
	{
		kcxt->vlpos = pos + len;
		return pos;
	}
	return NULL;
}

INLINE_FUNCTION(void)
kcxt_reset(kern_context *kcxt)
{
	if (kcxt->kvars_nslots > 0)
	{
		memset(kcxt->kvars_cmeta, 0, sizeof(void *) * kcxt->kvars_nslots);
		memset(kcxt->kvars_addr, 0, sizeof(void *) * kcxt->kvars_nslots);
		memset(kcxt->kvars_len, -1, sizeof(int) * kcxt->kvars_nslots);
	}
	kcxt->vlpos = kcxt->vlbuf;
}

INLINE_FUNCTION(void)
__STROM_EREPORT(kern_context *kcxt,
				int errcode,
				const char *filename,
				int lineno,
				const char *funcname,
				const char *message)
{
	if (kcxt->errcode == 0 && errcode != 0)
	{
		const char *pos;

		for (pos = filename; *pos != '\0'; pos++)
		{
			if (pos[0] == '/' && pos[1] != '\0')
				filename = pos + 1;
		}
		if (!message)
			message = "GPU kernel internal error";
		kcxt->errcode  = errcode;
		kcxt->error_filename = filename;
		kcxt->error_lineno   = lineno;
		kcxt->error_funcname = funcname;
		kcxt->error_message  = message;
	}
}
#define STROM_ELOG(kcxt, message)									\
	__STROM_EREPORT((kcxt),ERRCODE_INTERNAL_ERROR,					\
					__FILE__,__LINE__,__FUNCTION__,(message))
#define STROM_EREPORT(kcxt, errcode, message)						\
	__STROM_EREPORT((kcxt),(errcode),								\
					__FILE__,__LINE__,__FUNCTION__,(message))
#define STROM_CPU_FALLBACK(kcxt, errcode, message)					\
	__STROM_EREPORT((kcxt),(errcode) | ERRCODE_FLAGS_CPU_FALLBACK,	\
					__FILE__,__LINE__,__FUNCTION__,(message))

/* ----------------------------------------------------------------
 *
 * Definitions related to the kernel data store
 *
 * ----------------------------------------------------------------
 */
#include "arrow_defs.h"

#define TYPE_KIND__NULL			'n'		/* unreferenced column */
#define TYPE_KIND__BASE			'b'
#define TYPE_KIND__ARRAY		'a'
#define TYPE_KIND__COMPOSITE	'c'
#define TYPE_KIND__DOMAIN		'd'
#define TYPE_KIND__ENUM			'e'
#define TYPE_KIND__PSEUDO		'p'
#define TYPE_KIND__RANGE		'r'

struct kern_colmeta {
	/* true, if column is held by value. Elsewhere, a reference */
	bool			attbyval;
	/* alignment; 1,2,4 or 8, not characters in pg_attribute */
	int8_t			attalign;
	/* length of attribute */
	int16_t			attlen;
	/* attribute number */
	int16_t			attnum;
	/* offset of attribute location, if deterministic */
	int16_t			attcacheoff;
	/* oid of the SQL data type */
	Oid				atttypid;
	/* typmod of the SQL data type */
	int32_t			atttypmod;
	/* one of TYPE_KIND__* */
	int8_t			atttypkind;
	/*
	 * (for array and composite types)
	 * Some of types contain sub-fields like array or composite type.
	 * We carry type definition information (kern_colmeta) using the
	 * kds->colmeta[] array next to the top-level fields.
	 * An array type has relevant element type. So, its @num_subattrs
	 * is always 1, and kds->colmeta[@idx_subattrs] informs properties
	 * of the element type.
	 * A composite type has several fields.
	 * kds->colmeta[@idx_subattrs ... @idx_subattrs + @num_subattrs -1]
	 * carries its sub-fields properties.
	 */
	uint16_t		idx_subattrs;
	uint16_t		num_subattrs;

	/* column name */
	char			attname[NAMEDATALEN];

	/*
	 * (only arrow/column format)
	 * @attoptions keeps extra information of Apache Arrow type. Unlike
	 * PostgreSQL types, it can have variation of data accuracy in time
	 * related data types, or precision in decimal data type.
	 */
	ArrowTypeOptions attopts;
	uint32_t		nullmap_offset;
	uint32_t		nullmap_length;
	uint32_t		values_offset;
	uint32_t		values_length;
	uint32_t		extra_offset;
	uint32_t		extra_length;
};
typedef struct kern_colmeta		kern_colmeta;

#define KDS_FORMAT_HEAP			1
#define KDS_FORMAT_SLOT			2
#define KDS_FORMAT_HASH			3	/* inner hash table for GpuHashJoin */
#define KDS_FORMAT_BLOCK		4	/* raw blocks for direct loading */
#define KDS_FORMAT_COLUMN		5	/* columnar based storage format */
#define KDS_FORMAT_ARROW		6	/* apache arrow format */

struct kern_data_store {
	uint64_t		length;		/* length of this data-store */
	/*
	 * NOTE: {nitems + usage} must be aligned to 64bit because these pair of
	 * values can be updated atomically using cmpxchg.
	 */
	uint32_t		nitems; 	/* number of rows in this store */
	uint32_t		usage;		/* usage of this data-store (PACKED) */
	uint32_t		nrooms;		/* number of available rows in this store */
	uint32_t		ncols;		/* number of columns in this store */
	int8_t			format;		/* one of KDS_FORMAT_* above */
	bool			has_varlena; /* true, if any varlena attribute */
	bool			tdhasoid;	/* copy of TupleDesc.tdhasoid */
	Oid				tdtypeid;	/* copy of TupleDesc.tdtypeid */
	int32_t			tdtypmod;	/* copy of TupleDesc.tdtypmod */
	Oid				table_oid;	/* OID of the table (only if GpuScan) */
	uint32_t		nslots;		/* width of hash-slot (only HASH format) */
	uint32_t		nrows_per_block; /* average number of rows per
									  * PostgreSQL block (only BLOCK format) */
	uint32_t		nr_colmeta;	/* number of colmeta[] array elements;
								 * maybe, >= ncols, if any composite types */
	kern_colmeta	colmeta[1];	/* metadata of columns */
};
typedef struct kern_data_store		kern_data_store;

/*
 * kern_data_extra - extra buffer of KDS_FORMAT_COLUMN
 */
struct kern_data_extra
{
	uint64_t	length;
	uint64_t	usage;
	char		data[1];
};
typedef struct kern_data_extra		kern_data_extra;

/*
 * MEMO: Support of 32GB KDS - KDS with row-, hash- and column-format
 * internally uses 32bit offset value from the head or base address.
 * We have assumption here - any objects pointed by the offset value
 * is always aligned to MAXIMUM_ALIGNOF boundary (64bit).
 * It means we can use 32bit offset to represent up to 32GB range (35bit).
 */
INLINE_FUNCTION(uint32_t)
__kds_packed(size_t offset)
{
	assert((offset & ~(0xffffffffUL << MAXIMUM_ALIGNOF_SHIFT)) == 0);
	return (uint32_t)(offset >> MAXIMUM_ALIGNOF_SHIFT);
}

INLINE_FUNCTION(size_t)
__kds_unpack(uint32_t offset)
{
	return (size_t)offset << MAXIMUM_ALIGNOF_SHIFT;
}

/* ----------------------------------------------------------------
 *
 * Definitions of HeapTuple/IndexTuple and related
 *
 * ----------------------------------------------------------------
 */
#ifdef POSTGRES_H
#include "access/htup_details.h"
#include "access/itup.h"
#include "access/sysattr.h"
#else
/*
 * Attribute numbers for the system-defined attributes
 */
#define SelfItemPointerAttributeNumber			(-1)
#define ObjectIdAttributeNumber					(-2)
#define MinTransactionIdAttributeNumber			(-3)
#define MinCommandIdAttributeNumber				(-4)
#define MaxTransactionIdAttributeNumber			(-5)
#define MaxCommandIdAttributeNumber				(-6)
#define TableOidAttributeNumber					(-7)
#define FirstLowInvalidHeapAttributeNumber		(-8)

/*
 * ItemPointer:
 */
typedef struct
{
	struct {
		uint16_t	bi_hi;
		uint16_t	bi_lo;
	} ip_blkid;
	uint16_t		ip_posid;
} ItemPointerData;

/*
 * HeapTupleHeaderData
 */
typedef struct HeapTupleFields
{
	uint32_t		t_xmin;		/* inserting xact ID */
	uint32_t		t_xmax;		/* deleting or locking xact ID */
	union
	{
		uint32_t	t_cid;		/* inserting or deleting command ID, or both */
		uint32_t	t_xvac;		/* old-style VACUUM FULL xact ID */
	}	t_field3;
} HeapTupleFields;

typedef struct DatumTupleFields
{
    int32_t		datum_len_;		/* varlena header (do not touch directly!) */
    int32_t		datum_typmod;	/* -1, or identifier of a record type */
    Oid			datum_typeid;	/* composite type OID, or RECORDOID */
} DatumTupleFields;

typedef struct HeapTupleHeaderData
{
	union {
		HeapTupleFields		t_heap;
		DatumTupleFields	t_datum;
	} t_choice;

	ItemPointerData	t_ctid;			/* current TID of this or newer tuple */

	uint16_t		t_infomask2;	/* number of attributes + various flags */
    uint16_t		t_infomask;		/* various flag bits, see below */
    uint8_t			t_hoff;			/* sizeof header incl. bitmap, padding */
    /* ^ - 23 bytes - ^ */
    uint8_t			t_bits[1];		/* null-bitmap -- VARIABLE LENGTH */
} HeapTupleHeaderData;

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

#define HEAP_XMIN_COMMITTED		0x0100	/* t_xmin committed */
#define HEAP_XMIN_INVALID		0x0200	/* t_xmin invalid/aborted */
#define HEAP_XMAX_COMMITTED		0x0400	/* t_xmax committed */
#define HEAP_XMAX_INVALID		0x0800	/* t_xmax invalid/aborted */
#define HEAP_XMAX_IS_MULTI		0x1000	/* t_xmax is a MultiXactId */
#define HEAP_UPDATED			0x2000	/* this is UPDATEd version of row */
#define HEAP_MOVED_OFF			0x4000	/* unused in xPU */
#define HEAP_MOVED_IN			0x8000	/* unused in xPU */

/*
 * information stored in t_infomask2:
 */
#define HEAP_NATTS_MASK			0x07FF  /* 11 bits for number of attributes */
#define HEAP_KEYS_UPDATED		0x2000  /* tuple was updated and key cols
										 * modified, or tuple deleted */
#define HEAP_HOT_UPDATED		0x4000	/* tuple was HOT-updated */
#define HEAP_ONLY_TUPLE			0x8000	/* this is heap-only tuple */
#define HEAP2_XACT_MASK			0xE000	/* visibility-related bits */

/* null-bitmap checker */
#define att_isnull(ATT, BITS) (!((BITS)[(ATT) >> 3] & (1 << ((ATT) & 0x07))))
#define BITMAPLEN(NATTS)		(((int)(NATTS) + 7) / 8)

/*
 * Index tuple header structure
 *
 * All index tuples start with IndexTupleData.  If the HasNulls bit is set,
 * this is followed by an IndexAttributeBitMapData.  The index attribute
 * values follow, beginning at a MAXALIGN boundary.
 */
typedef struct IndexTupleData
{
	ItemPointerData		t_tid;		/* reference TID to heap tuple */

	/* ---------------
	 * t_info is laid out in the following fashion:
	 *
	 * 15th (high) bit: has nulls
	 * 14th bit: has var-width attributes
	 * 13th bit: AM-defined meaning
	 * 12-0 bit: size of tuple
	 * ---------------
	 */
	uint16_t			t_info;

	char				data[1];	/* data or IndexAttributeBitMapData */
} IndexTupleData;

typedef struct IndexAttributeBitMapData
{
	uint8_t				bits[(INDEX_MAX_KEYS + 8 - 1) / 8];
} IndexAttributeBitMapData;

#define INDEX_SIZE_MASK     0x1fff
#define INDEX_VAR_MASK      0x4000
#define INDEX_NULL_MASK     0x8000
#endif	/* POSTGRES_H */

/* ----------------------------------------------------------------
 *
 * Definitions of PageHeader/ItemId and related
 *
 * ----------------------------------------------------------------
 */
#ifdef POSTGRES_H
#include "access/gist.h"
#include "access/transam.h"
#include "storage/bufpage.h"
#include "storage/block.h"
#include "storage/itemid.h"
#include "storage/off.h"
#else
/* definitions in access/transam.h */
typedef uint32_t		TransactionId;
#define InvalidTransactionId		((TransactionId) 0)
#define BootstrapTransactionId		((TransactionId) 1)
#define FrozenTransactionId			((TransactionId) 2)
#define FirstNormalTransactionId	((TransactionId) 3)
#define MaxTransactionId			((TransactionId) 0xffffffff)

/* definitions in storage/block.h */
typedef uint32_t		BlockNumber;

#define InvalidBlockNumber			((BlockNumber) 0xffffffff)
#define MaxBlockNumber				((BlockNumber) 0xfffffffe)

/* definitions in storage/itemid.h */
typedef struct ItemIdData
{
	unsigned	lp_off:15,		/* offset to tuple (from start of page) */
				lp_flags:2,		/* state of item pointer, see below */
				lp_len:15;		/* byte length of tuple */
} ItemIdData;

#define LP_UNUSED		0		/* unused (should always have lp_len=0) */
#define LP_NORMAL		1		/* used (should always have lp_len>0) */
#define LP_REDIRECT		2		/* HOT redirect (should have lp_len=0) */
#define LP_DEAD			3		/* dead, may or may not have storage */

#define ItemIdGetOffset(itemId)		((itemId)->lp_off)
#define ItemIdGetLength(itemId)		((itemId)->lp_len)
#define ItemIdIsUsed(itemId)		((itemId)->lp_flags != LP_UNUSED)
#define ItemIdIsNormal(itemId)		((itemId)->lp_flags == LP_NORMAL)
#define ItemIdIsRedirected(itemId)	((itemId)->lp_flags == LP_REDIRECT)
#define ItemIdIsDead(itemId)		((itemId)->lp_flags == LP_DEAD)
#define ItemIdHasStorage(itemId)	((itemId)->lp_len != 0)
#define ItemIdSetUnused(itemId)			\
    do {								\
		(itemId)->lp_flags = LP_UNUSED; \
		(itemId)->lp_off = 0;           \
		(itemId)->lp_len = 0;           \
	} while(0)

/* definitions in storage/off.h */
typedef uint16_t			OffsetNumber;
#define InvalidOffsetNumber	((OffsetNumber) 0)
#define FirstOffsetNumber	((OffsetNumber) 1)
#define MaxOffsetNumber		((OffsetNumber) (BLCKSZ / sizeof(ItemIdData)))
#define OffsetNumberNext(offsetNumber)			\
	((OffsetNumber) (1 + (offsetNumber)))
#define OffsetNumberPrev(offsetNumber)			\
	((OffsetNumber) (-1 + (offsetNumber)))

/* definitions in storage/bufpage.h */
typedef struct PageHeaderData
{
#if 0
	/*
	 * NOTE: device code (ab-)uses this field to track parent block/item
	 * when GiST index is loaded. Without this hack, hard to implement
	 * depth-first search at GpuJoin.
	 */
	PageXLogRecPtr pd_lsn;		/* LSN: next byte after last byte of xlog
								 * record for last change to this page */
#else
	uint32_t	pd_parent_blkno;
	uint32_t	pd_parent_item;
#endif
	uint16_t	pd_checksum;	/* checksum */
	uint16_t	pd_flags;		/* flag bits, see below */
	uint16_t	pd_lower;		/* offset to start of free space */
	uint16_t	pd_upper;		/* offset to end of free space */
	uint16_t	pd_special;		/* offset to start of special space */
	uint16_t	pd_pagesize_version;
	TransactionId pd_prune_xid;	/* oldest prunable XID, or zero if none */
	ItemIdData	pd_linp[1];		/* line pointer array */
} PageHeaderData;

#define SizeOfPageHeaderData	(offsetof(PageHeaderData, pd_linp))
#define PageGetItemId(page, offsetNumber)			\
	(&((PageHeaderData *)(page))->pd_linp[(offsetNumber) - 1])
#define PageGetItem(page, lpp)						\
	((HeapTupleHeaderData *)((char *)(page) + ItemIdGetOffset(lpp)))
#define PageGetMaxOffsetNumber(page)				\
	(((PageHeaderData *) (page))->pd_lower <= SizeOfPageHeaderData ? 0 :	\
	 ((((PageHeaderData *) (page))->pd_lower - SizeOfPageHeaderData)		\
	  / sizeof(ItemIdData)))
#define PD_HAS_FREE_LINES	0x0001	/* are there any unused line pointers? */
#define PD_PAGE_FULL		0x0002	/* not enough free space for new tuple? */
#define PD_ALL_VISIBLE		0x0004	/* all tuples on page are visible to
									 * everyone */
#define PD_VALID_FLAG_BITS	0x0007	/* OR of all valid pd_flags bits */

/*
 * GiST index specific structures and labels
 */
#define F_LEAF              (1 << 0)    /* leaf page */
#define F_DELETED           (1 << 1)    /* the page has been deleted */
#define F_TUPLES_DELETED    (1 << 2)    /* some tuples on the page were deleted */
#define F_FOLLOW_RIGHT      (1 << 3)    /* page to the right has no downlink */
#define F_HAS_GARBAGE       (1 << 4)    /* some tuples on the page are dead */

#define GIST_PAGE_ID        0xFF81

typedef struct GISTPageOpaqueData
{
	struct {
		uint32_t	xlogid;
		uint32_t	xrecoff;
	} nsn;
	BlockNumber		rightlink;		/* next page if any */
	uint16_t		flags;			/* see bit definitions above */
	uint16_t		gist_page_id;	/* for identification of GiST indexes */
} GISTPageOpaqueData;

INLINE_FUNCTION(GISTPageOpaqueData *)
GistPageGetOpaque(PageHeaderData *page)
{
	return (GISTPageOpaqueData *)((char *)page + page->pd_special);
}

INLINE_FUNCTION(bool)
GistPageIsLeaf(PageHeaderData *page)
{
	return (GistPageGetOpaque(page)->flags & F_LEAF) != 0;
}

INLINE_FUNCTION(bool)
GistPageIsDeleted(PageHeaderData *page)
{
	return (GistPageGetOpaque(page)->flags & F_DELETED) != 0;
}

INLINE_FUNCTION(bool)
GistFollowRight(PageHeaderData *page)
{
	return (GistPageGetOpaque(page)->flags & F_FOLLOW_RIGHT) != 0;
}
/* root page of a gist index */
#define GIST_ROOT_BLKNO			0

#endif /* POSTGRES_H */

/*
 * Definition of KDS-Items and related
 */
struct kern_tupitem
{
	uint32_t		t_len;		/* length of tuple */
	uint32_t		rowid;		/* unique Id of this item */
	HeapTupleHeaderData	htup;
};
typedef struct kern_tupitem		kern_tupitem;

/*
 * kern_hashitem - individual items for KDS_FORMAT_HASH
 */
struct kern_hashitem
{
	uint32_t		hash;		/* 32-bit hash value */
	uint32_t		next;		/* offset of the next (PACKED) */
	kern_tupitem	t;			/* HeapTuple of this entry */
};
typedef struct kern_hashitem	kern_hashitem;

/* Length of the header postion of kern_data_store */
INLINE_FUNCTION(size_t)
KDS_HEAD_LENGTH(kern_data_store *kds)
{
	return MAXALIGN(offsetof(kern_data_store, colmeta[kds->nr_colmeta]));
}

/* Base address of the kern_data_store */
INLINE_FUNCTION(char *)
KDS_BODY_ADDR(kern_data_store *kds)
{
	return (char *)kds + KDS_HEAD_LENGTH(kds);
}

/* access functions for KDS_FORMAT_HEAP/HASH */
INLINE_FUNCTION(uint32_t *)
KDS_GET_ROWINDEX(kern_data_store *kds)
{
	Assert(kds->format == KDS_FORMAT_HEAP ||
		   kds->format == KDS_FORMAT_HASH);
	return (uint32_t *)KDS_BODY_ADDR(kds);
}

/* kern_tupitem by kds_index */
INLINE_FUNCTION(kern_tupitem *)
KDS_GET_TUPITEM(kern_data_store *kds, uint32_t kds_index)
{
	uint32_t	offset = KDS_GET_ROWINDEX(kds)[kds_index];

	if (!offset)
		return NULL;
	return (kern_tupitem *)((char *)kds + __kds_unpack(offset));
}

/* kern_tupitem by tuple-offset */
INLINE_FUNCTION(HeapTupleHeaderData *)
KDS_FETCH_TUPITEM(kern_data_store *kds,
				  uint32_t tuple_offset,
				  ItemPointerData *p_self,
				  uint32_t *p_len)
{
	kern_tupitem   *tupitem;

	Assert(kds->format == KDS_FORMAT_HEAP ||
		   kds->format == KDS_FORMAT_HASH);
	if (tuple_offset == 0)
		return NULL;
	tupitem = (kern_tupitem *)((char *)kds
							   + __kds_unpack(tuple_offset)
							   - offsetof(kern_tupitem, htup));
	if (p_self)
		*p_self = tupitem->htup.t_ctid;
	if (p_len)
		*p_len = tupitem->t_len;
	return &tupitem->htup;
}

INLINE_FUNCTION(uint32_t *)
KDS_GET_HASHSLOT(kern_data_store *kds)
{
	Assert(kds->format == KDS_FORMAT_HASH);
	return (uint32_t *)(KDS_BODY_ADDR(kds) +
						MAXALIGN(sizeof(uint32_t) * kds->nrooms));
}

INLINE_FUNCTION(kern_hashitem *)
KDS_HASH_FIRST_ITEM(kern_data_store *kds, uint32_t hash)
{
    uint32_t   *slot = KDS_GET_HASHSLOT(kds);
	size_t		offset = __kds_unpack(slot[hash % kds->nslots]);

	if (offset == 0)
		return NULL;
	Assert(offset < kds->length);
	return (kern_hashitem *)((char *)kds + offset);
}

INLINE_FUNCTION(kern_hashitem *)
KDS_HASH_NEXT_ITEM(kern_data_store *kds, kern_hashitem *khitem)
{
	size_t      offset;

	if (!khitem || khitem->next == 0)
		return NULL;
	offset = __kds_unpack(khitem->next);
	Assert(offset < kds->length);
	return (kern_hashitem *)((char *)kds + offset);
}

/* access macros for KDS_FORMAT_BLOCK */
#if 0
//Only GPU?
#define KERN_DATA_STORE_PARTSZ(kds)             \
    Min(((kds)->nrows_per_block +               \
         warpSize - 1) & ~(warpSize - 1),       \
        get_local_size())
#endif
#define KDS_BLOCK_BLCKNR(kds,block_id)				\
	(((BlockNumber *)KDS_BODY_ADDR(kds))[block_id])
#define KDS_BLOCK_PGPAGE(kds,block_id)					\
	((struct PageHeaderData *)							\
	 (KDS_BODY_ADDR(kds) +								\
	  MAXALIGN(sizeof(BlockNumber) * (kds)->nrooms) +	\
      BLCKSZ * block_id))

INLINE_FUNCTION(HeapTupleHeaderData *)
KDS_BLOCK_REF_HTUP(kern_data_store *kds,
				   uint32_t lp_offset,
				   ItemPointerData *p_self,
				   uint32_t *p_len)
{
	/*
	 * NOTE: lp_offset is not packed offset!
	 * KDS_FORMAT_BLOCK must never be larger than 4GB.
	 */
	ItemIdData	   *lpp = (ItemIdData *)((char *)kds + lp_offset);
	uint32_t		head_size;
	uint32_t		block_id;
	BlockNumber		block_nr;
	PageHeaderData *pg_page;

	Assert(kds->format == KDS_FORMAT_BLOCK);
	if (lp_offset == 0)
		return NULL;
	head_size = (KDS_HEAD_LENGTH(kds) +
				 MAXALIGN(sizeof(BlockNumber) * kds->nrooms));
	Assert(lp_offset >= head_size &&
		   lp_offset <  head_size + BLCKSZ * kds->nitems);
    block_id = (lp_offset - head_size) / BLCKSZ;
    block_nr = KDS_BLOCK_BLCKNR(kds, block_id);
	pg_page = KDS_BLOCK_PGPAGE(kds, block_id);

	Assert(lpp >= pg_page->pd_linp &&
		   lpp -  pg_page->pd_linp < PageGetMaxOffsetNumber(pg_page));
	if (p_self)
	{
		p_self->ip_blkid.bi_hi  = block_nr >> 16;
		p_self->ip_blkid.bi_lo  = block_nr & 0xffff;
		p_self->ip_posid        = lpp - pg_page->pd_linp;
	}
	if (p_len)
		*p_len = ItemIdGetLength(lpp);
	return (HeapTupleHeaderData *)PageGetItem(pg_page, lpp);
}

/* access functions for apache arrow format */
INLINE_FUNCTION(void *)
KDS_ARROW_REF_SIMPLE_DATUM(kern_data_store *kds,
						   kern_colmeta *cmeta,
						   uint32_t index,
						   uint32_t unitsz)
{
	uint8_t	   *nullmap;
	char	   *values;

	Assert(cmeta >= &kds->colmeta[0] &&
		   cmeta <= &kds->colmeta[kds->nr_colmeta - 1]);
	if (cmeta->nullmap_offset)
	{
		nullmap = (uint8_t *)kds + __kds_unpack(cmeta->nullmap_offset);
		if (att_isnull(index, nullmap))
			return NULL;
	}
	Assert(cmeta->values_offset > 0);
	Assert(cmeta->extra_offset == 0);
	Assert(cmeta->extra_length == 0);
	Assert(unitsz * (index+1) <= __kds_unpack(cmeta->values_length));
	values = (char *)kds + __kds_unpack(cmeta->values_offset);
	return values + unitsz * index;
}

INLINE_FUNCTION(void *)
KDS_ARROW_REF_VARLENA_DATUM(kern_data_store *kds,
							kern_colmeta *cmeta,
							uint32_t rowidx,
							uint32_t *p_length)
{
	uint8_t	   *nullmap;
	uint32_t   *offset;
	char	   *extra;

	Assert(cmeta >= &kds->colmeta[0] &&
		   cmeta <= &kds->colmeta[kds->nr_colmeta - 1]);
	if (cmeta->nullmap_offset)
	{
		nullmap = (uint8_t *)kds + __kds_unpack(cmeta->nullmap_offset);
		if (att_isnull(rowidx, nullmap))
			return NULL;
	}
	Assert(cmeta->values_offset > 0 &&
		   cmeta->extra_offset > 0 &&
		   sizeof(uint32_t) * (rowidx+1) <= __kds_unpack(cmeta->values_length));
	offset = (uint32_t *)(kds + __kds_unpack(cmeta->values_length));
	extra = (char *)kds + __kds_unpack(cmeta->extra_offset);

	Assert(offset[rowidx]   <= offset[rowidx+1] &&
		   offset[rowidx+1] <= __kds_unpack(cmeta->extra_length));
	*p_length = offset[rowidx+1] - offset[rowidx];
	return (extra + offset[rowidx]);	
}

/* ----------------------------------------------------------------
 *
 * Definitions of Varlena datum and related (mostly in postgres.h and c.h)
 *
 * ----------------------------------------------------------------
 */
typedef struct varlena		varlena;
#ifndef POSTGRES_H
struct varlena {
    char		vl_len_[4];	/* Do not touch this field directly! */
	char		vl_dat[1];
};

#define VARHDRSZ			((int) sizeof(uint32_t))
#define VARDATA(PTR)		VARDATA_4B(PTR)
#define VARSIZE(PTR)		VARSIZE_4B(PTR)

#define VARSIZE_SHORT(PTR)	VARSIZE_1B(PTR)
#define VARDATA_SHORT(PTR)	VARDATA_1B(PTR)

typedef union
{
	struct						/* Normal varlena (4-byte length) */
	{
		uint32_t	va_header;
		char		va_data[1];
	}		va_4byte;
	struct						/* Compressed-in-line format */
	{
		uint32_t	va_header;
		uint32_t	va_tcinfo;	/* Original data size (excludes header) and
								 * compression method; see va_extinfo */
		char		va_data[1];	/* Compressed data */
	}		va_compressed;
} varattrib_4b;

typedef struct
{
    uint8_t			va_header;
    char			va_data[1];	/* Data begins here */
} varattrib_1b;

/* inline portion of a short varlena pointing to an external resource */
typedef struct
{
	uint8_t			va_header;	/* Always 0x80 or 0x01 */
	uint8_t			va_tag;		/* Type of datum */
    char			va_data[1];	/* Data (of the type indicated by va_tag) */
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
	int32_t		va_rawsize;		/* Original data size (includes header) */
	uint32_t	va_extsize;		/* External saved size (doesn't) */
	Oid			va_valueid;		/* Unique ID of value within TOAST table */
	Oid			va_toastrelid;	/* RelID of TOAST table containing it */
} varatt_external;

typedef struct varatt_indirect
{
	uintptr_t	pointer;		/* Host pointer to in-memory varlena */
} varatt_indirect;

#define VARTAG_SIZE(tag) \
	((tag) == VARTAG_INDIRECT ? sizeof(varatt_indirect) :   \
	 (tag) == VARTAG_ONDISK ? sizeof(varatt_external) :     \
	 0 /* should not happen */)

#define VARHDRSZ_EXTERNAL		offsetof(varattrib_1b_e, va_data)
#define VARTAG_EXTERNAL(PTR)	VARTAG_1B_E(PTR)
#define VARSIZE_EXTERNAL(PTR)	\
	(VARHDRSZ_EXTERNAL + VARTAG_SIZE(VARTAG_EXTERNAL(PTR)))

/*
 * compressed varlena format
 */
typedef struct toast_compress_header
{
	int32_t		vl_len_;		/* varlena header (do not touch directly!) */
    uint32_t	rawsize;		/* 2 bits for compression method and 30bits
								 * external size; see va_extinfo */
} toast_compress_header;

#define TOAST_COMPRESS_HDRSZ        ((uint32_t)sizeof(toast_compress_header))
#define TOAST_COMPRESS_RAWSIZE(ptr)             \
    (((toast_compress_header *) (ptr))->rawsize)
#define TOAST_COMPRESS_RAWDATA(ptr)             \
    (((char *) (ptr)) + TOAST_COMPRESS_HDRSZ)
#define TOAST_COMPRESS_SET_RAWSIZE(ptr, len)    \
    (((toast_compress_header *) (ptr))->rawsize = (len))

/* basic varlena macros */
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
#define VARATT_IS_COMPRESSED(PTR)		VARATT_IS_4B_C(PTR)
#define VARATT_IS_EXTERNAL(PTR)			VARATT_IS_1B_E(PTR)
#define VARATT_IS_EXTERNAL_ONDISK(PTR)		\
	(VARATT_IS_EXTERNAL(PTR) && VARTAG_EXTERNAL(PTR) == VARTAG_ONDISK)
#define VARATT_IS_EXTERNAL_INDIRECT(PTR)	\
	(VARATT_IS_EXTERNAL(PTR) && VARTAG_EXTERNAL(PTR) == VARTAG_INDIRECT)
#define VARATT_IS_SHORT(PTR)			VARATT_IS_1B(PTR)
#define VARATT_IS_EXTENDED(PTR)			(!VARATT_IS_4B_U(PTR))
#define VARATT_NOT_PAD_BYTE(PTR)		(*((uint8_t *) (PTR)) != 0)

#define VARSIZE_4B(PTR)                     \
	((__Fetch(&((varattrib_4b *)(PTR))->va_4byte.va_header)>>2) & 0x3FFFFFFF)
#define VARSIZE_1B(PTR) \
	((((varattrib_1b *) (PTR))->va_header >> 1) & 0x7F)
#define VARTAG_1B_E(PTR) \
	(((varattrib_1b_e *) (PTR))->va_tag)

#define VARRAWSIZE_4B_C(PTR)    \
	__Fetch(&((varattrib_4b *) (PTR))->va_compressed.va_rawsize)

#define VARSIZE_ANY_EXHDR(PTR) \
	(VARATT_IS_1B_E(PTR) ? VARSIZE_EXTERNAL(PTR)-VARHDRSZ_EXTERNAL : \
	 (VARATT_IS_1B(PTR) ? VARSIZE_1B(PTR)-VARHDRSZ_SHORT :			 \
	  VARSIZE_4B(PTR)-VARHDRSZ))

#define VARSIZE_ANY(PTR)							\
    (VARATT_IS_1B_E(PTR) ? VARSIZE_EXTERNAL(PTR) :	\
     (VARATT_IS_1B(PTR) ? VARSIZE_1B(PTR) :			\
      VARSIZE_4B(PTR)))

#define VARDATA_4B(PTR)	(((varattrib_4b *) (PTR))->va_4byte.va_data)
#define VARDATA_1B(PTR)	(((varattrib_1b *) (PTR))->va_data)
#define VARDATA_ANY(PTR) \
	(VARATT_IS_1B(PTR) ? VARDATA_1B(PTR) : VARDATA_4B(PTR))

#define SET_VARSIZE(PTR, len)                       \
	(((varattrib_4b *)(PTR))->va_4byte.va_header = (((uint32_t) (len)) << 2))
#endif	/* POSTGRES_H */

/* ----------------------------------------------------------------
 *
 * Definitions for XPU device data types
 *
 * ----------------------------------------------------------------
 */
#define TYPE_OPCODE(NAME,a,b)	TypeOpCode__##NAME,
typedef enum {
	TypeOpCode__Invalid = 0,
#include "xpu_opcodes.h"
	TypeOpCode__composite,
	TypeOpCode__array,
	TypeOpCode__record,
	TypeOpCode__unsupported,
	TypeOpCode__BuiltInMax,
} TypeOpCode;

typedef struct xpu_datum_t		xpu_datum_t;
typedef struct xpu_datum_operators xpu_datum_operators;

#define XPU_DATUM_COMMON_FIELD			\
	const xpu_datum_operators *ops;		\
	bool		isnull

struct xpu_datum_t {
	XPU_DATUM_COMMON_FIELD;
};

struct xpu_datum_operators {
	const char *xpu_type_name;
	TypeOpCode	xpu_type_code;
	int			xpu_type_sizeof;	/* =sizeof(xpu_XXXX_t), not PG type! */
	bool	  (*xpu_datum_ref)(kern_context *kcxt,
							   xpu_datum_t *result,
							   const kern_colmeta *cmeta,
							   const void *addr, int len);
	int		  (*xpu_datum_store)(kern_context *kcxt,
								 char *buffer,
								 xpu_datum_t *arg);
	bool	  (*xpu_datum_hash)(kern_context *kcxt,
								uint32_t *p_hash,
								xpu_datum_t *arg);
};

#define PGSTROM_SQLTYPE_SIMPLE_DECLARATION(NAME,BASETYPE)	\
	typedef struct {										\
		XPU_DATUM_COMMON_FIELD;								\
		BASETYPE	value;									\
	} xpu_##NAME##_t;										\
	EXTERN_DATA xpu_datum_operators xpu_##NAME##_ops
#define PGSTROM_SQLTYPE_VARLENA_DECLARATION(NAME)			\
	typedef struct {										\
		XPU_DATUM_COMMON_FIELD;								\
		int			length;		/* -1, if PG verlena */		\
		const char *value;									\
	} xpu_##NAME##_t;										\
	EXTERN_DATA xpu_datum_operators xpu_##NAME##_ops
#define PGSTROM_SQLTYPE_OPERATORS(NAME)						\
	PUBLIC_DATA xpu_datum_operators xpu_##NAME##_ops = {	\
		.xpu_type_name = #NAME,								\
		.xpu_type_code = TypeOpCode__##NAME,				\
		.xpu_type_sizeof = sizeof(xpu_##NAME##_t),			\
		.xpu_datum_ref = xpu_##NAME##_datum_ref,			\
		.xpu_datum_store = xpu_##NAME##_datum_store,		\
		.xpu_datum_hash = xpu_##NAME##_datum_hash,			\
	}

#include "xpu_basetype.h"
#include "xpu_numeric.h"
#include "xpu_textlib.h"
#include "xpu_timelib.h"
#include "xpu_misclib.h"

/*
 * pg_array_t - array type support
 *
 * NOTE: pg_array_t is designed to store both of PostgreSQL / Arrow array
 * values. If @length < 0, it means @value points a varlena based PostgreSQL
 * array values; which includes nitems, dimension, nullmap and so on.
 * Elsewhere, @length means number of elements, from @start of the array on
 * the columnar buffer by @smeta.
 */
typedef struct {
	XPU_DATUM_COMMON_FIELD;
	char	   *value;
	int			length;
	uint32_t	start;
	kern_colmeta *smeta;
} pg_array_t;
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(array, pg_array_t);

/*
 * pg_composite_t - composite type support
 *
 * NOTE: pg_composite_t is designed to store both of PostgreSQL / Arrow composite
 * values. If @nfields < 0, it means @value.htup points a varlena base PostgreSQL
 * composite values. Elsewhere (@nfields >= 0), it points composite values on
 * KDS_FORMAT_ARROW chunk. In this case, smeta[0] ... smeta[@nfields-1] describes
 * the values array on the KDS.
 */
typedef struct {
	XPU_DATUM_COMMON_FIELD;
	int16_t		nfields;
	uint32_t	rowidx;
	char	   *value;
	Oid			comp_typid;
	int			comp_typmod;
	kern_colmeta *smeta;
} pg_composite_t;
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(composite, pg_composite_t);

typedef struct {
	TypeOpCode		type_opcode;
	xpu_datum_operators *type_ops;
} xpu_type_catalog_entry;

EXTERN_DATA xpu_type_catalog_entry	builtin_xpu_types_catalog[];

/* ----------------------------------------------------------------
 *
 * Definition of device flags
 *
 * ---------------------------------------------------------------- */
#define DEVKERN__NVIDIA_GPU			0x0001UL	/* for CUDA-based GPU */
#define DEVKERN__NVIDIA_DPU			0x0002UL	/* for BlueField-X DPU */
#define DEVKERN__ARMv8_SPU			0x0004UL	/* for ARMv8-based SPU */
#define DEVKERN__ANY				0x0007UL	/* Runnable on any xPU */
#define DEVFUNC__LOCALE_AWARE		0x0100UL	/* Device function is locale aware,
												 * thus, available only if "C" or
												 * no locale configuration */
#define DEVKERN__SESSION_TIMEZONE	0x0200UL	/* Device function needs session
												 * timezone */

/* ----------------------------------------------------------------
 *
 * Definition of device functions
 *
 * ---------------------------------------------------------------- */
#define FUNC_OPCODE(a,b,c,NAME,d,e)		FuncOpCode__##NAME,
typedef enum {
	FuncOpCode__Invalid = 0,
	FuncOpCode__ConstExpr,
	FuncOpCode__ParamExpr,
	FuncOpCode__VarExpr,
	//FuncOpCode__VarAsIsExpr ... only used in projection
	FuncOpCode__BoolExpr_And,
	FuncOpCode__BoolExpr_Or,
	FuncOpCode__BoolExpr_Not,
	FuncOpCode__NullTestExpr_IsNull,
	FuncOpCode__NullTestExpr_IsNotNull,
	FuncOpCode__BoolTestExpr_IsTrue,
	FuncOpCode__BoolTestExpr_IsNotTrue,
	FuncOpCode__BoolTestExpr_IsFalse,
	FuncOpCode__BoolTestExpr_IsNotFalse,
	FuncOpCode__BoolTestExpr_IsUnknown,
	FuncOpCode__BoolTestExpr_IsNotUnknown,
#include "xpu_opcodes.h"
	/* for projection */
	FuncOpCode__Projection,
	FuncOpCode__LoadVars,
	FuncOpCode__BuiltInMax,
} FuncOpCode;

typedef struct kern_expression	kern_expression;
#define XPU_PGFUNCTION_ARGS		kern_context *kcxt,			\
								const kern_expression *kexp,\
								xpu_datum_t *__result
typedef bool  (*xpu_function_t)(XPU_PGFUNCTION_ARGS);

struct kern_expression
{
	uint32_t		_vl_len;
	FuncOpCode		opcode;
	xpu_function_t	fn_dptr;		/* to be set by xPU service */
	int				nargs;
	TypeOpCode		exptype;
	const xpu_datum_operators *exptype_ops;	/* to be set by xPU service */
	union {
		char			data[1]			__attribute__((aligned(MAXIMUM_ALIGNOF)));
		struct {
			Oid			const_type;
			bool		const_isnull;
			char		const_value[1]	__attribute__((aligned(MAXIMUM_ALIGNOF)));
		} c;		/* ConstExpr */
		struct {
			uint32_t	param_id;
		} p;		/* ParamExpr */
		struct {
			int16_t		var_typlen;
			bool		var_typbyval;
			uint8_t		var_typalign;
			uint32_t	var_slot_id;
		} v;		/* VarExpr */
	} u;
};

typedef struct
{
	int16_t		var_depth;
	int16_t		var_resno;
	uint32_t	var_slot_id;
} __kern_preload_vars_item;

typedef struct
{
	uint32_t		_vl_len;
	int				nloads;
	__kern_preload_vars_item kvars[1];
} kern_preload_vars;

typedef struct
{
	uint32_t		_vl_len;
	int				nexprs;		/* number of expressions to be computed */
	int				nattrs;		/* number of destination attribute */
	uint32_t		slot_id[1];	/* destination or source slot-id */
	/*
	 * ----------------------
	 * ^  slot_id[0]            slot_id to store the projection result that
	 * |     :                  involves any calculations.
	 * V  slot_id[nexprs-1]     'nexprs' should equal to 'nargs' of Projection
	 * ----------------------
	 * ^  slot_id[nexprs]       source slot_id ordered by the attributes in
	 * |     :                  the destination tuple. Same slot_id may appear
	 * |     :                  several times.
	 * V  slot_id[nexprs + nattrs - 1]
	 *    slot_id[nexprs + nattrs] --> size of this kern_projection_map.
	 */
} kern_projection_map;

#define EXEC_KERN_EXPRESSION(__kcxt,__kexp,__retval)	\
	(__kexp)->fn_dptr((__kcxt),(__kexp),(xpu_datum_t *)__retval)
#define KEXP_OVERRUN_CHECKS(__kexp,__arg)						\
	assert((char *)(__arg) + VARSIZE(__arg) <= (char *)(__kexp) + VARSIZE(__kexp))

INLINE_FUNCTION(const kern_expression *)
__KEXP_FIRST_ARG(int nargs, const kern_expression *kexp, TypeOpCode exptype)
{
	const kern_expression *arg = NULL;

	assert(kexp->nargs == nargs || nargs < 0);
	if (kexp->nargs > 0)
	{
		arg = ((const kern_expression *)((kexp)->u.data));
		assert((char *)arg + VARSIZE(arg) <= (char *)kexp + VARSIZE(kexp));
		assert(arg->exptype == exptype || exptype == TypeOpCode__Invalid);
	}
	return arg;
}
#define KEXP_FIRST_ARG(__nargs, __exptype)		\
	__KEXP_FIRST_ARG((__nargs), kexp, TypeOpCode__##__exptype)

INLINE_FUNCTION(const kern_expression *)
__KEXP_NEXT_ARG(const kern_expression *kexp,
				const kern_expression *prev, TypeOpCode exptype)
{
	const kern_expression *next = (const kern_expression *)
		((const char *)prev + MAXALIGN(VARSIZE(prev)));
	assert((char *)next + VARSIZE(next) <= (char *)kexp + VARSIZE(kexp));
	assert(next->exptype == exptype);
	return next;
}
#define KEXP_NEXT_ARG(__prev, __exptype)		\
	__KEXP_NEXT_ARG(kexp, (__prev), TypeOpCode__##__exptype)

#define SizeOfKernExpr(__PAYLOAD_SZ)						\
	(offsetof(kern_expression, u.data) + (__PAYLOAD_SZ))
#define SizeOfKernExprConst(__PAYLOAD_SZ)	\
	(offsetof(kern_expression, u.c.const_value) + (__PAYLOAD_SZ))
#define SizeOfKernExprParam					\
	(offsetof(kern_expression, u.p.param_id) + sizeof(uint32_t))
#define SizeOfKernExprVar					\
	(offsetof(kern_expression, u.v.var_slot_id) + sizeof(int))
typedef struct {
	FuncOpCode		func_opcode;
	xpu_function_t	func_dptr;
} xpu_function_catalog_entry;

EXTERN_DATA xpu_function_catalog_entry	builtin_xpu_functions_catalog[];




/*
 * PG-Strom Command Tag
 */
#define XpuCommandTag__Success			0
#define XpuCommandTag__Error			1
#define XpuCommandTag__CPUFallback		2
#define XpuCommandTag__OpenSession		100
#define XpuCommandTag__XpuScanExec		200
#define XpuCommandMagicNumber			0xdeadbeafU

/*
 * kernSessionInfo - A set of immutable data during query execution
 * (like, transaction info, timezone, parameter buffer).
 */
typedef struct kernSessionInfo
{
	uint32_t	kcxt_extra_bufsz;	/* length of vlbuf[] */
	uint32_t	kcxt_kvars_nslots;	/* length of kvars slot */

	/* xpucode for this session */
	bool		xpucode_use_debug_code;
	uint32_t	xpucode_scan_quals;
	uint32_t	xpucode_scan_projs;

	/* database session info */
	uint64_t	xactStartTimestamp;	/* timestamp when transaction start */
	uint32_t	xact_id_array;		/* offset to array of xid */
	uint32_t	session_timezone;	/* offset to pg_tz */
	uint32_t	session_encode;		/* offset to xpu_encode_info;
									 * !! function pointer must be set by server */
	/* executor parameter buffer */
	uint32_t	nparams;	/* number of parameters */
	uint32_t	poffset[1];	/* offset of params */
} kernSessionInfo;

typedef struct {
	uint32_t	kds_src_offset;
	uint32_t	kds_dst_offset;
	char		data[1];
} kernExecScan;

typedef struct {
	kern_data_store		kds;
} kernExecResult;

#ifndef ILIST_H
typedef struct dlist_node dlist_node;
struct dlist_node
{
	dlist_node *prev;
	dlist_node *next;
};
#endif

typedef struct
{
	uint32_t	magic;
	uint32_t	tag;
	uint64_t	length;
	void	   *priv;
	dlist_node	chain;
	union {
		kern_errorbuf	error;
		kernSessionInfo	session;
		kernExecScan	scan;
		kernExecResult	result;
	} u;
} XpuCommand;

/*
 * kernSessionInfo utility functions.
 */
INLINE_FUNCTION(struct varlena *)
SESSION_XACT_ID_ARRAY(kernSessionInfo *session)
{
	if (session->xact_id_array == 0)
		return NULL;
	return (struct varlena *)((char *)session + session->xact_id_array);
}

INLINE_FUNCTION(struct pg_tz *)
SESSION_TIMEZONE(kernSessionInfo *session)
{
	if (session->session_timezone == 0)
		return NULL;
	return (struct pg_tz *)((char *)session + session->session_timezone);
}

INLINE_FUNCTION(struct xpu_encode_info *)
SESSION_ENCODE(kernSessionInfo *session)
{
	if (session->session_encode == 0)
		return NULL;
	return (struct xpu_encode_info *)((char *)session + session->session_encode);
}



/* ----------------------------------------------------------------
 *
 * PostgreSQL Device Functions (Built-in)
 *
 * ----------------------------------------------------------------
 */
#define FUNC_OPCODE(a,b,c,NAME,d,e)			\
	EXTERN_DATA bool pgfn_##NAME(XPU_PGFUNCTION_ARGS);
#include "xpu_opcodes.h"

/* ----------------------------------------------------------------
 *
 * Common XPU functions
 *
 * ----------------------------------------------------------------
 */
PUBLIC_FUNCTION(void)
pg_kern_ereport(kern_context *kcxt);	/* only host code */
PUBLIC_FUNCTION(uint32_t)
pg_hash_any(const void *ptr, int sz);

#endif	/* XPU_COMMON_H */
