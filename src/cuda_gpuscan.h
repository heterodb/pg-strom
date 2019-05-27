/*
 * cuda_gpuscan.h
 *
 * CUDA device code specific to GpuScan logic
 * --
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
#ifndef CUDA_GPUSCAN_H
#define CUDA_GPUSCAN_H
/*
 * kern_gpuscan
 */
struct kern_gpuscan {
	kern_errorbuf	kerror;
	cl_uint			grid_sz;
	cl_uint			block_sz;
	cl_uint			part_sz;			/* only KDS_FORMAT_BLOCK */
	cl_uint			nitems_in;
	cl_uint			nitems_out;
	cl_uint			extra_size;
	/* suspend/resume support */
	cl_uint			suspend_sz;			/* size of suspend context buffer */
	cl_uint			suspend_count;		/* # of suspended workgroups */
	cl_bool			resume_context;		/* true, if kernel should resume */
	kern_parambuf	kparams;
	/* <-- gpuscanSuspendContext --> */
	/* <-- gpuscanResultIndex (if KDS_FORMAT_ROW with no projection) -->*/
};
typedef struct kern_gpuscan		kern_gpuscan;

typedef struct
{
	cl_uint		part_index;
	cl_uint		line_index;
} gpuscanSuspendContext;

typedef struct
{
	cl_uint		nitems;
	cl_uint		results[FLEXIBLE_ARRAY_MEMBER];
} gpuscanResultIndex;

#define KERN_GPUSCAN_PARAMBUF(kgpuscan)			\
	(&((kern_gpuscan *)(kgpuscan))->kparams)
#define KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan)	\
	STROMALIGN(KERN_GPUSCAN_PARAMBUF(kgpuscan)->length)
#define KERN_GPUSCAN_SUSPEND_CONTEXT(kgpuscan, group_id) \
	((kgpuscan)->suspend_sz == 0				\
	 ? NULL										\
	 : ((gpuscanSuspendContext *)				\
		((char *)KERN_GPUSCAN_PARAMBUF(kgpuscan) + \
		 KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan))) + (group_id))
#define KERN_GPUSCAN_RESULT_INDEX(kgpuscan)		\
	((gpuscanResultIndex *)						\
	 ((char *)KERN_GPUSCAN_PARAMBUF(kgpuscan) +	\
	  KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan) +	\
	  STROMALIGN((kgpuscan)->suspend_sz)))
#define KERN_GPUSCAN_DMASEND_LENGTH(kgpuscan)	\
	(offsetof(kern_gpuscan, kparams) +			\
	 KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan))

#ifdef __CUDACC__
/*
 * Routines to support KDS_FORMAT_BLOCKS - This KDS format is used to load
 * raw PostgreSQL heap blocks to GPU without modification by CPU.
 * All CPU has to pay attention is, not to load rows which should not be
 * visible to the current scan snapshot.
 */
typedef cl_uint		TransactionId;

/* definitions at storage/itemid.h */
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
		(itemId)->lp_flags = LP_UNUSED;	\
		(itemId)->lp_off = 0;			\
		(itemId)->lp_len = 0;			\
	} while(0)

/* definitions at storage/off.h */
typedef cl_ushort		OffsetNumber;

#define InvalidOffsetNumber		((OffsetNumber) 0)
#define FirstOffsetNumber		((OffsetNumber) 1)
#define MaxOffsetNumber			((OffsetNumber) (BLCKSZ / sizeof(ItemIdData)))
#define OffsetNumberMask		(0xffff)	/* valid uint16 bits */

/* definitions at storage/bufpage.h */
typedef struct
{
	cl_uint			xlogid;			/* high bits */
	cl_uint			xrecoff;		/* low bits */
} PageXLogRecPtr;

typedef cl_ushort	LocationIndex;

typedef struct PageHeaderData
{
	/* XXX LSN is member of *any* block, not only page-organized ones */
	PageXLogRecPtr	pd_lsn;			/* LSN: next byte after last byte of xlog
									 * record for last change to this page */
	cl_ushort		pd_checksum;	/* checksum */
	cl_ushort		pd_flags;		/* flag bits, see below */
	LocationIndex	pd_lower;		/* offset to start of free space */
	LocationIndex	pd_upper;		/* offset to end of free space */
	LocationIndex	pd_special;		/* offset to start of special space */
	cl_ushort		pd_pagesize_version;
	TransactionId pd_prune_xid;		/* oldest prunable XID, or zero if none */
	ItemIdData		pd_linp[FLEXIBLE_ARRAY_MEMBER]; /* line pointer array */
} PageHeaderData;

#define SizeOfPageHeaderData	(offsetof(PageHeaderData, pd_linp))

#define PD_HAS_FREE_LINES	0x0001	/* are there any unused line pointers? */
#define PD_PAGE_FULL		0x0002	/* not enough free space for new tuple? */
#define PD_ALL_VISIBLE		0x0004	/* all tuples on page are visible to
									 * everyone */
#define PD_VALID_FLAG_BITS  0x0007	/* OR of all valid pd_flags bits */

#define PageGetItemId(page, offsetNumber)				\
	(&((PageHeaderData *)(page))->pd_linp[(offsetNumber) - 1])
#define PageGetItem(page, lpp)							\
	((HeapTupleHeaderData *)((char *)(page) + ItemIdGetOffset(lpp)))
STATIC_INLINE(cl_uint)
PageGetMaxOffsetNumber(PageHeaderData *page)
{
	cl_uint		pd_lower = page->pd_lower;

	return (pd_lower <= SizeOfPageHeaderData ? 0 :
			(pd_lower - SizeOfPageHeaderData) / sizeof(ItemIdData));
}
#endif	/* __CUDACC__ */

/*
 * KDS_BLOCK_REF_HTUP
 *
 * It pulls a HeapTupleHeader by a pair of KDS and lp_offset; 
 */
STATIC_INLINE(HeapTupleHeaderData *)
KDS_BLOCK_REF_HTUP(kern_data_store *kds,
				   cl_uint lp_offset,
				   ItemPointerData *p_self,
				   cl_uint *p_len)
{
	/*
	 * NOTE: lp_offset is not packed offset!
	 * KDS_FORMAT_BLOCK will be never larger than 4GB.
	 */
	ItemIdData	   *lpp = (ItemIdData *)((char *)kds + lp_offset);
	cl_uint			head_size;
	cl_uint			block_id;
	BlockNumber		block_nr;
	PageHeaderData *pg_page;

	Assert(kds->format == KDS_FORMAT_BLOCK);
	if (lp_offset == 0)
		return NULL;
	head_size = (KERN_DATA_STORE_HEAD_LENGTH(kds) +
				 STROMALIGN(sizeof(BlockNumber) * kds->nrooms));
	Assert(lp_offset >= head_size &&
		   lp_offset <  head_size + BLCKSZ * kds->nitems);
	block_id = (lp_offset - head_size) / BLCKSZ;
	block_nr = KERN_DATA_STORE_BLOCK_BLCKNR(kds, block_id);
	pg_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds, block_id);

	Assert(lpp >= pg_page->pd_linp &&
		   lpp -  pg_page->pd_linp <  PageGetMaxOffsetNumber(pg_page));
	if (p_self)
	{
		p_self->ip_blkid.bi_hi	= block_nr >> 16;
		p_self->ip_blkid.bi_lo	= block_nr & 0xffff;
		p_self->ip_posid		= lpp - pg_page->pd_linp;
	}
	if (p_len)
		*p_len = ItemIdGetLength(lpp);
	return (HeapTupleHeaderData *)PageGetItem(pg_page, lpp);
}

#ifdef __CUDACC__
/* to be generated from SQL */
DEVICE_FUNCTION(cl_bool)
gpuscan_quals_eval(kern_context *kcxt,
				   kern_data_store *kds,
				   ItemPointerData *t_self,
				   HeapTupleHeaderData *htup);

DEVICE_FUNCTION(cl_bool)
gpuscan_quals_eval_arrow(kern_context *kcxt,
						 kern_data_store *kds,
						 cl_uint src_index);

DEVICE_FUNCTION(void)
gpuscan_projection_tuple(kern_context *kcxt,
						 kern_data_store *kds_src,
						 HeapTupleHeaderData *htup,
						 ItemPointerData *t_self,
						 cl_char *tup_dclass,
						 Datum *tup_values);

DEVICE_FUNCTION(void)
gpuscan_projection_arrow(kern_context *kcxt,
						 kern_data_store *kds_src,
						 size_t src_index,
						 cl_char *tup_dclass,
						 Datum *tup_values);

/* libgpuscan.a */
DEVICE_FUNCTION(void)
kern_gpuscan_main_row(kern_context *kcxt,
					  kern_gpuscan *kgpuscan,
					  kern_data_store *kds_src,
					  kern_data_store *kds_dst,
					  bool has_device_projection);
DEVICE_FUNCTION(void)
kern_gpuscan_main_block(kern_context *kcxt,
						kern_gpuscan *kgpuscan,
						kern_data_store *kds_src,
						kern_data_store *kds_dst,
						bool has_device_projection);
DEVICE_FUNCTION(void)
kern_gpuscan_main_arrow(kern_context *kcxt,
						kern_gpuscan *kgpuscan,
						kern_data_store *kds_src,
						kern_data_store *kds_dst,
						bool has_device_projection);
#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUSCAN_H */
