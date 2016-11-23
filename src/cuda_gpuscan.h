/*
 * cuda_gpuscan.h
 *
 * CUDA device code specific to GpuScan logic
 * --
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
 * +-----------------+
 * | kern_gpuscan    |
 * | +---------------+
 * | | kern_errbuf   |
 * | +---------------+ ---
 * | | kern_parembuf |  ^
 * | |     :         |  | parameter buffer length
 * | |     :         |  v
 * | +---------------+ ---
 * | | kern_results  |
 * | |  @nrels = 1   |
 * | |     :         |
 * | |     :         |
 * +-+---------------+
 */
typedef struct {
	kern_errorbuf	kerror;
	/* performance profile */
	struct {
		cl_float	tv_kern_exec_quals;
		cl_float	tv_kern_projection;
	} pfm;
	kern_parambuf	kparams;
} kern_gpuscan;

#define KERN_GPUSCAN_PARAMBUF(kgpuscan)			\
	((kern_parambuf *)(&(kgpuscan)->kparams))
#define KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan)	\
	STROMALIGN((kgpuscan)->kparams.length)
#define KERN_GPUSCAN_RESULTBUF(kgpuscan)		\
	((kern_resultbuf *)((char *)&(kgpuscan)->kparams +				\
						STROMALIGN((kgpuscan)->kparams.length)))
#define KERN_GPUSCAN_RESULTBUF_LENGTH(kgpuscan)	\
	STROMALIGN(offsetof(kern_resultbuf,			\
		results[KERN_GPUSCAN_RESULTBUF(kgpuscan)->nrels * \
				KERN_GPUSCAN_RESULTBUF(kgpuscan)->nrooms]))
#define KERN_GPUSCAN_LENGTH(kgpuscan)			\
	(offsetof(kern_gpuscan, kparams) +			\
	 KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan) +	\
	 KERN_GPUSCAN_RESULTBUF_LENGTH(kgpuscan))
#define KERN_GPUSCAN_DMASEND_OFFSET(kgpuscan)	0
#define KERN_GPUSCAN_DMASEND_LENGTH(kgpuscan)	\
	(offsetof(kern_gpuscan, kparams) +			\
	 KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan) +	\
	 offsetof(kern_resultbuf, results[0]))
#define KERN_GPUSCAN_DMARECV_OFFSET(kgpuscan)	0
#define KERN_GPUSCAN_DMARECV_LENGTH(kgpuscan, nitems)	\
	(offsetof(kern_gpuscan, kparams) +					\
	 KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan) +			\
	 offsetof(kern_resultbuf, results[(nitems)]))

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
	cl_uint		pd_lower = __ldg(&page->pd_lower);

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
	ItemIdData	   *lpp = (ItemIdData *)((char *)kds + lp_offset);
	cl_uint			head_size;
	cl_uint			block_id;
	BlockNumber		block_nr;
	PageHeaderData *pg_page;

	Assert(__ldg(&kds->format) == KDS_FORMAT_BLOCK);
	head_size = (KERN_DATA_STORE_HEAD_LENGTH(kds) +
				 STROMALIGN(sizeof(BlockNumber) * __ldg(&kds->nrooms)));
	Assert(lp_offset >= head_size &&
		   lp_offset <  head_size + BLCKSZ * __ldg(&kds->nitems));
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
/*
 * forward declaration of the function to be generated on the fly
 */
STATIC_FUNCTION(cl_bool)
gpuscan_quals_eval(kern_context *kcxt,
				   kern_data_store *kds,
				   ItemPointerData *t_self,
				   HeapTupleHeaderData *htup);

/*
 * forward declaration of the function to be generated on the fly
 */
STATIC_FUNCTION(void)
gpuscan_projection(kern_context *kcxt,
				   kern_data_store *kds_src,
				   HeapTupleHeaderData *htup,
				   ItemPointerData *t_self,
				   kern_data_store *kds_dst,
				   cl_uint dst_nitems,
				   Datum *tup_values,
				   cl_bool *tup_isnull,
				   cl_bool *tup_internal);

/*
 * gpuscan_exec_quals_block
 *
 * kernel entrypoint of GpuScan for KDS_FORMAT_BLOCK
 */
KERNEL_FUNCTION(void)
gpuscan_exec_quals_block(kern_parambuf *kparams,
						 kern_resultbuf *kresults,
						 kern_data_store *kds_src,
						 kern_arg_t window_base,
						 kern_arg_t window_size)
{
	kern_context		kcxt;
	cl_uint				part_id;	/* partition index */
	cl_uint				part_sz;	/* partition size */
	cl_uint				curr_id;	/* index within partition */
	cl_uint				n_lines;
	cl_bool				rc;
	__shared__ cl_uint	gang_flag;
	__shared__ cl_uint	base;
	cl_uint				offset;
	cl_uint				count;
	PageHeaderData	   *pg_page;
	ItemPointerData		t_self;
	HeapTupleHeaderData *htup;

	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_exec_quals_block, kparams);

	/* sanity checks */
	assert(__ldg(&kds_src->format) == KDS_FORMAT_BLOCK);
	assert(!__ldg(&kresults->all_visible));
	assert(__ldg(&kds_src->nrows_per_block) > 1);
	assert(window_base + window_size <= kds_src->nitems);
	part_sz = (__ldg(&kds_src->nrows_per_block) +
			   warpSize - 1) & ~(warpSize - 1);
	part_sz = Min(part_sz, get_local_size());
	assert(get_local_size() % part_sz == 0);
	part_id = (get_global_index() * (get_local_size() / part_sz) +
			   get_local_id() / part_sz) + window_base;

	/* get a PostgreSQL block on which this thread will perform on */
	if (part_id < window_base + window_size)
	{
		BlockNumber	block_nr;

		pg_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds_src, part_id);
		n_lines = PageGetMaxOffsetNumber(pg_page);
		block_nr = KERN_DATA_STORE_BLOCK_BLCKNR(kds_src, part_id);
		t_self.ip_blkid.bi_hi = block_nr >> 16;
		t_self.ip_blkid.bi_lo = block_nr & 0xffff;
		t_self.ip_posid = InvalidOffsetNumber;
	}
	else
	{
		pg_page = NULL;
		n_lines = 0;
	}

	/*
	 * Walks on pd_linep[] array on the assigned PostgreSQL block.
	 * If a CUDA block is assigned to multiple PostgreSQL blocks, CUDA
	 * block should not exit until all the PostgreSQL blocks are processed
	 * due to restriction of thread synchronization.
	 */
	curr_id = get_local_id() % part_sz;
	do {
		ItemIdData	   *lpp;

		/* fetch a heap_tuple if valid */
		if (curr_id < n_lines)
		{
			lpp = PageGetItemId(pg_page, curr_id + 1);
			if (ItemIdIsNormal(lpp))
			{
				htup = PageGetItem(pg_page, lpp);
				t_self.ip_posid = curr_id + 1;
			}
			else
				htup = NULL;
		}
		else
			htup = NULL;

		/* evaluation of the qualifier */
		if (htup)
			rc = gpuscan_quals_eval(&kcxt, kds_src, &t_self, htup);
		else
			rc = false;
		__syncthreads();

		/* expand kresults to store the row offset */
		offset = pgstromStairlikeSum(rc ? 1 : 0, &count);
		if (count > 0)
		{
			if (get_local_id() == 0)
				base = atomicAdd(&kresults->nitems, count);
			__syncthreads();

			if (base + count > __ldg(&kresults->nrooms))
			{
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
				goto out;
			}
			else if (rc)
			{
				/* OK, store the result */
				kresults->results[base + offset] = (cl_uint)((char *)lpp -
															 (char *)kds_src);
			}
		}
		__syncthreads();

		/*
		 * Move to the next line item. If no threads in CUDA block wants to
		 * continue scan any more, we soon exit the loop.
		 */
		curr_id += part_sz;
		if (get_local_id() == 0)
			gang_flag = 0;
		__syncthreads();
		if (get_local_id() % part_sz == 0 && curr_id < n_lines)
			atomicExch(&gang_flag, 1);
		__syncthreads();
	} while (gang_flag);
out:		
	/* write back error status if any */
	kern_writeback_error_status(&kresults->kerror, kcxt.e);
}

/*
 * gpuscan_exec_quals_row
 *
 * kernel entrypoint of GpuScan for KDS_FORMAT_ROW
 */
KERNEL_FUNCTION(void)
gpuscan_exec_quals_row(kern_parambuf *kparams,
					   kern_resultbuf *kresults,
					   kern_data_store *kds_src,
					   kern_arg_t window_base,
					   kern_arg_t window_size)
{
	kern_context	kcxt;
	kern_tupitem   *tupitem = NULL;
	cl_bool			rc;
	cl_uint			offset;
	cl_uint			count;
	__shared__ cl_uint base;

	/* sanity checks */
	assert(kds_src->format == KDS_FORMAT_ROW);
	assert(!kresults->all_visible);
	assert(window_base + window_size <= kds_src->nitems);

	INIT_KERNEL_CONTEXT(&kcxt,gpuscan_exec_quals_row,kparams);

	if (get_global_id() < window_size)
	{
		tupitem = KERN_DATA_STORE_TUPITEM(kds_src, (window_base +
													get_global_id()));
		rc = gpuscan_quals_eval(&kcxt, kds_src,
								&tupitem->t_self,
								&tupitem->htup);
	}
	else
	{
		rc = false;
	}

	/* expand kresults buffer */
	offset = pgstromStairlikeSum(rc ? 1 : 0, &count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kresults->nitems, count);
		__syncthreads();

		if (base + count > kresults->nrooms)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			goto out;
		}
		else if (rc)
		{
			/* OK, store the result */
			kresults->results[base + offset] = (cl_uint)((char *)tupitem -
														 (char *)kds_src);
		}
	}
	__syncthreads();
out:
	/* write back error status if any */
	kern_writeback_error_status(&kresults->kerror, kcxt.e);
}

#ifdef GPUSCAN_DEVICE_PROJECTION
/*
 * gpuscan_projection_row
 *
 * It constructs a result tuple of GpuScan according to the required layout
 * of the result tuple. In case when row-format is required, host code never
 * call the device projection kernel unless result layout is not compatible.
 * So, entire kernel function is within #ifdef ... #endif block
 */
KERNEL_FUNCTION(void)
gpuscan_projection_row(kern_gpuscan *kgpuscan,
					   kern_data_store *kds_src,
					   kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	kern_resultbuf *kresults = KERN_GPUSCAN_RESULTBUF(kgpuscan);
	kern_context	kcxt;
	cl_uint			dst_nitems;
	cl_uint			kds_offset;
	cl_uint			offset;
	cl_uint			count;
	__shared__ cl_uint base;
	cl_uint			required;
	kern_tupitem   *tupitem;
	Datum			tup_values[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_isnull[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_internal[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_uint		   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
	HeapTupleHeaderData *htup;
	ItemPointerData t_self;

	/*
	 * immediate bailout if previous stage already have error status
	 */
	kcxt.e = kgpuscan->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_projection_row, kparams);

	/* sanity checks */
	assert(kresults->nrels == 1);
	assert(kds_src->format == KDS_FORMAT_ROW ||
		   (kds_src->format == KDS_FORMAT_BLOCK && !kresults->all_visible));
	assert(kds_dst->format == KDS_FORMAT_ROW && kds_dst->nslots == 0);
	/* update number of visible items */
	dst_nitems = (kresults->all_visible ? kds_src->nitems : kresults->nitems);
	if (get_global_id() == 0)
		kds_dst->nitems = dst_nitems;
	if (dst_nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		goto out;
	}

	/*
	 * step.1 - compute length of the result tuple to be written
	 */
	memset(tup_internal, 0, sizeof(tup_internal));

	if (get_global_id() < dst_nitems)
	{
		if (kds_src->format == KDS_FORMAT_ROW)
		{
			if (kresults->all_visible)
			{
				tupitem = KERN_DATA_STORE_TUPITEM(kds_src, get_global_id());
				t_self = tupitem->t_self;
				htup = &tupitem->htup;
			}
			else
			{
				kds_offset = kresults->results[get_global_id()];
				htup = KDS_ROW_REF_HTUP(kds_src, kds_offset, &t_self, NULL);
			}
		}
		else
		{
			assert(!kresults->all_visible);
			kds_offset = kresults->results[get_global_id()];
			htup = KDS_BLOCK_REF_HTUP(kds_src, kds_offset, &t_self, NULL);
		}
		/* extract to the private buffer */
		gpuscan_projection(&kcxt,
						   kds_src,
						   htup,
						   &t_self,
						   kds_dst,
						   dst_nitems,
						   tup_values,
						   tup_isnull,
						   tup_internal);
		required = MAXALIGN(offsetof(kern_tupitem, htup) +
							compute_heaptuple_size(&kcxt,
												   kds_dst,
												   tup_values,
												   tup_isnull,
												   tup_internal));
	}
	else
		required = 0;		/* out of range; never write anything */

	/*
	 * step.2 - increment the buffer usage of kds_dst
	 */
	offset = pgstromStairlikeSum(required, &count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kds_dst->usage, count);
		__syncthreads();

		if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
			STROMALIGN(sizeof(cl_uint) * kresults->nitems) +
			base + count > kds_dst->length)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			goto out;
		}
		else if (required > 0)
		{
			/*
			 * step.3 - extract the result heap-tuple
			 */
			cl_uint			pos = kds_dst->length - (base + offset + required);

			tup_index[get_global_id()] = pos;
			tupitem = (kern_tupitem *)((char *)kds_dst + pos);
			form_kern_heaptuple(&kcxt, kds_dst, tupitem,
								&t_self, tup_values, tup_isnull, tup_internal);
		}
	}
	__syncthreads();
out:
	/* write back error status if any */
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
}
#endif

KERNEL_FUNCTION(void)
gpuscan_projection_slot(kern_gpuscan *kgpuscan,
						kern_data_store *kds_src,
						kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	kern_resultbuf *kresults = KERN_GPUSCAN_RESULTBUF(kgpuscan);
	kern_context	kcxt;
	cl_uint			dst_nitems;
	cl_uint			kds_offset;
	Datum		   *tup_values;
	cl_bool		   *tup_isnull;
#ifdef GPUSCAN_DEVICE_PROJECTION
	cl_bool			tup_internal[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
#endif
	HeapTupleHeaderData *htup;
	ItemPointerData	t_self;

	/*
	 * immediate bailout if previous stage already have error status
	 */
	kcxt.e = kgpuscan->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_projection_slot, kparams);

	/* sanity checks */
	assert(kresults->nrels == 1);
	assert(kds_src->format == KDS_FORMAT_ROW ||
		   (kds_src->format == KDS_FORMAT_BLOCK && !kresults->all_visible));
	assert(kds_dst->format == KDS_FORMAT_SLOT);

	/* update number of visible items */
	dst_nitems = (kresults->all_visible ? kds_src->nitems : kresults->nitems);
	if (dst_nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		goto out;
	}
	if (get_global_id() == 0)
		kds_dst->nitems = dst_nitems;
	assert(get_global_size() >= dst_nitems);

	/*
	 * Fetch a HeapTuple / ItemPointer from KDS
	 */
	if (get_global_id() < dst_nitems)
	{
		if (kds_src->format == KDS_FORMAT_ROW)
		{
			if (kresults->all_visible)
			{
				kern_tupitem   *tupitem
					= KERN_DATA_STORE_TUPITEM(kds_src, get_global_id());
				t_self = tupitem->t_self;
				htup = &tupitem->htup;
			}
			else
			{
				kds_offset = kresults->results[get_global_id()];
				htup = KDS_ROW_REF_HTUP(kds_src, kds_offset, &t_self, NULL);
			}
		}
		else
		{
			assert(!kresults->all_visible);
			kds_offset = kresults->results[get_global_id()];
			htup = KDS_BLOCK_REF_HTUP(kds_src, kds_offset, &t_self, NULL);
		}
	}
	else
		htup = NULL;	/* out of range */

	/* destination buffer */
	tup_values = KERN_DATA_STORE_VALUES(kds_dst, get_global_id());
	tup_isnull = KERN_DATA_STORE_ISNULL(kds_dst, get_global_id());

#ifdef GPUSCAN_DEVICE_PROJECTION
	assert(kds_dst->ncols == GPUSCAN_DEVICE_PROJECTION_NFIELDS);
	gpuscan_projection(&kcxt,
					   kds_src,
					   htup,
					   &t_self,
					   kds_dst,
					   dst_nitems,
					   tup_values,
					   tup_isnull,
					   tup_internal);
#else
	deform_kern_heaptuple(&kcxt,
						  kds_src,
						  htup,
						  kds_dst->ncols,
						  true,
						  tup_values,
						  tup_isnull);
#endif
out:
	/* write back error status if any */
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
}

/*
 * gpuscan_main - controller function of GpuScan logic
 */
KERNEL_FUNCTION(void)
gpuscan_main(kern_gpuscan *kgpuscan,
			 kern_data_store *kds_src,
			 kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	kern_resultbuf *kresults = KERN_GPUSCAN_RESULTBUF(kgpuscan);
	kern_context	kcxt;
	void		   *kernel_func;
	void		  **kernel_args;
	dim3			grid_sz;
	dim3			block_sz;
	cl_uint			part_sz;
	cl_uint			ntuples;
	cl_ulong		tv1, tv2;
	cudaError_t		status = cudaSuccess;

	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_main, kparams);
	/* sanity checks */
	assert(get_global_size() == 1);		/* only single thread */
	assert(kds_src->format == KDS_FORMAT_ROW ||
		   (kds_src->format == KDS_FORMAT_BLOCK && !kresults->all_visible));
	assert(!kds_dst ||
		   kds_dst->format == KDS_FORMAT_ROW ||
		   kds_dst->format == KDS_FORMAT_SLOT);
	/*
	 * (1) Evaluation of Scan qualifiers
	 */
	if (!kresults->all_visible)
	{
		tv1 = GlobalTimer();

		if (kds_src->format == KDS_FORMAT_ROW)
		{
			kernel_func = (void *)gpuscan_exec_quals_row;
			part_sz = 0;
			ntuples = kds_src->nitems;
		}
		else
		{
			kernel_func = (void *)gpuscan_exec_quals_block;
			part_sz = (kds_src->nrows_per_block +
					   warpSize - 1) & ~(warpSize - 1);
			ntuples = part_sz * kds_src->nitems;
		}
		status = optimal_workgroup_size(&grid_sz,
										&block_sz,
										kernel_func,
										ntuples,
										0,
										sizeof(cl_uint));
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}

		/*
		 * adjust block-size if KDS_FORMAT_BLOCK and partition size is
		 * less than the block-size.
		 */
		if (kds_src->format == KDS_FORMAT_BLOCK &&
			part_sz < block_sz.x)
		{
			cl_uint		nparts_per_block = block_sz.x / part_sz;

			block_sz.x = nparts_per_block * part_sz;
			grid_sz.x = (kds_src->nitems +
						 nparts_per_block - 1) / nparts_per_block;
		}

		kernel_args = (void **)
			cudaGetParameterBuffer(sizeof(void *),
								   sizeof(void *) * 5);
		if (!kernel_args)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
			goto out;
		}
		kernel_args[0] = KERN_GPUSCAN_PARAMBUF(kgpuscan);
		kernel_args[1] = KERN_GPUSCAN_RESULTBUF(kgpuscan);
		kernel_args[2] = kds_src;
		kernel_args[3] = (void *)(0);				/* window_base */
		kernel_args[4] = (void *)(kds_src->nitems);	/* window_size */

		status = cudaLaunchDevice(kernel_func,
								  kernel_args,
								  grid_sz,
								  block_sz,
								  sizeof(cl_uint) * block_sz.x,
								  NULL);
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}

		if (kresults->kerror.errcode != StromError_Success)
		{
			kcxt.e = kresults->kerror;
			goto out;
		}
		ntuples = kresults->nitems;

		tv2 = GlobalTimer();
	}
	else
	{
		/* all-visible, thus all the source tuples are valid input */
		ntuples = kds_src->nitems;
		/* gpuscan_exec_quals_* was not executed  */
		tv1 = tv2 = 0;
	}
	kgpuscan->pfm.tv_kern_exec_quals = (cl_float)(tv2 - tv1) / 1000000.0;

	/*
     * (2) Projection of the results
     */
	if (kds_dst && ntuples > 0)
	{
		tv1 = GlobalTimer();

		if (kds_dst->format == KDS_FORMAT_ROW)
			kernel_func = (void *)gpuscan_projection_row;
		else
			kernel_func = (void *)gpuscan_projection_slot;

		status = optimal_workgroup_size(&grid_sz,
                                        &block_sz,
										kernel_func,
										ntuples,
										0, sizeof(cl_uint));
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}

		kernel_args = (void **)
			cudaGetParameterBuffer(sizeof(void *),
								   sizeof(void *) * 3);
		if (!kernel_args)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
			goto out;
		}
		kernel_args[0] = kgpuscan;
		kernel_args[1] = kds_src;
		kernel_args[2] = kds_dst;

		status = cudaLaunchDevice(kernel_func,
								  kernel_args,
								  grid_sz,
								  block_sz,
								  sizeof(cl_uint) * block_sz.x,
								  NULL);
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}

		if (kresults->kerror.errcode != StromError_Success)
		{
			kcxt.e = kresults->kerror;
			goto out;
		}
		tv2 = GlobalTimer();
	}
	else
	{
		/* gpuscan_projection_* was not launched */
		tv1 = tv2 = 0;
	}
	kgpuscan->pfm.tv_kern_projection = (cl_float)(tv2 - tv1) / 1000000.0;
out:
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUSCAN_H */
