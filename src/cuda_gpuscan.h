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
 * Sequential Scan using GPU/MIC acceleration
 *
 * It packs a kern_parambuf and kern_resultbuf structure within a continuous
 * memory ares, to transfer (usually) small chunk by one DMA call.
 *
 * +----------------+       -----
 * | kern_parambuf  |         ^
 * | +--------------+         |
 * | | length   o--------------------+
 * | +--------------+         |      | kern_vrelation is located just after
 * | | nparams      |         |      | the kern_parambuf (because of DMA
 * | +--------------+         |      | optimization), so head address of
 * | | poffset[0]   |         |      | kern_gpuscan + parambuf.length
 * | | poffset[1]   |         |      | points kern_resultbuf.
 * | |    :         |         |      |
 * | | poffset[M-1] |         |      |
 * | +--------------+         |      |
 * | | variable     |         |      |
 * | | length field |         |      |
 * | | for Param /  |         |      |
 * | | Const values |         |      |
 * | |     :        |         |      |
 * +-+--------------+  -----  |  <---+
 * | kern_resultbuf |    ^    |
 * | +--------------+    |    |  Area to be sent to OpenCL device.
 * | | nrels (=1)   |    |    |  Forward DMA shall be issued here.
 * | +--------------+    |    |
 * | | nitems       |    |    |
 * | +--------------+    |    |
 * | | nrooms (=N)  |    |    |
 * | +--------------+    |    |
 * | | errcode      |    |    V
 * | +--------------+    |  -----
 * | | rindex[0]    |    |
 * | | rindex[1]    |    |  Area to be written back from OpenCL device.
 * | |     :        |    |  Reverse DMA shall be issued here.
 * | | rindex[N-1]  |    V
 * +-+--------------+  -----
 *
 * Gpuscan kernel code assumes all the fields shall be initialized to zero.
 */
typedef struct {
	kern_errorbuf	kerror;
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
	(KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan) +	\
	 offsetof(kern_resultbuf, results[0]))
#define KERN_GPUSCAN_DMARECV_OFFSET(kgpuscan)	0
#define KERN_GPUSCAN_DMARECV_LENGTH(kgpuscan, nitems)	\
	(KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan) +			\
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

#define ItemIdGetOffset(itemId)		((itemId).lp_off)
#define ItemIdGetLength(itemId)		((itemId).lp_len)
#define ItemIdIsUsed(itemId)		((itemId).lp_flags != LP_UNUSED)
#define ItemIdIsNormal(itemId)		((itemId).lp_flags == LP_NORMAL)
#define ItemIdIsRedirected(itemId)	((itemId).lp_flags == LP_REDIRECT)
#define ItemIdIsDead(itemId)		((itemId).lp_flags == LP_DEAD)
#define ItemIdHasStorage(itemId)	((itemId).lp_len != 0)

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
	(((PageHeaderData *)(page))->pd_linp[(offsetNumber) - 1])
#define PageGetItem(page, itemId)				\
	((char *)(page) + ItemIdGetOffset(itemId))
STATIC_INLINE(cl_uint)
PageGetMaxOffsetNumber(PageHeaderData *page)
{
	cl_uint		pd_lower = __ldg(page->pd_lower);

	return (pd_lower <= SizeOfPageHeaderData ? 0 :
			(pd_lower - SizeOfPageHeaderData) / sizeof(ItemIdData));
}

/*
 * KERN_DATA_STORE_BLOCK_TUPITEM - setup a kern_tupitem from an offset
 * to itemId from the head of KDS
 */
STATIC_INLINE(void)
KERN_DATA_STORE_BLOCK_TUPITEM(kern_tupitem *tupitem,
							  kern_data_store *kds,
							  cl_uint lp_offset)
{
	ItemIdData	   *lpp = *((ItemIdData *)((char *)kds + lp_offset));
	ItemIdData		lp;
	cl_uint			head_size;
	cl_uint			block_id;
	BlockNumber		block_nr;
	PageHeaderData *pg_page;

	assert(__ldg(kds->format) == KDS_FORMAT_BLOCK);
	head_size = (KERN_DATA_STORE_HEAD_LENGTH(kds) +
				 STROMALIGN(sizeof(BlockNumber) * __ldg(kds->nrooms)));
	assert(lp_offset >= head_size &&
		   lp_offset <  head_size + BLCKSZ * __ldg(kds->nitems));
	block_id = (lp_offset - head_size) / BLOCKSZ;
	block_nr = KERN_DATA_STORE_BLOCK_BLCKNR(kds, block_id);
	pg_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds, block_id);

	assert(lpp >= pg_page->pd_linp &&
		   lpp -  pg_page->pd_linp <  PageGetMaxOffsetNumber(pg_page));

	tupitem->t_len	= ItemIdGetLength(*lpp);
	tupitem->t_self.ip_blkid.bi_hi = block_nr >> 16;
	tupitem->t_self.ip_blkid.bi_lo = block_nr & 0xffff;
	tupitem->t_self.ip_posid	= lpp - pg_page->pd_linp;
	tupitem->htup	= (HeapTupleHeaderData *)PageGetItem(pg_page, *lpp);
}

/*
 * forward declaration of the function to be generated on the fly
 */
STATIC_FUNCTION(cl_bool)
gpuscan_quals_eval(kern_context *kcxt,
				   kern_data_store *kds,
				   size_t kds_index);

/*
 * forward declaration of the function to be generated on the fly
 */
STATIC_FUNCTION(void)
gpuscan_projection(kern_context *kcxt,
				   kern_data_store *kds_src,
				   kern_tupitem *tupitem,
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
gpuscan_exec_quals_block(kern_gpuscan *kgpuscan,
						 kern_data_store *kds_src)
{
	kern_parambuf	   *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	kern_resultbuf	   *kresults = KERN_GPUSCAN_RESULTBUF(kgpuscan);
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
	HeapTupleHeaderData *htup;

	/* sanity checks */
	assert(__ldg(kds_src->format) == KDS_FORMAT_BLOCK);
	assert(__ldg(kds_src->nrows_per_block) > 1);
	part_sz = (__ldg(kds_src->nrows_per_block) + WARP_SZ - 1) & ~(WARP_SZ - 1);
	part_sz = Min(part_sz, get_local_size());
	assert(get_local_size() % part_sz == 0);
	part_id = (get_global_index() * (get_local_size() / part_id) +
			   get_local_id() / part_sz);
	/* get a PostgreSQL block on which this thread will perform on */
	if (part_id < kds_src->nitems)
	{
		pg_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds_src, part_id);
		n_lines = PageGetMaxOffsetNumber(pg_page);
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
	n_lines = PageGetMaxOffsetNumber(pg_page);
	curr_id = get_local_id() % part_sz;
	do {
		/* fetch a heap_tuple if valid */
		if (curr_id < n_lines)
		{
			ItemIdData	lp = PageGetItemId(pg_page, curr_id + 1);

			if (ItemIdIsNormal(lp))
				htup = PageGetItem(pg_page, lp);
			else
				htup = NULL;
		}
		else
			htup = NULL;

		/* evaluation of the qualifier */
		if (htup)
			rc = gpuscan_quals_eval(&kcxt, kds_src, htup);
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

			if (base + count > __ldg(kresults->nrooms))
			{
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
				goto out;
			}
			else if (rc)
			{
				/* OK, store the result */
				kresults->results[base + offset] = (cl_uint)....;
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
		if (get_local_id() % part_sz == 0 && curr_id < part_sz)
			atomicExch(&gand_flag, 1);
		__syncthreads();
	} while (gang_flag);
out:		
	/* write back error status if any */
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
}









/*
 * gpuscan_exec_quals_row
 *
 * kernel entrypoint of GpuScan for KDS_FORMAT_ROW
 */
KERNEL_FUNCTION(void)
gpuscan_exec_quals(kern_gpuscan *kgpuscan,
				   kern_data_store *kds_src)
{
	kern_parambuf  *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	kern_resultbuf *kresults = KERN_GPUSCAN_RESULTBUF(kgpuscan);
	kern_context	kcxt;
	size_t			kds_index = get_global_id();
	cl_bool			rc;
	cl_uint			offset;
	cl_uint			count;
	__shared__ cl_uint base;

	/* sanity checks */
	assert(kds_src->format == KDS_FORMAT_ROW);
	assert(!kresults->all_visible);

	INIT_KERNEL_CONTEXT(&kcxt,gpuscan_exec_quals_row,kparams);

	/* evaluate device qualifier */
	if (kds_index < kds_src->nitems)
		rc = gpuscan_quals_eval(&kcxt, kds_src, kds_index);
	else
		rc = false;

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
			kresults->results[base + offset] = (cl_uint)
				((char *)KERN_DATA_STORE_TUPITEM(kds_src, kds_index) -
				 (char *)kds_src);
		}
	}
	__syncthreads();
out:
	/* write back error status if any */
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
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
	cl_uint			offset;
	cl_uint			count;
	__shared__ cl_uint base;
	cl_uint			required;
	Datum			tup_values[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_isnull[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_internal[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_uint		   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);

	/*
	 * immediate bailout if previous stage already have error status
	 */
	kcxt.e = kgpuscan->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_projection_row, kparams);

	/* sanity checks */
	assert(kresults->nrels == 1);
	assert(kds_src->format == KDS_FORMAT_ROW);
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
		kern_tupitem   *tupitem_src;

		if (kresults->all_visible)
			tupitem_src = KERN_DATA_STORE_TUPITEM(kds_src, get_global_id());
		else
			tupitem_src = (kern_tupitem *)((char *)kds_src +
										   kresults->results[get_global_id()]);
		gpuscan_projection(&kcxt,
						   kds_src,
						   tupitem_src,
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
		required = 0;		/* not consume any buffer */

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
			kern_tupitem   *tupitem_dst
				= (kern_tupitem *)((char *)kds_dst + pos);

			tup_index[get_global_id()] = pos;
			form_kern_heaptuple(&kcxt, kds_dst, tupitem_dst,
								tup_values, tup_isnull, tup_internal);
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
	kern_tupitem   *tupitem;
	Datum		   *tup_values;
	cl_bool		   *tup_isnull;
#ifdef GPUSCAN_DEVICE_PROJECTION
	cl_bool			tup_internal[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
#endif

	/*
	 * immediate bailout if previous stage already have error status
	 */
	kcxt.e = kgpuscan->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_projection_slot, kparams);

	/* sanity checks */
	assert(kresults->nrels == 1);
	assert(kds_src->format == KDS_FORMAT_ROW);
	assert(kds_dst->format == KDS_FORMAT_SLOT);
	if (kresults->nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		goto out;
	}
	/* update number of visible items */
	dst_nitems = (kresults->all_visible ? kds_src->nitems : kresults->nitems);
	if (get_global_id() == 0)
		kds_dst->nitems = dst_nitems;
	if (dst_nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		goto out;
	}
	/* fetch the source tuple */
	if (get_global_id() < dst_nitems)
	{
		if (kresults->all_visible)
			tupitem = KERN_DATA_STORE_TUPITEM(kds_src, get_global_id());
		else
			tupitem = (kern_tupitem *)((char *)kds_src +
									   kresults->results[get_global_id()]);
	}
	else
		tupitem = NULL;

	tup_values = KERN_DATA_STORE_VALUES(kds_dst, get_global_id());
	tup_isnull = KERN_DATA_STORE_ISNULL(kds_dst, get_global_id());
#ifdef GPUSCAN_DEVICE_PROJECTION
	assert(kds_dst->ncols == GPUSCAN_DEVICE_PROJECTION_NFIELDS);
	gpuscan_projection(&kcxt,
					   kds_src,
					   tupitem,
					   kds_dst,
					   dst_nitems,
					   tup_values,
					   tup_isnull,
					   tup_internal);
#else
	if (tupitem != NULL)
	{
		deform_kern_heaptuple(&kcxt,
							  kds_src,
							  tupitem,
							  kds_dst->ncols,
							  true,
							  tup_values,
							  tup_isnull);
	}
#endif
out:
	/* write back error status if any */
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
}
#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUSCAN_H */
