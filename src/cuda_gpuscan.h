/*
 * cuda_gpuscan.h
 *
 * CUDA device code specific to GpuScan logic
 * --
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
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
	((kern_parambuf *)(&(kgpuscan)->kparams))
#define KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan)	\
	STROMALIGN((kgpuscan)->kparams.length)
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
	/*
	 * NOTE: lp_offset is not packed offset!
	 * KDS_FORMAT_BLOCK will be never larger than 4GB.
	 */
	ItemIdData	   *lpp = (ItemIdData *)((char *)kds + lp_offset);
	cl_uint			head_size;
	cl_uint			block_id;
	BlockNumber		block_nr;
	PageHeaderData *pg_page;

	Assert(__ldg(&kds->format) == KDS_FORMAT_BLOCK);
	if (lp_offset == 0)
		return NULL;
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

STATIC_FUNCTION(cl_bool)
gpuscan_quals_eval_column(kern_context *kcxt,
						  kern_data_store *kds,
						  cl_uint src_index);

STATIC_FUNCTION(void)
gpuscan_projection_tuple(kern_context *kcxt,
						 kern_data_store *kds_src,
						 HeapTupleHeaderData *htup,
						 ItemPointerData *t_self,
						 Datum *tup_values,
						 cl_bool *tup_isnull,
						 char *extra_buf);

STATIC_FUNCTION(void)
gpuscan_projection_column(kern_context *kcxt,
						  kern_data_store *kds_src,
						  size_t src_index,
						  Datum *tup_values,
						  cl_bool *tup_isnull,
						  char *extra_buf);

#ifdef GPUSCAN_KERNEL_REQUIRED
/*
 * gpuscan_exec_quals_row - GpuScan logic for KDS_FORMAT_ROW
 */
KERNEL_FUNCTION(void)
gpuscan_exec_quals_row(kern_gpuscan *kgpuscan,
					   kern_data_store *kds_src,
					   kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	gpuscanSuspendContext *my_suspend
		= KERN_GPUSCAN_SUSPEND_CONTEXT(kgpuscan, get_group_id());
	gpuscanResultIndex *gs_results	__attribute__((unused))
		= KERN_GPUSCAN_RESULT_INDEX(kgpuscan);
	kern_context	kcxt;
	cl_uint			part_index = 0;
	cl_uint			src_index;
	cl_uint			src_base;
	cl_uint			nitems_offset;
	cl_uint			usage_offset	__attribute__((unused));
	cl_uint			total_nitems_in = 0;	/* stat */
	cl_uint			total_nitems_out = 0;	/* stat */
	cl_uint			total_extra_size = 0;	/* stat */
#ifdef GPUSCAN_HAS_DEVICE_PROJECTION
#if GPUSCAN_DEVICE_PROJECTION_NFIELDS > 0
	Datum			tup_values[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_isnull[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
#else
	Datum		   *tup_values = NULL;
	cl_bool		   *tup_isnull = NULL;
#endif
#if GPUSCAN_DEVICE_PROJECTION_EXTRA_SIZE > 0
	char			tup_extra[GPUSCAN_DEVICE_PROJECTION_EXTRA_SIZE]
					__attribute__ ((aligned(MAXIMUM_ALIGNOF)));
#else
	char		   *tup_extra = NULL;
#endif
#endif
	__shared__ cl_uint	nitems_base;
	__shared__ cl_ulong	usage_base	__attribute__((unused));

	assert(kds_src->format == KDS_FORMAT_ROW);
	assert(!kds_dst || kds_dst->format == KDS_FORMAT_ROW);
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_exec_quals_row, kparams);
	/* resume kernel from the point where suspended, if any */
	if (kgpuscan->resume_context)
	{
		assert(my_suspend != NULL);
		part_index = my_suspend->part_index;
	}

	for (src_base = get_global_base() + part_index * get_global_size();
		 src_base < kds_src->nitems;
		 src_base += get_global_size(), part_index++)
	{
		kern_tupitem   *tupitem;
		cl_bool			rc;
		cl_uint			nvalids;
		cl_uint			required	__attribute__((unused));
		cl_uint			extra_sz = 0;
		cl_uint			suspend_kernel	__attribute__((unused)) = 0;

		/* Evalidation of the rows by WHERE-clause */
		src_index = src_base + get_local_id();
		if (src_index < kds_src->nitems)
		{
			tupitem = KERN_DATA_STORE_TUPITEM(kds_src, src_index);
			rc = gpuscan_quals_eval(&kcxt, kds_src,
									&tupitem->t_self,
									&tupitem->htup);
		}
		else
		{
			tupitem = NULL;
			rc = false;
		}
#ifdef GPUSCAN_HAS_WHERE_QUALS
		/* bailout if any error */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			goto out_nostat;
#endif
		/* how many rows servived WHERE-clause evaluation? */
		nitems_offset = pgstromStairlikeBinaryCount(tupitem && rc, &nvalids);
		if (nvalids == 0)
			goto skip;

#ifdef GPUSCAN_HAS_DEVICE_PROJECTION
		/* extract the source tuple to the private slot, if any */
		if (tupitem && rc)
		{
			gpuscan_projection_tuple(&kcxt,
									 kds_src,
									 &tupitem->htup,
									 &tupitem->t_self,
									 tup_values,
									 tup_isnull,
									 tup_extra);
			required = MAXALIGN(offsetof(kern_tupitem, htup) +
								compute_heaptuple_size(&kcxt,
													   kds_dst,
													   tup_values,
													   tup_isnull));
		}
		else
			required = 0;
		/* bailout if any error */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			goto out_nostat;

		usage_offset = pgstromStairlikeSum(required, &extra_sz);
		if (get_local_id() == 0)
		{
			union {
				struct {
					cl_uint	nitems;
					cl_uint	usage;
				} i;
				cl_ulong	v64;
			} oldval, curval, newval;

			curval.i.nitems	= kds_dst->nitems;
			curval.i.usage	= kds_dst->usage;
			do {
				newval = oldval = curval;
				newval.i.nitems += nvalids;
				newval.i.usage  += __kds_packed(extra_sz);

				if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
					STROMALIGN(sizeof(cl_uint) * newval.i.nitems) +
					__kds_unpack(newval.i.usage) > kds_dst->length)
				{
					atomicAdd(&kgpuscan->suspend_count, 1);
					suspend_kernel = 1;
					break;
				}
			} while ((curval.v64 = atomicCAS((cl_ulong *)&kds_dst->nitems,
											 oldval.v64,
											 newval.v64)) != oldval.v64);
			nitems_base = oldval.i.nitems;
			usage_base  = __kds_unpack(oldval.i.usage);
		}
		if (__syncthreads_count(suspend_kernel) > 0)
			break;

		/* store the result heap-tuple on destination buffer */
		if (tupitem && rc)
		{
			cl_uint	   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
			cl_uint		pos;
			cl_uint		htuple_oid = 0;

			if (kds_dst->tdhasoid)
			{
				htuple_oid = kern_getsysatt_oid(&tupitem->htup);
				if (htuple_oid == 0)
					htuple_oid = 0xffffffff;
			}
			pos = kds_dst->length - (usage_base + usage_offset + required);
			tup_index[nitems_base + nitems_offset] = __kds_packed(pos);
			form_kern_heaptuple((kern_tupitem *)((char *)kds_dst + pos),
								kds_dst->ncols,
								kds_dst->colmeta,
								&tupitem->t_self,
								&tupitem->htup.t_choice.t_heap,
								htuple_oid,
								tup_values,
								tup_isnull);
		}
#else
		if (get_local_id() == 0)
			nitems_base = atomicAdd(&gs_results->nitems, nvalids);
		__syncthreads();
		if (tupitem && rc)
		{
			assert(nitems_base + nitems_offset < kds_src->nrooms);
			gs_results->results[nitems_base + nitems_offset]
				= __kds_packed((char *)&tupitem->htup -
							   (char *)kds_src);
		}
#endif
	skip:
		/* update statistics */
		if (get_local_id() == 0)
		{
			total_nitems_in  += Min(kds_src->nitems - src_base,
									get_local_size());
			total_nitems_out += nvalids;
			total_extra_size += extra_sz;
		}
	}
	/* write back statistics and error code */
	if (get_local_id() == 0)
	{
		atomicAdd(&kgpuscan->nitems_in,  total_nitems_in);
		atomicAdd(&kgpuscan->nitems_out, total_nitems_out);
		atomicAdd(&kgpuscan->extra_size, total_extra_size);
	}
out_nostat:
	/* suspend the current position (even if normal exit) */
	if (my_suspend && get_local_id() == 0)
	{
		my_suspend->part_index = part_index;
		my_suspend->line_index = 0;
	}
	kern_writeback_error_status(&kgpuscan->kerror, &kcxt.e);
}
#endif	/* GPUSCAN_KERNEL_REQUIRED */

#ifdef GPUSCAN_KERNEL_REQUIRED
/*
 * gpuscan_exec_quals_block - GpuScan logic for KDS_FORMAT_BLOCK
 */
KERNEL_FUNCTION(void)
gpuscan_exec_quals_block(kern_gpuscan *kgpuscan,
						 kern_data_store *kds_src,
						 kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	gpuscanSuspendContext *my_suspend;
	kern_context	kcxt;
	cl_uint			src_nitems = kds_src->nitems;
	cl_uint			part_sz;
	cl_uint			n_parts;
	cl_uint			nitems_offset;
	cl_uint			usage_offset;
	cl_uint			window_sz;
	cl_uint			part_base;
	cl_uint			part_index = 0;
	cl_uint			line_index = 0;
	cl_uint			total_nitems_in = 0;	/* stat */
	cl_uint			total_nitems_out = 0;	/* stat */
	cl_uint			total_extra_size = 0;	/* stat */
	cl_bool			thread_is_valid = false;
#ifdef GPUSCAN_HAS_DEVICE_PROJECTION
#if GPUSCAN_DEVICE_PROJECTION_NFIELDS > 0
	Datum			tup_values[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_isnull[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
#else
	Datum		   *tup_values = NULL;
	cl_bool		   *tup_isnull = NULL;
#endif
#if GPUSCAN_DEVICE_PROJECTION_EXTRA_SIZE > 0
	char			tup_extra[GPUSCAN_DEVICE_PROJECTION_EXTRA_SIZE]
					__attribute__ ((aligned(MAXIMUM_ALIGNOF)));
#else
	char		   *tup_extra = NULL;
#endif
#endif
	__shared__ cl_uint	nitems_base;
	__shared__ cl_ulong	usage_base;

	assert(kds_src->format == KDS_FORMAT_BLOCK);
	assert(kds_dst->format == KDS_FORMAT_ROW);
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_exec_quals_block, kparams);

	part_sz = KERN_DATA_STORE_PARTSZ(kds_src);
	n_parts = get_local_size() / part_sz;
	if (get_global_id() == 0)
		kgpuscan->part_sz = part_sz;
	if (get_local_id() < part_sz * n_parts)
		thread_is_valid = true;
	window_sz = n_parts * get_num_groups();

	/* resume kernel from the point where suspended, if any */
	my_suspend = KERN_GPUSCAN_SUSPEND_CONTEXT(kgpuscan, get_group_id());
	if (kgpuscan->resume_context)
	{
		part_index = my_suspend->part_index;
		line_index = my_suspend->line_index;
	}
	__syncthreads();

	for (;;)
	{
		cl_uint		part_id;
		cl_uint		line_no;
		cl_uint		n_lines;
		cl_uint		nvalids;
		cl_uint		nitems_real;

		part_base = part_index * window_sz + get_group_id() * n_parts;
		if (part_base >= kds_src->nitems)
			break;
		part_id = get_local_id() / part_sz + part_base;
		line_no = get_local_id() % part_sz + line_index * part_sz;

		do {
			HeapTupleHeaderData *htup = NULL;
			ItemPointerData t_self;
			PageHeaderData *pg_page;
			BlockNumber	block_nr;
			cl_ushort	t_len	__attribute__((unused));
			cl_uint		required;
			cl_uint		extra_sz = 0;
			cl_uint		suspend_kernel = 0;
			cl_bool		rc;

			/* identify the block */
			if (thread_is_valid && part_id < src_nitems)
			{
				pg_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds_src, part_id);
				n_lines = PageGetMaxOffsetNumber(pg_page);
				block_nr = KERN_DATA_STORE_BLOCK_BLCKNR(kds_src, part_id);
				t_self.ip_blkid.bi_hi = block_nr >> 16;
				t_self.ip_blkid.bi_lo = block_nr & 0xffff;
				t_self.ip_posid = line_no + 1;

				if (line_no < n_lines)
				{
					ItemIdData *lpp = PageGetItemId(pg_page, line_no+1);
					if (ItemIdIsNormal(lpp))
						htup = PageGetItem(pg_page, lpp);
					t_len = ItemIdGetLength(lpp);
				}
			}

			/* evaluation of the qualifiers */
#ifdef GPUSCAN_HAS_WHERE_QUALS
			if (htup)
				rc = gpuscan_quals_eval(&kcxt, kds_src,
										&t_self,
										htup);
			else
				rc = false;
			/* bailout if any error */
			if (__syncthreads_count(kcxt.e.errcode) > 0)
				goto out_nostat;
#else
			rc = true;
#endif
			/* how many rows servived WHERE-clause evaluations? */
			nitems_offset = pgstromStairlikeBinaryCount(htup && rc, &nvalids);
			if (nvalids == 0)
				goto skip;

			/* store the result heap-tuple to destination buffer */
			if (htup && rc)
			{
#ifdef GPUSCAN_HAS_DEVICE_PROJECTION
				gpuscan_projection_tuple(&kcxt,
										 kds_src,
										 htup,
										 &t_self,
										 tup_values,
										 tup_isnull,
										 tup_extra);
				required = MAXALIGN(offsetof(kern_tupitem, htup) +
									compute_heaptuple_size(&kcxt,
														   kds_dst,
														   tup_values,
														   tup_isnull));
#else
				/* no projection; just write the source tuple as is */
				required = MAXALIGN(offsetof(kern_tupitem, htup) + t_len);
#endif
			}
			else
				required = 0;

			usage_offset = pgstromStairlikeSum(required, &extra_sz);
			if (get_local_id() == 0)
			{
				union {
					struct {
						cl_uint	nitems;
						cl_uint	usage;
					} i;
					cl_ulong	v64;
				} oldval, curval, newval;

				curval.i.nitems = kds_dst->nitems;
				curval.i.usage  = kds_dst->usage;
				do {
					newval = oldval = curval;
					newval.i.nitems += nvalids;
					newval.i.usage  += __kds_packed(extra_sz);

					if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
						STROMALIGN(sizeof(cl_uint) * newval.i.nitems) +
						__kds_unpack(newval.i.usage) > kds_dst->length)
					{
						atomicAdd(&kgpuscan->suspend_count, 1);
						suspend_kernel = 1;
						break;
					}
				} while ((curval.v64 = atomicCAS((cl_ulong *)&kds_dst->nitems,
												 oldval.v64,
												 newval.v64)) != oldval.v64);
				nitems_base = oldval.i.nitems;
				usage_base  = __kds_unpack(oldval.i.usage);
			}
			if (__syncthreads_count(suspend_kernel) > 0)
				goto out;

			/* store the result heap tuple */
			if (htup && rc)
			{
				cl_uint	   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
				cl_ulong	pos =
					(kds_dst->length - (usage_base + usage_offset + required));
#ifdef GPUSCAN_HAS_DEVICE_PROJECTION
				cl_uint		htuple_oid = 0;

				if (kds_dst->tdhasoid)
				{
					htuple_oid = kern_getsysatt_oid(htup);
					if (htuple_oid == 0)
						htuple_oid = 0xffffffff;
				}
				form_kern_heaptuple((kern_tupitem *)((char *)kds_dst + pos),
									kds_dst->ncols,
									kds_dst->colmeta,
									&t_self,
									&htup->t_choice.t_heap,
									htuple_oid,
									tup_values,
									tup_isnull);
#else
				kern_tupitem *tupitem;

				tupitem = (kern_tupitem *)((char *)kds_dst + pos);
				tupitem->t_len = t_len;
				tupitem->t_self = t_self;
				memcpy(&tupitem->htup, htup, t_len);
#endif
				tup_index[nitems_base + nitems_offset] = __kds_packed(pos);
			}
		skip:
			/* update statistics */
			pgstromStairlikeBinaryCount(htup != NULL, &nitems_real);
			if (get_local_id() == 0)
			{
				total_nitems_in		+= nitems_real;
				total_nitems_out	+= nvalids;
				total_extra_size	+= extra_sz;
			}

			/*
			 * Move to the next window of the line items, if any.
			 * If no threads in CUDA block wants to continue, exit the loop.
			 */
			line_index++;
			line_no += part_sz;
		} while (__syncthreads_count(thread_is_valid &&
									 line_no < n_lines) > 0);
		/* move to the next window */
		part_index++;
		line_index = 0;
	}
out:
	/* update statistics */
	if (get_local_id() == 0)
	{
		atomicAdd(&kgpuscan->nitems_in,  total_nitems_in);
		atomicAdd(&kgpuscan->nitems_out, total_nitems_out);
		atomicAdd(&kgpuscan->extra_size, total_extra_size);
	}
out_nostat:
	if (get_local_id() == 0)
	{
		my_suspend->part_index = part_index;
		my_suspend->line_index = line_index;
	}
	/* write back error code to the host */
	kern_writeback_error_status(&kgpuscan->kerror, &kcxt.e);
}
#endif	/* GPUSCAN_KERNEL_REQUIRED */

#ifdef GPUSCAN_KERNEL_REQUIRED
/*
 * gpuscan_exec_quals_column - GpuScan logic for KDS_FORMAT_COLUMN
 */
KERNEL_FUNCTION(void)
gpuscan_exec_quals_column(kern_gpuscan *kgpuscan,
						  kern_data_store *kds_src,
						  kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	gpuscanSuspendContext *my_suspend
		= KERN_GPUSCAN_SUSPEND_CONTEXT(kgpuscan, get_group_id());
	kern_context	kcxt;
	cl_uint			part_index = 0;
	cl_uint			src_base;
	cl_uint			src_index;
	cl_uint			nitems_offset;
	cl_uint			usage_offset	__attribute__((unused));
	cl_uint			total_nitems_in = 0;	/* stat */
	cl_uint			total_nitems_out = 0;	/* stat */
	cl_uint			total_extra_size = 0;	/* stat */
#if GPUSCAN_DEVICE_PROJECTION_NFIELDS > 0
	Datum			tup_values[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_isnull[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
#else
	Datum		   *tup_values = NULL;
	cl_bool		   *tup_isnull = NULL;
#endif
#if GPUSCAN_DEVICE_PROJECTION_EXTRA_SIZE > 0
	cl_char			tup_extra[GPUSCAN_DEVICE_PROJECTION_EXTRA_SIZE]
					__attribute__ ((aligned(MAXIMUM_ALIGNOF)));
#else
	cl_char		   *tup_extra __attribute__((unused)) = NULL;
#endif
	__shared__ cl_uint	nitems_base;
	__shared__ cl_ulong	usage_base	__attribute__((unused));

	assert(__ldg(&kds_src->format) == KDS_FORMAT_COLUMN);
	assert(!kds_dst || __ldg(&kds_dst->format) == KDS_FORMAT_ROW);
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_exec_quals_column, kparams);
	/* quick bailout if any error happen on the prior kernel */
	if (__syncthreads_count(kgpuscan->kerror.errcode) != 0)
		return;
	/* resume kernel from the point where suspended, if any */
	if (kgpuscan->resume_context)
	{
		assert(my_suspend != NULL);
		part_index = my_suspend->part_index;
	}

	for (src_base = get_global_base() + part_index * get_global_size();
		 src_base < kds_src->nitems;
		 src_base += get_global_size(), part_index++)
	{
		kern_tupitem   *tupitem		__attribute__((unused));
		cl_bool			rc;
		cl_uint			nvalids;
		cl_uint			required	__attribute__((unused));
		cl_uint			extra_sz = 0;
		cl_uint			suspend_kernel = 0;

		/* Evalidation of the rows by WHERE-clause */
		src_index = src_base + get_local_id();
		if (src_index < kds_src->nitems)
			rc = gpuscan_quals_eval_column(&kcxt, kds_src, src_index);
		else
			rc = false;
#ifdef GPUSCAN_HAS_WHERE_QUALS
		/* bailout if any error */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			goto out_nostat;
#endif
		/* how many rows servived WHERE-clause evaluation? */
		nitems_offset = pgstromStairlikeBinaryCount(rc, &nvalids);
		if (nvalids == 0)
			goto skip;

		/*
		 * OK, extract the source columns to form a result row
		 */
		if (rc)
		{
			gpuscan_projection_column(&kcxt,
									  kds_src,
									  src_index,
									  tup_values,
									  tup_isnull,
									  tup_extra);
			required = MAXALIGN(offsetof(kern_tupitem, htup) +
								compute_heaptuple_size(&kcxt,
													   kds_dst,
													   tup_values,
													   tup_isnull));
		}
		else
			required = 0;

		usage_offset = pgstromStairlikeSum(required, &extra_sz);
		if (get_local_id() == 0)
		{
			union {
				struct {
					cl_uint	nitems;
					cl_uint	usage;
				} i;
				cl_ulong	v64;
			} oldval, curval, newval;

			curval.i.nitems = kds_dst->nitems;
			curval.i.usage  = kds_dst->usage;
			do {
				newval = oldval = curval;
				newval.i.nitems += nvalids;
				newval.i.usage  += __kds_packed(extra_sz);

				if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
					STROMALIGN(sizeof(cl_uint) * newval.i.nitems) +
					__kds_unpack(newval.i.usage) > kds_dst->length)
				{
					atomicAdd(&kgpuscan->suspend_count, 1);
					suspend_kernel = 1;
					break;
				}
			} while ((curval.v64 = atomicCAS((cl_ulong *)&kds_dst->nitems,
											 oldval.v64,
											 newval.v64)) != oldval.v64);
			nitems_base = oldval.i.nitems;
			usage_base  = __kds_unpack(oldval.i.usage);
		}
		if (__syncthreads_count(suspend_kernel) > 0)
			break;

		/* store the result heap-tuple on destination buffer */
		if (required > 0)
		{
			cl_uint	   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
			cl_uint		pos;

			pos = kds_dst->length - (usage_base + usage_offset + required);
			tup_index[nitems_base + nitems_offset] = __kds_packed(pos);
			form_kern_heaptuple((kern_tupitem *)((char *)kds_dst + pos),
								kds_dst->ncols,
								kds_dst->colmeta,
								NULL,	/* ItemPointerData */
								NULL,	/* HeapTupleFields */
								kds_dst->tdhasoid ? 0xffffffff : 0,
								tup_values,
								tup_isnull);
		}
		/* bailout if any error */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			break;
	skip:
		/* update statistics */
		if (get_local_id() == 0)
		{
			total_nitems_in  += Min(kds_src->nitems - src_base,
									get_local_size());
			total_nitems_out += nvalids;
			total_extra_size += extra_sz;
		}
	}
	/* write back statistics and error code */
	if (get_local_id() == 0)
	{
		atomicAdd(&kgpuscan->nitems_in,  total_nitems_in);
		atomicAdd(&kgpuscan->nitems_out, total_nitems_out);
		atomicAdd(&kgpuscan->extra_size, total_extra_size);
	}
out_nostat:
	/* suspend the current position (even if normal exit) */
	if (my_suspend && get_local_id() == 0)
	{
		my_suspend->part_index = part_index;
		my_suspend->line_index = 0;
	}
	kern_writeback_error_status(&kgpuscan->kerror, &kcxt.e);
}
#endif	/* GPUSCAN_KERNEL_REQUIRED */
#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUSCAN_H */
