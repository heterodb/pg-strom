/*
 * cuda_gpuscan.h
 *
 * CUDA device code specific to GpuScan logic
 * --
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
 * +-+---------------+ ---
 */
struct kern_gpuscan {
	kern_errorbuf	kerror;
	cl_uint			read_src_pos;
	cl_uint			nitems_in;
	cl_uint			nitems_out;
	cl_uint			extra_size;
	/* performance profile */
	struct {
		cl_float	tv_kern_exec_quals;
		cl_float	tv_kern_projection;
	} pfm;
	kern_parambuf	kparams;
};

typedef struct kern_gpuscan		kern_gpuscan;

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
#define KERN_GPUSCAN_DMASEND_LENGTH(kgpuscan)	\
	(offsetof(kern_gpuscan, kparams) +			\
	 KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan) +	\
	 offsetof(kern_resultbuf, results[0]))
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
STATIC_FUNCTION(cl_bool)
gpuscan_quals_eval_column(kern_context *kcxt,
						  kern_data_store *kds,
						  cl_uint src_index)
{
	return false;
}

STATIC_FUNCTION(void)
gpuscan_projection(kern_context *kcxt,
				   kern_data_store *kds_src,
				   HeapTupleHeaderData *htup,
				   ItemPointerData *t_self,
				   Datum *tup_values,
				   cl_bool *tup_isnull,
				   cl_bool *tup_internal);

STATIC_FUNCTION(void)
gpuscan_projection_column(kern_context *kcxt,
						  kern_data_store *kds_src,
						  size_t src_index,
						  ItemPointerData *t_self,
						  HeapTupleFields *tx_attrs,
						  Datum *tup_values,
						  cl_bool *tup_isnull,
						  cl_char *tup_extra)
{}

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
	kern_resultbuf *kresults		__attribute__((unused))
		= KERN_GPUSCAN_RESULTBUF(kgpuscan);
	kern_context	kcxt;
	cl_uint			src_index;
	cl_uint			src_nitems = kds_src->nitems;
	cl_bool			try_next_window = true;
	cl_uint			nitems_offset;
	cl_uint			usage_offset	__attribute__((unused));
	cl_uint			total_nitems_out = 0;	/* stat */
	cl_uint			total_extra_size = 0;	/* stat */
#ifdef GPUSCAN_DEVICE_PROJECTION
	Datum			tup_values[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_isnull[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_internal[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
#endif
	__shared__ cl_int	src_base;
	__shared__ cl_int	nitems_base;
	__shared__ cl_int	usage_base	__attribute__((unused));
	__shared__ cl_int	status __attribute__((unused));

	assert(kds_src->format == KDS_FORMAT_ROW);
	assert(!kds_dst || kds_dst->format == KDS_FORMAT_ROW);
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_exec_quals_row, kparams);
	if (get_local_id() == 0)
		status = StromError_Success;
	__syncthreads();

	do {
		kern_tupitem   *tupitem;
		cl_bool			rc;
		cl_uint			nvalids;
		cl_uint			required	__attribute__((unused));
		cl_uint			extra_sz = 0;

		if (get_local_id() == 0)
			src_base = atomicAdd(&kgpuscan->read_src_pos, get_local_size());
		__syncthreads();

		if (src_base + get_local_size() >= src_nitems)
			try_next_window = false;
		if (src_base >= src_nitems)
			break;

		/* Evalidation of the rows by WHERE-clause */
		src_index = src_base + get_local_id();
		if (src_index < src_nitems)
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
		if (kcxt.e.errcode != StromError_Success)
			atomicCAS(&status, StromError_Success, kcxt.e.errcode);
		__syncthreads();
		if (status != StromError_Success)
			break;
#endif
		/* how many rows servived WHERE-clause evaluation? */
		nitems_offset = pgstromStairlikeBinaryCount(tupitem && rc, &nvalids);
		if (nvalids == 0)
			goto skip;

#ifdef GPUSCAN_DEVICE_PROJECTION
		if (get_local_id() == 0)
			nitems_base = atomicAdd(&kds_dst->nitems, nvalids);
		__syncthreads();

		/* extract the source tuple to the private slot, if any */
		if (tupitem && rc)
		{
			gpuscan_projection(&kcxt,
							   kds_src,
							   &tupitem->htup,
							   &tupitem->t_self,
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
			required = 0;

		usage_offset = pgstromStairlikeSum(required, &extra_sz);
		if (get_local_id() == 0)
			usage_base = atomicAdd(&kds_dst->usage, extra_sz);
		__syncthreads();

		if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
			STROMALIGN(sizeof(cl_uint) * (nitems_base + nvalids)) +
			usage_base + extra_sz > kds_dst->length)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			break;
		}

		/* store the result heap-tuple on destination buffer */
		if (tupitem && rc)
		{
			cl_uint	   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
			cl_uint		pos;

			pos = kds_dst->length - (usage_base + usage_offset + required);
			tup_index[nitems_base + nitems_offset] = pos;
			form_kern_heaptuple(&kcxt, kds_dst,
								(kern_tupitem *)((char *)kds_dst + pos),
								&tupitem->t_self,
								NULL,
								tup_values,
								tup_isnull,
								tup_internal);
		}

		/* bailout if any error */
		if (kcxt.e.errcode != StromError_Success)
			atomicCAS(&status, StromError_Success, kcxt.e.errcode);
		__syncthreads();
		if (status != StromError_Success)
			break;
#else
		if (get_local_id() == 0)
			nitems_base = atomicAdd(&kresults->nitems, nvalids);
		__syncthreads();

		if (nitems_base + nvalids >= kresults->nrooms)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			break;
		}

		/* OK, store the result index */
		if (tupitem && rc)
		{
			kresults->results[nitems_base + nitems_offset] = (cl_uint)
				((char *)(&tupitem->htup) - (char *)(kds_src));
		}
#endif
	skip:
		/* update statistics */
		if (get_local_id() == 0)
		{
			total_nitems_out += nvalids;
			total_extra_size += extra_sz;
		}
	} while (try_next_window);

	/* write back error code and statistics to the host */
	if (get_local_id() == 0)
	{
		atomicAdd(&kgpuscan->nitems_in,  kds_src->nitems);
		atomicAdd(&kgpuscan->nitems_out, total_nitems_out);
		atomicAdd(&kgpuscan->extra_size, total_extra_size);
	}
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
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
	kern_context	kcxt;
	cl_uint			src_nitems = kds_src->nitems;
	cl_uint			part_sz;
	cl_uint			n_parts;
	cl_uint			nitems_offset;
	cl_uint			usage_offset;
	cl_uint			total_nitems_in = 0;	/* stat */
	cl_uint			total_nitems_out = 0;	/* stat */
	cl_uint			total_extra_size = 0;	/* stat */
	cl_bool			try_next_window = true;
	__shared__ cl_uint	base;
	__shared__ cl_uint	nitems_base;
	__shared__ cl_uint	usage_base;
	__shared__ cl_int	status __attribute__((unused));
	__shared__ cl_int	gang_sync;

	assert(kds_src->format == KDS_FORMAT_BLOCK);
	assert(kds_dst->format == KDS_FORMAT_ROW);
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_exec_quals_block, kparams);
	if (get_local_id() == 0)
		status = StromError_Success;
	__syncthreads();

	part_sz = KERN_DATA_STORE_PARTSZ(kds_src);
	n_parts = get_local_size() / part_sz;
	do {
		cl_uint		part_id;
		cl_uint		line_no;
		cl_uint		n_lines;
		cl_uint		nvalids;
		cl_uint		nitems_real;

		if (get_local_id() == 0)
			base =  atomicAdd(&kgpuscan->read_src_pos, n_parts);
		__syncthreads();

		if (base + n_parts >= __ldg(&kds_src->nitems))
			try_next_window = false;
		if (base >= __ldg(&kds_src->nitems))
			break;

		part_id = base + get_local_id() / part_sz;
		line_no = get_local_id() % part_sz;

		do {
			HeapTupleHeaderData *htup = NULL;
			ItemPointerData t_self;
			PageHeaderData *pg_page;
			BlockNumber	block_nr;
			cl_ushort	t_len;
			cl_uint		required;
			cl_uint		extra_sz = 0;
			cl_bool		rc;

			/* identify the block */
			if (part_id < src_nitems)
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
			if (kcxt.e.errcode != StromError_Success)
				atomicCAS(&status, StromError_Success, kcxt.e.errcode);
			__syncthreads();
			if (status != StromError_Success)
				break;
#else
			rc = true;
#endif
			/* how many rows servived WHERE-clause evaluations? */
			nitems_offset = pgstromStairlikeBinaryCount(htup && rc, &nvalids);
			if (nvalids == 0)
				goto skip;

			/* store the result heap-tuple to destination buffer */
#ifdef GPUSCAN_DEVICE_PROJECTION_COMMON
			if (htup && rc)
			{
				gpuscan_projection(&kcxt,
								   kds_src,
								   htup,
								   &t_self,
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
				required = 0;

			usage_offset = pgstromStairlikeSum(required, &extra_sz);
			if (get_local_id() == 0)
				usage_base = atomicAdd(&kds_dst->usage, extra_sz);
			__syncthreads();

			if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
				STROMALIGN(sizeof(cl_uint) * (nitems_base + nvalids)) +
				usage_base + extra_sz > kds_dst->length)
			{
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
				break;
			}

			/* store the result heap tuple */
			if (tupitem && rc)
			{
				cl_uint	   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
				cl_uint		pos;

				pos = kds_dst->length - (usage_base + usage_offset + required);
				tup_index[nitems_base + nitems_offset] = pos;
				form_kern_heaptuple(&kcxt, kds_dst,
									(kern_tupitem *)((char *)kds_dst + pos),
									&t_self,
									NULL,
									tup_values,
									tup_isnull,
									tup_internal);
			}
#else
			/* no projection - write back souce tuple as is */
			if (get_local_id() == 0)
				nitems_base = atomicAdd(&kds_dst->nitems, nvalids);
			__syncthreads();

			if (htup && rc)
				required = MAXALIGN(offsetof(kern_tupitem, htup) + t_len);
			else
				required = 0;
			
			usage_offset = pgstromStairlikeSum(required, &extra_sz);
			if (get_local_id() == 0)
				usage_base = atomicAdd(&kds_dst->usage, extra_sz);
			__syncthreads();

			if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
				STROMALIGN(sizeof(cl_uint) * (nitems_base + nvalids)) +
				usage_base + extra_sz > kds_dst->length)
			{
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
				break;
			}

			if (htup && rc)
			{
				kern_tupitem *tupitem;
				cl_uint	   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
				cl_uint		pos = kds_dst->length - (usage_base +
													 usage_offset + required);
				tupitem = (kern_tupitem *)((char *)kds_dst + pos);
				tupitem->t_len = t_len;
				tupitem->t_self = t_self;
				memcpy(&tupitem->htup, htup, t_len);
				tup_index[nitems_base + nitems_offset] = pos;
			}
#endif
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
			line_no += part_sz;
			if (get_local_id() == 0)
				gang_sync = 0;
			__syncthreads();
			if (get_local_id() % part_sz == 0 && line_no < n_lines)
				gang_sync = 1;
			__syncthreads();
		} while (gang_sync > 0);
	} while (try_next_window);

	/* update statistics */
	if (get_local_id() == 0)
	{
		atomicAdd(&kgpuscan->nitems_in,  total_nitems_in);
		atomicAdd(&kgpuscan->nitems_out, total_nitems_out);
		atomicAdd(&kgpuscan->extra_size, total_extra_size);
	}
	/* write back error code to the host */
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
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
	kern_context	kcxt;
	cl_uint			src_index;
	cl_uint			src_nitems = kds_src->nitems;
	cl_bool			try_next_window = true;
	cl_uint			nitems_offset;
	cl_uint			usage_offset	__attribute__((unused));
	cl_uint			total_nitems_out = 0;	/* stat */
	cl_uint			total_extra_size = 0;	/* stat */
	Datum			tup_values[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_isnull[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
#if GPUSCAN_DEVICE_PROJECTION_EXTRA_SIZE > 0
	cl_char			tup_extra[GPUSCAN_DEVICE_PROJECTION_EXTRA_SIZE];
#else
	cl_char		   *tup_extra = NULL;
#endif
	__shared__ cl_int	src_base;
	__shared__ cl_int	nitems_base;
	__shared__ cl_int	usage_base	__attribute__((unused));
	__shared__ cl_int	status __attribute__((unused));

	assert(__ldg(&kds_src->format) == KDS_FORMAT_COLUMN);
	assert(!kds_dst || __ldg(&kds_dst->format) == KDS_FORMAT_ROW);
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_exec_quals_column, kparams);
	if (get_local_id() == 0)
		status = StromError_Success;
	__syncthreads();

	do {
		HeapTupleFields	tx_attrs;	/* xmin, xmax, cmin/cmax, if any */
		ItemPointerData	t_self;		/* t_self, if any */
		kern_tupitem   *tupitem;
		cl_bool			rc;
		cl_uint			nvalids;
		cl_uint			required	__attribute__((unused));
		cl_uint			extra_sz = 0;

		if (get_local_id() == 0)
			src_base = atomicAdd(&kgpuscan->read_src_pos, get_local_size());
		__syncthreads();

		if (src_base + get_local_size() >= src_nitems)
			try_next_window = false;
		if (src_base >= src_nitems)
			break;

		/* Evalidation of the rows by WHERE-clause */
		src_index = src_base + get_local_id();
		if (src_index < src_nitems)
			rc = gpuscan_quals_eval_column(&kcxt, kds_src, src_index);
		else
			rc = false;
#ifdef GPUSCAN_HAS_WHERE_QUALS
		/* bailout if any error */
		if (kcxt.e.errcode != StromError_Success)
			atomicCAS(&status, StromError_Success, kcxt.e.errcode);
		__syncthreads();
		if (status != StromError_Success)
			break;
#endif
		/* how many rows servived WHERE-clause evaluation? */
		nitems_offset = pgstromStairlikeBinaryCount(rc, &nvalids);
		if (nvalids == 0)
			goto skip;

		/*
		 * OK, extract the source columns to form a result row
		 */
		if (get_local_id() == 0)
			nitems_base = atomicAdd(&kds_dst->nitems, nvalids);
		__syncthreads();

		if (rc)
		{
			gpuscan_projection_column(&kcxt,
									  kds_src,
									  src_index,
									  &t_self,
									  &tx_attrs,
									  tup_values,
									  tup_isnull,
									  tup_extra);
			required = MAXALIGN(offsetof(kern_tupitem, htup) +
								compute_heaptuple_size(&kcxt,
													   kds_dst,
													   tup_values,
													   tup_isnull,
													   NULL));
		}
		else
			required = 0;

		usage_offset = pgstromStairlikeSum(required, &extra_sz);
		if (get_local_id() == 0)
			usage_base = atomicAdd(&kds_dst->usage, extra_sz);
		__syncthreads();

		if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
			STROMALIGN(sizeof(cl_uint) * (nitems_base + nvalids)) +
			usage_base + extra_sz > kds_dst->length)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			break;
		}

		/* store the result heap-tuple on destination buffer */
		if (required > 0)
		{
			cl_uint	   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
			cl_uint		pos;

			pos = kds_dst->length - (usage_base + usage_offset + required);
			tup_index[nitems_base + nitems_offset] = pos;
			form_kern_heaptuple(&kcxt,
								kds_dst,
								(kern_tupitem *)((char *)kds_dst + pos),
								&t_self,
								&tx_attrs,
								tup_values,
								tup_isnull,
								NULL);
		}

		/* bailout if any error */
		if (kcxt.e.errcode != StromError_Success)
			atomicCAS(&status, StromError_Success, kcxt.e.errcode);
		__syncthreads();
		if (status != StromError_Success)
			break;
	skip:
		/* update statistics */
		if (get_local_id() == 0)
		{
			total_nitems_out += nvalids;
			total_extra_size += extra_sz;
		}
	} while (try_next_window);

	/* write back error code and statistics to the host */
	if (get_local_id() == 0)
	{
		atomicAdd(&kgpuscan->nitems_in,  kds_src->nitems);
		atomicAdd(&kgpuscan->nitems_out, total_nitems_out);
		atomicAdd(&kgpuscan->extra_size, total_extra_size);
	}
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
}
#endif	/* GPUSCAN_KERNEL_REQUIRED */
#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUSCAN_H */
