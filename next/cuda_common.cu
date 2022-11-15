/*
 * cuda_common.cu
 *
 * Core implementation of GPU device code
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"

/*
 * __execProjectionCommon
 */
STATIC_FUNCTION(int)
__execProjectionCommon(kern_context *kcxt,
					   kern_expression *kexp,
					   kern_data_store *kds_dst,
					   uint32_t tupsz)
{
	size_t		offset;
	uint32_t	mask;
	uint32_t	nvalids;
	uint32_t	row_id;
	uint32_t	total_sz;
	bool		try_suspend = false;
	union {
		struct {
			uint32_t	nitems;
			uint32_t	usage;
		} i;
		uint64_t		v64;
	} oldval, curval, newval;

	/* allocation of the destination buffer */
	assert(kds_dst->format == KDS_FORMAT_ROW);
	mask = __ballot_sync(__activemask(), tupsz > 0);
	nvalids = __popc(mask);
	mask &= ((1U << LaneId()) - 1);
	row_id  = __popc(mask);
	assert(tupsz == 0 || row_id < nvalids);

	offset = __reduce_stair_add_sync(tupsz, &total_sz);
	if (LaneId() == 0)
	{
		curval.i.nitems = kds_dst->nitems;
		curval.i.usage  = kds_dst->usage;
		do {
			newval = oldval = curval;
			newval.i.nitems += nvalids;
			newval.i.usage  += __kds_packed(total_sz);

			if (KDS_HEAD_LENGTH(kds_dst) +
				MAXALIGN(sizeof(uint32_t) * newval.i.nitems) +
				__kds_unpack(newval.i.usage) > kds_dst->length)
			{
				try_suspend = true;
				break;
			}
		} while ((curval.v64 = atomicCAS((unsigned long long *)&kds_dst->nitems,
										 oldval.v64,
										 newval.v64)) != oldval.v64);
	}
	oldval.v64 = __shfl_sync(__activemask(), oldval.v64, 0);
	row_id += oldval.i.nitems;
	/* data store no space? */
	if (__any_sync(__activemask(), try_suspend))
		return 0;

	/* write out the tuple */
	if (tupsz > 0)
	{
		kern_expression *kproj = KEXP_FIRST_ARG(kexp);
		kern_tupitem *tupitem;

		assert(KEXP_IS_VALID(kproj, int4) &&
			   kproj->opcode == FuncOpCode__Projection);
		offset += __kds_unpack(oldval.i.usage);
		KDS_GET_ROWINDEX(kds_dst)[row_id] = __kds_packed(offset);
		tupitem = (kern_tupitem *)
			((char *)kds_dst + kds_dst->length - offset);
		tupitem->rowid = row_id;
		tupitem->t_len = kern_form_heaptuple(kcxt, kproj, kds_dst, &tupitem->htup);
	}
	return total_sz;
}

/*
 * ExecKernProjection
 */
PUBLIC_FUNCTION(int)
ExecProjectionOuterRow(kern_context *kcxt,
					   kern_expression *kexp,	/* LoadVars + Projection */
					   kern_data_store *kds_dst,
					   kern_data_store *kds_outer,
					   HeapTupleHeaderData *htup_outer,
					   int num_inners,
					   kern_data_store **kds_inners,
					   HeapTupleHeaderData **htup_inners)
{
	uint32_t	tupsz = 0;

	/*
	 * First, extract the variables from outer/inner tuples, and
	 * calculate expressions, if any.
	 */
	assert(__activemask() == 0xffffffffU);
	assert(kexp->opcode == FuncOpCode__LoadVars &&
		   kexp->exptype == TypeOpCode__int4);
	if (htup_outer != NULL)
	{
		xpu_int4_t	__tupsz;

		if (!ExecLoadVarsOuterRow(kcxt,
								  kexp,
								  (xpu_datum_t *)&__tupsz,
								  kds_outer,
								  htup_outer,
								  num_inners,
								  kds_inners,
								  htup_inners))
		{
			/* something wrong */
			assert(kcxt->errcode != 0);
		}
		else if (__tupsz.isnull || __tupsz.value <= 0)
		{
			STROM_ELOG(kcxt, "wrong calculation of tuple length");
		}
		else
		{
			tupsz = MAXALIGN(__tupsz.value);
		}
	}
	/* error checks */
	if (__ballot_sync(__activemask(), kcxt->errcode != 0) != 0)
		return -1;
	return __execProjectionCommon(kcxt, kexp, kds_dst, tupsz);
}
