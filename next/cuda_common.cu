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
#include "xpu_common.c"
#include "cuda_common.h"

/*
 * __execProjectionCommon
 */
STATIC_FUNCTION(bool)
__execProjectionCommon(kern_context *kcxt,
					   kern_expression *kexp,
					   kern_data_store *kds_dst,
					   int nvalids,
					   uint32_t tupsz)
{
	size_t		offset;
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
	/* data store no space? */
	if (__any_sync(__activemask(), try_suspend))
		return false;

	/* write out the tuple */
	if (LaneId() < nvalids)
	{
		kern_expression *kproj = KEXP_FIRST_ARG(kexp);
		uint32_t	row_id = oldval.i.nitems + LaneId();
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
	return true;
}

/*
 * ExecKernProjection
 */
PUBLIC_FUNCTION(bool)
ExecProjectionOuterRow(kern_context *kcxt,
					   kern_expression *kexp,	/* LoadVars + Projection */
					   kern_data_store *kds_dst,
					   kern_data_store *kds_outer,
					   HeapTupleHeaderData *htup_outer,
					   int num_inners,
					   kern_data_store **kds_inners,
					   HeapTupleHeaderData **htup_inners)
{
	uint32_t	nvalids;
	uint32_t	mask;
	uint32_t	tupsz = 0;

	assert(__activemask() == 0xffffffffU);
	mask = __ballot_sync(__activemask(), htup_outer != NULL);
	nvalids = __popc(mask);
	assert(htup_outer != NULL
		   ? LaneId() <  nvalids
		   : LaneId() >= nvalids);

	/*
	 * First, extract the variables from outer/inner tuples, and
	 * calculate expressions, if any.
	 */
	assert(kexp->opcode == FuncOpCode__LoadVars &&
		   kexp->exptype == TypeOpCode__int4);
	if (LaneId() < nvalids)
	{
		xpu_int4_t	__tupsz;

		if (ExecLoadVarsOuterRow(kcxt,
								 kexp,
								 (xpu_datum_t *)&__tupsz,
								 kds_outer,
								 htup_outer,
								 num_inners,
								 kds_inners,
								 htup_inners))
		{
			if (!__tupsz.isnull)
			{
				tupsz = MAXALIGN(__tupsz.value);
				assert(tupsz > 0);
			}
		}
	}
	return __execProjectionCommon(kcxt, kexp, kds_dst, nvalids, tupsz);
}
