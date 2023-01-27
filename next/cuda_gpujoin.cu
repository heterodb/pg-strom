/*
 * cuda_gpujoin.cu
 *
 * GPU accelerated parallel relations join based on hash-join or
 * nested-loop logic.
 * --
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"

/*
 * kern_gpujoin_main
 */
INLINE_FUNCTION(kern_warp_context *)
GPUJOIN_LOCAL_WARP_CONTEXT(int nrels)
{
	uint32_t	unitsz = KERN_WARP_CONTEXT_UNITSZ(nrels);

	return (kern_warp_context *)
		(__pgstrom_dynamic_shared_workmem + unitsz * WarpId());
}

KERNEL_FUNCTION(void)
kern_gpujoin_main(kern_session_info *session,
				  kern_gpujoin *kgjoin,
				  kern_multirels *kmrels,
				  kern_data_store *kds_src,
				  kern_data_extra *kds_extra,
				  kern_data_store *kds_dst)
{
	kern_warp_context *wp = GPUJOIN_LOCAL_WARP_CONTEXT(kmrels->num_rels);
	kern_expression *kexp_scan_quals = SESSION_KEXP_SCAN_QUALS(session);
	kern_context   *kcxt;
	int				depth;
	__shared__ uint32_t smx_row_count;

	INIT_KERNEL_CONTEXT(kcxt, session);
	/* sanity checks */
	assert(kgjoin->num_rels == kmrels->num_rels);
	/* resume the previous execution context */
	if (get_local_id() == 0)
		smx_row_count = 0;
	if (LaneId() == 0)
	{
		uint32_t	unitsz = KERN_WARP_CONTEXT_UNITSZ(kmrels->num_rels);
		uint32_t	offset = unitsz * (get_global_id() / warpSize);

		memcpy(wp, kgjoin->data + offset, unitsz);
		wp->nrels = kmrels->num_rels;
	}
	__syncthreads();

	/* main logic of GpuJoin */
	while ((depth = __shfl_sync(__activemask(), wp->depth, 0)) >= 0)
	{
		if (depth == 0)
		{
			/* LOAD FROM THE SOURCE */
			depth = execGpuScanLoadSource(kcxt, wp,
										  kds_src,
										  kds_extra,
										  kexp_scan_quals,
										  &smx_row_count);
			if (__any_sync(__activemask(), depth < 0) != 0)
			{
				assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
				depth = -1;		/* bailout */
			}
		}
		else if (depth > kmrels->num_rels)
		{
			/* PROJECTION */

		}
		else if (kmrels->chunks[depth-1].is_nestloop)
		{
			/* NEST-LOOP */

		}
#if 0
		else if (kmrels->chunks[depth-1].gist_offset != 0)
		{
			/* GiST-INDEX-JOIN */
		}
#endif
		else
		{
			/* HASH-JOIN */


		}
	}

	/* suspend the execution context */
	if (LaneId() == 0)
	{
		uint32_t    unitsz = KERN_WARP_CONTEXT_UNITSZ(kmrels->num_rels);
		uint32_t    offset = unitsz * (get_global_id() / warpSize);

		memcpy(kgjoin->data + offset, wp, unitsz);
	}
}
