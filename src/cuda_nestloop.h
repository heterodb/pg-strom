/*
 * cuda_nestloop.h
 *
 * Parallel hash join accelerated by OpenCL device
 * --
 * Copyright 2011-2015 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2015 (C) The PG-Strom Development Team
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
#ifndef CUDA_NESTLOOP_H
#define CUDA_NESTLOOP_H



typedef struct
{
	size_t			kresults_1_offset;
	size_t			kresults_2_offset;
	size_t			kresults_length;
	cl_uint			max_depth;
	cl_int			errcode;
	kern_parambuf	kparams;
} kern_nestloop;

#define KERN_NESTLOOP_PARAMBUF(knestloop)			\
	((kern_parambuf *)(&(knestloop)->kparams))
#define KERN_NESTLOOP_PARAMBUF_LENGTH(knestloop)	\
	STROMALIGN(KERN_NESTLOOP_PARAMBUF(knestloop)->length)
#define KERN_NESTLOOP_IN_RESULTSBUF(knestloop, depth)		\
	((kern_resultbuf *)((char *)(knestloop) +				\
						(((depth) & 0x01)					\
						 ? (knestloop)->kresults_1_offset	\
						 : (knestloop)->kresults_2_offset))))
#define KERN_NESTLOOP_OUT_RESULTSBUF(knestloop, depth)		\
	((kern_resultbuf *)((char *)(knestloop) +				\
						(((depth) & 0x01)					\
						 ? (knestloop)->kresults_2_offset	\
						 : (knestloop)->kresults_1_offset))))

typedef struct
{
	hostptr_t		hostptr;	/* address of this multihash on the host */
	cl_uint			nrels;		/* number of relations */
	struct {
		cl_uint		kds_offset;	/* offset of the kern_data_store */
		cl_uint		rmap_offset;/* offset of the reference map, if any */
	} rels[FLEXIBLE_ARRAY_MEMBER];
} kern_multi_relstore;

#define KERN_MULTI_RELSTORE_INNER_KDS(kmrels, depth)				\
	((kern_data_store *)											\
	 ((depth) > 1 && (depth) <= (kmrels)->nrels						\
	  ? ((char *)(kmrels) + (kmrels)->rels[(depth) - 1].kds_offset)	\
	  : NULL))

#define KERN_MULTI_RELSTORE_REFERENCE_MAP(kmrels, depth)			\
	((cl_bool *)													\
	 ((depth) > 1 && (depth) <= (kmrels)->nrels &&					\
	  (kmrels)->rels[(depth) - 1].rmap_offset > 0					\
	  ? ((char *)(kmrels) + (kmrels)->rels[(depth) - 1].rmap_offset)\
	  : NULL))



#ifdef __CUDACC__
STATIC_FUNCTION(cl_bool)
gpunestloop_qual_eval(cl_int *errcode,
					  kern_parambuf *kparams,
                      kern_data_store *kds,
					  size_t kds_index);

STATIC_FUNCTION(cl_bool)
gpunestloop_execute(cl_int *errcode,
					kern_parambuf *kparams,
					kern_multi_relstore *kmrels,
					kern_data_store *kds,
					int depth,
					cl_int *outer_index,
					cl_int inner_index);

STATIC_FUNCTION(void)
gpunestloop_projection_mapping(cl_int dest_colidx,
							   cl_int *src_depth,
							   cl_int *src_colidx);

KERNEL_FUNCTION(void)
gpunestloop_prep(kern_nestloop *knestloop,
                 kern_multi_relstore *kmrels,
                 kern_data_store *kds,
                 cl_int depth)
{
	kern_resultbuf *kresults_in;
	kern_resultbuf *kresults_out;
	cl_int			errcode = StromError_Success;

	/* sanity check */
	assert(depth > 0 && depth <= knestloop->max_depth);
	assert(knestloop->kresults_1_offset > 0);
	assert(knestloop->kresults_2_offset > 0);

	kresults_in = KERN_NESTLOOP_IN_RESULTSBUF(knestloop, depth);
	kresults_out = KERN_NESTLOOP_OUT_RESULTSBUF(knestloop, depth);

	/*
	 * In case of depth == 1, input result buffer is not initialized
	 * yet. So, we need to put initial values first of all.
	 *
	 * NOTE: gpunestloop_prep(depth==1) expects base portion of the
	 * kern_resultbuf structure is zero cleared, so host-side has to
	 * call cuMemsetD32() or others.
	 */
	if (depth == 1)
	{
		cl_bool		is_matched;
		cl_int		num_matched;

		/*
		 * Check qualifier of outer scan that was pulled-up (if any).
		 * then, it allocates result buffer on kresults_in and put
		 * get_global_id() if it match.
		 */
		if (get_global_id() < kds_in->nitems)
			is_matched = gpunestloop_qual_eval(&errcode, kparams, kds,
											   get_global_id());
		else
			is_matched = false;

		/* expand kresults_in->nitems */
		offset = arithmetic_stairlike_add(is_matched ? 1 : 0, &num_matched);
		if (get_local_id() == 0)
		{
			if (num_matched > 0)
				base = atomicAdd(&kresults_in->nitems, num_matched);
			else
				base = 0;
		}
		__synchthreads();

		if (base + num_matched > knestloop->kresults_total_items)
		{
			errcode = StromError_DataStoreNoSpace;
			goto out;
		}
		kresults->result[base + offset] = get_global_id();

		/* init other base portion */
		if (get_global_id() == 0)
		{
			kresults_in->nrels = depth;
            kresults_in->nrooms = kds->nitems;
            kresults_in->errcode = 0;   /* never used in gpunestloop */
		}
	}

	/* init output kresults buffer */
	if (get_global_id() == 0)
	{
		assert(kresults_in->nrels == depth);
		kresults_out->nrels = depth + 1;
		kresults_out->nrooms = knestloop->kresults_total_items / (depth + 1);
		kresults_out->nitems = 0;
		kresults_out->errcode = 0;	/* never used in gpunestloop */
	}
out:
	kern_writeback_error_status(&knestloop->errcode, errcode);
}

KERNEL_FUNCTION_MAXTHREADS(void)
gpunestloop_main(kern_nestloop *knestloop,
				 kern_multi_relstore *kmrels,
				 kern_data_store *kds,
				 cl_int depth)
{
	kern_parambuf	   *kparams = KERN_NESTLOOP_PARAMBUF(knestloop);
	kern_resultbuf	   *kresults_in;
	kern_resultbuf	   *kresults_out;
	kern_data_store	   *kds_in;
	cl_int				y_index;
	cl_int				x_index;
	cl_int				x_limit;
	cl_int				errcode = StromError_Success;

	/* already has an error status? */
	if (knestloop->errcode != StromError_Success)
		return;

	kresults_in = KERN_NESTLOOP_IN_RESULTSBUF(knestloop, depth);
	kresults_out = KERN_NESTLOOP_OUT_RESULTSBUF(knestloop, depth);

	/* sanity checks */
	assert(depth > 0 && depth <= knestloop->max_depth);
	assert(kresults_out->nrels == depth + 1);
	assert(kresults_out->nrels == (!kresults_in ? 2 : kresults_in->nrels + 1));

	/*
	 * NOTE: size of Y-axis deterministric on the time of kernel launch.
	 * host-side guarantees get_global_ysize() is larger then kds->nitems
	 * of the depth. On the other hands, nobody can know correct size of
	 * X-axis unless gpunestloop_main() of the previous stage.
	 * So, we ensure all the outer items are picked up by the loop below.
	 */
	kds_in = KERN_MULTI_RELSTORE_INNER_KDS(kmrels, depth);
	assert(kds_in != NULL);

	y_index = get_global_xid();
	if (y_index < kds_in->nitems)
	{
		x_limit = ((kresults_in->nitems + get_global_xsize() - 1) /
				   get_global_xsize()) * get_global_xsize();
		for (x_index = get_global_xid();
			 x_index < x_limit;
			 x_index += get_global_xsize())
		{
			cl_int		   *x_buffer = KERN_GET_RESULT(kresults_in, x_index);
			cl_int		   *r_buffer;
			cl_bool			is_matched;
			__shared__ cl_int base;

			if (x_index < kresults_in->nitems)
				matched = gpunestloop_execute(&errcode,
											  kparams,
											  kmrels,
											  kds,
											  depth,
											  x_buffer,
											  y_index);
			else
				matched = false;

			/* expand kresults_out->nitems */
			offset = arithmetic_stairlike_add(matched ? 1 : 0,
											  &num_matched);
			if (get_local_id() == 0)
			{
				if (num_matched > 0)
					base = atomicAdd(&kresults_out->nitems, num_matched);
				else
					base = 0;
			}
			__synchthreads();

			if (base + num_matched > knestloop->kresults_total_items)
			{
				STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);
				goto out;
			}

			if (is_matched)
			{
				rbuffer = KERN_GET_RESULT(kresults_out, base + offset);
				for (i=0; i < depth; i++)
					rbuffer[i] = xbuffer[i];
				rbuffer[depth] = y_index;
			}
		}
	}
out:
	kern_writeback_error_status(&knestloop->errcode, errcode);
}

/*
 * gpunestloop_outer_checkup
 *
 * It checks referencial map of inner relations, then if nobody didn't
 * pick up the entry, it adds an outer entry for each unreferenced one.
 */
KERNEL_FUNCTION(void)
gpunestloop_outer_checkup(kern_nestloop *knestloop,
						  kern_multi_relstore *kmrels,
						  kern_data_store *kds,
						  cl_int depth)
{
	kern_resultbuf	   *kresults_out;
	kern_data_store	   *kds_in;
	cl_bool			   *refmap;
	cl_bool				is_outer;
	cl_int				num_outer;
	cl_int				offset;
	cl_int			   *rbuffer;
	cl_int				errcode = StromError_Success;
	__shared__ cl_uint	base;
	cl_int				i;

	kresults_out = KERN_NESTLOOP_OUT_RESULTSBUF(knestloop, depth);
	kds_in = KERN_MULTI_RELSTORE_INNER_KDS(kmrels, depth);
	refmap = KERN_MULTI_RELSTORE_REFERENCE_MAP(kmrels, depth);
	if (!refmap)
	{
		STROM_SET_ERROR(&errcode, StromError_SanityCheckViolation);
		goto out;
	}

	/*
	 * check number of outer rows to be injected, then allocate slots
	 * for outer tuples.
	 */
	if (get_global_id() < kds_in->nitems)
		is_outer = !refmap[get_global_id()];
	else
		is_outer = false;
	offset = arithmetic_stairlike_add(is_outer ? 1 : 0, &num_outer);

	if (get_local_id() == 0)
	{
		if (num_outer > 0)
			base = atomicAdd(&kresults_out->nitems, num_outer);
		else
			base = 0;
	}
	__synchthreads();

	/* In case when (base + num_outer) is larger than or equal to the nrooms,
	 * it means we don't have enough space to write back nested-loop results.
	 * So, we have to tell the host-side to acquire larger kern_resultbuf.
	 */
	if (base + num_outer > kresults_out->nrooms)
	{
		errcode = StromError_DataStoreNoSpace;
		goto out;
    }

	/*
	 * OK, we know which row should be materialized using left outer join
	 * manner, and result buffer was acquired. Let's put result for the
	 * next stage.
	 */
	rbuffer = KERN_GET_RESULT(kresults_out, base + offset);
	for (i=0; i < depth; i++)
		rbuffer[i] = -1;	/* NULL */
	rbuffer[depth] = get_global_id();

out:
	kern_writeback_error_status(&knestloop->errcode, errcode);
}

KERNEL_FUNCTION(void)
gpunestloop_projection_row(kern_nestloop *knestloop,
						   kern_multi_relstore *kmrels,
						   kern_data_store *kds_src,
						   kern_data_store *kds_dst)
{
	
	
}

KERNEL_FUNCTION(void)
gpunestloop_projection_slot(kern_nestloop *knestloop,
							kern_multi_relstore *kmrels,
							kern_data_store *kds_src,
							kern_data_store *kds_dst)
{
	
	
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_NESTLOOP_H */
