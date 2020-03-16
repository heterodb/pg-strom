/*
 * cuda_gpujoin.h
 *
 * GPU accelerated parallel relations join based on hash-join or
 * nested-loop logic.
 * --
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#ifndef CUDA_GPUJOIN_H
#define CUDA_GPUJOIN_H
/*
 * definition of the inner relations structure. it can load multiple
 * kern_data_store or kern_hash_table.
 */
typedef struct
{
	cl_ulong		kmrels_length;	/* length of kern_multirels */
	cl_ulong		ojmaps_length;	/* length of outer-join map, if any */
	cl_uint			cuda_dindex;	/* device index of PG-Strom */
	cl_uint			nrels;			/* number of inner relations */
	struct
	{
		cl_ulong	chunk_offset;	/* offset to KDS or Hash */
		cl_ulong	ojmap_offset;	/* offset to outer-join map, if any */
		cl_bool		is_nestloop;	/* true, if NestLoop. */
		cl_bool		left_outer;		/* true, if JOIN_LEFT or JOIN_FULL */
		cl_bool		right_outer;	/* true, if JOIN_RIGHT or JOIN_FULL */
		cl_char		__padding__[5];
	} chunks[FLEXIBLE_ARRAY_MEMBER];
} kern_multirels;

#define KERN_MULTIRELS_INNER_KDS(kmrels, depth)	\
	((kern_data_store *)						\
	 ((char *)(kmrels) + (kmrels)->chunks[(depth)-1].chunk_offset))

#define KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth)					\
	((cl_bool *)((kmrels)->chunks[(depth)-1].right_outer				\
				 ? ((char *)(kmrels) +									\
					(size_t)(kmrels)->kmrels_length +					\
					(size_t)(kmrels)->chunks[(depth)-1].ojmap_offset)	\
				 : NULL))

#define KERN_MULTIRELS_LEFT_OUTER_JOIN(kmrels, depth)	\
	__ldg(&((kmrels)->chunks[(depth)-1].left_outer))

#define KERN_MULTIRELS_RIGHT_OUTER_JOIN(kmrels, depth)	\
	__ldg(&((kmrels)->chunks[(depth)-1].right_outer))

/*
 * kern_gpujoin - control object of GpuJoin
 *
 * The control object of GpuJoin has four segments as follows:
 * +------------------+
 * | 1. kern_gpujoin  |
 * +------------------+
 * | 2. kern_parambuf |
 * +------------------+
 * | 3. pseudo stack  |
 * +------------------+
 * | 4. saved context |
 * | for suspend and  |
 * | resume           |
 * +------------------+
 *
 * The first segment is the control object of GpuJoin itself, and the second
 * one is buffer of contrant variables.
 * The third segment is used to save the combination of joined rows as
 * intermediate results, performs like a pseudo-stack area. Individual SMs
 * have exclusive pseudo-stack, thus, can be utilized as a large but slow
 * shared memory. (If depth is low, we may be able to use the actual shared
 * memory instead.)
 * The 4th segment is used to save the execution context when GPU kernel
 * gets suspended. Both of shared memory contents (e.g, read_pos, write_pos)
 * and thread's private variables (e.g, depth, l_state, matched) are saved,
 * then, these state variables shall be restored on restart.
 * GpuJoin kernel will suspend the execution if and when destination buffer
 * gets filled up. Host code is responsible to detach the current buffer
 * and allocates a new one, then resume the GPU kernel.
 * The 4th segment for suspend / resume shall not be in-use unless destination
 * buffer does not run out, thus, it shall not consume devuce physical pages
 * because we allocate the control segment using unified managed memory.
 */
struct kern_gpujoin
{
	kern_errorbuf	kerror;				/* kernel error information */
	cl_uint			kparams_offset;		/* offset to the kparams */
	cl_uint			pstack_offset;		/* offset to the pseudo-stack */
	cl_uint			pstack_nrooms;		/* size of pseudo-stack */
	cl_uint			num_rels;			/* number of inner relations */
	cl_uint			grid_sz;			/* grid-size on invocation */
	cl_uint			block_sz;			/* block-size on invocation */
	/* suspend/resume related */
	cl_uint			suspend_offset;		/* offset to the suspend-backup */
	cl_uint			suspend_size;		/* length of the suspend buffer */
	cl_uint			suspend_count;		/* number of suspended blocks */
	cl_bool			resume_context;		/* resume context from suspend */
	cl_uint			src_read_pos;		/* position to read from kds_src */
	/* error status to be backed (OUT) */
	cl_uint			source_nitems;		/* out: # of source rows */
	cl_uint			outer_nitems;		/* out: # of filtered source rows */
	cl_uint			stat_nitems[FLEXIBLE_ARRAY_MEMBER]; /* out: stat nitems */
	/*-- pseudo-stack and suspend/resume context --*/
	/*-- kernel param/const buffer --*/
};
typedef struct kern_gpujoin		kern_gpujoin;

#ifndef __CUDACC__
/*
 * gpujoin_reset_kernel_task - reset kern_gpujoin status prior to resume
 */
STATIC_INLINE(void)
gpujoin_reset_kernel_task(kern_gpujoin *kgjoin, bool resume_context)
{
	memset(&kgjoin->kerror, 0, sizeof(kern_errorbuf));
	kgjoin->suspend_count	= 0;
	kgjoin->resume_context	= resume_context;
}
#endif

/*
 * suspend/resume context
 */
struct gpujoinSuspendContext
{
	cl_int			depth;
	cl_bool			scan_done;
	cl_uint			src_read_pos;
	cl_uint			stat_source_nitems;
	struct {
		cl_uint		wip_count;
		cl_uint		read_pos;
		cl_uint		write_pos;
		cl_uint		stat_nitems;
		cl_uint		l_state[MAXTHREADS_PER_BLOCK];	/* private variables */
		cl_bool		matched[MAXTHREADS_PER_BLOCK];	/* private variables */
	} pd[FLEXIBLE_ARRAY_MEMBER];	/* per-depth */
};
typedef struct gpujoinSuspendContext	gpujoinSuspendContext;

#define KERN_GPUJOIN_PARAMBUF(kgjoin)					\
	((kern_parambuf *)((char *)(kgjoin) + (kgjoin)->kparams_offset))
#define KERN_GPUJOIN_PARAMBUF_LENGTH(kgjoin)			\
	STROMALIGN(KERN_GPUJOIN_PARAMBUF(kgjoin)->length)
#define KERN_GPUJOIN_HEAD_LENGTH(kgjoin)				\
	STROMALIGN((char *)KERN_GPUJOIN_PARAMBUF(kgjoin) +	\
			   KERN_GPUJOIN_PARAMBUF_LENGTH(kgjoin) -	\
			   (char *)(kgjoin))
#define KERN_GPUJOIN_PSEUDO_STACK(kgjoin)					\
	((cl_uint *)((char *)(kgjoin) + (kgjoin)->pstack_offset))
#define KERN_GPUJOIN_SUSPEND_CONTEXT(kgjoin,group_id)		\
	((struct gpujoinSuspendContext *)						\
	 ((char *)(kgjoin) + (kgjoin)->suspend_offset +			\
	  (group_id) * STROMALIGN(offsetof(gpujoinSuspendContext, \
									   pd[(kgjoin)->num_rels + 1]))))
/* utility macros for automatically generated code */
#define GPUJOIN_REF_HTUP(chunk,offset)			\
	((offset) == 0								\
	 ? NULL										\
	 : (HeapTupleHeaderData *)((char *)(chunk) + (size_t)(offset)))
/* utility macros for automatically generated code */
#define GPUJOIN_REF_DATUM(colmeta,htup,colidx)	\
	(!(htup) ? NULL : kern_get_datum_tuple((colmeta),(htup),(colidx)))

#ifdef __CUDACC__
/*
 * gpujoin_quals_eval(_arrow)
 */
DEVICE_FUNCTION(cl_bool)
gpujoin_quals_eval(kern_context *kcxt,
				   kern_data_store *kds,
				   ItemPointerData *t_self,
				   HeapTupleHeaderData *htup);
DEVICE_FUNCTION(cl_bool)
gpujoin_quals_eval_arrow(kern_context *kcxt,
						 kern_data_store *kds,
						 cl_uint row_index);

/*
 * gpujoin_join_quals
 *
 * Evaluation of join qualifier in the given depth. It shall return true
 * if supplied pair of the rows matches the join condition.
 *
 * NOTE: if x-axil (outer input) or y-axil (inner input) are out of range,
 * we expect outer_index or inner_htup are NULL. Don't skip to call this
 * function, because nested-loop internally uses __syncthread operation
 * to reduce DRAM accesses.
 */
DEVICE_FUNCTION(cl_bool)
gpujoin_join_quals(kern_context *kcxt,
				   kern_data_store *kds,
				   kern_multirels *kmrels,
				   int depth,
				   cl_uint *x_buffer,
				   HeapTupleHeaderData *inner_htup,
				   cl_bool *joinquals_matched);

/*
 * gpujoin_hash_value
 *
 * Calculation of hash value if this depth uses hash-join logic.
 */
DEVICE_FUNCTION(cl_uint)
gpujoin_hash_value(kern_context *kcxt,
				   kern_data_store *kds,
				   kern_multirels *kmrels,
				   cl_int depth,
				   cl_uint *x_buffer,
				   cl_bool *is_null_keys);

/*
 * gpujoin_projection
 *
 * Implementation of device projection. Extract a pair of outer/inner tuples
 * on the tup_values/tup_isnull array.
 */
DEVICE_FUNCTION(cl_uint)
gpujoin_projection(kern_context *kcxt,
				   kern_data_store *kds_src,
				   kern_multirels *kmrels,
				   cl_uint *r_buffer,
				   kern_data_store *kds_dst,
				   cl_char *tup_dclass,
				   Datum   *tup_values,
				   cl_uint *tup_extras);
/*
 * gpujoin_main - main logic for inner-join
 */
DEVICE_FUNCTION(void)
gpujoin_main(kern_context *kcxt,
			 kern_gpujoin *kgjoin,
			 kern_multirels *kmrels,
			 kern_data_store *kds_src,
			 kern_data_store *kds_dst,
			 kern_parambuf *kparams_gpreagg,		/* only if combined Join */
			 cl_uint *l_state,
			 cl_bool *matched);
/*
 * gpujoin_right_outer - main logic for right-outer-join
 */
DEVICE_FUNCTION(void)
gpujoin_right_outer(kern_context *kcxt,
					kern_gpujoin *kgjoin,
					kern_multirels *kmrels,
					cl_int outer_depth,
					kern_data_store *kds_dst,
					kern_parambuf *kparams_gpreagg, /* only if combined Join */
					cl_uint *l_state,
					cl_bool *matched);
#endif	/* __CUDACC__ */

#ifdef __CUDACC_RTC__
/*
 * GPU kernel entrypoint - valid only NVRTC
 */
__shared__ cl_uint   wip_count[GPUJOIN_MAX_DEPTH+1];
__shared__ cl_uint   read_pos[GPUJOIN_MAX_DEPTH+1];
__shared__ cl_uint   write_pos[GPUJOIN_MAX_DEPTH+1];
__shared__ cl_uint   stat_nitems[GPUJOIN_MAX_DEPTH+1];

KERNEL_FUNCTION(void)
kern_gpujoin_main(kern_gpujoin *kgjoin,
				  kern_multirels *kmrels,
				  kern_data_store *kds_src,
				  kern_data_store *kds_dst,
				  kern_parambuf *kparams_gpreagg)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	cl_uint			l_state[GPUJOIN_MAX_DEPTH+1];
	cl_bool			matched[GPUJOIN_MAX_DEPTH+1];
	DECL_KERNEL_CONTEXT(u);

	assert(kgjoin->num_rels == GPUJOIN_MAX_DEPTH);
	memset(l_state, 0, sizeof(l_state));
	memset(matched, 0, sizeof(matched));
	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpujoin_main(&u.kcxt,
				 kgjoin,
				 kmrels,
				 kds_src,
				 kds_dst,
				 kparams_gpreagg,
				 l_state,
				 matched);
	kern_writeback_error_status(&kgjoin->kerror, &u.kcxt);
}

KERNEL_FUNCTION(void)
kern_gpujoin_right_outer(kern_gpujoin *kgjoin,
						 kern_multirels *kmrels,
						 cl_int outer_depth,
						 kern_data_store *kds_dst,
						 kern_parambuf *kparams_gpreagg)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	cl_uint			l_state[GPUJOIN_MAX_DEPTH+1];
	cl_bool			matched[GPUJOIN_MAX_DEPTH+1];
	DECL_KERNEL_CONTEXT(u);

	assert(kgjoin->num_rels == GPUJOIN_MAX_DEPTH);
	memset(l_state, 0, sizeof(l_state));
	memset(matched, 0, sizeof(matched));
	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpujoin_right_outer(&u.kcxt,
						kgjoin,
						kmrels,
						outer_depth,
						kds_dst,
						kparams_gpreagg,
						l_state,
						matched);
	kern_writeback_error_status(&kgjoin->kerror, &u.kcxt);
}

#ifndef GPUPREAGG_COMBINED_JOIN
DEVICE_FUNCTION(void)
gpupreagg_projection_slot(kern_context *kcxt_gpreagg,
						  cl_char *src_dclass,
						  Datum   *src_values,
						  cl_char *dst_dclass,
						  Datum   *dst_values)
{
	/* should never be called */
	assert(false);
}
#endif	/* !GPUPREAGG_COMBINED_JOIN */
#endif	/* __CUDACC_RTC__ */
#endif	/* CUDA_GPUJOIN_H */
