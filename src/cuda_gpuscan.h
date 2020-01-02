/*
 * cuda_gpuscan.h
 *
 * CUDA device code specific to GpuScan logic
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
/*
 * GpuScan main logic
 */
DEVICE_FUNCTION(void)
gpuscan_main_row(kern_context *kcxt,
				 kern_gpuscan *kgpuscan,
				 kern_data_store *kds_src,
				 kern_data_store *kds_dst,
				 bool has_device_projection);
DEVICE_FUNCTION(void)
gpuscan_main_block(kern_context *kcxt,
				   kern_gpuscan *kgpuscan,
				   kern_data_store *kds_src,
				   kern_data_store *kds_dst,
				   bool has_device_projection);
DEVICE_FUNCTION(void)
gpuscan_main_arrow(kern_context *kcxt,
				   kern_gpuscan *kgpuscan,
				   kern_data_store *kds_src,
				   kern_data_store *kds_dst,
				   bool has_device_projection);
#endif	/* __CUDACC__ */
#ifdef __CUDACC_RTC__
/*
 * GPU kernel entrypoint - valid only NVRTC
 */
KERNEL_FUNCTION(void)
kern_gpuscan_main_row(kern_gpuscan *kgpuscan,
                      kern_data_store *kds_src,
                      kern_data_store *kds_dst)
{
	kern_parambuf *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpuscan_main_row(&u.kcxt, kgpuscan, kds_src, kds_dst,
					 GPUSCAN_HAS_DEVICE_PROJECTION);
	kern_writeback_error_status(&kgpuscan->kerror, &u.kcxt);
}

KERNEL_FUNCTION(void)
kern_gpuscan_main_block(kern_gpuscan *kgpuscan,
                        kern_data_store *kds_src,
                        kern_data_store *kds_dst)
{
	kern_parambuf *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpuscan_main_block(&u.kcxt, kgpuscan, kds_src, kds_dst,
					   GPUSCAN_HAS_DEVICE_PROJECTION);
	kern_writeback_error_status(&kgpuscan->kerror, &u.kcxt);
}

KERNEL_FUNCTION(void)
kern_gpuscan_main_arrow(kern_gpuscan *kgpuscan,
                        kern_data_store *kds_src,
                        kern_data_store *kds_dst)
{
	kern_parambuf *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpuscan_main_arrow(&u.kcxt, kgpuscan, kds_src, kds_dst,
					   GPUSCAN_HAS_DEVICE_PROJECTION);
	kern_writeback_error_status(&kgpuscan->kerror, &u.kcxt);
}
#endif	/* __CUDACC_RTC__ */
#endif	/* CUDA_GPUSCAN_H */
