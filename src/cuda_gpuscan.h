/*
 * cuda_gpuscan.h
 *
 * CUDA device code specific to GpuScan logic
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
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
	cl_ulong		data[1];			/* variable length area */
	/* <-- gpuscanSuspendContext --> */
};
typedef struct kern_gpuscan		kern_gpuscan;

typedef struct gpuscanSuspendContext
{
	cl_uint		part_index;
	cl_uint		line_index;
} gpuscanSuspendContext;

#define KERN_GPUSCAN_SUSPEND_CONTEXT(kgpuscan, group_id)			\
	((kgpuscan)->suspend_sz == 0									\
	 ? NULL															\
	 : ((gpuscanSuspendContext *)(kgpuscan)->data) + (group_id))


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
DEVICE_FUNCTION(cl_bool)
gpuscan_quals_eval_column(kern_context *kcxt,
						  kern_data_store *kds,
						  kern_data_extra *extra,
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
DEVICE_FUNCTION(void)
gpuscan_projection_column(kern_context *kcxt,
						  kern_data_store *kds_src,
						  kern_data_extra *kds_extra,
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
DEVICE_FUNCTION(void)
gpuscan_main_column(kern_context *kcxt,
                    kern_gpuscan *kgpuscan,
                    kern_data_store *kds_src,
                    kern_data_extra *kds_extra,
                    kern_data_store *kds_dst);
#endif	/* __CUDACC__ */
#ifdef __CUDACC_RTC__
/*
 * GPU kernel entrypoint - valid only NVRTC
 */
KERNEL_FUNCTION(void)
kern_gpuscan_main_row(kern_gpuscan *kgpuscan,
					  kern_parambuf *kparams,
                      kern_data_store *kds_src,
					  kern_data_extra *__not_valid__,
                      kern_data_store *kds_dst)
{
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpuscan_main_row(&u.kcxt, kgpuscan, kds_src, kds_dst,
					 GPUSCAN_HAS_DEVICE_PROJECTION);
	kern_writeback_error_status(&kgpuscan->kerror, &u.kcxt);
}

KERNEL_FUNCTION(void)
kern_gpuscan_main_block(kern_gpuscan *kgpuscan,
						kern_parambuf *kparams,
                        kern_data_store *kds_src,
						kern_data_extra *__not_valid__,
                        kern_data_store *kds_dst)
{
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpuscan_main_block(&u.kcxt, kgpuscan, kds_src, kds_dst,
					   GPUSCAN_HAS_DEVICE_PROJECTION);
	kern_writeback_error_status(&kgpuscan->kerror, &u.kcxt);
}

KERNEL_FUNCTION(void)
kern_gpuscan_main_arrow(kern_gpuscan *kgpuscan,
						kern_parambuf *kparams,
						kern_data_store *kds_src,
						kern_data_extra *__not_valid__,
                        kern_data_store *kds_dst)
{
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpuscan_main_arrow(&u.kcxt, kgpuscan, kds_src, kds_dst,
					   GPUSCAN_HAS_DEVICE_PROJECTION);
	kern_writeback_error_status(&kgpuscan->kerror, &u.kcxt);
}

KERNEL_FUNCTION(void)
kern_gpuscan_main_column(kern_gpuscan *kgpuscan,
						 kern_parambuf *kparams,
						 kern_data_store *kds_src,
						 kern_data_extra *kds_extra,
						 kern_data_store *kds_dst)
{
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpuscan_main_column(&u.kcxt,
						kgpuscan,
						kds_src,
						kds_extra,
						kds_dst);
	kern_writeback_error_status(&kgpuscan->kerror, &u.kcxt);
}

#endif	/* __CUDACC_RTC__ */
#endif	/* CUDA_GPUSCAN_H */
