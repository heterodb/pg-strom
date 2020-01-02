/*
 * cuda_gpusort.h
 *
 * CUDA device code for GpuSort logic
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
#ifndef CUDA_GPUSORT_H
#define CUDA_GPUSORT_H
/*
 * kern_gpusort
 */
typedef struct {
	kern_errorbuf	kerror;
	cl_uint			nitems_in;
	cl_uint			nitems_out;
	kern_parambuf	kparams;
	/* <-- gpusortResultIndex --> */
} kern_gpusort;

typedef struct {
	cl_uint			nitems;
	cl_uint			results[FLEXIBLE_ARRAY_MEMBER];
} gpusortResultIndex;

#define KERN_GPUSORT_PARAMBUF(kgpusort)			\
	(&((kern_gpusort *)(kgpusort))->kparams)
#define KERN_GPUSORT_PARAMBUF_LENGTH(kgpusort)	\
	STROMALIGN(KERN_GPUSORT_PARAMBUF(kgpusort)->length)
#define KERN_GPUSORT_RESULT_INDEX(kgpusort)		\
	((gpusortResultIndex *)						\
	 ((char *)KERN_GPUSORT_PARAMBUF(kgpusort) + \
	  KERN_GPUSORT_PARAMBUF_LENGTH(kgpusort)))

#define BITONIC_MAX_LOCAL_SHIFT		12
#define BITONIC_MAX_LOCAL_SZ		(1<<BITONIC_MAX_LOCAL_SHIFT)

#ifdef __CUDACC__
/*
 * gpusort_quals_eval - evaluation of device qualifier
 */
DEVICE_FUNCTION(cl_bool)
gpusort_quals_eval(kern_context *kcxt,
				   kern_data_store *kds,
				   cl_uint row_index);
/*
 * gpusort_keycomp - comparison of two keys for sorting
 */
DEVICE_FUNCTION(cl_int)
gpusort_keycomp(kern_context *kcxt,
				kern_data_store *kds_src,
				cl_uint x_index,
				cl_uint y_index);
/*
 * GpuSort main logic;
 */
DEVICE_FUNCTION(void)
gpusort_setup_column(kern_context *kcxt,
					 kern_gpusort *kgpusort,
					 kern_data_store *kds_src);
DEVICE_FUNCTION(void)
gpusort_bitonic_local(kern_context *kcxt,
					  kern_gpusort *kgpusort,
					  kern_data_store *kds_src);
DEVICE_FUNCTION(void)
gpusort_bitonic_step(kern_context *kcxt,
					 kern_gpusort *kgpusort,
					 kern_data_store *kds_src,
					 cl_uint unitsz,
					 cl_bool reversing);
DEVICE_FUNCTION(void)
gpusort_bitonic_merge(kern_context *kcxt,
					  kern_gpusort *kgpusort,
					  kern_data_store *kds_src);
#endif	/* __CUDACC__ */

#ifdef	__CUDACC_RTC__
KERNEL_FUNCTION(void)
kern_gpusort_setup_column(kern_gpusort *kgpusort,
						  kern_data_store *kds_src)
{
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, &kgpusort->kparams);
	gpusort_setup_column(&u.kcxt, kgpusort, kds_src);
	kern_writeback_error_status(&kgpusort->kerror, &u.kcxt);
}

KERNEL_FUNCTION_MAXTHREADS(void)
kern_gpusort_bitonic_local(kern_gpusort *kgpusort,
						   kern_data_store *kds_src)
{
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, &kgpusort->kparams);
	gpusort_bitonic_local(&u.kcxt, kgpusort, kds_src);
	kern_writeback_error_status(&kgpusort->kerror, &u.kcxt);
}

KERNEL_FUNCTION_MAXTHREADS(void)
kern_gpusort_bitonic_step(kern_gpusort *kgpusort,
						  kern_data_store *kds_src,
						  cl_uint unitsz,
						  cl_bool reversing)
{
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, &kgpusort->kparams);
	gpusort_bitonic_step(&u.kcxt, kgpusort, kds_src, unitsz, reversing);
	kern_writeback_error_status(&kgpusort->kerror, &u.kcxt);
}

KERNEL_FUNCTION_MAXTHREADS(void)
kern_gpusort_bitonic_merge(kern_gpusort *kgpusort,
						   kern_data_store *kds_src)
{
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, &kgpusort->kparams);
	gpusort_bitonic_merge(&u.kcxt, kgpusort, kds_src);
	kern_writeback_error_status(&kgpusort->kerror, &u.kcxt);
}
#endif	/* __CUDACC_RTC__ */
#endif	/* CUDA_GPUSORT_H */
