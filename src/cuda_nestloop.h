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
	cl_uint			kresults_offset;
	cl_uint			innermap_offset;
	kern_parambuf	kparams;
} kern_nestloop;


typedef struct
{
	hostptr_t		hostptr;	/* address of this multihash on the host */
	cl_uint			ntables;	/* number of  */
	struct {
		cl_uint		kds_offset;	/* offset of the kern_data_store */
		cl_uint		rmap_offset;/* offset of the reference map, if any */
	} tables[FLEXIBLE_ARRAY_MEMBER];
} kern_multi_tables;





#ifdef __CUDACC__


#define GPUNESTLOOP_KERNEL_MAIN_TEMPLATE(depth)	\

KERNEL_FUNCTION(void)
gpunestloop_main(kern_nestloop *knl,
				 kern_multi_tables *km_tables,
				 kern_data_store *kds,
{



}




KERNEL_FUNCTION(void)
gpunestloop_main(kern_nestloop *knl,
				 kern_data_store *kds_inner,
				 kern_data_store *kds_outer)
{}

KERNEL_FUNCTION(void)
gpunestloop_projection_row()
{}

KERNEL_FUNCTION(void)
gpunestloop_projection_slot()
{}

#endif	/* __CUDACC__ */
#endif	/* CUDA_NESTLOOP_H */
