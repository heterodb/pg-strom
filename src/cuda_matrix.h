/*
 * cuda_matrix.h
 *
 * collection of matrix/vector/array support routines for CUDA devices
 * --
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H
#ifdef __CUDACC__

/*
 * Device side ArrayRef support
 *
 * PG-Strom support ArrayRef only if it references a scalar value in
 * an array value, like 'my_array[3][4]'.
 * 
 *
 *
 *
 *
 *
 *
 *
 */





#define ARR_SIZE(a)		VARSIZE_ANY(a)
#define ARR_NDIM(a)						\
	(pg_get_int32_value((char *)(a) + offsetof(ArrayType, ndim)))
#define ARR_HASNULL(a)					\
	(pg_get_int32_value((char *)(a) + offsetof(ArrayType, dataoffset)) != 0)
#define ARR_ELEMTYPE(a)					\
	(pg_get_int32_value((char *)(a) + offsetof(ArrayType, elemtype)))

/* copy of host-side ArrayGetOffset */
STATIC_INLINE(cl_uint)
pg_array_get_offset()
{
	cl_int		offset = 0;
	cl_int		scale = 1;
	cl_int		i;

	for (i = n - 1; i >= 0; i--)
	{
		offset += (index[i] - lbound[i]) * scale;
		scale *= dim[i];
	}
	return offset;
}

/* copy of host-side array_get_isnull */
STATIC_INLINE(cl_bool)
pg_array_get_isnull(const cl_uchar *nullbitmap, int offset))
{
	if (!nullbitmap)
		return false;		/* assume not null */
	if (nullbitmap[offset / BITS_PER_BYTE] & (1 << (offset % BITS_PER_BYTE)))
		return false;		/* not null */
	return true;			/* null */
}

STATIC_INLINE(void *)
pg_array_seek()
{
	/* easy if fixed-size elements and no NULLs */
	if (typlen > 0 && !nullbitmap)
		return ptr + nitems * ((size_t) att_align_nominal(typlen, typalign));

	/* seems worth having separate loops for NULL and no-NULLs cases */
	if (nullbitmap)
	{



	}
	else
	{
		for (i = 0; i < nitems; i++)
		{
			
			
		}
	}
	return addr;
}


STATIC_FUNCTION(void *)
pg_array_get_element(kern_context *kcxt,
					 pg_array_t array,
	)
{

	if (array.isnull)
		return NULL;

	ndim = ARR_NDIM(array.value);
	if (ndim != nscripts || ndim <= 0 || ndim > MAXDIM)
		return NULL;

	for (i = 0; i < ndim; i++)
	{
		if (index[i] < lb[i] || index[i] >= (dim[i] + lb[i]))
			return NULL;
	}

	/* calculate the element number */
	offset = pg_array_get_offset();



}





STATIC_FUNCTION(void *)
pg_array_reference_1d(kern_context *kcxt, pg_array_t *array,
					  cl_uint index_x)
{}

STATIC_FUNCTION(void *)
pg_array_reference_2d(kern_context *kcxt, pg_array_t *array,
					  cl_uint index_x, cl_uint index_y)
{}







#endif
