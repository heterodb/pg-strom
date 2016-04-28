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


/* copied from utils/array.h */
typedef struct
{
	cl_uint		vl_len_;		/* varlena header (do not touch directly!) */
	cl_int		ndim;			/* # of dimensions */
	cl_int		dataoffset;		/* offset to data, or 0 if no bitmap */
	Oid			elemtype;		/* element type OID */
} ArrayType;





#define ARR_SIZE(a)		VARSIZE_ANY(a)
#define ARR_NDIM(a)						\
	(get_int32_val((char *)(a) + offsetof(ArrayType, ndim)))
#define ARR_HASNULL(a)					\
	(get_int32_val((char *)(a) + offsetof(ArrayType, dataoffset)) != 0)
#define ARR_ELEMTYPE(a)					\
	(get_int32_val((char *)(a) + offsetof(ArrayType, elemtype)))
#define ARR_DIMS(a)		/* !!may be unaligned!! */		\
	((int *) (((char *) (a)) + sizeof(ArrayType)))
#define ARR_LBOUND(a)	/* !!may be unaligned!! */		\
	((int *) (((char *) (a)) + sizeof(ArrayType) +		\
			  sizeof(int) * ARR_NDIM(a)))
#define ARR_NULLBITMAP(a)									  \
	(ARR_HASNULL(a)											  \
	 ? (cl_char *) (((char *) (a)) + sizeof(ArrayType) +	  \
					2 * sizeof(int) * ARR_NDIM(a))			  \
	 : (cl_char *) NULL)


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
pg_array_seek(char *dataptr, cl_char *nullbitmap, cl_int nitems,
			  cl_short typlen, cl_char typbyval, cl_char typalign)
{
	int			i;

	if (!nullbitmap)
	{
		/* easy if fixed-size elements and no NULLs */
		if (typlen > 0)
			return dataptr + ((size_t)TYPEALIGN(typalign, typlen)) * nitems;

		/* walk on the array of variable length datum */
		for (i = 0; i < nitems; i++)
		{
			dataptr += VARSIZE_ANY(dataptr);
			dataptr = (char *) TYPEALIGN(typalign, dataptr);
		}
	}
	else
	{
		cl_uint		bitmask = 1;
		cl_uint		nullmask = *nullbitmap++;

		for (i = 0; i < nitems; i++)
		{
			if (nullmask & bitmask)
			{
				dataptr += (typlen > 0 ? typlen : VARSIZE_ANY(dataptr));
				dataptr = (char *) TYPEALIGN(dataptr, typalign);
			}
			bitmask <<= 1;
			if (bitmask == 0x100)
			{
				nullmask = *nullbitmap++;
				bitmask = 1;
			}
		}

	}
	return dataptr;
}


INLINE_FUNCTION(void *)
pg_array_get_element_fast()
{
	/* if not-null, 1D, fixed-length array */


}


STATIC_FUNCTION(void *)
pg_arrayref_get_element(kern_context *kcxt, pg_array_t array,
						int nscripts, cl_int *index)

{
	cl_int		i, ndim;
	cl_int	   *dim;
	cl_int	   *lb;
	size_t		offset;
	size_t		scale;
	cl_char	   *nullbitmap;

	if (array.isnull)
		return NULL;

	ndim = ARR_NDIM(array.value);
	if (ndim != nscripts || ndim <= 0 || ndim > MAXDIM)
		return NULL;

	dim = ARR_DIMS(array.value);
	lb = ARR_LBOUND(array.value);
	nullbitmap = ARR_NULLBITMAP(array.value);

	for (i = 0; i < ndim; i++)
	{
		if (index[i] < lb[i] || index[i] >= (dim[i] + lb[i]))
			return NULL;
	}


	/* not null, fixed length, 1D then fast path */
	if (ndim == 1 && !ARR_HASNULL(array.value) && element_type_len > 0)
	{
		// do fast path
	}

	/*
	 * calculate the element number (ArrayGetOffset in the host code)
	 */
	offset = 0;
	for (i = ndim - 1; i >= 0; i--)
	{
		offset += (index[i] - lb[i]) * scale;
		scale *= dim[i];
	}

	/*
	 * Check for NULL array element (array_get_isnull in the host code)
	 */
	if (nullbitmap && att_isnull(nullbitmap, offset))
		return NULL;

	return pg_array_seek();
}





STATIC_FUNCTION(void *)
pg_array_reference_1d(kern_context *kcxt, pg_array_t *array,
					  pg_int4_t index_1)
{
	if (aindex_1.isnull)
		return NULL;
	// NULL, if index is null




}

STATIC_FUNCTION(void *)
pg_array_reference_2d(kern_context *kcxt, pg_array_t *array,
					  cl_uint index_x, cl_uint index_y)
{}





pg_arrayref_int4(pg_int4_array_t)
{}




#endif
