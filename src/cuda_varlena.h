/*
 * cuda_varlena.h
 *
 * Definitions and routines for PostgreSQL variable length data types
 * --
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
#ifndef CUDA_VARLENA_H
#define CUDA_VARLENA_H

/*
 * Unlike host code, device code cannot touch external and/or compressed
 * toast datum. All the format device code can understand is usual
 * in-memory form; 4-bytes length is put on the head and contents follows.
 * So, it is a responsibility of host code to decompress the toast values
 * if device code may access compressed varlena.
 * In case when device code touches unsupported format, calculation result
 * shall be postponed to calculate on the host side.
 *
 * Note that it is harmless to have external and/or compressed toast datam
 * unless it is NOT referenced in the device code. It can understand the
 * length of these values, unlike contents.
 */
typedef struct varlena		varlena;
#ifndef POSTGRES_H
struct varlena {
	cl_char		vl_len_[4];	/* Do not touch this field directly! */
	cl_char		vl_dat[1];
};

#define VARHDRSZ			((int) sizeof(cl_int))
#define VARDATA(PTR)		VARDATA_4B(PTR)
#define VARSIZE(PTR)		VARSIZE_4B(PTR)
#define VARSIZE_EXHDR(PTR)	(VARSIZE(PTR) - VARHDRSZ)

#define VARSIZE_SHORT(PTR)	VARSIZE_1B(PTR)
#define VARDATA_SHORT(PTR)	VARDATA_1B(PTR)

typedef union
{
	struct						/* Normal varlena (4-byte length) */
	{
		cl_uint		va_header;
		cl_char		va_data[1];
    }		va_4byte;
	struct						/* Compressed-in-line format */
	{
		cl_uint		va_header;
		cl_uint		va_rawsize;	/* Original data size (excludes header) */
		cl_char		va_data[1];	/* Compressed data */
	}		va_compressed;
} varattrib_4b;

typedef struct
{
	cl_uchar	va_header;
	cl_char		va_data[1];		/* Data begins here */
} varattrib_1b;

/* inline portion of a short varlena pointing to an external resource */
typedef struct
{
	cl_uchar    va_header;		/* Always 0x80 or 0x01 */
	cl_uchar	va_tag;			/* Type of datum */
	cl_char		va_data[1];		/* Data (of the type indicated by va_tag) */
} varattrib_1b_e;

typedef enum vartag_external
{
	VARTAG_INDIRECT = 1,
	VARTAG_ONDISK = 18
} vartag_external;

#define VARHDRSZ_SHORT			offsetof(varattrib_1b, va_data)
#define VARATT_SHORT_MAX		0x7F

typedef struct varatt_external
{
	cl_int		va_rawsize;		/* Original data size (includes header) */
	cl_int		va_extsize;		/* External saved size (doesn't) */
	cl_int		va_valueid;		/* Unique ID of value within TOAST table */
	cl_int		va_toastrelid;	/* RelID of TOAST table containing it */
} varatt_external;

typedef struct varatt_indirect
{
	hostptr_t	pointer;	/* Host pointer to in-memory varlena */
} varatt_indirect;

#define VARTAG_SIZE(tag) \
	((tag) == VARTAG_INDIRECT ? sizeof(varatt_indirect) :	\
	 (tag) == VARTAG_ONDISK ? sizeof(varatt_external) :		\
	 0 /* should not happen */)

#define VARHDRSZ_EXTERNAL		offsetof(varattrib_1b_e, va_data)
#define VARTAG_EXTERNAL(PTR)	VARTAG_1B_E(PTR)
#define VARSIZE_EXTERNAL(PTR)	\
	(VARHDRSZ_EXTERNAL + VARTAG_SIZE(VARTAG_EXTERNAL(PTR)))

/*
 * compressed varlena format
 */
typedef struct toast_compress_header
{
	cl_int		vl_len_;	/* varlena header (do not touch directly!) */
	cl_int		rawsize;
} toast_compress_header;

#define TOAST_COMPRESS_HDRSZ		((cl_int)sizeof(toast_compress_header))
#define TOAST_COMPRESS_RAWSIZE(ptr)				\
	(((toast_compress_header *) (ptr))->rawsize)
#define TOAST_COMPRESS_RAWDATA(ptr)				\
	(((char *) (ptr)) + TOAST_COMPRESS_HDRSZ)
#define TOAST_COMPRESS_SET_RAWSIZE(ptr, len)	\
	(((toast_compress_header *) (ptr))->rawsize = (len))

/* basic varlena macros */
#define VARATT_IS_4B(PTR) \
	((((varattrib_1b *) (PTR))->va_header & 0x01) == 0x00)
#define VARATT_IS_4B_U(PTR) \
	((((varattrib_1b *) (PTR))->va_header & 0x03) == 0x00)
#define VARATT_IS_4B_C(PTR) \
	((((varattrib_1b *) (PTR))->va_header & 0x03) == 0x02)
#define VARATT_IS_1B(PTR) \
	((((varattrib_1b *) (PTR))->va_header & 0x01) == 0x01)
#define VARATT_IS_1B_E(PTR) \
	((((varattrib_1b *) (PTR))->va_header) == 0x01)
#define VARATT_IS_COMPRESSED(PTR)		VARATT_IS_4B_C(PTR)
#define VARATT_IS_EXTERNAL(PTR)			VARATT_IS_1B_E(PTR)
#define VARATT_IS_EXTERNAL_ONDISK(PTR)		\
	(VARATT_IS_EXTERNAL(PTR) && VARTAG_EXTERNAL(PTR) == VARTAG_ONDISK)
#define VARATT_IS_EXTERNAL_INDIRECT(PTR)	\
	(VARATT_IS_EXTERNAL(PTR) && VARTAG_EXTERNAL(PTR) == VARTAG_INDIRECT)
#define VARATT_IS_SHORT(PTR)			VARATT_IS_1B(PTR)
#define VARATT_IS_EXTENDED(PTR)			(!VARATT_IS_4B_U(PTR))
#define VARATT_NOT_PAD_BYTE(PTR) 		(*((cl_uchar *) (PTR)) != 0)

#define VARSIZE_4B(PTR)						\
	((__Fetch(&((varattrib_4b *)(PTR))->va_4byte.va_header)>>2) & 0x3FFFFFFF)
#define VARSIZE_1B(PTR) \
	((((varattrib_1b *) (PTR))->va_header >> 1) & 0x7F)
#define VARTAG_1B_E(PTR) \
	(((varattrib_1b_e *) (PTR))->va_tag)

#define VARRAWSIZE_4B_C(PTR)	\
	__Fetch(&((varattrib_4b *) (PTR))->va_compressed.va_rawsize)

#define VARSIZE_ANY_EXHDR(PTR) \
	(VARATT_IS_1B_E(PTR) ? VARSIZE_EXTERNAL(PTR)-VARHDRSZ_EXTERNAL : \
	 (VARATT_IS_1B(PTR) ? VARSIZE_1B(PTR)-VARHDRSZ_SHORT :			 \
	  VARSIZE_4B(PTR)-VARHDRSZ))

#define VARSIZE_ANY(PTR)							\
	(VARATT_IS_1B_E(PTR) ? VARSIZE_EXTERNAL(PTR) :	\
	 (VARATT_IS_1B(PTR) ? VARSIZE_1B(PTR) :			\
	  VARSIZE_4B(PTR)))

#define VARDATA_4B(PTR)	(((varattrib_4b *) (PTR))->va_4byte.va_data)
#define VARDATA_1B(PTR)	(((varattrib_1b *) (PTR))->va_data)
#define VARDATA_ANY(PTR) \
	(VARATT_IS_1B(PTR) ? VARDATA_1B(PTR) : VARDATA_4B(PTR))

#define SET_VARSIZE(PTR, len)						\
	(((varattrib_4b *)(PTR))->va_4byte.va_header = (((cl_uint) (len)) << 2))
#endif	/* POSTGRES_H */

/* ------------------------------------------------------------------
 *
 * Definitions for PostgreSQL's array structure
 *
 *    Definitions in this block are valid only if utils/array.h is not
 *    included yet.
 *
 * ------------------------------------------------------------------ */
#ifndef ARRAY_H
typedef struct
{
	/*
	 * NOTE: We assume 4bytes varlena header for array type. It allows
	 * aligned references to the array elements. Unlike CPU side, we
	 * cannot have extra malloc to ensure 4bytes varlena header. It is
	 * the reason why our ScalarArrayOp implementation does not support
	 * array data type referenced by Var node; which is potentially has
	 * short format.
	 */
	cl_uint		vl_len_;		/* don't touch this field */
	cl_int		ndim;			/* # of dimensions */
	cl_int		dataoffset;		/* offset to data, or 0 if no bitmap */
	cl_uint		elemtype;		/* element type OID */
} ArrayType;

#define MAXDIM			6

#define ARR_SIZE(a)		VARSIZE_ANY(a)
#define ARR_NDIM(a)		(((ArrayType *)(a))->ndim)
#define ARR_HASNULL(a)	(((ArrayType *)(a))->dataoffset != 0)
#define ARR_ELEMTYPE(a)	(((ArrayType *)(a))->elemtype)
#define ARR_DIMS(a)									\
	((int *) (((char *) (a)) + sizeof(ArrayType)))
#define ARR_LBOUND(a)								\
	((int *) (((char *) (a)) + sizeof(ArrayType) +	\
			  sizeof(int) * ARR_NDIM(a)))
#define ARR_NULLBITMAP(a)							\
	(ARR_HASNULL(a)									\
	 ? (((char *) (a)) + sizeof(ArrayType) +		\
		2 * sizeof(int) * ARR_NDIM(a))				\
	 : (char *) NULL)
/*
 * The total array header size (in bytes) for an array with the specified
 * number of dimensions and total number of items.
 */
#define ARR_OVERHEAD_NONULLS(ndims)					\
	MAXALIGN(sizeof(ArrayType) + 2 * sizeof(int) * (ndims))
#define ARR_OVERHEAD_WITHNULLS(ndims, nitems)		\
	MAXALIGN(sizeof(ArrayType) + 2 * sizeof(int) * (ndims) +	\
			 ((nitems) + 7) / 8)
/*
 * Returns a pointer to the actual array data.
 */
#define ARR_DATA_OFFSET(a)					\
	(ARR_HASNULL(a)							\
	 ? ((ArrayType *)(a))->dataoffset		\
	 : ARR_OVERHEAD_NONULLS(ARR_NDIM(a)))

#define ARR_DATA_PTR(a)		(((char *) (a)) + ARR_DATA_OFFSET(a))
#endif	/* ARRAY_H */

/*
 * Template of variable length data type on device
 */
#define STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)					\
	typedef struct {											\
		char	   *value;										\
		cl_bool		isnull;										\
		cl_int		length;		/* -1, if PG varlena */			\
	} pg_##NAME##_t;

#ifdef __CUDACC__
#define STROMCL_VARLENA_VARREF_TEMPLATE(NAME)					\
	STATIC_INLINE(pg_##NAME##_t)								\
	pg_##NAME##_datum_ref(kern_context *kcxt,					\
						  void *addr)							\
	{															\
		pg_##NAME##_t result;									\
																\
		if (!addr)												\
			result.isnull = true;								\
		else													\
		{														\
			result.isnull = false;								\
			result.length = -1;									\
			result.value = (char *)addr;						\
		}														\
		return result;											\
	}															\
	STATIC_INLINE(void)											\
	pg_datum_ref(kern_context *kcxt,							\
				 pg_##NAME##_t &result, void *addr)				\
	{															\
		result = pg_##NAME##_datum_ref(kcxt, addr);				\
	}															\
	STATIC_INLINE(void)											\
	pg_datum_ref_slot(kern_context *kcxt,						\
					  pg_##NAME##_t &result,					\
					  cl_char dclass, Datum datum)				\
	{															\
		if (dclass == DATUM_CLASS__NULL)						\
			result = pg_##NAME##_datum_ref(kcxt, NULL);			\
		else if (dclass == DATUM_CLASS__VARLENA)				\
			memcpy(&result, DatumGetPointer(datum), sizeof(result));	\
		else													\
		{														\
			assert(dclass == DATUM_CLASS__NORMAL);				\
			result = pg_##NAME##_datum_ref(kcxt, (char *)datum); \
		}														\
	}															\
	STATIC_INLINE(cl_int)										\
	pg_datum_store(kern_context *kcxt,							\
				   pg_##NAME##_t datum,							\
				   cl_char &dclass,								\
				   Datum &value)								\
	{															\
		if (datum.isnull)										\
			dclass = DATUM_CLASS__NULL;							\
		else if (datum.length < 0)								\
		{														\
			cl_uint		len = VARSIZE_ANY(datum.value);			\
																\
			dclass = DATUM_CLASS__NORMAL;						\
			value  = PointerGetDatum(datum.value);				\
			if (PTR_ON_VLBUF(kcxt, datum.value, len))			\
				return len;										\
		}														\
		else													\
		{														\
			pg_##NAME##_t  *vl_buf;								\
																\
			vl_buf = (pg_##NAME##_t *)							\
				kern_context_alloc(kcxt, sizeof(pg_##NAME##_t));\
			if (vl_buf)											\
			{													\
				memcpy(vl_buf, &datum, sizeof(pg_##NAME##_t));	\
				dclass = DATUM_CLASS__VARLENA;					\
				value  = PointerGetDatum(vl_buf);				\
				return sizeof(pg_##NAME##_t);					\
			}													\
			STROM_SET_ERROR(&kcxt->e, StromError_OutOfMemory);	\
			dclass = DATUM_CLASS__NULL;							\
		}														\
		return 0;												\
	}															\
	STATIC_INLINE(pg_##NAME##_t)								\
	pg_##NAME##_param(kern_context *kcxt, cl_uint param_id)		\
	{															\
		kern_parambuf  *kparams = kcxt->kparams;				\
		pg_##NAME##_t	result;									\
																\
		if (param_id < kparams->nparams &&						\
			kparams->poffset[param_id] > 0)						\
		{														\
			char	   *vl_val = ((char *)kparams +				\
								  kparams->poffset[param_id]);	\
			if (VARATT_IS_4B_U(vl_val) || VARATT_IS_1B(vl_val))	\
			{													\
				result.value = vl_val;							\
				result.length = -1;								\
				result.isnull = false;							\
			}													\
			else												\
			{													\
				result.isnull = true;							\
				STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck); \
			}													\
		}														\
		else													\
			result.isnull = true;								\
																\
		return result;											\
	}
#else	/* __CUDACC__ */
#define	STROMCL_VARLENA_VARREF_TEMPLATE(NAME)
#endif	/* __CUDACC__ */

#ifdef __CUDACC__
#define STROMCL_VARLENA_COMP_HASH_TEMPLATE(NAME)				\
	STATIC_INLINE(cl_uint)										\
	pg_comp_hash(kern_context *kcxt, pg_##NAME##_t datum)		\
	{                                                           \
		if (datum.isnull)										\
			return 0;											\
		if (datum.length >= 0)									\
			return pg_hash_any((cl_uchar *)datum.value,			\
							   datum.length);					\
		if (VARATT_IS_COMPRESSED(datum.value) ||                \
			VARATT_IS_EXTERNAL(datum.value))					\
		{														\
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);	\
			return 0;											\
		}														\
		return pg_hash_any((cl_uchar *)VARDATA_ANY(datum.value), \
						   VARSIZE_ANY_EXHDR(datum.value));		\
	}

#else	/* __CUDACC__ */
#define	STROMCL_VARLENA_COMP_HASH_TEMPLATE(NAME)
#endif	/* __CUDACC__ */

#ifdef __CUDACC__
#define STROMCL_VARLENA_ARROW_TEMPLATE(NAME)				\
	STATIC_INLINE(void)										\
	pg_datum_fetch_arrow(kern_context *kcxt,				\
						 pg_##NAME##_t &result,             \
						 kern_colmeta *cmeta,				\
						 char *base, cl_uint rowidx)		\
	{														\
		void           *addr;                               \
		cl_uint			length;								\
                                                            \
		addr = kern_fetch_varlena_datum_arrow(cmeta,		\
											  base,			\
											  rowidx,		\
											  &length);		\
		if (!addr)											\
		{													\
			result.isnull = true;							\
			return;											\
		}													\
		result.isnull = false;								\
		result.value  = (char *)addr;						\
		result.length = length;								\
	}

#define STROMCL_VARLENA_PGARRAY_TEMPLATE(NAME)				\
	STATIC_INLINE(cl_uint)									\
	pg_##NAME##_array_from_arrow(kern_context *kcxt,		\
								 char *dest,				\
								 kern_colmeta *cmeta,		\
								 char *base,				\
								 cl_uint start,				\
								 cl_uint end)				\
	{														\
		return pg_varlena_array_from_arrow<pg_##NAME##_t>	\
					(kcxt, dest, cmeta, base, start, end);	\
	}

/*
 * Generic interface of pg_XXXX_array_from_arrow
 */
template <typename T>
DEVICE_ONLY_INLINE(cl_uint)
pg_varlena_array_from_arrow(kern_context *kcxt,
							char *dest,
							kern_colmeta *cmeta,
							char *base,
							cl_uint start, cl_uint end)
{
	ArrayType  *res = (ArrayType *)dest;
	cl_uint		nitems = end - start;
	cl_uint		i, sz;
	char	   *nullmap = NULL;
	T			temp;

	Assert((cl_ulong)res == MAXALIGN(res));
	Assert(start <= end);
	if (cmeta->nullmap_offset == 0)
		sz = ARR_OVERHEAD_NONULLS(1);
	else
		sz = ARR_OVERHEAD_WITHNULLS(1, nitems);

	if (res)
	{
		res->ndim = 1;
		res->dataoffset = (cmeta->nullmap_offset == 0 ? 0 : sz);
		res->elemtype = cmeta->atttypid;
		ARR_DIMS(res)[0] = nitems;
		ARR_LBOUND(res)[0] = 1;

		nullmap = ARR_NULLBITMAP(res);
		Assert(dest + sz == ARR_DATA_PTR(res));
	}

	for (i=0; i < nitems; i++)
	{
		pg_datum_fetch_arrow(kcxt, temp, cmeta, base, start+i);
		if (temp.isnull)
		{
			if (nullmap)
				nullmap[i>>3] &= ~(1<<(i&7));
			else
				Assert(!dest);
		}
		else
		{
			if (nullmap)
				nullmap[i>>3] |= (1<<(i&7));

			sz = TYPEALIGN(cmeta->attalign, sz);
			if (temp.length < 0)
			{
				cl_uint		vl_len = VARSIZE_ANY(temp.value);

				if (dest)
					memcpy(dest + sz, DatumGetPointer(temp.value), vl_len);
				sz += vl_len;
			}
			else
			{
				if (dest)
				{
					memcpy(dest + sz + VARHDRSZ, temp.value, temp.length);
					SET_VARSIZE(dest + sz, VARHDRSZ + temp.length);
				}
				sz += VARHDRSZ + temp.length;
			}
		}
	}
	return sz;
}

#else
#define STROMCL_VARLENA_ARROW_TEMPLATE(NAME)
#define STROMCL_VARLENA_PGARRAY_TEMPLATE(NAME)
#endif

#define STROMCL_VARLENA_TYPE_TEMPLATE(NAME)					\
	STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)					\
	STROMCL_VARLENA_VARREF_TEMPLATE(NAME)					\
	STROMCL_VARLENA_COMP_HASH_TEMPLATE(NAME)

/* generic varlena */
#ifndef PG_VARLENA_TYPE_DEFINED
#define PG_VARLENA_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(varlena)
#endif	/* PG_VARLENA_TYPE_DEFINED */

/* pg_bytea_t */
#ifndef PG_BYTEA_TYPE_DEFINED
#define PG_BYTEA_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(bytea)
STROMCL_VARLENA_ARROW_TEMPLATE(bytea)
STROMCL_VARLENA_PGARRAY_TEMPLATE(bytea)
#endif	/* PG_BYTEA_TYPE_DEFINED */

#ifdef __CUDACC__
/*
 * for DATUM_CLASS__VARLENA handler
 */
STATIC_FUNCTION(cl_uint)
pg_varlena_datum_length(kern_context *kcxt, Datum datum)
{
	pg_varlena_t   *vl = (pg_varlena_t *) datum;

	if (vl->length < 0)
		return VARSIZE_ANY(vl->value);
	return VARHDRSZ + vl->length;
}

STATIC_FUNCTION(cl_uint)
pg_varlena_datum_write(kern_context *kcxt, char *dest, Datum datum)
{
	pg_varlena_t   *vl = (pg_varlena_t *) datum;
	cl_uint			vl_len;

	if (vl->length < 0)
	{
		vl_len = VARSIZE_ANY(vl->value);
		memcpy(dest, vl->value, vl_len);
	}
	else
	{
		vl_len = VARHDRSZ + vl->length;
		memcpy(dest + VARHDRSZ, vl->value, vl->length);
		SET_VARSIZE(dest, vl_len);
	}
	return vl_len;
}
#endif	/* __CUDACC__ */

#ifdef __CUDACC__
/*
 * for DATUM_CLASS__(ARRAY|COMPOSITE) handler (cuda_anytype.h)
 */
#ifdef PGSTROM_KERNEL_HAS_PGARRAY
STATIC_INLINE(cl_uint) pg_array_datum_length(kern_context *kcxt,
											 Datum datum);
STATIC_INLINE(cl_uint) pg_array_datum_write(kern_context *kcxt,
											char *dest, Datum datum);
#endif	/* PGSTROM_KERNEL_HAS_PGARRAY */
#ifdef PGSTROM_KERNEL_HAS_PGCOMPOSITE
STATIC_INLINE(cl_uint) pg_composite_datum_length(kern_context *kcxt,
												 Datum datum);
STATIC_INLINE(cl_uint) pg_composite_datum_write(kern_context *kcxt,
												char *dest, Datum datum);
#endif	/* PGSTROM_KERNEL_HAS_PGCOMPOSITE */
#endif	/* __CUDACC__ */

#ifdef __CUDACC__
template <typename T>
DEVICE_ONLY_INLINE(cl_bool)
pg_varlena_datum_extract(kern_context *kcxt, T &arg,
						 char **s, cl_int *len)
{
	if (arg.isnull)
		return false;
	if (arg.length < 0)
	{
		if (VARATT_IS_COMPRESSED(arg.value) ||
			VARATT_IS_EXTERNAL(arg.value))
        {
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			return false;
        }
		*s = VARDATA_ANY(arg.value);
		*len = VARSIZE_ANY_EXHDR(arg.value);
	}
	else
	{
		*s = arg.value;
		*len = arg.length;
	}
	return true;
}
#endif	/* __CUDACC__ */

#ifdef __CUDACC__
DEVICE_FUNCTION(size_t)
toast_raw_datum_size(kern_context *kcxt, varlena *attr);
DEVICE_FUNCTION(cl_int)
pglz_decompress(const char *source, cl_int slen,
				char *dest, cl_int rawsize);
DEVICE_FUNCTION(cl_bool)
toast_decompress_datum(char *buffer, cl_uint buflen,
					   const varlena *datum);
#endif	/* __CUDACC__ */













#if 0

#ifndef PG_ARRAY_TYPE_DEFINED
#define PG_ARRAY_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(array)
#endif

STATIC_FUNCTION(cl_uint)
pg_array_datum_length(kern_context *kcxt, Datum datum)
{
	//XXX: to be revised to support List of Apache Arrow
	pg_array_t *array = (pg_array_t *) datum;

	return VARSIZE_ANY(array->value);
}

STATIC_FUNCTION(cl_uint)
pg_array_datum_write(char *dest, Datum datum)
{
	return pg_varlena_datum_write(dest, datum);
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_VARLENA_H */
