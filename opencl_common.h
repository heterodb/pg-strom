/*
 * opencl_common.h
 *
 * A common header for OpenCL device code
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef OPENCL_COMMON_H
#define OPENCL_COMMON_H

/*
 * OpenCL background server always adds -DOPENCL_DEVICE_CODE on kernel build,
 * but not for the host code, so this #if ... #endif block is available only
 * OpenCL device code.
 */
#ifdef OPENCL_DEVICE_CODE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/* NULL definition */
#define NULL	((void *) 0UL)

/* basic type definitions */
typedef bool		cl_bool;
typedef char		cl_char;
typedef uchar		cl_uchar;
typedef short		cl_short;
typedef ushort		cl_ushort;
typedef int			cl_int;
typedef uint		cl_uint;
typedef long		cl_long;
typedef ulong		cl_ulong;
typedef float		cl_float;
typedef double		cl_double;

/* varlena related stuff */
typedef struct {
	int		vl_len;
	char	vl_dat[1];
} varlena;
#define VARHDRSZ			((int) sizeof(cl_int))
#define VARDATA(PTR)		(((varlena *)(PTR))->vl_dat)
#define VARSIZE(PTR)		(((varlena *)(PTR))->vl_len)
#define VARSIZE_EXHDR(PTR)	(VARSIZE(PTR) - VARHDRSZ)

/* template for native types */
#define STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)	\
	typedef struct {								\
		BASE	value;								\
		bool	isnull;								\
	} pg_##NAME##_t

#define STROMCL_VARLENA_DATATYPE_TEMPLACE(NAME)		\
	STROMCL_SIMPLE_TYPE_TEMPLATE(NAME, __global varlena *)

#define STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)	\
	static pg_##NAME##_t pg_##NAME##_vref(...)		\
	{												\
													\
													\
													\
													\
													\
													\
													\
													\
													\
													\
													\
													\
													\
													\
	}

#define STROMCL_VARLENA_VARREF_TEMPLATE(NAME)	\
	static pg_##NAME##_t pg_##NAME##_vref(...)	\
	{											\
												\
												\
												\
	}

#define STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)	\
	static pg_##NAME##_t pg_##NAME##_pref(...)		\
	{												\
													\
													\
	}

#define STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)	\
	static pg_##NAME##_t pg_##NAME##_pref(...)	\
	{											\
												\
												\
												\
	}

#define STROMCL_SIMPLE_TYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)

#define STROMCL_VRALENA_TYPE_TEMPLATE(NAME)		\
	STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)		\
	STROMCL_VARLENA_VARREF_TEMPLATE(NAME)		\
	STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)

#endif



/*
 * Simplified FormData_pg_attribute; that shall be referenced in OpenCL
 * device to understand the data format given by host.
 */
typedef struct {
	cl_uint		attrelid;
	cl_uint		atttypid;
	cl_short	attlen
	cl_short	attnum;
	cl_int		attndims;
	cl_int		attcacheoff;
	cl_int		atttypmod;
	cl_bool		attbyval;
	cl_char		attalign;
	cl_bool		attnotnull;
} SimpleFormData_pg_attribute;

typedef SimpleFormData_pg_attribute *SimpleForm_pg_attribute;


#endif	/* OPENCL_COMMON_H */
