/*
 * xpu_common.h
 *
 * Common header portion for both of GPU and DPU device code
 * --
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_COMMON_H
#define XPU_COMMON_H
#include <alloca.h>
#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>

/*
 * Functions with qualifiers
 */
#ifndef PGDLLEXPORT
#define PGDLLEXPORT
#endif

#if defined(__CUDACC__)
/* CUDA C++ */
#define INLINE_FUNCTION(RET_TYPE)				\
	__device__ __forceinline__					\
	static RET_TYPE __attribute__ ((unused))
#define STATIC_FUNCTION(RET_TYPE)		__device__ static RET_TYPE
#define PUBLIC_FUNCTION(RET_TYPE)		__device__ RET_TYPE
#define EXTERN_FUNCTION(RET_TYPE)		extern "C" __device__ RET_TYPE
#define KERNEL_FUNCTION(RET_TYPE)		extern "C" __global__ RET_TYPE
#define EXTERN_DATA(TYPE,NAME)			extern "C" __device__ TYPE NAME
#define EXTERN_SHARED_DATA(TYPE,NAME)	extern "C" __shared__ TYPE NAME
#define PUBLIC_DATA(TYPE,NAME)			__device__ TYPE NAME
#define PUBLIC_SHARED_DATA(TYPE,NAME)	__shared__ TYPE NAME
#define STATIC_DATA(TYPE,NAME)			static __device__ TYPE NAME
#define STATIC_SHARED_DATA(TYPE,NAME)	static __shared__ TYPE NAME
#elif defined(__cplusplus)
/* C++ */
#include <cstdio>						/* for printf in C++ */
#define INLINE_FUNCTION(RET_TYPE)		static inline RET_TYPE
#define STATIC_FUNCTION(RET_TYPE)		static RET_TYPE
#define PUBLIC_FUNCTION(RET_TYPE)		PGDLLEXPORT RET_TYPE
#define KERNEL_FUNCTION(RET_TYPE)		extern "C" RET_TYPE
#define EXTERN_FUNCTION(RET_TYPE)		extern "C" RET_TYPE
#define EXTERN_DATA(TYPE,NAME)			extern "C" TYPE NAME
#define EXTERN_SHARED_DATA(TYPE,NAME)	extern "C" TYPE NAME
#define PUBLIC_DATA(TYPE,NAME)			TYPE NAME
#define PUBLIC_SHARED_DATA(TYPE,NAME)	PUBLIC_DATA(TYPE,NAME)
#define STATIC_DATA(TYPE,NAME)			static TYPE NAME
#define STATIC_SHARED_DATA(TYPE,NAME)	STATIC_DATA(TYPE,NAME)
#else
/* C */
#define INLINE_FUNCTION(RET_TYPE)		static inline RET_TYPE
#define STATIC_FUNCTION(RET_TYPE)		static RET_TYPE
#define PUBLIC_FUNCTION(RET_TYPE)		PGDLLEXPORT RET_TYPE
#define KERNEL_FUNCTION(RET_TYPE)		RET_TYPE
#define EXTERN_FUNCTION(RET_TYPE)		extern RET_TYPE
#define EXTERN_DATA(TYPE,NAME)			extern TYPE NAME
#define EXTERN_SHARED_DATA(TYPE,NAME)	EXTERN_DATA(TYPE,NAME)
#define PUBLIC_DATA(TYPE,NAME)			TYPE NAME
#define PUBLIC_SHARED_DATA(TYPE,NAME)	PUBLIC_DATA(TYPE,NAME)
#define STATIC_DATA(TYPE,NAME)			static TYPE NAME
#define STATIC_SHARED_DATA(TYPE,NAME)	STATIC_DATA(TYPE,NAME)
#endif	/* __CUDACC__ */

/*
 * Limitation of types
 */
#ifndef SCHAR_MAX
#define SCHAR_MAX       127
#endif
#ifndef SCHAR_MIN
#define SCHAR_MIN       (-128)
#endif
#ifndef UCHAR_MAX
#define UCHAR_MAX       255
#endif
#ifndef SHRT_MAX
#define SHRT_MAX        32767
#endif
#ifndef SHRT_MIN
#define SHRT_MIN        (-32767-1)
#endif
#ifndef USHRT_MAX
#define USHRT_MAX       65535
#endif
#ifndef INT_MAX
#define INT_MAX         2147483647
#endif
#ifndef INT_MIN
#define INT_MIN         (-INT_MAX - 1)
#endif
#ifndef UINT_MAX
#define UINT_MAX        4294967295U
#endif
#ifndef LONG_MAX
#define LONG_MAX        0x7FFFFFFFFFFFFFFFLL
#endif
#ifndef LONG_MIN
#define LONG_MIN        (-LONG_MAX - 1LL)
#endif
#ifndef ULONG_MAX
#define ULONG_MAX       0xFFFFFFFFFFFFFFFFULL
#endif
#ifndef HALF_MAX
#define HALF_MAX        __short_as_half__(0x7bff)
#endif
#ifndef FLT_MAX
#define FLT_MAX         __int_as_float__(0x7f7fffffU)
#endif
#ifndef DBL_MAX
#define DBL_MAX         __longlong_as_double__(0x7fefffffffffffffULL)
#endif
#ifndef DBL_MIN
#define DBL_MIN			__longlong_as_double__(0x0010000000000000ULL)
#endif
#ifndef DBL_INF
#define DBL_INF			__longlong_as_double__(0x7ff0000000000000ULL)
#endif
#ifndef DBL_NAN
#define DBL_NAN			__longlong_as_double__(0x7fffffffffffffffULL)
#endif

#ifndef BITS_PER_BYTE
#define BITS_PER_BYTE	8
#endif
#ifndef SHRT_NBITS
#define SHRT_NBITS	(sizeof(int16_t) * BITS_PER_BYTE)
#endif
#ifndef INT_NBITS
#define INT_NBITS	(sizeof(int32_t) * BITS_PER_BYTE)
#endif
#ifndef LONG_NBITS
#define LONG_NBITS	(sizeof(int64_t) * BITS_PER_BYTE)
#endif

/*
 * Several fundamental data types and macros
 */
#ifndef Assert
#define Assert(cond)		assert(cond)
#endif
#ifndef Max
#define Max(a,b)			((a) > (b) ? (a) : (b))
#endif
#ifndef Min
#define Min(a,b)			((a) < (b) ? (a) : (b))
#endif
#ifndef Abs
#define Abs(x)				((x) >= 0 ? (x) : -(x))
#endif
#ifndef And
#define And(a,b)			((a) & (b))
#endif
#ifndef Or
#define Or(a,b)				((a) | (b))
#endif
#ifndef POSTGRES_H
typedef uint64_t			Datum;
typedef unsigned int		Oid;

#define NAMEDATALEN			64		/* must follow the host configuration */
#define BLCKSZ				8192	/* must follow the host configuration */

#ifndef lengthof
#define lengthof(array)		(sizeof (array) / sizeof ((array)[0]))
#endif
#define PointerGetDatum(X)	((Datum)(X))
#define DatumGetPointer(X)	((char *)(X))
#define TYPEALIGN(ALIGNVAL,LEN)         \
	(((uint64_t)(LEN) + ((ALIGNVAL) - 1)) & ~((uint64_t)((ALIGNVAL) - 1)))
#define TYPEALIGN_DOWN(ALIGNVAL,LEN)                        \
	(((uint64_t) (LEN)) & ~((uint64_t) ((ALIGNVAL) - 1)))
#define MAXIMUM_ALIGNOF		8
#define MAXALIGN(LEN)		TYPEALIGN(MAXIMUM_ALIGNOF,LEN)
#define MAXALIGN_DOWN(LEN)	TYPEALIGN_DOWN(MAXIMUM_ALIGNOF,LEN)
#define LONGALIGN(LEN)		TYPEALIGN(8,LEN)
#define INTALIGN(LEN)		TYPEALIGN(4,LEN)
#endif	/* POSTGRES_H */
#define __MAXALIGNED__		__attribute__((aligned(MAXIMUM_ALIGNOF)));
#define MAXIMUM_ALIGNOF_SHIFT 3

#ifndef HAS_GPUMASK_TYPEDEF
#define HAS_GPUMASK_TYPEDEF
#define INVALID_GPUMASK		(~0UL)
typedef int64_t				gpumask_t;
#endif	/* HAS_GPUMASK_TYPEDEF */

/* Definition of several primitive types */
typedef __int128	int128_t;
typedef struct
{
	uint64_t	u64_lo;
	uint64_t	u64_hi;
} int128_packed_t;

INLINE_FUNCTION(int128_t)
__fetch_int128_packed(const int128_packed_t *addr)
{
	return ((int128_t)addr->u64_hi << 64) | ((int128_t)addr->u64_lo);
}
INLINE_FUNCTION(void)
__store_int128_packed(int128_packed_t *addr, int128_t ival)
{
	addr->u64_lo = (uint64_t)(ival & ULONG_MAX);
	addr->u64_hi = (uint64_t)((ival >> 64) & ULONG_MAX);
}
#include "float2.h"

#ifndef __FILE_NAME__
INLINE_FUNCTION(const char *)
__runtime_file_name(const char *path)
{
	const char *s;

	for (s = path; *s != '\0'; s++)
	{
		if (*s == '/')
			path = s + 1;
	}
	return path;
}
#define __FILE_NAME__	__runtime_file_name(__FILE__)
#endif

#ifdef __cplusplus
template <typename T>
INLINE_FUNCTION(T)
__Fetch(const T *ptr)
{
	T	temp;
#if 0
	/*
	 * MEMO: probably, nvcc expects the 'ptr' is aligned by the caller
	 * of function, therefore, compiler optimization might consider
	 * the following if-block is always true, and condition checks can
	 * be removed. However, __Fetch() is used to the address where we
	 * cannot guarantee the alignment (e.g, payload of short-varlena).
	 * So, we always have to use memcpy() for the safe memory access.
	 */
	if ((sizeof(T) & (sizeof(T)-1)) == 0 &&
		(((uintptr_t)ptr) & (sizeof(T)-1)) == 0)
	{
		return *ptr;
	}
#endif
	memcpy(&temp, ptr, sizeof(T));
	return temp;
}

template <typename T>
INLINE_FUNCTION(void)
__FetchStore(T &dest, const T *ptr)
{
	if ((sizeof(T) & (sizeof(T)-1)) == 0 &&
		(((uintptr_t)ptr) & (sizeof(T)-1)) == 0)
	{
		dest = *ptr;
	}
	else
	{
		memcpy(&dest, ptr, sizeof(T));
	}
}

template <typename T>
INLINE_FUNCTION(T)
__volatileRead(const volatile T *ptr)
{
	return *ptr;
}

#else	/* __cplusplus */
#define __Fetch(PTR)			(*(PTR))
#define __FetchStore(DEST,PTR)	do { (DEST) = *(PTR); } while(0)
#define __volatileRead(PTR)		(*(PTR))
#endif

INLINE_FUNCTION(int)
__memcmp(const void *__s1, const void *__s2, size_t n)
{
	const unsigned char	*s1 = (const unsigned char *)__s1;
	const unsigned char *s2 = (const unsigned char *)__s2;

	for (size_t i=0; i < n; i++)
	{
		if (s1[i] < s2[i])
			return -1;
		if (s1[i] > s2[i])
			return 1;
	}
	return 0;
}

/* ----------------------------------------------------------------
 *
 * Fundamental CUDA definitions
 *
 * ----------------------------------------------------------------
 */
#define WARPSIZE				32
#define CUDA_L1_CACHELINE_SZ	128

#if defined(__CUDACC__)
/* Thread index at CUDA C++ */
#define get_group_id()			(blockIdx.x)
#define get_num_groups()		(gridDim.x)
#define get_local_id()			(threadIdx.x)
#define get_local_size()		(blockDim.x)
#define get_global_id()			(blockDim.x * blockIdx.x + threadIdx.x)
#define get_global_base()		(blockDim.x * blockIdx.x)
#define get_global_size()		(blockDim.x * gridDim.x)

/* Dynamic shared memory entrypoint */
extern __shared__ char __pgstrom_dynamic_shared_workmem[] __MAXALIGNED__;
#define SHARED_WORKMEM(OFFSET)					\
	(__pgstrom_dynamic_shared_workmem + (OFFSET))

/* Reference to the special registers */
INLINE_FUNCTION(uint32_t) LaneId(void)
{
	uint32_t	rv;

	asm volatile("mov.u32 %0, %laneid;" : "=r"(rv) );

	return rv;
}

INLINE_FUNCTION(uint32_t) DynamicShmemSize(void)
{
	uint32_t	rv;

	asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(rv) );

	return rv;
}

INLINE_FUNCTION(uint32_t) TotalShmemSize(void)
{
	uint32_t	rv;

	asm volatile("mov.u32 %0, %total_smem_size;" : "=r"(rv) );

	return rv;
}
#endif		/* __CUDACC__ */

/*
 * Current GPU-Task specific run-time properties
 */
EXTERN_SHARED_DATA(uint32_t, stromTaskProp__cuda_dindex);
EXTERN_SHARED_DATA(uint32_t, stromTaskProp__cuda_stack_limit);
EXTERN_SHARED_DATA(int32_t,  stromTaskProp__partition_divisor);
EXTERN_SHARED_DATA(int32_t,  stromTaskProp__partition_reminder);

/*
 * TypeOpCode / FuncOpCode
 */
#define TYPE_OPCODE(NAME,a,b)		TypeOpCode__##NAME,
typedef enum {
	TypeOpCode__Invalid = 0,
#include "xpu_opcodes.h"
	TypeOpCode__composite,
	TypeOpCode__array,
	TypeOpCode__internal,
	TypeOpCode__BuiltInMax,
} TypeOpCode;

#define FUNC_OPCODE(a,b,c,NAME,d,e)			FuncOpCode__##NAME,
#define DEVONLY_FUNC_OPCODE(a,NAME,b,c,d)	FuncOpCode__##NAME,

#define WITH_DEVICE_ONLY_FUNCTIONS		1
typedef enum {
	FuncOpCode__Invalid = 0,
	FuncOpCode__ConstExpr,
	FuncOpCode__ParamExpr,
	FuncOpCode__VarExpr,
	FuncOpCode__BoolExpr_And,
	FuncOpCode__BoolExpr_Or,
	FuncOpCode__BoolExpr_Not,
	FuncOpCode__NullTestExpr_IsNull,
	FuncOpCode__NullTestExpr_IsNotNull,
	FuncOpCode__BoolTestExpr_IsTrue,
	FuncOpCode__BoolTestExpr_IsNotTrue,
	FuncOpCode__BoolTestExpr_IsFalse,
	FuncOpCode__BoolTestExpr_IsNotFalse,
	FuncOpCode__BoolTestExpr_IsUnknown,
	FuncOpCode__BoolTestExpr_IsNotUnknown,
	FuncOpCode__DistinctFrom,
	FuncOpCode__CoalesceExpr,
	FuncOpCode__LeastExpr,
	FuncOpCode__GreatestExpr,
	FuncOpCode__CaseWhenExpr,
	FuncOpCode__ScalarArrayOpAny,
	FuncOpCode__ScalarArrayOpAll,
#include "xpu_opcodes.h"
	FuncOpCode__LoadVars = 9999,
	FuncOpCode__MoveVars,
	FuncOpCode__JoinQuals,
	FuncOpCode__HashValue,
	FuncOpCode__GiSTEval,
	FuncOpCode__SaveExpr,
	FuncOpCode__AggFuncs,
	FuncOpCode__Projection,
	FuncOpCode__SortKeys,
	FuncOpCode__Packed,		/* place-holder for the stacked expressions */
	FuncOpCode__BuiltInMax,
} FuncOpCode;

/*
 * Error status
 */
#define ERRCODE_STROM_SUCCESS		0
#define ERRCODE_SUSPEND_FALLBACK	'f'		/* suspend by CPU fallback */
#define ERRCODE_SUSPEND_NO_SPACE	'd'		/* suspend by buffer no space */
#define ERRCODE_DEVICE_ERROR		'E'		/* generic device error */
#define ERRCODE_DEVICE_FATAL		'F'		/* generic device fatal error */
#define ERRCODE_IS_SUSPEND(x)		((x) >= 'a' && (x) <= 'z')

#define KERN_ERRORBUF_FILENAME_LEN		32
#define KERN_ERRORBUF_FUNCNAME_LEN		64
#define KERN_ERRORBUF_MESSAGE_LEN		200
typedef struct {
	uint32_t	errcode;	/* one of the ERRCODE_* */
	int32_t		lineno;
	char		filename[KERN_ERRORBUF_FILENAME_LEN+1];
	char		funcname[KERN_ERRORBUF_FUNCNAME_LEN+1];
	char		message[KERN_ERRORBUF_MESSAGE_LEN+1];
} kern_errorbuf;

/*
 * kern_context - a set of run-time information
 */
typedef struct
{
	uint32_t		errcode;
	const char	   *error_filename;
	uint32_t		error_lineno;
	const char	   *error_funcname;
	const char	   *error_message;
	struct kern_session_info *session;
	struct kern_data_store *kds_fallback;

	/* the kernel variables slot */
	struct xpu_datum_t **kvars_slot;
	const struct kern_varslot_desc *kvars_desc;
	uint32_t		kvars_nslots;		/* length of kvars_values / desc */
	uint32_t		kvars_nrooms;		/* length of kvars_desc (incl. subfields) */
	uint32_t		kvecs_bufsz;		/* kvecs-buffer size per depth */
	uint32_t		kvecs_ndims;		/* number of kvecs-buffer per warp */
	char		   *kvecs_curr_buffer;	/* current kvecs-buffer */
	uint32_t		kvecs_curr_id;		/* current kvecs-id */

	/*
	 * GPU shared memory buffer for GPU-PreAgg boosting.
	 */
	uint32_t		groupby_prepfn_bufsz;
	uint32_t		groupby_prepfn_nbufs;
	char		   *groupby_prepfn_buffer;

	/*
	 * mode control flags
	 *
	 * kmode_compare_nulls - if true, equal/not-equal operators compare NULLs.
	 */
	bool			kmode_compare_nulls;

	/* variable length buffer */
	char		   *vlpos;
	char		   *vlend;
	char			vlbuf[1];
} kern_context;

#define INIT_KERNEL_CONTEXT(KCXT,SESSION,KDS_FALLBACK)					\
	do {																\
		const kern_varslot_desc *__vs_desc;								\
		uint32_t	__bufsz = Max(512, (SESSION)->kcxt_extra_bufsz);	\
		uint32_t	__len = offsetof(kern_context, vlbuf) +	__bufsz;	\
																		\
		KCXT = (kern_context *)alloca(__len);							\
		memset(KCXT, 0, __len);											\
		KCXT->session = (SESSION);										\
		KCXT->kds_fallback = (KDS_FALLBACK);							\
		KCXT->kvars_nrooms = (SESSION)->kcxt_kvars_nrooms;				\
		KCXT->kvars_nslots = (SESSION)->kcxt_kvars_nslots;				\
		KCXT->kvecs_bufsz  = (SESSION)->kcxt_kvecs_bufsz;				\
		KCXT->kvecs_ndims  = (SESSION)->kcxt_kvecs_ndims;				\
		KCXT->kvecs_curr_buffer = NULL;									\
		KCXT->kvecs_curr_id = 0;										\
		KCXT->kvars_slot = (struct xpu_datum_t **)						\
			alloca(sizeof(struct xpu_datum_t *) * KCXT->kvars_nslots);	\
		__vs_desc = SESSION_KVARS_SLOT_DESC(SESSION);					\
		for (int __i=0; __i < KCXT->kvars_nslots; __i++)				\
		{																\
			const xpu_datum_operators *vs_ops =	__vs_desc[__i].vs_ops;	\
			/* alloca() guarantees 16bytes-aligned */					\
			assert(vs_ops->xpu_type_alignof <= 16);						\
			KCXT->kvars_slot[__i] = (struct xpu_datum_t *)				\
				alloca(vs_ops->xpu_type_sizeof);						\
			KCXT->kvars_slot[__i]->expr_ops = NULL;						\
		}													   			\
		KCXT->kvars_desc = __vs_desc;									\
		KCXT->vlpos = KCXT->vlbuf;										\
		KCXT->vlend = KCXT->vlbuf + __bufsz;							\
	} while(0)

INLINE_FUNCTION(const char *)
__basename(const char *filename)
{
	const char *pos;

	for (pos = filename; *pos != '\0'; pos++)
	{
		if (pos[0] == '/' && pos[1] != '\0')
			filename = pos + 1;
	}
	return filename;
}

INLINE_FUNCTION(void)
__STROM_EREPORT(kern_context *kcxt,
				uint32_t errcode,
				const char *filename,
				int lineno,
				const char *funcname,
				const char *message)
{
	/* in case when no significant errors are reported... */
	if (errcode != ERRCODE_STROM_SUCCESS)
	{
		switch (kcxt->errcode)
		{
			case ERRCODE_SUSPEND_FALLBACK:
			case ERRCODE_SUSPEND_NO_SPACE:
				if (ERRCODE_IS_SUSPEND(errcode))
					return;
			case ERRCODE_STROM_SUCCESS:
				kcxt->errcode = errcode;
				kcxt->error_filename = __basename(filename);
				kcxt->error_lineno   = lineno;
				kcxt->error_funcname = funcname;
				kcxt->error_message  = message;
				break;
			default:
				break;
		}
	}
}
#define STROM_ELOG(kcxt, message)								\
	__STROM_EREPORT((kcxt),ERRCODE_DEVICE_ERROR,				\
					__FILE__,__LINE__,__FUNCTION__,(message))
#define SUSPEND_FALLBACK(kcxt, message)							\
	__STROM_EREPORT((kcxt),ERRCODE_SUSPEND_FALLBACK,			\
					__FILE__,__LINE__,__FUNCTION__,(message))
#define SUSPEND_NO_SPACE(kcxt, message)							\
	__STROM_EREPORT((kcxt),ERRCODE_SUSPEND_NO_SPACE,			\
					__FILE__,__LINE__,__FUNCTION__,(message))

INLINE_FUNCTION(void *)
kcxt_alloc(kern_context *kcxt, size_t len)
{
	char   *pos = (char *)MAXALIGN(kcxt->vlpos);

	if (pos >= kcxt->vlbuf && pos + len <= kcxt->vlend)
	{
		kcxt->vlpos = pos + len;
		return pos;
	}
	STROM_ELOG(kcxt, "out of kcxt memory");
	return NULL;
}

INLINE_FUNCTION(void)
kcxt_reset(kern_context *kcxt)
{
	kcxt->vlpos = kcxt->vlbuf;
}

INLINE_FUNCTION(void)
__strncpy(char *d, const char *s, uint32_t n)
{
	uint32_t	i, m = n-1;

	for (i=0; i < m && s[i] != '\0'; i++)
		d[i] = s[i];
	while (i < n)
		d[i++] = '\0';
}

/* ----------------------------------------------------------------
 *
 * Definition related to stack-overflow checker
 *
 * ----------------------------------------------------------------
 */
INLINE_FUNCTION(bool)
CHECK_CUDA_STACK_OVERFLOW(void)
{
#if defined(__CUDACC__)
	uint32_t	sp;

	/*
	 * MEMO: Even though it is not documented well, the stacksave instruction
	 * returns a negative value in 24bit.
	 * So, in case of zero stack-usage, the stack-pointer shall be 0x01000000U.
	 */
	asm volatile("stacksave.u32 %0;" : "=r"(sp) );

	/* 256b margin for the stack boundary */
	return (sp + stromTaskProp__cuda_stack_limit < 0x00ffff00U);
#else
	return false;
#endif
}

/* ----------------------------------------------------------------
 *
 * Definitions related to the kernel data store
 *
 * ----------------------------------------------------------------
 */
#include "arrow_defs.h"

#define TYPE_KIND__NULL			'n'		/* unreferenced column */
#define TYPE_KIND__BASE			'b'
#define TYPE_KIND__ARRAY		'a'
#define TYPE_KIND__COMPOSITE	'c'
#define TYPE_KIND__DOMAIN		'd'
#define TYPE_KIND__ENUM			'e'
#define TYPE_KIND__PSEUDO		'p'
#define TYPE_KIND__RANGE		'r'

struct kern_colmeta {
	/* true, if column is held by value. Elsewhere, a reference */
	bool			attbyval;
	/* alignment; 1,2,4 or 8, not characters in pg_attribute */
	int8_t			attalign;
	/* length of attribute */
	int16_t			attlen;
	/* attribute number */
	int16_t			attnum;
	/* offset of attribute location, if deterministic */
	int16_t			attcacheoff;
	/* oid of the SQL data type */
	Oid				atttypid;
	/* typmod of the SQL data type */
	int32_t			atttypmod;
	/* one of TYPE_KIND__* */
	int8_t			atttypkind;
	/* copy of kds->format */
	char			kds_format;
	/* copy of sizeof(xpu_xxxx_t) if any */
	int16_t			dtype_sizeof;
	/*
	 * offset from kds for the reverse reference.
	 * kds = (kern_data_store *)((char *)cmeta - cmeta->kds_offset)
	 */
	uint32_t		kds_offset;
	/*
	 * (for array and composite types)
	 * Some of types contain sub-fields like array or composite type.
	 * We carry type definition information (kern_colmeta) using the
	 * kds->colmeta[] array next to the top-level fields.
	 * An array type has relevant element type. So, its @num_subattrs
	 * is always 1, and kds->colmeta[@idx_subattrs] informs properties
	 * of the element type.
	 * A composite type has several fields.
	 * kds->colmeta[@idx_subattrs ... @idx_subattrs + @num_subattrs -1]
	 * carries its sub-fields properties.
	 */
	uint16_t		idx_subattrs;
	uint16_t		num_subattrs;

	/* column name */
	char			attname[NAMEDATALEN];

	/*
	 * (only arrow/column format)
	 * @attoptions keeps extra information of Apache Arrow type. Unlike
	 * PostgreSQL types, it can have variation of data accuracy in time
	 * related data types, or precision in decimal data type.
	 *
	 * 'virtual_offset' is not zero if this is a virtual column.
	 * if negative, it means NULL. Elsewhere, it points contents of the
	 * virtual column in the format of PostgreSQL datum.
	 */
	ArrowTypeOptions attopts;
	int64_t			virtual_offset;
	uint64_t		nullmap_offset;
	uint64_t		nullmap_length;
	uint64_t		values_offset;
	uint64_t		values_length;
	uint64_t		extra_offset;
	uint64_t		extra_length;
};
typedef struct kern_colmeta		kern_colmeta;

#define KDS_FORMAT_ROW			'r'		/* normal heap-tuples */
#define KDS_FORMAT_HASH			'h'		/* inner hash table for HashJoin */
#define KDS_FORMAT_BLOCK		'b'		/* raw blocks for direct loading */
#define KDS_FORMAT_COLUMN		'c'		/* columnar based storage format */
#define KDS_FORMAT_ARROW		'a'		/* apache arrow format */
#define KDS_FORMAT_FALLBACK		'f'		/* CPU-fallback buffer */

struct kern_data_store {
	uint64_t		length;		/* length of this data-store */
	uint64_t		usage;		/* usage of this data-store */
	uint32_t		nitems;		/* number of rows (or blocks) in this store */
	uint32_t		ncols;		/* number of columns in this store */
	char			format;		/* one of KDS_FORMAT_* above */
	bool			has_varlena; /* true, if any varlena attribute */
	bool			tdhasoid;	/* copy of TupleDesc.tdhasoid */
	Oid				tdtypeid;	/* copy of TupleDesc.tdtypeid */
	int32_t			tdtypmod;	/* copy of TupleDesc.tdtypmod */
	Oid				table_oid;	/* OID of the table (only if GpuScan) */
	/* only KDS_FORMAT_HASH */
	uint32_t		hash_nslots;	/* width of the hash-slot */
	/* only KDS_FORMAT_BLOCK */
	uint32_t		block_offset;	/* offset of blocks array */
	uint32_t		block_nloaded;	/* number of blocks already loaded by CPU */
	/* only KDS_FORMAT_COLUMN */
	uint32_t		column_nrooms;	/* = max_num_rows parameter */
	/* only KDS_FORMAT_ARROW */
	uint32_t		arrow_virtual_usage; /* usage of virtual column buffer */
	/* column definition */
	uint32_t		nr_colmeta;	/* number of colmeta[] array elements;
								 * maybe, >= ncols, if any composite types */
	kern_colmeta	colmeta[1];	/* metadata of columns */
};
typedef struct kern_data_store		kern_data_store;

/*
 * Layout of KDS_FORMAT_ROW / KDS_FORMAT_HASH
 *
 * +---------------------+
 * | kern_data_store     |
 * |        :            |
 * | +-------------------+
 * | | kern_colmeta      |
 * | |   colmeta[...]    |
 * +-+-------------------+  <-- KDS_BODY_ADDR(kds)
 * | ^                   |
 * | | Hash slots if any | (*) KDS_FORMAT_ROW always has 'hash_nslots' == 0,
 * | | (uint64 * nslots) |     thus, this field is only for KDS_FORMAT_HASH
 * | v                   |
 * +---------------------+
 * | ^                   |
 * | | Row index      o--------+  ((char *)kds + kds->length - row_index[i])
 * | | (uint64 * nitems) |     |
 * | v                   |     |
 * +---------------------+     |
 * |        :            |     |
 * +---------------------+ --- |
 * | ^                   |  ^  |
 * | | Buffer for        |  |  |
 * | | kern_tupitem,  <--------+
 * | | or kern_hashitem  |  | packed 'usage'
 * | v                   |  v
 * +---------------------+----
 *
 * Layout of KDS_FORMAT_BLOCK
 *
 * +-----------------------+
 * | kern_data_store       |
 * |        :              |
 * | +---------------------+
 * | | kern_colmeta        |
 * | |   colmeta[...]      |
 * +-+---------------------+ <-- KDS_BODY_ADDR(kds)
 * |                       |  ^
 * | Array of BlockNumber  |  | (BlockNumber * nitems)
 * |                       |  v
 * +-----------------------+ ---  <--- (char *)kds + kds->block_offset
 * |                       |  ^
 * | Raw blocks loaded by  |  | (BLCKSZ * block_nloaded)
 * | the host module.      |  |
 * |                       |  v
 * +-------------+---------+ -----
 * | Raw blocks  |   ^
 * | loaded aby  |   | (BLCKSZ * (nitems - block_nloaded)
 * | the device  |   |
 * | module      |   | (*) available only device side
 * |     :       |   v
 * +-------------+ -----
 *
 * Layout of KDS_FORMAT_ARROW
 *
 * +-----------------------+
 * | kern_data_store       |
 * |        :              |
 * | +---------------------+
 * | | kern_colmeta        |
 * | |   colmeta[...]      |
 * +-+---------------------+ <-- KDS_BODY_ADDR(kds)
 * |                       |  ^
 * | iovec of chunks to be |  | offsetof(strom_io_vector, ioc[nr_chunks])
 * | loaded                |  |
 * |                       |  v
 * +-----------------------+ ---
 */

/*
 * kern_data_extra - extra buffer of KDS_FORMAT_COLUMN
 */
struct kern_data_extra
{
	uint64_t	length;
	uint64_t	usage;
	uint64_t	deadspace;
	char		data[1];
};
typedef struct kern_data_extra		kern_data_extra;

/* ----------------------------------------------------------------
 *
 * Definitions of HeapTuple/IndexTuple and related
 *
 * ----------------------------------------------------------------
 */
#ifdef POSTGRES_H
#include "access/htup_details.h"
#include "access/itup.h"
#include "access/sysattr.h"
#else
/*
 * Attribute numbers for the system-defined attributes
 */
#define SelfItemPointerAttributeNumber			(-1)
#define MinTransactionIdAttributeNumber			(-2)
#define MinCommandIdAttributeNumber				(-3)
#define MaxTransactionIdAttributeNumber			(-4)
#define MaxCommandIdAttributeNumber				(-5)
#define TableOidAttributeNumber					(-6)
#define FirstLowInvalidHeapAttributeNumber		(-7)

/*
 * ItemPointer:
 */
typedef struct
{
	struct {
		uint16_t	bi_hi;
		uint16_t	bi_lo;
	} ip_blkid;
	uint16_t		ip_posid;
} ItemPointerData;

INLINE_FUNCTION(bool)
ItemPointerEquals(const ItemPointerData *ip1, const ItemPointerData *ip2)
{
	return (ip1->ip_blkid.bi_hi == ip2->ip_blkid.bi_hi &&
			ip1->ip_blkid.bi_lo == ip2->ip_blkid.bi_lo &&
			ip1->ip_posid       == ip2->ip_posid);
}

INLINE_FUNCTION(void)
ItemPointerSetInvalid(ItemPointerData *ip)
{
	ip->ip_blkid.bi_hi = 0xffffU;
	ip->ip_blkid.bi_lo = 0xffffU;
	ip->ip_posid = 0;
}

/*
 * HeapTupleHeaderData
 */
typedef struct HeapTupleFields
{
	uint32_t		t_xmin;		/* inserting xact ID */
	uint32_t		t_xmax;		/* deleting or locking xact ID */
	union
	{
		uint32_t	t_cid;		/* inserting or deleting command ID, or both */
		uint32_t	t_xvac;		/* old-style VACUUM FULL xact ID */
	}	t_field3;
} HeapTupleFields;

typedef struct DatumTupleFields
{
    int32_t		datum_len_;		/* varlena header (do not touch directly!) */
    int32_t		datum_typmod;	/* -1, or identifier of a record type */
    Oid			datum_typeid;	/* composite type OID, or RECORDOID */
} DatumTupleFields;

typedef struct HeapTupleHeaderData
{
	union {
		HeapTupleFields		t_heap;
		DatumTupleFields	t_datum;
	} t_choice;

	ItemPointerData	t_ctid;			/* current TID of this or newer tuple */

	uint16_t		t_infomask2;	/* number of attributes + various flags */
    uint16_t		t_infomask;		/* various flag bits, see below */
    uint8_t			t_hoff;			/* sizeof header incl. bitmap, padding */
    /* ^ - 23 bytes - ^ */
    uint8_t			t_bits[1];		/* null-bitmap -- VARIABLE LENGTH */
} HeapTupleHeaderData;

/*
 * information stored in t_infomask:
 */
#define HEAP_HASNULL			0x0001	/* has null attribute(s) */
#define HEAP_HASVARWIDTH		0x0002	/* has variable-width attribute(s) */
#define HEAP_HASEXTERNAL		0x0004	/* has external stored attribute(s) */
#define HEAP_HASOID				0x0008	/* has an object-id field */
#define HEAP_XMAX_KEYSHR_LOCK	0x0010	/* xmax is a key-shared locker */
#define HEAP_COMBOCID			0x0020	/* t_cid is a combo cid */
#define HEAP_XMAX_EXCL_LOCK		0x0040	/* xmax is exclusive locker */
#define HEAP_XMAX_LOCK_ONLY		0x0080	/* xmax, if valid, is only a locker */

#define HEAP_XMIN_COMMITTED		0x0100	/* t_xmin committed */
#define HEAP_XMIN_INVALID		0x0200	/* t_xmin invalid/aborted */
#define HEAP_XMAX_COMMITTED		0x0400	/* t_xmax committed */
#define HEAP_XMAX_INVALID		0x0800	/* t_xmax invalid/aborted */
#define HEAP_XMAX_IS_MULTI		0x1000	/* t_xmax is a MultiXactId */
#define HEAP_UPDATED			0x2000	/* this is UPDATEd version of row */
#define HEAP_MOVED_OFF			0x4000	/* unused in xPU */
#define HEAP_MOVED_IN			0x8000	/* unused in xPU */

/*
 * information stored in t_infomask2:
 */
#define HEAP_NATTS_MASK			0x07FF  /* 11 bits for number of attributes */
#define HEAP_KEYS_UPDATED		0x2000  /* tuple was updated and key cols
										 * modified, or tuple deleted */
#define HEAP_HOT_UPDATED		0x4000	/* tuple was HOT-updated */
#define HEAP_ONLY_TUPLE			0x8000	/* this is heap-only tuple */
#define HEAP2_XACT_MASK			0xE000	/* visibility-related bits */

/* null-bitmap checker */
#define att_isnull(ATT, BITS) (!((BITS)[(ATT) >> 3] & (1 << ((ATT) & 0x07))))
#define BITMAPLEN(NATTS)		(((int)(NATTS) + 7) / 8)

/*
 * Index tuple header structure
 *
 * All index tuples start with IndexTupleData.  If the HasNulls bit is set,
 * this is followed by an IndexAttributeBitMapData.  The index attribute
 * values follow, beginning at a MAXALIGN boundary.
 */
typedef struct IndexTupleData
{
	ItemPointerData		t_tid;		/* reference TID to heap tuple */

	/* ---------------
	 * t_info is laid out in the following fashion:
	 *
	 * 15th (high) bit: has nulls
	 * 14th bit: has var-width attributes
	 * 13th bit: AM-defined meaning
	 * 12-0 bit: size of tuple
	 * ---------------
	 */
	uint16_t			t_info;

	char				data[1];	/* data or IndexAttributeBitMapData */
} IndexTupleData;

#define INDEX_MAX_KEYS		32		/* pg_config_manual.h */
typedef struct IndexAttributeBitMapData
{
	uint8_t				bits[BITMAPLEN(INDEX_MAX_KEYS)];
} IndexAttributeBitMapData;

#define INDEX_SIZE_MASK     0x1fff
#define INDEX_VAR_MASK      0x4000
#define INDEX_NULL_MASK     0x8000

#define IndexTupleSize(itup)						\
	((size_t)((itup)->t_info & INDEX_SIZE_MASK))
#define IndexTupleHasNulls(itup)					\
	((((IndexTupleData *)(itup))->t_info & INDEX_NULL_MASK))
#define IndexTupleHasVarwidths(itup)				\
	((((IndexTupleData *)(itup))->t_info & INDEX_VAR_MASK))
#endif	/* POSTGRES_H */

/* ----------------------------------------------------------------
 *
 * Definitions of PageHeader/ItemId and related
 *
 * ----------------------------------------------------------------
 */
#ifdef POSTGRES_H
#include "access/gist.h"
#include "access/transam.h"
#include "storage/bufpage.h"
#include "storage/block.h"
#include "storage/itemid.h"
#include "storage/off.h"
#else
/* definitions in access/transam.h */
typedef uint32_t		TransactionId;
#define InvalidTransactionId		((TransactionId) 0)
#define BootstrapTransactionId		((TransactionId) 1)
#define FrozenTransactionId			((TransactionId) 2)
#define FirstNormalTransactionId	((TransactionId) 3)
#define MaxTransactionId			((TransactionId) 0xffffffff)

typedef struct
{
	uint64_t			value;
} FullTransactionId;

typedef uint32_t		CommandId;
#define FirstCommandId				((CommandId) 0)
#define InvalidCommandId			(~(CommandId)0)

/* definitions in storage/block.h */
typedef uint32_t		BlockNumber;

#define InvalidBlockNumber			((BlockNumber) 0xffffffff)
#define MaxBlockNumber				((BlockNumber) 0xfffffffe)

/* definitions in storage/itemid.h */
typedef struct ItemIdData
{
	unsigned	lp_off:15,		/* offset to tuple (from start of page) */
				lp_flags:2,		/* state of item pointer, see below */
				lp_len:15;		/* byte length of tuple */
} ItemIdData;

#define LP_UNUSED		0		/* unused (should always have lp_len=0) */
#define LP_NORMAL		1		/* used (should always have lp_len>0) */
#define LP_REDIRECT		2		/* HOT redirect (should have lp_len=0) */
#define LP_DEAD			3		/* dead, may or may not have storage */

#define ItemIdGetOffset(itemId)		((itemId)->lp_off)
#define ItemIdGetLength(itemId)		((itemId)->lp_len)
#define ItemIdIsUsed(itemId)		((itemId)->lp_flags != LP_UNUSED)
#define ItemIdIsNormal(itemId)		((itemId)->lp_flags == LP_NORMAL)
#define ItemIdIsRedirected(itemId)	((itemId)->lp_flags == LP_REDIRECT)
#define ItemIdIsDead(itemId)		((itemId)->lp_flags == LP_DEAD)
#define ItemIdHasStorage(itemId)	((itemId)->lp_len != 0)
#define ItemIdSetUnused(itemId)			\
    do {								\
		(itemId)->lp_flags = LP_UNUSED; \
		(itemId)->lp_off = 0;           \
		(itemId)->lp_len = 0;           \
	} while(0)

/* definitions in storage/off.h */
typedef uint16_t			OffsetNumber;
#define InvalidOffsetNumber	((OffsetNumber) 0)
#define FirstOffsetNumber	((OffsetNumber) 1)
#define MaxOffsetNumber		((OffsetNumber) (BLCKSZ / sizeof(ItemIdData)))
#define OffsetNumberNext(offsetNumber)			\
	((OffsetNumber) (1 + (offsetNumber)))
#define OffsetNumberPrev(offsetNumber)			\
	((OffsetNumber) (-1 + (offsetNumber)))

/* definitions in storage/bufpage.h */
typedef char   *Page;
typedef struct	PageHeaderData
{
#if 0
	/*
	 * NOTE: device code (ab-)uses this field to track parent block/item
	 * when GiST index is loaded. Without this hack, hard to implement
	 * depth-first search at GpuJoin.
	 */
	PageXLogRecPtr pd_lsn;		/* LSN: next byte after last byte of xlog
								 * record for last change to this page */
#else
	uint32_t	pd_parent_blkno;
	uint32_t	pd_parent_item;
#endif
	uint16_t	pd_checksum;	/* checksum */
	uint16_t	pd_flags;		/* flag bits, see below */
	uint16_t	pd_lower;		/* offset to start of free space */
	uint16_t	pd_upper;		/* offset to end of free space */
	uint16_t	pd_special;		/* offset to start of special space */
	uint16_t	pd_pagesize_version;
	TransactionId pd_prune_xid;	/* oldest prunable XID, or zero if none */
	ItemIdData	pd_linp[1];		/* line pointer array */
} PageHeaderData;

#define SizeOfPageHeaderData	(offsetof(PageHeaderData, pd_linp))
#define PageGetItemId(page, offsetNumber)			\
	(&((PageHeaderData *)(page))->pd_linp[(offsetNumber) - 1])
#define PageGetItem(page, lpp)						\
	((HeapTupleHeaderData *)((char *)(page) + ItemIdGetOffset(lpp)))
#define PageGetMaxOffsetNumber(page)				\
	(((PageHeaderData *) (page))->pd_lower <= SizeOfPageHeaderData ? 0 :	\
	 ((((PageHeaderData *) (page))->pd_lower - SizeOfPageHeaderData)		\
	  / sizeof(ItemIdData)))
#define PD_HAS_FREE_LINES	0x0001	/* are there any unused line pointers? */
#define PD_PAGE_FULL		0x0002	/* not enough free space for new tuple? */
#define PD_ALL_VISIBLE		0x0004	/* all tuples on page are visible to
									 * everyone */
#define PD_VALID_FLAG_BITS	0x0007	/* OR of all valid pd_flags bits */

/*
 * GiST index specific structures and labels
 */
#define F_LEAF              (1 << 0)    /* leaf page */
#define F_DELETED           (1 << 1)    /* the page has been deleted */
#define F_TUPLES_DELETED    (1 << 2)    /* some tuples on the page were deleted */
#define F_FOLLOW_RIGHT      (1 << 3)    /* page to the right has no downlink */
#define F_HAS_GARBAGE       (1 << 4)    /* some tuples on the page are dead */

#define GIST_PAGE_ID        0xFF81

typedef struct GISTPageOpaqueData
{
	struct {
		uint32_t	xlogid;
		uint32_t	xrecoff;
	} nsn;
	BlockNumber		rightlink;		/* next page if any */
	uint16_t		flags;			/* see bit definitions above */
	uint16_t		gist_page_id;	/* for identification of GiST indexes */
} GISTPageOpaqueData;

INLINE_FUNCTION(GISTPageOpaqueData *)
GistPageGetOpaque(PageHeaderData *page)
{
	return (GISTPageOpaqueData *)((char *)page + page->pd_special);
}

INLINE_FUNCTION(bool)
GistPageIsRoot(PageHeaderData *page)
{
	/* See, __innerPreloadSetupGiSTIndexWalker */
	/* This logic is valid only on the kds_gist, not host buffer */
	return (page->pd_parent_blkno == InvalidBlockNumber &&
			page->pd_parent_item  == InvalidOffsetNumber);
}

INLINE_FUNCTION(bool)
GistPageIsLeaf(PageHeaderData *page)
{
	return (GistPageGetOpaque(page)->flags & F_LEAF) != 0;
}

INLINE_FUNCTION(bool)
GistPageIsDeleted(PageHeaderData *page)
{
	return (GistPageGetOpaque(page)->flags & F_DELETED) != 0;
}

INLINE_FUNCTION(bool)
GistFollowRight(PageHeaderData *page)
{
	return (GistPageGetOpaque(page)->flags & F_FOLLOW_RIGHT) != 0;
}

/* root page of a gist index */
#define GIST_ROOT_BLKNO			0

#endif /* POSTGRES_H */

/*
 * Definition of KDS-Items and related
 */
struct kern_tupitem
{
	uint32_t		t_len;		/* length of tuple */
	uint32_t		rowid;		/* unique Id of this item */
	HeapTupleHeaderData	htup;
};
typedef struct kern_tupitem		kern_tupitem;

/*
 * kern_hashitem - individual items for KDS_FORMAT_HASH
 */
struct kern_hashitem
{
	uint64_t		next;		/* offset of the next entry */
	uint32_t		hash;		/* 32-bit hash value */
	uint32_t		__padding__;
	/* HeapTuple of this entry */
	kern_tupitem	t	__MAXALIGNED__;
};
typedef struct kern_hashitem	kern_hashitem;

/*
 * kern_fallbackitem - individual items for KDS_FORMAT_FALLBACK
 */
struct kern_fallbackitem
{
	uint32_t		t_len;
	uint16_t		depth;
	uint8_t			__reserved__;
	uint8_t			matched;
	uint64_t		l_state;
	HeapTupleHeaderData htup;
};
typedef struct kern_fallbackitem	kern_fallbackitem;

/* Length of the header postion of kern_data_store */
INLINE_FUNCTION(size_t)
KDS_HEAD_LENGTH(const kern_data_store *kds)
{
	return MAXALIGN(offsetof(kern_data_store, colmeta) +
					sizeof(kern_colmeta) * kds->nr_colmeta);
}

/* Base address of the kern_data_store */
INLINE_FUNCTION(char *)
KDS_BODY_ADDR(const kern_data_store *kds)
{
	return (char *)kds + KDS_HEAD_LENGTH(kds);
}

/* ------------------------------------------------
 *
 * access functions for KDS_FORMAT_ROW/HASH
 *
 * ------------------------------------------------
 */
INLINE_FUNCTION(uint64_t *)
KDS_GET_ROWINDEX(const kern_data_store *kds)
{
	assert(kds->format == KDS_FORMAT_ROW ||
		   kds->format == KDS_FORMAT_HASH ||
		   kds->format == KDS_FORMAT_FALLBACK);
	return (uint64_t *)KDS_BODY_ADDR(kds) + kds->hash_nslots;
}

INLINE_FUNCTION(kern_tupitem *)
KDS_GET_TUPITEM(const kern_data_store *kds, uint32_t kds_index)
{
	uint64_t	offset = __volatileRead(KDS_GET_ROWINDEX(kds) + kds_index);

	if (!offset)
		return NULL;
	return (kern_tupitem *)((char *)kds + kds->length - offset);
}

INLINE_FUNCTION(bool)
__KDS_TUPITEM_CHECK_VALID(const kern_data_store *kds, const kern_tupitem *tupitem)
{
	const char *head = (const char *)kds + sizeof(uint64_t) * (kds->hash_nslots +
															   kds->nitems);
	const char *tail = (const char *)kds + kds->length;

	return ((const char *)tupitem >= head &&
			(const char *)tupitem + tupitem->t_len <= tail);
}

INLINE_FUNCTION(bool)
__KDS_CHECK_OVERFLOW(const kern_data_store *kds, uint32_t nitems, uint64_t usage)
{
	assert(kds->format == KDS_FORMAT_ROW ||
		   kds->format == KDS_FORMAT_HASH ||
		   kds->format == KDS_FORMAT_FALLBACK);
	return (KDS_HEAD_LENGTH(kds) +
			sizeof(uint64_t) * (kds->hash_nslots + nitems) +
			usage) <= kds->length;
}

/* ------------------------------------------------
 *
 * access functions for KDS_FORMAT_HASH
 *
 * ------------------------------------------------
 */
INLINE_FUNCTION(uint64_t)
KDS_GET_HASHSLOT_WIDTH(uint64_t nitems)
{
	if (nitems <= 5000)
		return 20000UL;
	if (nitems <= 4000000)
		return 20000UL + 2 * nitems;
	return 8020000UL + nitems;
}

INLINE_FUNCTION(uint64_t *)
KDS_GET_HASHSLOT_BASE(const kern_data_store *kds)
{
	Assert(kds->format == KDS_FORMAT_HASH && kds->hash_nslots > 0);
	return (uint64_t *)(KDS_BODY_ADDR(kds));
}

INLINE_FUNCTION(uint64_t *)
KDS_GET_HASHSLOT(const kern_data_store *kds, uint32_t hash)
{
	uint64_t   *hslot = KDS_GET_HASHSLOT_BASE(kds);

	return hslot + (hash % kds->hash_nslots);
}

INLINE_FUNCTION(bool)
__KDS_HASH_ITEM_CHECK_VALID(const kern_data_store *kds, kern_hashitem *hitem)
{
	char   *tail = (char *)kds + kds->length;
	char   *head = (KDS_BODY_ADDR(kds) + sizeof(uint64_t) * (kds->hash_nslots +
															 kds->nitems));
	return ((char *)hitem >= head &&
			(char *)&hitem->t.htup + hitem->t.t_len <= tail);
}
#include <stdio.h>

INLINE_FUNCTION(kern_hashitem *)
KDS_HASH_FIRST_ITEM(const kern_data_store *kds, uint32_t hash)
{
	uint64_t   *hslot = KDS_GET_HASHSLOT(kds, hash);
	uint64_t	offset = __volatileRead(hslot);

	if (offset != 0 && offset != ULONG_MAX)
	{
		kern_hashitem *hitem = (kern_hashitem *)
			((char *)kds + kds->length - offset);
		assert(__KDS_HASH_ITEM_CHECK_VALID(kds, hitem));
		return hitem;
	}
	return NULL;
}

INLINE_FUNCTION(kern_hashitem *)
KDS_HASH_NEXT_ITEM(const kern_data_store *kds, uint64_t hnext_offset)
{
	if (hnext_offset != 0 && hnext_offset != ULONG_MAX)
	{
		kern_hashitem *hnext = (kern_hashitem *)
			((char *)kds + kds->length - hnext_offset);
		assert(__KDS_HASH_ITEM_CHECK_VALID(kds, hnext));
		return hnext;
	}
	return NULL;
}

/* ------------------------------------------------
 *
 * access functions for KDS_FORMAT_BLOCK
 *
 * ------------------------------------------------
 */
#define KDS_BLOCK_BLCKNR(kds,block_id)					\
	(((BlockNumber *)KDS_BODY_ADDR(kds))[block_id])
#define KDS_BLOCK_PGPAGE(kds,block_id)					\
	((struct PageHeaderData *)((char *)(kds) +			\
							   (kds)->block_offset +	\
							   BLCKSZ * (block_id)))

INLINE_FUNCTION(HeapTupleHeaderData *)
KDS_BLOCK_REF_HTUP(kern_data_store *kds,
				   uint32_t lp_offset,
				   ItemPointerData *p_self,
				   uint32_t *p_len)
{
	/*
	 * NOTE: lp_offset is not packed offset!
	 * KDS_FORMAT_BLOCK must never be larger than 4GB.
	 */
	ItemIdData	   *lpp = (ItemIdData *)((char *)kds + lp_offset);
	uint32_t		head_sz;
	uint32_t		block_id;
	BlockNumber		block_nr;
	PageHeaderData *pg_page;

	Assert(kds->format == KDS_FORMAT_BLOCK);
	if (lp_offset == 0)
		return NULL;
	head_sz = KDS_HEAD_LENGTH(kds) + MAXALIGN(sizeof(BlockNumber) * kds->nitems);
	Assert(lp_offset >= head_sz &&
		   lp_offset <  head_sz + BLCKSZ * kds->nitems);
	block_id = (lp_offset - head_sz) / BLCKSZ;
	block_nr = KDS_BLOCK_BLCKNR(kds, block_id);
	pg_page = KDS_BLOCK_PGPAGE(kds, block_id);

	Assert(lpp >= pg_page->pd_linp &&
		   lpp -  pg_page->pd_linp < PageGetMaxOffsetNumber((Page)pg_page));
	if (p_self)
	{
		p_self->ip_blkid.bi_hi  = block_nr >> 16;
		p_self->ip_blkid.bi_lo  = block_nr & 0xffff;
		p_self->ip_posid        = lpp - pg_page->pd_linp;
	}
	if (p_len)
		*p_len = ItemIdGetLength(lpp);
	return (HeapTupleHeaderData *)PageGetItem((Page)pg_page, lpp);
}

INLINE_FUNCTION(bool)
KDS_BLOCK_CHECK_VALID(const kern_data_store *kds,
					  const struct PageHeaderData *page)
{
	const char *base = (const char *)kds + kds->block_offset;

	return ((const char *)page          >= base &&
			(const char *)page + BLCKSZ <= (const char *)kds + kds->length &&
			(((const char *)page - base) & (BLCKSZ-1)) == 0);
}

/* access functions for apache arrow format */
INLINE_FUNCTION(bool)
KDS_ARROW_CHECK_ISNULL(const kern_data_store *kds,
					   const kern_colmeta *cmeta,
					   uint32_t index)
{
	Assert(cmeta >= kds->colmeta &&
		   cmeta <  kds->colmeta + kds->nr_colmeta);
	if (cmeta->nullmap_offset)
	{
		uint8_t	   *nullmap = (uint8_t *)kds + cmeta->nullmap_offset;
		uint32_t	mask = (1U << (index & 7));

		index = (index >> 3);
		if (index >= cmeta->nullmap_length ||
			(nullmap[index] & mask) == 0)
			return true;	/* NULL */
	}
	return false;
}

INLINE_FUNCTION(const void *)
KDS_ARROW_REF_SIMPLE_DATUM(const kern_data_store *kds,
						   const kern_colmeta *cmeta,
						   uint32_t index,
						   uint32_t unitsz)
{
	Assert(cmeta->values_offset > 0 &&
		   cmeta->extra_offset == 0 &&
		   cmeta->extra_length == 0);
	/* NOTE: caller should already apply NULL-checks, so we don't check
	 * it again. */
	if (unitsz * (index + 1) <= cmeta->values_length)
	{
		const char *values = ((const char *)kds + cmeta->values_offset);

		return values + unitsz * index;
	}
	return NULL;
}

#define VARATT_MAX		0x4ffffff8U

INLINE_FUNCTION(const void *)
KDS_ARROW_REF_VARLENA32_DATUM(const kern_data_store *kds,
							  const kern_colmeta *cmeta,
							  uint32_t index,
							  int *p_length)
{
	Assert(cmeta->values_offset > 0 &&
		   cmeta->extra_offset  > 0);
	/* NOTE: caller should already apply NULL-checks, so we don't check
	 * it again. */
	if (sizeof(uint32_t) * (index+1) <= cmeta->values_length)
	{
		const uint32_t *offset = (const uint32_t *)
			((const char *)kds + cmeta->values_offset);
		const char	   *extra  = (const char *)
			((const char *)kds + cmeta->extra_offset);
		if (offset[index]   <= offset[index+1] &&
			offset[index+1] <= cmeta->extra_length &&
			offset[index+1] - offset[index] <= VARATT_MAX)
		{
			*p_length = (int)(offset[index+1] - offset[index]);
			return (extra + offset[index]);
		}
	}
	return NULL;
}

INLINE_FUNCTION(const void *)
KDS_ARROW_REF_VARLENA64_DATUM(const kern_data_store *kds,
							  const kern_colmeta *cmeta,
							  uint32_t index,
							  int *p_length)
{
	Assert(cmeta->values_offset > 0 &&
		   cmeta->extra_offset  > 0);
	if (sizeof(uint32_t) * (index+1) <= cmeta->values_length)
	{
		const uint64_t *offset = (const uint64_t *)
			((const char *)kds + cmeta->values_offset);
		const char	   *extra  = (const char *)
			((const char *)kds + cmeta->extra_offset);
		if (offset[index]   <= offset[index+1] &&
			offset[index+1] <= cmeta->extra_length &&
			offset[index+1] - offset[index] <= VARATT_MAX)
		{
			*p_length = (int)(offset[index+1] - offset[index]);
			return (extra + offset[index]);
		}
	}
	return NULL;
}

INLINE_FUNCTION(bool)
KDS_COLUMN_ITEM_ISNULL(const kern_data_store *kds,
					   const kern_colmeta *cmeta,
					   uint32_t rowid)
{
	uint8_t	   *bitmap;
	uint8_t		mask = (1 << (rowid & 7));
	uint32_t	idx = (rowid >> 3);

	if (cmeta->nullmap_offset == 0)
		return false;	/* NOT NULL */
	if (idx >= cmeta->nullmap_length)
		return false;	/* NOT NULL */
	bitmap = (uint8_t *)kds + cmeta->nullmap_offset;

	return (bitmap[idx] & mask) == 0;
}

/*
 * GpuCacheSysattr
 *
 * An internal system attribute of GPU cache
 */
struct GpuCacheSysattr
{
	uint32_t	xmin;
	uint32_t	xmax;
	uint32_t	owner;
	ItemPointerData ctid;
	uint16_t	__padding__;	//can be used for t_infomask?
};
typedef struct GpuCacheSysattr	GpuCacheSysattr;

/* ----------------------------------------------------------------
 *
 * Definitions of Varlena datum and related (mostly in postgres.h and c.h)
 *
 * ----------------------------------------------------------------
 */
typedef struct varlena		varlena;
#ifndef POSTGRES_H
struct varlena {
    char		vl_len_[4];	/* Do not touch this field directly! */
	char		vl_dat[1];
};

#define VARHDRSZ			((int) sizeof(uint32_t))
#define VARDATA(PTR)		VARDATA_4B(PTR)
#define VARSIZE(PTR)		VARSIZE_4B(PTR)

#define VARSIZE_SHORT(PTR)	VARSIZE_1B(PTR)
#define VARDATA_SHORT(PTR)	VARDATA_1B(PTR)

typedef union
{
	struct						/* Normal varlena (4-byte length) */
	{
		uint32_t	va_header;
		char		va_data[1];
	}		va_4byte;
	struct						/* Compressed-in-line format */
	{
		uint32_t	va_header;
		uint32_t	va_tcinfo;	/* Original data size (excludes header) and
								 * compression method; see va_extinfo */
		char		va_data[1];	/* Compressed data */
	}		va_compressed;
} varattrib_4b;

typedef struct
{
    uint8_t			va_header;
    char			va_data[1];	/* Data begins here */
} varattrib_1b;

/* inline portion of a short varlena pointing to an external resource */
typedef struct
{
	uint8_t			va_header;	/* Always 0x80 or 0x01 */
	uint8_t			va_tag;		/* Type of datum */
    char			va_data[1];	/* Data (of the type indicated by va_tag) */
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
	int32_t		va_rawsize;		/* Original data size (includes header) */
	uint32_t	va_extinfo;		/* External saved size (doesn't) */
	Oid			va_valueid;		/* Unique ID of value within TOAST table */
	Oid			va_toastrelid;	/* RelID of TOAST table containing it */
} varatt_external;

/*
 * These macros define the "saved size" portion of va_extinfo.  Its remaining
 * two high-order bits identify the compression method.
 */
#define VARLENA_EXTSIZE_BITS	30
#define VARLENA_EXTSIZE_MASK	((1U << VARLENA_EXTSIZE_BITS) - 1)


typedef struct varatt_indirect
{
	uintptr_t	pointer;		/* Host pointer to in-memory varlena */
} varatt_indirect;

#define VARTAG_SIZE(tag) \
	((tag) == VARTAG_INDIRECT ? sizeof(varatt_indirect) :   \
	 (tag) == VARTAG_ONDISK ? sizeof(varatt_external) :     \
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
	int32_t		vl_len_;		/* varlena header (do not touch directly!) */
    uint32_t	tcinfo;			/* 2 bits for compression method and 30bits
								 * external size; see va_extinfo */
} toast_compress_header;

#define TOAST_COMPRESS_EXTSIZE(ptr)										\
	(((toast_compress_header *) (ptr))->tcinfo & VARLENA_EXTSIZE_MASK)
#define TOAST_COMPRESS_METHOD(ptr)										\
	(((toast_compress_header *) (ptr))->tcinfo >> VARLENA_EXTSIZE_BITS)

#define TOAST_COMPRESS_HDRSZ        ((uint32_t)sizeof(toast_compress_header))
#define TOAST_COMPRESS_RAWSIZE(ptr)             \
    (((toast_compress_header *) (ptr))->rawsize)
#define TOAST_COMPRESS_RAWDATA(ptr)             \
    (((char *) (ptr)) + TOAST_COMPRESS_HDRSZ)
#define TOAST_COMPRESS_SET_RAWSIZE(ptr, len)    \
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
#define VARATT_NOT_PAD_BYTE(PTR)		(*((uint8_t *) (PTR)) != 0)

#define VARSIZE_4B(PTR)                     \
	((__Fetch(&((varattrib_4b *)(PTR))->va_4byte.va_header)>>2) & 0x3FFFFFFF)
#define VARSIZE_1B(PTR) \
	((((varattrib_1b *) (PTR))->va_header >> 1) & 0x7F)
#define VARTAG_1B_E(PTR) \
	(((varattrib_1b_e *) (PTR))->va_tag)

#define VARRAWSIZE_4B_C(PTR)    \
	__Fetch(&((varattrib_4b *) (PTR))->va_compressed.va_rawsize)

#define VARSIZE_ANY_EXHDR(PTR) \
	(VARATT_IS_1B_E(PTR) ? VARSIZE_EXTERNAL(PTR)-VARHDRSZ_EXTERNAL : \
	 (VARATT_IS_1B(PTR) ? VARSIZE_1B(PTR)-VARHDRSZ_SHORT :			 \
	  VARSIZE_4B(PTR)-VARHDRSZ))

#define VARSIZE_ANY(PTR)							\
    (VARATT_IS_1B_E(PTR) ? VARSIZE_EXTERNAL(PTR) :	\
     (VARATT_IS_1B(PTR) ? VARSIZE_1B(PTR) :			\
      VARSIZE_4B(PTR)))

#define VARDATA_4B(PTR)		(((varattrib_4b *) (PTR))->va_4byte.va_data)
#define VARDATA_4B_C(PTR)	(((varattrib_4b *) (PTR))->va_compressed.va_data)
#define VARDATA_1B(PTR)		(((varattrib_1b *) (PTR))->va_data)
#define VARDATA_1B_E(PTR)	(((varattrib_1b_e *) (PTR))->va_data)
#define VARDATA_ANY(PTR)									\
	(VARATT_IS_1B(PTR) ? VARDATA_1B(PTR) : VARDATA_4B(PTR))

#define SET_VARSIZE(PTR, len)                       \
	(((varattrib_4b *)(PTR))->va_4byte.va_header = (((uint32_t) (len)) << 2))
#endif	/* POSTGRES_H */

/* ----------------------------------------------------------------
 *
 * Definition of vectorized xPU device data types
 *
 * ----------------------------------------------------------------
 */
#define KVEC_UNITSZ			(CUDA_MAXTHREADS_PER_BLOCK * 2)
#define KVEC_ALIGN(x)		TYPEALIGN(16,(x))	/* 128bit alignment */

#define KVEC_DATUM_COMMON_FIELD					\
	bool			isnull[KVEC_UNITSZ]
typedef struct kvec_datum_t {
	KVEC_DATUM_COMMON_FIELD;
} kvec_datum_t;

/* ----------------------------------------------------------------
 *
 * Definitions for XPU device data types
 *
 * ----------------------------------------------------------------
 */
typedef struct xpu_datum_t			xpu_datum_t;
typedef struct xpu_datum_operators	xpu_datum_operators;
typedef struct kern_varslot_desc	kern_varslot_desc;

#define XPU_DATUM_COMMON_FIELD			\
	const struct xpu_datum_operators *expr_ops

struct xpu_datum_t {
	XPU_DATUM_COMMON_FIELD;
};

/* if NULL, expr_ops is not set */
#define XPU_DATUM_ISNULL(xdatum)		(!(xdatum)->expr_ops)

struct xpu_datum_operators {
	const char *xpu_type_name;
	bool		xpu_type_byval;		/* = pg_type.typbyval */
	int8_t		xpu_type_align;		/* = pg_type.typalign */
	int16_t		xpu_type_length;	/* = pg_type.typlen */
	TypeOpCode	xpu_type_code;		/* = TypeOpCode__XXXX */
	int			xpu_type_sizeof;	/* = sizeof(xpu_XXXX_t), not PG type! */
	int			xpu_type_alignof;	/* = __alignof__(xpu_XXX_t), not PG type! */
	int			xpu_kvec_sizeof;	/* = sizeof(kvec_XXXX_t), not PG type! */
	int			xpu_kvec_alignof;	/* = __alignof__(kvec_XXX_t), not PG type! */

	/*
	 * xpu_datum_heap_read: called by LoadVars to load heap datum
	 * to the xpu_datum slot, prior to execution of the kexp in
	 * the current depth.
	 * also, it is used to reference const and param.
	 */
	bool	  (*xpu_datum_heap_read)(kern_context *kcxt,
									 const void *addr,		/* in */
									 xpu_datum_t *result);	/* out */
	/*
	 * xpu_datum_arrow_ref: called by LoadVars to load arrow datum
	 * to the xpu_datum slot, prior to execution of the kexp in the
	 * current depth.
	 */
	bool	  (*xpu_datum_arrow_read)(kern_context *kcxt,
									  const kern_data_store *kds,	/* in */
									  const kern_colmeta *cmeta,	/* in */
									  uint32_t kds_index,			/* in */
									  xpu_datum_t *result);			/* out */

	/*
	 * xpu_datum_kvec_load: it loads the xpu_datum_t (scalar value)
	 * from the vectorized buffer.
	 */
	bool	  (*xpu_datum_kvec_load)(kern_context *kcxt,
									 const kvec_datum_t *kvecs,	/* in */
									 uint32_t kvecs_id,			/* in */
									 xpu_datum_t *result);		/* out */
	/*
	 * xpu_datum_kvec_save: it saves the xpu_datum_t (scalar value)
	 * onto the vectorized buffer.
	 */
	bool	  (*xpu_datum_kvec_save)(kern_context *kcxt,
									 const xpu_datum_t *xdatum, /* in */
									 kvec_datum_t *kvecs,		/* out */
									 uint32_t kvecs_id);		/* out */
	/*
	 * xpu_datum_kvec_copy: it moves a vectorized buffer entry from the
	 * source to the destination.
	 */
	bool	  (*xpu_datum_kvec_copy)(kern_context *kcxt,
									 const kvec_datum_t *kvecs_src,	/* in */
									 uint32_t kvecs_src_id,			/* in */
									 kvec_datum_t *kvecs_dst,		/* out */
									 uint32_t kvecs_dst_id);		/* out */
	/*
	 * xpu_datum_write: called by Projection or GpuPreAgg to write out
	 * xdatum onto the destination/final buffer.
	 */
	int		  (*xpu_datum_write)(kern_context *kcxt,
								 char *buffer,					/* out */
								 const kern_colmeta *cmeta_dst,	/* in */
								 const xpu_datum_t *xdatum);	/* in */
	/*
	 * xpu_datum_hash: calculation of hash value using pg_hash_any()
	 */
	bool	  (*xpu_datum_hash)(kern_context *kcxt,
								uint32_t *p_hash,		/* out */
								xpu_datum_t *arg);		/* in */
	/*
	 * xpu_datum_comp: compares two xpu_datum values
	 */
	bool	  (*xpu_datum_comp)(kern_context *kcxt,
								int *p_comp,			/* out */
								xpu_datum_t *a,			/* in */
								xpu_datum_t *b);		/* in */
};

#define __PGSTROM_SQLTYPE_SIMPLE_DECLARATION(NAME,BASETYPE)	\
	typedef struct {										\
		XPU_DATUM_COMMON_FIELD;								\
		BASETYPE	value;									\
	} xpu_##NAME##_t;										\
	EXTERN_DATA(xpu_datum_operators, xpu_##NAME##_ops)
#define PGSTROM_SQLTYPE_SIMPLE_DECLARATION(NAME,BASETYPE)	\
	typedef struct {										\
		KVEC_DATUM_COMMON_FIELD;							\
		BASETYPE	values[KVEC_UNITSZ];					\
	} kvec_##NAME##_t;										\
	typedef struct {										\
		XPU_DATUM_COMMON_FIELD;								\
		BASETYPE	value;									\
	} xpu_##NAME##_t;										\
	EXTERN_DATA(xpu_datum_operators, xpu_##NAME##_ops)

#define __PGSTROM_SQLTYPE_VARLENA_DECLARATION(NAME)			\
	typedef struct {										\
		XPU_DATUM_COMMON_FIELD;								\
		int			length;		/* -1, if PG verlena */		\
		const char *value;									\
	} xpu_##NAME##_t;										\
	EXTERN_DATA(xpu_datum_operators, xpu_##NAME##_ops)
#define PGSTROM_SQLTYPE_VARLENA_DECLARATION(NAME)			\
	typedef struct {										\
		KVEC_DATUM_COMMON_FIELD;							\
		int			length[KVEC_UNITSZ];					\
		const char *values[KVEC_UNITSZ];					\
	} kvec_##NAME##_t;										\
	typedef struct {										\
		XPU_DATUM_COMMON_FIELD;								\
		int			length;		/* -1, if PG verlena */		\
		const char *value;									\
	} xpu_##NAME##_t;										\
	EXTERN_DATA(xpu_datum_operators, xpu_##NAME##_ops)

#define PGSTROM_SQLTYPE_OPERATORS(NAME,TYPBYVAL,TYPALIGN,TYPLENGTH) \
	PUBLIC_DATA(xpu_datum_operators, xpu_##NAME##_ops) = {			\
		.xpu_type_name        = #NAME,								\
		.xpu_type_byval       = TYPBYVAL,							\
		.xpu_type_align       = TYPALIGN,							\
		.xpu_type_length      = TYPLENGTH,							\
		.xpu_type_code        = TypeOpCode__##NAME,					\
		.xpu_type_sizeof      = sizeof(xpu_##NAME##_t),				\
		.xpu_type_alignof     = __alignof__(xpu_##NAME##_t),		\
		.xpu_kvec_sizeof      = sizeof(kvec_##NAME##_t),			\
		.xpu_kvec_alignof     = __alignof__(kvec_##NAME##_t),		\
		.xpu_datum_heap_read  = xpu_##NAME##_datum_heap_read,		\
		.xpu_datum_arrow_read = xpu_##NAME##_datum_arrow_read,		\
		.xpu_datum_kvec_load  = xpu_##NAME##_datum_kvec_load,		\
		.xpu_datum_kvec_save  = xpu_##NAME##_datum_kvec_save,		\
		.xpu_datum_kvec_copy  = xpu_##NAME##_datum_kvec_copy,		\
		.xpu_datum_write      = xpu_##NAME##_datum_write,			\
		.xpu_datum_hash       = xpu_##NAME##_datum_hash,			\
		.xpu_datum_comp       = xpu_##NAME##_datum_comp,			\
	}

#include "xpu_basetype.h"
#include "xpu_numeric.h"
#include "xpu_textlib.h"
#include "xpu_timelib.h"
#include "xpu_misclib.h"
#include "xpu_jsonlib.h"
#include "xpu_postgis.h"

/*
 * xpu_array_t - array type support
 *
 * NOTE: xpu_array_t is designed to store both of PostgreSQL / Arrow array
 * values. If @length < 0, it means @value points a varlena based PostgreSQL
 * array values; which includes nitems, dimension, nullmap and so on.
 * Elsewhere, @length means number of elements, from @start of the array on
 * the columnar buffer by @smeta. @kds can be pulled by @smeta->kds_offset.
 */
typedef struct {
	XPU_DATUM_COMMON_FIELD;
	int32_t		length;
	union {
		struct {
			const varlena *value;
		} heap;
		struct {
			const kern_colmeta *cmeta;
			uint32_t	start;
			uint32_t	slot_id;
		} arrow;
	} u;
} xpu_array_t;

typedef struct {
	KVEC_DATUM_COMMON_FIELD;
	const kern_colmeta *cmeta;		/* common in kvec */
	uint32_t	slot_id;			/* common in kvec */	
	int32_t		length[KVEC_UNITSZ];
	union {
		struct {
			const varlena  *values[KVEC_UNITSZ];
		} heap;		/* length < 0 */
		struct {
			uint32_t		start[KVEC_UNITSZ];
		} arrow;	/* length >= 0 */
	} u;
} kvec_array_t;
EXTERN_DATA(xpu_datum_operators, xpu_array_ops);

/* access macros for heap array */
typedef struct
{
	int32_t		ndim;			/* # of dimensions */
	int32_t		dataoffset;		/* offset to data, or 0 if no bitmap */
	Oid			elemtype;		/* element type OID */
	uint32_t	data[1];
} __ArrayTypeData;				/* payload of ArrayType */

INLINE_FUNCTION(int32_t)
__pg_array_ndim(const __ArrayTypeData *ar)
{
	return __Fetch(&ar->ndim);
}
INLINE_FUNCTION(int32_t)
__pg_array_dataoff(const __ArrayTypeData *ar)
{
	return __Fetch(&ar->dataoffset);
}
INLINE_FUNCTION(bool)
__pg_array_hasnull(const __ArrayTypeData *ar)
{
	return (__pg_array_dataoff(ar) != 0);
}
INLINE_FUNCTION(int32_t)
__pg_array_dim(const __ArrayTypeData *ar, int k)
{
	return __Fetch(&ar->data[k]);
}
INLINE_FUNCTION(uint8_t *)
__pg_array_nullmap(const __ArrayTypeData *ar)
{
	uint32_t	dataoff = __pg_array_dataoff(ar);

	if (dataoff > 0)
	{
		int32_t	ndim = __pg_array_ndim(ar);

		return (uint8_t *)((char *)&ar->data[2 * ndim]);
	}
	return NULL;
}
INLINE_FUNCTION(char *)
__pg_array_dataptr(const __ArrayTypeData *ar)
{
	uint32_t	dataoff = __pg_array_dataoff(ar);

	if (dataoff == 0)
	{
		int32_t	ndim = __pg_array_ndim(ar);

		dataoff = MAXALIGN(VARHDRSZ +
						   offsetof(__ArrayTypeData, data) +
						   2 * sizeof(uint32_t) * ndim);
	}
	assert(dataoff >= VARHDRSZ + offsetof(__ArrayTypeData, data));
	return (char *)ar + dataoff - VARHDRSZ;
}

/*
 * xpu_composite_t - composite type support
 *
 * NOTE: xpu_composite_t is designed to store both of PostgreSQL / Arrow composite
 * values. If @cmeta is NULL, it means @heap.value points a varlena based PostgreSQL
 * composite datum. Elsewhere, @arrow.value points the composite value on the
 * KDS_FORMAT_ARROW chunk,identified by the @rowidx.
 */
typedef struct {
	XPU_DATUM_COMMON_FIELD;
	const kern_colmeta *cmeta;		/* if Arrow::Composite type, it reference to the composite
									 * metadata to walk on the subfields. Elsewhere, @cmeta is
									 * NULL, and @u.heap.value points PostgreSQL record. */
	union {
		struct {
			const varlena *value;	/* composite varlena in heap-format */
		} heap;
		struct {
			uint32_t	rowidx;		/* composite row-index in arrow-format */
			uint32_t	slot_id;	/* vs_desc slot-id that stored in */
		} arrow;
	} u;
} xpu_composite_t;

typedef struct
{
	KVEC_DATUM_COMMON_FIELD;
	const kern_colmeta *cmeta;		/* to be identical in kvec */
	union {
		struct {
			const varlena *values[KVEC_UNITSZ];
		} heap;
		struct {
			uint32_t	slot_id;	/* to be identical in kvec */
			uint32_t	rowidx[KVEC_UNITSZ];
		} arrow;
	} u;
} kvec_composite_t;

EXTERN_DATA(xpu_datum_operators, xpu_composite_ops);

/*
 * xpu_internal_t - utility data type for internal usage such as:
 *
 * - to carry fixed-length datum for device projection
 * - to carry GiST-index pointer to the next depth
 */
typedef struct {
	XPU_DATUM_COMMON_FIELD;
	const void *value;
} xpu_internal_t;

typedef struct {
	KVEC_DATUM_COMMON_FIELD;
	const void *values[KVEC_UNITSZ];
} kvec_internal_t;

EXTERN_DATA(xpu_datum_operators, xpu_internal_ops);

/*
 * device type catalogs
 */
typedef struct {
	TypeOpCode		type_opcode;
	xpu_datum_operators *type_ops;
} xpu_type_catalog_entry;

EXTERN_DATA(xpu_type_catalog_entry, builtin_xpu_types_catalog[]);

/* device type hash for xPU service */
typedef struct xpu_type_hash_entry xpu_type_hash_entry;
struct xpu_type_hash_entry
{
	xpu_type_hash_entry	   *next;
	xpu_type_catalog_entry	cat;
};
typedef struct
{
	uint32_t		nitems;
	uint32_t		nslots;
	xpu_type_hash_entry *slots[1];	/* variable */
} xpu_type_hash_table;

/* ----------------------------------------------------------------
 *
 * Definition of device flags
 *
 * ---------------------------------------------------------------- */
#define DEVKIND__NONE				0x00000000U	/* no accelerator device */
#define DEVKIND__NVIDIA_GPU			0x00000001U	/* for CUDA-based GPU */
#define DEVKIND__NVIDIA_DPU			0x00000002U	/* for BlueField-X DPU */
#define DEVKIND__ANY				0x00000003U	/* Both of GPU and DPU */
#define DEVFUNC__LOCALE_AWARE		0x00000100U	/* Device function is locale aware,
												 * thus, available only if "C" or
												 * no locale configuration */
#define DEVKERN__SESSION_TIMEZONE	0x00000200U	/* Device function needs session
												 * timezone */
#define DEVFUNC__HAS_RECURSION		0x00000400U	/* Device function has recursive calls */
#define DEVTYPE__HAS_COMPARE		0x00000800U	/* Device type has compare handler */
#define DEVTASK__PINNED_HASH_RESULTS 0x00001000U/* Pinned results in HASH format */
#define DEVTASK__PINNED_ROW_RESULTS	0x00002000U	/* Pinned results in ROW format */
#define DEVTASK__USED_GPUDIRECT		0x00004000U	/* Task used GPU-Direct SQL */
#define DEVTASK__USED_GPUCACHE		0x00008000U	/* Task used GPU-Cache */
#define DEVTASK__PREAGG_FINAL_MERGE	0x00010000U	/* PreAgg final buffer should be merged
												 * on the xPU device side */

#define DEVTASK__SCAN				0x10000000U	/* xPU-Scan */
#define DEVTASK__JOIN				0x20000000U	/* xPU-Join */
#define DEVTASK__PREAGG				0x40000000U	/* xPU-PreAgg */
#define DEVTASK__SORT				0x80000000U	/* GPU-Sort */
#define DEVTASK__MASK				0x70000000U	/* mask of avove workloads */

#define TASK_KIND__GPUSCAN		(DEVTASK__SCAN   | DEVKIND__NVIDIA_GPU)
#define TASK_KIND__GPUJOIN		(DEVTASK__JOIN   | DEVKIND__NVIDIA_GPU)
#define TASK_KIND__GPUPREAGG	(DEVTASK__PREAGG | DEVKIND__NVIDIA_GPU)

#define TASK_KIND__DPUSCAN		(DEVTASK__SCAN   | DEVKIND__NVIDIA_DPU)
#define TASK_KIND__DPUJOIN		(DEVTASK__JOIN   | DEVKIND__NVIDIA_DPU)
#define TASK_KIND__DPUPREAGG	(DEVTASK__PREAGG | DEVKIND__NVIDIA_DPU)

#define TASK_KIND__MASK			(DEVTASK__MASK   | DEVKIND__ANY)

/* ----------------------------------------------------------------
 *
 * Definition of device functions
 *
 * ---------------------------------------------------------------- */
typedef struct kern_expression	kern_expression;
#define XPU_PGFUNCTION_ARGS		kern_context *kcxt,				\
								const kern_expression *kexp,	\
								xpu_datum_t *__result
typedef bool  (*xpu_function_t)(XPU_PGFUNCTION_ARGS);

#define KEXP_PROCESS_ARGS0(RETTYPE)								\
	xpu_##RETTYPE##_t *result = (xpu_##RETTYPE##_t *)__result;	\
																\
	assert(kexp->exptype == TypeOpCode__##RETTYPE &&			\
		   kexp->nr_args == 0)


#define KEXP_PROCESS_ARGS1(RETTYPE,ARGTYPE1,ARGNAME1)			\
	xpu_##RETTYPE##_t *result = (xpu_##RETTYPE##_t *)__result;	\
	xpu_##ARGTYPE1##_t ARGNAME1;								\
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																\
	assert(kexp->exptype == TypeOpCode__##RETTYPE &&			\
		   kexp->nr_args == 1 &&								\
		   KEXP_IS_VALID(karg,ARGTYPE1));						\
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &ARGNAME1))			\
		return false

#define KEXP_PROCESS_ARGS2(RETTYPE,ARGTYPE1,ARGNAME1,			\
								   ARGTYPE2,ARGNAME2)			\
	xpu_##RETTYPE##_t *result = (xpu_##RETTYPE##_t *)__result;	\
	xpu_##ARGTYPE1##_t ARGNAME1;								\
	xpu_##ARGTYPE2##_t ARGNAME2;								\
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																\
	assert(kexp->exptype == TypeOpCode__##RETTYPE &&			\
		   kexp->nr_args == 2 &&								\
		   KEXP_IS_VALID(karg,ARGTYPE1));						\
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &ARGNAME1))			\
		return false;											\
	karg = KEXP_NEXT_ARG(karg);									\
	assert(KEXP_IS_VALID(karg, ARGTYPE2));						\
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &ARGNAME2))			\
		return false

#define KEXP_PROCESS_ARGS3(RETTYPE,ARGTYPE1,ARGNAME1,			\
								   ARGTYPE2,ARGNAME2,			\
								   ARGTYPE3,ARGNAME3)			\
	xpu_##RETTYPE##_t *result = (xpu_##RETTYPE##_t *)__result;	\
	xpu_##ARGTYPE1##_t ARGNAME1;								\
	xpu_##ARGTYPE2##_t ARGNAME2;								\
	xpu_##ARGTYPE3##_t ARGNAME3;								\
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																\
	assert(kexp->exptype == TypeOpCode__##RETTYPE &&			\
		   kexp->nr_args == 3);									\
	assert(KEXP_IS_VALID(karg, ARGTYPE1));						\
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &ARGNAME1))			\
		return false;											\
	karg = KEXP_NEXT_ARG(karg);									\
	assert(KEXP_IS_VALID(karg, ARGTYPE2));						\
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &ARGNAME2))			\
		return false;											\
	karg = KEXP_NEXT_ARG(karg);									\
	assert(KEXP_IS_VALID(karg, ARGTYPE3));						\
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &ARGNAME3))			\
		return false

#define KEXP_PROCESS_ARGS4(RETTYPE,ARGTYPE1,ARGNAME1,			\
								   ARGTYPE2,ARGNAME2,			\
								   ARGTYPE3,ARGNAME3,			\
								   ARGTYPE4,ARGNAME4)			\
	xpu_##RETTYPE##_t *result = (xpu_##RETTYPE##_t *)__result;	\
	xpu_##ARGTYPE1##_t ARGNAME1;								\
	xpu_##ARGTYPE2##_t ARGNAME2;								\
	xpu_##ARGTYPE3##_t ARGNAME3;								\
	xpu_##ARGTYPE4##_t ARGNAME4;								\
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																\
	assert(kexp->exptype == TypeOpCode__##RETTYPE &&			\
		   kexp->nr_args == 4);									\
	assert(KEXP_IS_VALID(karg, ARGTYPE1));						\
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &ARGNAME1))			\
		return false;											\
	karg = KEXP_NEXT_ARG(karg);									\
	assert(KEXP_IS_VALID(karg, ARGTYPE2));						\
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &ARGNAME2))			\
		return false;											\
	karg = KEXP_NEXT_ARG(karg);									\
	assert(KEXP_IS_VALID(karg, ARGTYPE3));						\
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &ARGNAME3))			\
		return false;											\
	karg = KEXP_NEXT_ARG(karg);									\
	assert(KEXP_IS_VALID(karg, ARGTYPE4));						\
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &ARGNAME4))			\
		return false

#define __KAGG_ACTION__USE_FILTER	8192	/* only used in optimizer */
#define KAGG_ACTION__VREF			101		/* simple var copy */
#define KAGG_ACTION__VREF_NOKEY		102		/* simple var copy; but not a grouping-
											 * key, if GROUP-BY primary key.
											 * (only code generator internal use) */
#define KAGG_ACTION__NROWS_ANY		201		/* <int8> - increment 1 always */
#define KAGG_ACTION__NROWS_COND		202		/* <int8> - increment 1 if not NULL */
#define KAGG_ACTION__PMIN_INT32		302		/* <int4>,<int8> - min value */
#define KAGG_ACTION__PMIN_INT64		303		/* <int4>,<int8> - min value */
#define KAGG_ACTION__PMIN_FP64		304		/* <int4>,<float8> - min value */
#define KAGG_ACTION__PMAX_INT32		402		/* <int4>,<int8> - max value */
#define KAGG_ACTION__PMAX_INT64		403		/* <int4>,<int8> - max value */
#define KAGG_ACTION__PMAX_FP64		404		/* <int4>,<float8> - max value */
#define KAGG_ACTION__PSUM_INT		501		/* <int8> - sum of values */
#define KAGG_ACTION__PSUM_INT64		502		/* <int8>,<int8+8> */
#define KAGG_ACTION__PSUM_FP		503		/* <float8> - sum of values */
#define KAGG_ACTION__PSUM_NUMERIC	504		/* <int4>,<int8+8> - sum of values */
#define KAGG_ACTION__PAVG_INT		601		/* <int4>,<int8> - NROWS+PSUM */
#define KAGG_ACTION__PAVG_INT64		602		/* <int8>,<int8+8> */
#define KAGG_ACTION__PAVG_FP		603		/* <int4>,<float8> - NROWS+PSUM */
#define KAGG_ACTION__PAVG_NUMERIC	604		/* <int4>,<int8+8> - NROWS+PSUM */
#define KAGG_ACTION__STDDEV			701		/* <int4>,<float8>,<float8> - stddev */
#define KAGG_ACTION__COVAR			801		/* <int4>,<float8>x5 - covariance */

#define __PAGG_MINMAX_ATTRS__VALID	0x0001	/* value is not empty */

typedef struct
{
	int32_t		vl_len_;
	uint32_t	attrs;
	int64_t		value;
} kagg_state__pminmax_int64_packed;

typedef struct
{
	int32_t		vl_len_;
	uint32_t	attrs;
	float8_t	value;
} kagg_state__pminmax_fp64_packed;

typedef struct
{
	int32_t		vl_len_;
	uint32_t	attrs;			/* reserved for future use */
	int64_t		nitems;
	int64_t		sum;
} kagg_state__psum_int_packed;

typedef struct
{
	int32_t		vl_len_;
	uint32_t	attrs;			/* reserved for future use */
	int64_t		nitems;
	float8_t	sum;
} kagg_state__psum_fp_packed;

#define __PAGG_NUMERIC_ATTRS__WEIGHT	0x00ffffU
#define __PAGG_NUMERIC_ATTRS__NAN		0x010000U	/* NaN */
#define __PAGG_NUMERIC_ATTRS__PINF		0x020000U	/* +Inf */
#define __PAGG_NUMERIC_ATTRS__NINF		0x040000U	/* -Inf */
#define __PAGG_NUMERIC_ATTRS__MASK		0x070000U	/* Nan|+Inf|-Inf */
typedef struct
{
	int32_t		vl_len_;
	uint32_t	attrs;
	uint64_t	nitems;
	int128_packed_t sum;		/* int128 or uint64 x2 */
} kagg_state__psum_numeric_packed;

typedef struct
{
	int32_t		vl_len_;
	uint32_t	attrs;			/* reserved for future use */
	int64_t		nitems;
	float8_t	sum_x;
	float8_t	sum_x2;
} kagg_state__stddev_packed;

typedef struct
{
	int32_t		vl_len_;
	uint32_t	attrs;			/* reserved for future use */
	int64_t		nitems;
	float8_t	sum_x;
	float8_t	sum_xx;
	float8_t	sum_y;
	float8_t	sum_yy;
	float8_t	sum_xy;
} kagg_state__covar_packed;

typedef struct
{
	uint16_t	action;			/* any of KAGG_ACTION__* */
	int16_t		arg0_slot_id;	/* arg0 of partial aggregate function */
	int16_t		arg1_slot_id;	/* arg1 of partial aggregate function */
	int16_t		filter_slot_id;	/* if non-negative, slot-id of the filter */
	int32_t		typmod;			/* typmod of 1st arg - used for numeric */
} kern_aggregate_desc;


#define KSORT_KEY_ATTR__NULLS_FIRST			0x0400U
#define KSORT_KEY_ATTR__ORDER_ASC			0x8000U
#define KSORT_KEY_KIND__MASK				0x03ffU
#define KSORT_KEY_KIND__SHIFT				16
#define KSORT_KEY_KIND__VREF				0
#define KSORT_KEY_KIND__PMINMAX_INT64		1
#define KSORT_KEY_KIND__PMINMAX_FP64		2
#define KSORT_KEY_KIND__PSUM_INT64			3
#define KSORT_KEY_KIND__PSUM_FP64			4
#define KSORT_KEY_KIND__PSUM_NUMERIC		5
#define KSORT_KEY_KIND__PAVG_INT64			6
#define KSORT_KEY_KIND__PAVG_FP64			7
#define KSORT_KEY_KIND__PAVG_NUMERIC		8
#define KSORT_KEY_KIND__PVARIANCE_SAMP		9
#define KSORT_KEY_KIND__PVARIANCE_POP		10
#define KSORT_KEY_KIND__PCOVAR_CORR			11
#define KSORT_KEY_KIND__PCOVAR_SAMP			12
#define KSORT_KEY_KIND__PCOVAR_POP			13
#define KSORT_KEY_KIND__PCOVAR_AVGX			14
#define KSORT_KEY_KIND__PCOVAR_AVGY			15
#define KSORT_KEY_KIND__PCOVAR_COUNT		16
#define KSORT_KEY_KIND__PCOVAR_INTERCEPT	17
#define KSORT_KEY_KIND__PCOVAR_REGR_R2		18
#define KSORT_KEY_KIND__PCOVAR_REGR_SLOPE	19
#define KSORT_KEY_KIND__PCOVAR_REGR_SXX		20
#define KSORT_KEY_KIND__PCOVAR_REGR_SXY		21
#define KSORT_KEY_KIND__PCOVAR_REGR_SYY		22
#define KSORT_KEY_KIND__NITEMS				23

#define KSORT_WINDOW_FUNC__ROW_NUMBER		'n'
#define KSORT_WINDOW_FUNC__RANK				'r'
#define KSORT_WINDOW_FUNC__DENSE_RANK		'd'

typedef struct
{
	uint16_t	kind;			/* any of KSORT_KEY_KIND__* */
	int8_t		nulls_first;	/* true, if NULLs first */
	int8_t		order_asc;		/* true, if ORDER ASC */
	uint16_t	src_anum;		/* source attribute number of KDS */
	uint16_t	buf_offset;		/* if not KSORT_KEY_KIND__VREF, it means offset of
								 * the temporary calculated sorting key.
								 * location is:
								 * ((char *)&tupitem->htup + tupitem->t_len + key_offset)
								 */
	TypeOpCode	key_type_code;
	const struct xpu_datum_operators *key_ops;
} kern_sortkey_desc;

typedef struct
{
	int16_t		vl_resno;		/* resno of the source */
	int16_t		vl_slot_id;		/* slot-id to load the datum  */
} kern_varload_desc;

typedef struct
{
	int32_t		vm_offset;		/* source & destination kvecs-offset */
	int16_t		vm_slot_id;		/* source slot-id. */
	bool		vm_from_xdatum; /* true, if variable is originated from the current
								 * depth, so values must be copied from the xdatum,
								 * not kvecs-buffer.
								 */
} kern_varmove_desc;

struct kern_varslot_desc
{
	TypeOpCode	vs_type_code;
	bool		vs_typbyval;
	int8_t		vs_typalign;
	int16_t		vs_typlen;
	int32_t		vs_typmod;
	int32_t		vs_offset;		/* offset of kvec-buffer, if any. elsewhere -1. */
	uint16_t	idx_subfield;	/* offset to the subfield descriptor */
	uint16_t	num_subfield;	/* number of the subfield (array or composite) */
	const struct xpu_datum_operators *vs_ops;
};

typedef struct
{
	int16_t		fb_src_depth;	/* source depth of this fallback variable */
	int16_t		fb_src_resno;	/* source resno of this fallback variable */
	int16_t		fb_dst_resno;	/* resno of the host scan-slot */
	int16_t		fb_max_depth;	/* last depth that references this variable */
	uint16_t	fb_slot_id;		/* kernel slot-id of this fallback variable */
	int32_t		fb_kvec_offset;	/* kvec's buffer offset */
} kern_fallback_desc;

#define KERN_EXPRESSION_MAGIC			(0x4b657870)	/* 'K' 'e' 'x' 'p' */

#define KEXP_FLAG__IS_PUSHED_DOWN		0x0001U

#define SPECIAL_DEPTH__PREAGG_FINAL		(-2)

struct kern_expression
{
	uint32_t		len;			/* length of this expression */
	TypeOpCode		exptype;
	const xpu_datum_operators *expr_ops;
	uint32_t		expflags;		/* mask of KEXP_FLAG__* above */
	FuncOpCode		opcode;
	xpu_function_t	fn_dptr;		/* to be set by xPU service */
	uint32_t		nr_args;		/* number of arguments */
	uint32_t		args_offset;	/* offset to the arguments */
	union {
		char			data[1]			__MAXALIGNED__;
		struct {
			Oid			const_type;
			bool		const_isnull;
			char		const_value[1]	__MAXALIGNED__;
		} c;		/* ConstExpr */
		struct {
			uint32_t	param_id;
			char		__data[1];
		} p;		/* ParamExpr */
		struct {
			/* if var_offset < 0, it means variables should be loaded from
			 * the kcxt->kvars_values[] entries, used for the new values
			 * loaded in this depth, or temporary variables.
			 * elsewhere, 'var_offset' points particular region on the
			 * kernel vectorized values buffer (kvecs).
			 */
			int32_t		var_offset;		/* kvec's buffer offset */
			uint16_t	var_slot_id;	/* kvar's slot-id */
			char		__data[1];
		} v;		/* VarExpr */
		struct {
			uint32_t	case_comp;		/* key value to be compared, if any */
			uint32_t	case_else;		/* ELSE clause, if any */
			char		data[1]			__MAXALIGNED__;
		} casewhen;	/* Case-When */
		struct {
			uint16_t	elem_slot_id;	/* slot-id of temporary array element */
			char		data[1]			__MAXALIGNED__;
		} saop;		/* ScalarArrayOp */
		struct {
			int			depth;
			int			nitems;
			kern_varload_desc desc[1];
		} load;		/* VarLoads */
		struct {
			int			depth;		/* kvecs-buffer to write-out */
			int			nitems;
			kern_varmove_desc desc[1];
		} move;
		struct {
			uint32_t	gist_oid;		/* OID of GiST index (for EXPLAIN) */
			int16_t		gist_depth;		/* depth of index tuple */
			uint16_t	htup_slot_id;	/* slot_id to save htup pointer */
			int32_t		htup_offset;	/* kvec's buffer offset of htup pointer */
			kern_varload_desc ivar_desc; /* index-var load descriptor */
			char		data[1]			__MAXALIGNED__;
		} gist;		/* GiSTEval */
		struct {
			uint16_t	sv_slot_id;
			char		data[1]			__MAXALIGNED__;
		} save;		/* SaveExpr */
		struct {
			int			nattrs;
			kern_aggregate_desc desc[1];
		} pagg;		/* PreAggs */
		struct {
			uint32_t	hash;			/* kexp for hash-value calculation */
			int			nattrs;
			uint16_t	slot_id[1];
		} proj;		/* Projection */
		struct {
			int			nkeys;
			bool		needs_finalization;
			char		window_rank_func;		/* one of KSORT_WINDOW_FUNC__* */
			uint32_t	window_rank_limit;		/* rank() limit, if any */
			uint16_t	window_partby_nkeys;	/* # of partition keys */
			uint16_t	window_orderby_nkeys;	/* # of order-by keys */
			kern_sortkey_desc desc[1];
		} sort;		/* Sort */
		struct {
			uint32_t	npacked;	/* number of packed sub-expressions; including
									 * logical NULLs (npacked may be larger than
									 * nr_args) */
			uint32_t	offset[1];	/* offset to sub-expressions */
		} pack;		/* Packed */
	} u;
};

#define EXEC_KERN_EXPRESSION(__kcxt,__kexp,__retval)	\
	(__kexp)->fn_dptr((__kcxt),(__kexp),(xpu_datum_t *)__retval)

INLINE_FUNCTION(bool)
__KEXP_IS_VALID(const kern_expression *kexp,
				const kern_expression *karg)
{
	if (kexp)
	{
		if ((char *)karg >= kexp->u.data &&
			(char *)karg + karg->len <= (char *)kexp + kexp->len)
		{
			uint32_t   *magic = (uint32_t *)
				((char *)karg + karg->len - sizeof(uint32_t));
			if (*magic == (KERN_EXPRESSION_MAGIC
						   ^ ((uint32_t)karg->exptype << 6)
						   ^ ((uint32_t)karg->opcode << 14)))
			{
				/* ok, it looks valid */
				return true;
			}
		}
	}
	else if (!karg)
	{
		/* both kexp and karg is NULL */
		return true;
	}
	return false;
}
#define KEXP_IS_VALID(__karg,EXPTYPE)				\
	(__KEXP_IS_VALID(kexp,(__karg)) &&				\
	 (__karg)->exptype == TypeOpCode__##EXPTYPE)
#define KEXP_FIRST_ARG(__kexp)											\
	(((__kexp)->nr_args > 0 && (__kexp)->args_offset > 0)				\
	 ? ((kern_expression *)((char *)(__kexp) + (__kexp)->args_offset))	\
	 : NULL)
#define KEXP_NEXT_ARG(__karg)											\
	((kern_expression *)((char *)(__karg) + MAXALIGN((__karg)->len)))

#define SizeOfKernExpr(__PAYLOAD_SZ)						\
	(offsetof(kern_expression, u.data) + (__PAYLOAD_SZ))
#define SizeOfKernExprVar					\
	(offsetof(kern_expression, u.v.__data))
typedef struct {
	FuncOpCode		func_opcode;
	xpu_function_t	func_dptr;
} xpu_function_catalog_entry;

EXTERN_DATA(xpu_function_catalog_entry, builtin_xpu_functions_catalog[]);

/* device function hash for xPU service */
typedef struct xpu_func_hash_entry	xpu_func_hash_entry;
struct xpu_func_hash_entry
{
	xpu_func_hash_entry *next;
	xpu_function_catalog_entry cat;
};
typedef struct
{
	uint32_t	nitems;
	uint32_t	nslots;
	xpu_func_hash_entry *slots[1];	/* variable */
} xpu_func_hash_table;

/*
 * PG-Strom Command Tag
 */
#define XpuCommandTag__Success				0
#define XpuCommandTag__Error				1
#define XpuCommandTag__SuccessHalfWay		2
#define XpuCommandTag__OpenSession			100
#define XpuCommandTag__XpuTaskExec			110
#define XpuCommandTag__XpuTaskExecGpuCache	111
#define XpuCommandTag__XpuTaskFinal			119
#define XpuCommandMagicNumber				0xdeadbeafU

/*
 * kern_session_info - A set of immutable data during query execution
 * (like, transaction info, timezone, parameter buffer).
 */
typedef struct kern_session_info
{
	uint64_t	query_plan_id;		/* unique-id to use per-query buffer */
	uint32_t	kcxt_kvars_nrooms;	/* length of kvars_slot_desc[] array*/
	uint32_t	kcxt_kvars_nslots;	/* number of kvars slot */
	uint32_t	kcxt_kvars_defs;	/* offset of kvars_slot_desc[] array */
	uint32_t	kcxt_kvecs_bufsz;	/* length of kvecs buffer */
	uint32_t	kcxt_kvecs_ndims;	/* =(num_rels + 2) */
	uint32_t	kcxt_extra_bufsz;	/* length of vlbuf[] */
	uint32_t	cuda_stack_size;	/* estimated stack size */
	uint32_t	xpu_task_flags;		/* mask of device flags */
	gpumask_t	optimal_gpus;		/* mask of schedulable GPUs */
	/* xpucode for this session */
	uint32_t	xpucode_load_vars_packed;
	uint32_t	xpucode_move_vars_packed;
	uint32_t	xpucode_scan_quals;
	uint32_t	xpucode_join_quals_packed;
	uint32_t	xpucode_hash_values_packed;
	uint32_t	xpucode_gist_evals_packed;
	uint32_t	xpucode_projection;
	uint32_t	xpucode_groupby_keyhash;
	uint32_t	xpucode_groupby_keyload;
	uint32_t	xpucode_groupby_keycomp;
	uint32_t	xpucode_groupby_actions;
	uint32_t	xpucode_gpusort_keydesc;

	/* database session info */
	int64_t		hostEpochTimestamp;		/* = SetEpochTimestamp() */
	uint64_t	xactStartTimestamp;		/* timestamp when transaction start */
	uint32_t	session_xact_state;		/* offset to SerializedTransactionState */
	uint32_t	session_timezone;		/* offset to pg_tz */
	uint32_t	session_encode;			/* offset to xpu_encode_info;
										 * !! function pointer must be set by server */
	int32_t		session_currency_frac_digits;	/* copy of lconv::frac_digits */
	/* projection kds definition */
	uint32_t	projection_kds_dst;		/* header portion of kds_dst */

	/* join inner buffer */
	uint32_t	pgsql_port_number;		/* = PostPortNumber */
	uint32_t	pgsql_plan_node_id;		/* = Plan->plan_node_id */
	uint32_t	join_inner_handle;		/* key of join inner buffer */

	/* group-by final buffer */
	uint32_t	groupby_kds_final;		/* header portion of kds_final */
	uint32_t	groupby_prepfn_bufsz;	/* buffer size for preagg functions */
	float4_t	groupby_ngroups_estimation; /* planne's estimation of ngroups */

	/* gpu-sort final buffer */
	uint32_t	gpusort_htup_margin;	/* extra space at tail of the final
										 * kern_tupitem for finalization */
	uint32_t	gpusort_limit_count;	/* limit-pushdown, if positive */
	/* fallback buffer */
	uint32_t	fallback_kds_head;		/* offset to kds_fallback (header) */
	uint32_t	fallback_desc_defs;		/* offset to kern_fallback_desc array */
	uint32_t	fallback_desc_nitems;	/* number of kern_fallback_desc items */

	/* executor parameter buffer */
	uint32_t	nparams;	/* number of parameters */
	uint32_t	poffset[1];	/* offset of params */
} kern_session_info;

typedef struct {
	uint32_t	kds_src_pathname;	/* offset to const char *pathname */
	uint32_t	kds_src_iovec;		/* offset to strom_io_vector */
	uint32_t	kds_src_offset;		/* offset to kds_src */
	int32_t		scan_repeat_id;		/* current repeat count */
	char		data[1]				__MAXALIGNED__;
} kern_exec_task;

typedef struct {
	uint32_t	chunks_offset;		/* offset of kds_dst array */
	uint32_t	chunks_nitems;		/* number of kds_dst items */
	uint32_t	ojmap_offset;		/* offset of outer-join-map */
	uint32_t	ojmap_length;		/* length of outer-join-map */
	bool		right_outer_join;	/* true, if CPU should exex RIGHT-OUTER-JOIN */
	bool		final_plan_task;	/* true, if it is final response */
	uint32_t	final_nitems;		/* final buffer's nitems, if any */
	uint64_t	final_usage;		/* final buffer's usage, if any */
	uint64_t	final_total;		/* final buffer's total size, if any */
	/* statistics */
	uint32_t	npages_direct_read;	/* # of pages read by GPU-Direct Storage */
	uint32_t	npages_vfs_read;	/* # of pages read by VFS (fallback) */
	uint32_t	nitems_raw;		/* # of visible rows kept in the relation */
	uint32_t	nitems_in;		/* # of result rows in depth-0 after WHERE-clause */
	uint32_t	nitems_out;		/* # of result rows in final depth before host quals */
	uint32_t	num_rels;
	struct {
		uint32_t	nitems_roj;	/* # of generated rows by RIGHT-OUTER-JOIN (if any) */
		uint32_t	nitems_gist;/* # of results rows by GiST index (if any) */
		uint32_t	nitems_out;	/* # of results rows by JOIN in this depth */
	} stats[1];
} kern_exec_results;

typedef struct
{
	kern_errorbuf		error;		/* original error in kernel space */
	/* statistics */
	uint32_t			npages_direct_read;
	uint32_t			npages_vfs_read;
	kern_data_store		kds_src;
} kern_cpu_fallback;

#ifndef ILIST_H
typedef struct dlist_node
{
	struct dlist_node *prev;
	struct dlist_node *next;
} dlist_node;
typedef struct dlist_head
{
	dlist_node		head;
} dlist_head;
#endif

typedef struct
{
	uint32_t	magic;
	uint32_t	tag;
	uint64_t	length;
	void	   *priv;
	dlist_node	chain;
	union {
		kern_errorbuf		error;
		kern_session_info	session;
		kern_exec_task		task;
		kern_exec_results	results;
		kern_cpu_fallback	fallback;
	} u;
} XpuCommand;

/*
 * kern_session_info utility functions.
 */
INLINE_FUNCTION(const kern_data_store *)
SESSION_KDS_DST_HEAD(const kern_session_info *session)
{
	const kern_data_store *kds_dst_head = NULL;

	if (session->projection_kds_dst > 0)
		kds_dst_head = (const kern_data_store *)
			((char *)session + session->projection_kds_dst);

	return kds_dst_head;
}

INLINE_FUNCTION(bool)
SESSION_SUPPORTS_CPU_FALLBACK(const kern_session_info *session)
{
	return (session->fallback_kds_head != 0);
}

INLINE_FUNCTION(kern_varslot_desc *)
SESSION_KVARS_SLOT_DESC(const kern_session_info *session)
{
	if (session->kcxt_kvars_nslots == 0 ||
		session->kcxt_kvars_defs == 0)
		return NULL;

	return (kern_varslot_desc *)((char *)session + session->kcxt_kvars_defs);
}

INLINE_FUNCTION(kern_expression *)
__PICKUP_PACKED_KEXP(const kern_expression *kexp, int dindex)
{
	kern_expression *karg;
	uint32_t	offset;

	assert(kexp->opcode == FuncOpCode__Packed);
	if (dindex < 0 || dindex >= kexp->u.pack.npacked)
		return NULL;
	offset = kexp->u.pack.offset[dindex];
	if (offset == 0)
		return NULL;
	karg = (kern_expression *)((char *)kexp + offset);
	assert(karg->u.data < (char *)kexp + kexp->len &&
		   (char *)karg + karg->len <= (char *)kexp + kexp->len);
	return karg;
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_LOAD_VARS(const kern_session_info *session, int depth)
{
	kern_expression *kexp;
	kern_expression *karg;

	if (session->xpucode_load_vars_packed == 0)
		return NULL;
	kexp = (kern_expression *)
		((char *)session + session->xpucode_load_vars_packed);
	if (depth < 0)
		return kexp;
	karg = __PICKUP_PACKED_KEXP(kexp, depth);
	assert(!karg || (karg->opcode == FuncOpCode__LoadVars &&
					 karg->exptype == TypeOpCode__int4));
	return karg;
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_MOVE_VARS(const kern_session_info *session, int depth)
{
	kern_expression *kexp;
	kern_expression *karg;

	if (session->xpucode_move_vars_packed == 0)
		return NULL;
	kexp = (kern_expression *)
		((char *)session + session->xpucode_move_vars_packed);
	if (depth < 0)
		return kexp;
	karg = __PICKUP_PACKED_KEXP(kexp, depth);
	assert(!karg || (karg->opcode == FuncOpCode__MoveVars &&
					 karg->exptype == TypeOpCode__int4));
	return karg;
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_SCAN_QUALS(const kern_session_info *session)
{
	if (session->xpucode_scan_quals == 0)
		return NULL;
	return (kern_expression *)((char *)session + session->xpucode_scan_quals);
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_JOIN_QUALS(const kern_session_info *session, int depth)
{
	kern_expression *kexp;
	kern_expression *karg;

	if (session->xpucode_join_quals_packed == 0)
		return NULL;
	kexp = (kern_expression *)
		((char *)session + session->xpucode_join_quals_packed);
	if (depth < 0)
		return kexp;
	karg = __PICKUP_PACKED_KEXP(kexp, depth);
	assert(!karg || (karg->opcode == FuncOpCode__JoinQuals &&
					 karg->exptype == TypeOpCode__bool));
	return karg;
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_HASH_VALUE(const kern_session_info *session, int depth)
{
	kern_expression *kexp;
	kern_expression *karg;

	if (session->xpucode_hash_values_packed == 0)
		return NULL;
	kexp = (kern_expression *)
		((char *)session + session->xpucode_hash_values_packed);
	if (depth < 0)
		return kexp;
	karg = __PICKUP_PACKED_KEXP(kexp, depth);
	assert(!karg || (karg->opcode == FuncOpCode__HashValue &&
					 karg->exptype == TypeOpCode__int4));
	return karg;
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_GIST_EVALS(const kern_session_info *session, int depth)
{
	kern_expression *kexp;
	kern_expression *karg;

	if (session->xpucode_gist_evals_packed == 0)
		return NULL;
	kexp = (kern_expression *)
		((char *)session + session->xpucode_gist_evals_packed);
	if (depth < 0)
		return kexp;
	karg = __PICKUP_PACKED_KEXP(kexp, depth);
	assert(!karg || karg->opcode == FuncOpCode__GiSTEval);
	return karg;
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_PROJECTION(const kern_session_info *session)
{
	kern_expression *kexp = NULL;

	if (session->xpucode_projection)
	{
		kexp = (kern_expression *)
			((char *)session + session->xpucode_projection);
		assert(kexp->opcode == FuncOpCode__Projection &&
			   kexp->exptype == TypeOpCode__int4);
	}
	return kexp;
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_GROUPBY_KEYHASH(const kern_session_info *session)
{
	kern_expression *kexp = NULL;

	if (session->xpucode_groupby_keyhash)
	{
		kexp = (kern_expression *)
			((char *)session + session->xpucode_groupby_keyhash);
		assert(kexp->opcode == FuncOpCode__HashValue &&
			   kexp->exptype == TypeOpCode__int4);
	}
	return kexp;
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_GROUPBY_KEYLOAD(const kern_session_info *session)
{
	kern_expression *kexp = NULL;

	if (session->xpucode_groupby_keyload)
	{
		kexp = (kern_expression *)
			((char *)session + session->xpucode_groupby_keyload);
		assert(kexp->opcode == FuncOpCode__LoadVars &&
			   kexp->exptype == TypeOpCode__int4);
	}
	return kexp;
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_GROUPBY_KEYCOMP(const kern_session_info *session)
{
	kern_expression *kexp = NULL;

	if (session->xpucode_groupby_keycomp)
	{
		kexp = (kern_expression *)
			((char *)session + session->xpucode_groupby_keycomp);
		assert(kexp->exptype == TypeOpCode__bool);
	}
	return kexp;
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_GROUPBY_ACTIONS(const kern_session_info *session)
{
	kern_expression *kexp = NULL;

	if (session->xpucode_groupby_actions)
	{
		kexp = (kern_expression *)
			((char *)session + session->xpucode_groupby_actions);
		assert(kexp->opcode == FuncOpCode__AggFuncs &&
			   kexp->exptype == TypeOpCode__int4);
	}
	return kexp;
}

INLINE_FUNCTION(kern_expression *)
SESSION_KEXP_GPUSORT_KEYDESC(const kern_session_info *session)
{
	kern_expression *kexp = NULL;

	if (session->xpucode_gpusort_keydesc)
	{
		kexp = (kern_expression *)
			((char *)session + session->xpucode_gpusort_keydesc);
		assert(kexp->opcode == FuncOpCode__SortKeys &&
			   kexp->exptype == TypeOpCode__int4);
	}
	return kexp;
}

/* see access/transam/xact.c */
typedef struct
{
	int			xactIsoLevel;
	bool		xactDeferrable;
	FullTransactionId topFullTransactionId;
	FullTransactionId currentFullTransactionId;
	CommandId	currentCommandId;
	int			nParallelCurrentXids;
	TransactionId parallelCurrentXids[1];	/* variable */
} SerializedTransactionState;

INLINE_FUNCTION(SerializedTransactionState *)
SESSION_XACT_STATE(kern_session_info *session)
{
	if (session->session_xact_state == 0)
		return NULL;
	return (SerializedTransactionState *)((char *)session + session->session_xact_state);
}

INLINE_FUNCTION(struct pg_tz *)
SESSION_TIMEZONE(kern_session_info *session)
{
	if (session->session_timezone == 0)
		return NULL;
	return (struct pg_tz *)((char *)session + session->session_timezone);
}

INLINE_FUNCTION(struct xpu_encode_info *)
SESSION_ENCODE(kern_session_info *session)
{
	if (session->session_encode == 0)
		return NULL;
	return (struct xpu_encode_info *)((char *)session + session->session_encode);
}

/* ----------------------------------------------------------------
 *
 * Template for xPU connection commands receive
 *
 * ----------------------------------------------------------------
 */
#define TEMPLATE_XPU_CONNECT_RECEIVE_COMMANDS(__XPU_PREFIX)				\
	static int															\
	__XPU_PREFIX##ReceiveCommands(int sockfd, void *priv)				\
	{																	\
		char		buffer_local[10000];								\
		char	   *buffer;												\
		size_t		bufsz, offset;										\
		ssize_t		nbytes;												\
		int			recv_flags;											\
		int			count = 0;											\
		XpuCommand *curr = NULL;										\
																		\
	restart:															\
		buffer = buffer_local;											\
		bufsz  = sizeof(buffer_local);									\
		offset = 0;														\
		recv_flags = MSG_DONTWAIT;										\
		curr   = NULL;													\
																		\
		for (;;)														\
		{																\
			nbytes = recv(sockfd,										\
						  buffer + offset,								\
						  bufsz - offset,								\
						  recv_flags);									\
			if (nbytes < 0)												\
			{															\
				if (errno == EINTR)										\
					continue;											\
				if (errno == EAGAIN || errno == EWOULDBLOCK)			\
				{														\
					/*													\
					 * If we are in the halfway through the read of		\
					 * XpuCommand fraction, we have to wait for			\
					 * the complete XpuCommand.							\
					 * (The peer side should send the entire command	\
					 * very soon.) Elsewhere, we have no queued			\
					 * XpuCommand right now.							\
					 */													\
					if (!curr && offset == 0)							\
						return count;									\
					/* next recv(2) shall be blocking call */			\
					recv_flags = 0;										\
					continue;											\
				}														\
				fprintf(stderr, "[%s] failed on recv(2): %m\n",			\
						__FUNCTION__);									\
				return -1;												\
			}															\
			else if (nbytes == 0)										\
			{															\
				/* end of the stream */									\
				if (curr || offset > 0)									\
				{														\
					fprintf(stderr, "[%s] connection closed in the halfway through XpuCommands read\n", \
							__FUNCTION__);								\
					return -1;											\
				}														\
				return count;											\
			}															\
																		\
			offset += nbytes;											\
			if (!curr)													\
			{															\
				XpuCommand *temp, *xcmd;								\
			next:														\
				if (offset < offsetof(XpuCommand, u))					\
				{														\
					if (buffer != buffer_local)							\
					{													\
						memmove(buffer_local, buffer, offset);			\
						buffer = buffer_local;							\
						bufsz  = sizeof(buffer_local);					\
					}													\
					recv_flags = 0;		/* next recv(2) is blockable */	\
					continue;											\
				}														\
				temp = (XpuCommand *)buffer;							\
				if (temp->length <= offset)								\
				{														\
					assert(temp->magic == XpuCommandMagicNumber);		\
					xcmd = __XPU_PREFIX##AllocCommand(priv, temp->length); \
					if (!xcmd)											\
					{													\
						fprintf(stderr, "[%s] out of memory (sz=%lu): %m\n", \
								__FUNCTION__, temp->length);			\
						return -1;										\
					}													\
					memcpy(xcmd, temp, temp->length);					\
					__XPU_PREFIX##AttachCommand(priv, xcmd);			\
					count++;											\
																		\
					if (temp->length == offset)							\
						goto restart;									\
					/* read remained portion, if any */					\
					buffer += temp->length;								\
					offset -= temp->length;								\
					goto next;											\
				}														\
				else													\
				{														\
					curr = __XPU_PREFIX##AllocCommand(priv, temp->length); \
					if (!curr)											\
					{													\
						fprintf(stderr, "[%s] out of memory (sz=%lu): %m\n", \
								__FUNCTION__, temp->length);			\
						return -1;										\
					}													\
					memcpy(curr, temp, offset);							\
					buffer = (char *)curr;								\
					bufsz  = temp->length;								\
					recv_flags = 0;		/* blocking enabled */			\
				}														\
			}															\
			else if (offset >= curr->length)							\
			{															\
				assert(curr->magic == XpuCommandMagicNumber);			\
				assert(curr->length == offset);							\
				__XPU_PREFIX##AttachCommand(priv, curr);				\
				count++;												\
				goto restart;											\
			}															\
		}																\
		fprintf(stderr, "[%s] Bug? unexpected loop break\n",			\
				__FUNCTION__);											\
		return -1;														\
	}

/* ----------------------------------------------------------------
 *
 * Entrypoint for LoadVars
 *
 * ----------------------------------------------------------------
 */
EXTERN_FUNCTION(int)
kern_form_heaptuple(kern_context *kcxt,
					const kern_expression *kproj,
					const kern_data_store *kds_dst,
					HeapTupleHeaderData *htup);
EXTERN_FUNCTION(int)
kern_estimate_heaptuple(kern_context *kcxt,
						const kern_expression *kproj,
						const kern_data_store *kds_dst);
EXTERN_FUNCTION(const void *)
kern_fetch_heaptuple_attr(kern_context *kcxt,
						  const kern_data_store *kds,
						  const kern_tupitem *titem, int anum);
EXTERN_FUNCTION(bool)
ExecLoadVarsHeapTuple(kern_context *kcxt,
					  const kern_expression *kexp_load_vars,
					  int depth,
					  const kern_data_store *kds,
					  const HeapTupleHeaderData *htup);
EXTERN_FUNCTION(bool)
ExecLoadVarsOuterRow(kern_context *kcxt,
					 const kern_expression *kexp_load_vars,
					 const kern_expression *kexp_scan_quals,
					 const kern_data_store *kds_src,
					 const HeapTupleHeaderData *htup);
EXTERN_FUNCTION(bool)
ExecLoadVarsOuterArrow(kern_context *kcxt,
					   const kern_expression *kexp_load_vars,
					   const kern_expression *kexp_scan_quals,
					   const kern_data_store *kds_src,
					   uint32_t kds_index);
EXTERN_FUNCTION(bool)
ExecLoadVarsOuterColumn(kern_context *kcxt,
						const kern_expression *kexp_load_vars,
						const kern_expression *kexp_scan_quals,
						const kern_data_store *kds,
						const kern_data_extra *extra,
						uint32_t kds_index);
PUBLIC_FUNCTION(bool)
ExecLoadKeysFromGroupByFinal(kern_context *kcxt,
							 const kern_data_store *kds_final,
							 const kern_tupitem *tupitem,
							 const kern_expression *kexp_groupby_actions);
EXTERN_FUNCTION(bool)
ExecMoveKernelVariables(kern_context *kcxt,
						const kern_expression *kexp_move_vars,
                        char *dst_kvec_buffer,
                        int dst_kvec_id);
EXTERN_FUNCTION(bool)
ExecGpuJoinQuals(kern_context *kcxt,
				 const kern_expression *kexp_join_quals,
				 int *p_status);
EXTERN_FUNCTION(bool)
ExecGpuJoinOtherQuals(kern_context *kcxt,
					  const kern_expression *kexp_join_quals,
					  bool *p_status);
EXTERN_FUNCTION(uint64_t)
ExecGiSTIndexGetNext(kern_context *kcxt,
					 const kern_data_store *kds_hash,
					 const kern_data_store *kds_gist,
					 const kern_expression *kexp_gist,
					 uint64_t l_state);
EXTERN_FUNCTION(bool)
ExecGiSTIndexPostQuals(kern_context *kcxt,
					   int depth,
					   const kern_data_store *kds_hash,
					   const kern_expression *kexp_gist,
					   const kern_expression *kexp_load,
					   const kern_expression *kexp_join);
EXTERN_FUNCTION(int)
ExecKernProjection(kern_context *kcxt,
				   kern_expression *kexp,
				   uint32_t *combuf,
				   kern_data_store *kds_outer,
				   kern_data_extra *kds_extra,
				   int num_inners,
				   kern_data_store **kds_inners);
EXTERN_FUNCTION(bool)
HandleErrorIfCpuFallback(kern_context *kcxt,
						 int depth,
						 uint64_t l_state,
						 bool matched);

/* ----------------------------------------------------------------
 *
 * PostgreSQL Device Functions (Built-in)
 *
 * ----------------------------------------------------------------
 */
#define FUNC_OPCODE(a,b,c,NAME,d,e)			\
	EXTERN_FUNCTION(bool) pgfn_##NAME(XPU_PGFUNCTION_ARGS);
#define DEVONLY_FUNC_OPCODE(a,NAME,b,c,d)	\
	EXTERN_FUNCTION(bool) pgfn_##NAME(XPU_PGFUNCTION_ARGS);
#include "xpu_opcodes.h"

/* ----------------------------------------------------------------
 *
 * Common XPU functions
 *
 * ----------------------------------------------------------------
 */
EXTERN_FUNCTION(void)
pg_kern_ereport(kern_context *kcxt);	/* only host code */
EXTERN_FUNCTION(uint32_t)
pg_hash_any(const void *ptr, int sz);
INLINE_FUNCTION(uint32_t)
pg_hash_merge(uint32_t hash_prev, uint32_t hash_next)
{
	return ((hash_prev >> 3) | (hash_prev << 29)) ^ hash_next;
}

/* ----------------------------------------------------------------
 *
 * Definitions for xPU JOIN
 *
 * ----------------------------------------------------------------
 */
typedef struct
{
	int32_t			inner_depth;	/* partitioned depth */
	int32_t			hash_divisor;	/* divisor for the hash-value */
	struct {
		gpumask_t	available_gpus;	/* set of GPUs for this partition */
		kern_data_store *kds_in;	/* used by GPU-service */
	} parts[1];
} kern_buffer_partitions;

struct kern_multirels
{
	size_t		length;				/* total length of kern_multirels */
	size_t		ojmap_sz;			/* length of outer-join map */
	uint64_t	kbuf_part_offset;	/* offset to kern_buffer_partitions, if any */
	uint32_t	num_rels;
	struct
	{
		kern_data_store *kds_in;	/* pointer to KDS-inner (if non-partitioned) */
		kern_buffer_partitions *kbuf_parts; /* partition descriptor */
		/* --- aboves are valid only GPU-service --- */
		uint64_t	kds_offset;		/* offset to KDS */
		uint64_t	ojmap_offset;	/* offset to outer-join map, if any */
		uint64_t	gist_offset;	/* offset to GiST-index pages, if any */
		bool		is_nestloop;	/* true, if NestLoop */
		bool		left_outer;		/* true, if JOIN_LEFT or JOIN_FULL */
		bool		right_outer;	/* true, if JOIN_RIGHT or JOIN_FULL */
		bool		pinned_buffer;	/* true, if it uses pinned-buffer */
		uint64_t	buffer_id;		/* key to lookup pinned inner-buffer */
	} chunks[1];
};
typedef struct kern_multirels	kern_multirels;

INLINE_FUNCTION(kern_data_store *)
KERN_MULTIRELS_INNER_KDS(kern_multirels *kmrels, int depth)
{
#ifdef __CUDACC__
	kern_data_store *kds_in;

	assert(depth > 0 && depth <= kmrels->num_rels);
	kds_in = kmrels->chunks[depth-1].kds_in;
	if (!kds_in)
	{
		kern_buffer_partitions *kbuf_parts = kmrels->chunks[depth-1].kbuf_parts;

		if (kbuf_parts)
		{
			int		reminder = stromTaskProp__partition_reminder;

			assert(reminder >= 0 && reminder < kbuf_parts->hash_divisor);
			kds_in = kbuf_parts->parts[reminder].kds_in;
		}
	}
	return kds_in;
#else
	uint64_t	pos;

	assert(depth > 0 && depth <= kmrels->num_rels);
	pos = kmrels->chunks[depth-1].kds_offset;
	return (kern_data_store *)(pos == 0 ? NULL : ((char *)kmrels + pos));
#endif
}

INLINE_FUNCTION(bool *)
KERN_MULTIRELS_OUTER_JOIN_MAP(kern_multirels *kmrels, int depth)
{
	uint64_t	offset;

	assert(depth > 0 && depth <= kmrels->num_rels);
	offset = kmrels->chunks[depth-1].ojmap_offset;
	return (bool *)(offset == 0 ? NULL : ((char *)kmrels + offset));
}

INLINE_FUNCTION(bool *)
KERN_MULTIRELS_GPU_OUTER_JOIN_MAP(kern_multirels *kmrels, int depth,
								  uint32_t cuda_dindex)
{
	uint64_t	offset;

	assert(depth > 0 && depth <= kmrels->num_rels);
	offset = kmrels->chunks[depth-1].ojmap_offset;
	if (offset == 0)
		return NULL;
	offset += kmrels->ojmap_sz * cuda_dindex;
	return (bool *)((char *)kmrels + offset);
}

INLINE_FUNCTION(kern_data_store *)
KERN_MULTIRELS_GIST_INDEX(kern_multirels *kmrels, int depth)
{
	uint64_t	offset;

	assert(depth > 0 && depth <= kmrels->num_rels);
	offset = kmrels->chunks[depth-1].gist_offset;
	return (kern_data_store *)(offset == 0 ? NULL : ((char *)kmrels + offset));
}

INLINE_FUNCTION(kern_buffer_partitions *)
KERN_MULTIRELS_PARTITION_DESC(kern_multirels *kmrels, int depth)
{
	uint64_t	offset = kmrels->kbuf_part_offset;

	assert(depth < 0 || (depth > 0 && depth <= kmrels->num_rels));
	if (offset > 0)
	{
		kern_buffer_partitions *kbuf_parts
			= (kern_buffer_partitions *)((char *)kmrels + offset);
		if (depth < 0 || kbuf_parts->inner_depth == depth)
			return kbuf_parts;
	}
	return NULL;
}

/* ----------------------------------------------------------------
 *
 * Atomic Operations
 *
 * ----------------------------------------------------------------
 */
INLINE_FUNCTION(uint32_t)
__atomic_write_uint32(uint32_t *ptr, uint32_t ival)
{
#ifdef __CUDACC__
	return atomicExch((unsigned int *)ptr, ival);
#else
	return __atomic_exchange_n(ptr, ival, __ATOMIC_SEQ_CST);
#endif
}

INLINE_FUNCTION(uint64_t)
__atomic_write_uint64(uint64_t *ptr, uint64_t ival)
{
#ifdef __CUDACC__
	return atomicExch((unsigned long long int *)ptr, ival);
#else
	return __atomic_exchange_n(ptr, ival, __ATOMIC_SEQ_CST);
#endif
}

INLINE_FUNCTION(uint32_t)
__atomic_add_uint32(uint32_t *ptr, uint32_t ival)
{
#ifdef __CUDACC__
	return atomicAdd((unsigned int *)ptr, (unsigned int)ival);
#else
	return __atomic_fetch_add(ptr, ival, __ATOMIC_SEQ_CST);
#endif
}

INLINE_FUNCTION(uint64_t)
__atomic_add_uint64(uint64_t *ptr, uint64_t ival)
{
#ifdef __CUDACC__
	return atomicAdd((unsigned long long *)ptr, (unsigned long long)ival);
#else
	return __atomic_fetch_add(ptr, ival, __ATOMIC_SEQ_CST);
#endif
}

INLINE_FUNCTION(int64_t)
__atomic_add_int64(int64_t *ptr, int64_t ival)
{
#ifdef __CUDACC__
	return atomicAdd((unsigned long long int *)ptr, (unsigned long long int)ival);
#else
	return __atomic_fetch_add(ptr, ival, __ATOMIC_SEQ_CST);
#endif
}

INLINE_FUNCTION(float8_t)
__atomic_add_fp64(float8_t *ptr, float8_t fval)
{
#ifdef __CUDACC__
	return atomicAdd((double *)ptr, (double)fval);
#else
	union {
		uint64_t	ival;
		float8_t	fval;
	} oldval, newval;

	oldval.fval = __volatileRead(ptr);
	do {
		newval.fval = oldval.fval + fval;
	} while (!__atomic_compare_exchange_n((uint64_t *)ptr,
										  &oldval.ival,
										  newval.ival,
										  false,
										  __ATOMIC_SEQ_CST,
										  __ATOMIC_SEQ_CST));
	return oldval.fval;
#endif
}

INLINE_FUNCTION(uint32_t)
__atomic_and_uint32(uint32_t *ptr, uint32_t mask)
{
#ifdef __CUDACC__
	return atomicAnd((unsigned int *)ptr, (unsigned int)mask);
#else
	return __atomic_fetch_and(ptr, mask, __ATOMIC_SEQ_CST);
#endif
}

INLINE_FUNCTION(uint32_t)
__atomic_or_uint32(uint32_t *ptr, uint32_t mask)
{
#ifdef __CUDACC__
	return atomicOr((unsigned int *)ptr, (unsigned int)mask);
#else
	return __atomic_fetch_or(ptr, mask, __ATOMIC_SEQ_CST);
#endif
}

INLINE_FUNCTION(uint32_t)
__atomic_max_uint32(uint32_t *ptr, uint32_t ival)
{
#ifdef __CUDACC__
	return atomicMax((unsigned int *)ptr, (unsigned int)ival);
#else
	uint32_t	oldval = __volatileRead(ptr);

	while (oldval > ival)
	{
		if (__atomic_compare_exchange_n(ptr,
										&oldval,
										ival,
										false,
										__ATOMIC_SEQ_CST,
										__ATOMIC_SEQ_CST))
			break;
	}
	return oldval;
#endif
}

INLINE_FUNCTION(int64_t)
__atomic_min_int64(int64_t *ptr, int64_t ival)
{
#ifdef __CUDACC__
	return atomicMin((long long int *)ptr, (long long int)ival);
#else
	int64_t		oldval = __volatileRead(ptr);

	while (oldval > ival)
	{
		if (__atomic_compare_exchange_n(ptr,
										&oldval,
										ival,
										false,
										__ATOMIC_SEQ_CST,
										__ATOMIC_SEQ_CST))
			break;
	}
	return oldval;
#endif
}

INLINE_FUNCTION(int64_t)
__atomic_max_int64(int64_t *ptr, int64_t ival)
{
#ifdef __CUDACC__
	return atomicMax((long long int *)ptr, (long long int)ival);
#else
	int64_t		oldval = __volatileRead(ptr);

	while (oldval < ival)
	{
		if (__atomic_compare_exchange_n(ptr,
										&oldval,
										ival,
										false,
										__ATOMIC_SEQ_CST,
										__ATOMIC_SEQ_CST))
			break;
	}
	return oldval;
#endif
}

INLINE_FUNCTION(float8_t)
__atomic_min_fp64(float8_t *ptr, float8_t fval)
{
#ifdef __CUDACC__
	union {
		unsigned long long ival;
		float8_t	fval;
	} oldval, curval, newval;

	newval.fval = fval;
	curval.fval = __volatileRead(ptr);
	while (newval.fval < curval.fval)
	{
		oldval = curval;
		curval.ival = atomicCAS((unsigned long long *)ptr,
								oldval.ival,
								newval.ival);
		if (curval.ival == oldval.ival)
			break;
	}
	return curval.fval;
#else
	union {
		uint64_t	ival;
		float8_t	fval;
	} oldval, newval;

	newval.fval = fval;
	oldval.fval = __volatileRead(ptr);
	while (oldval.fval > newval.fval)
	{
		if (__atomic_compare_exchange_n((uint64_t *)ptr,
										&oldval.ival,
										newval.ival,
										false,
										__ATOMIC_SEQ_CST,
										__ATOMIC_SEQ_CST))
			break;
	}
	return oldval.fval;
#endif
}

INLINE_FUNCTION(float8_t)
__atomic_max_fp64(float8_t *ptr, float8_t fval)
{
#ifdef __CUDACC__
	union {
		unsigned long long ival;
		float8_t	fval;
	} oldval, curval, newval;

	newval.fval = fval;
	curval.fval = __volatileRead(ptr);
	while (newval.fval > curval.fval)
	{
		oldval = curval;
		curval.ival = atomicCAS((unsigned long long *)ptr,
								oldval.ival,
								newval.ival);
		if (curval.ival == oldval.ival)
			break;
	}
	return curval.fval;
#else
	union {
		uint64_t	ival;
		float8_t	fval;
	} oldval, newval;

	newval.fval = fval;
	oldval.fval = __volatileRead(ptr);
	while (oldval.fval > newval.fval)
	{
		if (__atomic_compare_exchange_n((uint64_t *)ptr,
										&oldval.ival,
										newval.ival,
										false,
										__ATOMIC_SEQ_CST,
										__ATOMIC_SEQ_CST))
			break;
	}
	return oldval.fval;
#endif
}

INLINE_FUNCTION(uint32_t)
__atomic_exchange_uint32(uint32_t *ptr, uint32_t newval)
{
#ifdef __CUDACC__
	return atomicExch((unsigned int *)ptr,
					  (unsigned int)newval);
#else
	return __atomic_exchange_n(ptr,
							   newval,
							   __ATOMIC_SEQ_CST);
#endif
}

INLINE_FUNCTION(uint64_t)
__atomic_exchange_uint64(uint64_t *ptr, uint64_t newval)
{
#ifdef __CUDACC__
	return atomicExch((unsigned long long int *)ptr,
					  (unsigned long long int)newval);
#else
	return __atomic_exchange_n(ptr,
							   newval,
							   __ATOMIC_SEQ_CST);
#endif
}

INLINE_FUNCTION(uint32_t)
__atomic_cas_uint32(uint32_t *ptr, uint32_t comp, uint32_t newval)
{
#ifdef __CUDACC__
	return atomicCAS((unsigned int *)ptr,
					 (unsigned int)comp,
					 (unsigned int)newval);
#else
	__atomic_compare_exchange_n(ptr,
								&comp,
								newval,
								false,
								__ATOMIC_SEQ_CST,
								__ATOMIC_SEQ_CST);
	return comp;
#endif
}

INLINE_FUNCTION(uint64_t)
__atomic_cas_uint64(uint64_t *ptr, uint64_t comp, uint64_t newval)
{
#ifdef __CUDACC__
	return atomicCAS((unsigned long long int *)ptr,
					 (unsigned long long int)comp,
					 (unsigned long long int)newval);
#else
	__atomic_compare_exchange_n(ptr,
								&comp,
								newval,
								false,
								__ATOMIC_SEQ_CST,
								__ATOMIC_SEQ_CST);
	return comp;
#endif
}

/* ----------------------------------------------------------------
 *
 * xPU PreAgg common utility functions
 *
 * ----------------------------------------------------------------
 */
INLINE_FUNCTION(bool)
__preagg_fetch_xdatum_as_int32(int32_t *p_ival, const xpu_datum_t *xdatum)
{
	if (xdatum->expr_ops == &xpu_int4_ops)
		*p_ival = ((const xpu_int4_t *)xdatum)->value;
	else if (xdatum->expr_ops == &xpu_date_ops)
		*p_ival = ((const xpu_date_t *)xdatum)->value;
	else
	{
		assert(XPU_DATUM_ISNULL(xdatum));
		return false;
	}
	return true;
}

INLINE_FUNCTION(bool)
__preagg_fetch_xdatum_as_int64(int64_t *p_ival, const xpu_datum_t *xdatum)
{
	if (xdatum->expr_ops == &xpu_int8_ops)
		*p_ival = ((const xpu_int8_t *)xdatum)->value;
	else if (xdatum->expr_ops == &xpu_timestamp_ops)
		*p_ival = ((const xpu_timestamp_t *)xdatum)->value;
	else if (xdatum->expr_ops == &xpu_timestamptz_ops)
		*p_ival = ((const xpu_timestamptz_t *)xdatum)->value;
	else if (xdatum->expr_ops == &xpu_time_ops)
		*p_ival = ((const xpu_time_t *)xdatum)->value;
	else if (xdatum->expr_ops == &xpu_money_ops)
		*p_ival = ((const xpu_money_t *)xdatum)->value;
	else
	{
		assert(XPU_DATUM_ISNULL(xdatum));
		return false;
	}
	return true;
}

INLINE_FUNCTION(bool)
__preagg_fetch_xdatum_as_float64(float8_t *p_fval, const xpu_datum_t *xdatum)
{
	if (xdatum->expr_ops == &xpu_float8_ops)
		*p_fval = ((const xpu_float8_t *)xdatum)->value;
	else
	{
		assert(XPU_DATUM_ISNULL(xdatum));
		return false;
	}
	return true;
}

/* ----------------------------------------------------------------
 *
 * Misc functions
 *
 * ----------------------------------------------------------------
 */
INLINE_FUNCTION(void)
print_kern_data_store(const kern_data_store *kds)
{
	printf("kds %p { length=%lu, usage=%lu, nitems=%u, ncols=%u, format=%c, has_varlena=%c, tdhasoid=%c, tdtypeid=%u, tdtypmod=%d, table_oid=%u, hash_nslots=%u, block_offset=%u, block_nloaded=%u, nr_colmeta=%u }\n",
		   kds,
		   kds->length,
		   kds->usage,
		   kds->nitems,
		   kds->ncols,
		   kds->format,
		   kds->has_varlena ? 't' : 'f',
		   kds->tdhasoid ? 't' : 'f',
		   kds->tdtypeid,
		   kds->tdtypmod,
		   kds->table_oid,
		   kds->hash_nslots,
		   kds->block_offset,
		   kds->block_nloaded,
		   kds->nr_colmeta);
	for (int j=0; j < kds->nr_colmeta; j++)
	{
		const kern_colmeta *cmeta = &kds->colmeta[j];

		printf("cmeta[%d] { attbyval=%c, attalign=%d, attlen=%d, attnum=%d, attcacheoff=%d, atttypid=%u, atttypmod=%d, atttypkind=%c, kds_format=%c, kds_offset=%u, idx_subattrs=%u, num_subattrs=%u, attname='%s' }\n",
			   j,
			   cmeta->attbyval ? 't' : 'f',
			   (int)cmeta->attalign,
			   (int)cmeta->attlen,
			   (int)cmeta->attnum,
			   (int)cmeta->attcacheoff,
			   cmeta->atttypid,
			   cmeta->atttypmod,
			   cmeta->atttypkind,
			   cmeta->kds_format,
			   cmeta->kds_offset,
			   (unsigned int)cmeta->idx_subattrs,
			   (unsigned int)cmeta->num_subattrs,
			   cmeta->attname);
	}
}
#endif	/* XPU_COMMON_H */
