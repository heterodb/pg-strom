/*
 * cuda_common.h
 *
 * A common header for CUDA device code
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
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

/*
 * Basic type definition - because of historical reason, we use "cl_"
 * prefix for the definition of data types below. It might imply
 * something related to OpenCL, but what we intend at this moment is
 * "CUDA Language".
 */
typedef char				cl_bool;
typedef char				cl_char;
typedef unsigned char		cl_uchar;
typedef short				cl_short;
typedef unsigned short		cl_ushort;
typedef int					cl_int;
typedef unsigned int		cl_uint;
#ifdef __CUDACC__
typedef long long			cl_long;
typedef unsigned long long	cl_ulong;
#else
typedef long				cl_long;
typedef unsigned long		cl_ulong;
#endif
typedef float				cl_float;
typedef double				cl_double;

/*
 * OpenCL intermediator always adds -DOPENCL_DEVICE_CODE on kernel build,
 * but not for the host code, so this #if ... #endif block is available
 * only OpenCL device code.
 */
#ifdef __CUDACC__

/* Misc definitions */
#ifdef offsetof
#undef offsetof
#endif
#define offsetof(TYPE,FIELD)	((devptr_t) &((TYPE *)0)->FIELD)

#ifdef lengthof
#undef lengthof
#endif
#define lengthof(ARRAY)			(sizeof(ARRAY) / sizeof((ARRAY)[0]))

#ifdef container_of
#undef container_of
#endif
#define container_of(TYPE,FIELD,PTR)			\
	((TYPE *)((char *) (PTR) - offsetof(TYPE, FIELD)))

#define BITS_PER_BYTE			8
#define FLEXIBLE_ARRAY_MEMBER	1
#define true			((cl_bool) 1)
#define false			((cl_bool) 0)

/* Another basic type definitions */
typedef cl_ulong	hostptr_t;
typedef size_t		devptr_t;
typedef cl_ulong	Datum;
#define PointerGetDatum(X)		((Datum) (X))
#define DatumGetPointer(X)		((char *) (X))

#define INT64CONST(x)	((cl_long) x##L)
#define UINT64CONST(x)	((cl_ulong) x##UL)

#define Max(a,b)		((a) > (b) ? (a) : (b))
#define Max3(a,b,c)		((a) > (b) ? Max((a),(c)) : Max((b),(c)))
#define Max4(a,b,c,d)	Max(Max((a),(b)),Max((c),(d)))

#define Min(a,b)		((a) < (b) ? (a) : (b))
#define Min3(a,b,c)		((a) < (b) ? Min((a),(c)) : Min((b),(c)))
#define Min4(a,b,c,d)	Min(Min((a),(b)),Min((c),(d)))

#define Add(a,b)		((a) + (b))
#define Add3(a,b,c)		((a) + (b) + (c))
#define Add4(a,b,c,d)	((a) + (b) + (c) + (d))

/* same as host side get_next_log2() */
#define get_next_log2(value)								\
	((value) == 0 ? 0 : (sizeof(cl_ulong) * BITS_PER_BYTE - \
						 __clzll((cl_ulong)(value) - 1)))

/*
 * Alignment macros
 */
#define TYPEALIGN(ALIGNVAL,LEN)	\
	(((devptr_t) (LEN) + ((ALIGNVAL) - 1)) & ~((devptr_t) ((ALIGNVAL) - 1)))
#define TYPEALIGN_DOWN(ALIGNVAL,LEN) \
	(((devptr_t) (LEN)) & ~((devptr_t) ((ALIGNVAL) - 1)))
#define INTALIGN(LEN)			TYPEALIGN(sizeof(cl_int), (LEN))
#define INTALIGN_DOWN(LEN)		TYPEALIGN_DOWN(sizeof(cl_int), (LEN))
#define LONGALIGN(LEN)          TYPEALIGN(sizeof(cl_long), (LEN))
#define LONGALIGN_DOWN(LEN)     TYPEALIGN_DOWN(sizeof(cl_long), (LEN))
#define MAXALIGN(LEN)			TYPEALIGN(MAXIMUM_ALIGNOF, (LEN))
#define MAXALIGN_DOWN(LEN)		TYPEALIGN_DOWN(MAXIMUM_ALIGNOF, (LEN))

/*
 * Limitation of types
 */
#define SHRT_MAX		32767
#define SHRT_MIN		(-32767-1)
#define USHRT_MAX		65535
#define INT_MAX			2147483647
#define INT_MIN			(-INT_MAX - 1)
#define UINT_MAX		4294967295U
#define LONG_MAX		0x7FFFFFFFFFFFFFFFLL
#define LONG_MIN        (-LONG_MAX - 1LL)
#define ULONG_MAX		0xFFFFFFFFFFFFFFFFULL
#define FLT_MAX			__int_as_float(0x7f7fffffU)
#define FLT_MIN			__int_as_float(0x00800000U)
#define FLT_DIG			6
#define FLT_MANT_DIG	24
#define DBL_MAX			__longlong_as_double(0x7fefffffffffffffULL)
#define DBL_MIN			__longlong_as_double(0x0010000000000000ULL)
#define DBL_DIG			15
#define DBL_MANT_DIG	53

/*
 * MEMO: We takes dynamic local memory using cl_ulong data-type because of
 * alignment problem. The nvidia's driver adjust alignment of local memory
 * according to the data type; 1byte for cl_char, 4bytes for cl_uint and
 * so on. Unexpectedly, void * pointer has 1byte alignment even if it is
 * expected to be casted another data types.
 * A pragma option __attribute__((aligned)) didn't work at least driver
 * version 340.xx. So, we declared the local_workmem as cl_ulong * pointer
 * as a workaround.
 */
#define SHARED_WORKMEM(TYPE)	((TYPE *) __pgstrom_dynamic_shared_workmem)
extern __shared__ cl_ulong __pgstrom_dynamic_shared_workmem[];

/*
 * Thread index like OpenCL style.
 *
 * Be careful to use this convenient alias if grid/block size may become
 * larger than INT_MAX, because threadIdx and blockDim are declared as
 * 32bit integer, thus, it makes overflow during intermediation results
 * if it is larger than INT_MAX.
 */
#define get_local_id()			(threadIdx.x)
#define get_local_size()		(blockDim.x)
#define get_global_id()			(threadIdx.x + blockIdx.x * blockDim.x)
#define get_global_size()		(blockDim.x * gridDim.x)
#define get_global_base()		(blockIdx.x * blockDim.x)
#define get_global_index()		(blockIdx.x)

#define lget_global_id()		((size_t)threadIdx.x +	\
								 (size_t)blockIdx.x *	\
								 (size_t)blockDim.x)
#define lget_global_size()		((size_t)blockDim.x * (size_t)gridDim.x)
#define lget_global_base()		((size_t)blockIdx.x * (size_t)blockDim.x)
#define lget_global_index()		((size_t)blockIdx.x)

#else	/* __CUDACC__ */
#include "access/htup_details.h"
#include "storage/itemptr.h"
#define __device__		/* address space qualifier is noise on host */
#define __global__		/* address space qualifier is noise on host */
#define __constant__	/* address space qualifier is noise on host */
#define __shared__		/* address space qualifier is noise on host */
typedef uintptr_t		hostptr_t;
#endif

/*
 * Template of static function declarations
 *
 * CUDA compilar raises warning if static functions are not used, but
 * we can restain this message with"unused" attribute of function/values.
 * STATIC_INLINE / STATIC_FUNCTION packs common attributes to be
 * assigned on host/device functions
 */
#ifdef __CUDACC__
#define STATIC_INLINE(RET_TYPE)						\
	__device__ __forceinline__ static RET_TYPE __attribute__ ((unused))
#define STATIC_FUNCTION(RET_TYPE)					\
	__device__ static RET_TYPE __attribute__ ((unused))
#define KERNEL_FUNCTION(RET_TYPE)	__global__ RET_TYPE
#if __CUDA_ARCH__ < 200
#define KERNEL_FUNCTION_MAXTHREADS(RET_TYPE)	\
	__global__ RET_TYPE __launch_bounds__(512)
#else
#define KERNEL_FUNCTION_MAXTHREADS(RET_TYPE)	\
	__global__ RET_TYPE __launch_bounds__(1024)
#endif
#else
#define STATIC_INLINE(RET_TYPE)		static inline RET_TYPE
#define STATIC_FUNCTION(RET_TYPE)	static inline RET_TYPE
#define KERNEL_FUNCTION(RET_TYPE)	RET_TYPE
#define KERNEL_FUNCTION_MAXTHREADS(RET_TYPE)	KERNEL_FUNCTION(RET_TYPE)
#endif

/*
 * Error code definition
 *
 *           0 : Success in all error code scheme
 *    1 -  999 : Error code in CUDA driver API
 * 1000 - 9999 : Error code of PG-Strom
 *     > 10000 : Error code in CUDA device runtime
 *
 * Error code definition
 */
#define StromError_Success					   0 /* OK */
#define StromError_CpuReCheck				1000 /* To be re-checked by CPU */
#define StromError_CudaInternal				1001 /* CUDA internal error */
#define StromError_OutOfMemory				1002 /* Out of memory */
#define StromError_OutOfSharedMemory		1003 /* Out of shared memory */
#define StromError_OutOfKernelArgs			1004 /* Out of kernel arguments */
#define StromError_InvalidValue				1999 /* Something invalid */
#define StromError_DataStoreCorruption		2000 /* KDS corrupted */
#define StromError_DataStoreNoSpace			2001 /* KDS has no space */
#define StromError_DataStoreOutOfRange		2002 /* out of KDS range access */
#define StromError_SanityCheckViolation		2003 /* sanity check violation */
#define StromError_WrongCodeGeneration		2004 /* Bugs on code generation */
#define StromError_CudaDevRunTimeBase	   10000 /* Base value for runtime */

/*
 * Kernel functions identifier
 */
#define StromKernel_HostPGStrom						0x0001
#define StromKernel_CudaRuntime						0x0002
#define StromKernel_gpuscan_exec_quals				0x0101
#define StromKernel_gpuscan_projection_row			0x0102
#define StromKernel_gpuscan_projection_slot			0x0103
#define StromKernel_gpuscan_main					0x0199
#define StromKernel_gpujoin_exec_outerscan			0x0201
#define StromKernel_gpujoin_exec_nestloop			0x0202
#define StromKernel_gpujoin_exec_hashjoin			0x0203
#define StromKernel_gpujoin_outer_nestloop			0x0204
#define StromKernel_gpujoin_outer_hashjoin			0x0205
#define StromKernel_gpujoin_projection_row			0x0206
#define StromKernel_gpujoin_projection_slot			0x0207
#define StromKernel_gpujoin_count_rows_dist			0x0208
#define StromKernel_gpujoin_main					0x0299
#define StromKernel_gpupreagg_preparation			0x0301
#define StromKernel_gpupreagg_local_reduction		0x0302
#define StromKernel_gpupreagg_global_reduction		0x0303
#define StromKernel_gpupreagg_nogroup_reduction		0x0304
#define StromKernel_gpupreagg_final_preparation		0x0305
#define StromKernel_gpupreagg_final_reduction		0x0306
#define StromKernel_gpupreagg_fixup_varlena			0x0307
#define StromKernel_gpupreagg_main					0x0399
#define StromKernel_gpusort_projection				0x0401
#define StromKernel_gpusort_bitonic_local			0x0402
#define StromKernel_gpusort_bitonic_step			0x0403
#define StromKernel_gpusort_bitonic_merge			0x0404
#define StromKernel_gpusort_fixup_pointers			0x0405
#define StromKernel_gpusort_main					0x0499
#define StromKernel_plcuda_prep_kernel				0x0501
#define StromKernel_plcuda_main_kernel				0x0502
#define StromKernel_plcuda_post_kernel				0x0503

typedef struct
{
	cl_int		errcode;	/* one of the StromError_* */
	cl_short	kernel;		/* one of the StromKernel_* */
	cl_short	lineno;		/* line number STROM_SET_ERROR is called */
#ifdef PGSTROM_DEBUG
	cl_long		extra_x;	/* for debug purpose */
	cl_long		extra_y;	/* for debug purpose */
	cl_long		extra_z;	/* for debug purpose */
#endif
} kern_errorbuf;

#ifdef PGSTROM_DEBUG
#define KERN_ERRORBUF_SET_EXTRA(p_kerror,x,y,z)	\
	do {										\
		(p_kerror)->extra_x = (cl_long)(x);		\
		(p_kerror)->extra_y = (cl_long)(y);		\
		(p_kerror)->extra_z = (cl_long)(z);		\
	} while(0)
#else
#define KERN_ERRORBUF_SET_EXTRA(kcxt,x,y,z)		do {} while(0)
#endif

/*
 * kern_context - a set of run-time information
 */
struct kern_parambuf;

typedef struct
{
	kern_errorbuf	e;
	struct kern_parambuf *kparams;
} kern_context;

#define INIT_KERNEL_CONTEXT(kcxt,kfunction,kparams)			\
	do {													\
		(kcxt)->e.errcode = StromError_Success;				\
		(kcxt)->e.kernel = StromKernel_##kfunction;			\
		(kcxt)->e.lineno = 0;								\
		KERN_ERRORBUF_SET_EXTRA(&(kcxt)->e,0,0,0);			\
		(kcxt)->kparams = (kparams);						\
		assert((cl_ulong)(kparams) == MAXALIGN(kparams));	\
	} while(0)

/*
 * It sets an error code unless no significant error code is already set.
 * Also, CpuReCheck has higher priority than RowFiltered because CpuReCheck
 * implies device cannot run the given expression completely.
 * (Usually, due to compressed or external varlena datum)
 */
#ifdef __CUDACC__
STATIC_INLINE(void)
__STROM_SET_ERROR(kern_errorbuf *p_kerror, cl_int errcode, cl_int lineno,
				  cl_long extra_x, cl_long extra_y, cl_long extra_z)
{
	cl_int			oldcode = p_kerror->errcode;

	if (oldcode == StromError_Success &&
		errcode != StromError_Success)
	{
		p_kerror->errcode = errcode;
		p_kerror->lineno = lineno;
		KERN_ERRORBUF_SET_EXTRA(p_kerror, extra_x, extra_y, extra_z);
	}
}

#define STROM_SET_ERROR(p_kerror, errcode)		\
	__STROM_SET_ERROR((p_kerror), (errcode), __LINE__, 123, 456, 789)
#define STROM_SET_ERROR_EXTRA(p_kerror, errcode, x, y, z)	\
	__STROM_SET_ERROR((p_kerror), (errcode), __LINE__, (x), (y), (z))
#define STROM_SET_RUNTIME_ERROR(p_kerror, errcode)						\
	STROM_SET_ERROR((p_kerror), (errcode) == cudaSuccess ?				\
					(cl_int)(errcode) :									\
					(cl_int)(errcode) + StromError_CudaDevRunTimeBase)
#define STROM_SET_RUNTIME_ERROR_EXTRA(p_kerror, errcode, x, y, z)		\
	STROM_SET_ERROR_EXTRA((p_kerror),									\
						  (errcode) == cudaSuccess ?					\
						  (cl_int)(errcode) :							\
						  (cl_int)(errcode) + StromError_CudaDevRunTimeBase, \
						  (x), (y), (z))

/*
 * kern_writeback_error_status
 */
STATIC_INLINE(void)
kern_writeback_error_status(kern_errorbuf *result, kern_errorbuf own_error)
{
	/*
	 * It writes back a thread local error status only when the global
	 * error status is not set yet and the caller thread contains any
	 * error status. Elsewhere, we don't involves any atomic operation
	 * in the most of code path.
	 */
	if (own_error.errcode != StromError_Success &&
		atomicCAS(&result->errcode,
				  StromError_Success,
				  own_error.errcode) == StromError_Success)
	{
		/* only primary error workgroup can come into */
		result->kernel = own_error.kernel;
		result->lineno = own_error.lineno;
#ifdef PGSTROM_DEBUG
		result->extra_x = own_error.extra_x;
		result->extra_y = own_error.extra_y;
		result->extra_z = own_error.extra_z;
#endif
	}
}

#else	/* __CUDACC__ */
#ifdef PGSTROM_DEBUG
#define KERROR_EXTRA_X(p_kerror)		((p_kerror)->extra_x)
#define KERROR_EXTRA_Y(p_kerror)		((p_kerror)->extra_y)
#define KERROR_EXTRA_Z(p_kerror)		((p_kerror)->extra_z)
#else
#define KERROR_EXTRA_X(p_kerror)		0UL
#define KERROR_EXTRA_Y(p_kerror)		0UL
#define KERROR_EXTRA_Z(p_kerror)		0UL
#endif
/*
 * If case when STROM_SET_ERROR is called in the host code,
 * it raises an error using ereport()
 */
#define STROM_SET_ERROR(p_kerror, errcode)		\
	elog(ERROR, "%s:%d %s", __FUNCTION__, __LINE__, errorText(errcode))
#endif	/* !__CUDACC__! */
#ifdef __CUDACC__
/*
 * NumSmx - reference to the %nsmid register
 */
STATIC_INLINE(cl_uint) NumSmx(void)
{
	cl_uint		ret;
	asm volatile("mov.u32 %0, %nsmid;" : "=r"(ret) );
	return ret;
}

/*
 * SmxId - reference to the %smid register
 */
STATIC_INLINE(cl_uint) SmxId(void)
{
	cl_uint		ret;
	asm volatile("mov.u32 %0, %smid;" : "=r"(ret) );
	return ret;
}

/*
 * WarpId() - reference to the %warpid register
 */
STATIC_INLINE(cl_uint) WarpId(void)
{
	cl_uint		ret;
	asm volatile("mov.u32 %0, %warpid;" : "=r"(ret) );
	return ret;
}

/*
 * GlobalTimer - A pre-defined, 64bit global nanosecond timer.
 *
 * NOTE: clock64() is not consistent across different SMX, thus, should not
 *       use this API in case when device time-run may reschedule the kernel.
 */
STATIC_INLINE(cl_ulong) GlobalTimer(void)
{
	cl_ulong	ret;
	asm volatile("mov.u64 %0, %globaltimer;" : "=l"(ret) );
	return ret;
}

/*
 * We need to re-define HeapTupleHeaderData and t_infomask related stuff
 */
typedef struct {
	struct {
		cl_ushort	bi_hi;
		cl_ushort	bi_lo;
	} ip_blkid;
	cl_ushort		ip_posid;
} ItemPointerData;

typedef struct {
	union {
		struct {
			cl_uint	t_xmin;		/* inserting xact ID */
			cl_uint	t_xmax;		/* deleting or locking xact ID */
			union {
				cl_uint	t_cid;	/* inserting or deleting command ID, or both */
				cl_uint	t_xvac;	/* old-style VACUUM FULL xact ID */
			} t_field3;
		} t_heap;
		struct {
			cl_uint	datum_len_;	/* varlena header (do not touch directly!) */
			cl_uint	datum_typmod;	/* -1, or identifier of a record type */
			cl_uint	datum_typeid;	/* composite type OID, or RECORDOID */
		} t_datum;
	} t_choice;

	ItemPointerData	t_ctid;			/* current TID of this or newer tuple */

	cl_ushort		t_infomask2;	/* number of attributes + various flags */
	cl_ushort		t_infomask;		/* various flag bits, see below */
	cl_uchar		t_hoff;			/* sizeof header incl. bitmap, padding */
	/* ^ - 23 bytes - ^ */
	cl_uchar		t_bits[1];		/* bitmap of NULLs -- VARIABLE LENGTH */
} HeapTupleHeaderData;

#define att_isnull(ATT, BITS) (!((BITS)[(ATT) >> 3] & (1 << ((ATT) & 0x07))))
#define bitmaplen(NATTS) (((int)(NATTS) + BITS_PER_BYTE - 1) / BITS_PER_BYTE)

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

/*
 * information stored in t_infomask2:
 */
#define HEAP_NATTS_MASK			0x07FF	/* 11 bits for number of attributes */
#define HEAP_KEYS_UPDATED		0x2000	/* tuple was updated and key cols
										 * modified, or tuple deleted */
#define HEAP_HOT_UPDATED		0x4000	/* tuple was HOT-updated */
#define HEAP_ONLY_TUPLE			0x8000	/* this is heap-only tuple */
#define HEAP2_XACT_MASK			0xE000	/* visibility-related bits */

#endif

/*
 * alignment for pg-strom
 */
#define STROMALIGN_LEN			16
#define STROMALIGN(LEN)			TYPEALIGN(STROMALIGN_LEN,(LEN))
#define STROMALIGN_DOWN(LEN)	TYPEALIGN_DOWN(STROMALIGN_LEN,(LEN))

#define GPUMEMALIGN_LEN			1024
#define GPUMEMALIGN(LEN)		TYPEALIGN(GPUMEMALIGN_LEN,(LEN))
#define GPUMEMALIGN_DOWN(LEN)	TYPEALIGN_DOWN(GPUMEMALIGN_LEN,(LEN))

#ifdef __CUDACC__
/*
 * alignment aware value reference
 */
STATIC_INLINE(cl_short)
get_uint16_val(const void *addr)
{
	return ((devptr_t)addr & (sizeof(cl_ushort) - 1)) == 0
		? *((cl_ushort *) addr)
		: ((cl_ushort)((cl_uchar *)addr)[0] |
		   (cl_ushort)((cl_uchar *)addr)[1] << 8);
}

STATIC_INLINE(cl_uint)
get_uint32_val(const void *addr)
{
	switch ((devptr_t)addr & (sizeof(cl_uint) - 1))
	{
		case 0:
			return *((cl_uint *) addr);
		case 2:
			return ((cl_uint)(*((cl_ushort *)((char *)addr))) |
					(cl_uint)(*((cl_ushort *)((char *)addr + 2))) << 16);
		default:
			return ((cl_uint)(*((cl_uchar *) ((char *)addr))) |
					(cl_uint)(*((cl_ushort *)((char *)addr + 1))) << 8 |
					(cl_uint)(*((cl_uchar *) ((char *)addr + 3))) << 24);
	}
}

STATIC_INLINE(cl_ulong)
get_uint64_val(const void *addr)
{
	switch ((devptr_t)addr & (sizeof(cl_ulong) - 1))
	{
		case 0:
			return *((cl_ulong *)addr);
		case 4:
			return ((cl_ulong)(*((cl_uint *)((char *)addr))) |
					(cl_ulong)(*((cl_uint *)((char *)addr + 4))));
		case 2:
		case 6:
			return ((cl_ulong)(*((cl_ushort *)((char *)addr))) |
					(cl_ulong)(*((cl_uint *)  ((char *)addr + 2))) << 16 |
					(cl_ulong)(*((cl_ushort *)((char *)addr + 6))) << 48);
		case 1:
		case 5:
			return ((cl_ulong)(*((cl_uchar *) ((char *)addr))) |
					(cl_ulong)(*((cl_ushort *)((char *)addr + 1))) <<  8 |
					(cl_ulong)(*((cl_uint *)  ((char *)addr + 3))) << 24 |
					(cl_ulong)(*((cl_uchar *) ((char *)addr + 7))) << 56);
		default:	/* 3 or 7 */
			return ((cl_ulong)(*((cl_uchar *) ((char *)addr))) |
					(cl_ulong)(*((cl_uint *)  ((char *)addr + 1))) <<  8 |
					(cl_ulong)(*((cl_ushort *)((char *)addr + 5))) << 40 |
					(cl_ulong)(*((cl_uchar *) ((char *)addr + 7))) << 56);
	}
}

#define get_int16_val(ADDR)		((cl_int)get_uint16_val(ADDR))
#define get_int32_val(ADDR)		((cl_int)get_uint32_val(ADDR))
#define get_int64_val(ADDR)		((cl_int)get_uint64_val(ADDR))
#define get_float32_val(ADDR)	__int_as_float(get_uint32_val(ADDR))
#define get_float64_val(ADDR)	__longlong_as_double(get_int64_val(ADDR))
#endif	/* __CUDACC__ */

/*
 * kern_data_store
 *
 * +----------------------------------------------+
 * | Common header portion of the kern_data_store |
 * |         :                                    |
 * | 'format' determines the layout below         |
 * +----------------------------------------------+
 * | Attributes of columns                        |
 * |                                              |
 * | kern_colmeta colmeta[0]                      |
 * | kern_colmeta colmeta[1]                      |
 * |        :                                     |
 * | kern_colmeta colmeta[M-1]                    |
 * +----------------------------------------------+
 * | <slot format> | <row format> / <hash format> |
 * +---------------+------------------------------+  -----
 * | values/isnull | Offset to the first hash-    |
 * | pair of the   | item for each slot (*).      |
 * | 1st tuple     |                              |
 * | +-------------+ (*) nslots=0 if row-format,  |
 * | | values[0]   | thus, it has no offset to    |
 * | |    :        | hash items.                  |
 * | | values[M-1] |                              |
 * | +-------------+  hash_slot[0]                |
 * | | isnull[0]   |  hash_slot[1]                |
 * | |    :        |      :                       |
 * | | isnull[M-1] |  hash_slot[nslots-1]         |
 * +-+-------------+------------------------------+
 * | values/isnull | Offset to the individual     |
 * | pair of the   | kern_tupitem.                |
 * | 2nd tuple     |                              |
 * | +-------------+ row_index[0]                 |
 * | | values[0]   | row_index[1]                 |
 * | |    :        |    :                         |
 * | | values[M-1] | row_index[nitems-1]          |
 * | +-------------+--------------+---------------+
 * | | isnull[0]   |    :         |       :       |
 * | |    :        +--------------+---------------+
 * | | isnull[M-1] | kern_tupitem | kern_hashitem |
 * +-+-------------+--------------+---------------+
 * | values/isnull | kern_tupitem | kern_hashitem |
 * | pair of the   +--------------+---------------+
 * | 3rd tuple     | kern_tupitem | kern_hashitem |
 * |      :        |     :        |     :         |
 * |      :        |     :        |     :         |
 * +---------------+--------------+---------------+
 */
typedef struct {
	/* true, if column is held by value. Elsewhere, a reference */
	cl_char			attbyval;
	/* alignment; 1,2,4 or 8, not characters in pg_attribute */
	cl_char			attalign;
	/* length of attribute */
	cl_short		attlen;
	/* attribute number */
	cl_short		attnum;
	/* offset of attribute location, if deterministic */
	cl_short		attcacheoff;
	/* oid of the SQL data type */
	cl_uint			atttypid;
	/* typmod of the SQL data type */
	cl_int			atttypmod;
} kern_colmeta;

/*
 * kern_tupitem - individual items for KDS_FORMAT_ROW
 */
typedef struct
{
	cl_ushort			t_len;	/* length of tuple */
	ItemPointerData		t_self;	/* SelfItemPointer */
	HeapTupleHeaderData	htup;
} kern_tupitem;

/*
 * kern_hashitem - individual items for KDS_FORMAT_HASH
 */
typedef struct
{
	cl_uint				hash;	/* 32-bit hash value */
	cl_uint				next;	/* offset of the next */
	cl_uint				rowid;	/* unique identifier of this hash entry */
	cl_uint				__padding__; /* for alignment */
	kern_tupitem		t;		/* HeapTuple of this entry */
} kern_hashitem;

#define KDS_FORMAT_ROW			1
#define KDS_FORMAT_SLOT			2
#define KDS_FORMAT_HASH			3	/* inner hash table for GpuHashJoin */

typedef struct {
	hostptr_t		hostptr;	/* address of kds on the host */
	cl_uint			length;		/* length of this data-store */
	cl_uint			usage;		/* usage of this data-store */
	cl_uint			ncols;		/* number of columns in this store */
	cl_uint			nitems; 	/* number of rows in this store */
	cl_uint			nrooms;		/* number of available rows in this store */
	cl_char			format;		/* one of KDS_FORMAT_* above */
	cl_char			has_notbyval; /* true, if any of column is !attbyval */
	cl_char			has_numeric;/* true, if any of column is numeric */
	cl_char			tdhasoid;	/* copy of TupleDesc.tdhasoid */
	cl_uint			tdtypeid;	/* copy of TupleDesc.tdtypeid */
	cl_int			tdtypmod;	/* copy of TupleDesc.tdtypmod */
	cl_uint			table_oid;	/* OID of the table (only if GpuScan) */
	cl_uint			nslots;		/* width of hash-slot (only HASH format) */
	cl_uint			hash_min;	/* minimum hash-value (only HASH format) */
	cl_uint			hash_max;	/* maximum hash-value (only HASH format) */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER]; /* metadata of columns */
} kern_data_store;


/* length estimator */
#define KDS_CALCULATE_HEAD_LENGTH(ncols)		\
	STROMALIGN(offsetof(kern_data_store, colmeta[(ncols)]))

#define KDS_CALCULATE_FRONTEND_LENGTH(ncols,nslots,nitems)	\
	(KDS_CALCULATE_HEAD_LENGTH(ncols) +			\
	 STROMALIGN(sizeof(cl_uint) * (nitems)) +	\
	 STROMALIGN(sizeof(cl_uint) * (nslots)))

/* 'nslots' estimation; 25% larger than nitems, but 128 at least */
#define __KDS_NSLOTS(nitems)					\
	Max(128, ((nitems) * 5) >> 2)
#define KDS_CALCULATE_ROW_FRONTLEN(ncols,nitems)	\
	KDS_CALCULATE_FRONTEND_LENGTH((ncols),0,(nitems))
#define KDS_CALCULATE_HASH_FRONTLEN(ncols,nitems)	\
	KDS_CALCULATE_FRONTEND_LENGTH((ncols),__KDS_NSLOTS(nitems),(nitems))
#define KDS_CALCULATE_ROW_LENGTH(ncols,nitems,data_len)		\
	(KDS_CALCULATE_ROW_FRONTLEN((ncols),(nitems)) + STROMALIGN(data_len))
#define KDS_CALCULATE_HASH_LENGTH(ncols,nitems,data_len)	\
	(KDS_CALCULATE_HASH_FRONTLEN((ncols),(nitems)) + STROMALIGN(data_len))
#define KDS_CALCULATE_SLOT_LENGTH(ncols,nitems)	\
	(KDS_CALCULATE_HEAD_LENGTH(ncols) +			\
	 LONGALIGN((sizeof(Datum) +					\
				sizeof(char)) * (ncols)) * (nitems))
/* gsage growth by the next tuple (row) */
#define KDS_ROW_USAGE_GROWTH(nitems, consumption)			\
	(STROMALIGN(sizeof(cl_uint) * ((nitems) + 1)) -			\
	 STROMALIGN(sizeof(cl_uint) * (nitems)) +				\
	 MAXALIGN(consumption))
/* usage growth by the next one tuple (hash) */
#define KDS_HASH_USAGE_GROWTH(nitems, consumption)			\
	(STROMALIGN(sizeof(cl_uint) * ((nitems) + 1)) -			\
	 STROMALIGN(sizeof(cl_uint) * (nitems)) +				\
	 STROMALIGN(sizeof(cl_uint) * __KDS_NSLOTS((nitems) + 1)) -	\
	 STROMALIGN(sizeof(cl_uint) * __KDS_NSLOTS(nitems)) +	\
	 MAXALIGN(consumption))

/* length of kern_data_store */
#define KERN_DATA_STORE_LENGTH(kds)		((kds)->length)

/* length of the header portion of kern_data_store */
#define KERN_DATA_STORE_HEAD_LENGTH(kds)			\
	KDS_CALCULATE_HEAD_LENGTH((kds)->ncols)

/* head address of data body */
#define KERN_DATA_STORE_BODY(kds)					\
	((char *)(kds) + KERN_DATA_STORE_HEAD_LENGTH(kds))

/* access macro for row- and hash-format */
#define KERN_DATA_STORE_ROWINDEX(kds)				\
	((cl_uint *)(KERN_DATA_STORE_BODY(kds)))

/* access macro for hash-format */
#define KERN_DATA_STORE_HASHSLOT(kds)				\
	((cl_uint *)(KERN_DATA_STORE_BODY(kds) +		\
				 STROMALIGN(sizeof(cl_uint) * (kds)->nitems)))

/* access macro for row- and hash-format */
#define KERN_DATA_STORE_TUPITEM(kds,kds_index)		\
	((kern_tupitem *)((char *)(kds) +				\
					  (KERN_DATA_STORE_ROWINDEX(kds)[(kds_index)])))

/* access macro for hash-format */
#define KERN_DATA_STORE_HASHITEM(kds,kds_index)		\
	((kern_hashitem *)								\
	 ((char *)KERN_DATA_STORE_TUPITEM(kds,kds_index) -	\
	  offsetof(kern_hashitem, t)))

/* access macro for tuple-slot format */
#define KERN_DATA_STORE_SLOT_LENGTH(kds,nitems)				\
	KDS_CALCULATE_SLOT_LENGTH((kds)->ncols,(nitems))

#define KERN_DATA_STORE_VALUES(kds,kds_index)				\
	((Datum *)((char *)(kds) +								\
			   KDS_CALCULATE_SLOT_LENGTH((kds)->ncols,(kds_index))))

#define KERN_DATA_STORE_ISNULL(kds,kds_index)				\
	((char *)(KERN_DATA_STORE_VALUES((kds),(kds_index)) + (kds)->ncols))


STATIC_INLINE(kern_hashitem *)
KERN_HASH_FIRST_ITEM(kern_data_store *kds, cl_uint hash)
{
	cl_uint	   *slot = KERN_DATA_STORE_HASHSLOT(kds);
	cl_uint		index = hash % kds->nslots;

	if (slot[index] == 0)
		return NULL;
	return (kern_hashitem *)((char *)kds + slot[index]);
}

STATIC_INLINE(kern_hashitem *)
KERN_HASH_NEXT_ITEM(kern_data_store *kds, kern_hashitem *khitem)
{
	if (!khitem || khitem->next == 0)
		return NULL;
	return (kern_hashitem *)((char *)kds + khitem->next);
}

/* transform device pointer to host address */
#define devptr_to_host(kds,devptr)		\
	((hostptr_t)(devptr) -				\
	 (hostptr_t)(&(kds)->hostptr) +		\
	 (hostptr_t)((kds)->hostptr))
/* transform host pointer to device address */
#define hostptr_to_dev(kds,hostptr)		\
	((hostptr_t)(hostptr) -				\
	 (hostptr_t)((kds)->hostptr) +		\
	 (hostptr_t)(&(kds)->hostptr))

/* check pointer's location */
STATIC_INLINE(cl_bool)
pointer_on_kds(void *ptr, kern_data_store *kds)
{
	return kds && ((char *)ptr >= (char *)kds &&
				   (char *)ptr <  (char *)kds + kds->length);
}

/*
 * kern_parambuf
 *
 * Const and Parameter buffer. It stores constant values during a particular
 * scan, so it may make sense if it is obvious length of kern_parambuf is
 * less than constant memory (NOTE: not implemented yet).
 */
typedef struct kern_parambuf
{
	hostptr_t	hostptr;	/* address of the parambuf on host-side */

	/*
	 * Fields of system information on execution
	 */
	cl_long		xactStartTimestamp;	/* timestamp when transaction start */

	/* variable length parameters / constants */
	cl_uint		length;		/* total length of parambuf */
	cl_uint		nparams;	/* number of parameters */
	cl_uint		poffset[FLEXIBLE_ARRAY_MEMBER];	/* offset of params */
} kern_parambuf;

STATIC_INLINE(void *)
kparam_get_value(kern_parambuf *kparams, cl_uint pindex)
{
	if (pindex >= kparams->nparams)
		return NULL;
	if (kparams->poffset[pindex] == 0)
		return NULL;
	return (char *)kparams + kparams->poffset[pindex];
}

STATIC_INLINE(cl_bool)
pointer_on_kparams(void *ptr, kern_parambuf *kparams)
{
	return kparams && ((char *)ptr >= (char *)kparams &&
					   (char *)ptr <  (char *)kparams + kparams->length);
}

/*
 * kern_resultbuf
 *
 * Output buffer to write back calculation results on a parciular chunk.
 * kern_errorbuf informs an error status that shall be raised on host-
 * side to abort current transaction.
 */
typedef struct {
	cl_uint		nrels;		/* number of relations to be appeared */
	cl_uint		nrooms;		/* max number of results rooms */
	cl_uint		nitems;		/* number of results being written */
	cl_char		all_visible;/* GpuScan dumps all the tuples in chunk */
	cl_char		__padding__[3];
	kern_errorbuf kerror;	/* error information */
	cl_uint		results[FLEXIBLE_ARRAY_MEMBER];
} kern_resultbuf;

#define KERN_GET_RESULT(kresults, index)		\
	((kresults)->results + (kresults)->nrels * (index))

#ifdef __CUDACC__
/*
 * PostgreSQL Data Type support in OpenCL kernel
 *
 * A stream of data sequencea are moved to OpenCL kernel, according to
 * the above row-/column-store definition. The device code being generated
 * by PG-Strom deals with each data item using the following data type;
 *
 * typedef struct
 * {
 *     BASE    value;
 *     bool    isnull;
 * } pg_##NAME##_t
 *
 * PostgreSQL has three different data classes:
 *  - fixed-length referenced by value
 *  - fixed-length referenced by pointer
 *  - variable-length value
 *
 * Right now, we support the two except for fixed-length referenced by
 * pointer (because these are relatively minor data type than others).
 * BASE reflects the data type in PostgreSQL; may be an integer, a float
 * or something others, however, all the variable-length value has same
 * BASE type; that is an offset of associated toast buffer, to reference
 * varlena structure on the global memory.
 */

/* forward declaration of access interface to kern_data_store */
STATIC_INLINE(void *)
kern_get_datum(kern_data_store *kds,
			   cl_uint colidx, cl_uint rowidx);

/*
 * common function to store a value on tuple-slot format
 */
STATIC_FUNCTION(void)
pg_common_vstore(kern_data_store *kds,
				 kern_context *kcxt,
				 cl_uint colidx, cl_uint rowidx,
				 Datum pg_value, cl_bool pg_isnull)
{
	if (kds->format != KDS_FORMAT_SLOT)
		STROM_SET_ERROR(&kcxt->e, StromError_SanityCheckViolation);
	else if (colidx >= kds->ncols || rowidx >= kds->nitems)
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreOutOfRange);
	else
	{
		Datum	   *ts_values = KERN_DATA_STORE_VALUES(kds, rowidx);
		cl_bool	   *ts_isnull = KERN_DATA_STORE_ISNULL(kds, rowidx);

		ts_isnull[colidx] = pg_isnull;
		ts_values[colidx] = pg_value;
	}
}

/*
 * Template of variable classes: fixed-length referenced by value
 * ---------------------------------------------------------------
 */
#define STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)			\
	typedef struct {										\
		BASE		value;									\
		cl_bool		isnull;									\
	} pg_##NAME##_t;

#define STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)			\
	STATIC_FUNCTION(pg_##NAME##_t)							\
	pg_##NAME##_datum_ref(kern_context *kcxt,				\
						  void *datum,						\
						  cl_bool internal_format)			\
	{														\
		pg_##NAME##_t	result;								\
															\
		if (!datum)											\
			result.isnull = true;							\
		else												\
		{													\
			result.isnull = false;							\
			result.value = *((BASE *) datum);				\
		}													\
		return result;										\
	}														\
															\
	STATIC_FUNCTION(pg_##NAME##_t)							\
	pg_##NAME##_vref(kern_data_store *kds,					\
					 kern_context *kcxt,					\
					 cl_uint colidx,						\
					 cl_uint rowidx)						\
	{														\
		void  *datum = kern_get_datum(kds,colidx,rowidx);	\
		return pg_##NAME##_datum_ref(kcxt,datum,false);		\
	}

#define STROMCL_SIMPLE_VARSTORE_TEMPLATE(NAME,BASE)			\
	STATIC_FUNCTION(void)									\
	pg_##NAME##_vstore(kern_data_store *kds,				\
					   kern_context *kcxt,					\
					   cl_uint colidx,						\
					   cl_uint rowidx,						\
					   pg_##NAME##_t pg_datum)				\
	{														\
		union {												\
			BASE		v_base;								\
			Datum		v_datum;							\
		} temp;												\
															\
		temp.v_datum = 0UL;									\
		temp.v_base = pg_datum.value;						\
		pg_common_vstore(kds, kcxt, colidx, rowidx,			\
						 temp.v_datum, pg_datum.isnull);	\
	}

#define STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)			\
	STATIC_FUNCTION(pg_##NAME##_t)							\
	pg_##NAME##_param(kern_context *kcxt,cl_uint param_id)	\
	{														\
		kern_parambuf *kparams = kcxt->kparams;				\
		pg_##NAME##_t result;								\
															\
		if (param_id < kparams->nparams &&					\
			kparams->poffset[param_id] > 0)					\
		{													\
			BASE *addr = (BASE *)							\
				((char *)kparams +							\
				 kparams->poffset[param_id]);				\
			result.value = *addr;							\
			result.isnull = false;							\
		}													\
		else												\
			result.isnull = true;							\
															\
		return result;										\
	}

#define STROMCL_SIMPLE_NULLTEST_TEMPLATE(NAME)				\
	STATIC_FUNCTION(pg_bool_t)								\
	pgfn_##NAME##_isnull(kern_context *kcxt, pg_##NAME##_t arg)	\
	{														\
		pg_bool_t result;									\
															\
		result.isnull = false;								\
		result.value = arg.isnull;							\
		return result;										\
	}														\
															\
	STATIC_FUNCTION(pg_bool_t)								\
	pgfn_##NAME##_isnotnull(kern_context *kcxt, pg_##NAME##_t arg)	\
	{														\
		pg_bool_t result;									\
															\
		result.isnull = false;								\
		result.value = !arg.isnull;							\
		return result;										\
	}

/*
 * Template of variable classes: fixed-length referenced by pointer
 * ----------------------------------------------------------------
 */
#define STROMCL_INDIRECT_VARREF_TEMPLATE(NAME,BASE)			\
	STATIC_FUNCTION(pg_##NAME##_t)							\
	pg_##NAME##_datum_ref(kern_context *kcxt,				\
						  void *datum,						\
						  cl_bool internal_format)			\
	{														\
		pg_##NAME##_t	result;								\
															\
		if (!datum)											\
			result.isnull = true;							\
		else												\
		{													\
			result.isnull = false;							\
			memcpy(&result.value, (BASE *) datum,			\
				   sizeof(BASE));							\
		}													\
		return result;										\
	}														\
															\
	STATIC_FUNCTION(pg_##NAME##_t)							\
	pg_##NAME##_vref(kern_data_store *kds,					\
					 kern_context *kcxt,					\
					 cl_uint colidx,						\
					 cl_uint rowidx)						\
	{														\
		void  *datum = kern_get_datum(kds,colidx,rowidx);	\
		return pg_##NAME##_datum_ref(kcxt,datum,false);		\
	}

#define STROMCL_INDIRECT_VARSTORE_TEMPLATE(NAME,BASE)		\
	STATIC_FUNCTION(void)									\
	pg_##NAME##_vstore(kern_data_store *kds,				\
					   kern_context *kcxt,					\
					   cl_uint colidx,						\
					   cl_uint rowidx,						\
					   pg_##NAME##_t pg_datum)				\
	{														\
		/* should not be used at this moment */				\
		assert(0);											\
	}

#define STROMCL_INDIRECT_PARAMREF_TEMPLATE(NAME,BASE)		\
	STATIC_FUNCTION(pg_##NAME##_t)							\
	pg_##NAME##_param(kern_context *kcxt,cl_uint param_id)	\
	{														\
		kern_parambuf *kparams = kcxt->kparams;				\
		pg_##NAME##_t result;								\
															\
		if (param_id < kparams->nparams &&					\
			kparams->poffset[param_id] > 0)					\
		{													\
			BASE *addr = (BASE *)((char *)kparams +			\
								  kparams->poffset[param_id]);	\
			memcpy(&result.value, addr, sizeof(BASE));		\
			result.isnull = false;							\
		}													\
		else												\
			result.isnull = true;							\
															\
		return result;										\
	}

/*
 * Macros to calculate CRC32 value.
 * (logic was copied from pg_crc32.c)
 */
#define INIT_LEGACY_CRC32(crc)		((crc) = 0xFFFFFFFF)
#define FIN_LEGACY_CRC32(crc)		((crc) ^= 0xFFFFFFFF)
#define EQ_LEGACY_CRC32(crc1,crc2)	((crc1) == (crc2))

STATIC_INLINE(cl_uint)
pg_common_comp_crc32(const cl_uint *crc32_table,
					 cl_uint hash,
					 const char *__data, cl_uint __len)
{
	cl_uint		__index;

	while (__len-- > 0)
	{
		__index = ((int) ((hash) >> 24) ^ *__data++) & 0xff;
		hash = crc32_table[__index] ^ ((hash) << 8);
	}
	return hash;
}

#define STROMCL_SIMPLE_COMP_CRC32_TEMPLATE(NAME,BASE)			\
	STATIC_FUNCTION(cl_uint)									\
	pg_##NAME##_comp_crc32(const cl_uint *crc32_table,			\
						   cl_uint hash, pg_##NAME##_t datum)	\
	{															\
		if (!datum.isnull)										\
		{														\
			union {												\
				BASE	as_base;								\
				char	as_char[sizeof(BASE)];					\
			} __data;											\
																\
			__data.as_base = datum.value;						\
			hash = pg_common_comp_crc32(crc32_table,			\
										hash,					\
										__data.as_char,			\
										sizeof(BASE));			\
		}														\
		return hash;											\
	}

#define STROMCL_SIMPLE_TYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_VARSTORE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_NULLTEST_TEMPLATE(NAME)			\
	STROMCL_SIMPLE_COMP_CRC32_TEMPLATE(NAME,BASE)

#define STROMCL_INDIRECT_TYPE_TEMPLATE(NAME,BASE)	\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_INDIRECT_VARREF_TEMPLATE(NAME,BASE)		\
	STROMCL_INDIRECT_VARSTORE_TEMPLATE(NAME,BASE)	\
	STROMCL_INDIRECT_PARAMREF_TEMPLATE(NAME,BASE)	\
	STROMCL_SIMPLE_NULLTEST_TEMPLATE(NAME)          \
	STROMCL_SIMPLE_COMP_CRC32_TEMPLATE(NAME,BASE)	\
	STATIC_INLINE(Datum)							\
	pg_##NAME##_to_datum(BASE *p_value)				\
	{												\
		return PointerGetDatum(p_value);			\
	}

/* pg_bool_t */
#ifndef PG_BOOL_TYPE_DEFINED
#define PG_BOOL_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(bool, cl_bool)
STATIC_INLINE(Datum)
pg_bool_to_datum(cl_bool value)
{
	return (Datum)(value ? true : false);
}
#endif

/* pg_int2_t */
#ifndef PG_INT2_TYPE_DEFINED
#define PG_INT2_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int2, cl_short)
STATIC_INLINE(Datum)
pg_int2_to_datum(cl_short value)
{
	return (Datum)(((Datum) value) & 0x0000ffffUL);
}
#endif

/* pg_int4_t */
#ifndef PG_INT4_TYPE_DEFINED
#define PG_INT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int4, cl_int)
STATIC_INLINE(Datum)
pg_int4_to_datum(cl_int value)
{
	return (Datum)(((Datum) value) & 0xffffffffUL);
}
#endif

/* pg_int8_t */
#ifndef PG_INT8_TYPE_DEFINED
#define PG_INT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int8, cl_long)
STATIC_INLINE(Datum)
pg_int8_to_datum(cl_long value)
{
	return (Datum)(value);
}
#endif

/* pg_float4_t */
#ifndef PG_FLOAT4_TYPE_DEFINED
#define PG_FLOAT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float4, cl_float)
STATIC_INLINE(Datum)
pg_float4_to_datum(cl_float value)
{
	return (Datum)((Datum)__float_as_int(value) & 0xffffffffUL);
}
#endif

/* pg_float8_t */
#ifndef PG_FLOAT8_TYPE_DEFINED
#define PG_FLOAT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float8, cl_double)
STATIC_INLINE(Datum)
pg_float8_to_datum(cl_double value)
{
	return (Datum)__double_as_longlong(value);
}
#endif

/*
 * Template of variable classes: variable-length variables
 * ---------------------------------------------------------------
 *
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
typedef struct varlena {
	cl_char		vl_len_[4];		/* Do not touch this field directly! */
	cl_char		vl_dat[1];
} varlena;

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
	((((varattrib_4b *) (PTR))->va_4byte.va_header >> 2) & 0x3FFFFFFF)
#define VARSIZE_1B(PTR) \
	((((varattrib_1b *) (PTR))->va_header >> 1) & 0x7F)
#define VARTAG_1B_E(PTR) \
	(((varattrib_1b_e *) (PTR))->va_tag)

#define VARRAWSIZE_4B_C(PTR)	\
	(((varattrib_4b *) (PTR))->va_compressed.va_rawsize)

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

/*
 * toast_raw_datum_size - return the raw (detoasted) size of a varlena
 * datum (including the VARHDRSZ header)
 */
STATIC_FUNCTION(size_t)
toast_raw_datum_size(kern_context *kcxt, varlena *attr)
{
	size_t		result;

	if (VARATT_IS_EXTERNAL(attr))
	{
		/* should not appear in kernel space */
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result = 0;
	}
	else if (VARATT_IS_COMPRESSED(attr))
	{
		/* here, va_rawsize is just the payload size */
		result = VARRAWSIZE_4B_C(attr) + VARHDRSZ;
	}
	else if (VARATT_IS_SHORT(attr))
	{
		/*
		 * we have to normalize the header length to VARHDRSZ or else the
		 * callers of this function will be confused.
		 */
		result = VARSIZE_SHORT(attr) - VARHDRSZ_SHORT + VARHDRSZ;
	}
	else
	{
		/* plain untoasted datum */
		result = VARSIZE(attr);
	}
	return result;
}

/*
 * Macro to extract a heap-tuple
 *
 * usage:
 * char   *addr;
 *
 * EXTRACT_HEAP_TUPLE_BEGIN(kds, htup, addr)
 *  -> addr shall point the device pointer of the first field, or NULL
 * EXTRACT_HEAP_TUPLE_NEXT(addr)
 *  -> addr shall point the device pointer of the second field, or NULL
 *     :
 * EXTRACT_HEAP_TUPLE_END()
 */
#define EXTRACT_HEAP_TUPLE_BEGIN(ADDR, kds, htup)						\
	do {																\
		HeapTupleHeaderData *__htup = (htup);							\
		kern_data_store	*__kds = (kds);									\
		kern_colmeta	__cmeta;										\
		cl_uint			__colidx = 0;									\
		cl_uint			__ncols;										\
		cl_bool			__heap_hasnull;									\
		char		   *__pos;											\
																		\
		if (!__htup)													\
			__ncols = 0;	/* to be considered as NULL */				\
		else															\
		{																\
			__heap_hasnull = ((__htup->t_infomask & HEAP_HASNULL) != 0); \
			__ncols = min((kds)->ncols,									\
						  __htup->t_infomask2 & HEAP_NATTS_MASK);		\
			__cmeta = __kds->colmeta[__colidx];							\
			__pos = (char *)(__htup) + __htup->t_hoff;					\
			assert(__pos == (char *)MAXALIGN(__pos));					\
		}																\
		if (__colidx < __ncols &&										\
			(!__heap_hasnull || !att_isnull(__colidx, __htup->t_bits)))	\
		{																\
			(ADDR) = __pos;												\
			__pos += (__cmeta.attlen > 0 ?								\
					  __cmeta.attlen :									\
					  VARSIZE_ANY(__pos));								\
		}																\
		else															\
			(ADDR) = NULL

#define EXTRACT_HEAP_TUPLE_NEXT(ADDR)									\
		__colidx++;														\
		if (__colidx < __ncols &&										\
			(!__heap_hasnull || !att_isnull(__colidx, __htup->t_bits)))	\
		{																\
			__cmeta = __kds->colmeta[__colidx];							\
																		\
			if (__cmeta.attlen > 0)										\
				__pos = (char *)TYPEALIGN(__cmeta.attalign, __pos);		\
			else if (!VARATT_NOT_PAD_BYTE(__pos))						\
				__pos = (char *)TYPEALIGN(__cmeta.attalign, __pos);		\
			(ADDR) = __pos;												\
			__pos += (__cmeta.attlen > 0 ?								\
					  __cmeta.attlen :									\
					  VARSIZE_ANY(__pos));								\
		}																\
		else															\
			(ADDR) = NULL

#define EXTRACT_HEAP_TUPLE_END()										\
	} while(0)

/*
 * kern_get_datum
 *
 * Reference to a particular datum on the supplied kernel data store.
 * It returns NULL, if it is a really null-value in context of SQL,
 * or in case when out of range with error code
 *
 * NOTE: We are paranoia for validation of the data being fetched from
 * the kern_data_store in row-format because we may see a phantom page
 * if the source transaction that required this kernel execution was
 * aborted during execution.
 * Once a transaction gets aborted, shared buffers being pinned are
 * released, even if DMA send request on the buffers are already
 * enqueued. In this case, the calculation result shall be discarded,
 * so no need to worry about correctness of the calculation, however,
 * needs to be care about address of the variables being referenced.
 */
STATIC_FUNCTION(void *)
kern_get_datum_tuple(kern_colmeta *colmeta,
					 HeapTupleHeaderData *htup,
					 cl_uint colidx)
{
	cl_bool		heap_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
	cl_uint		offset = htup->t_hoff;
	cl_uint		i, ncols = (htup->t_infomask2 & HEAP_NATTS_MASK);

	/* shortcut if colidx is obviously out of range */
	if (colidx >= ncols)
		return NULL;
	/* shortcut if tuple contains no NULL values */
	if (!heap_hasnull)
	{
		kern_colmeta	cmeta = colmeta[colidx];

		if (cmeta.attcacheoff >= 0)
			return (char *)htup + cmeta.attcacheoff;
	}
	/* regular path that walks on heap-tuple from the head */
	for (i=0; i < ncols; i++)
	{
		if (heap_hasnull && att_isnull(i, htup->t_bits))
		{
			if (i == colidx)
				return NULL;
		}
		else
		{
			kern_colmeta	cmeta = colmeta[i];
			char		   *addr;

			if (cmeta.attlen > 0)
				offset = TYPEALIGN(cmeta.attalign, offset);
			else if (!VARATT_NOT_PAD_BYTE((char *)htup + offset))
				offset = TYPEALIGN(cmeta.attalign, offset);
			/* TODO: overrun checks here */
			addr = ((char *) htup + offset);
			if (i == colidx)
				return addr;
			offset += (cmeta.attlen > 0
					   ? cmeta.attlen
					   : VARSIZE_ANY(addr));
		}
	}
	return NULL;
}

STATIC_FUNCTION(HeapTupleHeaderData *)
kern_get_tuple_row(kern_data_store *kds, cl_uint rowidx)
{
	kern_tupitem   *tupitem;

	if (rowidx >= kds->nitems)
		return NULL;	/* likely a BUG */
	tupitem = KERN_DATA_STORE_TUPITEM(kds, rowidx);
	return &tupitem->htup;
}

STATIC_FUNCTION(void *)
kern_get_datum_row(kern_data_store *kds,
				   cl_uint colidx, cl_uint rowidx)
{
	HeapTupleHeaderData *htup;

	if (colidx >= kds->ncols)
		return NULL;	/* likely a BUG */
	htup = kern_get_tuple_row(kds, rowidx);
	if (!htup)
		return NULL;
	return kern_get_datum_tuple(kds->colmeta, htup, colidx);
}

STATIC_FUNCTION(void *)
kern_get_datum_slot(kern_data_store *kds,
					cl_uint colidx, cl_uint rowidx)
{
	Datum	   *values = KERN_DATA_STORE_VALUES(kds,rowidx);
	cl_bool	   *isnull = KERN_DATA_STORE_ISNULL(kds,rowidx);
	kern_colmeta		cmeta = kds->colmeta[colidx];

	if (isnull[colidx])
		return NULL;
	if (cmeta.attbyval)
		return values + colidx;
	return (char *)values[colidx];
}

STATIC_INLINE(void *)
kern_get_datum(kern_data_store *kds,
			   cl_uint colidx, cl_uint rowidx)
{
	/* is it out of range? */
	if (colidx >= kds->ncols || rowidx >= kds->nitems)
		return NULL;
	if (kds->format == KDS_FORMAT_ROW ||
		kds->format == KDS_FORMAT_HASH)
		return kern_get_datum_row(kds, colidx, rowidx);
	if (kds->format == KDS_FORMAT_SLOT)
		return kern_get_datum_slot(kds, colidx, rowidx);
	/* TODO: put StromError_DataStoreCorruption error here */
	return NULL;
}

/*
 * functions to reference variable length variables
 */
STROMCL_SIMPLE_DATATYPE_TEMPLATE(varlena, varlena *)

STATIC_FUNCTION(pg_varlena_t)
pg_varlena_datum_ref(kern_context *kcxt,
					 void *datum,
					 cl_bool internal_format)
{
	varlena		   *vl_val = (varlena *) datum;
	pg_varlena_t	result;

	if (!datum)
		result.isnull = true;
	else
	{
		if (VARATT_IS_4B_U(vl_val) || VARATT_IS_1B(vl_val))
		{
			result.isnull = false;
			result.value = vl_val;
		}
		else
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_varlena_t)
pg_varlena_vref(kern_data_store *kds,
				kern_context *kcxt,
				cl_uint colidx,
				cl_uint rowidx)
{
	void   *datum = kern_get_datum(kds,colidx,rowidx);

	return pg_varlena_datum_ref(kcxt,datum,false);
}

STATIC_FUNCTION(pg_varlena_t)
pg_varlena_param(kern_context *kcxt, cl_uint param_id)
{
	kern_parambuf  *kparams = kcxt->kparams;
	pg_varlena_t	result;

	if (param_id < kparams->nparams &&
		kparams->poffset[param_id] > 0)
	{
		varlena *vl_val = (varlena *)((char *)kparams +
									  kparams->poffset[param_id]);
		if (VARATT_IS_4B_U(vl_val) || VARATT_IS_1B(vl_val))
		{
			result.value = vl_val;
			result.isnull = false;
		}
		else
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
	}
	else
		result.isnull = true;

	return result;
}

STROMCL_SIMPLE_NULLTEST_TEMPLATE(varlena)

STATIC_FUNCTION(cl_uint)
pg_varlena_comp_crc32(const cl_uint *crc32_table,
					  cl_uint hash, pg_varlena_t datum)
{
	if (!datum.isnull)
	{
		hash = pg_common_comp_crc32(crc32_table,
									hash,
									VARDATA_ANY(datum.value),
									VARSIZE_ANY_EXHDR(datum.value));
	}
	return hash;
}

#define STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)				\
	typedef pg_varlena_t	pg_##NAME##_t;

#define STROMCL_VARLENA_VARREF_TEMPLATE(NAME)				\
	STATIC_INLINE(pg_##NAME##_t)							\
	pg_##NAME##_datum_ref(kern_context *kcxt,				\
						  void *datum,						\
						  cl_bool internal_format)			\
	{														\
		return pg_varlena_datum_ref(kcxt,datum,internal_format);	\
	}														\
															\
	STATIC_INLINE(pg_##NAME##_t)							\
	pg_##NAME##_vref(kern_data_store *kds,					\
					 kern_context *kcxt,					\
					 cl_uint colidx,						\
					 cl_uint rowidx)						\
	{														\
		void  *datum = kern_get_datum(kds,colidx,rowidx);	\
		return pg_varlena_datum_ref(kcxt,datum,false);		\
	}

#define STROMCL_VARLENA_VARSTORE_TEMPLATE(NAME)				\
	STROMCL_SIMPLE_VARSTORE_TEMPLATE(NAME,varlena *)

#define STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)						\
	STATIC_INLINE(pg_##NAME##_t)									\
	pg_##NAME##_param(kern_context *kcxt, cl_uint param_id)			\
	{																\
		return pg_varlena_param(kcxt,param_id);						\
	}

#define STROMCL_VARLENA_NULLTEST_TEMPLATE(NAME)						\
	STATIC_INLINE(pg_bool_t)										\
	pgfn_##NAME##_isnull(kern_context *kcxt, pg_##NAME##_t arg)		\
	{																\
		return pgfn_varlena_isnull(kcxt, arg);					\
	}																\
	STATIC_INLINE(pg_bool_t)										\
	pgfn_##NAME##_isnotnull(kern_context *kcxt, pg_##NAME##_t arg)	\
	{																\
		return pgfn_varlena_isnotnull(kcxt, arg);					\
	}

#define STROMCL_VARLENA_COMP_CRC32_TEMPLATE(NAME)					\
	STATIC_INLINE(cl_uint)											\
	pg_##NAME##_comp_crc32(const cl_uint *crc32_table,				\
						   cl_uint hash, pg_##NAME##_t datum)		\
	{																\
		return pg_varlena_comp_crc32(crc32_table, hash, datum);		\
	}

#define STROMCL_VARLENA_TYPE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_VARREF_TEMPLATE(NAME)			\
	STROMCL_VARLENA_VARSTORE_TEMPLATE(NAME)			\
	STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)			\
	STROMCL_VARLENA_NULLTEST_TEMPLATE(NAME)			\
	STROMCL_VARLENA_COMP_CRC32_TEMPLATE(NAME)

/* pg_bytea_t */
#ifndef PG_BYTEA_TYPE_DEFINED
#define PG_BYTEA_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(bytea)
#endif

/* ------------------------------------------------------------------
 *
 * Declaration of utility functions
 *
 * ------------------------------------------------------------------ */

/*
 * pgstromStairlikeSum
 *
 * A utility routine to calculate sum of values when we have N items and 
 * want to know sum of items[i=0...k] (k < N) for each k, using reduction
 * algorithm on local memory (so, it takes log2(N) + 1 steps)
 *
 * The 'my_value' argument is a value to be set on the items[get_local_id(0)].
 * Then, these are calculate as follows:
 *
 *           init   1st      2nd         3rd         4th
 *           state  step     step        step        step
 * items[0] = X0 -> X0    -> X0       -> X0       -> X0
 * items[1] = X1 -> X0+X1 -> X0+X1    -> X0+X1    -> X0+X1
 * items[2] = X2 -> X2    -> X0+...X2 -> X0+...X2 -> X0+...X2
 * items[3] = X3 -> X2+X3 -> X0+...X3 -> X0+...X3 -> X0+...X3
 * items[4] = X4 -> X4    -> X4       -> X0+...X4 -> X0+...X4
 * items[5] = X5 -> X4+X5 -> X4+X5    -> X0+...X5 -> X0+...X5
 * items[6] = X6 -> X6    -> X4+...X6 -> X0+...X6 -> X0+...X6
 * items[7] = X7 -> X6+X7 -> X4+...X7 -> X0+...X7 -> X0+...X7
 * items[8] = X8 -> X8    -> X8       -> X8       -> X0+...X8
 * items[9] = X9 -> X8+X9 -> X8+9     -> X8+9     -> X0+...X9
 *
 * In Nth step, we split the array into 2^N blocks. In 1st step, a unit
 * containt odd and even indexed items, and this logic adds the last value
 * of the earlier half onto each item of later half. In 2nd step, you can
 * also see the last item of the earlier half (item[1] or item[5]) shall
 * be added to each item of later half (item[2] and item[3], or item[6]
 * and item[7]). Then, iterate this step until 2^(# of steps) less than N.
 *
 * Note that supplied items[] must have at least sizeof(cl_uint) *
 * get_local_size(0), and its contents shall be destroyed.
 * Also note that this function internally use barrier(), so unable to
 * use within if-blocks.
 */
STATIC_FUNCTION(cl_uint)
pgstromStairlikeSum(cl_uint my_value, cl_uint *total_sum)
{
	cl_uint	   *items = SHARED_WORKMEM(cl_uint);
	cl_uint		local_sz;
	cl_uint		local_id;
	cl_uint		unit_sz;
	cl_uint		stair_sum;
	cl_int		i, j;

	/* setup local size (pay attention, if 2D invocation) */
	local_sz = get_local_size();
	local_id = get_local_id();
	assert(local_id < local_sz);

	/* set initial value */
	items[local_id] = my_value;
	__syncthreads();

	for (i=1, unit_sz = local_sz; unit_sz > 0; i++, unit_sz >>= 1)
	{
		/* index of last item in the earlier half of each 2^i unit */
		j = (local_id & ~((1 << i) - 1)) | ((1 << (i-1)) - 1);

		/* add item[j] if it is later half in the 2^i unit */
		if ((local_id & (1 << (i - 1))) != 0)
			items[local_id] += items[j];
		__syncthreads();
	}
	if (total_sum)
		*total_sum = items[local_sz - 1];
	stair_sum = local_id == 0 ? 0 : items[local_id - 1];
	__syncthreads();
	return stair_sum;
}

/*
 * Utility functions to reference system columns
 */
STATIC_INLINE(Datum)
kern_getsysatt_ctid(kern_data_store *kds, HeapTupleHeaderData *htup)
{
	/*
	 * NOTE: Both of kern_tupitem and kern_hashitem put t_self just
	 * before the htup, with consideration of alignment.
	 */
	ItemPointerData	   *p_self = (ItemPointerData *)
		((char *)htup - (offsetof(kern_tupitem, htup) -
						 offsetof(kern_tupitem, t_self)));
	return (Datum) devptr_to_host(kds, p_self);
}

STATIC_INLINE(Datum)
kern_getsysatt_oid(kern_data_store *kds, HeapTupleHeaderData *htup)
{
	if ((htup->t_infomask & HEAP_HASOID) != 0)
		return *((cl_uint *)((char *) htup
							 + htup->t_hoff
							 - sizeof(cl_uint)));
	return 0;	/* InvalidOid */
}

STATIC_INLINE(Datum)
kern_getsysatt_xmin(kern_data_store *kds, HeapTupleHeaderData *htup)
{
	return (Datum) htup->t_choice.t_heap.t_xmin;
}

STATIC_INLINE(Datum)
kern_getsysatt_xmax(kern_data_store *kds, HeapTupleHeaderData *htup)
{
	return (Datum) htup->t_choice.t_heap.t_xmax;
}

STATIC_INLINE(Datum)
kern_getsysatt_cmin(kern_data_store *kds, HeapTupleHeaderData *htup)
{
	return (Datum) htup->t_choice.t_heap.t_field3.t_cid;
}

STATIC_INLINE(Datum)
kern_getsysatt_cmax(kern_data_store *kds, HeapTupleHeaderData *htup)
{
	return (Datum) htup->t_choice.t_heap.t_field3.t_cid;
}

STATIC_INLINE(Datum)
kern_getsysatt_tableoid(kern_data_store *kds, HeapTupleHeaderData *htup)
{
	return kds->table_oid;
}

/*
 * Forward function declarations) 
 * - kern_form_heaptuple needs type transform function on the data type; that
 *   has special internal format. Right now, only NUMERIC data type has own
 *   internal data format.
 */
STATIC_FUNCTION(cl_uint)
pg_numeric_to_varlena(kern_context *kcxt, char *vl_buffer,
					  Datum value, cl_bool isnull);

/*
 * compute_heaptuple_size
 */
STATIC_FUNCTION(cl_uint)
compute_heaptuple_size(kern_context *kcxt,
					   kern_data_store *kds,
					   Datum *tup_values,
					   cl_bool *tup_isnull,
					   cl_bool *tup_internal)
{
	cl_uint		t_hoff;
	cl_uint		datalen = 0;
	cl_uint		i, ncols = kds->ncols;
	cl_bool		heap_hasnull = false;

	/* compute data length */
	for (i=0; i < ncols; i++)
	{
		kern_colmeta	cmeta = kds->colmeta[i];

		if (tup_isnull[i])
			heap_hasnull = true;
		else
		{
			if (tup_internal && tup_internal[i])
			{
				/*
				 * NOTE: Right now, only numeric data type has internal
				 * data representation. It has to be transformed to the
				 * regular format prior to CPU write back.
				 */
				datalen = TYPEALIGN(sizeof(cl_uint), datalen);
				datalen += pg_numeric_to_varlena(kcxt, NULL,
												 tup_values[i],
												 tup_isnull[i]);
			}
			else if (cmeta.attlen > 0)
			{
				datalen = TYPEALIGN(cmeta.attalign, datalen);
			    datalen += cmeta.attlen;
			}
			else
			{
				Datum		datum = tup_values[i];
				cl_uint		vl_len = VARSIZE_ANY(datum);

				if (!VARATT_IS_1B(datum))
					datalen = TYPEALIGN(cmeta.attalign, datalen);
				datalen += vl_len;
			}
		}
	}

	/* compute header offset */
	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (heap_hasnull)
		t_hoff += bitmaplen(ncols);
	if (kds->tdhasoid)
		t_hoff += sizeof(cl_uint);
	t_hoff = MAXALIGN(t_hoff);

	return t_hoff + datalen;
}

/*
 * deform_kern_heaptuple
 *
 * Like deform_heap_tuple in host side, it extracts the supplied tuple-item
 * into tup_values / tup_isnull array. Note that pointer datum shall be
 * adjusted to the host-side address space.
 */
STATIC_FUNCTION(size_t)
deform_kern_heaptuple(kern_context *kcxt,
					  kern_data_store *kds,		/* in */
					  kern_tupitem *tupitem,	/* in */
					  cl_uint	nfields,		/* in */
					  cl_bool	as_host_addr,	/* in */
					  Datum	   *tup_values,		/* out */
					  cl_bool  *tup_isnull)		/* out */
{
	HeapTupleHeaderData *htup = &tupitem->htup;
	cl_uint		offset = htup->t_hoff;
	cl_uint		i, ncols = (htup->t_infomask2 & HEAP_NATTS_MASK);
	cl_bool		tup_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
	size_t		extra_len = 0;

	/* sanity check */
	assert(kds->format == KDS_FORMAT_ROW);

	/*
	 * In case of 'nfields' is less than length of array, we extract
	 * the first N columns only. On the other hands, t_informask2
	 * should not contain attributes than definition.
	 */
	assert(ncols <= kds->ncols);
	ncols = min(ncols, nfields);

	for (i=0; i < ncols; i++)
	{
		if (tup_hasnull && att_isnull(i, htup->t_bits))
		{
			tup_isnull[i] = true;
			tup_values[i] = 0;
		}
		else
		{
			kern_colmeta	cmeta = kds->colmeta[i];
			char		   *addr;

			if (cmeta.attlen > 0)
				offset = TYPEALIGN(cmeta.attalign, offset);
			else if (!VARATT_NOT_PAD_BYTE((char *)htup + offset))
				offset = TYPEALIGN(cmeta.attalign, offset);

			/*
			 * Store the value
			 */
			addr = ((char *) htup + offset);
			if (cmeta.attbyval)
			{
				if (cmeta.attlen == sizeof(cl_long))
					tup_values[i] = *((cl_long *) addr);
				else if (cmeta.attlen == sizeof(cl_int))
					tup_values[i] = *((cl_int *) addr);
				else if (cmeta.attlen == sizeof(cl_short))
					tup_values[i] = *((cl_short *) addr);
				else
				{
					assert(cmeta.attlen == sizeof(cl_char));
					tup_values[i] = *((cl_char *) addr);
				}
				offset += cmeta.attlen;
			}
			else
			{
				cl_uint		attlen = (cmeta.attlen > 0
									  ? cmeta.attlen
									  : VARSIZE_ANY(addr));
				/* store the device or host pointer according to the flag */
				tup_values[i] = (as_host_addr
								 ? devptr_to_host(kds, addr)
								 : PointerGetDatum(addr));
				offset += attlen;
				/* caller may need extra area */
				extra_len = TYPEALIGN(cmeta.attalign, extra_len);
				extra_len += attlen;
			}
			tup_isnull[i] = false;
		}
	}

	/*
	 * Fill up remaining columns if source tuple has less columns than
	 * length of the array; that is definition of the destination
	 */
	while (i < nfields)
		tup_isnull[i++] = true;

	return MAXALIGN(extra_len);
}

/*
 * form_kern_heaptuple
 *
 * A utility routine to build a kern_tupitem on the destination buffer
 * already allocated.
 *
 * kds          ... destination data-store
 * tupitem      ... kern_tupitem allocated on the kds
 * tuple_len    ... length of the tuple; shall be MAXALIGN(t_hoff) + data_len
 * heap_hasnull ... true, if tup_values/tup_isnull contains NULL
 * tup_values   ... array of result datum
 * tup_isnull   ... array of null flags
 * tup_internal ... array of internal flags
 */
STATIC_FUNCTION(cl_uint)
form_kern_heaptuple(kern_context *kcxt,
					kern_data_store *kds,
					kern_tupitem *tupitem,
					Datum *tup_values,
					cl_bool *tup_isnull,
					cl_bool *tup_internal)
{
	HeapTupleHeaderData *htup;
	cl_uint		i, ncols = kds->ncols;
	cl_bool		heap_hasnull;
	cl_ushort	t_infomask;
	cl_uint		t_hoff;
	cl_uint		curr;

	/* sanity checks */
	assert(kds->format == KDS_FORMAT_ROW);

	/* Does it have any NULL field? */
	heap_hasnull = false;
	for (i=0; i < ncols; i++)
	{
		if (tup_isnull[i])
		{
			heap_hasnull = true;
			break;
		}
	}
	t_infomask = (heap_hasnull ? HEAP_HASNULL : 0);

	/* Compute header offset */
	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (heap_hasnull)
		t_hoff += bitmaplen(ncols);
	if (kds->tdhasoid)
	{
		t_infomask |= HEAP_HASOID;
		t_hoff += sizeof(cl_uint);
	}
	t_hoff = MAXALIGN(t_hoff);

	/* setup header of kern_tupitem */
	// titem->t_len shall be set up later
	tupitem->t_self.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
	tupitem->t_self.ip_blkid.bi_lo = 0xffff;
	tupitem->t_self.ip_posid = 0;				/* InvalidOffsetNumber */
	htup = &tupitem->htup;

	/* setup HeapTupleHeader */
	// datum_len_ shall be set later
	htup->t_choice.t_datum.datum_typmod = kds->tdtypmod;
	htup->t_choice.t_datum.datum_typeid = kds->tdtypeid;
	htup->t_ctid.ip_blkid.bi_hi = 0xffff;
	htup->t_ctid.ip_blkid.bi_lo = 0xffff;
	htup->t_ctid.ip_posid = 0;
	htup->t_infomask2 = (ncols & HEAP_NATTS_MASK);
	// htup->t_infomask shall be set up later
	htup->t_hoff = t_hoff;
	curr = t_hoff;

	/* setup tuple body */
	for (i=0; i < ncols; i++)
	{
		kern_colmeta	cmeta = kds->colmeta[i];
		Datum			datum = tup_values[i];
		cl_bool			isnull = tup_isnull[i];

		if (isnull)
			htup->t_bits[i >> 3] &= ~(1 << (i & 0x07));
		else
		{
			if (heap_hasnull)
				htup->t_bits[i >> 3] |= (1 << (i & 0x07));

			if (tup_internal && tup_internal[i])
			{
				/*
				 * NOTE: Right now, only NUMERIC has internal data format.
				 * It has to be transformed again prior to CPU write back.
				 */
				curr = TYPEALIGN(sizeof(cl_uint), curr);
				curr += pg_numeric_to_varlena(kcxt, ((char *)htup + curr),
											  tup_values[i],
											  tup_isnull[i]);
			}
			else if (cmeta.attbyval)
			{
				char   *dest;

				while (TYPEALIGN(cmeta.attalign, curr) != curr)
					((char *)htup)[curr++] = '\0';
				dest = (char *)htup + curr;

				if (cmeta.attlen == sizeof(cl_long))
					*((cl_long *) dest) = (cl_long) datum;
				else if (cmeta.attlen == sizeof(cl_int))
					*((cl_int *) dest) = (cl_int) (datum & 0xffffffff);
				else if (cmeta.attlen == sizeof(cl_short))
					*((cl_short *) dest) = (cl_short) (datum & 0x0000ffff);
				else
				{
					assert(cmeta.attlen == sizeof(cl_char));
					*((cl_char *) dest) = (cl_char) (datum & 0x000000ff);
				}
				curr += cmeta.attlen;
			}
			else if (cmeta.attlen > 0)
			{
				while (TYPEALIGN(cmeta.attalign, curr) != curr)
					((char *)htup)[curr++] = '\0';

				memcpy((char *)htup + curr, (char *)datum, cmeta.attlen);

				curr += cmeta.attlen;
			}
			else
			{
				cl_uint		vl_len = VARSIZE_ANY(datum);

				t_infomask |= HEAP_HASVARWIDTH;
				/* put 0 and align here, if not a short varlena */
				if (!VARATT_IS_1B(datum))
				{
					while (TYPEALIGN(cmeta.attalign, curr) != curr)
						((char *)htup)[curr++] = '\0';
				}
				memcpy((char *)htup + curr, (char *)datum, vl_len);
				curr += vl_len;
			}
		}
	}
	curr += t_hoff;		/* add header length */
	tupitem->t_len = curr;
	SET_VARSIZE(&htup->t_choice.t_datum, MAXALIGN(curr));
	htup->t_infomask = t_infomask;

	return curr;
}

/* ------------------------------------------------------------
 *
 * Declarations of common built-in functions
 *
 * ------------------------------------------------------------
 */

/* A utility function to evaluate pg_bool_t value as if built-in
 * bool variable.
 */
STATIC_INLINE(cl_bool)
EVAL(pg_bool_t arg)
{
	if (!arg.isnull && arg.value != 0)
		return true;
	return false;
}

/*
 * Support routine for BoolExpr
 */
STATIC_INLINE(pg_bool_t)
operator ! (pg_bool_t arg)
{
	arg.value = !arg.value;
	return arg;
}

STATIC_INLINE(pg_bool_t)
operator && (pg_bool_t arg1, pg_bool_t arg2)
{
	pg_bool_t	result;

	/* If either of expression is FALSE, entire BoolExpr is also FALSE */
	if ((!arg1.isnull && !arg1.value) ||
		(!arg2.isnull && !arg2.value))
	{
		result.isnull = false;
		result.value  = false;
	}
	else
	{
		result.isnull = arg1.isnull | arg2.isnull;
		result.value  = (arg1.value && arg2.value ? true : false);
	}
	return result;
}

STATIC_INLINE(pg_bool_t)
operator || (pg_bool_t arg1, pg_bool_t arg2)
{
	pg_bool_t	result;

	/* If either of expression is TRUE, entire BoolExpr is also TRUE */
	if ((!arg1.isnull && arg1.value) ||
		(!arg2.isnull && arg2.value))
	{
		result.isnull = false;
		result.value  = true;
	}
	else
	{
		result.isnull = arg1.isnull | arg2.isnull;
		result.value  = (arg1.value || arg2.value ? true : false);
	}
	return result;
}

/*
 * macros for general binary compare functions
 */
#define devfunc_int_comp(x,y)					\
	((x) < (y) ? -1 : ((x) > (y) ? 1 : 0))

#define devfunc_float_comp(x,y)					\
	(isnan(x)									\
	 ? (isnan(y)								\
		? 0		/* NAN = NAM */					\
		: 1)	/* NAN > non-NAN */				\
	 : (isnan(y)								\
		? -1	/* non-NAN < NAN */				\
		: devfunc_int_comp((x),(y))))

/*
 * Functions for BooleanTest
 */
STATIC_INLINE(pg_bool_t)
pgfn_bool_is_true(kern_context *kcxt, pg_bool_t result)
{
	result.value = (!result.isnull && result.value);
	result.isnull = false;
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_bool_is_not_true(kern_context *kcxt, pg_bool_t result)
{
	result.value = (result.isnull || !result.value);
	result.isnull = false;
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_bool_is_false(kern_context *kcxt, pg_bool_t result)
{
	result.value = (!result.isnull && !result.value);
	result.isnull = false;
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_bool_is_not_false(kern_context *kcxt, pg_bool_t result)
{
	result.value = (result.isnull || result.value);
	result.isnull = false;
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_bool_is_unknown(kern_context *kcxt, pg_bool_t result)
{
	result.value = result.isnull;
	result.isnull = false;
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_bool_is_not_unknown(kern_context *kcxt, pg_bool_t result)
{
	result.value = !result.isnull;
	result.isnull = false;
	return result;
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_COMMON_H */
