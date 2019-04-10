/*
 * cuda_common.h
 *
 * A common header for CUDA device code
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
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

/* ---- Check minimum required CUDA version ---- */
#ifdef __CUDACC_RTC__
#if __CUDACC_VER_MAJOR__ < 9 || \
   (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ < 1)
#error PG-Strom requires CUDA 9.1 or later. Use newer version.
#endif
#endif /* __CUDACC_RTC__ */

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
#else	/* __CUDACC__ */
typedef long				cl_long;
typedef unsigned long		cl_ulong;
#endif	/* !__CUDACC__ */
#ifdef __CUDACC__
#include <cuda_fp16.h>
typedef __half				cl_half;
#else
/* Host code has no __half definition, so put dummy definition */
typedef unsigned short		cl_half;
#endif	/* __CUDACC__ */
typedef float				cl_float;
typedef double				cl_double;
#ifdef __CUDACC__
typedef cl_ulong			uintptr_t;
#endif

/* PG's utility macros */
#ifndef PG_STROM_H
#ifdef offsetof
#undef offsetof
#endif
#define offsetof(TYPE,FIELD)			((long) &((TYPE *)0UL)->FIELD)

#ifdef __NVCC__
/*
 * At CUDA10, we found nvcc replaces the offsetof above by __builtin_offsetof
 * regardless of our macro definitions. It is mostly equivalent, however, it
 * does not support offset calculation which includes run-time values.
 * E.g) offsetof(kds, colmeta[kds->ncols]) made an error.
 */
#define __builtin_offsetof(TYPE,FIELD)	((long) &((TYPE *)0UL)->FIELD)
#endif

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
#if MAXIMUM_ALIGNOF == 16
#define MAXIMUM_ALIGNOF_SHIFT	4
#elif MAXIMUM_ALIGNOF == 8
#define MAXIMUM_ALIGNOF_SHIFT	3
#elif MAXIMUM_ALIGNOF == 4
#define MAXIMUM_ALIGNOF_SHIFT	2
#else
#error Unexpected MAXIMUM_ALIGNOF definition
#endif	/* MAXIMUM_ALIGNOF */

/*
 * If NVCC includes this file, some inline function needs declarations of
 * basic utility functions.
 */
#ifndef __CUDACC_RTC__
#include <assert.h>
#include <stdio.h>
#endif

#define Assert(cond)	assert(cond)

/* Another basic type definitions */
typedef cl_ulong	hostptr_t;
typedef cl_ulong	Datum;
typedef struct nameData
{
	char		data[NAMEDATALEN];
} NameData;

#define PointerGetDatum(X)	((Datum) (X))
#define DatumGetPointer(X)	((char *) (X))

#define SET_1_BYTE(value)	(((Datum) (value)) & 0x000000ffL)
#define SET_2_BYTES(value)	(((Datum) (value)) & 0x0000ffffL)
#define SET_4_BYTES(value)	(((Datum) (value)) & 0xffffffffL)
#define SET_8_BYTES(value)	((Datum) (value))

#define READ_INT8_PTR(addr)		SET_1_BYTE(*((cl_uchar *)(addr)))
#define READ_INT16_PTR(addr)	SET_2_BYTES(*((cl_ushort *)(addr)))
#define READ_INT32_PTR(addr)	SET_4_BYTES(*((cl_uint *)(addr)))
#define READ_INT64_PTR(addr)	SET_8_BYTES(*((cl_ulong *)(addr)))

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

#define Compare(a,b)	((a) > (b) ? 1 : ((a) < (b) ? -1 : 0))

/* same as host side get_next_log2() */
#define get_next_log2(value)								\
	((value) == 0 ? 0 : (sizeof(cl_ulong) * BITS_PER_BYTE - \
						 __clzll((cl_ulong)(value) - 1)))
/*
 * Limitation of types
 */
#ifndef SHRT_MAX
#define SHRT_MAX		32767
#endif
#ifndef SHRT_MIN
#define SHRT_MIN		(-32767-1)
#endif
#ifndef USHRT_MAX
#define USHRT_MAX		65535
#endif
#ifndef INT_MAX
#define INT_MAX			2147483647
#endif
#ifndef INT_MIN
#define INT_MIN			(-INT_MAX - 1)
#endif
#ifndef UINT_MAX
#define UINT_MAX		4294967295U
#endif
#ifndef LONG_MAX
#define LONG_MAX		0x7FFFFFFFFFFFFFFFLL
#endif
#ifndef LONG_MIN
#define LONG_MIN        (-LONG_MAX - 1LL)
#endif
#ifndef ULONG_MAX
#define ULONG_MAX		0xFFFFFFFFFFFFFFFFULL
#endif
#ifndef HALF_MAX
#define HALF_MAX		__short_as_half(0x7bff)
#endif
#ifndef HALF_MIN
#define HALF_MIN		__short_as_half(0x0400)
#endif
#ifndef HALF_INFINITY
#define HALF_INFINITY	__short_as_half(0x0x7c00)
#endif
#ifndef FLT_MAX
#define FLT_MAX			__int_as_float(0x7f7fffffU)
#endif
#ifndef FLT_MIN
#define FLT_MIN			__int_as_float(0x00800000U)
#endif
#ifndef FLT_INFINITY
#define FLT_INFINITY	__int_as_float(0x7f800000U)
#endif
#ifndef FLT_NAN
#define FLT_NAN			__int_as_float(0x7fffffffU)
#endif
#ifndef DBL_MAX
#define DBL_MAX			__longlong_as_double(0x7fefffffffffffffULL)
#endif
#ifndef DBL_MIN
#define DBL_MIN			__longlong_as_double(0x0010000000000000ULL)
#endif
#ifndef DBL_INFINITY
#define DBL_INFINITY	__longlong_as_double(0x7ff0000000000000ULL)
#endif
#ifndef DBL_NAN
#define DBL_NAN			__longlong_as_double(0x7fffffffffffffffULL)
#endif

/*
 * Alignment macros
 */
#define TYPEALIGN(ALIGNVAL,LEN)	\
	(((uintptr_t) (LEN) + ((ALIGNVAL) - 1)) & ~((uintptr_t) ((ALIGNVAL) - 1)))
#define TYPEALIGN_DOWN(ALIGNVAL,LEN) \
	(((uintptr_t) (LEN)) & ~((uintptr_t) ((ALIGNVAL) - 1)))
#define INTALIGN(LEN)			TYPEALIGN(sizeof(cl_int), (LEN))
#define INTALIGN_DOWN(LEN)		TYPEALIGN_DOWN(sizeof(cl_int), (LEN))
#define LONGALIGN(LEN)          TYPEALIGN(sizeof(cl_long), (LEN))
#define LONGALIGN_DOWN(LEN)     TYPEALIGN_DOWN(sizeof(cl_long), (LEN))
#define MAXALIGN(LEN)			TYPEALIGN(MAXIMUM_ALIGNOF, (LEN))
#define MAXALIGN_DOWN(LEN)		TYPEALIGN_DOWN(MAXIMUM_ALIGNOF, (LEN))
#endif		/* PG_STROM_H */
/*
 * wider alignments
 */
#define STROMALIGN_LEN			16
#define STROMALIGN(LEN)			TYPEALIGN(STROMALIGN_LEN,(LEN))
#define STROMALIGN_DOWN(LEN)	TYPEALIGN_DOWN(STROMALIGN_LEN,(LEN))

#define GPUMEMALIGN_LEN			1024
#define GPUMEMALIGN(LEN)		TYPEALIGN(GPUMEMALIGN_LEN,(LEN))
#define GPUMEMALIGN_DOWN(LEN)	TYPEALIGN_DOWN(GPUMEMALIGN_LEN,(LEN))

#define BLCKALIGN(LEN)			TYPEALIGN(BLCKSZ,(LEN))
#define BLCKALIGN_DOWN(LEN)		TYPEALIGN_DOWN(BLCKSZ,(LEN))

#ifdef __CUDACC__
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
#define get_group_id()			(blockIdx.x)
#define get_num_groups()		(gridDim.x)
#define get_local_id()			(threadIdx.x)
#define get_local_size()		(blockDim.x)
#define get_global_id()			(threadIdx.x + blockIdx.x * blockDim.x)
#define get_global_size()		(blockDim.x * gridDim.x)
#define get_global_base()		(blockIdx.x * blockDim.x)

#else	/* __CUDACC__ */
typedef uintptr_t		hostptr_t;
#endif	/* !__CUDACC__ */

/*
 * Template of static function declarations
 *
 * CUDA compilar raises warning if static functions are not used, but
 * we can restain this message with"unused" attribute of function/values.
 * STATIC_INLINE / STATIC_FUNCTION packs common attributes to be
 * assigned on host/device functions
 */
#define MAXTHREADS_PER_BLOCK		1024
#ifdef __CUDACC__
#define STATIC_INLINE(RET_TYPE)					\
	__host__ __device__ __forceinline__			\
	static RET_TYPE __attribute__ ((unused))
#define STATIC_FUNCTION(RET_TYPE)				\
	__host__ __device__							\
	static RET_TYPE __attribute__ ((unused))
#define DEVICE_ONLY_INLINE(RET_TYPE)			\
	__device__ __forceinline__					\
	static RET_TYPE __attribute__ ((unused))
#define DEVICE_ONLY_FUNCTION(RET_TYPE)			\
	__device__									\
	static RET_TYPE __attribute__ ((unused))
#define KERNEL_FUNCTION(RET_TYPE)				\
	extern "C" __global__ RET_TYPE
#define KERNEL_FUNCTION_NUMTHREADS(RET_TYPE,NUM_THREADS) \
	extern "C" __global__ RET_TYPE __launch_bounds__(NUM_THREADS)
#define KERNEL_FUNCTION_MAXTHREADS(RET_TYPE)	\
	extern "C" __global__ RET_TYPE __launch_bounds__(MAXTHREADS_PER_BLOCK)
#else	/* __CUDACC__ */
#define STATIC_INLINE(RET_TYPE)		static inline RET_TYPE
#define STATIC_FUNCTION(RET_TYPE)	static inline RET_TYPE
#define KERNEL_FUNCTION(RET_TYPE)	RET_TYPE
#define KERNEL_FUNCTION_MAXTHREADS(RET_TYPE)	KERNEL_FUNCTION(RET_TYPE)
#endif	/* !__CUDACC__ */

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
#define StromError_Success				0		/* OK */
#define StromError_Generic				1001	/* General error */
#define StromError_CpuReCheck			1002	/* Re-checked by CPU */
#define StromError_InvalidValue			1003	/* Invalid values */
#define StromError_DataStoreNoSpace		1004	/* No space left on KDS */
#define StromError_WrongCodeGeneration	1005	/* Wrong GPU code generation */
#define StromError_OutOfMemory			1006	/* Out of Memory */
#define StromError_DataCorruption		1007	/* Data corruption */

/*
 * Kernel functions identifier
 */
#define StromKernel_HostPGStrom						0x0001
#define StromKernel_CudaRuntime						0x0002
#define StromKernel_NVMeStrom						0x0003
#define StromKernel_gpuscan_main_row				0x0101
#define StromKernel_gpuscan_main_block				0x0102
#define StromKernel_gpuscan_main_arrow				0x0103
#define StromKernel_gpujoin_main					0x0201
#define StromKernel_gpujoin_right_outer				0x0202
#define StromKernel_gpupreagg_setup_row				0x0301
#define StromKernel_gpupreagg_setup_block			0x0302
#define StromKernel_gpupreagg_setup_arrow			0x0303
#define StromKernel_gpupreagg_nogroup_reduction		0x0304
#define StromKernel_gpupreagg_groupby_reduction		0x0305
#define StromKernel_gpusort_setup_column			0x0401
#define StromKernel_gpusort_bitonic_local			0x0412
#define StromKernel_gpusort_bitonic_step			0x0413
#define StromKernel_gpusort_bitonic_merge			0x0414

#define KERN_ERRORBUF_FILENAME_LEN		24
typedef struct
{
	cl_int		errcode;	/* one of the StromError_* */
	cl_short	kernel;		/* one of the StromKernel_* */
	cl_short	lineno;		/* line number STROM_SET_ERROR is called */
	char		filename[KERN_ERRORBUF_FILENAME_LEN];
} kern_errorbuf;

/*
 * kern_context - a set of run-time information
 */
struct kern_parambuf;

#ifndef __CUDACC__
/* just a dummy for host code */
#define KERN_CONTEXT_VARLENA_BUFSZ			1
#endif	/* __CUDACC__ */
#define KERN_CONTEXT_VARLENA_BUFSZ_LIMIT	2048

typedef struct
{
	kern_errorbuf	e;
	struct kern_parambuf *kparams;
	cl_char		   *vlpos;
	cl_char			vlbuf[KERN_CONTEXT_VARLENA_BUFSZ];
} kern_context;

#define DECL_KERNEL_CONTEXT(NAME,VARLENA_BUFSZ)							\
	union {																\
		kern_context NAME;												\
		char __kern_context_dummy__[offsetof(kern_context, vlbuf) +		\
									MAXALIGN(VARLENA_BUFSZ)];			\
	}

#define INIT_KERNEL_CONTEXT(kcxt,kfunction,__kparams)		\
	do {													\
		(kcxt)->e.errcode = StromError_Success;				\
		(kcxt)->e.kernel = StromKernel_##kfunction;			\
		(kcxt)->e.lineno = 0;								\
		(kcxt)->e.filename[0] = '\0';						\
		(kcxt)->kparams = (__kparams);						\
		assert((cl_ulong)(__kparams) == MAXALIGN(__kparams)); \
		(kcxt)->vlpos = (kcxt)->vlbuf;						\
	} while(0)

#define PTR_ON_VLBUF(kcxt,ptr,len)							\
	((char *)(ptr) >= (kcxt)->vlbuf &&						\
	 (char *)(ptr) + (len) <= (kcxt)->vlbuf + KERN_CONTEXT_VARLENA_BUFSZ)

STATIC_INLINE(void *)
kern_context_alloc(kern_context *kcxt, size_t len)
{
	char   *pos = (char *)MAXALIGN(kcxt->vlpos);

	if (pos >= kcxt->vlbuf &&
		pos + len <= kcxt->vlbuf + KERN_CONTEXT_VARLENA_BUFSZ)
	{
		kcxt->vlpos = pos + len;
		return pos;
	}
	return NULL;
}

#ifdef __CUDACC__
/*
 * It sets an error code unless no significant error code is already set.
 * Also, CpuReCheck has higher priority than RowFiltered because CpuReCheck
 * implies device cannot run the given expression completely.
 * (Usually, due to compressed or external varlena datum)
 */
STATIC_INLINE(void)
__STROM_SET_ERROR(kern_errorbuf *p_kerror, cl_int errcode,
				  const char *filename, cl_int lineno)
{
	cl_int		oldcode = p_kerror->errcode;

	if (oldcode == StromError_Success &&
		errcode != StromError_Success)
	{
		const char *pos;
		cl_int		fn_len;

		for (pos=filename; *pos != '\0'; pos++)
		{
			if (pos[0] == '/' && pos[1] != '\0')
				filename = pos + 1;
		}
		p_kerror->errcode = errcode;
		p_kerror->lineno = lineno;

		fn_len = Min(pos - filename, KERN_ERRORBUF_FILENAME_LEN - 1);
		memcpy(p_kerror->filename, filename, fn_len);
		p_kerror->filename[fn_len] = '\0';
	}
}

#define STROM_SET_ERROR(p_kerror, errcode)		\
	__STROM_SET_ERROR((p_kerror), (errcode), __FILE__, __LINE__)

/*
 * kern_writeback_error_status
 */
DEVICE_ONLY_INLINE(void)
kern_writeback_error_status(kern_errorbuf *result, kern_errorbuf *my_error)
{
	/*
	 * It writes back a thread local error status only when the global
	 * error status is not set yet and the caller thread contains any
	 * error status. Elsewhere, we don't involves any atomic operation
	 * in the most of code path.
	 */
	if (my_error->errcode != StromError_Success &&
		atomicCAS(&result->errcode,
				  StromError_Success,
				  my_error->errcode) == StromError_Success)
	{
		/* only primary error workgroup can come into */
		result->kernel = my_error->kernel;
		result->lineno = my_error->lineno;
		memcpy(result->filename,
			   my_error->filename,
			   KERN_ERRORBUF_FILENAME_LEN);
	}
}
#else	/* __CUDACC__ */
/*
 * If case when STROM_SET_ERROR is called in the host code,
 * it raises an error using ereport()
 */
#define STROM_SET_ERROR(p_kerror, errcode)							\
	elog(ERROR, "%s:%d %s", __FUNCTION__, __LINE__, errorText(errcode))
#endif	/* !__CUDACC__ */

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
 * LaneId() - reference to the %laneid register
 */
STATIC_INLINE(cl_uint) LaneId(void)
{
	cl_uint		ret;
	asm volatile("mov.u32 %0, %laneid;" : "=r"(ret) );
	return ret;
}

/*
 * TotalShmemSize() - reference to the %total_smem_size
 */
STATIC_INLINE(cl_uint) TotalShmemSize(void)
{
	cl_uint		ret;
	asm volatile("mov.u32 %0, %total_smem_size;" : "=r"(ret) );
	return ret;
}

/*
 * DynamicShmemSize() - reference to the %dynamic_smem_size
 */
STATIC_INLINE(cl_uint) DynamicShmemSize(void)
{
	cl_uint		ret;
	asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret) );
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
#endif		/* __CUDACC__ */

#ifndef PG_STROM_H
/* definitions at storage/block.h */
typedef cl_uint		BlockNumber;
#define InvalidBlockNumber		((BlockNumber) 0xFFFFFFFF)
#define MaxBlockNumber			((BlockNumber) 0xFFFFFFFE)

/* details are defined at cuda_gpuscan.h */
struct PageHeaderData;

/* definitions at access/htup_details.h */
typedef struct {
	struct {
		cl_ushort	bi_hi;
		cl_ushort	bi_lo;
	} ip_blkid;
	cl_ushort		ip_posid;
} ItemPointerData;

typedef struct HeapTupleFields
{
	cl_uint			t_xmin;		/* inserting xact ID */
	cl_uint			t_xmax;		/* deleting or locking xact ID */
	union
	{
		cl_uint		t_cid;		/* inserting or deleting command ID, or both */
		cl_uint		t_xvac;		/* old-style VACUUM FULL xact ID */
    }	t_field3;
} HeapTupleFields;

typedef struct DatumTupleFields
{
	cl_int		datum_len_;		/* varlena header (do not touch directly!) */
	cl_int		datum_typmod;	/* -1, or identifier of a record type */
	cl_uint		datum_typeid;	/* composite type OID, or RECORDOID */
} DatumTupleFields;

typedef struct {
	union {
		HeapTupleFields		t_heap;
		DatumTupleFields	t_datum;
	} t_choice;

	ItemPointerData	t_ctid;			/* current TID of this or newer tuple */

	cl_ushort		t_infomask2;	/* number of attributes + various flags */
	cl_ushort		t_infomask;		/* various flag bits, see below */
	cl_uchar		t_hoff;			/* sizeof header incl. bitmap, padding */
	/* ^ - 23 bytes - ^ */
	cl_uchar		t_bits[1];		/* bitmap of NULLs -- VARIABLE LENGTH */
} HeapTupleHeaderData;

#define att_isnull(ATT, BITS) (!((BITS)[(ATT) >> 3] & (1 << ((ATT) & 0x07))))
#define BITMAPLEN(NATTS) (((int)(NATTS) + BITS_PER_BYTE - 1) / BITS_PER_BYTE)

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

#endif	/* PG_STROM_H */

/*
 * kern_data_store
 *
 * +---------------------------------------------------------------+
 * | Common header portion of the kern_data_store                  |
 * |         :                                                     |
 * | 'format' determines the layout below                          |
 * |                                                               |
 * | 'nitems' and 'nrooms' mean number of tuples except for BLOCK  |
 * | format. In BLOCK format, these fields mean number of the      |
 * | PostgreSQL blocks. We cannot know exact number of tuples      |
 * | without scanning of the data-store                            |
 * +---------------------------------------------------------------+
 * | Attributes of columns                                         |
 * |                                                               |
 * | kern_colmeta colmeta[0]                                       |
 * | kern_colmeta colmeta[1]       <--+                            |
 * |        :                         :   column definition of     |
 * | kern_colmeta colmeta[ncols-1] <--+-- regular tables           |
 * |        :                                                      |
 * | kern_colmeta colmeta[idx_subattrs]     <--+                   |
 * |        :                                  : field definition  |
 * | kern_colmeta colmeta[idx_subattrs +       : of composite type |
 * |                      num_subattrs - 1] <--+                   |
 * +---------------------------------------------------------------+
 * | <slot format> | <row format> / <hash format> | <block format> |
 * +---------------+------------------------------+----------------+
 * | values/isnull | Offset to the first hash-    | BlockNumber of |
 * | pair of the   | item for each slot (*).      | PostgreSQL;    |
 * | 1st tuple     |                              | used to setup  |
 * | +-------------+ (*) nslots=0 if row-format,  | ctid system    |
 * | | values[0]   | thus, it has no offset to    | column.        |
 * | |    :        | hash items.                  |                |
 * | | values[M-1] |                              | (*) N=nrooms   |
 * | +-------------+  hash_slot[0]                | block_num[0]   |
 * | | isnull[0]   |  hash_slot[1]                | block_num[1]   |
 * | |    :        |      :                       |      :         |
 * | | isnull[M-1] |  hash_slot[nslots-1]         |      :         |
 * +-+-------------+------------------------------+ block_num[N-1] |
 * | values/isnull | Offset to the individual     +----------------+ 
 * | pair of the   | kern_tupitem.                |     ^          |
 * | 2nd tuple     |                              |     |          |
 * | +-------------+ row_index[0]                 | Raw PostgreSQL |
 * | | values[0]   | row_index[1]                 | Block-0        |
 * | |    :        |    :                         |     |          |
 * | | values[M-1] | row_index[nitems-1]          |   BLCKSZ(8KB)  |
 * | +-------------+--------------+---------------+     |          |
 * | | isnull[0]   |    :         |       :       |     v          |
 * | |    :        +--------------+---------------+----------------+
 * | | isnull[M-1] | kern_tupitem | kern_hashitem |                |
 * +-+-------------+--------------+---------------+                |   
 * | values/isnull | kern_tupitem | kern_hashitem | Raw PostgreSQL |
 * | pair of the   +--------------+---------------+ Block-1        |
 * | 3rd tuple     | kern_tupitem | kern_hashitem |                |
 * |      :        |     :        |     :         |     :          |
 * |      :        |     :        |     :         |     :          |
 * +---------------+--------------+---------------+----------------+
 */
#include "arrow_defs.h"

#define TYPE_KIND__BASE			'b'
#define TYPE_KIND__ARRAY		'a'
#define TYPE_KIND__COMPOSITE	'c'
#define TYPE_KIND__DOMAIN		'd'
#define TYPE_KIND__ENUM			'e'
#define TYPE_KIND__PSEUDO		'p'
#define TYPE_KIND__RANGE		'r'

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
	/* one of TYPE_KIND__* */
	cl_char			atttypkind;
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
	cl_ushort		idx_subattrs;
	cl_ushort		num_subattrs;

	/*
	 * (only arrow format)
	 * @attoptions keeps extra information of Apache Arrow type. Unlike
	 * PostgreSQL types, it can have variation of data accuracy in time
	 * related data types, or precision in decimal data type.
	 */
	ArrowTypeOptions attopts;
	cl_uint			nullmap_offset;
	cl_uint			nullmap_length;
	cl_uint			values_offset;
	cl_uint			values_length;
	cl_uint			extra_offset;
	cl_uint			extra_length;

	/*
	 * (only column format)
	 * @va_offset is offset of the values array from the kds-head.
	 * @va_length is length of the values array and extra area which is
	 * used to dictionary of varlena or nullmap of fixed-length values.
	 */
	cl_uint			va_offset;
	cl_uint			va_length;
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
	cl_uint				next;	/* offset of the next (PACKED) */
	cl_uint				rowid;	/* unique identifier of this hash entry */
	cl_uint				__padding__; /* for alignment */
	kern_tupitem		t;		/* HeapTuple of this entry */
} kern_hashitem;

#define KDS_FORMAT_ROW			1
#define KDS_FORMAT_SLOT			2
#define KDS_FORMAT_HASH			3	/* inner hash table for GpuHashJoin */
#define KDS_FORMAT_BLOCK		4	/* raw blocks for direct loading */
#define KDS_FORMAT_COLUMN		5	/* columnar based storage format */
#define KDS_FORMAT_ARROW		6	/* apache arrow format */

typedef struct {
	size_t			length;		/* length of this data-store */
	/*
	 * NOTE: {nitems + usage} must be aligned to 64bit because these pair of
	 * values can be updated atomically using cmpxchg.
	 */
	cl_uint			nitems; 	/* number of rows in this store */
	cl_uint			usage;		/* usage of this data-store (PACKED) */
	cl_uint			nrooms;		/* number of available rows in this store */
	cl_uint			ncols;		/* number of columns in this store */
	cl_char			format;		/* one of KDS_FORMAT_* above */
	cl_char			has_notbyval; /* true, if any of column is !attbyval */
	cl_char			has_attnames; /* true, if attname array exists next to
								   * to the colmeta array */
	cl_char			tdhasoid;	/* copy of TupleDesc.tdhasoid */
	cl_uint			tdtypeid;	/* copy of TupleDesc.tdtypeid */
	cl_int			tdtypmod;	/* copy of TupleDesc.tdtypmod */
	cl_uint			table_oid;	/* OID of the table (only if GpuScan) */
	cl_uint			nslots;		/* width of hash-slot (only HASH format) */
	cl_uint			nrows_per_block; /* average number of rows per
									  * PostgreSQL block (only BLOCK format) */
	cl_uint			nr_colmeta;	/* number of colmeta[] array elements;
								 * maybe, >= ncols, if any composite types */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER]; /* metadata of columns */
} kern_data_store;

/* attribute number of system columns */
#define SelfItemPointerAttributeNumber			(-1)
#define ObjectIdAttributeNumber					(-2)
#define MinTransactionIdAttributeNumber			(-3)
#define MinCommandIdAttributeNumber				(-4)
#define MaxTransactionIdAttributeNumber			(-5)
#define MaxCommandIdAttributeNumber				(-6)
#define TableOidAttributeNumber					(-7)
#define FirstLowInvalidHeapAttributeNumber		(-8)

/*
 * MEMO: Support of 32GB KDS - KDS with row-, hash- and column-format
 * internally uses 32bit offset value from the head or base address.
 * We have assumption here - any objects pointed by the offset value
 * is always aligned to MAXIMUM_ALIGNOF boundary (64bit).
 * It means we can use 32bit offset to represent up to 32GB range (35bit).
 */
STATIC_INLINE(cl_uint)
__kds_packed(size_t offset)
{
	Assert((offset & (~((size_t)(~0U) << MAXIMUM_ALIGNOF_SHIFT))) == 0);
	return (cl_uint)(offset >> MAXIMUM_ALIGNOF_SHIFT);
}

STATIC_INLINE(size_t)
__kds_unpack(cl_uint offset)
{
	return (size_t)offset << MAXIMUM_ALIGNOF_SHIFT;
}
#define KDS_OFFSET_MAX_SIZE		((size_t)UINT_MAX << MAXIMUM_ALIGNOF_SHIFT)

/* 'nslots' estimation; 25% larger than nitems, but 128 at least */
#define __KDS_NSLOTS(nitems)					\
	Max(128, ((nitems) * 5) >> 2)
/*
 * NOTE: For strict correctness, header portion of kern_data_store may
 * have larger number of colmeta[] items than 'ncols', if array or composite
 * types are in the field definition.
 * However, it is relatively rare, and 'ncols' == 'nr_colmeta' in most cases.
 * The macros below are used for just cost estimation; no need to be strict
 * connect for size estimatino.
 */
#define KDS_ESTIMATE_HEAD_LENGTH(ncols)					\
	STROMALIGN(offsetof(kern_data_store, colmeta[(ncols)]))
#define KDS_ESTIMATE_ROW_LENGTH(ncols,nitems,htup_sz)					\
	(KDS_ESTIMATE_HEAD_LENGTH(ncols) +									\
	 STROMALIGN(sizeof(cl_uint) * (nitems)) +							\
	 STROMALIGN(MAXALIGN(offsetof(kern_tupitem,							\
								  htup) + htup_sz) * (nitems)))
#define KDS_ESTIMATE_HASH_LENGTH(ncols,nitems,htup_sz)					\
	(KDS_ESTIMATE_HEAD_LENGTH(ncols) +									\
	 STROMALIGN(sizeof(cl_uint) * (nitems)) +							\
	 STROMALIGN(sizeof(cl_uint) * __KDS_NSLOTS(nitems)) +				\
	 STROMALIGN(MAXALIGN(offsetof(kern_hashitem,						\
								  t.htup) + htup_sz) * (nitems)))

/* Length of the header postion of kern_data_store */
STATIC_INLINE(size_t)
KERN_DATA_STORE_HEAD_LENGTH(kern_data_store *kds)
{
	size_t	headsz = STROMALIGN(offsetof(kern_data_store,
										 colmeta[kds->nr_colmeta]));
	if (kds->has_attnames)
		headsz += STROMALIGN(sizeof(NameData) * kds->nr_colmeta);
	return headsz;
}
/* attname array, if any */
STATIC_INLINE(NameData *)
KERN_DATA_STORE_ATTNAMES(kern_data_store *kds)
{
	size_t	offset;

	if (!kds->has_attnames)
		return NULL;
	offset = STROMALIGN(offsetof(kern_data_store,
								 colmeta[kds->nr_colmeta]));
	return (NameData *)((char *)kds + offset);
}
/* Base address of the data body */
STATIC_INLINE(char *)
KERN_DATA_STORE_BODY(kern_data_store *kds)
{
	return (char *)kds + KERN_DATA_STORE_HEAD_LENGTH(kds);
}

/* access function for row- and hash-format */
STATIC_INLINE(cl_uint *)
KERN_DATA_STORE_ROWINDEX(kern_data_store *kds)
{
	Assert(kds->format == KDS_FORMAT_ROW ||
		   kds->format == KDS_FORMAT_HASH);
	return (cl_uint *)KERN_DATA_STORE_BODY(kds);
}

/* access function for hash-format */
STATIC_INLINE(cl_uint *)
KERN_DATA_STORE_HASHSLOT(kern_data_store *kds)
{
	Assert(kds->format == KDS_FORMAT_HASH);
	return (cl_uint *)(KERN_DATA_STORE_BODY(kds) +
					   STROMALIGN(sizeof(cl_uint) * kds->nitems));
}

/* access function for row- and hash-format */
STATIC_INLINE(kern_tupitem *)
KERN_DATA_STORE_TUPITEM(kern_data_store *kds, cl_uint kds_index)
{
	size_t	offset = KERN_DATA_STORE_ROWINDEX(kds)[kds_index];

	if (!offset)
		return NULL;
	return (kern_tupitem *)((char *)kds + __kds_unpack(offset));
}

/* access macro for row-format by tup-offset */
STATIC_INLINE(HeapTupleHeaderData *)
KDS_ROW_REF_HTUP(kern_data_store *kds,
				 cl_uint tup_offset,
				 ItemPointerData *p_self,
				 cl_uint *p_len)
{
	kern_tupitem   *tupitem;

	Assert(kds->format == KDS_FORMAT_ROW ||
		   kds->format == KDS_FORMAT_HASH);
	if (tup_offset == 0)
		return NULL;
	tupitem = (kern_tupitem *)((char *)(kds)
							   + __kds_unpack(tup_offset)
							   - offsetof(kern_tupitem, htup));
	if (p_self)
		*p_self = tupitem->t_self;
	if (p_len)
		*p_len = tupitem->t_len;
	return &tupitem->htup;
}

STATIC_INLINE(kern_hashitem *)
KERN_HASH_FIRST_ITEM(kern_data_store *kds, cl_uint hash)
{
	cl_uint	   *slot = KERN_DATA_STORE_HASHSLOT(kds);
	size_t		offset = __kds_unpack(slot[hash % kds->nslots]);

	if (offset == 0)
		return NULL;
	Assert(offset < kds->length);
	return (kern_hashitem *)((char *)kds + offset);
}

STATIC_INLINE(kern_hashitem *)
KERN_HASH_NEXT_ITEM(kern_data_store *kds, kern_hashitem *khitem)
{
	size_t		offset;

	if (!khitem || khitem->next == 0)
		return NULL;
	offset = __kds_unpack(khitem->next);
	Assert(offset < kds->length);
	return (kern_hashitem *)((char *)kds + offset);
}

/* access macro for tuple-slot format */
STATIC_INLINE(size_t)
KERN_DATA_STORE_SLOT_LENGTH(kern_data_store *kds, cl_uint nitems)
{
	size_t	headsz = KERN_DATA_STORE_HEAD_LENGTH(kds);
	size_t	unitsz = LONGALIGN((sizeof(Datum) + sizeof(char)) * kds->ncols);

	return headsz + unitsz * nitems;
}

STATIC_INLINE(Datum *)
KERN_DATA_STORE_VALUES(kern_data_store *kds, cl_uint row_index)
{
	size_t	offset = KERN_DATA_STORE_SLOT_LENGTH(kds, row_index);

	return (Datum *)((char *)kds + offset);
}

STATIC_INLINE(cl_char *)
KERN_DATA_STORE_DCLASS(kern_data_store *kds, cl_uint row_index)
{
	Datum  *values = KERN_DATA_STORE_VALUES(kds, row_index);

	return (cl_char *)(values + kds->ncols);
}

/* access macro for block format */
#define KERN_DATA_STORE_PARTSZ(kds)				\
	Min(((kds)->nrows_per_block +				\
		 warpSize - 1) & ~(warpSize - 1),		\
		get_local_size())
#define KERN_DATA_STORE_BLOCK_BLCKNR(kds,kds_index)			\
	(((BlockNumber *)KERN_DATA_STORE_BODY(kds))[kds_index])
#define KERN_DATA_STORE_BLOCK_PGPAGE(kds,kds_index)			\
	((struct PageHeaderData *)								\
	 (KERN_DATA_STORE_BODY(kds) +							\
	  STROMALIGN(sizeof(BlockNumber) * (kds)->nrooms) +		\
	  BLCKSZ * kds_index))

/* access functions for apache arrow format */
STATIC_FUNCTION(void *)
kern_get_simple_datum_arrow(kern_data_store *kds,
							kern_colmeta *cmeta,
							cl_uint index,
							cl_uint unitsz)
{
	cl_char	   *nullmap = NULL;
	cl_char	   *values;

	Assert(index < kds->nitems);
	if (cmeta->nullmap_offset)
	{
		nullmap = (char *)kds + __kds_unpack(cmeta->nullmap_offset);
		if (att_isnull(index, nullmap))
			return NULL;
	}
	Assert(cmeta->values_offset > 0 &&
		   cmeta->extra_offset == 0 &&
		   cmeta->extra_length == 0);
	values = (char *)kds + __kds_unpack(cmeta->values_offset);
	return values + unitsz * index;
}

STATIC_FUNCTION(void *)
kern_get_varlena_datum_arrow(kern_data_store *kds,
							 kern_colmeta *cmeta,
							 cl_uint index,
							 cl_uint *p_length)
{
	cl_char	   *nullmap;
	cl_uint	   *offset;
	cl_char	   *extra;

	Assert(index < kds->nitems);
	if (cmeta->nullmap_offset)
	{
		nullmap = (char *)kds + __kds_unpack(cmeta->nullmap_offset);
		if (att_isnull(index, nullmap))
			return NULL;
	}
	Assert(cmeta->values_offset > 0 &&
		   cmeta->extra_offset > 0);
	offset = (cl_uint *)((char *)kds + __kds_unpack(cmeta->values_offset));
	extra = (char *)kds + __kds_unpack(cmeta->extra_offset);

	Assert(offset[index] <= offset[index+1] &&
		   offset[index+1] <= __kds_unpack(cmeta->extra_length));
	*p_length = offset[index+1] - offset[index];
	return (extra + offset[index]);
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
 * GstoreIpcHandle
 *
 * Format definition when Gstore_fdw exports IPChandle of the GPU memory.
 */

/* Gstore_fdw's internal data format */
#define GSTORE_FDW_FORMAT__PGSTROM		50		/* KDS_FORMAT_COLUMN */
//#define GSTORE_FDW_FORMAT__ARROW		51		/* Apache Arrow compatible */

/* column 'compression' option */
#define GSTORE_COMPRESSION__NONE		0
#define GSTORE_COMPRESSION__PGLZ		1

typedef struct
{
	cl_uint		__vl_len;		/* 4B varlena header */
	cl_short	device_id;		/* GPU device where pinning on */
	cl_char		format;			/* one of GSTORE_FDW_FORMAT__* */
	cl_char		__padding__;	/* reserved */
	cl_long		rawsize;		/* length in bytes */
	union {
#ifdef CU_IPC_HANDLE_SIZE
		CUipcMemHandle		d;	/* CUDA driver API */
#endif
#ifdef CUDA_IPC_HANDLE_SIZE
		cudaIpcMemHandle_t	r;	/* CUDA runtime API */
#endif
		char				data[64];
	} ipc_mhandle;
} GstoreIpcHandle;

/* ----------------------------------------------------------------
 * PostgreSQL Data Type support in CUDA kernel
 *
 * Device code will have the following representation for data types of
 * PostgreSQL, once it gets loaded from any type of data store above.
 *
 * typedef struct
 * {
 *     bool    isnull;
 *     BASE    value;
 * } pg_XXXX_t
 *
 * PostgreSQL has four different classes:
 *  - fixed-length referenced by value (simple)
 *  - fixed-length referenced by pointer (indirect)
 *  - variable-length (varlena)
 *
 * The 'simple' and 'indirect' types are always inlined to pg_XXX_t,
 * and often transformed to internal representation.
 * Some of 'varlena' types are also inlined to pg_XXXX_t variable,
 * mostly, if this varlena type has upper limit length short enough.
 */

/* ----------------------------------------------------------------
 *
 * About GPU Projection Support
 *
 * A typical projection code path is below:
 *
 * 1. Extract values from heap-tuple or column-store onto tup_dclass[] and
 *    tup_values[] array, and calculate length of the new heap-tuple.
 * 2. Allocation of the destination buffer, per threads-group
 * 3. Write out the heap-tuple
 *
 * Step-1 is usually handled by auto-generated code. In some case, it is not
 * reasonable to extract values to in-storage format prior to allocation of
 * the destination buffer, like a long text value that references a source
 * buffer in Apache Arrow.
 * Right now, we pay attention on simple varlena (Binary of Arrow that is
 * bytes in PG, and Utf8 of Arrow that is text in PG), and array of fixed-
 * length values (List of Arrow).
 * If tup_values[] hold a pointer to pg_varlena_t or pg_array_t, not raw-
 * varlena image, tup_dclass[] will have special flag to inform indirect
 * reference to the value.
 *
 * pg_XXXX_datum_ref() routine of types are responsible to transform disk
 * format to internal representation.
 * pg_XXXX_datum_store() routine of types are also responsible to transform
 * internal representation to disk format. We need to pay attention on
 * projection stage. If and when GPU code tries to store expressions which
 * are not simple Var, Const or Param, these internal representation must
 * be written to extra-buffer first.
 *
 * Also note that KDS_FORMAT_SLOT is designed to have compatible layout to
 * pair of tup_dclass[] / tup_values[] array if all the items have NULL or
 * NORMAL state. Other state should be normalized prior to CPU writeback.
 *
 * ----------------------------------------------------------------
 */
#define DATUM_CLASS__NORMAL		0	/* datum is normal value */
#define DATUM_CLASS__NULL		1	/* datum is NULL */
#define DATUM_CLASS__VARLENA	2	/* datum is pg_varlena_t reference */
#define DATUM_CLASS__ARRAY		3	/* datum is pg_array_t reference */

/*
 * Template of variable classes: fixed-length referenced by value
 * ---------------------------------------------------------------
 */
#define STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)					\
	typedef struct {												\
		BASE		value;											\
		cl_bool		isnull;											\
	} pg_##NAME##_t;

#ifdef __CUDACC__
#define STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE,AS_DATUM)			\
	STATIC_INLINE(pg_##NAME##_t)									\
	pg_##NAME##_datum_ref(kern_context *kcxt, void *addr)			\
	{																\
		pg_##NAME##_t	result;										\
																	\
		if (!addr)													\
			result.isnull = true;									\
		else														\
		{															\
			result.isnull = false;									\
			result.value = *((BASE *) addr);						\
		}															\
		return result;												\
	}																\
	STATIC_INLINE(void)												\
	pg_datum_ref(kern_context *kcxt,								\
				 pg_##NAME##_t &result, void *addr)					\
	{																\
		result = pg_##NAME##_datum_ref(kcxt, addr);					\
	}																\
	STATIC_INLINE(pg_##NAME##_t)									\
	pg_##NAME##_arrow_ref(kern_context *kcxt,						\
						  kern_data_store *kds,						\
						  kern_colmeta *cmeta, cl_int index);		\
	STATIC_INLINE(void)												\
	pg_datum_ref_slot(kern_context *kcxt,							\
					  pg_##NAME##_t &result,						\
					  cl_char dclass, Datum datum)					\
	{																\
		if (dclass == DATUM_CLASS__NULL)							\
			result = pg_##NAME##_datum_ref(kcxt, NULL);				\
		else														\
		{															\
			assert(dclass == DATUM_CLASS__NORMAL);					\
			result = pg_##NAME##_datum_ref(kcxt, &datum);			\
		}															\
	}																\
	STATIC_INLINE(cl_int)											\
	pg_datum_store(kern_context *kcxt,								\
				   pg_##NAME##_t datum,								\
				   cl_char &dclass,									\
				   Datum &value)									\
	{																\
		if (!datum.isnull)											\
		{															\
			dclass = DATUM_CLASS__NORMAL;							\
			value = AS_DATUM(datum.value);							\
			return sizeof(BASE);									\
		}															\
		dclass = DATUM_CLASS__NULL;									\
		return 0;													\
	}
#else	/* __CUDACC__ */
#define	STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE,AS_DATUM)
#endif	/* __CUDACC__ */

/*
 * Template of variable classes: fixed-length referenced by pointer
 * ----------------------------------------------------------------
 */
#ifdef __CUDACC__
#define STROMCL_INDIRECT_VARREF_TEMPLATE(NAME,BASE)					\
	STATIC_INLINE(pg_##NAME##_t)									\
	pg_##NAME##_datum_ref(kern_context *kcxt, void *addr)			\
	{																\
		pg_##NAME##_t	result;										\
																	\
		if (!addr)													\
			result.isnull = true;									\
		else														\
		{															\
			result.isnull = false;									\
			memcpy(&result.value, (BASE *)addr, sizeof(BASE));		\
		}															\
		return result;												\
	}																\
	STATIC_INLINE(void)												\
	pg_datum_ref(kern_context *kcxt,								\
				 pg_##NAME##_t &result, void *addr)					\
	{																\
		result = pg_##NAME##_datum_ref(kcxt, addr);					\
	}																\
	STATIC_INLINE(void)												\
	pg_datum_ref_slot(kern_context *kcxt,							\
					  pg_##NAME##_t &result,						\
					  cl_char dclass, Datum datum)					\
	{																\
		if (dclass == DATUM_CLASS__NULL)							\
			result = pg_##NAME##_datum_ref(kcxt, NULL);				\
		else														\
		{															\
			assert(dclass == DATUM_CLASS__NORMAL);					\
			result = pg_##NAME##_datum_ref(kcxt, (char *)datum);	\
		}															\
	}																\
	STATIC_INLINE(cl_int)											\
	pg_datum_store(kern_context *kcxt,								\
				   pg_##NAME##_t datum,								\
				   cl_char &dclass,									\
				   Datum &value)									\
	{																\
		void	   *res;											\
																	\
		if (datum.isnull)											\
		{															\
			dclass = DATUM_CLASS__NULL;								\
			return 0;												\
		}															\
		res = kern_context_alloc(kcxt, sizeof(BASE));				\
		if (!res)													\
		{															\
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);		\
			dclass = DATUM_CLASS__NULL;								\
			return 0;												\
		}															\
		memcpy(res, &datum.value, sizeof(BASE));					\
		dclass = DATUM_CLASS__NORMAL;								\
		value = PointerGetDatum(res);								\
		return sizeof(BASE);										\
	}
#else	/* __CUDACC__ */
#define	STROMCL_INDIRECT_VARREF_TEMPLATE(NAME,BASE)
#endif	/* __CUDACC__ */

/*
 * General purpose hash-function which is compatible to PG's hash_any()
 * These are basis of Hash-Join and Group-By reduction.
 */
#ifdef __CUDACC__
DEVICE_ONLY_FUNCTION(cl_uint)
pg_hash_any(const cl_uchar *k, cl_int keylen);

#define STROMCL_SIMPLE_COMP_HASH_TEMPLATE(NAME,BASE)			\
	STATIC_INLINE(cl_uint)										\
	pg_comp_hash(kern_context *kcxt, pg_##NAME##_t datum)		\
	{															\
		if (datum.isnull)										\
			return 0;											\
		return pg_hash_any((cl_uchar *)&datum.value,			\
						   sizeof(BASE));						\
	}
#else	/* __CUDACC__ */
#define	STROMCL_SIMPLE_COMP_HASH_TEMPLATE(NAME,BASE)
#endif	/* __CUDACC__ */

/*
 * References to Const/Param values
 */
#ifdef __CUDACC__
#define STROMCL_SIMPLE_PARAM_TEMPLATE(NAME)					\
	STATIC_INLINE(pg_##NAME##_t)							\
	pg_##NAME##_param(kern_context *kcxt,cl_uint param_id)	\
	{														\
		kern_parambuf *kparams = kcxt->kparams;				\
		pg_##NAME##_t result;								\
															\
		if (param_id < kparams->nparams &&					\
			kparams->poffset[param_id] > 0)					\
		{													\
			void   *addr = ((char *)kparams +				\
							kparams->poffset[param_id]);	\
			result = pg_##NAME##_datum_ref(kcxt, addr);		\
		}													\
		else												\
			result.isnull = true;							\
		return result;										\
	}
#else
#define STROMCL_SIMPLE_PARAM_TEMPLATE(NAME)
#endif

/*
 * Reference to Arrow values without any transformation
 */
#ifdef __CUDACC__
#define STROMCL_SIMPLE_ARROW_TEMPLATE(NAME,BASE)			\
	STATIC_INLINE(void)										\
	pg_datum_ref_arrow(kern_context *kcxt,					\
					   pg_##NAME##_t &result,				\
					   kern_data_store *kds,				\
					   cl_uint colidx, cl_uint rowidx)		\
	{														\
		kern_colmeta   *cmeta = &kds->colmeta[colidx];		\
		void		   *addr;								\
															\
		assert(kds->format == KDS_FORMAT_ARROW);			\
		assert(colidx < kds->nr_colmeta &&					\
			   rowidx < kds->nitems);						\
		addr = kern_get_simple_datum_arrow(kds,cmeta,		\
										   rowidx,			\
										   sizeof(BASE));	\
		if (!addr)											\
			result.isnull = true;							\
		else												\
		{													\
			result.isnull = false;							\
			result.value  = *((BASE *)addr);				\
		}													\
	}
#define STROMCL_NOSUPPORT_ARROW_TEMPLATE(NAME)				\
	STATIC_INLINE(void)                                     \
    pg_datum_ref_arrow(kern_context *kcxt,                  \
					   pg_##NAME##_t &result,               \
					   kern_data_store *kds,                \
					   cl_uint colidx, cl_uint rowidx)      \
	{														\
		result.isnull = true;								\
		STROM_SET_ERROR(&kcxt->e,StromError_WrongCodeGeneration);	\
	}

#else
#define STROMCL_SIMPLE_ARROW_TEMPLATE(NAME,BASE)
#define STROMCL_NOSUPPORT_ARROW_TEMPLATE(NAME,BASE)
#endif

#define STROMCL_SIMPLE_TYPE_TEMPLATE(NAME,BASE,AS_DATUM)	\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)				\
	STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE,AS_DATUM)		\
	STROMCL_SIMPLE_COMP_HASH_TEMPLATE(NAME,BASE)			\
	STROMCL_SIMPLE_PARAM_TEMPLATE(NAME)

#define STROMCL_INDIRECT_TYPE_TEMPLATE(NAME,BASE)	\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_INDIRECT_VARREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_COMP_HASH_TEMPLATE(NAME,BASE)	\
	STROMCL_SIMPLE_PARAM_TEMPLATE(NAME)

#define __STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,OPER,EXTRA) \
	STATIC_INLINE(pg_bool_t)								\
	pgfn_##FNAME##EXTRA(kern_context *kcxt,					\
						pg_##LNAME##_t arg1,				\
						pg_##RNAME##_t arg2)				\
	{														\
		pg_bool_t result;									\
															\
		result.isnull = arg1.isnull | arg2.isnull;			\
		if (!result.isnull)									\
			result.value = ((CAST)arg1.value OPER			\
							(CAST)arg2.value);				\
		return result;										\
	}
#define STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST)		\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,==,eq)	\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,!=,ne)	\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,<, lt)	\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,<=,le)	\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,>, gt)	\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,>=,ge)

/* pg_bool_t */
#ifndef PG_BOOL_TYPE_DEFINED
#define PG_BOOL_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(bool, cl_bool, )
#ifdef __CUDACC__
STATIC_INLINE(void)
pg_datum_ref_arrow(kern_context *kcxt,
				   pg_bool_t &result,
				   kern_data_store *kds,
				   cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta   *cmeta = &kds->colmeta[colidx];
	cl_uchar	   *bitmap;
	cl_uchar		mask = (1 << (rowidx & 7));

	assert(kds->format == KDS_FORMAT_ARROW);
	assert(colidx < kds->nr_colmeta && rowidx < kds->nitems);
	bitmap = (cl_uchar *)
		kern_get_simple_datum_arrow(kds,cmeta,rowidx>>3,
									sizeof(cl_uchar));
	if (!bitmap)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value  = ((*bitmap & mask) ? true : false);
	}
}
#endif	/* __CUDACC__ */
#endif	/* PG_BOOL_TYPE_DEFINED */

/* pg_int2_t */
#ifndef PG_INT2_TYPE_DEFINED
#define PG_INT2_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int2, cl_short, )
STROMCL_SIMPLE_ARROW_TEMPLATE(int2, cl_short)
#endif	/* PG_INT2_TYPE_DEFINED */

/* pg_int4_t */
#ifndef PG_INT4_TYPE_DEFINED
#define PG_INT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int4, cl_int, )
STROMCL_SIMPLE_ARROW_TEMPLATE(int4, cl_int)
#endif	/* PG_INT4_TYPE_DEFINED */

/* pg_int8_t */
#ifndef PG_INT8_TYPE_DEFINED
#define PG_INT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int8, cl_long, )
STROMCL_SIMPLE_ARROW_TEMPLATE(int8, cl_long)
#endif	/* PG_INT8_TYPE_DEFINED */

/* pg_float2_t */
#ifndef PG_FLOAT2_TYPE_DEFINED
#define PG_FLOAT2_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float2, cl_half, __half_as_short)
STROMCL_SIMPLE_ARROW_TEMPLATE(float2, cl_half)
#endif	/* PG_FLOAT2_TYPE_DEFINED */

/* pg_float4_t */
#ifndef PG_FLOAT4_TYPE_DEFINED
#define PG_FLOAT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float4, cl_float, __float_as_int)
STROMCL_SIMPLE_ARROW_TEMPLATE(float4, cl_float)
#endif	/* PG_FLOAT4_TYPE_DEFINED */

/* pg_float8_t */
#ifndef PG_FLOAT8_TYPE_DEFINED
#define PG_FLOAT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float8, cl_double, __double_as_longlong)
STROMCL_SIMPLE_ARROW_TEMPLATE(float8, cl_double)
#endif	/* PG_FLOAT8_TYPE_DEFINED */

/* definitions for variable-length data types */
#include "cuda_varlena.h"

#ifdef	__CUDACC__
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
 *
 * EXTRACT_HEAP_READ_XXXX()
 *  -> load raw values to dclass[]/values[], and update extras[]
 */
#define EXTRACT_HEAP_TUPLE_BEGIN(ADDR, kds, htup)						\
	do {																\
		const HeapTupleHeaderData * __restrict__ __htup = (htup);		\
		const kern_colmeta * __restrict__ __kds_colmeta = (kds)->colmeta; \
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
			__ncols = Min((kds)->ncols,									\
						  __htup->t_infomask2 & HEAP_NATTS_MASK);		\
			__cmeta = __kds_colmeta[__colidx];							\
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
			__cmeta = __kds_colmeta[__colidx];							\
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
#endif	/* __CUDACC__ */

#define EXTRACT_HEAP_READ_8BIT(ADDR,ATT_DCLASS,ATT_VALUES)	 \
	do {													 \
		if (!(ADDR))										 \
			(ATT_DCLASS) = DATUM_CLASS__NULL;				 \
		else												 \
		{													 \
			(ATT_DCLASS) = DATUM_CLASS__NORMAL;				 \
			(ATT_VALUES) = *((cl_uchar *)(ADDR));			 \
		}												 	 \
	} while(0)

#define EXTRACT_HEAP_READ_16BIT(ADDR,ATT_DCLASS,ATT_VALUES)	 \
	do {													 \
		if (!(ADDR))										 \
			(ATT_DCLASS) = DATUM_CLASS__NULL;				 \
		else												 \
		{													 \
			(ATT_DCLASS) = DATUM_CLASS__NORMAL;				 \
			(ATT_VALUES) = *((cl_ushort *)(ADDR));			 \
		}												 	 \
	} while(0)

#define EXTRACT_HEAP_READ_32BIT(ADDR,ATT_DCLASS,ATT_VALUES)	 \
	do {													 \
		if (!(ADDR))										 \
			(ATT_DCLASS) = DATUM_CLASS__NULL;				 \
		else												 \
		{													 \
			(ATT_DCLASS) = DATUM_CLASS__NORMAL;				 \
			(ATT_VALUES) = *((cl_uint *)(ADDR));			 \
		}												 	 \
	} while(0)

#define EXTRACT_HEAP_READ_64BIT(ADDR,ATT_DCLASS,ATT_VALUES)	 \
	do {													 \
		if (!(ADDR))										 \
			(ATT_DCLASS) = DATUM_CLASS__NULL;				 \
		else												 \
		{													 \
			(ATT_DCLASS) = DATUM_CLASS__NORMAL;				 \
			(ATT_VALUES) = *((cl_ulong *)(ADDR));			 \
		}												 	 \
	} while(0)

#define EXTRACT_HEAP_READ_POINTER(ADDR,ATT_DCLASS,ATT_VALUES)	\
	do {											\
		if (!(ADDR))											\
			(ATT_DCLASS) = DATUM_CLASS__NULL;					\
		else													\
		{														\
			(ATT_DCLASS) = DATUM_CLASS__NORMAL;					\
			(ATT_VALUES) = PointerGetDatum(ADDR);				\
		}														\
	} while(0)

/*
 * kern_get_datum_xxx
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
			if (cmeta.attlen > 0)
				offset += cmeta.attlen;
			else
				offset += VARSIZE_ANY(addr);
		}
	}
	return NULL;
}

//who use this API?
STATIC_FUNCTION(void *)
kern_get_datum_row(kern_data_store *kds,
				   cl_uint colidx, cl_uint rowidx)
{
	kern_tupitem   *tupitem;

	if (colidx >= kds->ncols ||
		rowidx >= kds->nitems)
		return NULL;	/* likely a BUG */
	tupitem = KERN_DATA_STORE_TUPITEM(kds, rowidx);

	return kern_get_datum_tuple(kds->colmeta, &tupitem->htup, colidx);
}

#if 0
//not used any more. Use pg_datum_ref_slot() instead
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
#endif

STATIC_FUNCTION(void *)
kern_get_datum_column(kern_data_store *kds,
					  cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta *cmeta;
	size_t		offset;
	size_t		length;
	char	   *values;
	char	   *nullmap;

	Assert(colidx < kds->ncols);
	cmeta = &kds->colmeta[colidx];
	/* special case handling if 'tableoid' system column */
	if (cmeta->attnum == TableOidAttributeNumber)
		return &kds->table_oid;
	offset = __kds_unpack(cmeta->va_offset);
	if (offset == 0)
		return NULL;
	values = (char *)kds + offset;
	length = __kds_unpack(cmeta->va_length);
	if (cmeta->attlen < 0)
	{
		Assert(!cmeta->attbyval);
		offset = ((cl_uint *)values)[rowidx];
		if (offset == 0)
			return NULL;
		Assert(offset < length);
		values += __kds_unpack(offset);
	}
	else
	{
		cl_int	unitsz = TYPEALIGN(cmeta->attalign,
								   cmeta->attlen);
		size_t	array_sz = MAXALIGN(unitsz * kds->nitems);

		Assert(length >= array_sz);
		if (length > array_sz)
		{
			length -= array_sz;
			Assert(MAXALIGN(BITMAPLEN(kds->nitems)) == length);
			nullmap = values + array_sz;
			if (att_isnull(rowidx, nullmap))
				return NULL;
		}
		values += unitsz * rowidx;
	}
	return (void *)values;
}

/*
 * Utility functions to reference system columns
 *   (except for ctid and table_oid)
 */
#ifdef __CUDACC__
STATIC_INLINE(cl_int)
pg_sysattr_ctid_store(kern_context *kcxt,
					  kern_data_store *kds,
					  HeapTupleHeaderData *htup,
					  ItemPointerData *t_self,
					  cl_char &dclass,
					  Datum   &value)
{
	void	   *temp;

	if (!t_self)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		if (kds && ((char *)t_self >= (char *)kds &&
					(char *)t_self <  (char *)kds + kds->length))
		{
			value = PointerGetDatum(t_self);
		}
		else
		{
			temp = kern_context_alloc(kcxt, sizeof(ItemPointerData));
			if (temp)
			{
				memcpy(temp, t_self, sizeof(ItemPointerData));
				value = PointerGetDatum(temp);
				return sizeof(ItemPointerData);
			}
			dclass = DATUM_CLASS__NULL;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_oid_store(kern_context *kcxt,
					 kern_data_store *kds,
					 HeapTupleHeaderData *htup,
					 ItemPointerData *t_self,
					 cl_char &dclass,
					 Datum   &value)
{
	if (!htup)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		if ((htup->t_infomask & HEAP_HASOID) == 0)
			value = 0;
		else
			value = *((cl_uint *)((char *) htup
								  + htup->t_hoff
								  - sizeof(cl_uint)));
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_xmin_store(kern_context *kcxt,
					  kern_data_store *kds,
					  HeapTupleHeaderData *htup,
					  ItemPointerData *t_self,
					  cl_char &dclass,
					  Datum   &value)
{
	if (!htup)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = htup->t_choice.t_heap.t_xmin;
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_cmin_store(kern_context *kcxt,
					  kern_data_store *kds,
					  HeapTupleHeaderData *htup,
					  ItemPointerData *t_self,
					  cl_char &dclass,
					  Datum   &value)
{
	if (!htup)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = htup->t_choice.t_heap.t_field3.t_cid;
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_xmax_store(kern_context *kcxt,
					  kern_data_store *kds,
					  HeapTupleHeaderData *htup,
					  ItemPointerData *t_self,
					  cl_char &dclass,
					  Datum   &value)
{
	if (!htup)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = htup->t_choice.t_heap.t_xmax;
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_cmax_store(kern_context *kcxt,
					  kern_data_store *kds,
					  HeapTupleHeaderData *htup,
					  ItemPointerData *t_self,
					  cl_char &dclass,
					  Datum   &value)
{
	if (!htup)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = htup->t_choice.t_heap.t_field3.t_cid;
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_tableoid_store(kern_context *kcxt,
						  kern_data_store *kds,
						  HeapTupleHeaderData *htup,
						  ItemPointerData *t_self,
						  cl_char &dclass,
						  Datum   &value)
{
	if (!kds)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = kds->table_oid;
	}
	return 0;
}
#endif	/* __CUDACC__ */

/*
 * compute_heaptuple_size
 */
STATIC_FUNCTION(cl_uint)
compute_heaptuple_size(kern_context *kcxt,
					   kern_data_store *kds,
					   cl_char *tup_dclass,
					   Datum   *tup_values)
{
	cl_uint		t_hoff;
	cl_uint		datalen = 0;
	cl_uint		i, ncols = kds->ncols;
	cl_bool		heap_hasnull = false;

	/* compute data length */
	for (i=0; i < ncols; i++)
	{
		kern_colmeta   *cmeta = &kds->colmeta[i];
		cl_char			dclass = tup_dclass[i];

		if (dclass == DATUM_CLASS__NULL)
			heap_hasnull = true;
		else if (cmeta->attlen > 0)
		{
			assert(dclass == DATUM_CLASS__NORMAL);
			datalen = TYPEALIGN(cmeta->attalign, datalen);
			datalen += cmeta->attlen;
		}
		else
		{
			Datum		datum = tup_values[i];
			cl_uint		vl_len;

			switch (dclass)
			{
				case DATUM_CLASS__VARLENA:
					vl_len = pg_varlena_datum_length(kcxt,datum);
					datalen = TYPEALIGN(cmeta->attalign, datalen);
					break;
				case DATUM_CLASS__ARRAY:
					vl_len = pg_array_datum_length(kcxt,datum);
					datalen = TYPEALIGN(cmeta->attalign, datalen);
					break;
				default:
					assert(dclass == DATUM_CLASS__NORMAL);
					vl_len = VARSIZE_ANY(datum);
					if (!VARATT_IS_1B(datum))
						datalen = TYPEALIGN(cmeta->attalign, datalen);
					break;
			}
			datalen += vl_len;
		}
	}
	/* compute header offset */
	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (heap_hasnull)
		t_hoff += BITMAPLEN(ncols);
	if (kds->tdhasoid)
		t_hoff += sizeof(cl_uint);
	t_hoff = MAXALIGN(t_hoff);

	return t_hoff + datalen;
}

/*
 * deform_kern_heaptuple
 *
 * Like deform_heap_tuple in host side, it extracts the supplied tuple-item
 * into tup_values / tup_isnull array.
 *
 * NOTE: composite datum which is built-in other composite datum might not
 * be aligned to 4-bytes boundary. So, we don't touch htup fields directly,
 * except for 1-byte datum.
 */
STATIC_FUNCTION(void)
deform_kern_heaptuple(cl_int	nattrs,			/* in */
					  kern_colmeta *tup_attrs,	/* in */
					  HeapTupleHeaderData *htup,/* in */
					  cl_char  *tup_dclass,		/* out */
					  Datum	   *tup_values)		/* out */
{
	/* 'htup' must be aligned to 8bytes */
	assert(((cl_ulong)htup & (MAXIMUM_ALIGNOF-1)) == 0);
	if (!htup)
	{
		int		i;

		for (i=0; i < nattrs; i++)
			tup_dclass[i] = DATUM_CLASS__NULL;
	}
	else
	{
		cl_uint		offset = htup->t_hoff;
		cl_bool		tup_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
		cl_uint		i, ncols = (htup->t_infomask2 & HEAP_NATTS_MASK);

		ncols = Min(ncols, nattrs);
		for (i=0; i < ncols; i++)
		{
			if (tup_hasnull && att_isnull(i, htup->t_bits))
			{
				tup_dclass[i] = DATUM_CLASS__NULL;
				tup_values[i] = 0;
			}
			else
			{
				kern_colmeta   *cmeta = &tup_attrs[i];
				char		   *addr;

				if (cmeta->attlen > 0)
					offset = TYPEALIGN(cmeta->attalign, offset);
				else if (!VARATT_NOT_PAD_BYTE((char *)htup + offset))
					offset = TYPEALIGN(cmeta->attalign, offset);

				/* Store the value */
				addr = ((char *) htup + offset);
				if (cmeta->attbyval)
				{
					if (cmeta->attlen == sizeof(cl_char))
						tup_values[i] = *((cl_char *)addr);
					else if (cmeta->attlen == sizeof(cl_short))
						tup_values[i] = *((cl_short *)addr);
					else if (cmeta->attlen == sizeof(cl_int))
						tup_values[i] = *((cl_int *)addr);
					else if (cmeta->attlen == sizeof(cl_long))
						tup_values[i] = *((cl_long *)addr);
					else
					{
						assert(cmeta->attlen <= sizeof(Datum));
						memcpy(&tup_values[i], addr, cmeta->attlen);
					}
					offset += cmeta->attlen;
				}
				else
				{
					cl_uint		attlen = (cmeta->attlen > 0
										  ? cmeta->attlen
										  : VARSIZE_ANY(addr));
					tup_values[i] = PointerGetDatum(addr);
					offset += attlen;
				}
				tup_dclass[i] = DATUM_CLASS__NORMAL;
			}
		}
		/*
		 * Fill up remaining columns if source tuple has less columns than
		 * length of the array; that is definition of the destination
		 */
		while (i < nattrs)
			tup_dclass[i++] = DATUM_CLASS__NORMAL;
	}
}

/*
 * __form_kern_heaptuple
 */
STATIC_FUNCTION(cl_uint)
__form_kern_heaptuple(void	   *buffer,			/* out */
					  cl_int	ncols,			/* in */
					  kern_colmeta *colmeta,	/* in */
					  HeapTupleHeaderData *htup_orig, /* in: if heap-tuple */
					  cl_int	comp_typmod,	/* in: if composite type */
					  cl_uint	comp_typeid,	/* in: if composite type */
					  cl_uint	htuple_oid,		/* in */
					  cl_char  *tup_dclass,		/* in */
					  Datum	   *tup_values)		/* in */
{
	HeapTupleHeaderData *htup = (HeapTupleHeaderData *)buffer;
	cl_bool		tup_hasnull = false;
	cl_ushort	t_infomask;
	cl_uint		t_hoff;
	cl_uint		i, curr;

	/* alignment checks */
	assert((uintptr_t)htup == MAXALIGN(htup));

	/* has any NULL attributes? */
	if (tup_dclass != NULL)
	{
		for (i=0; i < ncols; i++)
		{
			if (tup_dclass[i] == DATUM_CLASS__NULL)
			{
				tup_hasnull = true;
				break;
			}
		}
	}
	t_infomask = (tup_hasnull ? HEAP_HASNULL : 0);

	/* preserve HeapTupleHeaderData, if any */
	if (htup_orig)
		memcpy(htup, htup_orig, offsetof(HeapTupleHeaderData,
										 t_ctid) + sizeof(ItemPointerData));
	else
	{
		/* datum_len_ shall be set on the tail  */
		htup->t_choice.t_datum.datum_typmod = comp_typmod;
		htup->t_choice.t_datum.datum_typeid = comp_typeid;
		htup->t_ctid.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
		htup->t_ctid.ip_blkid.bi_lo = 0xffff;
		htup->t_ctid.ip_posid = 0;				/* InvalidOffsetNumber */
	}
	htup->t_infomask2 = (ncols & HEAP_NATTS_MASK);

	/* computer header size */
	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (tup_hasnull)
		t_hoff += BITMAPLEN(ncols);
	if (htuple_oid != 0)
	{
		t_infomask |= HEAP_HASOID;
		t_hoff += sizeof(cl_uint);
	}
	t_hoff = MAXALIGN(t_hoff);
	if (htuple_oid != 0)
		*((cl_uint *)((char *)htup + t_hoff - sizeof(cl_uint))) = htuple_oid;

	/* walk on the regular columns */
	htup->t_hoff = t_hoff;
	curr = t_hoff;

	for (i=0; i < ncols; i++)
	{
		kern_colmeta *cmeta = &colmeta[i];
		Datum		datum = tup_values[i];
		cl_char		dclass;
		cl_int		padding;

		dclass = (!tup_dclass ? DATUM_CLASS__NORMAL : tup_dclass[i]);
		if (dclass == DATUM_CLASS__NULL)
		{
			assert(tup_hasnull);
			htup->t_bits[i >> 3] &= ~(1 << (i & 0x07));
			continue;
		}

		if (tup_hasnull)
			htup->t_bits[i >> 3] |= (1 << (i & 0x07));

		padding = TYPEALIGN(cmeta->attalign, curr) - curr;
		if (cmeta->attbyval)
		{
			assert(dclass == DATUM_CLASS__NORMAL);
			while (padding-- > 0)
				((char *)htup)[curr++] = '\0';
			assert(cmeta->attlen <= sizeof(datum));
			memcpy((char *)htup + curr, &datum, cmeta->attlen);
			curr += cmeta->attlen;
		}
		else if (cmeta->attlen > 0)
		{
			assert(dclass == DATUM_CLASS__NORMAL);
			while (padding-- > 0)
				((char *)htup)[curr++] = '\0';
			memcpy((char *)htup + curr,
				   DatumGetPointer(datum), cmeta->attlen);
			curr += cmeta->attlen;
		}
		else
		{
			cl_int		vl_len;

			switch (dclass)
			{
				case DATUM_CLASS__VARLENA:
					while (padding-- > 0)
						((char *)htup)[curr++] = '\0';
					vl_len = pg_varlena_datum_write((char *)htup+curr, datum);
					break;
				case DATUM_CLASS__ARRAY:
					while (padding-- > 0)
						((char *)htup)[curr++] = '\0';
					vl_len = pg_array_datum_write((char *)htup+curr, datum);
					break;
				default:
					assert(dclass == DATUM_CLASS__NORMAL);
					vl_len = VARSIZE_ANY(datum);
					if (!VARATT_IS_1B(datum))
					{
						while (padding-- > 0)
							((char *)htup)[curr++] = '\0';
					}
					memcpy((char *)htup + curr,
						   DatumGetPointer(datum), vl_len);
					break;
			}
			t_infomask |= HEAP_HASVARWIDTH;
			curr += vl_len;
		}
	}
	htup->t_infomask = t_infomask;
	if (!htup_orig)
		SET_VARSIZE(&htup->t_choice.t_datum, curr);
	return curr;
}

/*
 * form_kern_heaptuple
 *
 * A utility routine to build a kern_tupitem on the destination buffer
 * already allocated.
 *
 * tupitem      ... kern_tupitem allocated on the kds
 * kds_dst      ... destination data store
 * tup_self     ... item pointer of the tuple, if any
 * htup         ... tuple-header of the original tuple, if any
 * tup_dclass   ... any of DATUM_CLASS__*
 * tup_values   ... array of values to be written
 */
STATIC_INLINE(cl_uint)
form_kern_heaptuple(kern_tupitem    *tupitem,		/* out */
					kern_data_store	*kds_dst,		/* in */
					ItemPointerData *tup_self,		/* in, optional */
					HeapTupleHeaderData *htup,		/* in, optional */
					cl_char         *tup_dclass,	/* in */
					Datum           *tup_values)	/* in */
{
	cl_uint		htuple_oid = 0;

	assert((uintptr_t)tupitem == MAXALIGN(tupitem));
	assert((char *)tupitem >= (char *)kds_dst &&
		   (char *)tupitem <  (char *)kds_dst + kds_dst->length);
	/* setup kern_tupitem */
	if (tup_self)
		tupitem->t_self = *tup_self;
	else
	{
		tupitem->t_self.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
		tupitem->t_self.ip_blkid.bi_lo = 0xffff;
		tupitem->t_self.ip_posid = 0;				/* InvalidOffsetNumber */
	}
	/* OID of tuple; deprecated at PG12 */
	if (kds_dst->tdhasoid &&
		htup && (htup->t_infomask & HEAP_HASOID) != 0)
	{
		htuple_oid = *((cl_uint *)((char *)htup
								   + htup->t_hoff
								   - sizeof(cl_uint)));
	}
	tupitem->t_len = __form_kern_heaptuple(&tupitem->htup,
										   kds_dst->ncols,
										   kds_dst->colmeta,
										   htup,
										   0,	/* not a composite type */
										   0,	/* not a composite type */
										   htuple_oid,
										   tup_dclass,
										   tup_values);
	return tupitem->t_len;
}

/*
 * form_kern_composite_type
 *
 * A utility routine to set up a composite type data structure
 * on the supplied pre-allocated region. It in
 *
 * @buffer     ... pointer to global memory where caller wants to construct
 *                 a composite datum. It must have enough length and also
 *                 must be aligned to DWORD.
 * @typeoid    ... type OID of the composite type 
 * @typemod    ... type modifier of the composite type
 * @nfields    ... number of sub-fields of the composite type
 * @colmeta    ... array of kern_colmeta for sub-field types
 * @tup_dclass ... any of DATUM_CLASS__*
 * @tup_values ... values of the sub-fields
 */
STATIC_INLINE(cl_uint)
form_kern_composite_type(void      *buffer,      /* out */
						 cl_uint    comp_typeid, /* in: type OID */
						 cl_int		comp_typmod, /* in: type modifier */
						 cl_int		nfields,     /* in: # of attributes */
						 kern_colmeta *colmeta,  /* in: sub-type attributes */
						 cl_char   *tup_dclass,  /* in: */
						 Datum	   *tup_values)	 /* in: */
{
	return __form_kern_heaptuple(buffer,
								 nfields,
								 colmeta,
								 NULL,
								 comp_typmod,
								 comp_typeid,
								 0,	/* composite type never have OID */
								 tup_dclass,
								 tup_values);
}

#ifdef __CUDACC__
/* ------------------------------------------------------------------
 *
 * Declaration of utility functions
 *
 * ------------------------------------------------------------------ */

/*
 * Device version of hash_any() in PG host code
 */
#define rot(x,k)		(((x)<<(k)) | ((x)>>(32-(k))))
#define mix(a,b,c)								\
	{											\
		a -= c;  a ^= rot(c, 4);  c += b;		\
		b -= a;  b ^= rot(a, 6);  a += c;		\
		c -= b;  c ^= rot(b, 8);  b += a;		\
		a -= c;  a ^= rot(c,16);  c += b;		\
		b -= a;  b ^= rot(a,19);  a += c;		\
		c -= b;  c ^= rot(b, 4);  b += a;		\
	}

#define final(a,b,c)							\
	{											\
		c ^= b; c -= rot(b,14);					\
		a ^= c; a -= rot(c,11);					\
		b ^= a; b -= rot(a,25);					\
		c ^= b; c -= rot(b,16);					\
		a ^= c; a -= rot(c, 4);					\
		b ^= a; b -= rot(a,14);					\
		c ^= b; c -= rot(b,24);					\
	}

DEVICE_ONLY_FUNCTION(cl_uint)
pg_hash_any(const cl_uchar *k, cl_int keylen)
{
	cl_uint		a, b, c;
	cl_uint		len;

	/* Set up the internal state */
	len = keylen;
	a = b = c = 0x9e3779b9 + len + 3923095;

	/* If the source pointer is word-aligned, we use word-wide fetches */
	if (((uintptr_t) k & (sizeof(cl_uint) - 1)) == 0)
	{
		/* Code path for aligned source data */
		const cl_uint	*ka = (const cl_uint *) k;

		/* handle most of the key */
		while (len >= 12)
		{
			a += ka[0];
			b += ka[1];
			c += ka[2];
			mix(a, b, c);
			ka += 3;
			len -= 12;
		}

		/* handle the last 11 bytes */
		k = (const unsigned char *) ka;
		switch (len)
		{
			case 11:
				c += ((cl_uint) k[10] << 24);
				/* fall through */
			case 10:
				c += ((cl_uint) k[9] << 16);
				/* fall through */
			case 9:
				c += ((cl_uint) k[8] << 8);
				/* the lowest byte of c is reserved for the length */
				/* fall through */
			case 8:
				b += ka[1];
				a += ka[0];
				break;
			case 7:
				b += ((cl_uint) k[6] << 16);
				/* fall through */
			case 6:
				b += ((cl_uint) k[5] << 8);
				/* fall through */
			case 5:
				b += k[4];
				/* fall through */
			case 4:
				a += ka[0];
				break;
			case 3:
				a += ((cl_uint) k[2] << 16);
				/* fall through */
			case 2:
				a += ((cl_uint) k[1] << 8);
				/* fall through */
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
	}
	else
	{
		/* Code path for non-aligned source data */

		/* handle most of the key */
		while (len >= 12)
		{
			a += k[0] + (((cl_uint) k[1] << 8) +
						 ((cl_uint) k[2] << 16) +
						 ((cl_uint) k[3] << 24));
			b += k[4] + (((cl_uint) k[5] << 8) +
						 ((cl_uint) k[6] << 16) +
						 ((cl_uint) k[7] << 24));
			c += k[8] + (((cl_uint) k[9] << 8) +
						 ((cl_uint) k[10] << 16) +
						 ((cl_uint) k[11] << 24));
			mix(a, b, c);
			k += 12;
			len -= 12;
		}

		/* handle the last 11 bytes */
		switch (len)            /* all the case statements fall through */
		{
			case 11:
				c += ((cl_uint) k[10] << 24);
			case 10:
				c += ((cl_uint) k[9] << 16);
			case 9:
				c += ((cl_uint) k[8] << 8);
				/* the lowest byte of c is reserved for the length */
			case 8:
				b += ((cl_uint) k[7] << 24);
			case 7:
				b += ((cl_uint) k[6] << 16);
			case 6:
				b += ((cl_uint) k[5] << 8);
			case 5:
				b += k[4];
			case 4:
				a += ((cl_uint) k[3] << 24);
			case 3:
				a += ((cl_uint) k[2] << 16);
			case 2:
				a += ((cl_uint) k[1] << 8);
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
	}
	final(a, b, c);

	return c;
}
#undef rot
#undef mix
#undef final

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
DEVICE_ONLY_FUNCTION(cl_uint)
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
 * pgstromStairlikeBinaryCount
 *
 * A special optimized version of pgstromStairlikeSum, for binary count.
 * It has smaller number of __syncthreads().
 */
DEVICE_ONLY_FUNCTION(cl_uint)
pgstromStairlikeBinaryCount(int predicate, cl_uint *total_count)
{
	cl_uint	   *items = SHARED_WORKMEM(cl_uint);
	cl_uint		nwarps = get_local_size() / warpSize;
	cl_uint		warp_id = get_local_id() / warpSize;
	cl_uint		w_bitmap;
	cl_uint		stair_count;
	cl_int		unit_sz;
	cl_int		i, j;

	w_bitmap = __ballot_sync(__activemask(), predicate);
	if ((get_local_id() & (warpSize-1)) == 0)
		items[warp_id] = __popc(w_bitmap);
	__syncthreads();

	for (i=1, unit_sz = nwarps; unit_sz > 0; i++, unit_sz >>= 1)
	{
		/* index of last item in the earlier half of each 2^i unit */
		j = (get_local_id() & ~((1<<i)-1)) | ((1<<(i-1))-1);

		/* add item[j] if it is later half in the 2^i unit */
		if (get_local_id() < nwarps &&
			(get_local_id() & (1 << (i-1))) != 0)
			items[get_local_id()] += items[j];
		__syncthreads();
	}
	if (total_count)
		*total_count = items[nwarps - 1];
	w_bitmap &= (1U << (get_local_id() & (warpSize-1))) - 1;
	stair_count = (warp_id == 0 ? 0 : items[warp_id - 1]) + __popc(w_bitmap);
	__syncthreads();

	return stair_count;
}

/*
 * pgstromTotalSum
 *
 * A utility routine to calculate total sum of the supplied array which are
 * consists of primitive types.
 * Unlike pgstromStairLikeSum, it accepts larger length of the array than
 * size of thread block, and unused threads shall be relaxed earlier.
 *
 * Restrictions:
 * - array must be a primitive types, like int, double.
 * - array must be on the shared memory.
 * - all the threads must call the function simultaneously.
 *   (Unacceptable to call the function in if-block)
 */
template <typename T>
DEVICE_ONLY_FUNCTION(T)
pgstromTotalSum(T *values, cl_uint nitems)
{
	cl_uint		nsteps = get_next_log2(nitems);
	cl_uint		nthreads;
	cl_uint		step;
	cl_uint		loop;
	T			retval;

	if (nitems == 0)
		return (T)(0);
	__syncthreads();
	for (step=1; step <= nsteps; step++)
	{
		nthreads = ((nitems - 1) >> step) + 1;

		for (loop=get_local_id(); loop < nthreads; loop += get_local_size())
		{
			cl_uint		dst = (loop << step);
			cl_uint		src = dst + (1U << (step - 1));

			if (src < nitems)
				values[dst] += values[src];
		}
		__syncthreads();
	}
	retval = values[0];
	__syncthreads();

	return retval;
}
#endif	/* __CUDACC__ */

#ifdef __CUDACC__
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

STATIC_INLINE(pg_bool_t)
NOT(pg_bool_t arg)
{
	arg.value = !arg.value;
	return arg;
}

STATIC_INLINE(pg_bool_t)
to_bool(cl_bool value)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value  = value;

	return result;
}

/*
 * Support routine for CASE x WHEN y then ... else ... end
 */
template <typename E, typename T>
STATIC_INLINE(T)
PG_CASEWHEN_ELSE(kern_context *kcxt,
				 const E& case_val,
				 const T& else_val)
{
	return else_val;
}

template <typename E, typename T, typename ...R>
STATIC_INLINE(T)
PG_CASEWHEN_ELSE(kern_context *kcxt,
				 const E& case_val,
				 const T& else_val,
				 const E& test_val,
				 const T& then_val,
				 const R&... args_rest)
{
	pg_int4_t	cmp;

	if (!case_val.isnull && !test_val.isnull)
	{
		cmp = pgfn_type_compare(kcxt, case_val, test_val);
		if (!cmp.isnull && cmp.value == 0)
			return then_val;
	}
	return PG_CASEWHEN_ELSE(kcxt, case_val, else_val, args_rest...);
}

template <typename E, typename T, typename ...R>
STATIC_INLINE(T)
PG_CASEWHEN_EXPR(kern_context *kcxt,
				 const E& case_val,
				 const E& test_val,
				 const T& then_val,
				 const R&... args_rest)
{
	pg_int4_t	cmp;
	E			else_val;

	if (!case_val.isnull && !test_val.isnull)
	{
		cmp = pgfn_type_compare(kcxt, case_val, test_val);
		if (!cmp.isnull && cmp.value == 0)
			return then_val;
	}
	else_val.isnull = true;
	return PG_CASEWHEN_ELSE(kcxt, case_val, else_val, args_rest...);
}

/*
 * Support routine for COALESCE / GREATEST / LEAST
 */
template <typename T>
STATIC_INLINE(T)
PG_COALESCE(kern_context *kcxt, const T& arg)
{
	return arg;
}

template <typename T, typename ...R>
STATIC_INLINE(T)
PG_COALESCE(kern_context *kcxt, const T& arg1, const R&... args_rest)
{
	if (!arg1.isnull)
		return arg1;
	return PG_COALESCE(kcxt, args_rest...);
}

template <typename T>
STATIC_INLINE(T)
PG_GREATEST(kern_context *kcxt, const T& arg)
{
	return arg;
}

template <typename T, typename ...R>
STATIC_INLINE(T)
PG_GREATEST(kern_context *kcxt, const T& arg1, const R&... args_rest)
{
	if (arg1.isnull)
		return PG_GREATEST(kcxt, args_rest...);
	else
	{
		T			arg2 = PG_GREATEST(kcxt, args_rest...);
		pg_int4_t	cmp;

		cmp = pgfn_type_compare(kcxt, arg1, arg2);
		if (cmp.isnull)
			return arg1;
		else if (cmp.value > 0)
			return arg1;
		else
			return arg2;
	}
}

template <typename T>
STATIC_INLINE(T)
PG_LEAST(kern_context *kcxt, const T& arg)
{
	return arg;
}

template <typename T, typename... R>
STATIC_INLINE(T)
PG_LEAST(kern_context *kcxt, const T& arg1, const R&... args_rest)
{
	if (arg1.isnull)
		return PG_LEAST(kcxt, args_rest...);
	else
	{
		T			arg2 = PG_LEAST(kcxt, args_rest...);
		pg_int4_t	cmp;

		cmp = pgfn_type_compare(kcxt, arg1, arg2);
		if (cmp.isnull)
			return arg1;
		else if (cmp.value > 0)
			return arg2;
		else
			return arg1;
	}
}

/*
 * Support routine for NullTest
 */
template <typename T>
STATIC_INLINE(pg_bool_t)
PG_ISNULL(kern_context *kcxt, T arg)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value = arg.isnull;

	return result;
}

template <typename T>
STATIC_INLINE(pg_bool_t)
PG_ISNOTNULL(kern_context *kcxt, T arg)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value = !arg.isnull;

	return result;
}

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
