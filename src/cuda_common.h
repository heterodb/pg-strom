/*
 * cuda_common.h
 *
 * A common header for CUDA device code
 * --
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
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
#else
typedef long				cl_long;
typedef unsigned long		cl_ulong;
#endif
#ifdef __CUDACC__
#include <cuda_fp16.h>
typedef __half				cl_half;
#endif
typedef float				cl_float;
typedef double				cl_double;

/* CUDA NVRTC always adds -D__CUDACC__ on device code. */
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

#define Assert(cond)	assert(cond)

/* Another basic type definitions */
typedef cl_ulong	hostptr_t;
typedef size_t		devptr_t;
typedef cl_ulong	Datum;

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
#define HALF_MAX		__short_as_half(0x7bff)
#define HALF_MIN		__short_as_half(0x0400)
#define HALF_INFINITY	__short_as_half(0x0x7c00)
#define FLT_MAX			__int_as_float(0x7f7fffffU)
#define FLT_MIN			__int_as_float(0x00800000U)
#define FLT_INFINITY	__int_as_float(0x7f800000U)
#define FLT_NAN			__int_as_float(0x7fffffffU)
#define DBL_MAX			__longlong_as_double(0x7fefffffffffffffULL)
#define DBL_MIN			__longlong_as_double(0x0010000000000000ULL)
#define DBL_INFINITY	__longlong_as_double(0x7ff0000000000000ULL)
#define DBL_NAN			__longlong_as_double(0x7fffffffffffffffULL)

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
#include "access/htup_details.h"
#include "storage/itemptr.h"
#define __device__		/* address space qualifier is noise on host */
#define __global__		/* address space qualifier is noise on host */
#define __constant__	/* address space qualifier is noise on host */
#define __shared__		/* address space qualifier is noise on host */
typedef uintptr_t		hostptr_t;
#define __ldg(x)		(*(x))	/* cache access hint is just a noise on host */
#endif	/* __CUDACC__ */

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
	__device__ __forceinline__ static RET_TYPE __attribute__ ((unused))
#define STATIC_FUNCTION(RET_TYPE)				\
	__device__ static RET_TYPE __attribute__ ((unused))
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
#define StromError_Success				0		/* OK */
#define StromError_Suspend				1001	/* GPU kernel suspend */
#define StromError_CpuReCheck			1002	/* Re-checked by CPU */
#define StromError_InvalidValue			1003	/* Invalid values */
#define StromError_DataStoreNoSpace		1004	/* No space left on KDS */
#define StromError_WrongCodeGeneration	1005	/* Wrong GPU code generation */
#define StromError_OutOfMemory			1006	/* Out of Memory */

#define StromError_CudaDevRunTimeBase	1000000

/*
 * Kernel functions identifier
 */
#define StromKernel_HostPGStrom						0x0001
#define StromKernel_CudaRuntime						0x0002
#define StromKernel_NVMeStrom						0x0003
#define StromKernel_gpuscan_exec_quals_row			0x0101
#define StromKernel_gpuscan_exec_quals_block		0x0102
#define StromKernel_gpuscan_exec_quals_column		0x0103
#define StromKernel_gpujoin_main					0x0201
#define StromKernel_gpujoin_right_outer				0x0202
#define StromKernel_gpupreagg_setup_row				0x0301
#define StromKernel_gpupreagg_setup_block			0x0302
#define StromKernel_gpupreagg_setup_column			0x0303
#define StromKernel_gpupreagg_nogroup_reduction		0x0304
#define StromKernel_gpupreagg_groupby_reduction		0x0305
#define StromKernel_plcuda_prep_kernel				0x0501
#define StromKernel_plcuda_main_kernel				0x0502
#define StromKernel_plcuda_post_kernel				0x0503

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

typedef struct
{
	kern_errorbuf	e;
	struct kern_parambuf *kparams;
} kern_context;

#define INIT_KERNEL_CONTEXT(kcxt,kfunction,__kparams)		\
	do {													\
		(kcxt)->e.errcode = StromError_Success;				\
		(kcxt)->e.kernel = StromKernel_##kfunction;			\
		(kcxt)->e.lineno = 0;								\
		(kcxt)->e.filename[0] = '\0';						\
		(kcxt)->kparams = (__kparams);						\
		assert((cl_ulong)(__kparams) == MAXALIGN(__kparams));	\
	} while(0)

/*
 * It sets an error code unless no significant error code is already set.
 * Also, CpuReCheck has higher priority than RowFiltered because CpuReCheck
 * implies device cannot run the given expression completely.
 * (Usually, due to compressed or external varlena datum)
 */
#ifdef __CUDACC__
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
#define STROM_SET_RUNTIME_ERROR(p_kerror, errcode)						\
	STROM_SET_ERROR((p_kerror), (errcode) == cudaSuccess ?				\
					(cl_int)(errcode) :									\
					(cl_int)(errcode) + StromError_CudaDevRunTimeBase)
/*
 * kern_writeback_error_status
 */
STATIC_INLINE(void)
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
 * | kern_colmeta colmeta[1]                                       |
 * |        :                                                      |
 * | kern_colmeta colmeta[M-1]                                     |
 * +----------------------------------------------+----------------+
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
	/* (only column) offset to the values array; that is always aligned by
	 * MAXALIGN(). For large KDS >4GB, va_offset shall be used with 3bits-
	 * shift to represents up to 32GB. */
	cl_uint			va_offset;
	/* (only column) total size of varlena body or NULL bitmap if attbyval;
	 * because of the same reason, extra_sz shall be used with 3bits shift. */
	cl_uint			extra_sz;
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
	cl_char			tdhasoid;	/* copy of TupleDesc.tdhasoid */
	cl_uint			tdtypeid;	/* copy of TupleDesc.tdtypeid */
	cl_int			tdtypmod;	/* copy of TupleDesc.tdtypmod */
	cl_uint			table_oid;	/* OID of the table (only if GpuScan) */
	cl_uint			nslots;		/* width of hash-slot (only HASH format) */
	cl_uint			hash_min;	/* minimum hash-value (only HASH format) */
	cl_uint			hash_max;	/* maximum hash-value (only HASH format) */
	cl_uint			nrows_per_block; /* average number of rows per
									  * PostgreSQL block (only BLOCK format) */
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
#define KDS_CALCULATE_ROW_LENGTH(ncols,nitems,data_len)		\
	(KDS_CALCULATE_FRONTEND_LENGTH((ncols),0,(nitems)) +	\
	 MAXALIGN(data_len))
#define KDS_CALCULATE_HASH_LENGTH(ncols,nitems,data_len)	\
	(KDS_CALCULATE_FRONTEND_LENGTH((ncols),__KDS_NSLOTS(nitems),(nitems)) +	\
	 MAXALIGN(data_len))
#define KDS_CALCULATE_SLOT_LENGTH(ncols,nitems)	\
	(KDS_CALCULATE_HEAD_LENGTH(ncols) +			\
	 LONGALIGN((sizeof(Datum) +					\
				sizeof(char)) * (ncols)) * (nitems))

/* length of the header portion of kern_data_store */
#define KERN_DATA_STORE_HEAD_LENGTH(kds)			\
	KDS_CALCULATE_HEAD_LENGTH((kds)->ncols)

/* head address of data body */
#define KERN_DATA_STORE_BODY(kds)					\
	((char *)(kds) + KERN_DATA_STORE_HEAD_LENGTH(kds))

/* access function for row- and hash-format */
STATIC_INLINE(cl_uint *)
KERN_DATA_STORE_ROWINDEX(kern_data_store *kds)
{
	Assert(__ldg(&kds->format) == KDS_FORMAT_ROW ||
		   __ldg(&kds->format) == KDS_FORMAT_HASH);
	return (cl_uint *)KERN_DATA_STORE_BODY(kds);
}

/* access function for hash-format */
STATIC_INLINE(cl_uint *)
KERN_DATA_STORE_HASHSLOT(kern_data_store *kds)
{
	Assert(__ldg(&kds->format) == KDS_FORMAT_HASH);
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

	Assert(__ldg(&kds->format) == KDS_FORMAT_ROW ||
		   __ldg(&kds->format) == KDS_FORMAT_HASH);
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
	Assert(offset < __ldg(&kds->length));
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
#define KERN_DATA_STORE_SLOT_LENGTH(kds,nitems)				\
	KDS_CALCULATE_SLOT_LENGTH((kds)->ncols,(nitems))

#define KERN_DATA_STORE_VALUES(kds,kds_index)				\
	((Datum *)((char *)(kds) +								\
			   KDS_CALCULATE_SLOT_LENGTH((kds)->ncols,(kds_index))))

#define KERN_DATA_STORE_ISNULL(kds,kds_index)				\
	((char *)(KERN_DATA_STORE_VALUES((kds),(kds_index)) + (kds)->ncols))

/* access macro for block format */
#define KERN_DATA_STORE_PARTSZ(kds)							\
	Min((__ldg(&(kds)->nrows_per_block) +					\
		 warpSize - 1) & ~(warpSize - 1), get_local_size())

#define KERN_DATA_STORE_BLOCK_BLCKNR(kds, kds_index)			\
	(((BlockNumber *)KERN_DATA_STORE_BODY(kds))[(kds_index)])

#define KERN_DATA_STORE_BLOCK_PGPAGE(kds, kds_index)				\
	(struct PageHeaderData *)(KERN_DATA_STORE_BODY(kds) +			\
							  STROMALIGN(sizeof(BlockNumber) *		\
										 __ldg(&(kds)->nrooms)) +	\
							  BLCKSZ * kds_index)
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

#ifdef __CUDACC__
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
 *
 * pg_XXXX_datum_ref() routine of types are responsible to transform disk
 * format to internal representation.
 * pg_XXXX_datum_store() routine of types are also responsible to transform
 * internal representation to disk format. We need to pay attention on
 * projection stage. If and when GPU code tries to store expressions which
 * are not simple Var, Const or Param, these internal representation must
 * be written to extra-buffer first.
 */

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
	STATIC_INLINE(pg_##NAME##_t)							\
	pg_##NAME##_datum_ref(kern_context *kcxt,				\
						  void *datum)						\
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
	STATIC_INLINE(void)										\
	pg_datum_ref(kern_context *kcxt,						\
				 pg_##NAME##_t &result, void *datum)		\
	{														\
		result = pg_##NAME##_datum_ref(kcxt, datum);		\
	}														\
															\
	STATIC_INLINE(cl_uint)									\
	pg_##NAME##_datum_store(kern_context *kcxt,				\
							void *extra_buf,				\
							pg_##NAME##_t datum)			\
	{														\
		if (datum.isnull)									\
			return 0;										\
		if (extra_buf)										\
			*((BASE *)extra_buf) = datum.value;				\
		return sizeof(BASE);								\
	}														\
															\
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

/*
 * Template of variable classes: fixed-length referenced by pointer
 * ----------------------------------------------------------------
 */
#define STROMCL_INDIRECT_VARREF_TEMPLATE(NAME,BASE)			\
	STATIC_INLINE(pg_##NAME##_t)							\
	pg_##NAME##_datum_ref(kern_context *kcxt,				\
						  void *datum)						\
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
	STATIC_INLINE(void)										\
	pg_datum_ref(kern_context *kcxt,						\
				 pg_##NAME##_t &result, void *datum)		\
	{														\
		result = pg_##NAME##_datum_ref(kcxt, datum);		\
	}														\
															\
	STATIC_INLINE(cl_uint)									\
	pg_##NAME##_datum_store(kern_context *kcxt,				\
							void *extra_buf,				\
							pg_##NAME##_t datum)			\
	{														\
		if (datum.isnull)									\
			return 0;										\
		if (extra_buf)										\
			memcpy(extra_buf, &datum.value, sizeof(BASE));	\
		return sizeof(BASE);								\
	}														\
															\
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

STATIC_FUNCTION(cl_uint)
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
	STATIC_INLINE(cl_uint)										\
	pg_##NAME##_comp_crc32(const cl_uint *crc32_table,			\
						   cl_uint hash, pg_##NAME##_t datum)	\
	{															\
		if (!datum.isnull)										\
		{														\
			hash = pg_common_comp_crc32(crc32_table,			\
										hash,					\
										(char *)&datum.value,	\
										sizeof(BASE));			\
		}														\
		return hash;											\
	}

#define STROMCL_SIMPLE_TYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_COMP_CRC32_TEMPLATE(NAME,BASE)

#define STROMCL_INDIRECT_TYPE_TEMPLATE(NAME,BASE)	\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)		\
	STROMCL_INDIRECT_VARREF_TEMPLATE(NAME,BASE)		\
	STROMCL_SIMPLE_COMP_CRC32_TEMPLATE(NAME,BASE)	\
	STATIC_INLINE(Datum)							\
	pg_##NAME##_as_datum(void *addr)				\
	{												\
		return PointerGetDatum(addr);				\
	}

/* template for binary compatible type-cast */
#define STROMCL_SIMPLE_TYPECAST_TEMPLATE(SOURCE,TARGET)		\
	STATIC_INLINE(pg_##TARGET##_t)							\
	to_##TARGET(pg_##SOURCE##_t arg)						\
	{														\
		pg_##TARGET##_t r;									\
		r.isnull = arg.isnull;								\
		r.value = arg.value;								\
		return r;											\
	}

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
STROMCL_SIMPLE_TYPE_TEMPLATE(bool, cl_bool)
STATIC_INLINE(Datum)
pg_bool_as_datum(void *addr)
{
	cl_bool		val = *((cl_bool *)addr);
	return SET_1_BYTE(val);
}
#endif

/* pg_int2_t */
#ifndef PG_INT2_TYPE_DEFINED
#define PG_INT2_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int2, cl_short)
STATIC_INLINE(Datum)
pg_int2_as_datum(void *addr)
{
	cl_short	val = *((cl_short *)addr);
	return SET_2_BYTES(val);
}
#endif

/* pg_int4_t */
#ifndef PG_INT4_TYPE_DEFINED
#define PG_INT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int4, cl_int)
STATIC_INLINE(Datum)
pg_int4_as_datum(void *addr)
{
	cl_int		val = *((cl_int *)addr);
	return SET_4_BYTES(val);
}
#endif

/* pg_int8_t */
#ifndef PG_INT8_TYPE_DEFINED
#define PG_INT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int8, cl_long)
STATIC_INLINE(Datum)
pg_int8_as_datum(void *addr)
{
	cl_long		val = *((cl_long *)addr);
	return SET_8_BYTES(val);
}
#endif

/* pg_float2_t */
#ifndef PG_FLOAT2_TYPE_DEFINED
#define PG_FLOAT2_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float2, cl_half)
STATIC_INLINE(Datum)
pg_float2_as_datum(void *addr)
{
	cl_half		val = *((cl_half *)addr);
	return SET_2_BYTES(__half_as_short(val));
}
#endif

/* pg_float4_t */
#ifndef PG_FLOAT4_TYPE_DEFINED
#define PG_FLOAT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float4, cl_float)
STATIC_INLINE(Datum)
pg_float4_as_datum(void *addr)
{
	cl_float	val = *((cl_float *)addr);
	return SET_4_BYTES(__float_as_int(val));
}
#endif

/* pg_float8_t */
#ifndef PG_FLOAT8_TYPE_DEFINED
#define PG_FLOAT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float8, cl_double)
STATIC_INLINE(Datum)
pg_float8_as_datum(void *addr)
{
	cl_double	val = *((cl_double *)addr);
	return SET_8_BYTES(__double_as_longlong(val));
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
 * toast_decompress_datum - decompress a compressed version of a varlena datum
 */
STATIC_FUNCTION(cl_int)
pglz_decompress(const char *source, cl_int slen,
				char *dest, cl_int rawsize)
{
	const cl_uchar *sp;
	const cl_uchar *srcend;
	cl_uchar	   *dp;
	cl_uchar	   *destend;

	sp = (const cl_uchar *) source;
	srcend = ((const cl_uchar *) source) + slen;
	dp = (cl_uchar *) dest;
	destend = dp + rawsize;

	while (sp < srcend && dp < destend)
	{
		/*
		 * Read one control byte and process the next 8 items (or as many as
		 * remain in the compressed input).
		 */
		cl_uchar	ctrl = *sp++;
		int			ctrlc;

		for (ctrlc = 0; ctrlc < 8 && sp < srcend; ctrlc++)
		{
			if (ctrl & 1)
			{
				/*
				 * Otherwise it contains the match length minus 3 and the
				 * upper 4 bits of the offset. The next following byte
				 * contains the lower 8 bits of the offset. If the length is
				 * coded as 18, another extension tag byte tells how much
				 * longer the match really was (0-255).
				 */
				cl_int		len;
				cl_int		off;

				len = (sp[0] & 0x0f) + 3;
				off = ((sp[0] & 0xf0) << 4) | sp[1];
				sp += 2;
				if (len == 18)
					len += *sp++;

				/*
				 * Check for output buffer overrun, to ensure we don't clobber
				 * memory in case of corrupt input.  Note: we must advance dp
				 * here to ensure the error is detected below the loop.  We
				 * don't simply put the elog inside the loop since that will
				 * probably interfere with optimization.
				 */
				if (dp + len > destend)
				{
					dp += len;
					break;
				}
				/*
				 * Now we copy the bytes specified by the tag from OUTPUT to
				 * OUTPUT. It is dangerous and platform dependent to use
				 * memcpy() here, because the copied areas could overlap
				 * extremely!
				 */
				while (len--)
				{
					*dp = dp[-off];
					dp++;
				}
			}
			else
			{
				/*
				 * An unset control bit means LITERAL BYTE. So we just copy
				 * one from INPUT to OUTPUT.
				 */
				if (dp >= destend)		/* check for buffer overrun */
					break;				/* do not clobber memory */

				*dp++ = *sp++;
			}

			/*
			 * Advance the control bit
			 */
			ctrl >>= 1;
		}
	}

	/*
	 * Check we decompressed the right amount.
	 */
	if (dp != destend || sp != srcend)
		return -1;

	/*
	 * That's it.
	 */
	return rawsize;
}

STATIC_INLINE(cl_bool)
toast_decompress_datum(char *buffer, cl_uint buflen,
					   const struct varlena *datum)
{
	cl_int		rawsize;

	assert(VARATT_IS_COMPRESSED(datum));
	rawsize = TOAST_COMPRESS_RAWSIZE(datum);
	if (rawsize + VARHDRSZ > buflen)
		return false;
	SET_VARSIZE(buffer, rawsize + VARHDRSZ);
	if (pglz_decompress(TOAST_COMPRESS_RAWDATA(datum),
						VARSIZE(datum) - TOAST_COMPRESS_HDRSZ,
						buffer + VARHDRSZ,
						rawsize) < 0)
	{
		printf("GPU kernel: compressed varlena datum is corrupted\n");
		return false;
	}
	return true;
}

/*
 * template to reference variable length variables
 */
#define STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)					\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME, varlena *)

#define STROMCL_VARLENA_VARREF_TEMPLATE(NAME)					\
	STATIC_INLINE(pg_##NAME##_t)								\
	pg_##NAME##_datum_ref(kern_context *kcxt,					\
						  void *datum)							\
	{															\
		pg_##NAME##_t result;									\
																\
		if (!datum)												\
			result.isnull = true;								\
		else if (!VARATT_IS_4B_U(datum) &&						\
				 !VARATT_IS_1B(datum))							\
		{														\
			result.isnull = true;								\
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);	\
		}														\
		else													\
		{														\
			result.isnull = false;								\
			result.value = (varlena *)datum;					\
		}														\
		return result;											\
	}															\
	STATIC_INLINE(void)											\
	pg_datum_ref(kern_context *kcxt,							\
				 pg_##NAME##_t &result, void *datum)			\
	{															\
		result = pg_##NAME##_datum_ref(kcxt, datum);			\
	}															\
																\
	STATIC_INLINE(cl_uint)										\
	pg_##NAME##_datum_store(kern_context *kcxt,					\
							void *extra_buf,					\
							pg_##NAME##_t datum)				\
	{															\
		cl_uint		vl_size;									\
																\
		if (datum.isnull)										\
			return 0;											\
		vl_size = VARSIZE_ANY(datum.value);						\
		if (extra_buf)											\
			memcpy(extra_buf, datum.value, vl_size);			\
		return vl_size;											\
	}															\
																\
	STATIC_INLINE(pg_##NAME##_t)								\
	pg_##NAME##_param(kern_context *kcxt, cl_uint param_id)		\
	{															\
		kern_parambuf  *kparams = kcxt->kparams;				\
		pg_##NAME##_t	result;									\
																\
		if (param_id < kparams->nparams &&						\
			kparams->poffset[param_id] > 0)						\
		{														\
			varlena *vl_val = (varlena *)						\
				((char *)kparams + kparams->poffset[param_id]);	\
			if (VARATT_IS_4B_U(vl_val) || VARATT_IS_1B(vl_val))	\
			{													\
				result.value = vl_val;							\
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

#define STROMCL_VARLENA_COMP_CRC32_TEMPLATE(NAME)				\
	STATIC_INLINE(cl_uint)										\
	pg_##NAME##_comp_crc32(const cl_uint *crc32_table,			\
						   cl_uint hash, pg_##NAME##_t datum)	\
	{															\
		if (datum.isnull)										\
			return hash;										\
		/* TODO: CpuReCheck if compressed or external */		\
		hash = pg_common_comp_crc32(crc32_table,				\
									hash,						\
									VARDATA_ANY(datum.value),	\
									VARSIZE_ANY_EXHDR(datum.value)); \
		return hash;											\
	}

#define STROMCL_VARLENA_TYPE_TEMPLATE(NAME)						\
	STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)						\
	STROMCL_VARLENA_VARREF_TEMPLATE(NAME)						\
	STROMCL_VARLENA_COMP_CRC32_TEMPLATE(NAME)					\
	STATIC_INLINE(Datum)										\
	pg_##NAME##_as_datum(void *addr)							\
	{															\
		return PointerGetDatum(addr);							\
	}

/* generic varlena */
#ifndef PG_VARLENA_TYPE_DEFINED
#define PG_VARLENA_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(varlena)
#endif

/* pg_bytea_t */
#ifndef PG_BYTEA_TYPE_DEFINED
#define PG_BYTEA_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(bytea)
#endif

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
			__ncols = min(__ldg(&(kds)->ncols),							\
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

STATIC_FUNCTION(void *)
kern_get_datum_column(kern_data_store *kds,
					  cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta *cmeta;
	size_t		offset;
	char	   *values;
	char	   *nullmap;

	Assert(colidx < kds->ncols);
	cmeta = &kds->colmeta[colidx];
	/* special case handling if 'tableoid' system column */
	if (cmeta->attnum == TableOidAttributeNumber)
		return &kds->table_oid;
	offset = __kds_unpack(__ldg(&cmeta->va_offset));
	if (offset == 0)
		return NULL;
	values = (char *)kds + offset;
	if (__ldg(&cmeta->attlen) < 0)
	{
		Assert(!__ldg(&cmeta->attbyval));
		offset = ((cl_uint *)values)[rowidx];
		if (offset == 0)
			return NULL;
		Assert(offset < ((size_t)__ldg(&kds->nitems) * sizeof(cl_uint) +
						 (size_t)__ldg(&cmeta->extra_sz)));
		values += __kds_unpack(offset);
	}
	else
	{
		size_t	extra_sz = __kds_unpack(__ldg(&cmeta->extra_sz));
		cl_int	unitsz = TYPEALIGN(__ldg(&cmeta->attalign),
								   __ldg(&cmeta->attlen));
		if (extra_sz > 0)
		{
			Assert(MAXALIGN(BITMAPLEN(__ldg(&kds->nitems))) == extra_sz);
			nullmap = values + MAXALIGN(unitsz * __ldg(&kds->nitems));
			if (att_isnull(rowidx, nullmap))
				return NULL;
		}
		values += unitsz * rowidx;
	}
	return (void *)values;
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
	if (kds->format == KDS_FORMAT_COLUMN)
		return kern_get_datum_column(kds, colidx, rowidx);
	/* TODO: put StromError_DataStoreCorruption error here */
	return NULL;
}

#ifdef __CUDACC__
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
 * pgstromStairlikeBinaryCount
 *
 * A special optimized version of pgstromStairlikeSum, for binary count.
 * It has smaller number of __syncthreads().
 */
STATIC_FUNCTION(cl_uint)
pgstromStairlikeBinaryCount(int predicate, cl_uint *total_count)
{
	cl_uint	   *items = SHARED_WORKMEM(cl_uint);
	cl_uint		nwarps = get_local_size() / warpSize;
	cl_uint		warp_id = get_local_id() / warpSize;
	cl_uint		w_bitmap;
	cl_uint		stair_count;
	cl_int		unit_sz;
	cl_int		i, j;

#if CUDA_VERSION < 9000
	w_bitmap = __ballot(predicate);
#else
	w_bitmap = __ballot_sync(__activemask(), predicate);
#endif
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
STATIC_FUNCTION(T)
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

/*
 * Utility functions to reference system columns
 *   (except for ctid and table_oid)
 */
STATIC_INLINE(Datum)
kern_getsysatt_oid(HeapTupleHeaderData *htup)
{
	if ((htup->t_infomask & HEAP_HASOID) != 0)
		return *((cl_uint *)((char *) htup
							 + htup->t_hoff
							 - sizeof(cl_uint)));
	return 0;	/* InvalidOid */
}

STATIC_INLINE(Datum)
kern_getsysatt_xmin(HeapTupleHeaderData *htup)
{
	return (Datum) htup->t_choice.t_heap.t_xmin;
}

STATIC_INLINE(Datum)
kern_getsysatt_xmax(HeapTupleHeaderData *htup)
{
	return (Datum) htup->t_choice.t_heap.t_xmax;
}

STATIC_INLINE(Datum)
kern_getsysatt_cmin(HeapTupleHeaderData *htup)
{
	return (Datum) htup->t_choice.t_heap.t_field3.t_cid;
}

STATIC_INLINE(Datum)
kern_getsysatt_cmax(HeapTupleHeaderData *htup)
{
	return (Datum) htup->t_choice.t_heap.t_field3.t_cid;
}

/*
 * compute_heaptuple_size
 */
STATIC_FUNCTION(cl_uint)
compute_heaptuple_size(kern_context *kcxt,
					   kern_data_store *kds,
					   Datum *tup_values,
					   cl_bool *tup_isnull)
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
			if (cmeta.attlen > 0)
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
					  Datum	   *tup_values,		/* out */
					  cl_bool  *tup_isnull)		/* out */
{
	/* 'htup' must be aligned to 8bytes */
	assert(((cl_ulong)htup & (MAXIMUM_ALIGNOF-1)) == 0);
	if (!htup)
	{
		memset(tup_isnull, -1, sizeof(cl_bool) * nattrs);
	}
	else
	{
		cl_uint		offset = htup->t_hoff;
		cl_bool		tup_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
		cl_uint		i, ncols = (htup->t_infomask2 & HEAP_NATTS_MASK);

		ncols = min(ncols, nattrs);
		for (i=0; i < ncols; i++)
		{
			if (tup_hasnull && att_isnull(i, htup->t_bits))
			{
				tup_isnull[i] = true;
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
						tup_isnull[i] = true;
						assert(false);
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
				tup_isnull[i] = false;
			}
		}
		/*
		 * Fill up remaining columns if source tuple has less columns than
		 * length of the array; that is definition of the destination
		 */
		while (i < nattrs)
			tup_isnull[i++] = true;
	}
}

/*
 * __form_kern_heaptuple
 */
STATIC_FUNCTION(cl_uint)
__form_kern_heaptuple(void    *buffer,		/* out */
					  cl_int   ncols,		/* in; num of attributes */
					  kern_colmeta *colmeta,/* in: column definition */
					  HeapTupleFields *htup_field, /* in: only if tuple */
					  cl_int   comp_typmod,	/* in */
					  cl_uint  comp_typeid,	/* in */
					  cl_uint  htuple_oid,	/* in */
					  Datum   *tup_values,	/* in */
					  cl_bool *tup_isnull)	/* in */
{
	HeapTupleHeaderData *htup = (HeapTupleHeaderData *)buffer;
	cl_bool		tup_hasnull = false;
	cl_ushort	t_infomask;
	cl_uint		t_hoff;
	cl_uint		i, curr;

	/* alignment checks */
	assert((devptr_t)htup == MAXALIGN(htup));

	/* has any NULL attribute? */
	if (tup_isnull != NULL)
	{
		for (i=0; i < ncols; i++)
		{
			if (tup_isnull[i])
			{
				tup_hasnull = true;
				break;
			}
		}
	}
	t_infomask = (tup_hasnull ? HEAP_HASNULL : 0);

	/* setup HeapTupleHeaderData */
	if (htup_field)
		memcpy(&htup->t_choice.t_heap, htup_field, sizeof(HeapTupleFields));
	else
	{
		/* datum_len_ shall be set on the tail  */
		htup->t_choice.t_datum.datum_typmod = comp_typmod;
		htup->t_choice.t_datum.datum_typeid = comp_typeid;
	}
	htup->t_ctid.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
	htup->t_ctid.ip_blkid.bi_lo = 0xffff;
	htup->t_ctid.ip_posid = 0;				/* InvalidOffsetNumber */
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
		kern_colmeta   *cmeta = &colmeta[i];
		Datum			datum = tup_values[i];
		cl_bool			isnull = (!tup_isnull ? false : tup_isnull[i]);
		cl_int			padding;

		if (isnull)
		{
			assert(tup_hasnull);
			htup->t_bits[i >> 3] &= ~(1 << (i & 0x07));
		}
		else
		{
			if (tup_hasnull)
				htup->t_bits[i >> 3] |= (1 << (i & 0x07));

			padding = TYPEALIGN(cmeta->attalign, curr) - curr;
			if (cmeta->attbyval)
			{
				while (padding-- > 0)
					((char *)htup)[curr++] = '\0';
				assert(cmeta->attlen <= sizeof(datum));
				memcpy((char *)htup + curr, &datum, cmeta->attlen);
				curr += cmeta->attlen;
			}
			else if (cmeta->attlen > 0)
			{
				while (padding-- > 0)
					((char *)htup)[curr++] = '\0';
				memcpy((char *)htup + curr,
					   DatumGetPointer(datum), cmeta->attlen);
				curr += cmeta->attlen;
			}
			else
			{
				cl_uint		vl_len = VARSIZE_ANY(datum);

				if (!VARATT_IS_1B(datum))
				{
					while (padding-- > 0)
						((char *)htup)[curr++] = '\0';
				}
				t_infomask |= HEAP_HASVARWIDTH;
				memcpy((char *)htup + curr,
					   DatumGetPointer(datum), vl_len);
				curr += vl_len;
			}
		}
	}
	htup->t_infomask = t_infomask;
	curr = MAXALIGN(curr);
	if (!htup_field)
		SET_VARSIZE(&htup->t_choice.t_datum, curr);

	return curr;
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
 * tup_self     ... item pointer of the tuple, if any
 * tx_attrs     ... xmin,xmax,cmin/cmax, if any
 * tup_values   ... array of result datum
 * tup_isnull   ... array of null flags
 */
STATIC_INLINE(cl_uint)
form_kern_heaptuple(kern_tupitem	*tupitem,	/* out */
					cl_int			 ncols,		/* in */
					kern_colmeta	*colmeta,	/* in */
					ItemPointerData *tup_self,	/* in, optional */
					HeapTupleFields *htup_field,/* in, optional */
					cl_uint			 htuple_oid,/* in, optional */
					Datum           *tup_values,/* in */
					cl_bool         *tup_isnull)/* in */
{
	/* check alignment */
	assert((devptr_t)tupitem == MAXALIGN(tupitem));

	/* setup kern_tupitem */
	if (tup_self)
		tupitem->t_self = *tup_self;
	else
	{
		tupitem->t_self.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
		tupitem->t_self.ip_blkid.bi_lo = 0xffff;
		tupitem->t_self.ip_posid = 0;				/* InvalidOffsetNumber */
	}
	tupitem->t_len = __form_kern_heaptuple(&tupitem->htup,
										   ncols,
										   colmeta,
										   htup_field,
										   0,	/* not composite type */
										   0,	/* not composite type */
										   htuple_oid,
										   tup_values,
										   tup_isnull);
	return tupitem->t_len;
}

/*
 * setup_kern_heaptuple
 *
 * A utility routine to set up a composite type data structure
 * on the supplied pre-allocated region. It in
 *
 * @buffer     ... pointer to global memory where caller wants to construct
 *                 a composite datum. It must have enough length and also
 *                 must be aligned to DWORD.
 * @comp_type  ... reference to kern_colmeta of the composite type itself
 * @sub_types  ... reference to an array of kern_colmeta of the sub-fields of
 *                 the destination composite type.
 * @nfields    ... number of sub-fields of the composite type.
 * @tup_datum  ... in: it gives length of individual sub-fields.
 *                 out: it returns pointer of the individual sub-fields.
 * @tup_isnull ... flag to indicate which fields are NULL. If @tup_isnull is
 *                 NULL, it means the composite type contains no NULL items.
 */
STATIC_INLINE(cl_uint)
form_kern_composite_type(void *buffer,			/* out */
						 kern_colmeta *typmeta, /* in: column definitions */
						 cl_int        ncols,   /* in: # of attributes */
						 kern_colmeta *colmeta, /* in: type attributes */
						 Datum        *tup_values,	/* in: */
						 cl_bool      *tup_isnull)	/* in: */
{
	return __form_kern_heaptuple(buffer,
								 ncols,
								 colmeta,
								 NULL,
								 typmeta->atttypmod,
								 typmeta->atttypid,
								 0,	/* composite type never have OID */
								 tup_values,
								 tup_isnull);
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

/* memory compare */
STATIC_INLINE(cl_int)
memcmp(const void *s1, const void *s2, size_t n)
{
	const cl_uchar *p1 = (const cl_uchar *)s1;
	const cl_uchar *p2 = (const cl_uchar *)s2;

	while (n--)
	{
		if (*p1 != *p2)
			return ((int)*p1) - ((int)*p2);
		p1++;
		p2++;
	}
	return 0;
}

/* string comparison */
STATIC_INLINE(cl_int)
strcmp(const char *__s1, const char *__s2)
{
	const cl_uchar *s1 = (const cl_uchar *) __s1;
	const cl_uchar *s2 = (const cl_uchar *) __s2;
	cl_int		c1, c2;

	do {
		c1 = (cl_uchar) *s1++;
		c2 = (cl_uchar) *s2++;

		if (c1 == '\0')
			return c1 - c2;
	} while (c1 == c2);

	return c1 - c2;
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
