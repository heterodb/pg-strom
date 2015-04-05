/*
 * cuda_common.h
 *
 * A common header for CUDA device code
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

#define INT64CONST(x)	((cl_long) x##L)
#define UINT64CONST(x)	((cl_ulong) x##UL)

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
 * MEMO: Manner like OpenCL style.
 */
#define get_local_id()			(threadIdx.x)
#define get_local_size()		(blockDim.x)
#define get_global_id()			(threadIdx.x + blockIdx.x * blockDim.x)
#define get_global_size()		(blockDim.x * gridDim.x)


#else	/* __CUDACC__ */
#include "access/htup_details.h"
#include "storage/itemptr.h"
#define __device__		/* address space qualifier is noise on host */
#define __global__		/* address space qualifier is noise on host */
#define __constant__	/* address space qualifier is noise on host */
#define __shared__		/* address space qualifier is noise on host */
typedef uintptr_t	hostptr_t;
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
#else
#define STATIC_INLINE(RET_TYPE)		static inline RET_TYPE
#define STATIC_FUNCTION(RET_TYPE)	static inline RET_TYPE
#define KERNEL_FUNCTION(RET_TYPE)	RET_TYPE
#endif

/*
 * Error code definition
 */
#define StromError_Success					   0 /* OK */
#define StromError_CpuReCheck				1000 /* To be re-checked by CPU */
#define StromError_CudaInternal				1001 /* CUDA internal error */
#define StromError_OutOfMemory				1002 /* Out of memory */
#define StromError_OutOfSharedMemory		1003 /* Out of shared memory */
#define StromError_DataStoreCorruption		2000 /* KDS corrupted */
#define StromError_DataStoreNoSpace			2001 /* KDS has no space */
#define StromError_DataStoreOutOfRange		2002 /* out of KDS range access */
#define StromError_SanityCheckViolation		2003 /* sanity check violation */

#ifdef __CUDACC__
/*
 * Hint for compiler
 */
#if __CUDA_ARCH__ < 200
#define __likely_max_threads__		__launch_bounds__(512)
#else
#define __likely_max_threads__		__launch_bounds__(1024)
#endif

/*
 * It sets an error code unless no significant error code is already set.
 * Also, CpuReCheck has higher priority than RowFiltered because CpuReCheck
 * implies device cannot run the given expression completely.
 * (Usually, due to compressed or external varlena datum)
 */
STATIC_INLINE(void)
STROM_SET_ERROR(cl_int *p_error, cl_int errcode)
{
	cl_int	oldcode = *p_error;

	if (oldcode == StromError_Success &&
		errcode != StromError_Success)
		*p_error = errcode;
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
#define STROMALIGN(LEN)			TYPEALIGN(STROMALIGN_LEN,LEN)
#define STROMALIGN_DOWN(LEN)	TYPEALIGN_DOWN(STROMALIGN_LEN,LEN)

/*
 * kern_data_store
 *
 * It stores row- and column-oriented values in the kernel space.
 *
 * +--------------------------------+
 * | hostptr                        |
 * +--------------------------------+
 * | length                         |
 * +--------------------------------+
 * | ncols                          |
 * +--------------------------------+
 * | nitems                         |
 * +--------------------------------+
 * | nrooms                         |
 * +--------------------------------+
 * | format                         |
 * +--------------------------------+
 * | colmeta[0]                     | aligned to
 * | colmeta[1]                     | STROMALIGN()
 * |   :                            |    |
 * | colmeta[M-1]                   |    V
 * +----------------+---------------+---------
 * |  <row-format>  | <slot-format> |
 * +----------------+---------------+
 * | tup_offset[0]  | values/isnull |
 * | tup_offset[1]  | pair of the   |
 * |    :           | 1st row       |
 * | tup_offset[N-1]| +-------------+
 * +----------------+ | values[0]   |
 * |    :           | |    :        |
 * |    :           | | values[N-1] |
 * +----------------+ +-------------+
 * | htup_item[N-1] | | isnull[0]   |
 * | +--------------+ |    :        |
 * | | t_len        | | isnull[N-1] |
 * | +--------------+-+-------------+
 * | | t_self       | values/isnull |
 * | +--------------+ pair of the   |
 * | | heap_tuple   | 2nd row       |
 * | |   :          | +-------------+
 * | |   :          | | values[0]   |
 * +-+--------------+ |    :        |
 * |     :          | | values[N-1] |
 * |     :          | +-------------+
 * +----------------+ | isnull[0]   |
 * | htup_item[0]   | |    :        |
 * | +--------------+ | isnull[N-1] |
 * | | t_len        +-+-------------+
 * | +--------------+       :       |
 * | | t_self       |       :       |
 * | +--------------+       :       |
 * | | heap_tuple   |       :       |
 * | |   :          |       :       |
 * | |   :          |       :       |
 * +-+--------------+---------------+
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
} kern_colmeta;


typedef struct
{
	cl_ushort			t_len;		/* length of tuple */
	ItemPointerData		t_self;		/* SelfItemPointer */
	HeapTupleHeaderData	htup;
} kern_tupitem;

#define KDS_FORMAT_ROW			1
#define KDS_FORMAT_SLOT			2

typedef struct {
	hostptr_t		hostptr;	/* address of kds on the host */
	cl_uint			length;		/* length of this data-store */
	cl_uint			usage;		/* usage of this data-store */
	cl_uint			ncols;		/* number of columns in this store */
	cl_uint			nitems; 	/* number of rows in this store */
	cl_uint			nrooms;		/* number of available rows in this store */
	cl_char			format;		/* one of KDS_FORMAT_* above */
	cl_char			tdhasoid;	/* copy of TupleDesc.tdhasoid */
	cl_uint			tdtypeid;	/* copy of TupleDesc.tdtypeid */
	cl_int			tdtypmod;	/* copy of TupleDesc.tdtypmod */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER]; /* metadata of columns */
} kern_data_store;

/* head address of data body */
#define KERN_DATA_STORE_BODY(kds)							\
	((char *)(kds) + STROMALIGN(offsetof(kern_data_store,	\
										 colmeta[(kds)->ncols])))

/* access macro for row-format */
#define KERN_DATA_STORE_TUPITEM(kds,kds_index)				\
	((kern_tupitem *)										\
	 ((char *)(kds) +										\
	  ((cl_uint *)KERN_DATA_STORE_BODY(kds))[(kds_index)]))

/* access macro for tuple-slot format */
#define KERN_DATA_STORE_VALUES(kds,kds_index)				\
	((Datum *)(KERN_DATA_STORE_BODY(kds) +					\
			   LONGALIGN((sizeof(Datum) + sizeof(char)) *	\
						 (kds)->ncols) * (kds_index)))
#define KERN_DATA_STORE_ISNULL(kds,kds_index)				\
	((cl_bool *)(KERN_DATA_STORE_VALUES((kds),(kds_index)) + (kds)->ncols))

/* length of kern_data_store */
#define KERN_DATA_STORE_LENGTH(kds)		\
	STROMALIGN((kds)->length)

/* length of the header portion of kern_data_store */
#define KERN_DATA_STORE_HEAD_LENGTH(kds)			\
	STROMALIGN(offsetof(kern_data_store, colmeta[(kds)->ncols]))

/*
 * kern_parambuf
 *
 * Const and Parameter buffer. It stores constant values during a particular
 * scan, so it may make sense if it is obvious length of kern_parambuf is
 * less than constant memory (NOTE: not implemented yet).
 */
typedef struct {
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

/*
 * kern_resultbuf
 *
 * Output buffer to write back calculation results on a parciular chunk.
 * 'errcode' informs a significant error that shall raise an error on
 * host side and abort transactions. 'results' informs row-level status.
 */
typedef struct {
	cl_uint		nrels;		/* number of relations to be appeared */
	cl_uint		nrooms;		/* max number of results rooms */
	cl_uint		nitems;		/* number of results being written */
	cl_int		errcode;	/* chunk-level error */
	cl_char		has_rechecks;
	cl_char		all_visible;
	cl_char		__padding__[2];
	cl_int		results[FLEXIBLE_ARRAY_MEMBER];
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
			   kern_data_store *ktoast,
			   cl_uint colidx, cl_uint rowidx);
/* forward declaration of writer interface to kern_data_store */
STATIC_INLINE(Datum *)
pg_common_vstore(kern_data_store *kds,
				 kern_data_store *ktoast,
				 int *errcode,
				 cl_uint colidx,
				 cl_uint rowidx,
				 cl_bool isnull);

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
	pg_##NAME##_datum_ref(int *errcode,						\
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
	__device__ pg_##NAME##_t								\
	pg_##NAME##_vref(kern_data_store *kds,					\
					 kern_data_store *ktoast,				\
					 int *errcode,							\
					 cl_uint colidx,						\
					 cl_uint rowidx)						\
	{														\
		void  *datum =										\
			kern_get_datum(kds,ktoast,colidx,rowidx);		\
		return pg_##NAME##_datum_ref(errcode,datum,false);	\
	}

#define STROMCL_SIMPLE_VARSTORE_TEMPLATE(NAME,BASE)			\
	STATIC_FUNCTION(void)									\
	pg_##NAME##_vstore(kern_data_store *kds,				\
					   kern_data_store *ktoast,				\
					   int *errcode,						\
					   cl_uint colidx,						\
					   cl_uint rowidx,						\
					   pg_##NAME##_t datum)					\
	{														\
		Datum *daddr;										\
		union {												\
			BASE		v_base;								\
			Datum		v_datum;							\
		} temp;												\
		daddr = pg_common_vstore(kds, ktoast, errcode,		\
								 colidx, rowidx,			\
								 datum.isnull);				\
		if (daddr)											\
		{													\
			temp.v_datum = 0;								\
			temp.v_base = datum.value;						\
			*daddr = temp.v_datum;							\
		}													\
	}

#define STROMCL_SIMPLE_PARAMREF_TEMPLATE(NAME,BASE)			\
	STATIC_FUNCTION(pg_##NAME##_t)							\
	pg_##NAME##_param(kern_parambuf *kparams,				\
					  int *errcode,							\
					  cl_uint param_id)						\
	{														\
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
	pgfn_##NAME##_isnull(int *errcode, pg_##NAME##_t arg)	\
	{														\
		pg_bool_t result;									\
															\
		result.isnull = false;								\
		result.value = arg.isnull;							\
		return result;										\
	}														\
															\
	STATIC_FUNCTION(pg_bool_t)								\
	pgfn_##NAME##_isnotnull(int *errcode, pg_##NAME##_t arg)\
	{														\
		pg_bool_t result;									\
															\
		result.isnull = false;								\
		result.value = !arg.isnull;							\
		return result;										\
	}

/*
 * Macros to calculate CRC32 value.
 * (logic was copied from pg_crc32.c)
 */
#define INIT_CRC32C(crc)		((crc) = 0xFFFFFFFF)
#define FIN_CRC32C(crc)			((crc) ^= 0xFFFFFFFF)
#define EQ_CRC32C(crc1,crc2)	((crc1) == (crc2))

#define STROMCL_SIMPLE_COMP_CRC32_TEMPLATE(NAME,BASE)			\
	STATIC_FUNCTION(cl_uint)									\
	pg_##NAME##_comp_crc32(const cl_uint *crc32_table,			\
						   cl_uint hash, pg_##NAME##_t datum)	\
	{															\
		cl_uint         __len = sizeof(BASE);					\
		cl_uint         __index;								\
		union {													\
			BASE        as_base;								\
			cl_uint     as_int;									\
			cl_ulong    as_long;								\
		} __data;												\
																\
		if (!datum.isnull)										\
		{														\
			__data.as_base = datum.value;						\
			while (__len-- > 0)									\
			{													\
				__index = (hash ^ __data.as_int) & 0xff;		\
				hash = crc32_table[__index] ^ ((hash) >> 8);	\
				__data.as_long = (__data.as_long >> 8);			\
			}													\
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

/* pg_bool_t */
#ifndef PG_BOOL_TYPE_DEFINED
#define PG_BOOL_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(bool, cl_bool)
#endif

/* pg_int2_t */
#ifndef PG_INT2_TYPE_DEFINED
#define PG_INT2_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int2, cl_short)
#endif

/* pg_int4_t */
#ifndef PG_INT4_TYPE_DEFINED
#define PG_INT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int4, cl_int)
#endif

/* pg_int8_t */
#ifndef PG_INT8_TYPE_DEFINED
#define PG_INT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int8, cl_long)
#endif

/* pg_float4_t */
#ifndef PG_FLOAT4_TYPE_DEFINED
#define PG_FLOAT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float4, cl_float)
#endif

/* pg_float8_t */
#ifndef PG_FLOAT8_TYPE_DEFINED
#define PG_FLOAT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float8, cl_double)
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

#define VARSIZE_4B(PTR) \
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
toast_raw_datum_size(cl_int *errcode, varlena *attr)
{
	size_t		result;

	if (VARATT_IS_EXTERNAL(attr))
	{
		/* should not appear in kernel space */
		STROM_SET_ERROR(errcode, StromError_CpuReCheck);
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
					kern_data_store *ktoast,
					cl_uint colidx, cl_uint rowidx)
{
	Datum	   *values = KERN_DATA_STORE_VALUES(kds,rowidx);
	cl_bool	   *isnull = KERN_DATA_STORE_ISNULL(kds,rowidx);
	kern_colmeta		cmeta = kds->colmeta[colidx];

	if (isnull[colidx])
		return NULL;
	if (cmeta.attlen > 0)
		return values + colidx;
	return (char *)(&ktoast->hostptr) + values[colidx];
}

STATIC_INLINE(void *)
kern_get_datum(kern_data_store *kds,
			   kern_data_store *ktoast,
			   cl_uint colidx, cl_uint rowidx)
{
	/* is it out of range? */
	if (colidx >= kds->ncols || rowidx >= kds->nitems)
		return NULL;
	if (kds->format == KDS_FORMAT_ROW)
		return kern_get_datum_row(kds, colidx, rowidx);
	if (kds->format == KDS_FORMAT_SLOT)
		return kern_get_datum_slot(kds,ktoast,colidx,rowidx);
	/* TODO: put StromError_DataStoreCorruption error here */
	return NULL;
}

/*
 * common function to store a value on tuple-slot format
 */
STATIC_FUNCTION(Datum *)
pg_common_vstore(kern_data_store *kds,
				 kern_data_store *ktoast,
				 int *errcode,
				 cl_uint colidx, cl_uint rowidx,
				 cl_bool isnull)
{
	Datum	   *slot_values;
	cl_bool	   *slot_isnull;
	/*
	 * Only tuple-slot is acceptable destination format.
	 * Only row- and row-flat are acceptable source format.
	 */
	if (kds->format != KDS_FORMAT_SLOT || ktoast->format != KDS_FORMAT_ROW)
	{
		STROM_SET_ERROR(errcode, StromError_SanityCheckViolation);
		return NULL;
	}
	/* out of range? */
	if (colidx >= kds->ncols || rowidx >= kds->nrooms)
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreOutOfRange);
		return NULL;
	}
	slot_values = KERN_DATA_STORE_VALUES(kds, rowidx);
	slot_isnull = KERN_DATA_STORE_ISNULL(kds, rowidx);

	slot_isnull[colidx] = (cl_char)(isnull ? 1 : 0);

	return slot_values + colidx;
}

/*
 * kern_fixup_data_store
 *
 * pg_xxx_vstore() interface stores varlena datum on kern_data_store with
 * KDS_FORMAT_TUP_SLOT format using device address space. Because tup-slot
 * format intends to store host accessable representation, we need to fix-
 * up pointers in the tuple store.
 * In case of any other format, we don't need to modify the data.
 */
STATIC_FUNCTION(void)
pg_fixup_tupslot_varlena(int *errcode,
						 kern_data_store *kds,
						 kern_data_store *ktoast,
						 cl_uint colidx, cl_uint rowidx)
{
	Datum		   *values;
	cl_bool		   *isnull;
	kern_colmeta	cmeta;

	/* only tuple-slot format needs to fixup pointer values */
	if (kds->format != KDS_FORMAT_SLOT)
		return;
	/* out of range? */
	if (rowidx >= kds->nitems || colidx >= kds->ncols)
		return;

	/* fixed length variable? */
	cmeta = kds->colmeta[colidx];
	if (cmeta.attlen > 0)
		return;

	values = KERN_DATA_STORE_VALUES(kds, rowidx);
	isnull = KERN_DATA_STORE_ISNULL(kds, rowidx);
	/* no need to care about NULL values */
	if (isnull[colidx])
		return;
	if (ktoast->format == KDS_FORMAT_ROW)
	{
		hostptr_t	offset = values[colidx];

		if (offset < ktoast->length)
		{
			values[colidx] = (Datum)((hostptr_t)ktoast->hostptr +
									 (hostptr_t)offset);
		}
		else
		{
			isnull[colidx] = true;
			values[colidx] = 0;
			STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
		}
	}
	else
		STROM_SET_ERROR(errcode, StromError_SanityCheckViolation);
}

/*
 * functions to reference variable length variables
 */
STROMCL_SIMPLE_DATATYPE_TEMPLATE(varlena, varlena *)

STATIC_FUNCTION(pg_varlena_t)
pg_varlena_datum_ref(int *errcode,
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_varlena_t)
pg_varlena_vref(kern_data_store *kds,
				kern_data_store *ktoast,
				int *errcode,
				cl_uint colidx,
				cl_uint rowidx)
{
	void   *datum = kern_get_datum(kds,ktoast,colidx,rowidx);

	return pg_varlena_datum_ref(errcode,datum,false);
}

STATIC_FUNCTION(void)
pg_varlena_vstore(kern_data_store *kds,
				  kern_data_store *ktoast,
				  int *errcode,
				  cl_uint colidx,
				  cl_uint rowidx,
				  pg_varlena_t datum)
{
	Datum  *daddr = pg_common_vstore(kds, ktoast, errcode,
									 colidx, rowidx, datum.isnull);
	if (daddr)
	{
		*daddr = (Datum)((char *)datum.value -
						 (char *)&ktoast->hostptr);
	}
	/*
	 * NOTE: pg_fixup_tupslot_varlena() shall be called, prior to
	 * the write-back of kern_data_store.
	 */
}

STATIC_FUNCTION(pg_varlena_t)
pg_varlena_param(kern_parambuf *kparams,
				 int *errcode,
				 cl_uint param_id)
{
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
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
		const cl_char  *__data = VARDATA_ANY(datum.value);
		cl_uint			__len = VARSIZE_ANY_EXHDR(datum.value);
		cl_uint			__index;

		while (__len-- > 0)
		{
			__index = (hash ^ *__data++) & 0xff;
			hash = crc32_table[__index] ^ (hash >> 8);
		}
	}
	return hash;
}

#define STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)				\
	typedef pg_varlena_t	pg_##NAME##_t;

#define STROMCL_VARLENA_VARREF_TEMPLATE(NAME)				\
	STATIC_INLINE(pg_##NAME##_t)							\
	pg_##NAME##_datum_ref(int *errcode,						\
						  void *datum,						\
						  cl_bool internal_format)			\
	{														\
		return pg_varlena_datum_ref(errcode,datum,			\
									internal_format);		\
	}														\
															\
	STATIC_INLINE(pg_##NAME##_t)							\
	pg_##NAME##_vref(kern_data_store *kds,					\
					 kern_data_store *ktoast,				\
					 int *errcode,							\
					 cl_uint colidx,						\
					 cl_uint rowidx)						\
	{														\
		void  *datum										\
			= kern_get_datum(kds,ktoast,colidx,rowidx);		\
		return pg_varlena_datum_ref(errcode,datum,false);	\
	}

#define STROMCL_VARLENA_VARSTORE_TEMPLATE(NAME)				\
	STATIC_INLINE(void)										\
	pg_##NAME##_vstore(kern_data_store *kds,				\
					   kern_data_store *ktoast,				\
					   int *errcode,						\
					   cl_uint colidx,						\
					   cl_uint rowidx,						\
					   pg_##NAME##_t datum)					\
	{														\
		return pg_varlena_vstore(kds,ktoast,errcode,		\
								 colidx,rowidx,datum);		\
	}

#define STROMCL_VARLENA_PARAMREF_TEMPLATE(NAME)						\
	STATIC_INLINE(pg_##NAME##_t)									\
	pg_##NAME##_param(kern_parambuf *kparams,						\
					  int *errcode, cl_uint param_id)				\
	{																\
		return pg_varlena_param(kparams,errcode,param_id);			\
	}

#define STROMCL_VARLENA_NULLTEST_TEMPLATE(NAME)						\
	STATIC_INLINE(pg_bool_t)										\
	pgfn_##NAME##_isnull(int *errcode, pg_##NAME##_t arg)			\
	{																\
		return pgfn_varlena_isnull(errcode, arg);					\
	}																\
	STATIC_INLINE(pg_bool_t)										\
	pgfn_##NAME##_isnotnull(int *errcode, pg_##NAME##_t arg)		\
	{																\
		return pgfn_varlena_isnotnull(errcode, arg);				\
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
 * arithmetic_stairlike_add
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
arithmetic_stairlike_add(cl_uint my_value, cl_uint *total_sum)
{
	cl_uint	   *items = SHARED_WORKMEM(cl_uint);
	size_t		unitsz = get_local_size();
	cl_int		i, j;

	/* set initial value */
	items[get_local_id()] = my_value;
	__syncthreads();

	for (i=1; unitsz > 0; i++, unitsz >>= 1)
	{
		/* index of last item in the earlier half of each 2^i unit */
		j = (get_local_id() & ~((1 << i) - 1)) | ((1 << (i-1)) - 1);

		/* add item[j] if it is later half in the 2^i unit */
		if ((get_local_id() & (1 << (i - 1))) != 0)
			items[get_local_id()] += items[j];
		__syncthreads();
	}
	if (total_sum)
		*total_sum = items[get_local_size() - 1];
	return items[get_local_id()] - my_value;
}

/*
 * kern_writeback_error_status
 *
 * It set thread local error code on the status variable on the global
 * memory, if status code is still StromError_Success and any of thread
 * has a particular error code.
 * NOTE: It does not look at significance of the error, so caller has
 * to clear its error code if it is a minor one.
 */
STATIC_FUNCTION(void)
kern_writeback_error_status(cl_int *error_status, int own_errcode)
{
	cl_int	   *error_temp = SHARED_WORKMEM(cl_int);
	size_t		unitsz;
	size_t		mask;
	size_t		buddy;
	cl_int		errcode_0;
	cl_int		errcode_1;
	cl_int		i;

	error_temp[get_local_id()] = own_errcode;
	__syncthreads();

	for (i=1, unitsz = get_local_size();
		 unitsz > 0;
		 i++, unitsz >>= 1)
	{
		mask = (1 << i) - 1;

		if ((get_local_id() & mask) == 0)
		{
			buddy = get_local_id() + (1 << (i - 1));

			errcode_0 = error_temp[get_local_id()];
			errcode_1 = (buddy < get_local_size()
						 ? error_temp[buddy]
						 : StromError_Success);
			if (errcode_0 == StromError_Success &&
				errcode_1 != StromError_Success)
				error_temp[get_local_id()] = errcode_1;
		}
		__syncthreads();
	}

	/*
	 * It writes back a statement level error, unless no other workgroup
	 * put a significant error status.
	 * This atomic operation set an error code, if it is still
	 * StromError_Success.
	 */
	errcode_0 = error_temp[0];
	if (get_local_id() == 0)
		atomicCAS(error_status, StromError_Success, errcode_0);
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
STATIC_FUNCTION(pg_bool_t)
pgfn_bool_is_true(cl_int *errcode, pg_bool_t result)
{
	result.value = (!result.isnull && result.value);
	result.isnull = false;
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bool_is_not_true(cl_int *errcode, pg_bool_t result)
{
	result.value = (result.isnull || !result.value);
	result.isnull = false;
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bool_is_false(cl_int *errcode, pg_bool_t result)
{
	result.value = (!result.isnull && !result.value);
	result.isnull = false;
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bool_is_not_false(cl_int *errcode, pg_bool_t result)
{
	result.value = (result.isnull || result.value);
	result.isnull = false;
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bool_is_unknown(cl_int *errcode, pg_bool_t result)
{
	result.value = result.isnull;
	result.isnull = false;
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bool_is_not_unknown(cl_int *errcode, pg_bool_t result)
{
	result.value = !result.isnull;
	result.isnull = false;
	return result;
}

/*
 * Functions for BoolOp (EXPR_AND and EXPR_OR shall be constructed on demand)
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_boolop_not(cl_int *errcode, pg_bool_t result)
{
	result.value = !result.value;
	/* if null is given, result is also null */
	return result;
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_COMMON_H */
