/*
 * opencl_hashjoin.h
 *
 * Parallel hash join accelerated by OpenCL device
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef OPENCL_HASHJOIN_H
#define OPENCL_HASHJOIN_H

/*
 * Format of kernel hash table; to be prepared
 *
 * +--------------------+
 * | kern_gpuhash_table |  total length of hash table incl.
 * | +------------------+  hash entries; size of DMA send
 * | | length        o---------------------------------+
 * | +------------------+                              |
 * | | nslots (=N)      |                              |
 * | +------------------+                              |
 * | | nkeys (=M)       |                              |
 * | +------------------+                              |
 * | | colmeta[0]       |                              |
 * | | colmeta[1]       |                              |
 * | |    :             |                              |
 * | | colmeta[M-1]     |                              |
 * | +------------------+                              |
 * | | hash_slot[0]     |                              |
 * | | hash_slot[1]     |                              |
 * | |     :            |                              |
 * | | hash_slot[N-2] o-------+  single directioned    |
 * | | hash_slot[N-1]   |     |  link from the         |
 * +-+------------------+ <---+  hash_slot[]           |
 * | kern_gpuhash_entry |                              |
 * | +------------------+                              |
 * | | next       o-----------+  If multiple entries   |
 * | +------------------+     |  has same hash value,  |
 * | | hash             |     |  these are linked.     |
 * | +------------------+     |                        |
 * | | keylen           |     |                        |
 * | +------------------+     |                        |
 * | | keydata:         |     |                        |
 * | | (actual values   |     |                        |
 * | |  to be joined)   |     |                        |
 * +-+------------------+ <---+                        |
 * | kern_gpuhash_entry |                              |
 * | +-----------------+-                              |
 * | | next       o-----------+                        |
 * | +------------------+     |                        |
 * | | hash             |     |                        |
 * | +------------------+     |                        |
 * | |      :           |     |                        |
 * |        :           |     |                        |
 * | |      :           |     |                        |
 * +-+------------------+ <---+                        |
 * | kern_gpuhash_entry |                              |
 * | +------------------+                              |
 * | | next             |                              |
 * | +------------------+                              |
 * | | hash             |                              |
 * | +------------------+                              |
 * | | keylen           |                              |
 * | +------------------+                              |
 * | | keydata          |                              |
 * | | +----------------+                              |
 * | | | nullmap(16bit) |                              |
 * | | | keydata:       |                              |
 * | | | (actual values |                              |
 * | | |  to be joined) |                              |
 * +-+------------------+  <---------------------------+
 */
typedef struct
{
	cl_uint			next;	/* offset of the next */
	cl_uint			hash;	/* 32-bit hash value */
	cl_ushort		keylen;	/* length of key data */
	cl_char			keydata[FLEXIBLE_ARRAY_MEMBER];
} kern_hash_entry;

typedef struct
{
	cl_uint			length;	/* total length of hash table */
	cl_uint			nslots;	/* width of hash slot */
	cl_uint			nkeys;	/* number of keys to be compared */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER];
} kern_hash_table;


/*
 * Sequential Scan using GPU/MIC acceleration
 *
 * It packs a kern_parambuf and kern_resultbuf structure within a continuous
 * memory ares, to transfer (usually) small chunk by one DMA call.
 *
 * +----------------+       -----
 * | kern_parambuf  |         ^
 * | +--------------+         |
 * | | length   o--------------------+
 * | +--------------+         |      | kern_resultbuf is located just after
 * | | nparams      |         |      | the kern_parambuf (because of DMA
 * | +--------------+         |      | optimization), so head address of
 * | | poffset[0]   |         |      | kern_gpuscan + parambuf.length
 * | | poffset[1]   |         |      | points kern_resultbuf.
 * | |    :         |         |      |
 * | | poffset[M-1] |         |      |
 * | +--------------+         |      |
 * | | variable     |         |      |
 * | | length field |         |      |
 * | | for Param /  |         |      |
 * | | Const values |         |      |
 * | |     :        |         |      |
 * +-+--------------+  -----  |  <---+
 * | kern_resultbuf |    ^    |
 * | +--------------+    |    |  Area to be sent to OpenCL device.
 * | | nrooms       |    |    |  Forward DMA shall be issued here.
 * | +--------------+    |    |
 * | | nitems       |    |    |
 * | +--------------+    |    |
 * | | debug_usage  |    |    |
 * | +--------------+    |    |
 * | | errcode      |    |    V
 * | +--------------+    |  -----
 * | | results[0]   |    |
 * | | results[1]   |    |  Area to be written back from OpenCL device.
 * | |     :        |    |  Reverse DMA shall be issued here.
 * | | results[N-1] |    V
 * +-+--------------+  -----
 */
typedef struct
{
	kern_parambuf	kparam;
	/* also, resultbuf shall be placed next to the parambuf */
} kern_hash_join;

#define KERN_GPUHJ_PARAMBUF(kghashjoin)					\
	((__global kern_parambuf *)(&(kghashjoin)->kparam))
#define KERN_GPUHJ_PARAMBUF_LENGTH(kghashjoin)			\
	STROMALIGN(KERN_GPUHJ_PARAMBUF(kghashjoin)->length)
#define KERN_GPUHJ_RESULTBUF(kghashjoin)								\
	((__global kern_resultbuf *)((__global char *)kghashjoin +			\
								 KERN_GPUHJ_PARAMBUF_LENGTH(kghashjoin))
#define KERN_GPUHJ_RESULTBUF_LENGTH(kghashjoin)							\
	STROMALIGN(offsetof(kern_resultbuf,									\
						results[KERN_GPUHJ_RESULTBUF(kghashjoin)->nrooms]))
#define KERN_GPUHJ_DMA_SENDLEN(kghashjoin)		\
	(KERN_GPUHJ_PARAMBUF_LENGTH(kghashjoin) +	\
	 offsetof(kern_resultbuf, results[0]))
#define KERN_GPUHJ_DMA_RECVLEN(kghashjoin)		\
	(offsetof(kern_resultbuf,					\
			  results[KERN_GPUHJ_RESULTBUF(kghashjoin)->nrooms]))



#ifdef OPENCL_DEVICE_CODE
/*
 * forward declaration of run-time generated functions.
 */

/*
 * gpuhashjoin_hashkey
 *
 * It generates a hash-key of the specified row in the column-store.
 */
static cl_uint
gpuhashjoin_hashkey(__private cl_int *errcode,
					__global kern_parambuf *kparam,
					__global kern_column_store *kcs,
					__global kern_toastbuf *ktoast,
					size_t row_index);
/*
 * gpuhashjoin_keycomp
 *
 * It compares a hash-item with the specified row in the column-store.
 */
static cl_bool
gpuhashjoin_keycomp(__private cl_int *errcode,
					__global kern_hash_item *hitem,
					__global kern_parambuf *kparam,
					__global kern_column_store *kcs,
					__global kern_toastbuf *ktoast,
					size_t row_index);
/*
 * gpuhashjoin_qual_eval
 *
 * It preprocesses the specified row according to the qualifier, if
 * GpuHashJoin pulled-up device executable qualifiers from underlying
 * scan plan.
 */
static pg_bool_t
gpuhashjoin_qual_eval(__private cl_int *errcode,
					  __global kern_parambuf *kparam,
					  __global kern_column_store *kcs,
					  __global kern_toastbuf *ktoast,
					  size_t row_index);

static void
gpuhashjoin_inner(__private cl_int *errcode,
				  __global kern_parambuf *kparam,
				  __global kern_resultbuf *kresult,
				  __global kern_hash_table *khashtbl,
				  __global kern_column_store *kcs,
				  __global kern_toastbuf *ktoast,
				  __local void *local_workbuf)
{
	pg_bool_t	rc;
	cl_int		errcode = StromError_Success;

	/* generate a hash key */
	hashkey = gpuhashjoin_hashkey(get_global_id(0));

	/* step.0: check qualifier around kcs, to check it should be filtered */
	if (get_global_id(0) < kcs->nrows)
		rc = gpuhashjoin_qual_eval(&errcode,
								   kparam, kcs, ktoast,
								   get_global_id(0));
	else
		rc.isnull = true;

	/* step.1: walks on the hash slot and count number of matches */

	/* step.2: allocate result buffer for this workgroup */
	if (get_local_id(0) == 0)
	{}
	/* if overflow, we shall return without writing */

	/* step.3: again, walks on the hash slot again, and put pair of ids */
	
}


__kernel void
gpuhashjoin_inner_cs(__global kern_hash_join *kgpuhj,
					 __global kern_hash_table *khashtbl,
					 __global kern_column_store *kcs,
					 __global kern_toastbuf *toast,
					 __local void *local_workmem)
{


}

__kernel void
gpuhashjoin_inner_rs(__global kern_hash_join *kgpuhj,
					 __global kern_hash_table *khashtbl,
					 __global kern_row_store *krs,
					 __global kern_column_store *kcs,
					 __local void *local_workmem)
{
	__global kern_parambuf *kparams = KERN_GPUHJ_PARAMBUF(kgpuhj);
	__global kern_resultbuf *kresult = KERN_GPUHJ_RESULTBUF(kgpuhj);
	pg_bytea_t		kparam_0 = pg_bytea_param(kparams,&errcode,0);
	cl_int			errcode = StromError_Success;
	__local size_t	kcs_offset;
	__local size_t	kcs_nitems;

	/* acquire a slot of this workgroup */
	if (get_local_id(0) == 0)
	{
		if (get_global_id(0) + get_local_size(0) < krs->nrows)
			kcs_nitems = get_local_size(0);
		else if (get_global_id(0) < krs->nrows)
			kcs_nitems = krs->nrows - get_global_id(0);
		else
			kcs_nitems = 0;
		kcs_offset = atomic_add(&kcs->nrows, kcs_nitems);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/* move data into column store */
	kern_row_to_column(&errcode,
					   (__global cl_char *)VARDATA(kparam_0.value),
					   krs,
					   get_global_id(0),
					   kcs,
					   NULL,
					   kcs_offset,
					   kcs_nitems,
					   local_workmem);
	/* OK, run gpu hash join */
	gpuhashjoin_inner(&errcode,
					  kparams,
					  kresult,
					  khashtbl,
					  kcs,
					  (__global kern_toastbuf *)krs,
					  kcs_offset + get_local_id(0),
					  local_workbuf);
	/* back execution status into host-side */
	kern_writeback_error_status(&kresult->errcode, errcode, local_workbuf);
}


#endif

typedef struct
{
	pgstrom_message		msg;	/* = StromTag_GpuHashTable */
	kern_hash_table		kern;
} pgstrom_gpu_hash_table;


typedef struct
{
	pgstrom_message		msg;	/* = StromTag_GpuHashJoin */
	Datum				dprog_key;
	StromObject		   *rc_store;
	kern_hash_join		kern;
} pgstrom_gpu_hash_join;




#endif	/* OPENCL_HASHJOIN_H */
