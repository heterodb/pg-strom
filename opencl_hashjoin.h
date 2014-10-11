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
 * | kern_multihash     |
 * | +------------------+
 * | | length           | <--- total length of multiple hash-tables; that
 * | +------------------+      also meand length to be send via DMA
 * | | ntables (=M)     | <--- number of hash-tables
 * | +------------------+
 * | | htbl_offset[0] o---> htbl_offset[0] is always NULL
 * | | htbl_offset[1] o------+
 * | |     :            |    |
 * | |     :            |    |
 * | | htbl_offset[M-1] |    |
 * +-+------------------+    |
 * |       :            |    |
 * +--------------------+    |
 * | kern_hashtable(0)  |    |
 * |       :            |    |
 * +--------------------+ <--+
 * | kern_hashtable(1)  |
 * |       :            |
 * +--------------------+
 * |       :            |
 * +--------------------+
 * | kern_hashtable(M-1)|
 * |       :            |
 * +--------------------+
 * | region for each    |
 * | kern_hashentry     |
 * | items              |
 * |                    |
 * |                    |
 * +--------------------+
 *
 * +--------------------+
 * | kern_hashtable     |
 * | +------------------+
 * | | nslots (=N)      |
 * | +------------------+
 * | | nkeys (=M)       |
 * | +------------------+
 * | | colmeta[0]       |
 * | | colmeta[1]       |
 * | |    :             |
 * | | colmeta[M-1]     |
 * | +------------------+
 * | | hash_slot[0]     |
 * | | hash_slot[1]     |
 * | |     :            |
 * | | hash_slot[N-2] o-------+  single directioned link
 * | | hash_slot[N-1]   |     |  from the hash_slot[]
 * +-+------------------+ <---+
 * | kern_hashentry     |
 * | +------------------+
 * | | next      o------------+  If multiple entries
 * | +------------------+     |  has same hash value,
 * | | hash             |     |  these are linked.
 * | +------------------+     |
 * | | rowidx           |     |
 * | +------------------+     |
 * | | matched          |     |
 * | +------------------+     |
 * | | keydata:         |     |
 * | | nullmap[...]     |     |
 * | | values[...]      |     |
 * | |                  |     |
 * | | values are put   |     |
 * | | next to nullmap  |     |
 * +-+------------------+ <---+
 * | kern_hashentry     |
 * | +------------------+
 * | | next       o-----------> NULL
 * | +------------------+
 * | | hash             |
 * | +------------------+
 * | |      :           |
 * | |      :           |
 * +-+------------------+
 */
typedef struct
{
	cl_uint			next;	/* offset of the next */
	cl_uint			hash;	/* 32-bit hash value */
	cl_uint			rowid;	/* identifier of inner rows */
	cl_uint			t_len;	/* length of the tuple */
	HeapTupleHeaderData htup;	/* tuple of the inner relation */
} kern_hashentry;

typedef struct
{
	cl_uint			ncols;		/* number of inner relation's columns */
	cl_uint			nslots;		/* width of hash slot */
	cl_char			is_outer;	/* true, if outer join (not supported now) */
	cl_char			__padding__[7];	/* for 64bit alignment */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER];
} kern_hashtable;

typedef struct
{
	hostptr_t		hostptr;	/* address of this multihash on the host */
	cl_uint			pg_crc32_table[256];
	/* MEMO: Originally, we put 'pg_crc32_table[]' as a static array
	 * deployed on __constant memory region, however, a particular
	 * OpenCL runtime had (has?) a problem on references to values
	 * on __constant memory. So, we moved the 'pg_crc32_table' into
	 * __global memory area as a workaround....
	 */
	cl_uint			ntables;	/* number of hash tables (= # of inner rels) */
	cl_uint			htable_offset[FLEXIBLE_ARRAY_MEMBER];
} kern_multihash;

#define KERN_HASHTABLE(kmhash, depth)								\
	((__global kern_hashtable *)((__global char *)(kmhash) +		\
								 (kmhash)->htable_offset[(depth)]))
#define KERN_HASHTABLE_SLOT(khtable)								\
	((__global cl_uint *)((__global char *)(khtable)+				\
						  LONGALIGN(offsetof(kern_hashtable,		\
											 colmeta[(khtable)->ncols]))))
#define KERN_HASHENTRY_SIZE(khentry)								\
	LONGALIGN(offsetof(kern_hashentry, htup) + (khentry)->t_len)

static inline __global kern_hashentry *
KERN_HASH_FIRST_ENTRY(__global kern_hashtable *khtable, cl_uint hash)
{
	__global cl_uint *slot = KERN_HASHTABLE_SLOT(khtable);
	cl_uint		index = hash % khtable->nslots;

	if (slot[index] == 0)
		return NULL;
	return (__global kern_hashentry *)((__global char *) khtable +
									   slot[index]);
}

static inline __global kern_hashentry *
KERN_HASH_NEXT_ENTRY(__global kern_hashtable *khtable,
					 __global kern_hashentry *khentry)
{
	if (khentry->next == 0)
		return NULL;
	return (__global kern_hashentry *)((__global char *)khtable +
									   khentry->next);
}

/*
 * Hash-Joining using GPU/MIC acceleration
 *
 * It packs a kern_parambuf and kern_resultbuf structure within a continuous
 * memory ares, to transfer (usually) small chunk by one DMA call.
 *
 * +----------------+       -----
 * | kern_parambuf  |         ^
 * | +--------------+         |
 * | | length   o-------------------+
 * | +--------------+         |     | kern_resultbuf is located just after
 * | | nparams      |         |     | the kern_parambuf (because of DMA
 * | +--------------+         |     | optimization), so head address of
 * | | poffset[0]   |         |     | kern_gpuscan + parambuf.length
 * | | poffset[1]   |         |     | points kern_resultbuf.
 * | |    :         |         |     |
 * | | poffset[M-1] |         |     |
 * | +--------------+         |     |
 * | | variable     |         |     |
 * | | length field |         |     |
 * | | for Param /  |         |     |
 * | | Const values |         |     |
 * | |     :        |         |     |
 * +-+--------------+  -----  |  <--+
 * | kern_resultbuf |    ^    |
 * | +--------------+    |    |  Area to be sent to OpenCL device.
 * | | nrooms       |    |    |  Forward DMA shall be issued here.
 * | +--------------+    |    |
 * | | nitems       |    |    |
 * | +--------------+    |    |
 * | | errcode      |    V    V
 * | +--------------+ -----------
 * | | results[0]   |  Area to be written back from the OpenCL device.
 * | | results[1]   |  Reverse DMA shall be issued here.
 * | |     :        |
 * | | results[N-1] |  results[] array is only in-kernel
 * +-+--------------+
 *
 * Things to be written into result-buffer:
 * Unlike simple scan cases, GpuHahJoin generate a tuple combined from
 * two different relation stream; inner and outer. We invoke a kernel
 * for each row-/column-store of outer relation stream, so it is obvious
 * which row-/column-store is pointed by the result. However, the inner
 * relation stream, that is hashed on the table, may have multiple row-
 * column-stores within a hash-table. So, it takes 8bytes to identify
 * a particular tuple on inner-side (index of rcstore and offset in the
 * rcstore).
 * So, we expect the result buffer shall be used as follows:
 *   results[3 * i + 0] = rcs-index of inner side (upper 32bits of rowid)
 *   results[3 * i + 1] = row-offset of inner side (lower 32bits of rowid)
 *   results[3 * i + 2] = row-index of outer relation stream
 *
 * MEMO: In the future enhancement, we may set invalid inner identifier,
 * if not valid pair was not found, for LEFT OUTER JOIN, but not now.
 */
typedef struct
{
	kern_parambuf	kparams;
	/* also, resultbuf shall be placed next to the parambuf */
} kern_hashjoin;

#define KERN_HASHJOIN_LENGTH(khashjoin)						\
	KERN_HASHJOIN_PARAMBUF(khashjoin)->length
#define KERN_HASHJOIN_PARAMBUF(khashjoin)					\
	((__global kern_parambuf *)(&(khashjoin)->kparams))

#define KERN_HASHJOIN_PARAMBUF_LENGTH(khashjoin)			\
	STROMALIGN(KERN_HASHJOIN_PARAMBUF(khashjoin)->length)
#define KERN_HASHJOIN_RESULTBUF(khashjoin)								\
	((__global kern_resultbuf *)((__global char *)(khashjoin) +			\
								 KERN_HASHJOIN_PARAMBUF_LENGTH(khashjoin)))
#define KERN_HASHJOIN_RESULTBUF_LENGTH(khashjoin)						\
	STROMALIGN(offsetof(kern_resultbuf,									\
						results[KERN_HASHJOIN_RESULTBUF(khashjoin)->nrels * \
								KERN_HASHJOIN_RESULTBUF(khashjoin)->nrooms]))
#define KERN_HASHJOIN_DMA_SENDOFS(khashjoin)	\
	((uintptr_t)&(khashjoin)->kparams - (uintptr_t)khashjoin)
#define KERN_HASHJOIN_DMA_SENDLEN(khashjoin)	\
	(KERN_HASHJOIN_PARAMBUF_LENGTH(khashjoin) +	\
	 offsetof(kern_resultbuf, results[0]))
#define KERN_HASHJOIN_DMA_RECVOFS(khashjoin)	\
	((uintptr_t)KERN_HASHJOIN_RESULTBUF(khashjoin) - (uintptr_t)(khashjoin))
#define KERN_HASHJOIN_DMA_RECVLEN(khashjoin)	\
	(offsetof(kern_resultbuf,					\
		results[KERN_HASHJOIN_RESULTBUF(khashjoin)->nrels *	\
				KERN_HASHJOIN_RESULTBUF(khashjoin)->nrooms]))

#ifdef OPENCL_DEVICE_CODE
/*
 * gpuhashjoin_execute
 *
 * main routine of gpuhashjoin - it run hash-join logic on the supplied
 * hash-tables and kds/ktoast pair, then stores its result on the "results"
 * array. caller already acquires (n_matches * n_rels) slot from "results".
 */
static cl_uint
gpuhashjoin_execute(__private cl_int *errcode,
					__global kern_parambuf *kparams,
					__global kern_multihash *kmhash,
					__global kern_data_store *kds,
					__global kern_toastbuf *ktoast,
					size_t kds_index,
					__global cl_int *rbuffer);

/*
 * kern_gpuhashjoin_main
 *
 * entrypoint of kernel gpuhashjoin implementation. Its job can be roughly
 * separated into two portions; the first one is to count expected number
 * of matched items (that should be acquired on the kern_resultbuf), then
 * the second job is to store the hashjoin result - for more correctness,
 * it shall be done in gpuhashjoin_main automatically generated.
 * In case when the result buffer does not have sufficient space, it
 * returns StromError_DataStoreNoSpace to inform host system this hashjoin
 * needs larger result buffer.
 */
__kernel void
kern_gpuhashjoin_main(__global kern_hashjoin *khashjoin,
					  __global kern_multihash *kmhash,
					  __global kern_data_store *kds,
					  __global kern_toastbuf *ktoast,
					  __global kern_row_map *krowmap,
					  KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
{
	__global kern_parambuf  *kparams = KERN_HASHJOIN_PARAMBUF(khashjoin);
	__global kern_resultbuf *kresults = KERN_HASHJOIN_RESULTBUF(khashjoin);
	cl_int			errcode = StromError_Success;
	cl_uint			n_matches;
	cl_uint			offset;
	cl_uint			nitems;
	size_t			kds_index;
	__local cl_uint	base;

	/* sanity check - kresults must have sufficient width of slots for the
	 * required hash-tables within kern_multihash.
	 */
	if (kresults->nrels != kmhash->ntables + 1)
	{
		errcode = StromError_DataStoreCorruption;
		goto out;
	}

	/* In case when kern_row_map (krowmap) is given, it means all the items
	 * are not valid and some them have to be dealt as like invisible rows.
	 * krowmap is an array of valid row-index.
	 */
	if (!krowmap)
		kds_index = get_global_id(0);
	else if (get_global_id(0) < krowmap->nvalids)
		kds_index = (size_t) krowmap->rindex[get_global_id(0)];
	else
		kds_index = kds->nitems;	/* ensure this thread is out of range */

	/* 1st-stage: At first, we walks on the hash tables to count number of
	 * expected number of matched hash entries towards the items being in
	 * the kern_data_store; to be aquired later for writing back the results.
	 * Also note that a thread not mapped on a particular valid item in kds
	 * can be simply assumed n_matches == 0.
	 */
	if (kds_index < kds->nitems)
		n_matches = gpuhashjoin_execute(&errcode,
										kparams,
										kmhash,
										kds, ktoast,
										kds_index,
										NULL);
	else
		n_matches = 0;

	/*
	 * XXX - calculate total number of matched tuples being searched
	 * by this workgroup
	 */
	offset = arithmetic_stairlike_add(n_matches, LOCAL_WORKMEM, &nitems);

	/*
	 * XXX - allocation of result buffer. A tuple takes 2 * sizeof(cl_uint)
	 * to store pair of row-indexes.
	 * If no space any more, return an error code to retry later.
	 *
	 * use atomic_add(&kresults->nitems, nitems) to determine the position
	 * to write. If expected usage is larger than kresults->nrooms, it
	 * exceeds the limitation of result buffer.
	 *
	 * MEMO: we may need to re-define nrooms/nitems using 64bit variables
	 * to avoid overflow issues, but has special platform capability on
	 * 64bit atomic-write...
	 */
	if(get_local_id(0) == 0)
	{
		if (nitems > 0)
			base = atomic_add(&kresults->nitems, nitems);
		else
			base = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/* In case when (base + nitems) is larger than or equal to the nrooms,
	 * it means we don't have enough space to write back hash-join results
	 * to host-side. So, we have to tell the host code the provided
	 * kern_resultbuf didn't have enough space.
	 */
	if (base + nitems > kresults->nrooms)
	{
		errcode = StromError_DataStoreNoSpace;
		goto out;
	}

	/*
	 * 2nd-stage: we already know how many items shall be generated on
	 * this hash-join. So, all we need to do is to invoke auto-generated
	 * hash-joining function with a certain address on the result-buffer.
	 */
	if (n_matches > 0 && kds_index < kds->nitems)
	{
		__global cl_int	   *rbuffer
			= kresults->results + kresults->nrels * (base + offset);

		n_matches = gpuhashjoin_execute(&errcode,
										kparams,
										kmhash,
										kds, ktoast,
										kds_index,
										rbuffer);
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode, LOCAL_WORKMEM);
}

/*
 * kern_gpuhashjoin_projection
 *
 *
 *
 */
static void
gpuhashjoin_projection_datum(__private cl_int *errcode,
							 __global kern_data_store *kds_dest,
							 size_t kds_index,
							 cl_int depth,
							 cl_int colidx,
							 __global hostptr_t *hostptr,
							 __global void *srcaddr);

__kernel void
kern_gpuhashjoin_projection(__global kern_hashjoin *khashjoin,	/* in */
							__global kern_multihash *kmhash,	/* in */
							__global kern_data_store *kds,		/* in */
							__global kern_toastbuf *ktoast,		/* in */
							__global kern_data_store *kds_dest,	/* out */
							KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
{
	__global kern_parambuf  *kparams = KERN_HASHJOIN_PARAMBUF(khashjoin);
	__global kern_resultbuf *kresults = KERN_HASHJOIN_RESULTBUF(khashjoin);
	__global cl_int	   *rbuffer;
	__global HeapTupleHeaderData *htup;
	__global hostptr_t *hostptr;
	__global void	   *addr;
	cl_int				depth, nrels;
	cl_int				i, ncols;
	cl_uint				offset;
	cl_uint				nattrs;
	cl_bool				heap_hasnull;
	cl_int				errcode = StromError_Success;

	/* Case of overflow; it shall be retried or executed by CPU instead,
	 * so no projection is needed anyway. We quickly exit the kernel.
	 * No need to set an error code because kern_gpuhashjoin_main()
	 * should already set it.
	 */
	if (kresults->nitems > kresults->nrooms ||
		kresults->nitems > kds_dest->nrooms)
	{
		STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);
		goto out;
	}
	/* Update the nitems of kds_dest */
	if (get_global_id(0) == 0)
		kds_dest->nitems = kresults->nitems;
	/* Do projection if thread is responsible */
	if (get_global_id(0) >= kresults->nitems)
		goto out;
	/* Ensure format of the kds_dest */
	if (kds_dest->format != KDS_FORMAT_TUPSLOT &&
		kds_dest->format == KDS_FORMAT_COLUMN)
	{
		STROM_SET_ERROR(&errcode, StromError_DataStoreCorruption);
		goto out;
	}

	/*
	 * Projection of the outer relation
	 * Note that outer relation (kds, ktoast) may have row or column
	 * format.
	 */
	rbuffer = kresults->results + nrels * get_global_id(0);
	if (kds->format == KDS_FORMAT_ROW)
	{
		cl_uint		i, ncols = kds->ncols;

		htup = kern_get_tuple_rs(kds, rbuffer[0]);
		if (htup)
		{
			offset = htup->t_hoff;
			nattrs = (htup->t_infomask2 & HEAP_NATTS_MASK);
			heap_hasnull = (htup->t_infomask & HEAP_HASNULL);

			for (i=0; i < ncols; i++)
			{
				if (i >= nattrs)
					addr = NULL;
				else if (heap_hasnull && att_isnull(i, htup->t_bits))
					addr = NULL;
				else
				{
					kern_colmeta	cmeta = kds->colmeta[i];

					if (cmeta.attlen > 0)
						offset = TYPEALIGN(cmeta.attalign, offset);
					else if (!VARATT_NOT_PAD_BYTE((__global char *)htup +
												  offset))
						offset = TYPEALIGN(cmeta.attalign, offset);

					addr = ((__global char *) htup + offset);
				}
				gpuhashjoin_projection_datum(&errcode,
											 kds_dest,
											 get_global_id(0),
											 0,	/* depth */
											 i,	/* colidx */
											 &kds->hostptr,
											 addr);
			}
		}
		else
		{
			/* fill up the slot with null */
			for (i=0; i < ncols; i++)
				gpuhashjoin_projection_datum(&errcode,
											 kds_dest,
											 get_global_id(0),
											 0,	/* depth */
											 i,	/* colidx */
											 NULL,	/* hostptr */
											 NULL);	/* addr */
		}
	}
	else if (kds->format == KDS_FORMAT_COLUMN)
	{
		cl_uint		i, ncols = kds->ncols;
		cl_uint		rowidx = rbuffer[0];

		for (i=0; i < ncols; i++)
		{
			addr = kern_get_datum_cs(kds, ktoast, i, rbuffer[0]);
			hostptr = (kds->colmeta[i].attlen < 0
					   ? &ktoast->hostptr
					   : &kds->hostptr);
			gpuhashjoin_projection_datum(&errcode,
										 kds_dest,
										 get_global_id(0),
										 0,	/* depth */
										 i,	/* colidx */
										 hostptr,
										 addr);
		}
	}
	else
	{
		STROM_SET_ERROR(&errcode, StromError_DataStoreCorruption);
		goto out;
	}

	/*
	 * Projection of each inner relations. Unlike outer relations,
	 * each rows are stores as row-format in kern_hashentry.
	 */
	for (depth=1; depth < nrels; depth++)
	{
		__global kern_hashtable *khtable = KERN_HASHTABLE(kmhash, depth);
		__global kern_hashentry *kentry;
		cl_uint		i, ncols = khtable->ncols;
		cl_uint		offset;

		kentry = (__global kern_hashentry *)
			((__global char *)khtable + rbuffer[depth]);
		htup = &kentry->htup;
		offset = htup->t_hoff;
		nattrs = (htup->t_infomask2 & HEAP_NATTS_MASK);
		heap_hasnull = (htup->t_infomask & HEAP_HASNULL);
		
		for (i=0; i < ncols; i++)
		{
			if (i >= nattrs)
				addr = NULL;
			else if (heap_hasnull && att_isnull(i, htup->t_bits))
				addr = NULL;
			else
			{
				if (khtable->colmeta[i].attlen > 0)
					offset = TYPEALIGN(khtable->colmeta[i].attlen, offset);
				else if (!VARATT_NOT_PAD_BYTE((__global char *)htup + offset))
					offset = TYPEALIGN(khtable->colmeta[i].attalign, offset);
				addr = ((__global char *) htup + offset);
			}
			gpuhashjoin_projection_datum(&errcode,
										 kds_dest,
										 get_global_id(0),
										 depth,
										 i,
										 &kmhash->hostptr,
										 addr);
		}
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode, LOCAL_WORKMEM);
}

/*
 * Helper routine of auto-generated gpuhashjoin_projection_datum()
 */
static inline void
gpuhashjoin_put_datum(__private cl_int *errcode,
					  __global kern_data_store *kds_dest,
					  size_t kds_index,
					  cl_uint colidx,
					  cl_int  typlen,
					  cl_bool typbyval,
					  __global hostptr_t *hostptr,
					  __global void *addr)
{
	if (kds_dest->format == KDS_FORMAT_TUPSLOT)
	{
		__global Datum   *values = KERN_DATA_STORE_VALUES(kds_dest, kds_index);
		__global cl_char *isnull = KERN_DATA_STORE_ISNULL(kds_dest, kds_index);

		if (!addr)
			isnull[colidx] = (cl_char) 1;
		else{
			isnull[colidx] = (cl_char) 0;
			if (typbyval)
			{
				if (typlen == sizeof(cl_char))
					values[colidx] = (Datum)(*((cl_char *)addr));
				else if (typlen == sizeof(cl_short))
					values[colidx] = (Datum)(*((cl_short *)addr));
				else if (typlen == sizeof(cl_int))
					values[colidx] = (Datum)(*((cl_int *)addr));
				else
					values[colidx] = (Datum)(*((cl_long *)addr));
			}
			else
			{
				values[colidx] = (Datum)(*hostptr + ((uintptr_t) addr -
													 (uintptr_t) hostptr));
			}
		}
	}
	else if (kds_dest->format == KDS_FORMAT_COLUMN)
	{
		kern_colmeta		cmeta = kds_dest->colmeta[colidx];
		__global void	   *destp;
		__global cl_uint   *nullmap;
		cl_uint				nullmask;
		cl_uint				offset;

		/* only null can be allowed to store value on invalid column */
		if (!cmeta.attvalid)
		{
			if (addr)
				STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
			return;
		}

		offset = cmeta.cs_offset;
		nullmap = ((__global cl_uint *)
				   ((__global cl_char *)kds_dest + offset)) + (kds_index >> 5);
		nullmask = (1U << (kds_index & 0x1f));
		if (!addr)
		{
			if (!cmeta.attnotnull)
				atomic_and(nullmap, ~nullmask);
			else
				STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
		}
		else
		{
			if (!cmeta.attnotnull)
			{
				atomic_or(nullmap, nullmask);
				offset += STROMALIGN(bitmaplen(kds_dest->nrooms));
			}
			if (cmeta.attlen > 0)
			{
				destp = ((__global cl_char *)kds_dest + offset +
						 cmeta.attlen * kds_index);
				if (typlen == sizeof(cl_char))
					*((cl_char *) destp) = *((cl_char *) addr);
				else if (typlen == sizeof(cl_short))
					*((cl_short *) destp) = *((cl_short *) addr);
				else if (typlen == sizeof(cl_int))
					*((cl_int *) destp) = *((cl_int *) addr);
				else if (typlen == sizeof(cl_long))
					*((cl_long *) destp) = *((cl_long *) addr);
				else
					memcpy(destp, addr, typlen);
			}
			else
			{
				destp = ((__global cl_char *)kds_dest + offset +
						 sizeof(cl_uint) * kds_index);
				*((cl_uint *) destp)
					= (cl_uint)((__global char *) addr -
								(__global char *) hostptr);
			}
		}
	}
	else
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
}

#define GHJOIN_PUT_DATUM(colidx,typlen,typbyval)	\
	gpuhashjoin_put_datum(errcode,					\
						  kds_dest,					\
						  kds_index,				\
						  (colidx),					\
						  (typlen),					\
						  (typbyval),				\
						  hostptr,					\
						  addr);

/*
 * Template of variable reference on the hash-entry
 */
#define STROMCL_SIMPLE_HASHREF_TEMPLATE(NAME,BASE)				\
	static pg_##NAME##_t										\
	pg_##NAME##_hashref(__global kern_hashtable *khtable,		\
						__global kern_hashentry *kentry,		\
						__private int *p_errcode,				\
						cl_uint colidx)							\
	{															\
		pg_##NAME##_t result;									\
		__global BASE *addr										\
			= kern_get_datum_tuple(khtable->colmeta,			\
								   &kentry->htup,				\
								   colidx);						\
		if (!addr)												\
			result.isnull = true;								\
		else													\
		{														\
			result.isnull = false;								\
			result.value = *addr;								\
		}														\
		return result;											\
	}

static pg_varlena_t
pg_varlena_hashref(__global kern_hashtable *khtable,
				   __global kern_hashentry *kentry,
				   __private int *p_errcode,
				   cl_uint colidx)
{
	pg_varlena_t	result;
	__global varlena *vl_ptr
		= kern_get_datum_tuple(khtable->colmeta,
							   &kentry->htup,
							   colidx);
	if (!vl_ptr)
		result.isnull = true;
	else if (VARATT_IS_4B_U(vl_ptr) || VARATT_IS_1B(vl_ptr))
	{
		result.value = vl_ptr;
		result.isnull = false;
	}
	else
	{
		result.isnull = true;
		STROM_SET_ERROR(p_errcode, StromError_CpuReCheck);
	}
	return result;
}

#define STROMCL_VARLENA_HASHREF_TEMPLATE(NAME)				\
	static pg_##NAME##_t									\
	pg_##NAME##_hashref(__global kern_hashtable *khtable,	\
						__global kern_hashentry *kentry,	\
						__private int *p_errcode,			\
						cl_uint colidx)						\
	{														\
		return pg_varlena_hashref(khtable, kentry,			\
								  p_errcode, colidx);		\
	}

/*
 * Macros to calculate hash key-value.
 * (logic was copied from pg_crc32.c)
 */
#define INIT_CRC32(crc)		((crc) = 0xFFFFFFFF)
#define FIN_CRC32(crc)		((crc) ^= 0xFFFFFFFF)

#define STROMCL_SIMPLE_HASHKEY_TEMPLATE(NAME,BASE)			\
	static inline cl_uint									\
	pg_##NAME##_hashkey(__global kern_multihash *kmhash,	\
						cl_uint hash, pg_##NAME##_t datum)	\
	{														\
		__global const cl_uint *crc32_table					\
			= kmhash->pg_crc32_table;						\
		if (!datum.isnull)									\
		{													\
			BASE		__data = datum.value;				\
			cl_uint		__len = sizeof(BASE);				\
			cl_uint		__index;							\
															\
			while (__len-- > 0)								\
			{												\
				__index = ((hash >> 24) ^ (__data)) & 0xff;	\
				hash = crc32_table[__index] ^ (hash << 8);	\
				__data = (__data >> 8);						\
			}												\
		}													\
		return hash;										\
	}

#define STROMCL_VARLENA_HASHKEY_TEMPLATE(NAME)				\
	static inline cl_uint									\
	pg_##NAME##_hashkey(__global kern_multihash *kmhash,	\
						cl_uint hash, pg_##NAME##_t datum)	\
	{														\
		__global const cl_uint *crc32_table					\
			= kmhash->pg_crc32_table;						\
		if (!datum.isnull)									\
		{													\
			__global const cl_char *__data =				\
				VARDATA_ANY(datum.value);					\
			cl_uint		__len = VARSIZE_ANY_EXHDR(datum.value); \
			cl_uint		__index;							\
			while (__len-- > 0)								\
			{												\
				__index = ((hash >> 24) ^ *__data++) & 0xff;\
				hash = crc32_table[__index] ^ (hash << 8);	\
			}												\
		}													\
		return hash;										\
	}

#else	/* OPENCL_DEVICE_CODE */

typedef struct pgstrom_multihash_tables
{
	StromObject		sobj;		/* = StromTab_HashJoinTable */
	cl_uint			maxlen;		/* max available length (also means size
								 * of allocated shared memory region) */
	cl_uint			length;		/* total usage of allocated shmem
								 * (also means length of DMA send) */
	slock_t			lock;		/* protection of the fields below */
	cl_int			refcnt;		/* reference counter of this hash table */
	cl_int			dindex;		/* device to load the hash table */
	cl_int			n_kernel;	/* number of active running kernel */
	cl_mem			m_hash;		/* in-kernel buffer object. Once n_kernel
								 * backed to zero, valid m_hash needs to
								 * be released. */
	cl_event		ev_hash;	/* event to load hash table to kernel */
	kern_multihash	kern;
} pgstrom_multihash_tables;

typedef struct
{
	pgstrom_message	msg;		/* = StromTag_GpuHashJoin */
	Datum			dprog_key;	/* device key for gpuhashjoin */
	pgstrom_multihash_tables *mhtables;	/* inner hashjoin tables */
	pgstrom_data_store *pds;	/* data store of outer relation */
	kern_hashjoin  *khashjoin;	/* a pair of kparams/kresults */
	kern_row_map	krowmap;	/* valid row mapping */
} pgstrom_gpuhashjoin;

#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_HASHJOIN_H */
