/*
 * opencl_hashjoin.h
 *
 * Parallel hash join accelerated by OpenCL device
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
	cl_uint			length;		/* length of this hashtable chunk */
	cl_uint			ncols;		/* number of inner relation's columns */
	cl_uint			nslots;		/* width of hash slot */
	cl_char			is_outer;	/* true, if outer join (not supported now) */
	cl_char			__padding__[3];	/* for 64bit alignment */
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
 *
 *
 * +-+-----------------+ ---
 * | kern_parambuf     |  ^
 * | +-----------------+  | Region to be sent to the m_join device memory
 * | | length          |  | 
 * | +-----------------+  |
 * | | nparams         |  |
 * | +-----------------+  |
 * | | poffset[0]      |  |
 * | | poffset[1]      |  |
 * | |    :            |  |
 * | | poffset[M-1]    |  |
 * | +-----------------+  |
 * | | variable length |  |
 * | | fields for      |  |
 * | | Param / Const   |  |
 * | |     :           |  |
 * +-------------------+ -|----
 * | kern_resultbuf    |  |  ^
 * |(only fixed fields)|  |  | Region to be written back from the device
 * | +-----------------+  |  | memory to the host-side
 * | | nrels           |  |  |
 * | +-----------------+  |  |
 * | | nrooms          |  |  |
 * | +-----------------+  |  |
 * | | nitems          |  |  |
 * | +-----------------+  |  |
 * | | errcode         |  |  |
 * | +-----------------+  |  |
 * | | has_recheckes   |  |  |
 * | +-----------------+  |  |
 * | | __padding__[]   |  |  V
 * +-+-----------------+ ------
 * | kern_row_map      |  ^  Region to be sent to the m_rowmap device memory,
 * | +-----------------+  |  on demand.
 * | | nvalids         |  |
 * | +-----------------+  |
 * | | rindex[0]       |  |
 * | | rindex[1]       |  |
 * | |   :             |  |
 * | | rindex[N-1]     |  V
 * +-+-----------------+ ---
 */
typedef struct
{
	kern_parambuf	kparams;
} kern_hashjoin;

#define KERN_HASHJOIN_PARAMBUF(khashjoin)					\
	((__global kern_parambuf *)(&(khashjoin)->kparams))
#define KERN_HASHJOIN_PARAMBUF_LENGTH(khashjoin)			\
	STROMALIGN(KERN_HASHJOIN_PARAMBUF(khashjoin)->length)
#define KERN_HASHJOIN_RESULTBUF(khashjoin)					\
	((__global kern_resultbuf *)							\
	 ((__global char *)KERN_HASHJOIN_PARAMBUF(khashjoin) +	\
	  KERN_HASHJOIN_PARAMBUF_LENGTH(khashjoin)))
#define KERN_HASHJOIN_RESULTBUF_LENGTH(khashjoin)			\
	STROMALIGN(offsetof(kern_resultbuf, results[0]))
#define KERN_HASHJOIN_ROWMAP(khashjoin)						\
	((__global kern_row_map *)								\
	 ((__global char *)KERN_HASHJOIN_RESULTBUF(khashjoin) +	\
	  KERN_HASHJOIN_RESULTBUF_LENGTH(khashjoin)))
#define KERN_HASHJOIN_ROWMAP_LENGTH(khashjoin)		\
	(KERN_HASHJOIN_ROWMAP(khashjoin)->nvalids < 0 ?	\
	 STROMALIGN(offsetof(kern_row_map, rindex[0])) :\
	 STROMALIGN(offsetof(kern_row_map,				\
				rindex[KERN_HASHJOIN_ROWMAP(khashjoin)->nvalids])))
#define KERN_HASHJOIN_DMA_SENDPTR(khashjoin)	\
	KERN_HASHJOIN_PARAMBUF(khashjoin)
#define KERN_HASHJOIN_DMA_SENDOFS(khashjoin)		0UL
#define KERN_HASHJOIN_DMA_SENDLEN(khashjoin)		\
	((uintptr_t)KERN_HASHJOIN_ROWMAP(khashjoin) -	\
	 (uintptr_t)KERN_HASHJOIN_PARAMBUF(khashjoin))
#define KERN_HASHJOIN_DMA_RECVPTR(khashjoin)	\
	KERN_HASHJOIN_RESULTBUF(khashjoin)
#define KERN_HASHJOIN_DMA_RECVOFS(khashjoin)	\
	KERN_HASHJOIN_PARAMBUF_LENGTH(khashjoin)
#define KERN_HASHJOIN_DMA_RECVLEN(khashjoin)	\
	KERN_HASHJOIN_RESULTBUF_LENGTH(khashjoin)

#ifdef OPENCL_DEVICE_CODE

/*
 * gpuhashjoin_qual_eval
 *
 * simple evaluation of qualifier, if any
 */
static bool
gpuhashjoin_qual_eval(__private cl_int *errcode,
					  __global kern_parambuf *kparams,
					  __global kern_data_store *kds,
					  __global kern_data_store *ktoast,
					  size_t kds_index);

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
					__local cl_uint *crc32_table,
					__global kern_data_store *kds,
					__global kern_data_store *ktoast,
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
					  __global kern_data_store *ktoast,
					  __global kern_row_map   *krowmap,
					  KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
{
	__global kern_parambuf  *kparams = KERN_HASHJOIN_PARAMBUF(khashjoin);
	__global kern_resultbuf *kresults = KERN_HASHJOIN_RESULTBUF(khashjoin);
	cl_int			errcode = StromError_Success;
	cl_uint			n_matches;
	cl_uint			offset;
	cl_uint			nitems;
	size_t			kds_index;
	size_t			crc_index;
	__local cl_uint	base;
	__local cl_uint	crc32_table[256];

	/* sanity check - kresults must have sufficient width of slots for the
	 * required hash-tables within kern_multihash.
	 */
	if (kresults->nrels != kmhash->ntables + 1)
	{
		errcode = StromError_DataStoreCorruption;
		goto out;
	}

	/* move crc32 table to __local memory from __global memory.
	 *
	 * NOTE: calculation of hash value (based on crc32 in GpuHashJoin) is
	 * the core of calculation workload in the GpuHashJoin implementation.
	 * If we keep the master table is global memory, it will cause massive
	 * amount of computing core stall because of RAM access latency.
	 * So, we try to move them into local shared memory at the beginning.
	 */
	for (crc_index = get_local_id(0);
		 crc_index < 256;
		 crc_index += get_local_size(0))
	{
		crc32_table[crc_index] = kmhash->pg_crc32_table[crc_index];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

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

	/*
	 * 0th-stage: if gpuhashjoin pulled-up device executable qualifiers
	 * from the outer relation scan, we try to evaluate it on the outer
	 * relation (that is kern_data_store), then exclude this thread if
	 * record shouldn't be visible
	 */
	if (!gpuhashjoin_qual_eval(&errcode, kparams, kds, ktoast, kds_index))
		kds_index = kds->nitems;	/* ensure this thread is not visible */

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
										crc32_table,
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
										crc32_table,
										kds, ktoast,
										kds_index,
										rbuffer);
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode, LOCAL_WORKMEM);
}

/*
 * kern_gpuhashjoin_projection_xxx
 *
 *
 *
 */
static void
gpuhashjoin_projection_mapping(cl_int dest_colidx,
								__private cl_uint *src_depth,
								__private cl_uint *src_colidx);
static void
gpuhashjoin_projection_datum(__private cl_int *errcode,
							 __global Datum *slot_values,
							 __global cl_char *slot_isnull,
							 cl_int depth,
							 cl_int colidx,
							 hostptr_t hostaddr,
							 __global void *datum);

__kernel void
kern_gpuhashjoin_projection_row(__global kern_hashjoin *khashjoin,	/* in */
								__global kern_multihash *kmhash,	/* in */
								__global kern_data_store *kds,		/* in */
								__global kern_data_store *ktoast,	/* in */
								__global kern_data_store *kds_dest,	/* out */
								KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
{
	__global kern_resultbuf *kresults = KERN_HASHJOIN_RESULTBUF(khashjoin);
	__global cl_int	   *rbuffer;
	__global void	   *datum;
	cl_uint				nrels = kresults->nrels;
	cl_bool				heap_hasnull = false;
	cl_uint				t_hoff;
	cl_uint				t_len;
	cl_uint				data_len;
	cl_uint				required;
	cl_uint				offset;
	cl_uint				total_len;
	cl_uint				usage_head;
	__local cl_uint		usage_prev;
	cl_int				errcode = StromError_Success;

	/* update nitems of kds_dest. note that get_global_id(0) is not always
     * called earlier than other thread. So, we should not expect nitems
	 * of kds_dest is initialized.
	 */
	if (get_global_id(0) == 0)
		kds_dest->nitems = kresults->nitems;

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

	/* Ensure format of the kern_data_store (source/destination) */
	if ((kds->format != KDS_FORMAT_ROW &&
		 kds->format != KDS_FORMAT_ROW_FLAT) ||
		kds_dest->format != KDS_FORMAT_ROW_FLAT)
	{
		STROM_SET_ERROR(&errcode, StromError_DataStoreCorruption);
		goto out;
	}

	/* combination of rows in this join */
	rbuffer = kresults->results + nrels * get_global_id(0);

	/*
	 * Step.1 - compute length of the joined tuple
	 */
	if (get_global_id(0) < kresults->nitems)
	{
		cl_uint		i, ncols = kds_dest->ncols;

		/* t_len and ctid */
		required = offsetof(kern_tupitem, htup);

		/* estimation of data length */
		data_len = 0;
		for (i=0; i < ncols; i++)
		{
			kern_colmeta	cmeta = kds_dest->colmeta[i];
			cl_uint			depth;
			cl_uint			colidx;

			/* ask auto generated code */
			gpuhashjoin_projection_mapping(i, &depth, &colidx);

			if (depth == 0)
				datum = kern_get_datum(kds, ktoast, colidx, rbuffer[0] - 1);
			else if (depth < nrels)
			{
				__global kern_hashtable *khtable;
				__global kern_hashentry *kentry;

				khtable = KERN_HASHTABLE(kmhash, depth);
				kentry = (__global kern_hashentry *)
					((__global char *)khtable + rbuffer[depth]);

				datum = kern_get_datum_tuple(khtable->colmeta,
											 &kentry->htup,
											 colidx);
			}
			else
				datum = NULL;

			if (!datum)
				heap_hasnull = true;
			else
			{
				/* att_align_datum */
				if (cmeta.attlen > 0 || !VARATT_IS_1B(datum))
					data_len = TYPEALIGN(cmeta.attalign, data_len);
				/* att_addlength_datum */
				if (cmeta.attlen > 0)
					data_len += cmeta.attlen;
				else
					data_len += VARSIZE_ANY(datum);
			}
		}
		t_hoff = offsetof(HeapTupleHeaderData, t_bits);
		if (heap_hasnull)
			t_hoff += bitmaplen(ncols);
		if (kds->tdhasoid)
			t_hoff += sizeof(cl_uint);
		t_hoff = MAXALIGN(t_hoff);
		required += t_hoff + MAXALIGN(data_len);
	}
	else
		required = 0;

	/*
	 * Step.2 - takes advance usage counter of kds_dest->usage
	 */
	offset = arithmetic_stairlike_add(required, LOCAL_WORKMEM, &total_len);
	if (get_local_id(0) == 0)
	{
		if (total_len > 0)
			usage_prev = atomic_add(&kds_dest->usage, total_len);
		else
			usage_prev = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Check expected usage of the buffer */
	usage_head = (STROMALIGN(offsetof(kern_data_store,
									  colmeta[kds_dest->ncols])) +
				  STROMALIGN(sizeof(kern_blkitem) * kds_dest->maxblocks) +
				  STROMALIGN(sizeof(kern_rowitem) * kresults->nitems));
	if (usage_head + usage_prev + total_len > kds_dest->length)
	{
		errcode = StromError_DataStoreNoSpace;
		goto out;
	}

	/*
	 * Step.3 - construction of a heap-tuple
	 */
	if (required > 0)
	{
		__global HeapTupleHeaderData *htup;
		__global kern_rowitem *ritem;
		__global kern_tupitem *titem;
		cl_uint		htup_offset;
		cl_uint		i, ncols = kds_dest->ncols;
		cl_uint		curr;

		/* put index of heap-tuple */
		htup_offset = kds_dest->length - (usage_prev + offset + required);
		ritem = KERN_DATA_STORE_ROWITEM(kds_dest, get_global_id(0));
		ritem->htup_offset = htup_offset;

		/* build a heap-tuple */
		titem = (__global kern_tupitem *)
			((__global char *)kds_dest + htup_offset);
		titem->t_len = t_hoff + data_len;
		titem->t_self.ip_blkid.bi_hi = 0x1234;	/* InvalidBlockNumber */
		titem->t_self.ip_blkid.bi_lo = 0x5678;
		titem->t_self.ip_posid = 0x9876;		/* InvalidOffsetNumber */
		htup = &titem->htup;

		SET_VARSIZE(&htup->t_choice.t_datum, required);
		htup->t_choice.t_datum.datum_typmod = kds_dest->tdtypmod;
		htup->t_choice.t_datum.datum_typeid = kds_dest->tdtypeid;
		htup->t_ctid.ip_blkid.bi_hi = 0x2345;
		htup->t_ctid.ip_blkid.bi_lo = 0x6789;
		htup->t_ctid.ip_posid = 0x1369;
		htup->t_infomask2 = (ncols & HEAP_NATTS_MASK);
		htup->t_infomask = (heap_hasnull ? HEAP_HASNULL : 0);
		htup->t_hoff = t_hoff;
		curr = t_hoff;

		for (i=0; i < ncols; i++)
		{
			kern_colmeta	cmeta = kds_dest->colmeta[i];
			cl_uint			depth;
			cl_uint			colidx;

			/* ask auto generated code again */
			gpuhashjoin_projection_mapping(i, &depth, &colidx);

			if (depth == 0)
				datum = kern_get_datum(kds, ktoast, colidx, rbuffer[0] - 1);
			else
            {
				__global kern_hashtable *khtable;
				__global kern_hashentry *kentry;

				khtable = KERN_HASHTABLE(kmhash, depth);
				kentry = (__global kern_hashentry *)
					((__global char *)khtable + rbuffer[depth]);

				datum = kern_get_datum_tuple(khtable->colmeta,
											 &kentry->htup,
											 colidx);
			}

			/* put datum on the destination kds */
			if (!datum)
				htup->t_bits[i >> 3] &= ~(1 << (i & 0x07));
			else
			{
				if (cmeta.attlen > 0)
				{
					__global char *dest;

					while (TYPEALIGN(cmeta.attalign, curr) != curr)
						((__global char *)htup)[curr++] = 0;
					dest = (__global char *)htup + curr;

					switch (cmeta.attlen)
					{
						case sizeof(cl_char):
							*((__global cl_char *) dest)
								= *((__global cl_char *) datum);
							break;
						case sizeof(cl_short):
							*((__global cl_short *) dest)
								= *((__global cl_short *) datum);
							break;
						case sizeof(cl_int):
							*((__global cl_int *) dest)
							  = *((__global cl_int *) datum);
							break;
						case sizeof(cl_long):
							*((__global cl_long *) dest)
							  = *((__global cl_long *) datum);
							break;
						default:
							memcpy(dest, datum, cmeta.attlen);
							break;
					}
					curr += cmeta.attlen;
				}
				else
				{
					cl_uint		vl_len = VARSIZE_ANY(datum);

					/* put 0 and align here, if not a short varlena */
					if (!VARATT_IS_1B(datum))
					{
						while (TYPEALIGN(cmeta.attalign, curr) != curr)
							((__global char *)htup)[curr++] = 0;
					}
					memcpy((__global char *)htup + curr, datum, vl_len);
					curr += vl_len;
				}
				if (heap_hasnull)
					htup->t_bits[i >> 3] |= (1 << (i & 0x07));
			}
		}
		titem->t_len = curr;
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode, LOCAL_WORKMEM);
}

__kernel void
kern_gpuhashjoin_projection_slot(__global kern_hashjoin *khashjoin,	/* in */
								 __global kern_multihash *kmhash,	/* in */
								 __global kern_data_store *kds,		/* in */
								 __global kern_data_store *ktoast,	/* in */
								 __global kern_data_store *kds_dest, /* out */
								 KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
{
	__global kern_resultbuf *kresults = KERN_HASHJOIN_RESULTBUF(khashjoin);
	__global cl_int	   *rbuffer;
	__global Datum	   *slot_values;
	__global cl_char   *slot_isnull;
	cl_int				nrels = kresults->nrels;
	cl_int				depth;
	cl_int				errcode = StromError_Success;

	/* Update the nitems of kds_dest */
	if (get_global_id(0) == 0)
		kds_dest->nitems = kresults->nitems;

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
	/* Do projection if thread is responsible */
	if (get_global_id(0) >= kresults->nitems)
		goto out;
	/* Ensure format of the kern_data_store */
	if ((kds->format != KDS_FORMAT_ROW &&
		 kds->format != KDS_FORMAT_ROW_FLAT) ||
		kds_dest->format != KDS_FORMAT_TUPSLOT)
	{
		STROM_SET_ERROR(&errcode, StromError_DataStoreCorruption);
		goto out;
	}

	/* Extract each tuple and projection */
	rbuffer = kresults->results + nrels * get_global_id(0);
	slot_values = KERN_DATA_STORE_VALUES(kds_dest, get_global_id(0));
	slot_isnull = KERN_DATA_STORE_ISNULL(kds_dest, get_global_id(0));

	for (depth=0; depth < nrels; depth++)
	{
		__global HeapTupleHeaderData *htup;
		__global kern_colmeta  *p_colmeta;
		__global void		   *datum;
		__global char		   *baseaddr;
		hostptr_t				hostaddr;
		cl_uint					i, ncols;
		cl_uint					offset;
		cl_uint					nattrs;
		cl_bool					heap_hasnull;

		if (depth == 0)
		{
			ncols = kds->ncols;
			p_colmeta = kds->colmeta;
			if (kds->format == KDS_FORMAT_ROW)
			{
				__global kern_blkitem *bitem;
				cl_uint		blkidx;

				htup = kern_get_tuple_rs(kds, rbuffer[0] - 1, &blkidx);
				baseaddr = (__global char *)
					KERN_DATA_STORE_ROWBLOCK(kds, blkidx);
				bitem = KERN_DATA_STORE_BLKITEM(kds, blkidx);
				hostaddr = bitem->page;
			}
			else
			{
				htup = kern_get_tuple_rsflat(kds, rbuffer[0] - 1);
				baseaddr = (__global char *)&kds->hostptr;
				hostaddr = kds->hostptr;
			}
		}
		else
		{
			__global kern_hashtable *khtable = KERN_HASHTABLE(kmhash, depth);
			__global kern_hashentry *kentry;

			kentry = (__global kern_hashentry *)
				((__global char *)khtable + rbuffer[depth]);
			htup = &kentry->htup;

			ncols = khtable->ncols;
			p_colmeta = khtable->colmeta;
			baseaddr = (__global char *)&kmhash->hostptr;
			hostaddr = kmhash->hostptr;
		}

		/* fill up the slot with null */
		if (!htup)
		{
			for (i=0; i < ncols; i++)
				gpuhashjoin_projection_datum(&errcode,
											 slot_values,
											 slot_isnull,
											 depth,
											 i,
											 0,
											 NULL);
			continue;
		}
		offset = htup->t_hoff;
		nattrs = (htup->t_infomask2 & HEAP_NATTS_MASK);
		heap_hasnull = (htup->t_infomask & HEAP_HASNULL);

		for (i=0; i < ncols; i++)
		{
			if (i >= nattrs)
				datum = NULL;
			else if (heap_hasnull && att_isnull(i, htup->t_bits))
				datum = NULL;
			else
			{
				kern_colmeta	cmeta = p_colmeta[i];

				if (cmeta.attlen > 0)
					offset = TYPEALIGN(cmeta.attlen, offset);
				else if (!VARATT_NOT_PAD_BYTE((__global char *)htup + offset))
					offset = TYPEALIGN(cmeta.attalign, offset);

				datum = ((__global char *) htup + offset);
				offset += (cmeta.attlen > 0
						   ? cmeta.attlen
						   : VARSIZE_ANY(datum));
			}
			/* put datum */
			gpuhashjoin_projection_datum(&errcode,
										 slot_values,
										 slot_isnull,
										 depth,
										 i,
										 hostaddr + ((uintptr_t) datum -
													 (uintptr_t) baseaddr),
										 datum);
		}
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode, LOCAL_WORKMEM);
}

/*
 * Template of variable reference on the hash-entry
 */
#define STROMCL_ANYTYPE_HASHREF_TEMPLATE(NAME)						\
	pg_##NAME##_t													\
	pg_##NAME##_hashref(__global kern_hashtable *khtable,			\
						__global kern_hashentry *kentry,			\
						__private int *p_errcode,					\
						cl_uint colidx)								\
	{																\
		__global void *datum										\
			= kern_get_datum_tuple(khtable->colmeta,				\
								   &kentry->htup,					\
								   colidx);							\
		return pg_##NAME##_datum_ref(p_errcode, datum, false);		\
	}

#else	/* OPENCL_DEVICE_CODE */

typedef struct pgstrom_multihash_tables
{
	StromObject		sobj;		/* = StromTab_HashJoinTable */
	Size			length;		/* max available length of this mhash */
	Size			usage;		/* current usage of this mhash */
	double			ntuples;	/* number of tuples in this mhash */
	bool			is_divided;	/* true, if not whole of the inner relation */
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
	pgstrom_message		msg;		/* = StromTag_GpuHashJoin */
	Datum				dprog_key;	/* device key for gpuhashjoin */
	pgstrom_multihash_tables *mhtables;	/* inner hashjoin tables */
	pgstrom_data_store *pds;		/* data store of outer relation */
	pgstrom_data_store *pds_dest;	/* data store of result buffer */
	kern_hashjoin		khashjoin;		/* kern_hashjoin of this request */
} pgstrom_gpuhashjoin;

#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_HASHJOIN_H */
