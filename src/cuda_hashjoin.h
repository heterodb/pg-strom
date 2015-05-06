/*
 * cuda_hashjoin.h
 *
 * Parallel hash join accelerated by OpenCL device
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
#ifndef CUDA_HASHJOIN_H
#define CUDA_HASHJOIN_H

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
	((kern_hashtable *)((char *)(kmhash) +							\
						(kmhash)->htable_offset[(depth)]))
#define KERN_HASHTABLE_SLOT(khtable)								\
	((cl_uint *)((char *)(khtable) +								\
				 LONGALIGN(offsetof(kern_hashtable,					\
									colmeta[(khtable)->ncols]))))
#define KERN_HASHENTRY_SIZE(khentry)								\
	LONGALIGN(offsetof(kern_hashentry, htup) + (khentry)->t_len)

STATIC_INLINE(kern_hashentry *)
KERN_HASH_FIRST_ENTRY(kern_hashtable *khtable, cl_uint hash)
{
	cl_uint	   *slot = KERN_HASHTABLE_SLOT(khtable);
	cl_uint		index = hash % khtable->nslots;

	if (slot[index] == 0)
		return NULL;
	return (kern_hashentry *)((char *) khtable + slot[index]);
}

STATIC_INLINE(kern_hashentry *)
KERN_HASH_NEXT_ENTRY(kern_hashtable *khtable, kern_hashentry *khentry)
{
	if (khentry->next == 0)
		return NULL;
	return (kern_hashentry *)((char *)khtable + khentry->next);
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
 */
typedef struct
{
	kern_parambuf	kparams;
} kern_hashjoin;

#define KERN_HASHJOIN_PARAMBUF(khashjoin)			\
	((kern_parambuf *)(&(khashjoin)->kparams))
#define KERN_HASHJOIN_PARAMBUF_LENGTH(khashjoin)	\
	STROMALIGN(KERN_HASHJOIN_PARAMBUF(khashjoin)->length)
#define KERN_HASHJOIN_RESULTBUF(khashjoin)			\
	((kern_resultbuf *)((char *)KERN_HASHJOIN_PARAMBUF(khashjoin) +	\
						KERN_HASHJOIN_PARAMBUF_LENGTH(khashjoin)))
#define KERN_HASHJOIN_RESULTBUF_LENGTH(khashjoin)	\
	STROMALIGN(offsetof(kern_resultbuf, results[0]))
#define KERN_HASHJOIN_DMA_SENDPTR(khashjoin)		\
	KERN_HASHJOIN_PARAMBUF(khashjoin)
#define KERN_HASHJOIN_DMA_SENDOFS(khashjoin)		0UL
#define KERN_HASHJOIN_DMA_SENDLEN(khashjoin)		\
	(KERN_HASHJOIN_PARAMBUF_LENGTH(khashjoin) +		\
	 KERN_HASHJOIN_RESULTBUF_LENGTH(khashjoin))
#define KERN_HASHJOIN_DMA_RECVPTR(khashjoin)		\
	KERN_HASHJOIN_RESULTBUF(khashjoin)
#define KERN_HASHJOIN_DMA_RECVOFS(khashjoin)		\
	KERN_HASHJOIN_PARAMBUF_LENGTH(khashjoin)
#define KERN_HASHJOIN_DMA_RECVLEN(khashjoin)		\
	KERN_HASHJOIN_RESULTBUF_LENGTH(khashjoin)

#ifdef __CUDACC__

/*
 * gpuhashjoin_qual_eval
 *
 * simple evaluation of qualifier, if any
 */
STATIC_FUNCTION(bool)
gpuhashjoin_qual_eval(cl_int *errcode,
					  kern_parambuf *kparams,
					  kern_data_store *kds,
					  kern_data_store *ktoast,
					  size_t kds_index);

/*
 * gpuhashjoin_execute
 *
 * main routine of gpuhashjoin - it run hash-join logic on the supplied
 * hash-tables and kds/ktoast pair, then stores its result on the "results"
 * array. caller already acquires (n_matches * n_rels) slot from "results".
 */
STATIC_FUNCTION(cl_uint)
gpuhashjoin_execute(cl_int *errcode,
					kern_parambuf *kparams,
					kern_multihash *kmhash,
					cl_uint *crc32_table,	/* shared memory */
					kern_data_store *kds,
					kern_data_store *ktoast,
					size_t kds_index,
					cl_int *rbuffer);

/*
 * gpuhashjoin_projection_mapping/_datum
 *
 * support routine of projection
 */
STATIC_FUNCTION(void)
gpuhashjoin_projection_mapping(cl_int dest_colidx,
							   cl_uint *src_depth,
							   cl_uint *src_colidx);
STATIC_FUNCTION(void)
gpuhashjoin_projection_datum(cl_int *errcode,
							 Datum *slot_values,
							 cl_char *slot_isnull,
							 cl_int depth,
							 cl_int colidx,
							 hostptr_t hostaddr,
							 void *datum);
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
KERNEL_FUNCTION(void)
kern_gpuhashjoin_main(kern_hashjoin *khashjoin,
					  kern_multihash *kmhash,
					  kern_data_store *kds)
{
	kern_parambuf  *kparams = KERN_HASHJOIN_PARAMBUF(khashjoin);
	kern_resultbuf *kresults = KERN_HASHJOIN_RESULTBUF(khashjoin);
	cl_int			errcode = StromError_Success;
	cl_uint			n_matches;
	cl_uint			offset;
	cl_uint			nitems;
	size_t			kds_index = get_global_id();
	size_t			crc_index;
	__shared__ cl_uint base;
	__shared__ cl_uint crc32_table[256];

	/* sanity check - kresults must have sufficient width of slots for the
	 * required hash-tables within kern_multihash.
	 */
	if (kresults->nrels != kmhash->ntables + 1)
	{
		errcode = StromError_DataStoreCorruption + 1000;
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
	for (crc_index = get_local_id();
		 crc_index < 256;
		 crc_index += get_local_size())
	{
		crc32_table[crc_index] = kmhash->pg_crc32_table[crc_index];
	}
	__syncthreads();

	/*
	 * 0th-stage: if gpuhashjoin pulled-up device executable qualifiers
	 * from the outer relation scan, we try to evaluate it on the outer
	 * relation (that is kern_data_store), then exclude this thread if
	 * record shouldn't be visible
	 */
	if (!gpuhashjoin_qual_eval(&errcode, kparams, kds, NULL, kds_index))
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
										kds, NULL,
										kds_index,
										NULL);
	else
		n_matches = 0;

	/*
	 * XXX - calculate total number of matched tuples being searched
	 * by this workgroup
	 */
	offset = arithmetic_stairlike_add(n_matches, &nitems);

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
	if (get_local_id() == 0)
	{
		if (nitems > 0)
			base = atomicAdd(&kresults->nitems, nitems);
		else
			base = 0;
	}
	__syncthreads();

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
	    cl_int	   *rbuffer = KERN_GET_RESULT(kresults, base+offset);

		n_matches = gpuhashjoin_execute(&errcode,
										kparams,
										kmhash,
										crc32_table,
										kds, NULL,
										kds_index,
										rbuffer);
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode);
}

/*
 * kern_gpuhashjoin_projection_row
 *
 * It puts hashjoin results on the data-store with row-format.
 * This function takes __syncthreads()
 */
STATIC_FUNCTION(bool)
__gpuhashjoin_projection_row(cl_int *p_errcode,			/* in/out */
							 kern_resultbuf *kresults,	/* in */
							 size_t res_index,			/* in */
							 kern_multihash *kmhash,	/* in */
							 kern_data_store *kds_src,	/* in */
							 kern_data_store *kds_dst)	/* out */
{
	cl_int		   *rbuffer = KERN_GET_RESULT(kresults, res_index);
	void		   *datum;
	cl_uint			nrels = kresults->nrels;
	cl_bool			heap_hasnull = false;
	cl_uint			t_hoff;
	cl_uint			data_len;
	cl_uint			required;
	cl_uint			offset;
	cl_uint			total_len;
	__shared__ cl_uint usage_prev;

	/*
	 * Step.1 - compute length of the joined tuple
	 */
	if (res_index < kresults->nitems)
	{
		cl_uint		i, ncols = kds_dst->ncols;

		/* t_len and ctid */
		required = offsetof(kern_tupitem, htup);

		/* estimation of data length */
		data_len = 0;
		for (i=0; i < ncols; i++)
		{
			kern_colmeta	cmeta = kds_dst->colmeta[i];
			cl_uint			depth;
			cl_uint			colidx;

			/* ask auto generated code */
			gpuhashjoin_projection_mapping(i, &depth, &colidx);

			if (depth == 0)
				datum = kern_get_datum_row(kds_src, colidx, rbuffer[0] - 1);
			else if (depth < nrels)
			{
				kern_hashtable *khtable = KERN_HASHTABLE(kmhash, depth);
				kern_hashentry *kentry = (kern_hashentry *)
					((char *)khtable + rbuffer[depth]);

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
		if (kds_src->tdhasoid)
			t_hoff += sizeof(cl_uint);
		t_hoff = MAXALIGN(t_hoff);
		required += t_hoff + MAXALIGN(data_len);
	}
	else
		required = 0;

	/*
	 * Step.2 - takes advance usage counter of kds_dst->usage
	 */
	offset = arithmetic_stairlike_add(required, &total_len);
	if (get_local_id() == 0)
	{
		if (total_len > 0)
			usage_prev = atomicAdd(&kds_dst->usage, total_len);
		else
			usage_prev = 0;
	}
	__syncthreads();

	/* check expected usage of the buffer */
	if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
		STROMALIGN(sizeof(cl_uint) * kresults->nitems) +
		usage_prev + total_len > kds_dst->length)
	{
		*p_errcode = StromError_DataStoreNoSpace;
		return false;
	}

	/*
	 * Step.3 - construction of a heap-tuple
	 */
	if (required > 0)
	{
		HeapTupleHeaderData *htup;
		kern_tupitem   *titem;
		cl_uint		   *htup_index;
		cl_uint			htup_offset;
		cl_uint			i, ncols = kds_dst->ncols;
		cl_uint			curr;

		/* setup kern_tupitem */
		htup_offset = kds_dst->length - (usage_prev + offset + required);
		htup_index = (cl_uint *)KERN_DATA_STORE_BODY(kds_dst);
		htup_index[get_global_id()] = htup_offset;

		titem = (kern_tupitem *)((char *)kds_dst + htup_offset);
		titem->t_len = t_hoff + data_len;
		titem->t_self.ip_blkid.bi_hi = 0x1234;	/* InvalidBlockNumber */
		titem->t_self.ip_blkid.bi_lo = 0x5678;
		titem->t_self.ip_posid = 0x9876;		/* InvalidOffsetNumber */
		htup = &titem->htup;

		SET_VARSIZE(&htup->t_choice.t_datum, required);
		htup->t_choice.t_datum.datum_typmod = kds_dst->tdtypmod;
		htup->t_choice.t_datum.datum_typeid = kds_dst->tdtypeid;
		htup->t_ctid.ip_blkid.bi_hi = 0x2345;
		htup->t_ctid.ip_blkid.bi_lo = 0x6789;
		htup->t_ctid.ip_posid = 0x1369;
		htup->t_infomask2 = (ncols & HEAP_NATTS_MASK);
		htup->t_infomask = (heap_hasnull ? HEAP_HASNULL : 0);
		htup->t_hoff = t_hoff;
		curr = t_hoff;

		for (i=0; i < ncols; i++)
		{
			kern_colmeta	cmeta = kds_dst->colmeta[i];
			cl_uint			depth;
			cl_uint			colidx;

			/* ask auto generated code again */
			gpuhashjoin_projection_mapping(i, &depth, &colidx);

			if (depth == 0)
				datum = kern_get_datum_row(kds_src, colidx, rbuffer[0] - 1);
			else
            {
				kern_hashtable *khtable = KERN_HASHTABLE(kmhash, depth);
				kern_hashentry *kentry = (kern_hashentry *)
					((char *)khtable + rbuffer[depth]);

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
					char	   *dest;

					while (TYPEALIGN(cmeta.attalign, curr) != curr)
						((char *)htup)[curr++] = 0;
					dest = (char *)htup + curr;

					switch (cmeta.attlen)
					{
						case sizeof(cl_char):
							*((cl_char *) dest) = *((cl_char *) datum);
							break;
						case sizeof(cl_short):
							*((cl_short *) dest) = *((cl_short *) datum);
							break;
						case sizeof(cl_int):
							*((cl_int *) dest) = *((cl_int *) datum);
							break;
						case sizeof(cl_long):
							*((cl_long *) dest) = *((cl_long *) datum);
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
							((char *)htup)[curr++] = 0;
					}
					memcpy((char *)htup + curr, datum, vl_len);
					curr += vl_len;
				}
				if (heap_hasnull)
					htup->t_bits[i >> 3] |= (1 << (i & 0x07));
			}
		}
		titem->t_len = curr;
	}
	return true;
}

KERNEL_FUNCTION(void)
kern_gpuhashjoin_projection_row(kern_hashjoin *khashjoin,	/* in */
								kern_multihash *kmhash,		/* in */
								kern_data_store *kds_src,	/* in */
								kern_data_store *kds_dst)	/* out */
{
	kern_resultbuf *kresults = KERN_HASHJOIN_RESULTBUF(khashjoin);
	size_t			res_index;
	size_t			res_limit;
	cl_int			errcode = StromError_Success;

	/* update nitems of kds_dst. note that get_global_id(0) is not always
     * called earlier than other thread. So, we should not expect nitems
	 * of kds_dst is initialized.
	 */
	if (get_global_id() == 0)
		kds_dst->nitems = kresults->nitems;

	/* Case of overflow; it shall be retried or executed by CPU instead,
	 * so no projection is needed anyway. We quickly exit the kernel.
	 * No need to set an error code because kern_gpuhashjoin_main()
	 * should already set it.
	 */
	if (kresults->nitems > kresults->nrooms ||
		kresults->nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);
		goto out;
	}

	/* Ensure format of the kern_data_store (source/destination) */
	if (kds_src->format != KDS_FORMAT_ROW ||
		kds_dst->format != KDS_FORMAT_ROW)
	{
		STROM_SET_ERROR(&errcode, StromError_DataStoreCorruption + 2000);
		goto out;
	}

	/* Do projection if thread is responsible */
	res_limit = ((kresults->nitems + get_local_size() - 1) /
				 get_local_size()) * get_local_size();
	for (res_index = get_global_id();
		 res_index < res_limit;
		 res_index += get_global_size())
	{
		if (!__gpuhashjoin_projection_row(&errcode, kresults, res_index,
										  kmhash, kds_src, kds_dst))
			break;
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode);
}

/*
 * kern_gpuhashjoin_projection_slot
 *
 * It puts hashjoin results on the data-store with slot format
 */
STATIC_FUNCTION(bool)
__gpuhashjoin_projection_slot(cl_int *p_errcode,		/* in/out */
							  kern_resultbuf *kresults,	/* in */
							  size_t res_index,			/* in */
							  kern_multihash *kmhash,	/* in */
							  kern_data_store *kds_src,	/* in */
							  kern_data_store *kds_dst) /* out */
{
	cl_int	   *rbuffer = KERN_GET_RESULT(kresults, res_index);
	Datum	   *slot_values = KERN_DATA_STORE_VALUES(kds_dst, res_index);
	cl_char	   *slot_isnull = KERN_DATA_STORE_ISNULL(kds_dst, res_index);
	cl_int		nrels = kresults->nrels;
	cl_int		depth;

	for (depth=0; depth < nrels; depth++)
	{
		HeapTupleHeaderData *htup;
		kern_colmeta  *p_colmeta;
		void		   *datum;
		char		   *baseaddr;
		hostptr_t		hostaddr;
		cl_uint			i, ncols;
		cl_uint			offset;
		cl_uint			nattrs;
		cl_bool			heap_hasnull;

		if (depth == 0)
		{
			ncols = kds_src->ncols;
			p_colmeta = kds_src->colmeta;

			htup = kern_get_tuple_row(kds_src, rbuffer[0] - 1);
			baseaddr = (char *)&kds_src->hostptr;
			hostaddr = kds_src->hostptr;
		}
		else
		{
			kern_hashtable *khtable = KERN_HASHTABLE(kmhash, depth);
			kern_hashentry *kentry;

			kentry = (kern_hashentry *)((char *)khtable + rbuffer[depth]);
			htup = &kentry->htup;

			ncols = khtable->ncols;
			p_colmeta = khtable->colmeta;
			baseaddr = (char *)&kmhash->hostptr;
			hostaddr = kmhash->hostptr;
		}

		/* fill up the slot with null */
		if (!htup)
		{
			for (i=0; i < ncols; i++)
				gpuhashjoin_projection_datum(p_errcode,
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
				else if (!VARATT_NOT_PAD_BYTE((char *)htup + offset))
					offset = TYPEALIGN(cmeta.attalign, offset);

				datum = ((char *) htup + offset);
				offset += (cmeta.attlen > 0
						   ? cmeta.attlen
						   : VARSIZE_ANY(datum));
			}
			/* put datum */
			gpuhashjoin_projection_datum(p_errcode,
										 slot_values,
										 slot_isnull,
										 depth,
										 i,
										 hostaddr + ((char *) datum -
													 (char *) baseaddr),
										 datum);
		}
	}
	return true;
}

KERNEL_FUNCTION(void)
kern_gpuhashjoin_projection_slot(kern_hashjoin *khashjoin,	/* in */
								 kern_multihash *kmhash,	/* in */
								 kern_data_store *kds_src,	/* in */
								 kern_data_store *kds_dst) /* out */
{
	kern_resultbuf *kresults = KERN_HASHJOIN_RESULTBUF(khashjoin);
	size_t		res_index;
	size_t		res_limit;
	cl_int		errcode = StromError_Success;

	/* Update the nitems of kds_dst */
	if (get_global_id() == 0)
		kds_dst->nitems = kresults->nitems;

	/* Case of overflow; it shall be retried or executed by CPU instead,
	 * so no projection is needed anyway. We quickly exit the kernel.
	 * No need to set an error code because kern_gpuhashjoin_main()
	 * should already set it.
	 */
	if (kresults->nitems > kresults->nrooms ||
		kresults->nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);
		goto out;
	}

	/* Ensure format of the kern_data_store */
	if (kds_src->format != KDS_FORMAT_ROW ||
		kds_dst->format != KDS_FORMAT_SLOT)
	{
		STROM_SET_ERROR(&errcode, StromError_DataStoreCorruption + 3000);
		goto out;
	}

	/* Do projection if thread is responsible */
	res_limit = kresults->nitems;
	for (res_index = get_global_id();
		 res_index < res_limit;
		 res_index += get_global_size())
	{
		if (!__gpuhashjoin_projection_slot(&errcode, kresults, res_index,
										   kmhash, kds_src, kds_dst))
			break;
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode);
}

/*
 * Template of variable reference on the hash-entry
 */
#define STROMCL_ANYTYPE_HASHREF_TEMPLATE(NAME)						\
	STATIC_FUNCTION(pg_##NAME##_t)									\
	pg_##NAME##_hashref(kern_hashtable *khtable,					\
						kern_hashentry *kentry,						\
						int *p_errcode,								\
						cl_uint colidx)								\
	{																\
		void *datum = kern_get_datum_tuple(khtable->colmeta,		\
										   &kentry->htup,			\
										   colidx);					\
		return pg_##NAME##_datum_ref(p_errcode, datum, false);		\
	}

#endif	/* __CUDACC__ */
#endif	/* CUDA_HASHJOIN_H */
