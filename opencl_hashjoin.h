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
	cl_char			matched;/* flag to track whether this entry get matched */
	/* above fields take 13bytes, so the keydata will usually start from
	 * the unaligned address. However, header portion keydata is used to
	 * nullmap. As long as nkey is less than or equal to 24 (almost right),
	 * key values shall be start from the aligned offset without loss.
	 */
	cl_char			keydata[FLEXIBLE_ARRAY_MEMBER];
} kern_hashentry;

typedef struct
{
	cl_uint			nslots;		/* width of hash slot */
	cl_uint			nkeys;		/* number of keys to be compared */
	cl_char			is_outer;	/* true, if outer join */
	cl_char			__padding__[3];
	struct {
		cl_char		attnotnull;	/* true, if always not null */
		cl_char		attalign;	/* type of alignment */
		cl_short	attlen;		/* length of type */
	} colmeta[FLEXIBLE_ARRAY_MEMBER];	/* simplified kern_colmeta */
} kern_hashtable;

typedef struct
{
	cl_uint			pg_crc32_table[256];
	/* MEMO: Originally, we put 'pg_crc32_table[]' as a static array
	 * deployed on __constant memory region, however, a particular
	 * OpenCL runtime had (has?) a problem on references to values
	 * on __constant memory. So, we moved the 'pg_crc32_table' into
	 * __global memory area as a workaround....
	 */
	hostptr_t		hostptr;	/* address of this mhash on the host-side */
	cl_uint			ntables;	/* number of hash tables (= # of inner rels) */
	cl_uint			htable_offset[FLEXIBLE_ARRAY_MEMBER];
} kern_multihash;

#define KERN_HASHTABLE(kmhash, depth)								\
	((__global kern_hashtable *)((__global char *)(kmhash) +		\
								 (kmhash)->htable_offset[(depth)]))
#define KERN_HASHTABLE_SLOT(khtable)								\
	((__global cl_uint *)((__global char *)(khtable)+				\
						  LONGALIGN(offsetof(kern_hashtable,		\
											 colmeta[(khtable)->nkeys]))))

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
 * | | errcode      |    |    V
 * | +--------------+    |  -----
 * | | results[0]   |    |
 * | | results[1]   |    |  Area to be written back from OpenCL device.
 * | |     :        |    |  Reverse DMA shall be issued here.
 * | | results[N-1] |    V
 * +-+--------------+  -----
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
 * Template of variable reference on the hash-entry
 */
#define STROMCL_SIMPLE_HASHREF_TEMPLATE(NAME,BASE)				\
	static pg_##NAME##_t										\
	pg_##NAME##_hashref(__global kern_hashentry *kentry,		\
						__private int *p_errcode,				\
						cl_uint key_index,						\
						cl_uint key_offset)						\
	{															\
		pg_##NAME##_t result;									\
																\
		if (att_isnull(key_index, kentry->keydata))				\
			result.isnull = true;								\
		else													\
		{														\
			__global BASE *ptr = (__global BASE *)				\
				((__global char *)(kentry) + (key_offset));		\
			result.isnull = false;								\
			result.value = *ptr;								\
		}														\
		return result;											\
	}

static pg_varlena_t
pg_varlena_hashref(__global kern_hashentry *kentry,
				   __private int *p_errcode,
				   cl_uint key_index,
				   cl_uint key_offset)
{
	pg_varlena_t	result;
	__global varlena *vl;

	if (att_isnull(key_index, kentry->keydata))
		result.isnull = true;
	else
	{
		vl = (__global varlena *)((__global char *)kentry +
								  key_offset);
		if (VARATT_IS_4B_U(vl) || VARATT_IS_1B(vl))
		{
			result.value = vl;
			result.isnull = false;
		}
		else
		{
			result.isnull = true;
			STROM_SET_ERROR(p_errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

#define STROMCL_VARLENA_HASHREF_TEMPLATE(NAME)				\
	static pg_##NAME##_t									\
	pg_##NAME##_hashref(__global kern_hashentry *kentry,	\
						__private int *p_errcode,			\
						cl_uint key_index,					\
						cl_uint key_offset)					\
	{														\
		return pg_varlena_hashref(kentry, p_errcode,		\
								  key_index,key_offset);	\
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
