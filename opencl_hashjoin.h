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
 * | | length        o-----------------------------------+
 * | +------------------+                                |
 * | | is_matched    o---------------------------------+ |
 * | +------------------+  offset towards array to     | |
 * | | nslots (=N)      |  track which row was matched | |
 * | +------------------+  for outer join.             | |
 * | | nkeys (=M)       |                              | |
 * | +------------------+                              | |
 * | | colmeta[0]       |                              | |
 * | | colmeta[1]       |                              | |
 * | |    :             |                              | |
 * | | colmeta[M-1]     |                              | |
 * | +------------------+                              | |
 * | | hash_slot[0]     |                              | |
 * | | hash_slot[1]     |                              | |
 * | |     :            |                              | |
 * | | hash_slot[N-2] o-------+  single directioned    | |
 * | | hash_slot[N-1]   |     |  link from the         | |
 * +-+------------------+ <---+  hash_slot[]           | |
 * | kern_gpuhash_entry |                              | |
 * | +------------------+                              | |
 * | | next       o-----------+  If multiple entries   | |
 * | +------------------+     |  has same hash value,  | |
 * | | hash             |     |  these are linked.     | |
 * | +------------------+     |                        | |
 * | | rowidx           |     |                        | |
 * | +------------------+     |                        | |
 * | | keydata:         |     |                        | |
 * | | (actual values   |     |                        | |
 * | |  to be joined)   |     |                        | |
 * +-+------------------+ <---+                        | |
 * | kern_gpuhash_entry |                              | |
 * | +-----------------+-                              | |
 * | | next       o-----------+                        | |
 * | +------------------+     |                        | |
 * | | hash             |     |                        | |
 * | +------------------+     |                        | |
 * | |      :           |     |                        | |
 * |        :           |     |                        | |
 * | |      :           |     |                        | |
 * +-+------------------+ <---+                        | |
 * | kern_gpuhash_entry |                              | |
 * | +------------------+                              | |
 * | | next             |                              | |
 * | +------------------+                              | |
 * | | hash             |                              | |
 * | +------------------+                              | |
 * | | rowidx           |                              | |
 * | +------------------+                              | |
 * | | keydata          |                              | |
 * | | +----------------+                              | |
 * | | | nullmap(16bit) |                              | |
 * | | | keydata:       |                              | |
 * | | | (actual values |                              | |
 * | | |  to be joined) |                              | |
 * +-+------------------+  <---------------------------+ |
 * | is_matched         |     ^                          |
 * | (array of cl_char) |     |                          |
 * | Non-zero means     |     | sizeof(cl_char) * M      |
 * | associated inner   |     |                          |
 * | row was matched    |     |                          |
 * |        :           |     |                          |
 * |        :           |     v                          |
 * +--------------------+  <-----------------------------+
 */
typedef struct
{
	cl_uint			next;	/* offset of the next */
	cl_uint			hash;	/* 32-bit hash value */
	cl_ulong		rowid;	/* identifier of inner rows; upper 32bits are used
							 * to point a particular rc-store, lower 32bits
							 * are index within the rc-store.
							 */
	cl_char			keydata[FLEXIBLE_ARRAY_MEMBER];
} kern_hashentry;

typedef struct
{
	cl_uint			length;	/* length of this hash table */
	cl_uint			nslots;	/* width of hash slot */
	cl_uint			nkeys;	/* number of keys to be compared */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER];
} kern_hashtable;

#define KERN_HASHTABLE_SLOT(khash)										\
	(__global cl_uint *)((__global char *)(khash) +						\
						 LONGALIGN(offsetof(kern_hashtable,				\
											colmeta[(khash)->nkeys])))

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
} kern_hashjoin;

#define KERN_HASHJOIN_PARAMBUF(kghashjoin)					\
	((__global kern_parambuf *)(&(khashjoin)->kparam))
#define KERN_HASHJOIN_PARAMBUF_LENGTH(kghashjoin)			\
	STROMALIGN(KERN_GPUHJ_PARAMBUF(khashjoin)->length)
#define KERN_HASHJOIN_RESULTBUF(kghashjoin)								\
	((__global kern_resultbuf *)((__global char *)kghashjoin +			\
								 KERN_GPUHJ_PARAMBUF_LENGTH(kghashjoin))
#define KERN_HASHJOIN_RESULTBUF_LENGTH(kghashjoin)						\
	STROMALIGN(offsetof(kern_resultbuf,									\
						results[KERN_GPUHJ_RESULTBUF(kghashjoin)->nrooms]))
#define KERN_HASHJOIN_DMA_SENDLEN(kghashjoin)	\
	(KERN_GPUHJ_PARAMBUF_LENGTH(kghashjoin) +	\
	 offsetof(kern_resultbuf, results[0]))
#define KERN_HASHJOIN_DMA_RECVLEN(kghashjoin)	\
	(offsetof(kern_resultbuf,					\
			  results[KERN_GPUHJ_RESULTBUF(kghashjoin)->nrooms]))



#ifdef OPENCL_DEVICE_CODE
/*
 * declaration of functions being generated on the fly.
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
					__global kern_parambuf *kparams,
					__global kern_hashentry *kentry,
					__global kern_column_store *kcs,
					__global kern_toastbuf *ktoast,
					size_t row_index);

/*
 * gpuhashjoin_inner
 *
 * Routine of INNER HASH-JOIN. It counts up number of matched rows on the
 * hash-table using hash-key comparison function. Then, it acquires this
 * number of result slot in the result buffer, and write a pair of row-
 * indexes back. In case of result-buffer is not sufficient, it returns
 * StromError_DataStoreNoSpace, to inform host system with mode larger
 * result buffer.
 */
static void
gpuhashjoin_inner(__private cl_int *errcode,
				  __global kern_parambuf *kparams,
				  __global kern_resultbuf *kresults,
				  __global kern_hashtable *khashtbl,
				  __global kern_column_store *kcs,
				  __global kern_toastbuf *ktoast,
				  size_t kcs_index,
				  __local void *local_workbuf)
{
	cl_int		errcode = StromError_Success;
	cl_uint		hash_value;
	cl_uint		slot_index;

	/* calculate a hash value */
	hash_value = gpuhashjoin_hashkey(&errcode, kparams,
									 kcs, ktoast, kcs_index);
	slot_index = hash_value % khashtbl->nslots;

	/*
	 * XXX - walk on the hash table to count number of matched tuples
	 */


	/*
	 * XXX - calculate total number of matched tuples being searched
	 * by this workgroup
	 */


	/*
	 * XXX - allocation of result buffer. A tuple takes 2 * sizeof(cl_uint)
	 * to store pair of row-indexes.
	 * If no space any more, return an error code to retry later.
	 *
	 * use atomic_add(&kresults->nrows, nitems) to determine the position
	 * to write. If expected usage is larger than kresults->nrooms, it
	 * exceeds the limitation of result buffer.
	 *
	 * MEMO: we may need to re-define nrows/nitems using 64bit variables
	 * to avoid overflow issues, but has special platform capability on
	 * 64bit atomic-write...
	 */

	/*
	 * XXX - walk on the hash table again, to write pair of row-index
	 * on the acquired slot
	 */



}


__kernel void
gpuhashjoin_inner_cs(__global kern_hashjoin *khashjoin,
					 __global kern_hashtable *khashtbl,
					 __global kern_column_store *kcs,
					 __global kern_toastbuf *toast,
					 __local void *local_workmem)
{
	__global kern_parambuf *kparams = KERN_GPUHJ_PARAMBUF(kghashjoin);
	__global kern_resultbuf *kresults = KERN_GPUHJ_RESULTBUF(kghashjoin);


}

__kernel void
gpuhashjoin_inner_rs(__global kern_hashjoin *khashjoin,
					 __global kern_hashtable *khashtbl,
					 __global kern_row_store *krs,
					 __global kern_column_store *kcs,
					 __local void *local_workmem)
{
	__global kern_parambuf *kparams = KERN_GPUHJ_PARAMBUF(kghashjoin);
	__global kern_resultbuf *kresults = KERN_GPUHJ_RESULTBUF(kghashjoin);
	pg_bytea_t		kparam_0 = pg_bytea_param(kparams,&errcode,0);
	cl_int			errcode = StromError_Success;
	__local size_t	kcs_offset;
	__local size_t	kcs_nitems;

	/*
	 * map this workgroup on a particular range on the column-store
	 */
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
					  kresults,
					  khashtbl,
					  kcs,
					  (__global kern_toastbuf *)krs,
					  kcs_offset + get_local_id(0),
					  local_workbuf);
	/* back execution status into host-side */
	kern_writeback_error_status(&kresult->errcode, errcode, local_workbuf);
}

/*
 * Template of variable reference on the hash-entry
 */
#define STROMCL_SIMPLE_HASHREF_TEMPLATE(NAME,BASE)			\
	static pg_##NAME##_t									\
	pg_##NAME##_hashref(__global kern_hashentry *kentry,		\
						__private int *p_errcode,			\
						cl_uint key_index,					\
						cl_uint key_offset)					\
	{														\
		pg_##NAME##_t result;								\
															\
		if (att_isnull(key_index, khe->keydata))			\
			result.isnull = true;							\
		else												\
		{													\
			result.isnull = false;							\
			result.value = *((__global BASE *)				\
							 (khe->keydata + key_offset));	\
		}													\
		return result;										\
	}

static pg_varlena_t
pg_varlena_hashref(__global kern_hashentry *kentry,
				   __private int *p_errcode,
				   cl_uint key_index,
				   cl_uint key_offset)
{
	pg_varlena_t	result;
	__global varlena *vl;

	if (att_isnull(key_index, khe->keydata))
		result.isnull = true;
	else
	{
		vl = (__global varlena *)((__global char *)khe +
								  key_offset);
		if (VARATT_IS_4B_U(val) || VARATT_IS_1B(val))
		{
			result.value = vl;
			result.isnull = false;
		}
		else
		{
			result.isnull = true;
			STROM_SET_ERROR(p_errcode, StromError_RowReCheck);
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
		return pg_varlena_hashref(khe,p_errcode,			\
								  key_index,key_offset);	\
	}

/*
 * Macros to calculate hash key-value.
 * (logic was copied from pg_crc32.c)
 */
/*
 * This table is based on the polynomial
 *  x^32+x^26+x^23+x^22+x^16+x^12+x^11+x^10+x^8+x^7+x^5+x^4+x^2+x+1.
 * (This is the same polynomial used in Ethernet checksums, for instance.)
 */
__constant cl_uint pg_crc32_table[256] = {
	0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA,
	0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
	0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988,
	0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91,
	0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE,
	0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
	0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC,
	0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
	0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
	0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
	0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940,
	0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
	0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116,
	0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
	0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
	0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
	0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A,
	0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
	0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818,
	0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01,
	0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
	0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
	0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C,
	0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
	0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2,
	0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB,
	0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
	0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
	0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086,
	0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
	0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4,
	0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD,
	0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A,
	0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683,
	0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8,
	0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
	0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE,
	0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7,
	0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC,
	0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5,
	0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252,
	0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
	0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60,
	0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79,
	0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236,
	0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F,
	0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04,
	0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
	0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A,
	0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713,
	0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38,
	0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21,
	0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E,
	0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
	0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C,
	0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
	0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2,
	0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB,
	0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0,
	0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
	0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6,
	0xBAD03605, 0xCDD70693, 0x54DE5729, 0x23D967BF,
	0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
	0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
};

#define INIT_CRC32(crc)		((crc) = 0xFFFFFFFF)
#define COMP_CRC32(crc, data, len)									\
	do {															\
		__global const cl_uchar *__data								\
			= (__global const cl_uchar *)(data);					\
		uint32 __len = (len);										\
																	\
		while (__len-- > 0)											\
		{															\
			cl_int     __tab_index =								\
				((cl_int) ((crc) >> 24) ^ *__data++) & 0xFF;		\
			(crc) = pg_crc32_table[__tab_index] ^ ((crc) << 8);		\
		}															\
	} while (0)
#define FIN_CRC32(crc)		((crc) ^= 0xFFFFFFFF)








#endif

typedef struct pgstrom_hashjoin_table
{
	StromObject		sobj;		/* = StromTab_HashJoinTable */
	cl_uint			maxlen;		/* max length of hash-table; be allocated */
	slock_t			lock;		/* protection of the fields below */
	cl_int			refcnt;		/* reference counter of this hash table */
	cl_int			n_kernel;	/* number of active running kernel */
	cl_mem			m_hash;		/* in-kernel buffer object. Once n_kernel
								 * backed to zero, valid m_hash needs to
								 * be released. */
	cl_int			num_rcs;	/* number of inner row/column-stores */
	cl_int			max_rcs;	/* max number of inner row/column-stores */
	StromObject	  **rcstore;	/* array of row/column-store */
	kern_hashtable	kern;
} pgstrom_hashjoin_table;

typedef struct
{
	pgstrom_message	msg;		/* = StromTag_GpuHashJoin */
	dlist_node		chain;		/* to chain free-list */
	Datum			dprog_key;	/* device key for gpuhashjoin */
	

	bool			build_pscan;/* true, if want pseudo-scan view */
	StromObject	   *rcs_in;
	StromObject	   *rcs_out;
	kern_hashjoin  *kern;
} pgstrom_gpuhashjoin;

#endif	/* OPENCL_HASHJOIN_H */
