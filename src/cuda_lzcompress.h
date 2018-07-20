/*
 * cuda_lzcompress.h
 *
 * GPU aware LZ decompression routine
 * ----
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
 */
#ifndef CUDA_LZCOMPRESS_H
#define CUDA_LZCOMPRESS_H

typedef struct
{
	char		_vl_head[VARHDRSZ];
	cl_ushort	magic;		/* = GPULZ_MAGIC_NUMBER */
	cl_ushort	blocksz;	/* = GPULZ_BLOCK_SIZE */
	cl_uint		rawsize;	/* uncompressed data size w/o header */
	char		data[FLEXIBLE_ARRAY_MEMBER];
} GPULZ_compressed;

#ifdef __CUDACC__
/*
 * gpulz_decompress
 *
 * In-GPU LZ-decompression routine; this function must be called per workgroup.
 */
STATIC_FUNCTION(cl_bool)
gpulz_decompress(const char *source, cl_int slen, char *dest, cl_int rawsize)
{
	const cl_uchar *sp = (const cl_uchar *) source;
	const cl_uchar *send = sp + slen;
	char		   *dp = dest;
	char		   *dend = dp + rawsize;
	cl_uint			mask = (1 << (get_local_id() & 7));
	__shared__ void *ptr;	/* for sanity check */

	while (sp < send && dp < dend)
	{
		const cl_uchar *cntl = sp;
		const cl_uchar *data = sp + (get_local_size() >> 3);
		const cl_uchar *extra = data + get_local_size();
		int				c, d, mask;
		cl_uint			index;
		cl_uint			required;
		cl_uint			usage;
		cl_uint			ofs, len;

		if (extra > send)
			return false;		/* should not happen */
		/*
		 * MEMO: First 1024bit of the current iteration block is control bit
		 * for each threads in workgroup. If zero, its relevant data byte is
		 * uncompressed character. Otherwise, a pair of data byte and extra
		 * byte(s) are backward offset and length on the current sliding-
		 * window. First of all, we have to ensure this iteration block does
		 * not overrun the source memory region.
		 */
		c = cntl[get_local_id() >> 3] & mask;
		d = data[get_local_id()];
		if (c == 0)
			required = 0;		/* uncompressed character */
		else if ((d & 0xe0) != 0xe0)
			required = 1;		/* short offset */
		else
			required = 2;		/* long offset */

		index = pgstromStairlikeSum(required, &usage);
		if (extra + usage > send)
			return false;		/* should not happen */

		/* move 'sp' to the next location to read */
		sp = extra + usage;

		/*
		 * MEMO: Once extra byte(s) of thread are identified, we can know
		 * the length to write on the destination buffer for each.
		 * Then, we have to ensure the destination buffer has enough space
		 * to write out.
		 */
		extra += index;			/* thread's own extra byte(s), if any */
		if (c == 0)
			len = 1;
		else
		{
			len = (d & 0x1f) + GPULZ_MIN_MATCH;
			ofs = (d & 0xe0);
			if (ofs != 0xe0)
				ofs = (ofs << 3) | *extra;		/* short offset */
			else
				ofs = ((extra[1] << 8) | extra[0]) + GPULZ_MIN_LONG_OFFSET;
		}
		index = pgstromStairlikeSum(len, &usage);
		if (dp + usage > dend)
			return false;		/* should not happen */

		/*
		 * OK, write out the uncompressed character or compressed string.
		 */
		if (c == 0)
			dp[index] = d;
		else
		{
			Assert(dp - ofs >= dest);
			memcpy(dp + index,  dp - ofs, len);
		}
		dp += usage;

		/* sanity checks */
		if (get_local_id() == 0)
			ptr = sp;
		__syncthreads();
		assert(__syncthreads_count(ptr == sp) == 0);
		if (get_local_id() == 0)
			ptr = dp;
		assert(__syncthreads_count(ptr == dp) == 0);
	}
	return (sp == send && dp == dend);
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_LZCOMPRESS_H */
