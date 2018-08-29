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

#define GPULZ_MAGIC_NUMBER		0x4c7a		/* 'Lz' */
#define GPULZ_BLOCK_SIZE		1024
#define GPULZ_MIN_MATCH			3
#define GPULZ_MAX_MATCH			34
#define GPULZ_MIN_LONG_OFFSET	0x0700
#define GPULZ_MAX_SHORT_OFFSET	0x06ff
#define GPULZ_MAX_OFFSET		0x106ff

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
		int				c, d;
		cl_uint			index;
		cl_uint			required;
		cl_uint			usage;
		cl_uint			ofs, len;

		if (__syncthreads_count(extra > send) != 0)
		{
			if (get_local_id() == 0)
				printf("gpulz_decompress: control+data block overrun (cntl=%p data=%p extra=%p send=%p)\n", cntl, data, extra, send);
			return false;		/* should not happen */
		}

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
		if (__syncthreads_count(extra + usage > send) != 0)
		{
			if (get_local_id() == 0)
				printf("gpulz_decompress: extra block overrun (extra=%p usage=%u send=%p)\n", extra, usage, send);
			return false;		/* should not happen */
		}

		/* move 'sp' to the next location to read */
		sp = extra + usage;

		/*
		 * MEMO: Once extra byte(s) of thread are identified, we can know
		 * exact length to write out on the destination buffer for each.
		 * Also note that some inline characters may point beyond the @dend
		 * around the boundary of the compressed data tail. These characters
		 * are out of the range, so to be skipped.
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
		if (dp + index < dend)
		{
			/*
			 * OK, write out the uncompressed character or dictionary text
			 */
			assert(dp + index + len <= dend);
			if (c == 0)
				dp[index] = d;
			else
			{
				Assert(dp - ofs >= dest);
				memcpy(dp + index,  dp - ofs, len);
			}
		}
		else
		{
			assert(c == 0 && d == 0);
		}
		dp += usage;

		/* sanity checks */
		if (get_local_id() == 0)
			ptr = (void *)sp;
		__syncthreads();
		assert(__syncthreads_count(ptr != sp) == 0);
		if (get_local_id() == 0)
			ptr = dp;
		__syncthreads();
		assert(__syncthreads_count(ptr != dp) == 0);
	}
	if (__syncthreads_count(sp == send && dp == dend) != 0)
	{
		if (get_local_id() == 0)
			printf("gpulz_decompress: compressed data was not terminated correctly (sp=%p send=%p dp=%p dend=%p)\n", sp, send, dp, dend);
		return false;
	}
	return true;
}

/*
 * kernel_gpulz_decompression
 *
 * An extra kernel to decompress an compressed KDS_FORMAT_COLUMN
 */
KERNEL_FUNCTION_NUMTHREADS(void, GPULZ_BLOCK_SIZE)
kernel_gpulz_decompression(kern_errorbuf *kerror,
						   kern_data_store *kds)
{
	kern_errorbuf ebuf;
	cl_int		cindex;
	__shared__ cl_uint base;

	/* sanity checks */
	assert(kds->format == KDS_FORMAT_COLUMN);
	if (kds->ncols == 0)
		return;

	/* urgent bailout if prior steps already raised an error */
	if (__syncthreads_count(kerror->errcode) > 0)
		return;

	memset(&ebuf, 0, sizeof(ebuf));
	ebuf.kernel = StromKernel_kernel_gpulz_decompression;

	for (cindex = get_group_id();
		 cindex < kds->ncols;
		 cindex += get_num_groups())
	{
		kern_colmeta *cmeta = &kds->colmeta[cindex];
		GPULZ_compressed *comp;
		size_t		offset;
		size_t		rawsize;
		char	   *dest;

		if (cmeta->va_rawsize == 0)
			continue;
		/* fetch compressed image and validation */
		offset = __kds_unpack(cmeta->va_offset);
		if (offset == 0)
			continue;	/* should not happen, but legal */
		comp = (GPULZ_compressed *)((char *)kds + offset);
		if (MAXALIGN(VARSIZE(comp)) != __kds_unpack(cmeta->va_length))
		{
			printf("%s(gid=%d): inconsistent chunk size (%ld of %ld)\n",
				   __FUNCTION__, get_global_id(),
				   VARSIZE(comp), __kds_unpack(cmeta->va_length));
			STROM_SET_ERROR(&ebuf, StromError_DataCorruption);
		}
		else if (MAXALIGN(comp->rawsize) != __kds_unpack(cmeta->va_rawsize))
		{
			printf("%s(gid=%d): inconsistent rawsize (%ld of %ld)\n",
				   __FUNCTION__, get_global_id(),
				   MAXALIGN(comp->rawsize), __kds_unpack(cmeta->va_rawsize));
			STROM_SET_ERROR(&ebuf, StromError_DataCorruption);
		}
		else if (comp->magic != GPULZ_MAGIC_NUMBER)
		{
			printf("%s(gid=%d): wrong magic number (%04x)\n",
				   __FUNCTION__, get_global_id(), comp->magic);
			STROM_SET_ERROR(&ebuf, StromError_DataCorruption);
		}
		else if (comp->blocksz != get_local_size())
		{
			printf("%s(gid=%d): workgroup size mismatch (%d for %d)\n",
				   __FUNCTION__, get_global_id(),
				   get_local_size(), comp->blocksz);
			STROM_SET_ERROR(&ebuf, StromError_DataCorruption);
		}
		if (__syncthreads_count(ebuf.errcode) > 0)
			break;

		/* allocation of the destination buffer */
		rawsize = MAXALIGN(comp->rawsize);
		if (get_local_id() == 0)
			base = atomicAdd(&kds->usage, __kds_packed(rawsize));
		__syncthreads();
		dest = (char *)kds + __kds_unpack(base);
		assert(dest + rawsize <= (char *)kds + kds->length);
		if (!gpulz_decompress(comp->data, VARSIZE(comp), dest, comp->rawsize))
		{
			STROM_SET_ERROR(&ebuf, StromError_DataCorruption);
			break;
		}
		/* update kern_colmeta */
		if (get_local_id() == 0)
		{
			cmeta->va_offset = __kds_packed((char *)dest - (char *)kds);
			cmeta->va_length = __kds_packed(MAXALIGN(rawsize));
			cmeta->va_rawsize = 0;	/* uncompressed */
		}
	}
	kern_writeback_error_status(kerror, &ebuf);
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_LZCOMPRESS_H */
