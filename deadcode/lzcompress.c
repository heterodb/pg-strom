/*
 * lzcompress.c
 *
 * Routines for GPU aware compression/decompression
 * ----
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
#include "pg_strom.h"
#include "cuda_lzcompress.h"

struct GPULZ_HistEntry
{
	struct GPULZ_HistEntry *prev;
	struct GPULZ_HistEntry *next;
	cl_int			hindex;	/* hash index */
	const cl_uchar *pos;	/* my position */
};
typedef struct GPULZ_HistEntry	GPULZ_HistEntry;

#define GPULZ_HISTORY_NSLOTS	65536
#define GPULZ_HISTORY_NROOMS	262144

static GPULZ_HistEntry *hist_slots[GPULZ_HISTORY_NSLOTS];
static GPULZ_HistEntry	hist_entries[GPULZ_HISTORY_NROOMS];

static inline cl_int
gpulz_hist_index(const cl_uchar *sp, const cl_uchar *send)
{
	return (sp < send
			? hash_any(sp, Min(send - sp, 4))
			: 0) & (GPULZ_HISTORY_NSLOTS - 1);
}

static void
gpulz_hist_add(const cl_uchar *sp, const cl_uchar *send)
{
	static cl_bool	hent_recycle = false;
	static cl_int	hent_usage = 0;
	cl_int			hindex = gpulz_hist_index(sp, send);
	GPULZ_HistEntry *hent;
	GPULZ_HistEntry *prev;
	GPULZ_HistEntry *next;

	hent = &hist_entries[hent_usage];
	if (hent_recycle)
	{
		prev = hent->prev;
		next = hent->next;

		if (!prev)
			hist_slots[hent->hindex] = next;
		else
			prev->next = next;
		if (next)
			next->prev = prev;
	}
	next = hist_slots[hindex];
	Assert(!next || next->prev == NULL);
	if (next)
		next->prev = hent;

	hent->next = next;
	hent->prev = NULL;
	hent->hindex = hindex;
	hent->pos = sp;
	hist_slots[hindex] = hent;

	if (++hent_usage >= GPULZ_HISTORY_NROOMS)
	{
		hent_usage = 0;
		hent_recycle = true;
	}
}

static bool
gpulz_find_match(const cl_uchar *sp,
				 const cl_uchar *sbase,
				 const cl_uchar *send,
				 int good_match,
				 int good_drop,
				 cl_int *p_length,
				 cl_int *p_offset)
{
	GPULZ_HistEntry *hent;
	cl_int		hindex;
	cl_int		len = 0;
	cl_int		off = INT_MAX;

	hindex = gpulz_hist_index(sp, send);
	for (hent = hist_slots[hindex];
		 hent != NULL && len < GPULZ_MAX_MATCH;
		 hent = hent->next)
	{
		const cl_uchar *ip = sp;
		const cl_uchar *hp = hent->pos;
		int		thisoff;
		int		thislen;

		/* this entry is not visible on decompression time */
		if (hp >= sbase)
			continue;

		/* stop if the offset does not fit */
		thisoff = sbase - hp;
		if (thisoff > GPULZ_MAX_OFFSET)
			break;
		thislen = 0;
		while (ip < send && hp < sbase && *ip == *hp &&
			   thislen < GPULZ_MAX_MATCH)
		{
			thislen++;
			ip++;
			hp++;
		}
		/*
		 * Remember this match as the best, if so.
		 */
		if (thislen > len)
		{
			len = thislen;
			off = thisoff;
		}

		/*
         * Be happy with lesser good matches the more entries we visited. But
         * no point in doing calculation if we're at end of list.
         */
		if (len >= good_match)
			break;
		good_match -= (good_match * good_drop) / 100;
	}

	/*
	 * Return match information only if it results at least in one byte
	 * reduction
	 */
	if (len >= GPULZ_MIN_MATCH)
	{
		*p_length = len;
		*p_offset = off;
		return true;
	}
	return false;
}

cl_int
gpulz_compress(const char *source, cl_int slen, char *dest)
{
	const cl_uchar *sp = (cl_uchar *) source;
	const cl_uchar *send = sp + slen;
	const cl_uchar *sbase;
	cl_uchar   *dp = (cl_uchar *)dest;
	cl_uchar   *dend = dp + slen;
	cl_uchar   *cntl;
	cl_uchar   *ep;
	cl_uint		local_id;
	cl_int		good_match = 128;
	cl_int		good_drop = 10;

	memset(hist_slots, 0, sizeof(hist_slots));
	while (sp < send)
	{
		cntl = dp;
		dp += GPULZ_BLOCK_SIZE / 8;
		ep = dp + GPULZ_BLOCK_SIZE;
		if (ep >= dend)
			return -1;
		memset(cntl, 0, GPULZ_BLOCK_SIZE / 8);

		sbase = sp;
		for (local_id = 0; local_id < GPULZ_BLOCK_SIZE; local_id++)
		{
			cl_int		length;
			cl_int		offset;
			cl_int		nloops;

			if (sp >= send)
			{
				dp[local_id] = 0;
			}
			else if (gpulz_find_match(sp, sbase, send,
									  good_match, good_drop,
									  &length, &offset))
			{
				Assert(length >= GPULZ_MIN_MATCH &&
					   length <= GPULZ_MAX_MATCH);
				Assert(offset <= GPULZ_MAX_OFFSET);
				Assert(sp + length <= send);
				Assert(memcmp(sp, sbase - offset, length) == 0);
				for (nloops = length; nloops > 0; nloops--)
					gpulz_hist_add(sp++, send);
				if (offset < GPULZ_MIN_LONG_OFFSET)
				{
					Assert(((length - GPULZ_MIN_MATCH) & ~0x001f) == 0);
					dp[local_id] = ((length - GPULZ_MIN_MATCH) |
									((offset >> 3) & 0xe0));
					if (ep+1 >= dend)
						return -1;
					*ep++ = (offset & 0x00ff);
				}
				else
				{
					dp[local_id] = (length - GPULZ_MIN_MATCH) | 0xe0;
					offset -= GPULZ_MIN_LONG_OFFSET;
					if (ep+2 >= dend)
						return -1;
					*ep++ = (offset & 0x00ff);
					*ep++ = (offset & 0xff00) >> 8;
				}
				cntl[local_id / 8] |= (1 << (local_id % 8));
			}
			else
			{
				gpulz_hist_add(sp, send);
				dp[local_id] = *sp++;
			}
		}
		/* move to the next block */
		dp = ep;
	}
	return (int)((char *)dp - (char *)dest);
}

cl_bool
gpulz_decompress(const char *source, cl_int slen, char *dest, cl_int rawsize)
{
	const cl_uchar *sp = (const cl_uchar *) source;
	const cl_uchar *send = sp + slen;
	char		   *dp = dest;
	char		   *dend = dp + rawsize;

	while (sp < send && dp < dend)
	{
		const cl_uchar *cntl = sp;
		const cl_uchar *data = sp + GPULZ_BLOCK_SIZE / 8;
		const cl_uchar *extra = data + GPULZ_BLOCK_SIZE;
		char		   *dbase = dp;
		cl_int			local_id;

		if (extra > send)
			return false;				/* should not happen */

		for (local_id = 0; local_id < GPULZ_BLOCK_SIZE; local_id++)
		{
			if (dp >= dend)
				break;		/* no need to write out any more */
			if ((cntl[local_id / 8] & (1 << (local_id % 8))) != 0)
			{
				int		d = data[local_id];
				int		len;
				int		ofs;

				len = (d & 0x1f) + GPULZ_MIN_MATCH;
				ofs = (d & 0xe0);
				if (ofs != 0xe0)
				{
					if (extra + 1 > send)
						return false;	/* should not happen */
					/* short offset */
					ofs = (ofs << 3) | *extra++;
				}
				else
				{
					/* long offset */
					if (extra + 2 > send)
						return false;	/* should not happen */
					ofs = ((extra[1] << 8) | extra[0]) + GPULZ_MIN_LONG_OFFSET;
					extra += 2;
				}
				Assert(dbase - ofs >= dest);
				Assert(dp + len <= dend);
				memcpy(dp, dbase - ofs, len);
				dp += len;
			}
			else
			{
				/* uncompressed byte */
				*dp++ = data[local_id];
			}
		}
		sp = extra;
	}
	if (sp == send && dp >= dend)
		return true;

	return false;
}

/* SQL stubs */
Datum pgstrom_gpulz_compress(PG_FUNCTION_ARGS);
Datum pgstrom_gpulz_compress_raw(PG_FUNCTION_ARGS);
Datum pgstrom_gpulz_decompress(PG_FUNCTION_ARGS);

Datum
pgstrom_gpulz_compress(PG_FUNCTION_ARGS)
{
	struct varlena *src = PG_GETARG_VARLENA_P(0);
	int		rawsize = VARSIZE_ANY_EXHDR(src);
	GPULZ_compressed *dst;
	int		dstlen;
	cl_uint *magic;

	dst = palloc(offsetof(GPULZ_compressed, data) + rawsize + sizeof(cl_uint));
	dst->magic = GPULZ_MAGIC_NUMBER;
	dst->blocksz = GPULZ_BLOCK_SIZE;
	dst->rawsize = rawsize;
	magic = (cl_uint *)((char *)dst +
						offsetof(GPULZ_compressed, data) + rawsize);
	*magic = 0xdeadbeaf;
	dstlen = gpulz_compress(VARDATA(src), rawsize, dst->data);
	Assert(*magic == 0xdeadbeaf);
	if (dstlen < 0)
	{
		elog(NOTICE, "GPULz: unable to compress the data");
		PG_RETURN_NULL();
	}
	SET_VARSIZE(dst, offsetof(GPULZ_compressed, data) + dstlen);

	elog(INFO, "GPULz: %d -> %d (%.2f%%)",
		 rawsize, dstlen, 100.0 * (double)dstlen / (double)rawsize);

	PG_RETURN_POINTER(dst);
}
PG_FUNCTION_INFO_V1(pgstrom_gpulz_compress);

Datum
pgstrom_gpulz_compress_raw(PG_FUNCTION_ARGS)
{
	char   *rawdata = PG_GETARG_POINTER(0);		/* internal */
	int64	rawsize = PG_GETARG_INT64(1);		/* int8 */
	GPULZ_compressed *dst;
	int		dstlen;

	if (rawsize < 0)
		elog(ERROR, "GPULz: invalid rawsize %ld", rawsize);
	dst = palloc(offsetof(GPULZ_compressed, data) + rawsize + sizeof(cl_uint));
	dst->magic = GPULZ_MAGIC_NUMBER;
	dst->blocksz = GPULZ_BLOCK_SIZE;
	dst->rawsize = (cl_uint)rawsize;
	*((cl_uint *)((char *)dst->data + rawsize)) = 0xdeadbeaf;
	dstlen = gpulz_compress(rawdata, rawsize, dst->data);
	if (dstlen < 0)
	{
		pfree(dst);
		elog(NOTICE, "GPULz: unable to compress the data");
		PG_RETURN_NULL();
	}
	Assert(*((cl_uint *)((char *)dst->data + rawsize)) == 0xdeadbeaf);
	SET_VARSIZE(dst, offsetof(GPULZ_compressed, data) + dstlen);

	elog(INFO, "GPULz: %ld -> %d (%.2f%%)",
		 rawsize, dstlen, 100.0 * (double)dstlen / (double)rawsize);

	PG_RETURN_POINTER(dst);
}
PG_FUNCTION_INFO_V1(pgstrom_gpulz_compress_raw);

Datum
pgstrom_gpulz_decompress(PG_FUNCTION_ARGS)
{
	GPULZ_compressed *src = (GPULZ_compressed *)PG_GETARG_VARLENA_P(0);
	int		srclen;
	int		rawsize;
	struct varlena *dst;

	if (VARSIZE(src) < offsetof(GPULZ_compressed, data) ||
		src->magic != GPULZ_MAGIC_NUMBER ||
		src->blocksz != GPULZ_BLOCK_SIZE)
		elog(ERROR, "GPULz: wrong data format");
	srclen = VARSIZE(src) - offsetof(GPULZ_compressed, data);
	rawsize = src->rawsize;
	dst = palloc(rawsize + VARHDRSZ);
	if (!gpulz_decompress(src->data, srclen, dst->vl_dat, rawsize))
		elog(ERROR, "GPULz: unable to decompress the data");
	SET_VARSIZE(dst, rawsize + VARHDRSZ);

	elog(INFO, "GPULz: %d -> %d", srclen, rawsize);

	PG_RETURN_POINTER(dst);
}
PG_FUNCTION_INFO_V1(pgstrom_gpulz_decompress);
