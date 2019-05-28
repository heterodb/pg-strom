/*
 * libgpucore.cu
 *
 * Core implementation of GPU device code.
 * ----
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
#define KERN_CONTEXT_VARLENA_BUFSZ 10 //tentative
#include "cuda_common.h"


/*
 * __compute_heaptuple_size
 */
DEVICE_FUNCTION(cl_uint)
__compute_heaptuple_size(kern_context *kcxt,
						 kern_colmeta *__cmeta,
						 cl_bool heap_hasoid,
						 cl_uint ncols,
						 cl_char *tup_dclass,
						 Datum   *tup_values)
{
	cl_uint		t_hoff;
	cl_uint		datalen = 0;
	cl_uint		j;
	cl_bool		heap_hasnull = false;

	/* compute data length */
	for (j=0; j < ncols; j++)
	{
		kern_colmeta   *cmeta = &__cmeta[j];
		cl_char			dclass = tup_dclass[j];

		if (dclass == DATUM_CLASS__NULL)
			heap_hasnull = true;
		else if (cmeta->attlen > 0)
		{
			assert(dclass == DATUM_CLASS__NORMAL);
			datalen = TYPEALIGN(cmeta->attalign, datalen);
			datalen += cmeta->attlen;
		}
		else
		{
			Datum		datum = tup_values[j];
			cl_uint		vl_len;

			switch (dclass)
			{
				case DATUM_CLASS__VARLENA:
					vl_len = pg_varlena_datum_length(kcxt,datum);
					datalen = TYPEALIGN(cmeta->attalign, datalen);
					break;
#ifdef PGSTROM_KERNEL_HAS_PGARRAY
				case DATUM_CLASS__ARRAY:
					vl_len = pg_array_datum_length(kcxt,datum);
					datalen = TYPEALIGN(cmeta->attalign, datalen);
					break;
#endif
#ifdef PGSTROM_KERNEL_HAS_PGCOMPOSITE
				case DATUM_CLASS__COMPOSITE:
					vl_len = pg_composite_datum_length(kcxt,datum);
					datalen = TYPEALIGN(cmeta->attalign, datalen);
					break;
#endif
				default:
					assert(dclass == DATUM_CLASS__NORMAL);
					vl_len = VARSIZE_ANY(datum);
					if (!VARATT_IS_1B(datum))
						datalen = TYPEALIGN(cmeta->attalign, datalen);
					break;
			}
			datalen += vl_len;
		}
	}
	/* compute header offset */
	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (heap_hasnull)
		t_hoff += BITMAPLEN(ncols);
	if (heap_hasoid)
		t_hoff += sizeof(cl_uint);
	t_hoff = MAXALIGN(t_hoff);

	return t_hoff + datalen;
}

/*
 * deform_kern_heaptuple
 *
 * Like deform_heap_tuple in host side, it extracts the supplied tuple-item
 * into tup_values / tup_isnull array.
 *
 * NOTE: composite datum which is built-in other composite datum might not
 * be aligned to 4-bytes boundary. So, we don't touch htup fields directly,
 * except for 1-byte datum.
 */
DEVICE_FUNCTION(void)
deform_kern_heaptuple(cl_int	nattrs,			/* in */
					  kern_colmeta *tup_attrs,	/* in */
					  HeapTupleHeaderData *htup,/* in */
					  cl_char  *tup_dclass,		/* out */
					  Datum	   *tup_values)		/* out */
{
	/* 'htup' must be aligned to 8bytes */
	assert(((cl_ulong)htup & (MAXIMUM_ALIGNOF-1)) == 0);
	if (!htup)
	{
		int		i;

		for (i=0; i < nattrs; i++)
			tup_dclass[i] = DATUM_CLASS__NULL;
	}
	else
	{
		cl_uint		offset = htup->t_hoff;
		cl_bool		tup_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
		cl_uint		i, ncols = (htup->t_infomask2 & HEAP_NATTS_MASK);

		ncols = Min(ncols, nattrs);
		for (i=0; i < ncols; i++)
		{
			if (tup_hasnull && att_isnull(i, htup->t_bits))
			{
				tup_dclass[i] = DATUM_CLASS__NULL;
				tup_values[i] = 0;
			}
			else
			{
				kern_colmeta   *cmeta = &tup_attrs[i];
				char		   *addr;

				if (cmeta->attlen > 0)
					offset = TYPEALIGN(cmeta->attalign, offset);
				else if (!VARATT_NOT_PAD_BYTE((char *)htup + offset))
					offset = TYPEALIGN(cmeta->attalign, offset);

				/* Store the value */
				addr = ((char *) htup + offset);
				if (cmeta->attbyval)
				{
					if (cmeta->attlen == sizeof(cl_char))
						tup_values[i] = *((cl_char *)addr);
					else if (cmeta->attlen == sizeof(cl_short))
						tup_values[i] = *((cl_short *)addr);
					else if (cmeta->attlen == sizeof(cl_int))
						tup_values[i] = *((cl_int *)addr);
					else if (cmeta->attlen == sizeof(cl_long))
						tup_values[i] = *((cl_long *)addr);
					else
					{
						assert(cmeta->attlen <= sizeof(Datum));
						memcpy(&tup_values[i], addr, cmeta->attlen);
					}
					offset += cmeta->attlen;
				}
				else
				{
					cl_uint		attlen = (cmeta->attlen > 0
										  ? cmeta->attlen
										  : VARSIZE_ANY(addr));
					tup_values[i] = PointerGetDatum(addr);
					offset += attlen;
				}
				tup_dclass[i] = DATUM_CLASS__NORMAL;
			}
		}
		/*
		 * Fill up remaining columns if source tuple has less columns than
		 * length of the array; that is definition of the destination
		 */
		while (i < nattrs)
			tup_dclass[i++] = DATUM_CLASS__NORMAL;
	}
}

/*
 * __form_kern_heaptuple
 */
DEVICE_FUNCTION(cl_uint)
__form_kern_heaptuple(kern_context *kcxt,
					  void	   *buffer,			/* out */
					  cl_int	ncols,			/* in */
					  kern_colmeta *colmeta,	/* in */
					  HeapTupleHeaderData *htup_orig, /* in: if heap-tuple */
					  cl_int	comp_typmod,	/* in: if composite type */
					  cl_uint	comp_typeid,	/* in: if composite type */
					  cl_uint	htuple_oid,		/* in */
					  cl_char  *tup_dclass,		/* in */
					  Datum	   *tup_values)		/* in */
{
	HeapTupleHeaderData *htup = (HeapTupleHeaderData *)buffer;
	cl_bool		tup_hasnull = false;
	cl_ushort	t_infomask;
	cl_uint		t_hoff;
	cl_uint		i, curr;

	/* alignment checks */
	assert((uintptr_t)htup == MAXALIGN(htup));

	/* has any NULL attributes? */
	if (tup_dclass != NULL)
	{
		for (i=0; i < ncols; i++)
		{
			if (tup_dclass[i] == DATUM_CLASS__NULL)
			{
				tup_hasnull = true;
				break;
			}
		}
	}
	t_infomask = (tup_hasnull ? HEAP_HASNULL : 0);

	/* preserve HeapTupleHeaderData, if any */
	if (htup_orig)
		memcpy(htup, htup_orig, offsetof(HeapTupleHeaderData,
										 t_ctid) + sizeof(ItemPointerData));
	else
	{
		/* datum_len_ shall be set on the tail  */
		htup->t_choice.t_datum.datum_typmod = comp_typmod;
		htup->t_choice.t_datum.datum_typeid = comp_typeid;
		htup->t_ctid.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
		htup->t_ctid.ip_blkid.bi_lo = 0xffff;
		htup->t_ctid.ip_posid = 0;				/* InvalidOffsetNumber */
	}
	htup->t_infomask2 = (ncols & HEAP_NATTS_MASK);

	/* computer header size */
	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (tup_hasnull)
		t_hoff += BITMAPLEN(ncols);
	if (htuple_oid != 0)
	{
		t_infomask |= HEAP_HASOID;
		t_hoff += sizeof(cl_uint);
	}
	t_hoff = MAXALIGN(t_hoff);
	if (htuple_oid != 0)
		*((cl_uint *)((char *)htup + t_hoff - sizeof(cl_uint))) = htuple_oid;

	/* walk on the regular columns */
	htup->t_hoff = t_hoff;
	curr = t_hoff;

	for (i=0; i < ncols; i++)
	{
		kern_colmeta *cmeta = &colmeta[i];
		Datum		datum = tup_values[i];
		cl_char		dclass;
		cl_int		padding;

		dclass = (!tup_dclass ? DATUM_CLASS__NORMAL : tup_dclass[i]);
		if (dclass == DATUM_CLASS__NULL)
		{
			assert(tup_hasnull);
			htup->t_bits[i >> 3] &= ~(1 << (i & 0x07));
			continue;
		}

		if (tup_hasnull)
			htup->t_bits[i >> 3] |= (1 << (i & 0x07));

		padding = TYPEALIGN(cmeta->attalign, curr) - curr;
		if (cmeta->attbyval)
		{
			assert(dclass == DATUM_CLASS__NORMAL);
			while (padding-- > 0)
				((char *)htup)[curr++] = '\0';
			assert(cmeta->attlen <= sizeof(datum));
			memcpy((char *)htup + curr, &datum, cmeta->attlen);
			curr += cmeta->attlen;
		}
		else if (cmeta->attlen > 0)
		{
			assert(dclass == DATUM_CLASS__NORMAL);
			while (padding-- > 0)
				((char *)htup)[curr++] = '\0';
			memcpy((char *)htup + curr,
				   DatumGetPointer(datum), cmeta->attlen);
			curr += cmeta->attlen;
		}
		else
		{
			cl_int		vl_len;

			switch (dclass)
			{
				case DATUM_CLASS__VARLENA:
					while (padding-- > 0)
						((char *)htup)[curr++] = '\0';
					vl_len = pg_varlena_datum_write(kcxt,
													(char *)htup+curr,
													datum);
					break;
#ifdef PGSTROM_KERNEL_HAS_PGARRAY
				case DATUM_CLASS__ARRAY:
					while (padding-- > 0)
						((char *)htup)[curr++] = '\0';
					vl_len = pg_array_datum_write(kcxt,
												  (char *)htup+curr,
												  datum);
					break;
#endif
#ifdef  PGSTROM_KERNEL_HAS_PGCOMPOSITE
				case DATUM_CLASS__COMPOSITE:
					while (padding-- > 0)
						((char *)htup)[curr++] = '\0';
					vl_len = pg_composite_datum_write(kcxt,
													  (char *)htup+curr,
													  datum);
					break;
#endif
				default:
					assert(dclass == DATUM_CLASS__NORMAL);
					vl_len = VARSIZE_ANY(datum);
					if (!VARATT_IS_1B(datum))
					{
						while (padding-- > 0)
							((char *)htup)[curr++] = '\0';
					}
					memcpy((char *)htup + curr,
						   DatumGetPointer(datum), vl_len);
					break;
			}
			t_infomask |= HEAP_HASVARWIDTH;
			curr += vl_len;
		}
	}
	htup->t_infomask = t_infomask;
	if (!htup_orig)
		SET_VARSIZE(&htup->t_choice.t_datum, curr);
	return curr;
}

/*
 * pgstromStairlikeSum
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
DEVICE_FUNCTION(cl_uint)
pgstromStairlikeSum(cl_uint my_value, cl_uint *total_sum)
{
	cl_uint	   *items = SHARED_WORKMEM(cl_uint);
	cl_uint		local_sz;
	cl_uint		local_id;
	cl_uint		unit_sz;
	cl_uint		stair_sum;
	cl_int		i, j;

	/* setup local size (pay attention, if 2D invocation) */
	local_sz = get_local_size();
	local_id = get_local_id();
	assert(local_id < local_sz);

	/* set initial value */
	items[local_id] = my_value;
	__syncthreads();

	for (i=1, unit_sz = local_sz; unit_sz > 0; i++, unit_sz >>= 1)
	{
		/* index of last item in the earlier half of each 2^i unit */
		j = (local_id & ~((1 << i) - 1)) | ((1 << (i-1)) - 1);

		/* add item[j] if it is later half in the 2^i unit */
		if ((local_id & (1 << (i - 1))) != 0)
			items[local_id] += items[j];
		__syncthreads();
	}
	if (total_sum)
		*total_sum = items[local_sz - 1];
	stair_sum = local_id == 0 ? 0 : items[local_id - 1];
	__syncthreads();
	return stair_sum;
}

/*
 * pgstromStairlikeBinaryCount
 *
 * A special optimized version of pgstromStairlikeSum, for binary count.
 * It has smaller number of __syncthreads().
 */
DEVICE_FUNCTION(cl_uint)
pgstromStairlikeBinaryCount(int predicate, cl_uint *total_count)
{
	cl_uint	   *items = SHARED_WORKMEM(cl_uint);
	cl_uint		nwarps = get_local_size() / warpSize;
	cl_uint		warp_id = get_local_id() / warpSize;
	cl_uint		w_bitmap;
	cl_uint		stair_count;
	cl_int		unit_sz;
	cl_int		i, j;

	w_bitmap = __ballot_sync(__activemask(), predicate);
	if ((get_local_id() & (warpSize-1)) == 0)
		items[warp_id] = __popc(w_bitmap);
	__syncthreads();

	for (i=1, unit_sz = nwarps; unit_sz > 0; i++, unit_sz >>= 1)
	{
		/* index of last item in the earlier half of each 2^i unit */
		j = (get_local_id() & ~((1<<i)-1)) | ((1<<(i-1))-1);

		/* add item[j] if it is later half in the 2^i unit */
		if (get_local_id() < nwarps &&
			(get_local_id() & (1 << (i-1))) != 0)
			items[get_local_id()] += items[j];
		__syncthreads();
	}
	if (total_count)
		*total_count = items[nwarps - 1];
	w_bitmap &= (1U << (get_local_id() & (warpSize-1))) - 1;
	stair_count = (warp_id == 0 ? 0 : items[warp_id - 1]) + __popc(w_bitmap);
	__syncthreads();

	return stair_count;
}

/*
 * Device version of hash_any() in PG host code
 */
#define rot(x,k)		(((x)<<(k)) | ((x)>>(32-(k))))
#define mix(a,b,c)								\
	{											\
		a -= c;  a ^= rot(c, 4);  c += b;		\
		b -= a;  b ^= rot(a, 6);  a += c;		\
		c -= b;  c ^= rot(b, 8);  b += a;		\
		a -= c;  a ^= rot(c,16);  c += b;		\
		b -= a;  b ^= rot(a,19);  a += c;		\
		c -= b;  c ^= rot(b, 4);  b += a;		\
	}

#define final(a,b,c)							\
	{											\
		c ^= b; c -= rot(b,14);					\
		a ^= c; a -= rot(c,11);					\
		b ^= a; b -= rot(a,25);					\
		c ^= b; c -= rot(b,16);					\
		a ^= c; a -= rot(c, 4);					\
		b ^= a; b -= rot(a,14);					\
		c ^= b; c -= rot(b,24);					\
	}

__device__ cl_uint
pg_hash_any(const cl_uchar *k, cl_int keylen)
{
	cl_uint		a, b, c;
	cl_uint		len;

	/* Set up the internal state */
	len = keylen;
	a = b = c = 0x9e3779b9 + len + 3923095;

	/* If the source pointer is word-aligned, we use word-wide fetches */
	if (((uintptr_t) k & (sizeof(cl_uint) - 1)) == 0)
	{
		/* Code path for aligned source data */
		const cl_uint	*ka = (const cl_uint *) k;

		/* handle most of the key */
		while (len >= 12)
		{
			a += ka[0];
			b += ka[1];
			c += ka[2];
			mix(a, b, c);
			ka += 3;
			len -= 12;
		}

		/* handle the last 11 bytes */
		k = (const unsigned char *) ka;
		switch (len)
		{
			case 11:
				c += ((cl_uint) k[10] << 24);
				/* fall through */
			case 10:
				c += ((cl_uint) k[9] << 16);
				/* fall through */
			case 9:
				c += ((cl_uint) k[8] << 8);
				/* the lowest byte of c is reserved for the length */
				/* fall through */
			case 8:
				b += ka[1];
				a += ka[0];
				break;
			case 7:
				b += ((cl_uint) k[6] << 16);
				/* fall through */
			case 6:
				b += ((cl_uint) k[5] << 8);
				/* fall through */
			case 5:
				b += k[4];
				/* fall through */
			case 4:
				a += ka[0];
				break;
			case 3:
				a += ((cl_uint) k[2] << 16);
				/* fall through */
			case 2:
				a += ((cl_uint) k[1] << 8);
				/* fall through */
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
	}
	else
	{
		/* Code path for non-aligned source data */

		/* handle most of the key */
		while (len >= 12)
		{
			a += k[0] + (((cl_uint) k[1] << 8) +
						 ((cl_uint) k[2] << 16) +
						 ((cl_uint) k[3] << 24));
			b += k[4] + (((cl_uint) k[5] << 8) +
						 ((cl_uint) k[6] << 16) +
						 ((cl_uint) k[7] << 24));
			c += k[8] + (((cl_uint) k[9] << 8) +
						 ((cl_uint) k[10] << 16) +
						 ((cl_uint) k[11] << 24));
			mix(a, b, c);
			k += 12;
			len -= 12;
		}

		/* handle the last 11 bytes */
		switch (len)            /* all the case statements fall through */
		{
			case 11:
				c += ((cl_uint) k[10] << 24);
			case 10:
				c += ((cl_uint) k[9] << 16);
			case 9:
				c += ((cl_uint) k[8] << 8);
				/* the lowest byte of c is reserved for the length */
			case 8:
				b += ((cl_uint) k[7] << 24);
			case 7:
				b += ((cl_uint) k[6] << 16);
			case 6:
				b += ((cl_uint) k[5] << 8);
			case 5:
				b += k[4];
			case 4:
				a += ((cl_uint) k[3] << 24);
			case 3:
				a += ((cl_uint) k[2] << 16);
			case 2:
				a += ((cl_uint) k[1] << 8);
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
	}
	final(a, b, c);

	return c;
}
#undef rot
#undef mix
#undef final

/*
 * varlena compress/decompress functions
 */

/*
 * toast_raw_datum_size - return the raw (detoasted) size of a varlena
 * datum (including the VARHDRSZ header)
 */
DEVICE_FUNCTION(size_t)
toast_raw_datum_size(kern_context *kcxt, varlena *attr)
{
	size_t		result;

	if (VARATT_IS_EXTERNAL(attr))
	{
		if (VARATT_IS_EXTERNAL_ONDISK(attr))
		{
			varatt_external	va_ext;

			memcpy(&va_ext, ((varattrib_1b_e *)attr)->va_data,
				   sizeof(varatt_external));
			result = va_ext.va_rawsize;
		}
		else
		{
			/* should not appear in the kernel space */
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			result = 0;
		}
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
 * toast_decompress_datum - decompress a compressed version of a varlena datum
 */
DEVICE_FUNCTION(cl_int)
pglz_decompress(const char *source, cl_int slen,
				char *dest, cl_int rawsize)
{
	const cl_uchar *sp;
	const cl_uchar *srcend;
	cl_uchar	   *dp;
	cl_uchar	   *destend;

	sp = (const cl_uchar *) source;
	srcend = ((const cl_uchar *) source) + slen;
	dp = (cl_uchar *) dest;
	destend = dp + rawsize;

	while (sp < srcend && dp < destend)
	{
		/*
		 * Read one control byte and process the next 8 items (or as many as
		 * remain in the compressed input).
		 */
		cl_uchar	ctrl = *sp++;
		int			ctrlc;

		for (ctrlc = 0; ctrlc < 8 && sp < srcend; ctrlc++)
		{
			if (ctrl & 1)
			{
				/*
				 * Otherwise it contains the match length minus 3 and the
				 * upper 4 bits of the offset. The next following byte
				 * contains the lower 8 bits of the offset. If the length is
				 * coded as 18, another extension tag byte tells how much
				 * longer the match really was (0-255).
				 */
				cl_int		len;
				cl_int		off;

				len = (sp[0] & 0x0f) + 3;
				off = ((sp[0] & 0xf0) << 4) | sp[1];
				sp += 2;
				if (len == 18)
					len += *sp++;

				/*
				 * Check for output buffer overrun, to ensure we don't clobber
				 * memory in case of corrupt input.  Note: we must advance dp
				 * here to ensure the error is detected below the loop.  We
				 * don't simply put the elog inside the loop since that will
				 * probably interfere with optimization.
				 */
				if (dp + len > destend)
				{
					dp += len;
					break;
				}
				/*
				 * Now we copy the bytes specified by the tag from OUTPUT to
				 * OUTPUT. It is dangerous and platform dependent to use
				 * memcpy() here, because the copied areas could overlap
				 * extremely!
				 */
				while (len--)
				{
					*dp = dp[-off];
					dp++;
				}
			}
			else
			{
				/*
				 * An unset control bit means LITERAL BYTE. So we just copy
				 * one from INPUT to OUTPUT.
				 */
				if (dp >= destend)		/* check for buffer overrun */
					break;				/* do not clobber memory */

				*dp++ = *sp++;
			}

			/*
			 * Advance the control bit
			 */
			ctrl >>= 1;
		}
	}

	/*
	 * Check we decompressed the right amount.
	 */
	if (dp != destend || sp != srcend)
		return -1;

	/*
	 * That's it.
	 */
	return rawsize;
}

DEVICE_FUNCTION(cl_bool)
toast_decompress_datum(char *buffer, cl_uint buflen,
					   const varlena *datum)
{
	cl_int		rawsize;

	assert(VARATT_IS_COMPRESSED(datum));
	rawsize = TOAST_COMPRESS_RAWSIZE(datum);
	if (rawsize + VARHDRSZ > buflen)
		return false;
	SET_VARSIZE(buffer, rawsize + VARHDRSZ);
	if (pglz_decompress(TOAST_COMPRESS_RAWDATA(datum),
						VARSIZE(datum) - TOAST_COMPRESS_HDRSZ,
						buffer + VARHDRSZ,
						rawsize) < 0)
	{
		printf("GPU kernel: compressed varlena datum is corrupted\n");
		return false;
	}
	return true;
}
