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
