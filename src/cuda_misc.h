/*
 * cuda_misc.h
 *
 * Collection of various data type support on CUDA devices
 * --
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
#ifndef CUDA_MISC_H
#define CUDA_MISC_H
#ifdef __CUDACC__

/* pg_money_t */
#ifndef PG_MONEY_TYPE_DEFINED
#define PG_MONEY_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(money, cl_long)
STATIC_INLINE(Datum)
pg_monery_as_datum(void *addr)
{
	cl_long		val = *((cl_long *)addr);
	return SET_8_BYTES(val);
}
STROMCL_SIMPLE_COMPARE_TEMPLATE(cash_,money,money,cl_long)
#endif

/*
 * Cast function to currency data type
 */
#ifdef PG_NUMERIC_TYPE_DEFINED
STATIC_FUNCTION(pg_money_t)
pgfn_numeric_cash(kern_context *kcxt, pg_numeric_t arg1)
{
	pg_int8			temp = { PGLC_CURRENCY_SCALE, false };
	pg_numeric_t	div;
	pg_money_t		result;

	div = pgfn_int8_numeric(kcxt, temp);
	temp = pgfn_numeric_int8(kcxt, pgfn_numeric_mul(kcxt, arg1, div));
	result.isnull = temp.isnull;
	result.value = temp.value;

	return result;
}
#endif

STATIC_FUNCTION(pg_money_t)
pgfn_int4_cash(kern_context *kcxt, pg_int4_t arg1)
{
	pg_money_t	result;

	if (arg1.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_long) arg1.value / (cl_long)PGLC_CURRENCY_SCALE;
	}
	return result;
}

STATIC_FUNCTION(pg_money_t)
pgfn_int8_cash(kern_context *kcxt, pg_int8_t arg1)
{
	pg_money_t	result;

	if (arg1.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_long) arg1.value / (cl_long)PGLC_CURRENCY_SCALE;
	}
	return result;
}

/*
 * Currency operator functions
 */
STATIC_FUNCTION(pg_money_t)
pgfn_cash_pl(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_money_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = arg1.value + arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_money_t)
pgfn_cash_mi(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_money_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = arg1.value + arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_float8_t)
pgfn_cash_div_cash(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_float8_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
		else
		{
			result.isnull = false;
			result.value = (cl_double)arg1.value / (cl_double)arg2.value;
		}
	}
	return result;
}

#define PGFN_MONEY_MULFUNC_TEMPLATE(name,d_type,d_cast)				\
	STATIC_FUNCTION(pg_money_t)										\
	pgfn_cash_mul_##name(kern_context *kcxt,						\
						 pg_money_t arg1, pg_##d_type##_t arg2)		\
	{																\
		pg_money_t	result;											\
																	\
		if (arg1.isnull || arg2.isnull)								\
			result.isnull = true;									\
		else														\
		{															\
			result.isnull = false;									\
			result.value = arg1.value * (d_cast)arg2.value;			\
		}															\
		return result;												\
	}

PGFN_MONEY_MULFUNC_TEMPLATE(int2, int2, cl_long)
PGFN_MONEY_MULFUNC_TEMPLATE(int4, int4, cl_long)
//PGFN_MONEY_MULFUNC_TEMPLATE(int8, int8)
PGFN_MONEY_MULFUNC_TEMPLATE(flt2, float2, cl_float)
PGFN_MONEY_MULFUNC_TEMPLATE(flt4, float4, cl_float)
PGFN_MONEY_MULFUNC_TEMPLATE(flt8, float8, cl_double)
#undef PGFN_MONEY_MULFUNC_TEMPLATE

#define PGFN_MONEY_DIVFUNC_TEMPLATE(name,d_type,zero)				\
	STATIC_FUNCTION(pg_money_t)										\
	pgfn_cash_div_##name(kern_context *kcxt,						\
						 pg_money_t arg1, pg_##d_type##_t arg2)		\
	{																\
		pg_money_t	result;											\
																	\
		if (arg1.isnull || arg2.isnull)								\
			result.isnull = true;									\
		else														\
		{															\
			if (arg2.value == (zero))								\
			{														\
				result.isnull = true;								\
				STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);	\
			}														\
			else													\
			{														\
				result.isnull = false;								\
				result.value = rint((cl_double)arg1.value /			\
									(cl_double)arg2.value);			\
			}														\
		}															\
		return result;												\
	}

PGFN_MONEY_DIVFUNC_TEMPLATE(int2, int2, 0)
PGFN_MONEY_DIVFUNC_TEMPLATE(int4, int4, 0)
//PGFN_MONEY_DIVFUNC_TEMPLATE(int8, int8, 0)
PGFN_MONEY_DIVFUNC_TEMPLATE(flt2, float2, (__half)0.0)
PGFN_MONEY_DIVFUNC_TEMPLATE(flt4, float4, 0.0)
PGFN_MONEY_DIVFUNC_TEMPLATE(flt8, float8, 0.0)
#undef PGFN_MONEY_DIVFUNC_TEMPLATE

STATIC_INLINE(pg_money_t)
pgfn_int2_mul_cash(kern_context *kcxt, pg_int2_t arg1, pg_money_t arg2)
{
	return pgfn_cash_mul_int2(kcxt, arg2, arg1);
}

STATIC_INLINE(pg_money_t)
pgfn_int4_mul_cash(kern_context *kcxt, pg_int4_t arg1, pg_money_t arg2)
{
	return pgfn_cash_mul_int4(kcxt, arg2, arg1);
}

STATIC_INLINE(pg_money_t)
pgfn_flt2_mul_cash(kern_context *kcxt, pg_float2_t arg1, pg_money_t arg2)
{
	return pgfn_cash_mul_flt2(kcxt, arg2, arg1);
}

STATIC_INLINE(pg_money_t)
pgfn_flt4_mul_cash(kern_context *kcxt, pg_float4_t arg1, pg_money_t arg2)
{
	return pgfn_cash_mul_flt4(kcxt, arg2, arg1);
}

STATIC_INLINE(pg_money_t)
pgfn_flt8_mul_cash(kern_context *kcxt, pg_float8_t arg1, pg_money_t arg2)
{
	return pgfn_cash_mul_flt8(kcxt, arg2, arg1);
}

/*
 * Currency comparison functions
 */
STATIC_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_int4_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		if (arg1.value > arg2.value)
			result.value = 1;
		else if (arg1.value < arg2.value)
			result.value = -1;
		else
			result.value = 0;
	}
	return result;
}

/* pg_uuid_t */
#define UUID_LEN 16
typedef struct
{
	cl_uchar data[UUID_LEN];
} pgsql_uuid_t;

#ifndef PG_UUID_TYPE_DEFINED
#define PG_UUID_TYPE_DEFINED
STROMCL_INDIRECT_TYPE_TEMPLATE(uuid, pgsql_uuid_t)
#endif	/* PG_UUID_TYPE_DEFINED */

STATIC_INLINE(int)
uuid_internal_cmp(pg_uuid_t *arg1, pg_uuid_t *arg2)
{
	cl_uchar   *data1 = arg1->value.data;
	cl_uchar   *data2 = arg2->value.data;
	cl_int		i, cmp = 0;

	for (i=0; cmp == 0 && i < UUID_LEN; i++)
		cmp = (int)data1[i] - (int)data2[i];
	return cmp;
}

STATIC_INLINE(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2)
{
	pg_int4_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = uuid_internal_cmp(&arg1, &arg2);
	}
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_uuid_lt(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(uuid_internal_cmp(&arg1, &arg2) < 0);
	}
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_uuid_le(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(uuid_internal_cmp(&arg1, &arg2) <= 0);
	}
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_uuid_eq(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(uuid_internal_cmp(&arg1, &arg2) == 0);
	}
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_uuid_ge(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(uuid_internal_cmp(&arg1, &arg2) >= 0);
	}
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_uuid_gt(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(uuid_internal_cmp(&arg1, &arg2) > 0);
	}
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_uuid_ne(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(uuid_internal_cmp(&arg1, &arg2) != 0);
	}
	return result;
}

/*
 * Data Types for network address types
 * ---------------------------------------------------------------- */

/* pg_macaddr_t */
typedef struct macaddr
{
	cl_uchar	a;
	cl_uchar	b;
	cl_uchar	c;
	cl_uchar	d;
	cl_uchar	e;
	cl_uchar	f;
} macaddr;

#define hibits(addr) \
	((unsigned long)(((addr)->a<<16)|((addr)->b<<8)|((addr)->c)))

#define lobits(addr) \
	((unsigned long)(((addr)->d<<16)|((addr)->e<<8)|((addr)->f)))

#ifndef PG_MACADDR_TYPE_DEFINED
#define PG_MACADDR_TYPE_DEFINED
STROMCL_INDIRECT_TYPE_TEMPLATE(macaddr,macaddr)
#endif	/* PG_MACADDR_TYPE_DEFINED */

STATIC_FUNCTION(pg_macaddr_t)
pgfn_macaddr_trunc(kern_context *kcxt, pg_macaddr_t arg1)
{
	arg1.value.d = 0;
	arg1.value.e = 0;
	arg1.value.f = 0;

	return arg1;
}

STATIC_INLINE(cl_int)
macaddr_cmp_internal(macaddr *a1, macaddr *a2)
{
	if (hibits(a1) < hibits(a2))
		return -1;
	else if (hibits(a1) > hibits(a2))
		return 1;
	else if (lobits(a1) < lobits(a2))
		return -1;
	else if (lobits(a1) > lobits(a2))
		return 1;
	else
		return 0;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_macaddr_eq(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (macaddr_cmp_internal(&arg1.value, &arg2.value) == 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_macaddr_lt(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (macaddr_cmp_internal(&arg1.value, &arg2.value) < 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_macaddr_le(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (macaddr_cmp_internal(&arg1.value, &arg2.value) <= 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_macaddr_gt(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (macaddr_cmp_internal(&arg1.value, &arg2.value) > 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_macaddr_ge(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (macaddr_cmp_internal(&arg1.value, &arg2.value) >= 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_macaddr_ne(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (macaddr_cmp_internal(&arg1.value, &arg2.value) != 0);
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = macaddr_cmp_internal(&arg1.value, &arg2.value);
	return result;
}

STATIC_FUNCTION(pg_macaddr_t)
pgfn_macaddr_not(kern_context *kcxt, pg_macaddr_t arg1)
{
	pg_macaddr_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		result.value.a	= ~arg1.value.a;
		result.value.b	= ~arg1.value.b;
		result.value.c	= ~arg1.value.c;
		result.value.d	= ~arg1.value.d;
		result.value.e	= ~arg1.value.e;
		result.value.f	= ~arg1.value.f;
	}
	return result;
}

STATIC_FUNCTION(pg_macaddr_t)
pgfn_macaddr_and(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2)
{
	pg_macaddr_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value.a	= arg1.value.a & arg2.value.a;
		result.value.b	= arg1.value.b & arg2.value.b;
		result.value.c	= arg1.value.c & arg2.value.c;
		result.value.d	= arg1.value.d & arg2.value.d;
		result.value.e	= arg1.value.e & arg2.value.e;
		result.value.f	= arg1.value.f & arg2.value.f;
	}
	return result;
}

STATIC_FUNCTION(pg_macaddr_t)
pgfn_macaddr_or(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2)
{
	pg_macaddr_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value.a	= arg1.value.a | arg2.value.a;
		result.value.b	= arg1.value.b | arg2.value.b;
		result.value.c	= arg1.value.c | arg2.value.c;
		result.value.d	= arg1.value.d | arg2.value.d;
		result.value.e	= arg1.value.e | arg2.value.e;
		result.value.f	= arg1.value.f | arg2.value.f;
	}
	return result;
}

/* pg_inet_t */

/*
 *  This is the internal storage format for IP addresses
 *  (both INET and CIDR datatypes):
 */
typedef struct
{
	cl_uchar	family;		/* PGSQL_AF_INET or PGSQL_AF_INET6 */
	cl_uchar	bits;		/* number of bits in netmask */
	cl_uchar	ipaddr[16];	/* up to 128 bits of address */
} inet_struct;

#define PGSQL_AF_INET		(AF_INET + 0)
#define PGSQL_AF_INET6		(AF_INET + 1)

typedef struct
{
	char		vl_len_[4];	/* Do not touch this field directly! */
	inet_struct	inet_data;
} inet;

#define ip_family(inetptr)		(inetptr)->family
#define ip_bits(inetptr)		(inetptr)->bits
#define ip_addr(inetptr)		(inetptr)->ipaddr
#define ip_addrsize(inetptr)	\
	((inetptr)->family == PGSQL_AF_INET ? 4 : 16)
#define ip_maxbits(inetptr)		\
	((inetptr)->family == PGSQL_AF_INET ? 32 : 128)

#ifndef PG_INET_TYPE_DEFINED
#define PG_INET_TYPE_DEFINED
STROMCL_SIMPLE_DATATYPE_TEMPLATE(inet,inet_struct)

STATIC_INLINE(pg_inet_t)
pg_inet_datum_ref(kern_context *kcxt, void *datum)
{
	pg_inet_t	result;

	result.isnull = !datum;
	if (datum)
	{
		if (VARATT_IS_COMPRESSED(datum))
		{
			inet	temp;

			if (toast_decompress_datum((char *)&temp, sizeof(inet),
									   (struct varlena *)datum))
			{
				memcpy(&result.value, &temp.inet_data, sizeof(inet_struct));
			}
			else
			{
				STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
				result.isnull = true;
			}
		}
		else if (VARATT_IS_EXTERNAL(datum) ||
				 VARSIZE_ANY_EXHDR(datum) < offsetof(inet_struct, ipaddr))
		{
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			result.isnull = true;
		}
		else
		{
			inet_struct	   *ip_data = (inet_struct *)VARDATA_ANY(datum);
			cl_int			ip_size = ip_addrsize(ip_data);

			if (VARSIZE_ANY_EXHDR(datum) >= offsetof(inet_struct,
													 ipaddr[ip_size]))
			{
				memcpy(&result.value, VARDATA_ANY(datum),
					   offsetof(inet_struct, ipaddr[ip_size]));
				result.isnull = false;
			}
			else
			{
				STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
				result.isnull = true;
			}
		}
	}
	return result;
}
STATIC_INLINE(void)
pg_datum_ref(kern_context *kcxt, pg_inet_t &result, void *datum)
{
	result = pg_inet_datum_ref(kcxt, datum);
}

STATIC_INLINE(void *)
pg_inet_datum_store(kern_context *kcxt, pg_inet_t datum)
{
	char	   *pos;

	if (datum.isnull)
		return NULL;
	pos = (char *)MAXALIGN(kcxt->vlpos);
	if (!PTR_ON_VLBUF(kcxt, pos, VARHDRSZ + sizeof(inet_struct)))
	{
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		return NULL;
	}
	SET_VARSIZE(pos, VARHDRSZ + sizeof(inet_struct));
	memcpy(pos + VARHDRSZ, &datum.value, sizeof(inet_struct));
	kcxt->vlpos += VARHDRSZ + sizeof(inet_struct);
	return pos;
}

STATIC_FUNCTION(pg_inet_t)
pg_inet_param(kern_context *kcxt, cl_uint param_id)
{
	kern_parambuf  *kparams = kcxt->kparams;
	void		   *paddr;

	if (param_id < kparams->nparams &&
		kparams->poffset[param_id] > 0)
		paddr = ((char *)kparams + kparams->poffset[param_id]);
	else
		paddr = NULL;

	return pg_inet_datum_ref(kcxt,paddr);
}

STATIC_INLINE(cl_uint)
pg_inet_comp_crc32(const cl_uint *crc32_table,
				   kern_context *kcxt,
				   cl_uint hash, pg_inet_t datum)
{
	if (!datum.isnull)
	{
		int		len = (datum.value.family == PGSQL_AF_INET  ? 4 :
					   datum.value.family == PGSQL_AF_INET6 ? 16 : 0);
		if (len > 0)
		{
			hash = pg_common_comp_crc32(crc32_table, hash,
										(char *)&datum.value,
										offsetof(inet_struct, ipaddr[len]));
		}
		else
		{
			STROM_SET_ERROR(&kcxt->e, StromError_InvalidValue);
		}
	}
	return hash;
}
#endif	/* PG_INET_TYPE_DEFINED */

#ifndef PG_CIDR_TYPE_DEFINED
#define PG_CIDR_TYPE_DEFINED
typedef pg_inet_t					pg_cidr_t;
#define pg_cidr_datum_ref(a,b)		pg_inet_datum_ref(a,b)
#define pg_cidr_datum_store(a,b,c)	pg_inet_datum_store(a,b,c)
#define pg_cidr_param(a,b)			pg_inet_param(a,b)
#define pg_cidr_isnull(a,b)			pg_inet_isnull(a,b)
#define pg_cidr_isnotnull(a,b)		pg_inet_isnotnull(a,b,c)
#define pg_cidr_comp_crc32(a,b,c)	pg_inet_comp_crc32(a,b,c)
#define pg_cidr_as_datum(a)			pg_inet_as_datum(a)
#endif	/* PG_CIDR_TYPE_DEFINED */

/* memory comparison */
STATIC_INLINE(cl_int)
__memcmp(const void *s1, const void *s2, size_t n)
{
	const cl_uchar *p1 = (const cl_uchar *)s1;
	const cl_uchar *p2 = (const cl_uchar *)s2;

	while (n--)
	{
		if (*p1 != *p2)
			return ((int)*p1) - ((int)*p2);
		p1++;
		p2++;
	}
	return 0;
}

/*
 * int
 * bitncmp(l, r, n)
 *      compare bit masks l and r, for n bits.
 * return:
 *      <0, >0, or 0 in the libc tradition.
 * note:
 *      network byte order assumed.  this means 192.5.5.240/28 has
 *      0x11110000 in its fourth octet.
 * author:
 *      Paul Vixie (ISC), June 1996
 */
#define IS_HIGHBIT_SET(ch)		((unsigned char)(ch) & 0x80)

STATIC_INLINE(int)
bitncmp(const unsigned char *l, const unsigned char *r, int n)
{
	unsigned int	lb, rb;
	int				x, b;

	b = n / 8;
	x = __memcmp(l, r, b);
	if (x || (n % 8) == 0)
		return x;

	lb = l[b];
	rb = r[b];
	for (b = n % 8; b > 0; b--)
	{
		if (IS_HIGHBIT_SET(lb) != IS_HIGHBIT_SET(rb))
		{
			if (IS_HIGHBIT_SET(lb))
				return 1;
			return -1;
		}
		lb <<= 1;
		rb <<= 1;
	}
	return 0;
}

STATIC_INLINE(cl_int)
network_cmp_internal(inet_struct *a1, inet_struct *a2)
{
	if (ip_family(a1) == ip_family(a2))
	{
		int		order;

		order = bitncmp(ip_addr(a1), ip_addr(a2),
						Min(ip_bits(a1), ip_bits(a2)));
		if (order != 0)
			return order;
		order = ((int) ip_bits(a1)) - ((int) ip_bits(a2));
		if (order != 0)
			return order;
		return bitncmp(ip_addr(a1), ip_addr(a2), ip_maxbits(a1));
	}
	return ip_family(a1) - ip_family(a2);
}

/*
 * network_lt
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_network_lt(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (network_cmp_internal(&arg1.value, &arg2.value) < 0);
	return result;
}

/*
 * network_le
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_network_le(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (network_cmp_internal(&arg1.value, &arg2.value) <= 0);
	return result;
}

/*
 * network_eq
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_network_eq(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (network_cmp_internal(&arg1.value, &arg2.value) == 0);
	return result;
}

/*
 * network_ge
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_network_ge(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (network_cmp_internal(&arg1.value, &arg2.value) >= 0);
	return result;
}

/*
 * network_gt
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_network_gt(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (network_cmp_internal(&arg1.value, &arg2.value) > 0);
	return result;
}

/*
 * network_ne
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_network_ne(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (network_cmp_internal(&arg1.value, &arg2.value) != 0);
	return result;
}

/*
 * network_cmp
 */
STATIC_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = network_cmp_internal(&arg1.value, &arg2.value);
	return result;
}

/*
 * network_larger
 */
STATIC_FUNCTION(pg_inet_t)
pgfn_network_larger(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	if (arg1.isnull || arg2.isnull)
	{
		pg_inet_t	dummy = { 0, true };

		return dummy;
	}

	if (network_cmp_internal(&arg1.value, &arg2.value) > 0)
		return arg1;
	else
		return arg2;
}

/*
 * network_smaller
 */
STATIC_FUNCTION(pg_inet_t)
pgfn_network_smaller(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	if (arg1.isnull || arg2.isnull)
	{
		pg_inet_t	dummy = { 0, true };

		return dummy;
	}

	if (network_cmp_internal(&arg1.value, &arg2.value) < 0)
		return arg1;
	else
		return arg2;
}

/*
 * network_sub
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_network_sub(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (ip_family(&arg1.value) == ip_family(&arg2.value) &&
			ip_bits(&arg1.value) > ip_bits(&arg2.value))
			result.value = (bitncmp(ip_addr(&arg1.value),
									ip_addr(&arg2.value),
									ip_bits(&arg2.value)) == 0);
		else
			result.value = false;
	}
	return result;
}

/*
 * network_subeq
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_network_subeq(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (ip_family(&arg1.value) == ip_family(&arg2.value) &&
			ip_bits(&arg1.value) >= ip_bits(&arg2.value))
			result.value = (bitncmp(ip_addr(&arg1.value),
									ip_addr(&arg2.value),
									ip_bits(&arg2.value)) == 0);
		else
			result.value = false;
	}
	return result;
}

/*
 * network_sup
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_network_sup(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (ip_family(&arg1.value) == ip_family(&arg2.value) &&
			ip_bits(&arg1.value) < ip_bits(&arg2.value))
			result.value = (bitncmp(ip_addr(&arg1.value),
									ip_addr(&arg2.value),
									ip_bits(&arg1.value)) == 0);
		else
			result.value = false;
	}
	return result;

}

/*
 * network_supeq(inet)
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_network_supeq(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (ip_family(&arg1.value) == ip_family(&arg2.value) &&
			ip_bits(&arg1.value) <= ip_bits(&arg2.value))
			result.value = (bitncmp(ip_addr(&arg1.value),
									ip_addr(&arg2.value),
									ip_bits(&arg1.value)) == 0);
		else
			result.value = false;
	}
	return result;
}

/*
 * network_overlap(inet)
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_network_overlap(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (ip_family(&arg1.value) == ip_family(&arg2.value))
			result.value = (bitncmp(ip_addr(&arg1.value),
									ip_addr(&arg2.value),
									Min(ip_bits(&arg1.value),
										ip_bits(&arg2.value))) == 0);
		else
			result.value = false;
	}
	return result;
}

/*
 * set_masklen(inet,int)
 */
STATIC_FUNCTION(pg_inet_t)
pgfn_inet_set_masklen(kern_context *kcxt, pg_inet_t arg1, pg_int4_t arg2)
{
	pg_inet_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		cl_int		bits = arg2.value;

		if (bits == -1)
			bits = ip_maxbits(&arg1.value);
		if (bits < 0 || bits > ip_maxbits(&arg1.value))
		{
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			result.isnull = true;
		}
		else
		{
			memcpy(&result.value, &arg1.value, sizeof(inet_struct));
			ip_bits(&result.value) = bits;
		}
	}
	return result;
}

/*
 * set_masklen(cidr,int)
 */
STATIC_FUNCTION(pg_cidr_t)
pgfn_cidr_set_masklen(kern_context *kcxt, pg_cidr_t arg1, pg_int4_t arg2)
{
	pg_cidr_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		cl_int	bits = arg2.value;
		cl_int	byte;
		cl_int	nbits;
		cl_int	maxbytes;

		if (bits == -1)
			bits = ip_maxbits(&arg1.value);
		if (bits < 0 || bits > ip_maxbits(&arg1.value))
		{
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			result.isnull = true;
		}
		else
		{
			/* clone the original data */
			memcpy(&result.value, &arg1.value, sizeof(inet_struct));
			ip_bits(&result.value) = bits;

			/* zero out any bits to the right of the new netmask */
			byte = bits / 8;
			nbits = bits % 8;
			/* clear the first byte, this might be a partial byte */
			if (nbits != 0)
			{
				ip_addr(&result.value)[byte] &= ~(0xff >> nbits);
				byte++;
			}
			/* clear remaining bytes */
			maxbytes = ip_addrsize(&result.value);
			while (byte < maxbytes)
			{
				ip_addr(&result.value)[byte] = 0;
				byte++;
			}
		}
	}
	return result;
}

/*
 * family(inet)
 */
STATIC_FUNCTION(pg_int4_t)
pgfn_inet_family(kern_context *kcxt, pg_inet_t arg1)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		switch (ip_family(&arg1.value))
		{
			case PGSQL_AF_INET:
				result.value = 4;
				break;
			case PGSQL_AF_INET6:
				result.value = 6;
				break;
			default:
				result.value = 0;
				break;
		}
	}
	return result;
}

/*
 * network(inet)
 */
STATIC_FUNCTION(pg_cidr_t)
pgfn_network_network(kern_context *kcxt, pg_inet_t arg1)
{
	pg_cidr_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		cl_int		byte = 0;
		cl_int		bits;
		cl_uchar	mask;
		cl_uchar   *a, *b;

		/* make sure any unused bits are zeroed */
		memset(&result.value, 0, sizeof(result.value));
		bits = ip_bits(&arg1.value);
		a = ip_addr(&arg1.value);
		b = ip_addr(&result.value);

		while (bits)
		{
			if (bits >= 8)
			{
				mask = 0xff;
				bits -= 8;
			}
			else
			{
				mask = 0xff << (8 - bits);
				bits = 0;
			}
			b[byte] = a[byte] & mask;
			byte++;
		}
		ip_family(&result.value) = ip_family(&arg1.value);
		ip_bits(&result.value) = ip_bits(&arg1.value);
	}
	return result;
}

/*
 * netmask(inet)
 */
STATIC_FUNCTION(pg_inet_t)
pgfn_inet_netmask(kern_context *kcxt, pg_inet_t arg1)
{
	pg_inet_t	result;
	cl_int		byte;
	cl_int		bits;
	cl_uchar	mask;
	cl_uchar   *a, *b;

	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		byte = 0;
		bits = ip_bits(&arg1.value);
		a = ip_addr(&arg1.value);
		b = ip_addr(&result.value);

		while (bits)
		{
			if (bits >= 8)
			{
				mask = 0xff;
				bits -= 8;
			}
			else
			{
				mask = 0xff << (8 - bits);
				bits = 0;
			}
			b[byte] = a[byte] & mask;
			byte++;
		}

		ip_family(&result.value) = ip_family(&arg1.value);
		ip_bits(&result.value) = ip_family(&arg1.value);
	}
	return result;
}

/*
 * masklen(inet)
 */
STATIC_FUNCTION(pg_int4_t)
pgfn_inet_masklen(kern_context *kcxt, pg_inet_t arg1)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
		result.value = ip_bits(&arg1.value);
	return result;
}

/*
 * broadcast(inet)
 */
STATIC_FUNCTION(pg_inet_t)
pgfn_inet_broadcast(kern_context *kcxt, pg_inet_t arg1)
{
	pg_inet_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		cl_int		byte;
		cl_int		bits;
		cl_int		maxbytes;
		cl_uchar	mask;
		cl_uchar   *a, *b;

		/* make sure any unused bits are zeroed */
		memset(&result.value, 0, sizeof(result.value));
		if (ip_family(&arg1.value))
			maxbytes = 4;
		else
			maxbytes = 16;

		bits = ip_bits(&arg1.value);
		a = ip_addr(&arg1.value);
		b = ip_addr(&result.value);

		for (byte = 0; byte < maxbytes; byte++)
		{
			if (bits >= 8)
			{
				mask = 0x00;
				bits -= 8;
			}
			else if (bits == 0)
				mask = 0xff;
			else
			{
				mask = 0xff >> bits;
				bits = 0;
			}
			b[byte] = a[byte] | mask;
		}
		ip_family(&result.value) = ip_family(&arg1.value);
		ip_bits(&result.value) = ip_bits(&arg1.value);
	}
	return result;
}

/*
 * host(inet)
 */
STATIC_FUNCTION(pg_inet_t)
pgfn_inet_hostmask(kern_context *kcxt, pg_inet_t arg1)
{
	pg_inet_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		cl_int		byte;
		cl_int		bits;
		cl_int		maxbytes;
		cl_uchar	mask;
		cl_uchar   *b;

		/* make sure any unused bits are zeroed */
		memset(&result.value, 0, sizeof(result.value));

		if (ip_family(&arg1.value) == PGSQL_AF_INET)
			maxbytes = 4;
		else
			maxbytes = 16;

		bits = ip_maxbits(&arg1.value) - ip_bits(&arg1.value);
		b = ip_addr(&result.value);

		byte = maxbytes - 1;

		while (bits)
		{
			if (bits >= 8)
			{
				mask = 0xff;
				bits -= 8;
			}
			else
			{
				mask = 0xff >> (8 - bits);
				bits = 0;
			}
			b[byte] = mask;
			byte--;
		}
		ip_family(&result.value) = ip_family(&arg1.value);
		ip_bits(&result.value) = ip_maxbits(&arg1.value);
	}
	return result;
}

/*
 * cidr(inet)
 */
STATIC_FUNCTION(pg_cidr_t)
pgfn_inet_to_cidr(kern_context *kcxt, pg_inet_t arg1)
{
	pg_cidr_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		cl_int	bits = ip_bits(&arg1.value);
		cl_int	byte;
		cl_int	nbits;
		cl_int	maxbytes;

		/* sanity check */
		if (bits < 0 || bits > ip_maxbits(&arg1.value))
		{
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			result.isnull = true;
		}
		else
		{
			/* clone the original data */
			memcpy(&result.value, &arg1.value, sizeof(inet_struct));

			/* zero out any bits to the right of the netmask */
			byte = bits / 8;

			nbits = bits % 8;
			/* clear the first byte, this might be a partial byte */
			if (nbits != 0)
			{
				ip_addr(&result.value)[byte] &= ~(0xFF >> nbits);
				byte++;
			}
			/* clear remaining bytes */
			maxbytes = ip_addrsize(&result.value);
			while (byte < maxbytes)
			{
				ip_addr(&result.value)[byte] = 0;
				byte++;
			}
		}
	}
	return result;
}

/*
 * inetnot(inet)
 */
STATIC_FUNCTION(pg_inet_t)
pgfn_inet_not(kern_context *kcxt, pg_inet_t arg1)
{
	pg_inet_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		cl_int		nb = ip_addrsize(&arg1.value);
		cl_uchar   *pip = ip_addr(&arg1.value);
		cl_uchar   *pdst = ip_addr(&result.value);

		memset(&result.value, 0, sizeof(result.value));

		while (nb-- > 0)
			pdst[nb] = ~pip[nb];

		ip_bits(&result.value) = ip_bits(&arg1.value);
		ip_family(&result.value) = ip_family(&arg1.value);
	}
	return result;
}

/*
 * inetand(inet)
 */
STATIC_FUNCTION(pg_inet_t)
pgfn_inet_and(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_inet_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		memset(&result.value, 0, sizeof(result.value));
		if (ip_family(&arg1.value) != ip_family(&arg2.value))
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
		else
		{
			cl_int		nb = ip_addrsize(&arg1.value);
			cl_uchar   *pip = ip_addr(&arg1.value);
			cl_uchar   *pip2 = ip_addr(&arg2.value);
			cl_uchar   *pdst = ip_addr(&result.value);

			while (nb-- > 0)
				pdst[nb] = pip[nb] & pip2[nb];

			ip_bits(&result.value) = Max(ip_bits(&arg1.value),
										 ip_bits(&arg2.value));
			ip_family(&result.value) = ip_family(&arg1.value);
		}
	}
	return result;
}

/*
 * inetor(inet)
 */
STATIC_FUNCTION(pg_inet_t)
pgfn_inet_or(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_inet_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		memset(&result.value, 0, sizeof(result.value));
		if (ip_family(&arg1.value) != ip_family(&arg2.value))
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
		else
		{
			cl_int		nb = ip_addrsize(&arg1.value);
			cl_uchar   *pip = ip_addr(&arg1.value);
			cl_uchar   *pip2 = ip_addr(&arg2.value);
			cl_uchar   *pdst = ip_addr(&result.value);

			while (nb-- > 0)
				pdst[nb] = pip[nb] | pip2[nb];

			ip_bits(&result.value) = Max(ip_bits(&arg1.value),
										 ip_bits(&arg2.value));
			ip_family(&result.value) = ip_family(&arg1.value);
		}
	}
	return result;
}

STATIC_INLINE(pg_inet_t)
internal_inetpl(kern_context *kcxt, inet_struct *ip, cl_long addend)
{
	pg_inet_t	result;
	cl_int		nb = ip_addrsize(ip);
	cl_uchar   *pip = ip_addr(ip);
	cl_uchar   *pdst = ip_addr(&result.value);
	int			carry = 0;

	memset(&result, 0, sizeof(result));
	while (nb-- > 0)
	{
		carry = pip[nb] + (int) (addend & 0xFF) + carry;
		pdst[nb] = (unsigned char) (carry & 0xFF);
		carry >>= 8;

		/*
		 * We have to be careful about right-shifting addend because
		 * right-shift isn't portable for negative values, while simply
		 * dividing by 256 doesn't work (the standard rounding is in the
		 * wrong direction, besides which there may be machines out there
		 * that round the wrong way).  So, explicitly clear the low-order
		 * byte to remove any doubt about the correct result of the
		 * division, and then divide rather than shift.
		 */
		addend &= ~((cl_long) 0xFF);
		addend /= 0x100;
	}

	/*
	 * At this point we should have addend and carry both zero if original
	 * addend was >= 0, or addend -1 and carry 1 if original addend was <
	 * 0.  Anything else means overflow.
	 */
	if (!((addend == 0 && carry == 0) || (addend == -1 && carry == 1)))
	{
		result.isnull = true;
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	}
	else
	{
		ip_bits(&result.value) = ip_bits(ip);
		ip_family(&result.value) = ip_family(ip);
	}
	return result;
}

/*
 * inetpl(inet,bigint)
 */
STATIC_FUNCTION(pg_inet_t)
pgfn_inetpl_int8(kern_context *kcxt, pg_inet_t arg1, pg_int8_t arg2)
{
	if (arg1.isnull | arg2.isnull)
	{
		pg_inet_t	dummy;

		dummy.isnull = true;
		return dummy;
	}
	return internal_inetpl(kcxt, &arg1.value, arg2.value);
}

/*
 * inetmi(inet,bigint)
 */
STATIC_FUNCTION(pg_inet_t)
pgfn_inetmi_int8(kern_context *kcxt, pg_inet_t arg1, pg_int8_t arg2)
{
	if (arg1.isnull | arg2.isnull)
	{
		pg_inet_t	dummy;

		dummy.isnull = true;
		return dummy;
	}
	return internal_inetpl(kcxt, &arg1.value, -arg2.value);
}

/*
 * inetmi(inet,inet)
 */
STATIC_FUNCTION(pg_int8_t)
pgfn_inetmi(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (ip_family(&arg1.value) != ip_family(&arg2.value))
		{
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			result.isnull = true;
		}
		else
		{
			/*
			 * We form the difference using the traditional complement,
			 * increment, and add rule, with the increment part being handled
			 * by starting the carry off at 1.  If you don't think integer
			 * arithmetic is done in two's complement, too bad.
			 */
			cl_int		nb = ip_addrsize(&arg1.value);
			cl_int		byte = 0;
			cl_uchar   *pip = ip_addr(&arg1.value);
			cl_uchar   *pip2 = ip_addr(&arg2.value);
			cl_int		carry = 1;
			cl_long		res = 0;

			while (nb-- > 0)
			{
				int		lobyte;

				carry = pip[nb] + (~pip2[nb] & 0xFF) + carry;
				lobyte = carry & 0xFF;
				if (byte < sizeof(cl_long))
				{
					res |= ((cl_long) lobyte) << (byte * 8);
				}
				else
				{
					/*
					 * Input wider than int64: check for overflow.
					 * All bytes to the left of what will fit should be 0 or
					 * 0xFF, depending on sign of the now-complete result.
					 */
					if ((res < 0) ? (lobyte != 0xFF) : (lobyte != 0))
					{
						STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
						result.isnull = true;
						return result;
					}
				}
				carry >>= 8;
				byte++;
			}
			/*
			 * If input is narrower than int64, overflow is not possible,
			 * but we have to do proper sign extension.
			 */
			if (carry == 0 && byte < sizeof(cl_long))
				res |= ((cl_long) -1) << (byte * 8);
			result.value = res;
		}
	}
	return result;
}

/*
 * inet_same_family(inet,inet)
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_inet_same_family(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (ip_family(&arg1.value) == ip_family(&arg2.value));
	return result;
}

#else	/* __CUDACC__ */
#include "utils/pg_locale.h"

STATIC_INLINE(void)
assign_misclib_session_info(StringInfo buf)
{
	struct lconv *lconvert = PGLC_localeconv();
	cl_int		fpoint;
	cl_long		scale;
	cl_int		i;

	/* see comments about frac_digits in cash_in() */
	fpoint = lconvert->frac_digits;
	if (fpoint < 0 || fpoint > 10)
		fpoint = 2;

	/* compute required scale factor */
	scale = 1;
	for (i=0; i < fpoint; i++)
		scale *= 10;

	appendStringInfo(
		buf,
		"#ifdef __CUDACC__\n"
		"/* ================================================\n"
		" * session information for cuda_misc.h\n"
		" * ================================================ */\n"
		"\n"
		"#define PGLC_CURRENCY_SCALE_LOG10  %d\n"
		"#define PGLC_CURRENCY_SCALE        %ld\n"
		"#define AF_INET                    %d\n"
		"\n"
		"#endif /* __CUDACC__ */\n",
		fpoint,
		scale,
		AF_INET);
}
#endif	/* __CUDACC__ */
#endif	/* CUDA_MISC_H */
