/*
 * cuda_misc.h
 *
 * Collection of various data type support on CUDA devices
 * --
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#ifndef CUDA_MISCLIB_H
#define CUDA_MISCLIB_H

/* pg_money_t */
#ifndef PG_MONEY_TYPE_DEFINED
#define PG_MONEY_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(money, cl_long, )
STROMCL_SIMPLE_COMP_HASH_TEMPLATE(money, cl_long)
STROMCL_UNSUPPORTED_ARROW_TEMPLATE(money)
#endif
#ifdef __CUDACC__
/*
 * Cast function to currency data type
 */
DEVICE_FUNCTION(pg_money_t)
pgfn_numeric_cash(kern_context *kcxt, pg_numeric_t arg1);
DEVICE_FUNCTION(pg_money_t)
pgfn_int4_cash(kern_context *kcxt, pg_int4_t arg1);
DEVICE_FUNCTION(pg_money_t)
pgfn_int8_cash(kern_context *kcxt, pg_int8_t arg1);

/*
 * Currency operator functions
 */
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_pl(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_mi(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2);
DEVICE_FUNCTION(pg_float8_t)
pgfn_cash_div_cash(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_mul_int2(kern_context *kcxt, pg_money_t arg1, pg_int2_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_mul_int4(kern_context *kcxt, pg_money_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_mul_flt2(kern_context *kcxt, pg_money_t arg1, pg_float2_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_mul_flt4(kern_context *kcxt, pg_money_t arg1, pg_float4_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_mul_flt8(kern_context *kcxt, pg_money_t arg1, pg_float8_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_int2_mul_cash(kern_context *kcxt, pg_int2_t arg1, pg_money_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_int4_mul_cash(kern_context *kcxt, pg_int4_t arg1, pg_money_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_flt2_mul_cash(kern_context *kcxt, pg_float2_t arg1, pg_money_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_flt4_mul_cash(kern_context *kcxt, pg_float4_t arg1, pg_money_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_flt8_mul_cash(kern_context *kcxt, pg_float8_t arg1, pg_money_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_div_int2(kern_context *kcxt, pg_money_t arg1, pg_int2_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_div_int4(kern_context *kcxt, pg_money_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_div_flt2(kern_context *kcxt, pg_money_t arg1, pg_float2_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_div_flt4(kern_context *kcxt, pg_money_t arg1, pg_float4_t arg2);
DEVICE_FUNCTION(pg_money_t)
pgfn_cash_div_flt8(kern_context *kcxt, pg_money_t arg1, pg_float8_t arg2);
/*
 * Currency comparison functions
 */
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2);
#endif /* __CUDACC__ */

/* pg_uuid_t */
#define UUID_LEN 16
typedef struct
{
	cl_uchar data[UUID_LEN];
} pgsql_uuid_t;

#ifndef PG_UUID_TYPE_DEFINED
#define PG_UUID_TYPE_DEFINED
STROMCL_INDIRECT_TYPE_TEMPLATE(uuid, pgsql_uuid_t)
STROMCL_SIMPLE_COMP_HASH_TEMPLATE(uuid, pgsql_uuid_t)
STROMCL_UNSUPPORTED_ARROW_TEMPLATE(uuid)
#endif	/* PG_UUID_TYPE_DEFINED */
#ifdef __CUDACC__
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_uuid_lt(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_uuid_le(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_uuid_eq(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_uuid_ge(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_uuid_gt(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_uuid_ne(kern_context *kcxt, pg_uuid_t arg1, pg_uuid_t arg2);
#endif /* __CUDACC__ */

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
STROMCL_SIMPLE_COMP_HASH_TEMPLATE(macaddr,macaddr)
STROMCL_UNSUPPORTED_ARROW_TEMPLATE(macaddr)
#endif	/* PG_MACADDR_TYPE_DEFINED */
#ifdef __CUDACC__
DEVICE_FUNCTION(pg_macaddr_t)
pgfn_macaddr_trunc(kern_context *kcxt, pg_macaddr_t arg1);
DEVICE_FUNCTION(pg_bool_t)
pgfn_macaddr_eq(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_macaddr_lt(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_macaddr_le(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_macaddr_gt(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_macaddr_ge(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_macaddr_ne(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2);
DEVICE_FUNCTION(pg_macaddr_t)
pgfn_macaddr_not(kern_context *kcxt, pg_macaddr_t arg1);
DEVICE_FUNCTION(pg_macaddr_t)
pgfn_macaddr_and(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2);
DEVICE_FUNCTION(pg_macaddr_t)
pgfn_macaddr_or(kern_context *kcxt, pg_macaddr_t arg1, pg_macaddr_t arg2);
#endif /* __CUDACC__ */

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
STROMCL_EXTERNAL_VARREF_TEMPLATE(inet)
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(inet)
STROMCL_UNSUPPORTED_ARROW_TEMPLATE(inet)
#endif	/* PG_INET_TYPE_DEFINED */

#ifndef PG_CIDR_TYPE_DEFINED
#define PG_CIDR_TYPE_DEFINED
typedef pg_inet_t					pg_cidr_t;
#define pg_cidr_datum_ref(a,b)		pg_inet_datum_ref(a,b)
#define pg_cidr_param(a,b)			pg_inet_param(a,b)
#endif	/* PG_CIDR_TYPE_DEFINED */

#ifdef __CUDACC__
/* binary compatible type cast */
DEVICE_INLINE(pg_inet_t)
to_inet(pg_cidr_t arg)
{
	return arg;
}
DEVICE_FUNCTION(pg_bool_t)
pgfn_network_lt(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_network_le(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_network_eq(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_network_ge(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_network_gt(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_network_ne(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_inet_t)
pgfn_network_larger(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_inet_t)
pgfn_network_smaller(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_network_sub(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_network_subeq(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_network_sup(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_network_supeq(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_network_overlap(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_inet_t)
pgfn_inet_set_masklen(kern_context *kcxt, pg_inet_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_cidr_t)
pgfn_cidr_set_masklen(kern_context *kcxt, pg_cidr_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_inet_family(kern_context *kcxt, pg_inet_t arg1);
DEVICE_FUNCTION(pg_cidr_t)
pgfn_network_network(kern_context *kcxt, pg_inet_t arg1);
DEVICE_FUNCTION(pg_inet_t)
pgfn_inet_netmask(kern_context *kcxt, pg_inet_t arg1);
DEVICE_FUNCTION(pg_int4_t)
pgfn_inet_masklen(kern_context *kcxt, pg_inet_t arg1);
DEVICE_FUNCTION(pg_inet_t)
pgfn_inet_broadcast(kern_context *kcxt, pg_inet_t arg1);
DEVICE_FUNCTION(pg_inet_t)
pgfn_inet_hostmask(kern_context *kcxt, pg_inet_t arg1);
DEVICE_FUNCTION(pg_cidr_t)
pgfn_inet_to_cidr(kern_context *kcxt, pg_inet_t arg1);
DEVICE_FUNCTION(pg_inet_t)
pgfn_inet_not(kern_context *kcxt, pg_inet_t arg1);
DEVICE_FUNCTION(pg_inet_t)
pgfn_inet_and(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_inet_t)
pgfn_inet_or(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_inet_t)
pgfn_inetpl_int8(kern_context *kcxt, pg_inet_t arg1, pg_int8_t arg2);
DEVICE_FUNCTION(pg_inet_t)
pgfn_inetmi_int8(kern_context *kcxt, pg_inet_t arg1, pg_int8_t arg2);
DEVICE_FUNCTION(pg_int8_t)
pgfn_inetmi(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_inet_same_family(kern_context *kcxt, pg_inet_t arg1, pg_inet_t arg2);

/*
 * Misc mathematic functions
 */
DEVICE_FUNCTION(pg_float8_t)
pgfn_cbrt(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_ceil(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_exp(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_floor(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_ln(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_log10(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_dpi(kern_context *kcxt);
DEVICE_FUNCTION(pg_float8_t)
pgfn_round(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_sign(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_dsqrt(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_dpow(kern_context *kcxt, pg_float8_t arg1, pg_float8_t arg2);
DEVICE_FUNCTION(pg_float8_t)
pgfn_trunc(kern_context *kcxt, pg_float8_t arg1);
/*
 * Trigonometric function
 */
DEVICE_FUNCTION(pg_float8_t)
pgfn_degrees(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_radians(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_acos(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_asin(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_atan(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_atan2(kern_context *kcxt, pg_float8_t arg1, pg_float8_t arg2);
DEVICE_FUNCTION(pg_float8_t)
pgfn_cos(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_cot(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_sin(kern_context *kcxt, pg_float8_t arg1);
DEVICE_FUNCTION(pg_float8_t)
pgfn_tan(kern_context *kcxt, pg_float8_t arg1);

#endif	/* __CUDACC__ */
#endif	/* CUDA_MISCLIB_H */
