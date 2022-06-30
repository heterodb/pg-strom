/*
 * xpu_misclib.c
 *
 * Collection of misc functions and operators for xPU(GPU/DPU/SPU)
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 *
 */
#include "xpu_common.h"

/*
 * Currency data type (xpu_money_t), functions and operators
 */
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(money, Cash);

/*
 * UUID data type (xpu_uuid_t), functions and operators
 */
STATIC_FUNCTION(bool)
xpu_uuid_datum_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const kern_colmeta *cmeta,
				   const void *addr, int len)
{
	xpu_uuid_t *result = (xpu_uuid_t *)__result;

	memset(result, 0, sizeof(xpu_uuid_t));
	result->ops = &xpu_uuid_ops;
	if (!addr)
		result->isnull = true;
	else
	{
		if (cmeta &&
			cmeta->attopts.fixed_size_binary.byteWidth != UUID_LEN)
		{
			STROM_ELOG(kcxt, "Arrow::FixedSizeBinary has wrong byteWidth");
			return false;
		}
		memcpy(&result->value.data, addr, UUID_LEN);
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_uuid_datum_store(kern_context *kcxt,
					 char *buffer,
					 xpu_datum_t *__arg)
{
	xpu_uuid_t *arg = (xpu_uuid_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		memcpy(buffer, arg->value.data, UUID_LEN);
	return UUID_LEN;
}

PUBLIC_FUNCTION(bool)
xpu_uuid_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					xpu_datum_t *__arg)
{
	xpu_uuid_t *arg = (xpu_uuid_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(arg->value.data, UUID_LEN);
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(uuid);

/*
 * Macaddr data type (xpu_macaddr_t), functions and operators
 */
STATIC_FUNCTION(bool)
xpu_macaddr_datum_ref(kern_context *kcxt,
					  xpu_datum_t *__result,
					  const kern_colmeta *cmeta,
					  const void *addr, int len)
{
	xpu_macaddr_t *result = (xpu_macaddr_t *)__result;

	memset(result, 0, sizeof(xpu_macaddr_t));
	result->ops = &xpu_macaddr_ops;
	if (!addr)
		result->isnull = true;
	else
	{
		if (cmeta &&
			cmeta->attopts.fixed_size_binary.byteWidth != sizeof(macaddr))
		{
			STROM_ELOG(kcxt, "Arrow::FixedSizeBinary has wrong byteWidth");
			return false;
		}
		memcpy(&result->value, addr, sizeof(macaddr));
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_macaddr_datum_store(kern_context *kcxt,
						char *buffer,
						xpu_datum_t *__arg)
{
	xpu_macaddr_t *arg = (xpu_macaddr_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		memcpy(buffer, &arg->value, sizeof(macaddr));
	return sizeof(macaddr);
}

PUBLIC_FUNCTION(bool)
xpu_macaddr_datum_hash(kern_context *kcxt,
					   uint32_t *p_hash,
					   xpu_datum_t *__arg)
{
	xpu_macaddr_t *arg = (xpu_macaddr_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(macaddr));
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(macaddr);

/*
 * Inet data type (xpu_iner_t), functions and operators
 */
PUBLIC_FUNCTION(bool)
xpu_inet_datum_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const kern_colmeta *cmeta,
				   const void *addr, int len)
{
	xpu_inet_t *result = (xpu_inet_t *)__result;

	memset(result, 0, sizeof(xpu_inet_t));
	result->ops = &xpu_inet_ops;
	if (!addr)
	{
		result->isnull = true;
	}
	else if (!cmeta)
	{
		if (VARATT_IS_COMPRESSED(addr) || VARATT_IS_EXTERNAL(addr))
		{
			STROM_CPU_FALLBACK(kcxt, ERRCODE_INTERNAL_ERROR,
							   "inet value is compressed or toasted");
			return false;
		}
		else if (VARSIZE_ANY_EXHDR(addr) < offsetof(inet_struct, ipaddr))
		{
			STROM_ELOG(kcxt, "corrupted inet datum");
			return false;
		}
		else
		{
			inet_struct *ip_data = (inet_struct *)VARDATA_ANY(addr);
			int		ip_size = ip_addrsize(ip_data);

			memcpy(&result->value, VARDATA_ANY(addr),
				   offsetof(inet_struct, ipaddr[ip_size]));		
		}
	}
	else
	{
		int		byteWidth = cmeta->attopts.fixed_size_binary.byteWidth;

		if (byteWidth == 4)
		{
			result->value.family = PGSQL_AF_INET;
			result->value.bits = 32;
			memcpy(result->value.ipaddr, addr, byteWidth);
		}
		else if (byteWidth == 16)
		{
			result->value.family = PGSQL_AF_INET6;
			result->value.bits = 128;
			memcpy(result->value.ipaddr, addr, byteWidth);
		}
		else
		{
			STROM_ELOG(kcxt, "corrupted inet datum");
			return false;
		}
	}
	return true;
}

PUBLIC_FUNCTION(int)
xpu_inet_datum_store(kern_context *kcxt,
					 char *buffer,
					 xpu_datum_t *__arg)
{
	xpu_inet_t *arg = (xpu_inet_t *)__arg;
	int			len;

	if (arg->isnull)
		return 0;
	if (arg->value.family == PGSQL_AF_INET)
		len = offsetof(inet_struct, ipaddr) + 4;
	else if (arg->value.family == PGSQL_AF_INET6)
		len = offsetof(inet_struct, ipaddr) + 16;
	else
	{
		STROM_ELOG(kcxt, "corrupted inet datum");
		return -1;
	}
	if (buffer)
	{
		memcpy(buffer + VARHDRSZ, &arg->value, len);
		SET_VARSIZE(buffer, len + VARHDRSZ);
	}
	return len + VARHDRSZ;
}

PUBLIC_FUNCTION(bool)
xpu_inet_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					xpu_datum_t *__arg)
{
	xpu_inet_t *arg = (xpu_inet_t *)__arg;
	int			len;

	if (arg->isnull)
		*p_hash = 0;
	else
	{
		if (arg->value.family == PGSQL_AF_INET)
			len = offsetof(inet_struct, ipaddr[4]);		/* IPv4 */
		else if (arg->value.family == PGSQL_AF_INET6)
			len = offsetof(inet_struct, ipaddr[16]);	/* IPv6 */
		else
		{
			STROM_ELOG(kcxt, "corrupted inet datum");
			return false;
		}
		*p_hash = pg_hash_any(&arg->value, len);
	}
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(inet);
