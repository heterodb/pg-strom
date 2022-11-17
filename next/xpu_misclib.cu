/*
 * xpu_misclib.cu
 *
 * Collection of misc functions and operators for both of GPU and DPU
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
STATIC_FUNCTION(bool)
xpu_money_datum_ref(kern_context *kcxt,
					xpu_datum_t *__result,
					const void *addr)
{
	xpu_money_t *result = (xpu_money_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		result->value  = *((Cash *)addr);
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_money_arrow_move(kern_context *kcxt,
					 char *buffer,
					 const kern_colmeta *cmeta,
					 const void *addr, int len)
{
	if (!addr)
		return 0;
	if (len != sizeof(Cash))
	{
		STROM_ELOG(kcxt, "Arrow value is not convertible to money");
		return -1;
	}
	if (buffer)
		memcpy(buffer, addr, sizeof(Cash));
	return sizeof(Cash);
}

STATIC_FUNCTION(bool)
xpu_money_arrow_ref(kern_context *kcxt,
					xpu_datum_t *__result,
					const kern_colmeta *cmeta,
					const void *addr, int len)
{
	xpu_money_t *result = (xpu_money_t *)__result;
	int		sz;

	sz = xpu_money_arrow_move(kcxt, (char *)&result->value,
							  cmeta, addr, len);
	if (sz < 0)
		return false;
	result->isnull = (sz == 0);
	return true;
}

STATIC_FUNCTION(int)
xpu_money_datum_store(kern_context *kcxt,
					  char *buffer,
					  const xpu_datum_t *__arg)
{
	const xpu_money_t *arg = (const xpu_money_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		*((Cash *)buffer) = arg->value;
	return sizeof(Cash);
}

STATIC_FUNCTION(bool)
xpu_money_datum_hash(kern_context *kcxt,
					 uint32_t *p_hash,
					 const xpu_datum_t *__arg)
{
	const xpu_money_t *arg = (const xpu_money_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(Cash));
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(money, true, 8, sizeof(Cash));
PG_SIMPLE_COMPARE_TEMPLATE(cash_,money,money,Cash)
/*
 * UUID data type (xpu_uuid_t), functions and operators
 */
STATIC_FUNCTION(bool)
xpu_uuid_datum_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const void *addr)
{
	xpu_uuid_t *result = (xpu_uuid_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		memcpy(result->value.data, addr, UUID_LEN);
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_uuid_arrow_move(kern_context *kcxt,
					char *buffer,
					const kern_colmeta *cmeta,
					const void *addr, int len)
{
	if (len != UUID_LEN)
	{
		STROM_ELOG(kcxt, "Arrow value is not convertible to uuid");
		return -1;
	}
	if (!addr)
		return 0;
	if (buffer)
		memcpy(buffer, addr, UUID_LEN);
	return UUID_LEN;
}

STATIC_FUNCTION(bool)
xpu_uuid_arrow_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const kern_colmeta *cmeta,
				   const void *addr, int len)
{
	xpu_uuid_t *result = (xpu_uuid_t *)__result;
	int		sz;

	sz = xpu_uuid_arrow_move(kcxt, (char *)result->value.data,
							 cmeta, addr, len);
	if (sz < 0)
		return false;
	result->isnull = (sz == 0);
	return true;
}

STATIC_FUNCTION(int)
xpu_uuid_datum_store(kern_context *kcxt,
					 char *buffer,
					 const xpu_datum_t *__arg)
{
	const xpu_uuid_t *arg = (const xpu_uuid_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		memcpy(buffer, arg->value.data, UUID_LEN);
	return UUID_LEN;
}

STATIC_FUNCTION(bool)
xpu_uuid_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					const xpu_datum_t *__arg)
{
	const xpu_uuid_t *arg = (const xpu_uuid_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(arg->value.data, UUID_LEN);
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(uuid, false, 1, UUID_LEN);

/*
 * Macaddr data type (xpu_macaddr_t), functions and operators
 */
STATIC_FUNCTION(bool)
xpu_macaddr_datum_ref(kern_context *kcxt,
					  xpu_datum_t *__result,
					  const void *addr)
{
	xpu_macaddr_t *result = (xpu_macaddr_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		memcpy(&result->value, addr, sizeof(macaddr));
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_macaddr_arrow_move(kern_context *kcxt,
					   char *buffer,
					   const kern_colmeta *cmeta,
					   const void *addr, int len)
{
	if (!addr)
		return 0;
	if (len != sizeof(macaddr))
	{
		STROM_ELOG(kcxt, "Arrow value is not convertible to macaddr");
		return -1;
	}
	if (buffer)
		memcpy(buffer, addr, sizeof(macaddr));
	return sizeof(macaddr);
}

STATIC_FUNCTION(bool)
xpu_macaddr_arrow_ref(kern_context *kcxt,
					  xpu_datum_t *__result,
					  const kern_colmeta *cmeta,
					  const void *addr, int len)
{
	xpu_macaddr_t *result = (xpu_macaddr_t *)__result;
	int		sz;

	sz = xpu_macaddr_arrow_move(kcxt, (char *)&result->value,
								cmeta, addr, len);
	if (sz < 0)
		return false;
	result->isnull = (sz == 0);
	return true;
}

STATIC_FUNCTION(int)
xpu_macaddr_datum_store(kern_context *kcxt,
						char *buffer,
						const xpu_datum_t *__arg)
{
	const xpu_macaddr_t *arg = (const xpu_macaddr_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		memcpy(buffer, &arg->value, sizeof(macaddr));
	return sizeof(macaddr);
}

STATIC_FUNCTION(bool)
xpu_macaddr_datum_hash(kern_context *kcxt,
					   uint32_t *p_hash,
					   const xpu_datum_t *__arg)
{
	const xpu_macaddr_t *arg = (const xpu_macaddr_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(macaddr));
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(macaddr, false, 4, sizeof(macaddr));

/*
 * Inet data type (xpu_iner_t), functions and operators
 */
STATIC_FUNCTION(bool)
xpu_inet_datum_ref(kern_context *kcxt,
                   xpu_datum_t *__result,
				   const void *addr)
{
	xpu_inet_t *result = (xpu_inet_t *)__result;

	if (!addr)
		result->isnull = true;
	else if (VARATT_IS_EXTERNAL(addr) || VARATT_IS_COMPRESSED(addr))
	{
		STROM_CPU_FALLBACK(kcxt, "inet value is compressed or toasted");
		return false;
	}
	else
	{
		int		sz = VARSIZE_ANY_EXHDR(addr);

		if (sz == offsetof(inet_struct, ipaddr[4]))
		{
			memcpy(&result->value, VARDATA_ANY(addr), sz);
			if (result->value.family != PGSQL_AF_INET)
			{
				STROM_ELOG(kcxt, "inet (ipv4) value corruption");
				return false;
			}
			result->isnull = false;
		}
		else if (sz == offsetof(inet_struct, ipaddr[16]))
		{
			memcpy(&result->value, VARDATA_ANY(addr), sz);
			if (result->value.family != PGSQL_AF_INET6)
			{
				STROM_ELOG(kcxt, "inet (ipv6) value corruption");
				return false;
			}
			result->isnull = false;
		}
		else
		{
			STROM_ELOG(kcxt, "Bug? inet value is corrupted");
			return false;
		}
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_inet_arrow_move(kern_context *kcxt,
					char *buffer,
					const kern_colmeta *cmeta,
					const void *addr, int len)
{
	uint8_t	family;
	int		sz;

	if (!addr)
		return 0;
	if (len == 4)
		family = PGSQL_AF_INET;
	else if (len == 16)
		family = PGSQL_AF_INET6;
	else
	{
		STROM_ELOG(kcxt, "Arrow value is not convertible to inet");
		return -1;
	}

	sz = VARHDRSZ + offsetof(inet_struct, ipaddr[len]);
	if (buffer)
	{
		inet   *idata = (inet *)buffer;
		
		idata->inet_data.family = family;
		idata->inet_data.bits   = 8 * len;
		memcpy(idata->inet_data.ipaddr, addr, len);
		SET_VARSIZE(&idata, sz);
		memcpy(buffer, &idata, sz);
	}
	return sz;
}

STATIC_FUNCTION(bool)
xpu_inet_arrow_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const kern_colmeta *cmeta,
				   const void *addr, int len)
{
	xpu_inet_t *result = (xpu_inet_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		if (len == 4)
			result->value.family = PGSQL_AF_INET;
		else if (len == 16)
			result->value.family = PGSQL_AF_INET6;
		else
		{
			STROM_ELOG(kcxt, "Arrow value is not convertible to inet");
			return false;
		}
		result->value.bits = (len << 3);
		memcpy(result->value.ipaddr, addr, len);
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_inet_datum_store(kern_context *kcxt,
					 char *buffer,
					 const xpu_datum_t *__arg)
{
	const xpu_inet_t *arg = (const xpu_inet_t *)__arg;
	int		sz;

	if (arg->isnull)
		return 0;
	sz = (arg->value.family == PGSQL_AF_INET
		  ? offsetof(inet_struct, ipaddr[4])
		  : offsetof(inet_struct, ipaddr[16]));
	if (buffer)
	{
		memcpy(buffer + VARHDRSZ, &arg->value, sz);
		SET_VARSIZE(buffer, VARHDRSZ + sz);
	}
	return VARHDRSZ + sz;
}

STATIC_FUNCTION(bool)
xpu_inet_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					const xpu_datum_t *__arg)
{
	const xpu_inet_t *arg = (const xpu_inet_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
	{
		int		sz;

		if (arg->value.family == PGSQL_AF_INET)
			sz = offsetof(inet_struct, ipaddr[4]);		/* IPv4 */
		else if (arg->value.family == PGSQL_AF_INET6)
			sz = offsetof(inet_struct, ipaddr[16]);	/* IPv6 */
		else
		{
			STROM_ELOG(kcxt, "corrupted inet datum");
			return false;
		}
		*p_hash = pg_hash_any(&arg->value, sz);
	}
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(inet, false, 4, -1);
