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
					int vclass,
					const kern_variable *kvar)
{
	xpu_money_t *result = (xpu_money_t *)__result;

	result->expr_ops = &xpu_money_ops;
	if (vclass == KVAR_CLASS__INLINE)
		result->value = kvar->i64;
	else if (vclass >= sizeof(Cash))
		result->value = *((Cash *)kvar->ptr);
	else
	{
		STROM_ELOG(kcxt, "unexpected vclass for device money data type.");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_money_datum_store(kern_context *kcxt,
					  const xpu_datum_t *__arg,
					  int *p_vclass,
					  kern_variable *p_kvar)
{
	const xpu_money_t *arg = (const xpu_money_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_vclass = KVAR_CLASS__NULL;
	else
	{
		*p_vclass = KVAR_CLASS__INLINE;
		p_kvar->i64 = arg->value;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_money_datum_write(kern_context *kcxt,
					  char *buffer,
					  const xpu_datum_t *__arg)
{
	const xpu_money_t *arg = (const xpu_money_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
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

	if (XPU_DATUM_ISNULL(arg))
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
				   int vclass,
				   const kern_variable *kvar)
{
	xpu_uuid_t *result = (xpu_uuid_t *)__result;

	result->expr_ops = &xpu_uuid_ops;
	if (vclass >= sizeof(pg_uuid_t))
	{
		memcpy(&result->value, kvar->ptr, sizeof(pg_uuid_t));
	}
	else
	{
		STROM_ELOG(kcxt, "unexpected vclass for device uuid data type.");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_uuid_datum_store(kern_context *kcxt,
					 const xpu_datum_t *__arg,
					 int *p_vclass,
					 kern_variable *p_kvar)
{
	xpu_uuid_t	   *arg = (xpu_uuid_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_vclass = KVAR_CLASS__NULL;
	else
	{
		xpu_uuid_t *buf = (xpu_uuid_t *)kcxt_alloc(kcxt, sizeof(pg_uuid_t));

		if (!buf)
			return false;
		memcpy(buf, &arg->value, sizeof(pg_uuid_t));
		p_kvar->ptr = buf;
		*p_vclass = sizeof(pg_uuid_t);
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_uuid_datum_write(kern_context *kcxt,
					 char *buffer,
					 const xpu_datum_t *__arg)
{
	const xpu_uuid_t *arg = (const xpu_uuid_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		return 0;
	if (buffer)
		memcpy(buffer, &arg->value, sizeof(pg_uuid_t));
	return sizeof(pg_uuid_t);
}

STATIC_FUNCTION(bool)
xpu_uuid_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					const xpu_datum_t *__arg)
{
	const xpu_uuid_t *arg = (const xpu_uuid_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
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
					  int vclass,
					  const kern_variable *kvar)
{
	xpu_macaddr_t *result = (xpu_macaddr_t *)__result;

	result->expr_ops = &xpu_macaddr_ops;
	if (vclass >= sizeof(macaddr))
		memcpy(&result->value, kvar->ptr, sizeof(macaddr));
	else
	{
		STROM_ELOG(kcxt, "unexpected vclass for device macaddr data type.");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_macaddr_datum_store(kern_context *kcxt,
						const xpu_datum_t *__arg,
						int *p_vclass,
						kern_variable *p_kvar)
{
	xpu_macaddr_t  *arg = (xpu_macaddr_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_vclass = KVAR_CLASS__NULL;
	else
	{
		macaddr	   *buf = (macaddr *)kcxt_alloc(kcxt, sizeof(macaddr));

		if (!buf)
			return false;
		memcpy(buf, &arg->value, sizeof(macaddr));
		p_kvar->ptr = buf;
		*p_vclass = sizeof(macaddr);
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_macaddr_datum_write(kern_context *kcxt,
						char *buffer,
						const xpu_datum_t *__arg)
{
	const xpu_macaddr_t  *arg = (xpu_macaddr_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
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

	if (XPU_DATUM_ISNULL(arg))
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
				   int vclass,
				   const kern_variable *kvar)
{
	xpu_inet_t *result = (xpu_inet_t *)__result;

	if (vclass == KVAR_CLASS__VARLENA)
	{
		const char *addr = (const char *)kvar->ptr;

		if (VARATT_IS_EXTERNAL(addr) || VARATT_IS_COMPRESSED(addr))
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
			}
			else if (sz == offsetof(inet_struct, ipaddr[16]))
			{
				memcpy(&result->value, VARDATA_ANY(addr), sz);
				if (result->value.family != PGSQL_AF_INET6)
				{
					STROM_ELOG(kcxt, "inet (ipv6) value corruption");
					return false;
				}
			}
			else
			{
				STROM_ELOG(kcxt, "Bug? inet value is corrupted");
				return false;
			}
			result->expr_ops = &xpu_inet_ops;
		}
	}
	else
	{
		STROM_ELOG(kcxt, "unexpected vclass for device inet data type.");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_inet_datum_store(kern_context *kcxt,
					 const xpu_datum_t *__arg,
					 int *p_vclass,
					 kern_variable *p_kvar)
{
	xpu_inet_t *arg = (xpu_inet_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_vclass = KVAR_CLASS__NULL;
	else
	{
		inet   *in;
		int		sz;

		if (arg->value.family == PGSQL_AF_INET)
			sz = offsetof(inet_struct, ipaddr) + 4;
		else if (arg->value.family == PGSQL_AF_INET6)
			sz = offsetof(inet_struct, ipaddr) + 16;
		else
		{
			STROM_ELOG(kcxt, "Bug? inet value is corrupted");
			return false;
		}
		in = (inet *)kcxt_alloc(kcxt, offsetof(inet, inet_data) + sz);
		if (!in)
			return false;
		memcpy(&in->inet_data, &arg->value, sz);
		SET_VARSIZE(in, VARHDRSZ + sz);

		p_kvar->ptr = in;
		*p_vclass = KVAR_CLASS__VARLENA;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_inet_datum_write(kern_context *kcxt,
					 char *buffer,
					 const xpu_datum_t *__arg)
{
	const xpu_inet_t  *arg = (xpu_inet_t *)__arg;
	int		sz;

	if (XPU_DATUM_ISNULL(arg))
		return 0;
	if (arg->value.family == PGSQL_AF_INET)
		sz = offsetof(inet_struct, ipaddr) + 4;
	else if (arg->value.family == PGSQL_AF_INET6)
		sz = offsetof(inet_struct, ipaddr) + 16;
	else
	{
		STROM_ELOG(kcxt, "Bug? inet value is corrupted");
		return -1;
	}
	if (buffer)
	{
		memcpy(buffer+VARHDRSZ, &arg->value, sz);
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

	if (XPU_DATUM_ISNULL(arg))
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
