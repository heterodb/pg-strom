/*
 * xpu_misclib.h
 *
 * Misc definitions for both of GPU and DPU
 * --
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_MISCLIB_H
#define XPU_MISCLIB_H
#include <sys/socket.h>

#ifndef CASH_H
typedef int64_t		Cash;
#endif

#ifndef UUID_H
/* sql_uuid_t */
#define UUID_LEN	16
typedef struct
{
	uint8_t		data[UUID_LEN];
} pg_uuid_t;
#endif	/* UUID_H */

#ifndef INET_H
/* sql_macaddr_t */
typedef struct
{
	uint8_t		a;
	uint8_t		b;
	uint8_t		c;
	uint8_t		d;
	uint8_t		e;
	uint8_t		f;
} macaddr;

/* sql_inet_t */
typedef struct
{
	uint8_t		family;		/* PGSQL_AF_INET or PGSQL_AF_INET6 */
	uint8_t		bits;		/* number of bits in netmask */
	uint8_t		ipaddr[16];	/* up to 128 bits of address */
} inet_struct;

#define PGSQL_AF_INET		(AF_INET + 0)
#define PGSQL_AF_INET6		(AF_INET + 1)

typedef struct
{
	uint32_t	vl_len_; /* Do not touch this field directly! */
	inet_struct	inet_data;
} inet;

#define ip_family(inetptr)		(inetptr)->family
#define ip_bits(inetptr)		(inetptr)->bits
#define ip_addr(inetptr)		(inetptr)->ipaddr
#define ip_addrsize(inetptr)	\
	((inetptr)->family == PGSQL_AF_INET ? 4 : 16)
#define ip_maxbits(inetptr)		\
	((inetptr)->family == PGSQL_AF_INET ? 32 : 128)
#endif	/* INET_H */

PGSTROM_SQLTYPE_SIMPLE_DECLARATION(money, int64_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(uuid, pg_uuid_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(macaddr, macaddr);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(inet, inet_struct);

EXTERN_FUNCTION(int)
xpu_interval_write_heap(kern_context *kcxt,
						char *buffer,
						const xpu_datum_t *arg);

#endif	/* XPU_MISCLIB_H */
