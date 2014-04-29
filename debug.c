/*
 * debug.c
 *
 * Various debugging stuff of PG-Strom
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "pg_strom.h"

bool		pgstrom_kernel_debug;

/*
 * pgstrom_dump_kernel_debug
 *
 * It dumps all the debug message during kernel execution, if any.
 */
void
pgstrom_dump_kernel_debug(int elevel, kern_resultbuf *kresult)
{
	kern_debug *kdebug;
	char	   *baseptr;
	cl_uint		i, j, offset = 0;

	if (kresult->debug_usage == KERN_DEBUG_UNAVAILABLE ||
		kresult->debug_nums == 0)
		return;

	baseptr = (char *)&kresult->results[kresult->nrooms];
	for (i=0; i < kresult->debug_nums; i++)
	{
		char		buf[1024];

		kdebug = (kern_debug *)(baseptr + offset);
		j = snprintf(buf, sizeof(buf),
					 "Global(%u/%u+%u) Local (%u/%u) %s = ",
					 kdebug->global_id,
					 kdebug->global_sz,
					 kdebug->global_ofs,
					 kdebug->local_id,
					 kdebug->local_sz,
					 kdebug->label);
		switch (kdebug->v_class)
		{
			case 'c':
				snprintf(buf + j, sizeof(buf) - j, "%hhd",
						 (cl_char)(kdebug->value.v_int & 0x000000ff));
				break;
			case 's':
				snprintf(buf + j, sizeof(buf) - j, "%hd",
						 (cl_short)(kdebug->value.v_int & 0x0000ffff));
				break;
			case 'i':
				snprintf(buf + j, sizeof(buf) - j, "%d",
						 (cl_int)(kdebug->value.v_int & 0xffffffff));
				break;
			case 'l':
				snprintf(buf + j, sizeof(buf) - j, "%ld",
						 (cl_long)kdebug->value.v_int);
				break;
			case 'f':
			case 'd':
				snprintf(buf + j, sizeof(buf) - j, "%f",
						 (cl_double)kdebug->value.v_fp);
				break;
			default:
				snprintf(buf + j, sizeof(buf) - j,
						 "0x%016lx (unknown class)", kdebug->value.v_int);
				break;
		}
		elog(elevel, "kdebug: %s", buf);

		offset += kdebug->length;
	}
}

/*
 * Debugging facilities for shmem.c
 */
Datum
pgstrom_shmem_alloc_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	Size	size = PG_GETARG_INT64(0);
	void   *address;

	address = pgstrom_shmem_alloc(size);

	PG_RETURN_INT64((Size) address);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);

	PG_RETURN_NULL();
#endif
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_alloc_func);

Datum
pgstrom_shmem_free_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	void		   *address = (void *) PG_GETARG_INT64(0);

	pgstrom_shmem_free(address);

	PG_RETURN_BOOL(true);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);

	PG_RETURN_NULL();
#endif
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_free_func);

/*
 * initialization of debugging facilities
 */
void
pgstrom_init_debug(void)
{
	/* turn on/off kernel device debug support */
	DefineCustomBoolVariable("pg_strom.kernel_debug",
							 "turn on/off kernel debug support",
							 NULL,
							 &pgstrom_kernel_debug,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
}
