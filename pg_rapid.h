/*
 * pg_rapid.h
 *
 * Header file of pg_rapid module
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#ifndef PG_RAPID_H
#define PG_RAPID_H

#include "fmgr.h"

/*
 * utilcmds.c
 */
extern void	pgrapid_utilcmds_init(void);

/*
 * blkload.c
 */


/*
 * pg_rapid.c
 */
extern Datum pgrapid_fdw_handler(PG_FUNCTION_ARGS);
extern Datum pgrapid_fdw_validator(PG_FUNCTION_ARGS);
extern void _PG_init(void);

#endif	/* PG_RAPID_H */
