/*
 * pg_strom.h
 *
 * Header file of pg_strom module
 *
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef PG_STROM_H
#define PG_STROM_H

/*
 * opencl_entry.c
 */
extern void pgstrom_init_opencl_entry(void);

/*
 * main.c
 */
extern void _PG_init(void);

#endif	/* PG_STROM_H */
