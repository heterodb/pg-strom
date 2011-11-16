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

/*
 * XXX - Some of naming scheme conflicts between PostgreSQL and
 * CUDA, so we make declarations of PostgreSQL side invisible in
 * the case when npp.h was included.
 */
#ifdef POSTGRES_H
#include "fmgr.h"
#endif
#include <driver_types.h>

#define PGRAPID_SCHEMA_NAME		"pg_rapid"

/*
 * utilcmds.c
 */
#ifdef POSTGRES_H
extern void	pgrapid_utilcmds_init(void);

#endif
/*
 * blkload.c
 */
#ifdef POSTGRES_H


#endif

/*
 * pg_rapid.c
 */
#ifdef POSTGRES_H
extern bool pgrapid_fdw_handler_is_called;
extern Datum pgrapid_fdw_handler(PG_FUNCTION_ARGS);
extern Datum pgrapid_fdw_validator(PG_FUNCTION_ARGS);
#endif

/*
 * cuda.c
 */
extern const char *
pgcuda_get_error_string(cudaError_t error);
extern cudaError_t
pgcuda_get_device_count(int *count);
extern cudaError_t
pgcuda_get_device_properties(struct cudaDeviceProp *prop, int device);

#endif	/* PG_RAPID_H */
