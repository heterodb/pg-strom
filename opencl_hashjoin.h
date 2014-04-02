/*
 * opencl_hashjoin.h
 *
 * Parallel hash join accelerated by OpenCL device
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef OPENCL_HASHJOIN_H
#define OPENCL_HASHJOIN_H

typedef struct {
	MessageTag		mtag;
	cl_uint			next;	/* offset of next hash-item, or 0 if not exists */
	cl_uint			hash;	/* 32-bit hash value */
	cl_char			keydata[FLEXIBLE_ARRAY_MEMBER];
} kern_hash_item;

typedef struct {
	MessageTag		mtag;
	cl_uint			nslots;
	cl_uint			nkeys;
	kern_colmeta	keyatts[FLEXIBLE_ARRAY_MEMBER];
} kern_hash_table;


#ifdef OPENCL_DEVICE_CODE
#endif
#endif	/* OPENCL_HASHJOIN_H */
