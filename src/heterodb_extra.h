/*
 * heterodb_extra.h
 *
 * Definitions of HeteroDB Extra Package
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2017-2021 (C) HeteroDB,Inc
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef HETERODB_EXTRA_H
#define HETERODB_EXTRA_H
#include <stdbool.h>
#include <stdint.h>

#define HETERODB_EXTRA_FILENAME		"heterodb_extra.so"
#define HETERODB_EXTRA_PATHNAME		"/usr/lib64/" HETERODB_EXTRA_FILENAME
#define HETERODB_EXTRA_MAX_GPUS		63
#ifndef HAS_GPUMASK_TYPEDEF
#define HAS_GPUMASK_TYPEDEF
#define INVALID_GPUMASK				(~0UL)
typedef int64_t						gpumask_t;
#endif	/* HAS_GPUMASK_TYPEDEF */

#define HETERODB_LICENSE_PATHNAME	"/etc/heterodb.license"
/* fixed length of the license key (2048bits) */
#define HETERODB_LICENSE_KEYLEN		256
#define HETERODB_LICENSE_KEYBITS	(8 * HETERODB_LICENSE_KEYLEN)

#define HETERODB_EXTRA_CURRENT_API_VERSION	20240725
#define HETERODB_EXTRA_OLDEST_API_VERSION	20240418

/* cufile.c */
typedef struct
{
	unsigned long	m_offset;	/* destination offset from the base address
								 * base = mgmem + offset */
	unsigned int	fchunk_id;	/* source page index of the file. */
	unsigned int	nr_pages;	/* number of pages to be loaded */
} strom_io_chunk;

typedef struct
{
	unsigned int	nr_chunks;
	strom_io_chunk	ioc[1];
} strom_io_vector;

#endif	/* HETERODB_EXTRA_H */
