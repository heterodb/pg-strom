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

#define HETERODB_EXTRA_FILENAME		"heterodb_extra.so"
#define HETERODB_EXTRA_PATHNAME		"/usr/lib64/" HETERODB_EXTRA_FILENAME

#define HETERODB_EXTRA_API_VERSION	20221225

/* gpudirect.c */
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

/* misc.c */
typedef struct
{
	const char *filename;
	int			lineno;
	const char *funcname;
	char		message[2000];
} heterodb_extra_error_info;

#endif	/* HETERODB_EXTRA_H */
