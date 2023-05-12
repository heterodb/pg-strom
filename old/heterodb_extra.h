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

#define HETERODB_EXTRA_API_VERSION	20211018

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

typedef struct GPUDirectFileDesc
{
	int			rawfd;
	void	   *fhandle;
	size_t		bytesize;
	/* CUfileHandle_t is an alias of 'void *' defined at cufile.h */
} GPUDirectFileDesc;

/* sysfs.c */
typedef struct
{
	int			device_id;
	char		device_name[128];
	const char *cpu_affinity;	/* __internal use__ */
	int			pci_domain;		/* PCI_DOMAIN_ID */
	int			pci_bus_id;		/* PCI_BUS_ID */
	int			pci_dev_id;		/* PCI_DEVICE_ID */
	int			pci_func_id;	/* MULTI_GPU_BOARD ? MULTI_GPU_BOARD_GROUP_ID : 0 */
} GpuPciDevItem;

/* misc.c */
typedef struct
{
	const char *filename;
	int			lineno;
	const char *funcname;
	char		message[2000];
} heterodb_extra_error_info;

#endif	/* HETERODB_EXTRA_H */
