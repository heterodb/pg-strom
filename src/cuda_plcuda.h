/*
 * cuda_plcuda.h
 *
 * CUDA device code for PL/CUDA
 * --
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#ifndef CUDA_PLCUDA_H
#define CUDA_PLCUDA_H

/*
 * GstoreIpcMapping
 */
typedef struct
{
	GstoreIpcHandle	h;		/* unique identifier of Gstore_Fdw */
	void		   *map;	/* mapped device address
							 * in the CUDA program context */
} GstoreIpcMapping;

#define PLCUDA_ARGMENT_FDESC	4
#define PLCUDA_RESULT_FDESC		5

/*
 * Error handling
 */
#define EEXIT(fmt,...)									\
	do {												\
		fprintf(stderr, "Error(L%d): " fmt "\n",		\
				__LINE__, ##__VA_ARGS__);				\
		exit(1);										\
	} while(0)

#define CUEXIT(rc,fmt,...)								\
	do {												\
		fprintf(stderr, "Error(L%d): " fmt " (%s)\n",	\
				__LINE__, ##__VA_ARGS__,				\
				cudaGetErrorName(rc));					\
	} while(0)

#endif	/* CUDA_PLCUDA_H */
