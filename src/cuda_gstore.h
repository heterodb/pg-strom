/*
 * cuda_gstore.h
 *
 * CUDA device code specific to GstoreFdw in-memory data store
 * --
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#ifndef CUDA_GSTORE_H
#define CUDA_GSTORE_H

#define GSTORE_TX_LOG__MAGIC		0xEBAD7C00
#define GSTORE_TX_LOG__INSERT		(GSTORE_TX_LOG__MAGIC | 'I')
#define GSTORE_TX_LOG__UPDATE		(GSTORE_TX_LOG__MAGIC | 'U')
#define GSTORE_TX_LOG__DELETE		(GSTORE_TX_LOG__MAGIC | 'D')
#define GSTORE_TX_LOG__COMMIT		(GSTORE_TX_LOG__MAGIC | 'C')

typedef struct {
	cl_uint		crc;
	cl_uint		type;
	cl_uint		length;
	cl_uint		xid;
	cl_ulong	timestamp;
	char		data[1];		/* variable length */
} GstoreTxLogCommon;

/*
 * INSERT/UPDATE/DELETE
 *
 * +----------------------+        -----
 * | GstoreTxLogCommon    |          ^
 * | - u32 crc32          |          |
 * | - u32 type           |          |
 * | - u32 length       o-------> length
 * | - u32 xid            |          |
 * | - u64 timestamp      |          |
 * +----------------------|          |
 * | - u32 rowid          |          |
 * | - u32 update_mask  o------+     |
 * | - HeapTupleHeaderData|    |
 * | +--------------------+    | ((char *)tx_log_row
 * | | PostgreSQL's       |    |        + tx_log_row->update_mask)
 * | | HeapTupleHeader    |    |
 * | |  + Nullmap + Data  |    |     |
 * = =                    =    |     |
 * +-+--------------------+ <--+     |
 * | Update Mask          |          |
 * | (bitmap of updated   |          |
 * |  columns)            |          v
 * +----------------------+        -----
 *
 * GstoreTxLogRow contains HeapTupleHeaderData that is usual PostgreSQL's
 * heap-tuple structure, and update-mask at tail of the transaction log.
 * It indicates which columns were updated on UPDATE to save the extra
 * buffer area if schema contains variable-length field.
 * INSERT/DELETE log shall not have the update-mask, and usually DELETE
 * log does not carray any values (t_infomask & HEAP_NATTS_MASK) == 0.
 */
typedef struct {
	cl_uint		crc;
	cl_uint		type;
	cl_uint		length;
	cl_uint		xid;
	cl_ulong	timestamp;
	/* above are common */
	cl_uint		rowid;
	cl_uint		update_mask;
	HeapTupleHeaderData htup;
} GstoreTxLogRow;

/*
 * COMMIT
 *
 * +---------------------+    -----
 * | GstoreTxLogCommon   |      ^
 * | - u32 crc32         |      |
 * | - u32 type          |      |
 * | - u32 length      o---> length
 * | - u32 xid           |      |
 * | - u64 timestamp     |      |
 * +---------------------|      |
 * | - u32 nitems        |      |
 * | - u32 rowids[]      |      |
 * | +-------------------+      |
 * | | rowid-1 committed |      |
 * | | rowid-2 committed |      |
 * | |        :          |      |
 * | | rowid-N committed |      v
 * +-+-------------------+    -----
 *
 * GstoreTxLogCommit is just a hint for CPU code, but informs GPU code
 * which rows are committed. The synchronizer kernel shall update the
 * xmin/xmax field of GPU device buffer according to the commit-log.
 * GPU code can see the rows only (1) Xmin=Frozen or own transactions
 * and (2) Xmax=Invalid or not own transactions.
 */
typedef struct {
	cl_uint		crc;
	cl_uint		type;
	cl_uint		length;
	cl_uint		xid;
	cl_ulong	timestamp;
	/* above fields are common */
	cl_uint		nitems;
	cl_uint		rowids[1];		/* variable length */
} GstoreTxLogCommit;


/*
 * GstoreFdwSysattr
 *
 * A fixed-length system attribute for each row.
 */
typedef struct
{
	cl_uint		xmin;
	cl_uint		xmax;
#ifndef __CUDACC__
	cl_uint		cid;
#else
	/* get_global_id() of the thread who tries to update the row. */
	cl_uint		owner_id;
#endif
} GstoreFdwSysattr;

/*
 * kern_gpustore_redolog
 */
typedef struct
{
	kern_errorbuf	kerror;
	size_t			length;
	cl_uint			nrooms;
	cl_uint			nitems;
	cl_uint			log_index[FLEXIBLE_ARRAY_MEMBER];
} kern_gpustore_redolog;

#endif /* CUDA_GSTORE_H */
