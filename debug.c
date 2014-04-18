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
 * Debugging facilities for mqueue.c
 */
#ifdef PGSTROM_DEBUG
typedef struct {
	pgstrom_message	msg;
	cl_uint			seconds;
	char			label[FLEXIBLE_ARRAY_MEMBER];
} pgstrom_test_message;

static void
pgstrom_process_testmsg(pgstrom_message *msg)
{
	pgstrom_test_message   *tmsg = (pgstrom_test_message *)msg;

	elog(LOG, "test message: '%s' in process (refcnt=%d)",
		 tmsg->label, tmsg->msg.refcnt);
	sleep(tmsg->seconds);
	elog(LOG, "test message: '%s' being replied (refcnt=%d)",
		 tmsg->label, tmsg->msg.refcnt);
	pgstrom_reply_message(&tmsg->msg);
}

static void
pgstrom_release_testmsg(pgstrom_message *msg)
{
	elog(INFO, "test message %p is released", msg);
	pgstrom_put_queue(msg->respq);
	pgstrom_shmem_free(msg);
}
#endif

/*
 * pgstrom_create_queue_func
 *
 * creates a message queue for test-message facility
 */
Datum
pgstrom_create_queue_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	pgstrom_queue  *mqueue = pgstrom_create_queue();
	if (!mqueue)
		elog(ERROR, "out of shared memory");
	PG_RETURN_INT64((Size)mqueue);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);
	PG_RETURN_NULL();
#endif
}
PG_FUNCTION_INFO_V1(pgstrom_create_queue_func);

/*
 * pgstrom_close_queue_func
 *
 * close the supplied message queue
 * arg1: pointer of pgstrom_queue object
 */
Datum
pgstrom_close_queue_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	pgstrom_queue  *mqueue = (pgstrom_queue *)PG_GETARG_POINTER(0);

	pgstrom_close_queue(mqueue);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);
#endif
	PG_RETURN_BOOL(true);
}
PG_FUNCTION_INFO_V1(pgstrom_close_queue_func);

/*
 * pgstrom_create_testmsg_func
 *
 * creates a new test-message object and returns its pointer.
 * arg1: pointer of pgstrom_queue object
 * arg2: seconds to be slept in the server
 * arg3: text label to be printed on the server
 */
Datum
pgstrom_create_testmsg_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	pgstrom_test_message   *tmsg;
	pgstrom_queue  *mqueue = (pgstrom_queue *)PG_GETARG_POINTER(0);
	int32			seconds = PG_GETARG_INT32(1);
	char		   *test_label = text_to_cstring(PG_GETARG_TEXT_P(2));
	Size			length = offsetof(pgstrom_test_message,
									  label[strlen(test_label) + 1]);

	tmsg = pgstrom_shmem_alloc(length);
	if (!tmsg)
		elog(ERROR, "out of shared memory");

	tmsg->msg.stag = StromTag_TestMessage;
	SpinLockInit(&tmsg->msg.lock);
	tmsg->msg.refcnt = 1;
	tmsg->msg.respq = pgstrom_get_queue(mqueue);
	tmsg->msg.cb_process = pgstrom_process_testmsg;
	tmsg->msg.cb_release = pgstrom_release_testmsg;
	tmsg->seconds = seconds;
	strcpy(tmsg->label, test_label);

	PG_RETURN_INT64((Size)tmsg);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);
#endif
	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(pgstrom_create_testmsg_func);

/*
 * pgstrom_enqueue_testmsg_func
 *
 * enqueues a message object into server message queue
 * arg1: pointer of pgstrom_message object
 */
Datum
pgstrom_enqueue_testmsg_func(PG_FUNCTION_ARGS)
{
	bool		result = false;
#ifdef PGSTROM_DEBUG
	pgstrom_test_message   *tmsg
		= (pgstrom_test_message *)PG_GETARG_POINTER(0);
	elog(INFO, "tmsg {refcnt=%d}", tmsg->msg.refcnt);
	result = pgstrom_enqueue_message(&tmsg->msg);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);
#endif
	PG_RETURN_BOOL(result);
}
PG_FUNCTION_INFO_V1(pgstrom_enqueue_testmsg_func);

/*
 * pgstrom_dequeue_testmsg_func
 *
 * dequeue a message object from a reply queue, and returns its pointer
 * arg1: pointer of pgstrom_queue object
 */
Datum
pgstrom_dequeue_testmsg_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	pgstrom_queue		   *mqueue = (pgstrom_queue *)PG_GETARG_POINTER(0);
	pgstrom_test_message   *tmsg
		= (pgstrom_test_message *)pgstrom_dequeue_message(mqueue);
	if (!tmsg)
		elog(INFO, "pgstrom_dequeue_message timeout!");
	else
		elog(INFO, "tmsg {refcnt=%d, label=%s}", tmsg->msg.refcnt, tmsg->label);
	PG_RETURN_INT64((Size)tmsg);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);
	PG_RETURN_NULL();
#endif
}
PG_FUNCTION_INFO_V1(pgstrom_dequeue_testmsg_func);

/*
 * pgstrom_release_testmsg_func
 *
 * it releases (actually, decrement of reference counter of) a message object
 * arg2: pointer of pgstrom_message object
 */
Datum
pgstrom_release_testmsg_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	pgstrom_test_message   *tmsg
		= (pgstrom_test_message *)PG_GETARG_POINTER(0);
	elog(INFO, "tmsg {refcnt=%d}", tmsg->msg.refcnt);
	pgstrom_put_message(&tmsg->msg);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);
#endif
	PG_RETURN_BOOL(true);
}
PG_FUNCTION_INFO_V1(pgstrom_release_testmsg_func);

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
