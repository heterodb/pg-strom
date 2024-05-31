/*
 * pg_utils.h
 *
 * Inline routines of misc utility purposes
 * --
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef PG_UTILS_H
#define PG_UTILS_H

/* Max/Min macros that takes 3 or more arguments */
#define Max3(a,b,c)		((a) > (b) ? Max((a),(c)) : Max((b),(c)))
#define Max4(a,b,c,d)	Max(Max((a),(b)), Max((c),(d)))

#define Min3(a,b,c)		((a) > (b) ? Min((a),(c)) : Min((b),(c)))
#define Min4(a,b,c,d)	Min(Min((a),(b)), Min((c),(d)))


/*
 * transformation from align character into width
 */
static inline int
typealign_get_width(char type_align)
{
	switch (type_align)
	{
		case 'c':
			return 1;
		case 's':
			return ALIGNOF_SHORT;
		case 'i':
			return ALIGNOF_INT;
		case 'd':
			return ALIGNOF_DOUBLE;
		default:
			elog(ERROR, "unexpected type alignment: %c", type_align);
	}
	return -1;  /* be compiler quiet */
}

/*
 * get_next_log2
 *
 * It returns N of the least 2^N value that is larger than or equal to
 * the supplied value.
 */
static inline int
get_next_log2(Size size)
{
	int		shift = 0;

	if (size == 0 || size == 1)
		return 0;
	size--;
#ifdef __GNUC__
	shift = sizeof(Size) * BITS_PER_BYTE - __builtin_clzl(size);
#else
#if SIZEOF_VOID_P == 8
	if ((size & 0xffffffff00000000UL) != 0)
	{
		size >>= 32;
		shift += 32;
	}
#endif
	if ((size & 0xffff0000UL) != 0)
	{
		size >>= 16;
		shift += 16;
	}
	if ((size & 0x0000ff00UL) != 0)
	{
		size >>= 8;
		shift += 8;
	}
	if ((size & 0x000000f0UL) != 0)
	{
		size >>= 4;
		shift += 4;
	}
	if ((size & 0x0000000cUL) != 0)
	{
		size >>= 2;
		shift += 2;
	}
	if ((size & 0x00000002UL) != 0)
	{
		size >>= 1;
		shift += 1;
	}
	if ((size & 0x00000001UL) != 0)
		shift += 1;
#endif  /* !__GNUC__ */
	return shift;
}

/*
 * __trim - remove whitespace at the head/tail of cstring
 */
static inline char *
__trim(char *token)
{
	char   *tail = token + strlen(token) - 1;

	while (isspace(*token))
		token++;
	while (tail >= token && isspace(*tail))
		*tail-- = '\0';
	return token;
}

/*
 * __strtol / __strtoul / __strtosz - set errno if token is not pure digits
 */
static inline long int
__strtol(const char *token)
{
	long int ival;
	char   *end;

	errno = 0;		/* clear */
	ival = strtol(token, &end, 10);
	if (*end != '\0')
		errno = EINVAL;
	return ival;
}

static inline unsigned long int
__strtoul(const char *token)
{
	unsigned long int ival;
	char   *end;

	errno = 0;		/* clear */
	ival = strtoul(token, &end, 10);
	if (*end != '\0')
		errno = EINVAL;
	return ival;
}

static inline size_t
__strtosz(const char *token)
{
	size_t	sz;
	char   *end;

	errno = 0;		/* clear */
	sz = strtoul(token, &end, 10);
	if (errno == 0)
	{
		if (strcasecmp(end, "t") == 0 || strcasecmp(end, "tb") == 0)
		{
			if (sz > 0x0000000000ffffffUL)
				errno = ERANGE;
			else
				sz <<= 40;
		}
		else if (strcasecmp(end, "g") == 0 || strcasecmp(end, "gb") == 0)
		{
			if (sz > 0x00000003ffffffffUL)
				errno = ERANGE;
			else
				sz <<= 30;
		}
		else if (strcasecmp(end, "m") == 0 || strcasecmp(end, "mb") == 0)
		{
			if (sz > 0x00000fffffffffffUL)
				errno = ERANGE;
			else
				sz <<= 20;
		}
		else if (strcasecmp(end, "k") == 0 || strcasecmp(end, "kb") == 0)
		{
			if (sz > 0x003fffffffffffffUL)
				errno = ERANGE;
			else
				sz <<= 10;
		}
		else if (*end != '\0')
			errno = EINVAL;
	}
	return sz;
}

/* lappend on the specified memory-context */
static inline List *
lappend_cxt(MemoryContext memcxt, List *list, void *datum)
{
	MemoryContext oldcxt = MemoryContextSwitchTo(memcxt);
	List   *r;

	r = lappend(list, datum);
	MemoryContextSwitchTo(oldcxt);

	return r;
}

/* initStringInfo on the specified memory-context */
static inline void
initStringInfoCxt(MemoryContext memcxt, StringInfo buf)
{
	MemoryContext	oldcxt;

	oldcxt = MemoryContextSwitchTo(memcxt);
	initStringInfo(buf);
	MemoryContextSwitchTo(oldcxt);
}

/*
 * formater of numeric/bytesz/millisec
 */
static inline char *
format_numeric(int64 value)
{
	if (value > 8000000000000L   || value < -8000000000000L)
		return psprintf("%.2fT", (double)value / 1000000000000.0);
	else if (value > 8000000000L || value < -8000000000L)
		return psprintf("%.2fG", (double)value / 1000000000.0);
	else if (value > 8000000L    || value < -8000000L)
		return psprintf("%.2fM", (double)value / 1000000.0);
	else if (value > 8000L       || value < -8000L)
		return psprintf("%.2fK", (double)value / 1000.0);
	else
		return psprintf("%ld", value);
}

static inline char *
format_bytesz(size_t nbytes)
{
	if (nbytes > (1UL<<43))
		return psprintf("%.2fTB", (double)nbytes / (double)(1UL<<40));
	else if (nbytes > (1UL<<33))
		return psprintf("%.2fGB", (double)nbytes / (double)(1UL<<30));
	else if (nbytes > (1UL<<23))
		return psprintf("%.2fMB", (double)nbytes / (double)(1UL<<20));
	else if (nbytes > (1UL<<13))
		return psprintf("%.2fKB", (double)nbytes / (double)(1UL<<10));
	return psprintf("%uB", (unsigned int)nbytes);
}

static inline char *
format_millisec(double milliseconds)
{
	if (milliseconds > 300000.0)    /* more then 5min */
		return psprintf("%.2fmin", milliseconds / 60000.0);
	else if (milliseconds > 8000.0) /* more than 8sec */
		return psprintf("%.2fsec", milliseconds / 1000.0);
	return psprintf("%.2fms", milliseconds);
}

/*
 * pmemdup
 */
static inline void *
pmemdup(const void *src, Size sz)
{
	void   *dst = palloc(sz);

	memcpy(dst, src, sz);

	return dst;
}

/*
 * buildoidvector
 */
static inline oidvector *
__buildoidvector1(Oid a)
{
	return buildoidvector(&a, 1);
}

static inline oidvector *
__buildoidvector2(Oid a, Oid b)
{
	Oid		oids[2];

	oids[0] = a;
	oids[1] = b;
	return buildoidvector(oids, 2);
}

static inline oidvector *
__buildoidvector3(Oid a, Oid b, Oid c)
{
	Oid		oids[3];

	oids[0] = a;
	oids[1] = b;
	oids[2] = c;
	return buildoidvector(oids, 3);
}

static inline oidvector *
__buildoidvector4(Oid a, Oid b, Oid c, Oid d)
{
	Oid		oids[4];

	oids[0] = a;
	oids[1] = b;
	oids[2] = c;
	oids[3] = d;
	return buildoidvector(oids, 4);
}

/*
 * Macros for worker threads
 */
#define __FATAL(fmt,...)                        \
	do {                                        \
		fprintf(stderr, "(%s:%d) " fmt "\n",    \
				basename(__FILE__), __LINE__,   \
				##__VA_ARGS__);                 \
		_exit(1);                               \
	} while(0)

static inline void
pthreadMutexInit(pthread_mutex_t *mutex)
{
	if ((errno = pthread_mutex_init(mutex, NULL)) != 0)
		__FATAL("failed on pthread_mutex_init: %m");
}

static inline void
pthreadMutexInitShared(pthread_mutex_t *mutex)
{
	pthread_mutexattr_t mattr;

	if ((errno = pthread_mutexattr_init(&mattr)) != 0)
		__FATAL("failed on pthread_mutexattr_init: %m");
	if ((errno = pthread_mutexattr_setpshared(&mattr, 1)) != 0)
		__FATAL("failed on pthread_mutexattr_setpshared: %m");
	if ((errno = pthread_mutex_init(mutex, &mattr)) != 0)
        __FATAL("failed on pthread_mutex_init: %m");
    if ((errno = pthread_mutexattr_destroy(&mattr)) != 0)
        __FATAL("failed on pthread_mutexattr_destroy: %m");
}

static inline void
pthreadMutexLock(pthread_mutex_t *mutex)
{
	if ((errno = pthread_mutex_lock(mutex)) != 0)
		__FATAL("failed on pthread_mutex_lock: %m");
}

static inline bool
pthreadMutexTryLock(pthread_mutex_t *mutex)
{
	if ((errno = pthread_mutex_trylock(mutex)) == 0)
		return true;
	if (errno != EBUSY)
		__FATAL("failed on pthread_mutex_trylock: %m");
	return false;
}

static inline void
pthreadMutexUnlock(pthread_mutex_t *mutex)
{
	if ((errno = pthread_mutex_unlock(mutex)) != 0)
		__FATAL("failed on pthread_mutex_unlock: %m");
}

static inline void
pthreadRWLockInit(pthread_rwlock_t *rwlock)
{
	if ((errno = pthread_rwlock_init(rwlock, NULL)) != 0)
		__FATAL("failed on pthread_rwlock_init: %m");
}

static inline void
pthreadRWLockInitShared(pthread_rwlock_t *rwlock)
{
	pthread_rwlockattr_t rwattr;

	if ((errno = pthread_rwlockattr_init(&rwattr)) != 0)
		__FATAL("failed on pthread_rwlockattr_init: %m");
    if ((errno = pthread_rwlockattr_setpshared(&rwattr, 1)) != 0)
		__FATAL("failed on pthread_rwlockattr_setpshared: %m");
    if ((errno = pthread_rwlock_init(rwlock, &rwattr)) != 0)
		__FATAL("failed on pthread_rwlock_init: %m");
}

static inline void
pthreadRWLockReadLock(pthread_rwlock_t *rwlock)
{
	if ((errno = pthread_rwlock_rdlock(rwlock)) != 0)
        __FATAL("failed on pthread_rwlock_rdlock: %m");
}

static inline void
pthreadRWLockWriteLock(pthread_rwlock_t *rwlock)
{
	if ((errno = pthread_rwlock_wrlock(rwlock)) != 0)
		__FATAL("failed on pthread_rwlock_wrlock: %m");
}

static inline void
pthreadRWLockUnlock(pthread_rwlock_t *rwlock)
{
	if ((errno = pthread_rwlock_unlock(rwlock)) != 0)
		__FATAL("failed on pthread_rwlock_unlock: %m");
}

static inline void
pthreadCondInit(pthread_cond_t *cond)
{
	if ((errno = pthread_cond_init(cond, NULL)) != 0)
		__FATAL("failed on pthread_cond_init: %m");
}

static inline void
pthreadCondInitShared(pthread_cond_t *cond)
{
	pthread_condattr_t condattr;

	if ((errno = pthread_condattr_init(&condattr)) != 0)
		__FATAL("failed on pthread_condattr_init: %m");
	if ((errno = pthread_condattr_setpshared(&condattr, 1)) != 0)
		__FATAL("failed on pthread_condattr_setpshared: %m");
	if ((errno = pthread_cond_init(cond, &condattr)) != 0)
		__FATAL("failed on pthread_cond_init: %m");
	if ((errno = pthread_condattr_destroy(&condattr)) != 0)
		__FATAL("failed on pthread_condattr_destroy: %m");
}

static inline void
pthreadCondWait(pthread_cond_t *cond, pthread_mutex_t *mutex)
{
    if ((errno = pthread_cond_wait(cond, mutex)) != 0)
		__FATAL("failed on pthread_cond_wait: %m");
}

static inline bool
pthreadCondWaitTimeout(pthread_cond_t *cond, pthread_mutex_t *mutex,
					   long timeout_ms)
{
	struct timespec tm;

	clock_gettime(CLOCK_REALTIME, &tm);
	tm.tv_sec  += (timeout_ms / 1000);
	tm.tv_nsec += (timeout_ms % 1000) * 1000000;
	if (tm.tv_nsec >= 1000000000L)
	{
		tm.tv_sec += tm.tv_nsec / 1000000000;
		tm.tv_nsec = tm.tv_nsec % 1000000000;
	}
	errno = pthread_cond_timedwait(cond, mutex, &tm);
	if (errno == 0)
		return true;
	else if (errno == ETIMEDOUT)
		return false;
	__FATAL("failed on pthread_cond_timedwait: %m");
}

static inline void
pthreadCondBroadcast(pthread_cond_t *cond)
{
	if ((errno = pthread_cond_broadcast(cond)) != 0)
		__FATAL("failed on pthread_cond_broadcast: %m");
}

static inline void
pthreadCondSignal(pthread_cond_t *cond)
{
	if ((errno = pthread_cond_signal(cond)) != 0)
		__FATAL("failed on pthread_cond_signal: %m");
}

/*
 * Misc debug functions
 */
INLINE_FUNCTION(void)
dump_tuple_desc(const TupleDesc tdesc)
{
	fprintf(stderr, "tupdesc %p { natts=%d, tdtypeid=%u, tdtypmod=%d, tdrefcount=%d }\n",
			tdesc,
			tdesc->natts,
			tdesc->tdtypeid,
			tdesc->tdtypmod,
			tdesc->tdrefcount);
	for (int j=0; j < tdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tdesc, j);

		fprintf(stderr, "attr[%d] { attname='%s', atttypid=%u, attlen=%d, attnum=%d, atttypmod=%d, attbyval=%c, attalign=%c, attnotnull=%c attisdropped=%c }\n",
				j,
				NameStr(attr->attname),
				attr->atttypid,
				(int)attr->attlen,
				(int)attr->attnum,
				(int)attr->atttypmod,
				attr->attbyval ? 't' : 'f',
				attr->attalign,
				attr->attnotnull ? 't' : 'f',
				attr->attisdropped ? 't' : 'f');
	}
}

/*
 * dump_kern_data_store
 */
INLINE_FUNCTION(void)
dump_kern_data_store(const kern_data_store *kds)
{
	fprintf(stderr, "kds %p { length=%lu, usage=%lu, nitems=%u, ncols=%u, format=%c, has_varlena=%c, tdhasoid=%c, tdtypeid=%u, tdtypmod=%d, table_oid=%u, hash_nslots=%u, block_offset=%u, block_nloaded=%u, nr_colmeta=%u }\n",
			kds,
			kds->length,
			kds->usage,
			kds->nitems,
			kds->ncols,
			kds->format,
			kds->has_varlena ? 't' : 'f',
			kds->tdhasoid ? 't' : 'f',
			kds->tdtypeid,
			kds->tdtypmod,
			kds->table_oid,
			kds->hash_nslots,
			kds->block_offset,
			kds->block_nloaded,
			kds->nr_colmeta);
	for (int j=0; j < kds->nr_colmeta; j++)
	{
		const kern_colmeta *cmeta = &kds->colmeta[j];

		fprintf(stderr, "cmeta[%d] { attbyval=%c, attalign=%d, attlen=%d, attnum=%d, attcacheoff=%d, atttypid=%u, atttypmod=%d, atttypkind=%c, kds_format=%c, kds_offset=%u, idx_subattrs=%u, num_subattrs=%u, attname='%s' }\n",
				j,
				cmeta->attbyval ? 't' : 'f',
				(int)cmeta->attalign,
				(int)cmeta->attlen,
				(int)cmeta->attnum,
				(int)cmeta->attcacheoff,
				cmeta->atttypid,
				cmeta->atttypmod,
				cmeta->atttypkind,
				cmeta->kds_format,
				cmeta->kds_offset,
				(unsigned int)cmeta->idx_subattrs,
				(unsigned int)cmeta->num_subattrs,
				cmeta->attname);
	}
}

#endif	/* PG_UTILS_H */
