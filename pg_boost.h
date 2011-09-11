/*
 * pg_boost.h - Header file of pg_boost module
 *
 * Copyright (c) 2011 KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#ifndef PG_BOOST_H
#define PG_BOOST_H
#include <pthread.h>

/*
 * pg_boost.c - entrypoint of this module
 */
extern int		guc_segment_size;
extern bool		guc_with_hugetlb;
extern char	   *guc_unbuffered_dir;

/*
 * type/macro definition of dual linked list on shared memory segment
 */
typedef struct
{
	uintptr_t	prev;
	uintptr_t	next;
} mlist_t;

#define offset_of(type, member)					\
	__builtin_offsetof (type, member)
#define container_of(addr, type, member)		\
	(type *)(((uintptr_t)(addr)) - offset_of(type, member))
#define list_entry(offset, type, member)		\
	container_of(offset_to_addr(offset), type, member)

/*
 * shmmgr.c - management of shared memory segment / buffers
 */
typedef struct msegment_s		msegment_t;
typedef struct mchunk_s			mchunk_t;
typedef struct mbuffer_s		mbuffer_t;
typedef struct mslab_head_s		mslab_head_t;
typedef struct mslab_body_s		mslab_body_t;

extern msegment_t   *msegment;
#define addr_to_offset(p)		\
	((p) == NULL ? 0 : (((uintptr_t)(p)) - (uintptr_t)(msegment)))
#define offset_to_addr(p)		\
	((p) == 0 ? NULL : (void *)((p) + (uintptr_t)(msegment)))

extern mchunk_t	   *shmmgr_alloc_chunk(size_t size);
extern mchunk_t	   *shmmgr_try_alloc_chunk(size_t size);
extern void			shmmgr_free_chunk(mchunk_t *mchunk);
extern void		   *shmmgr_get_chunk_data(mchunk_t *mchunk);
extern int			shmmgr_get_chunk_class(mchunk_t *mchunk);

extern mbuffer_t   *shmmgr_alloc_buffer(size_t size);
extern mbuffer_t   *shmmgr_try_alloc_buffer(size_t size);
extern void			shmmgr_free_buffer(mbuffer_t *mbuffer);
extern void		   *shmmgr_get_read_buffer(mbuffer_t *mbuffer);
extern void		   *shmmgr_get_write_buffer(mbuffer_t *mbuffer);
extern void			shmmgr_put_buffer(mbuffer_t *mbuffer, bool is_dirty);

extern mslab_head_t *shmmgr_create_slab(const char *slabname, int unit_size);
extern void			shmmgr_destroy_slab(mslab_head_t *mslab);
extern void		   *shmmgr_alloc_slab(mslab_head_t *mslab);
extern void		   *shmmgr_try_alloc_slab(mslab_head_t *mslab);
extern void			shmmgr_free_slab(mslab_head_t *mslab, void *ptr);

extern void			shmmgr_init_mutex(pthread_mutex_t *lock);
extern void			shmmgr_init_rwlock(pthread_rwlock_t *lock);
extern void			shmmgr_init(void);

/*
 * shmbtree.c - translation index ItemPointer -> uintptr_t
 */



/*
 * Inline routines and iterator of dual linked list on shared memory segment
 */
static inline void mlist_init(mlist_t *mlist)
{
	mlist->next = mlist->prev = addr_to_offset(mlist);
}

static inline bool mlist_empty(mlist_t *mlist)
{
	return mlist->next == addr_to_offset(mlist) ? true : false;
}

static inline void mlist_add(mlist_t *base, mlist_t *list)
{
	mlist_t  *next = offset_to_addr(base->next);

	base->next = addr_to_offset(list);
	list->prev = addr_to_offset(base);
	list->next = addr_to_offset(next);
	next->prev = addr_to_offset(list);
}

static inline void mlist_del(mlist_t *mlist)
{
	mlist_t	   *plist = offset_to_addr(mlist->prev);
	mlist_t	   *nlist = offset_to_addr(mlist->next);

	plist->next = mlist->next;
	nlist->prev = mlist->prev;

	mlist_init(mlist);
}

/*
 * mlist_foreach_entry(_safe)
 *
 * Iterator of mlist. "_safe" version is safe against remove items.
 * @entry  : pointer of type * owning the list as a loop cursor.
 * @temp   : another type * to use as temporary storage
 * @head   : head of the list to be iterated
 * @member : name of the shmlist_t within the struct
 */
#define mlist_foreach_entry(entry, head, member)						\
	for (entry = list_entry((head)->next, typeof(*entry), member);		\
		 &entry->member != (head);										\
		 entry = list_entry(entry->member.next, typeof(*entry), member))

#define mlist_foreach_entry_safe(entry, temp, head, member)				\
	for (entry = list_entry((head)->next, typeof(*entry), member),		\
		 temp = list_entry(entry->member.next, typeof(*entry), member); \
		 &entry->member != (head);										\
		 entry = temp,													\
		 temp = list_entry(entry->member.next, typeof(*entry), member))

#endif	/* PG_BOOST_H */
