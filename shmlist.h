/*
 * shmlist.h
 *
 * Routines to handle dual-linked list on shared memory segment.
 *
 * Copyright (C) 2011 KaiGai Kohei <kaigai@kaigai.gr.jp>
 */
#ifndef SHMLIST_H

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
 * mlist_init - initialize shmlist_t structure
 */
static inline void mlist_init(mlist_t *mlist)
{
	mlist->next = mlist->prev = addr_to_offset(mlist);
}

/*
 * mlist_empty - check whether the list is empty, or not
 */
static inline bool shmlist_empty(mlist *mlist)
{
	return mlist->next == addr_to_offset(mlist) ? true : false;
}

/*
 * mlist_add - add a new entry to the base list
 */
static inline void shmlist_add(mlist_t *base, mlist_t *list)
{
	shmlist_t  *next = offset_to_addr(base->next);

	base->next = addr_to_offset(list);
	list->prev = addr_to_offset(base);
	list->next = addr_to_offset(next);
	next->prev = addr_to_offset(list);
}

/*
 * mlist_del - remove an entry from the current list
 */
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

#endif	/* SHMLIST_H */
