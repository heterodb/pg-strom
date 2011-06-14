/*
 * shmlist.h - Dual linked list on shared memory segment
 *
 * Copyright (c) 2011 KaiGai Kohei <kaigai@kaigai.gr.jp>
 */
#ifndef SHMLIST_H
#define SHMLIST_H

/*
 * Dual linked list on shared memory segment
 */
typedef struct {
	shmptr_t		prev;
	shmptr_t		next;
} shmlist_t;

#define offsetof(type, member)	__builtin_offsetof (type, member)
#define container_of(addr, type, member)				\
	(type *)(((char *)addr) - offset_of(type, member))
#define list_entry(shmptr, type, member)				\
	container_of(shmptr_to_addr(shmptr), type, member)

/*
 * shmlist_foreach_entry(_safe)
 *
 * iterator of the shmlist. _safe version is safe against remove items.
 * @entry  : pointer of type * owning the list as a loop cursor.
 * @temp   : another type * to use as temporary storage
 * @head   : head of the list to be iterated
 * @member : name of the shmlist_t within the struct
 */
#define shmlist_foreach_entry(entry, head, member)						\
	for (entry = list_entry((head)->next, typeof(*entry), member);		\
		 &entry->member != (head);										\
		 entry = list_entry(entry->member.next, typeof(*entry), member))

#define shmlist_foreach_entry_safe(entry, temp, head, member)			\
	for (entry = list_entry((head)->next, typeof(*entry), member),		\
		 temp = list_entry(entry->member.next, typeof(*entry), member); \
		 &entry->member != (head);										\
		 entry = temp,													\
		 temp = list_entry(entry->member.next, typeof(*entry), member))

static inline bool
shmlist_empty(shmlist_t *list)
{
	return shmptr_to_addr(list->next) == list ? true : false;
}

static inline void
shmlist_init(shmlist_t *list)
{
	list->next = list->prev = shmptr_from_addr(list);
}

static inline void
shmlist_add(shmlist_t *base, shmlist_t *list)
{
	shmlist_t  *nlist = shmptr_to_addr(base->next);

	base->next = shmptr_from_addr(list);
	list->prev = shmptr_from_addr(base);
	list->next = shmptr_from_addr(nlist);
	nlist->prev = shmptr_from_addr(list);
}

static inline void
shmlist_del(shmlist_t *list)
{
	shmlist_t  *plist = shmptr_to_addr(list->prev);
	shmlist_t  *nlist = shmptr_to_addr(list->next);

	plist->next = shmptr_from_addr(nlist);
	nlist->prev = shmptr_from_addr(plist);

	shmlist_init(list);
}

#endif	/* SHMLIST_H */

