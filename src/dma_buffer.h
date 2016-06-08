/*
 * dma_buffer.h
 *
 * Definition of the portable shared address support and relevant
 * dual-linked list support.
 * --
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
#ifndef DMA_BUFFER_H
#define DMA_BUFFER_H

/* definition of the portable shared address */
typedef cl_ulong		port_addr_t;

extern void *paddr_to_local(port_addr_t paddr);
extern port_addr_t local_to_paddr(void *l_ptr);

extern void *dmaBufferAlloc(GpuContext_v2 *gcontext);
extern void dmaBufferFree(void *l_ptr);

/*
 * Dual linked list operations
 */
typedef struct plist_node
{
	port_addr_t		prev;
	port_addr_t		next;
} plist_node;

typedef struct plist_head
{
	plist_node		head;
} plist_head;

typedef struct plist_iter
{
	plist_node	   *cur;	/* current element */
	plist_node	   *end;	/* last node we'll iterate to */
} plist_iter;

static inline void
plist_init(plist_head *head)
{
	head->head.next = head->head.prev = local_to_paddr(&head->head);
}

static inline bool
plist_is_empty(plist_head *head)
{
	return (head->head.next == 0UL ||
			paddr_to_local(head->head.next) == &head->head);
}

static inline void
plist_push_head(plist_head *head, plist_node *node)
{
	plist_node	   *next;

	if (head->head.next == 0UL || head->head.prev == 0UL)
		plist_init(head);
	next = paddr_to_local(head->head.next);

	node->next = head->head.next;
	node->prev = local_to_paddr(&head->head);
	head->head.next = next->prev = local_to_paddr(node);
}

static inline void
plist_push_tail(plist_head *head, plist_node *node)
{
	plist_node	   *prev;

	if (head->head.next == 0UL || head->head.prev == 0UL)
		plist_init(head);
	prev = paddr_to_local(head->head.prev);

	node->next = local_to_paddr(&head->head);
	node->prev = head->head.prev;
	head->head.prev = prev->next = local_to_paddr(node);
}

static inline void
plist_delete(plist_node *node)
{
	plist_node	   *next = paddr_to_local(node->next);
	plist_node	   *prev = paddr_to_local(node->prev);

	prev->next = node->next;
	next->prev = node->prev;

	memset(node, 0, sizeof(plist_node));
}

static inline plist_node *
plist_pop_head_node(plist_head *head)
{
	plist_node	   *node;
	Assert(!plist_is_empty(head));
	node = paddr_to_local(head->head.next);
	dlist_delete(node);
	return node;
}

static inline bool
plist_has_next(plist_head *head, plist_node *node)
{
	return node->next != local_to_paddr(&head->head);
}

static inline bool
plist_has_prev(plist_head *head, plist_node *node)
{
	return node->prev != local_to_addr(&head->head);
}

static inline plist_node *
plist_next_node(plist_head *head, plist_node *node)
{
	Assert(plist_has_next(head, node));
	return paddr_to_local(node->next);
}

static inline plist_node *
plist_prev_node(plist_head *head, plist_node *node)
{
	Assert(plist_has_prev(head, node));
	return paddr_to_local(node->prev);
}

static inline plist_node *
plist_head_node(plist_head *head)
{
	Assert(!plist_is_empty(head));
	return paddr_to_local(head->head.next);
}

static inline plist_node *
plist_tail_node(plist_head *head)
{
	Assert(!plist_is_empty(head));
	return paddr_to_local(head->head.prev);
}

#define plist_foreach(iter, lhead)								\
	for (AssertVariableIsOfTypeMacro(iter, plist_iter),			\
		 AssertVariableIsOfTypeMacro(lhead, plist_head *),		\
		 (iter).end = &(lhead)->head,							\
		 (iter).cur = ((iter).end->next							\
					   ? paddr_to_local((iter).end->next)		\
					   : (iter).end),							\
		 (iter).cur != (iter).end;								\
		 (iter).cur = paddr_to_local((iter).next))

#endif	/* DMA_BUFFER_H */
