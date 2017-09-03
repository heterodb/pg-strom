/*
 * dsm_list.h
 *
 * inline functions to handle dual-linked list on DSM segment
 * ----
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
#ifndef DSM_LIST_H
#define DSM_LIST_H

typedef uint32		dsm_offset_t;

typedef struct
{
	dsm_offset_t	prev;
	dsm_offset_t	next;
} dsm_list_node;

typedef struct
{
	dsm_offset_t	prev;
	dsm_offset_t	next;
} dsm_list_head;

static inline void
dsm_list_init(dsm_list_head *dhead)
{
	dhead->prev = UINT_MAX;
	dhead->next = UINT_MAX;
}

static inline bool
dsm_list_is_empty(dsm_list_head *dhead)
{
	return dhead->prev == UINT_MAX && dhead->next == UINT_MAX;
}

static inline void
check_dsm_list_node(dsm_segment *dsm_seg, dsm_list_node *dnode)
{
	uintptr_t	base __attribute__((unused))
		= (uintptr_t) dsm_segment_address(dsm_seg);

	Assert((uintptr_t)(dnode) >= base &&
		   (uintptr_t)(dnode) <  base + dsm_segment_map_length(dsm_seg));
}

static inline void
dsm_list_push_head(dsm_segment *dsm_seg,
				   dsm_list_head *dhead, dsm_list_node *dnode)
{
	char		   *base = dsm_segment_address(dsm_seg);
	dsm_list_node  *dnext;
	dsm_offset_t	offset;

	check_dsm_list_node(dsm_seg, dnode);
	offset = (dsm_offset_t)((uintptr_t)dnode - (uintptr_t)base);
	if (dsm_list_is_empty(dhead))
	{
		dnode->prev = dnode->next = UINT_MAX;
		dhead->prev = dhead->next = offset;
	}
	else
	{
		dnext = (dsm_list_node *)(base + dhead->next);
		check_dsm_list_node(dsm_seg, dnext);

		dnode->next = dhead->next;
		dnode->prev = dnext->prev;
		dhead->next = offset;
		dhead->prev = offset;
	}
}

static inline void
dsm_list_push_tail(dsm_segment *dsm_seg,
				   dsm_list_head *dhead, dsm_list_node *dnode)
{
	char		   *base = dsm_segment_address(dsm_seg);
	dsm_list_node  *dprev;
	dsm_offset_t	offset;

	check_dsm_list_node(dsm_seg, dnode);
	offset = (dsm_offset_t)((uintptr_t)dnode - (uintptr_t)base);
	if (dsm_list_is_empty(dhead))
	{
		dnode->prev = dnode->next = UINT_MAX;
		dhead->prev = dhead->next = offset;
	}
	else
	{
		dprev = (dsm_list_node *)(base + dhead->prev);
		check_dsm_list_node(dsm_seg, dprev);

		dnode->next = dprev->next;
		dnode->prev = dhead->prev;
		dhead->prev = offset;
		dprev->next = offset;
	}
}

static inline void
dsm_list_delete(dsm_segment *dsm_seg,
				dsm_list_head *dhead, dsm_list_node *dnode)
{
	char		   *base= dsm_segment_address(dsm_seg);
	dsm_list_node  *dprev = NULL;
	dsm_list_node  *dnext = NULL;

	check_dsm_list_node(dsm_seg, dnode);
	if (dnode->prev == UINT_MAX)
		dhead->next = dnode->next;
	else
	{
		dprev = (dsm_list_node *)(base + dnode->prev);
		check_dsm_list_node(dsm_seg, dprev);
		Assert(dprev->next == ((uintptr_t)dnode - (uintptr_t)base));
		dprev->next = dnode->next;
	}

	if (dnode->next == UINT_MAX)
		dhead->prev = dnode->prev;
	else
	{
		dnext = (dsm_list_node *)(base + dnode->next);
		check_dsm_list_node(dsm_seg, dnext);
		Assert(dnext->prev == ((uintptr_t)dnode - (uintptr_t)base));
		dnext->prev = dnode->prev;
	}
	/* set poison */
	dnode->prev = dnode->next = UINT_MAX;
}

#define dsm_list_next(dsm_seg,offset)					\
	((offset) == UINT_MAX ? NULL : (dsm_list_node *)	\
	 ((char *)dsm_segment_address(dsm_seg) + (offset)))

#define dsm_list_foreach(pos, dsm_seg, dhead)			\
	for (pos = dsm_list_next(dsm_seg, (dhead)->next);	\
		 pos != NULL;									\
		 pos = dsm_list_next(dsm_seg, (pos)->next))

#define dsm_list_container(type, field, dnode)						  \
	(AssertVariableIsOfTypeMacro(dnode, dsm_list_node *),			  \
	 AssertVariableIsOfTypeMacro(((type *) NULL)->field, dms_list_node), \
	 ((type *)((char *)(dnode) - offsetof(type, field))))

#endif	/* DSM_LIST_H */





