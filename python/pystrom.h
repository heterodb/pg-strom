#include <stdint.h>

#ifndef PYSTROM_H
#define PYSTROM_H
/* boolean */
#ifndef bool
typedef char bool;
#endif
#ifndef true
#define true	((bool) 1)
#endif
#ifndef false
#define false	((bool) 0)
#endif

/* simple list operations */
typedef struct dlist_node dlist_node;
struct dlist_node
{
	dlist_node	   *prev;
	dlist_node	   *next;
};

typedef struct dlist_head
{
	dlist_node		head;
} dlist_head;

static inline void
dlist_init(dlist_head *head)
{
	head->head.next = head->head.prev = &head->head;
}

static inline void
dlist_push_head(dlist_head *head, dlist_node *node)
{
	node->next = head->head.next;
	node->prev = &head->head;
	node->next->prev = node;
	head->head.next = node;
}

static inline void
dlist_push_tail(dlist_head *head, dlist_node *node)
{
    node->next = &head->head;
    node->prev = head->head.prev;
    node->prev->next = node;
    head->head.prev = node;
}

static inline void
dlist_delete(dlist_node *node)
{
	node->prev->next = node->next;
	node->next->prev = node->prev;
}

/*
 * simple hash function
 */
static inline uint32_t
hash_value(const unsigned char *ptr, size_t len)
{
	uint64_t	value = 0;
	uint64_t	extra = 0;

	while (len >= sizeof(uint64_t))
	{
		value ^= *((uint64_t *)ptr);
		ptr += sizeof(uint64_t);
		len -= sizeof(uint64_t);
	}
	switch (len)
	{
		case 7:
			extra |= ((uint64_t)ptr[6]) << 48;
		case 6:
			extra |= ((uint64_t)ptr[5]) << 40;
		case 5:
			extra |= ((uint64_t)ptr[4]) << 32;
		case 4:
			extra |= ((uint64_t)ptr[3]) << 24;
		case 3:
			extra |= ((uint64_t)ptr[2]) << 16;
		case 2:
			extra |= ((uint64_t)ptr[1]) <<  8;
		case 1:
			extra |= ((uint64_t)ptr[0]);
		default:
			break;
	}
	value ^= extra;

	return ((value * 0x61C8864680B583EBULL) >> 16) & 0xffffffffU;
}

/*
 * encode / decode hex cstring
 */
static inline int
hex_decode(const char *src, unsigned int len, char *dst)
{
	const char *pos = src;
	const char *end = src + len;
	char	   *start = dst;
	int			c, h, l;

	while (pos < end)
	{
		c = *pos++;
		if (c >= '0' && c <= '9')
			h = c - '0';
		else if (c >= 'a' && c <= 'f')
			h = c - 'a' + 10;
		else if (c >= 'A' && c <= 'F')
			h = c - 'A' + 10;
		else
		{
			PyErr_Format(PyExc_ValueError, "invalid hexadecimal data");
			return -1;
		}

		c = *pos++;
		if (c >= '0' && c <= '9')
			l = c - '0';
		else if (c >= 'a' && c <= 'f')
			l = c - 'a' + 10;
		else if (c >= 'A' && c <= 'F')
			l = c - 'A' + 10;
		else
		{
			PyErr_Format(PyExc_ValueError, "invalid hexadecimal data");
			return -1;
		}
		*dst++ = (h << 4) | l;
	}
	return (dst - start);
}

/*
 * function declarations
 */
extern uintptr_t cupy_strom__ipcmem_open(const char *ident,
										 int *p_device_id,
										 size_t *p_bytesize,
										 char *p_type_code,
										 int *p_width,
										 long *p_height);
extern int	cupy_strom__ipcmem_close(uintptr_t device_ptr);

#endif	/* PYSTROM_H */
