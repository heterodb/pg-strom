/*
 * inners.c
 *
 * Portion of GpuJoin to build inner hash/heap table (WIP).
 * ----
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
#include "pg_strom.h"

struct shared_mmap_segment
{
	dlist_node		chain;
	ResourceOwner	resowner;
	uint32			handle;
	bool			needs_cleanup;
	void		   *mapped_address;
	size_t			mapped_length;
};

static dlist_head	shared_mmap_segment_list;





/*
 * NOTE: PG12 removed dsm_resize() because it didn't handle failure of
 * posix_fallocate() correctly and not supported on all platforms.
 * So, we have *_expand() API instead, because we only try to expand
 * shared memory segment, and it nevet hits problematic scenario.
 */
#define shared_mmap_filename(namebuf, handle)	\
	snprintf((namebuf), sizeof(namebuf), "/pg_strom.%08x", (handle))

shared_mmap_segment *
shared_mmap_create(size_t required)
{
	size_t		size = TYPEALIGN(PAGE_SIZE, required);
	char		name[64];
	uint32		handle;
	int			rc, fdesc = -1;
	void	   *address;
	shared_mmap_segment *shm_seg;

	shm_seg = MemoryContextAllocZero(TopMemoryContext,
									 sizeof(shared_mmap_segment));
	PG_TRY();
	{
		do {
			handle = ((uint32)MyProcPid ^ (uint32)random());
			shared_mmap_filename(name, handle);

			fdesc = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
			if (fdesc < 0 && errno != EEXIST)
				elog(ERROR, "could not open shared memory segment \"%s\": %m",
					 name);
		} while (fdesc < 0);

		do {
			rc = posix_fallocate(fdesc, 0, size);
			if (rc < 0 && errno != EINTR)
				elog(ERROR, "failed on posix_fallocate on \"%s\": %m", name);
		} while (rc != 0);

		address = mmap(NULL, size, PROT_READ | PROT_WRITE,
					   MAP_SHARED, fdesc, 0);
		if (address == MAP_FAILED)
			elog(ERROR, "failed on mmap on \"%s\": %m", name);

		close(fdesc);
	}
	PG_CATCH();
	{
		if (fdesc >= 0)
		{
			shm_unlink(name);
			close(fdesc);
		}
		pfree(shm_seg);
		PG_RE_THROW();
	}
	PG_END_TRY();

	shm_seg->resowner = CurrentResourceOwner;
	shm_seg->handle = handle;
	shm_seg->needs_cleanup = true;
	shm_seg->mapped_address = address;
	shm_seg->mapped_length = size;
	dlist_push_tail(&shared_mmap_segment_list, &shm_seg->chain);

	return shm_seg;
}

shared_mmap_segment *
shared_mmap_attach(uint32 handle)
{
	char		name[64];
	int			fdesc = -1;
	struct stat	st_buf;
	void	   *address;
	shared_mmap_segment *shm_seg;

	shm_seg = MemoryContextAllocZero(TopMemoryContext,
									 sizeof(shared_mmap_segment));
	PG_TRY();
	{
		shared_mmap_filename(name, handle);
		fdesc = shm_open(name, O_RDWR, 0600);
		if (fdesc < 0)
			elog(ERROR, "could not open shared memory segment \"%s\": %m",
				 name);

		if (fstat(fdesc, &st_buf) != 0)
			elog(ERROR, "failed on fstat(\"%s\"): %m", name);

		address = mmap(NULL, st_buf.st_size, PROT_READ | PROT_WRITE,
					   MAP_SHARED, fdesc, 0);
		if (address == MAP_FAILED)
			elog(ERROR, "failed on mmap on \"%s\": %m", name);

		close(fdesc);
	}
	PG_CATCH();
	{
		if (fdesc >= 0)
			close(fdesc);
		pfree(shm_seg);
		PG_RE_THROW();
	}
	PG_END_TRY();

	shm_seg->resowner = CurrentResourceOwner;
	shm_seg->handle = handle;
	shm_seg->needs_cleanup = false;
	shm_seg->mapped_address = address;
	shm_seg->mapped_length = st_buf.st_size;
	dlist_push_tail(&shared_mmap_segment_list, &shm_seg->chain);

	return shm_seg;
}

void
shared_mmap_detach(shared_mmap_segment *shm_seg)
{
	char		name[64];

	shared_mmap_filename(name, shm_seg->handle);
	if (munmap(shm_seg->mapped_address,
			   shm_seg->mapped_length) != 0)
	{
		elog(WARNING, "failed on munmap(\"%s\", %zu): %m",
			 name, shm_seg->mapped_length);
	}
	if (shm_seg->needs_cleanup)
	{
		if (shm_unlink(name) != 0)
			elog(WARNING, "failed on shm_link(\"%s\"): %m", name);
	}
	dlist_delete(&shm_seg->chain);
	pfree(shm_seg);
}

void *
shared_mmap_expand(shared_mmap_segment *shm_seg, size_t new_size)
{
	char		name[64];
	int			fdesc;
	void	   *address;
	struct stat	st_buf;

	new_size = TYPEALIGN(PAGE_SIZE, new_size);
	if (new_size <= shm_seg->mapped_length)
		goto skip;	/* nothing to do */

	shared_mmap_filename(name, shm_seg->handle);
	fdesc = shm_open(name, O_RDWR, 0600);
	if (fdesc < 0)
		elog(ERROR, "could not open shared memory segment \"%s\": %m",
			 name);
	PG_TRY();
	{
		if (fstat(fdesc, &st_buf) != 0)
			elog(ERROR, "failed on fstat(\"%s\"): %m", name);
		if (new_size > st_buf.st_size)
		{
			if (posix_fallocate(fdesc, 0, new_size) != 0)
				elog(ERROR, "failed on posix_fallocate(\"%s\"): %m", name);
		}
		address = mremap(shm_seg->mapped_address,
						 shm_seg->mapped_length,
						 new_size, MREMAP_MAYMOVE);
		if (address == MAP_FAILED)
			elog(ERROR, "failed on mremap(\"%s\", %zu): %m", name, new_size);
		shm_seg->mapped_address = address;
		shm_seg->mapped_length = new_size;
	}
	PG_CATCH();
	{
		close(fdesc);
		PG_RE_THROW();
	}
	PG_END_TRY();
skip:
	return shm_seg->mapped_address;
}

void *
shared_mmap_address(shared_mmap_segment *shm_seg)
{
	return shm_seg->mapped_address;
}

size_t
shared_mmap_length(struct shared_mmap_segment *shm_seg)
{
	return shm_seg->mapped_length;
}

uint64
shared_mmap_handle(struct shared_mmap_segment *shm_seg)
{
	return shm_seg->handle;
}

static void
shared_mmap_callback(ResourceReleasePhase phase,
					 bool is_commit, bool is_toplevel, void *arg)
{
	dlist_mutable_iter iter;

	if (phase != RESOURCE_RELEASE_BEFORE_LOCKS)
		return;

	dlist_foreach_modify(iter, &shared_mmap_segment_list)
	{
		shared_mmap_segment *shm_seg = (shared_mmap_segment *)
			dlist_container(shared_mmap_segment, chain, iter.cur);

		if (shm_seg->resowner == CurrentResourceOwner)
			shared_mmap_detach(shm_seg);
	}
}

/*
 * pgstrom_init_inners
 */
void
pgstrom_init_inners(void)
{
	dlist_init(&shared_mmap_segment_list);

	RegisterResourceReleaseCallback(shared_mmap_callback, NULL);
}
