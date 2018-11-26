#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char * const argv[])
{
	const char *arg_catalog = NULL;
#if PLCUDA_NUM_ARGS == 0
	void	  **arg_ptrs = NULL;
#else
	void	   *arg_ptrs[PLCUDA_NUM_ARGS];
	char		arg_kind[PLCUDA_NUM_ARGS];
	char	   *arg_buffer = NULL;
	long		arg_bufsz = 128 * 1024;	/* 128kB in default */
	char	   *tail;
	const char *pos, *cat;
#endif
	ssize_t		sz, nbytes;
	char	   *buffer;
	int			c, i, j;
	cudaError_t	rc;
	PLCUDA_RESULT_TYPE result;

	/* command line options */
	while ((c = getopt(argc, argv, "s:c:")) >= 0)
	{
		switch (c)
		{
			case 's':
				arg_bufsz = atol(optarg);
				if (arg_bufsz < 0)
					EEXIT("invalid argument buffer size");
				break;
			case 'c':
				if (arg_catalog)
					EEXIT("argument catalog specified twice");
				arg_catalog = strdup(optarg);
				break;
			default:
				EEXIT("unknown option '%c'", c);
				break;
		}
	}
#if PLCUDA_NUM_ARGS > 0
	if (!arg_catalog)
		EEXIT("no argument catalog was specified");
	memset(arg_ptrs, 0, sizeof(arg_ptrs));

	/* read arguments from stdin */
	rc = cudaMallocManaged(&arg_buffer, arg_bufsz);
	if (rc != cudaSuccess)
		CUEXIT(rc, "out of managed memory");
	nbytes = arg_bufsz;
	tail = arg_buffer;
	do {
		sz = read(PLCUDA_ARGMENT_FDESC, tail, nbytes);
		if (sz < 0)
		{
			if (errno == EINTR)
				continue;
			EEXIT("failed on read(stdin): %m");
		}
		else if (sz == 0)
			break;		/* end of file */
		assert(sz <= nbytes);
		tail += sz;
		nbytes -= sz;
	} while(nbytes > 0);
	close(PLCUDA_ARGMENT_FDESC);

	pos = arg_buffer;
	cat = arg_catalog;
	for (i=0; i < PLCUDA_NUM_ARGS; i++)
	{
		assert(pos == (char *)MAXALIGN(pos));
		arg_kind[i] = *cat++;
		switch (arg_kind[i])
		{
			case 'N':		/* null value */
				break;
			case 'i':		/* immediate datum */
				if (pos + sizeof(Datum) > tail)
					EEXIT("argument buffer out of range pos=%p tail=%p", pos, tail);
				arg_ptrs[i] = (void *)pos;
				pos += sizeof(Datum);
				break;
			case 'r':		/* indirect fixed-length datum */
				{
					size_t	sz = 0;

					while (*cat >= '0' && *cat <= '9')
						sz = sz * 10 + (*cat++ - '0');
					if (pos + sz > tail)
						EEXIT("argument buffer out of range");
					arg_ptrs[i] = (void *)pos;
					pos += MAXALIGN(sz);
				}
				break;
			case 'v':		/* varlena datum */
				{
					size_t	sz;

					if (pos + VARHDRSZ > tail)
						EEXIT("argument buffer out of range");
					sz = VARSIZE_ANY(pos);
					if (pos + sz > tail)
						EEXIT("argument buffer out of range");
					arg_ptrs[i] = (void *)pos;
					pos += MAXALIGN(sz);
				}
				break;
			case 'g':		/* Gstore_fdw */
				{
					GstoreIpcMapping *temp, *prev;
					cudaError_t	rc;

					if (pos + sizeof(GstoreIpcHandle) > tail)
						EEXIT("argument buffer out of range");
					temp = (GstoreIpcMapping *)
						calloc(1, sizeof(GstoreIpcMapping));
					memcpy(&temp->h, pos, sizeof(GstoreIpcHandle));
					pos += MAXALIGN(sizeof(GstoreIpcHandle));
					for (j=0; j < i; j++)
					{
						if (arg_kind[j] != 'g')
							continue;
						prev = (GstoreIpcMapping *)arg_ptrs[j];
						if (memcmp(&prev->h.ipc_mhandle.r,
								   &temp->h.ipc_mhandle.r,
								   sizeof(cudaIpcMemHandle_t)) == 0)
						{
							temp->map = prev->map;
							break;
						}
					}
					if (!temp->map)
					{
						rc = cudaIpcOpenMemHandle(&temp->map,
												  temp->h.ipc_mhandle.r,
											   cudaIpcMemLazyEnablePeerAccess);
						if (rc != cudaSuccess)
							CUEXIT(rc, "failed on cudaIpcOpenMemHandle");
					}
					arg_ptrs[i] = temp;
				}
				break;
			default:
				EEXIT("wrong argument catalog: %s", arg_catalog);
				break;
		}
	}
	if (i != PLCUDA_NUM_ARGS)
		EEXIT("invalid argument catalog: %s", arg_catalog);
#else
	if (arg_catalog && strlen(arg_catalog) > 0)
		EEXIT("argument catalog is longer than expected");
#endif
	/* launch user defined code block */
	result = plcuda_main(arg_ptrs);
#if PLCUDA_RESULT_TYPLEN == -1
	if (!result)
		return 1;		/* returns NULL */
	buffer = (char *)result;
	nbytes = VARSIZE_ANY(buffer);
#elif PLCUDA_RESULT_TYPLEN > 0
#if PLCUDA_RESULT_TYPBYVAL
	buffer = (char *)&result;
#else
	if (!result)
		return 1;		/* return NULL */
	buffer = (char *)result;
#endif
	nbytes = PLCUDA_RESULT_TYPLEN;
#else
#error "unexpected result type properties"
#endif
	/* write back the result of PL/CUDA */
	do {
		sz = write(PLCUDA_RESULT_FDESC, buffer, nbytes);
		if (sz < 0)
		{
			if (errno == EINTR)
				continue;
			EEXIT("failed on write: %m");
		}
		buffer += sz;
		nbytes -= sz;
	} while (nbytes > 0);
	close(PLCUDA_RESULT_FDESC);

	return 0;
}
