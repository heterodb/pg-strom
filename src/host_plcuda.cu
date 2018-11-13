#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>

#define EEXIT(fmt, ...)								\
	do {											\
		fprintf(stderr, fmt "\n",##__VA_ARGS__);	\
		exit(2);									\
	} while(0)

static cudaIpcMemHandle_t
decode_ipcmhandle(const char *hex)
{
	cudaIpcMemHandle_t handle;
	const char	   *pos = hex;
	unsigned char  *dst = (unsigned char *)&handle;
	int				i, c0, c1;

	for (i=0; *pos != '\0'; i++)
	{
		if (i >= sizeof(cudaIpcMemHandle_t))
			EEXIT("IPC mhandle too large");

		c0 = *pos++;
		if (c0 >= '0' && c0 <= '9')
			c0 = c0 - '0';
		else if (c0 >= 'a' && c0 <= 'f')
			c0 = c0 - 'a' + 10;
		else if (c0 >= 'A' && c0 <= 'F')
			c0 = c0 - 'A' + 10;
		else
			EEXIT("invalid HEX character: %s", hex);
		assert((c0 & 0xfff0) == 0);

		c1 = *pos++;
		if (c1 >= '0' && c1 <= '9')
			c1 = c1 - '0';
		else if (c1 >= 'a' && c1 <= 'f')
			c1 = c1 - 'a' + 10;
		else if (c1 >= 'A' && c1 <= 'F')
			c1 = c1 - 'A' + 10;
		else
			EEXIT("invalid HEX character: %s", hex);
		assert((c1 & 0xfff0) == 0);

		dst[i] = (c0 << 4) | c1;
	}
	if (i != sizeof(cudaIpcMemHandle_t))
		EEXIT("IPC mhandle length mismatch");
	return handle;
}

int main(int argc, char * const argv[])
{
	int			fdesc;
	struct stat	stbuf;
	void	   *buffer;
	ssize_t		s, nbytes;
	int			c, i, j, k;
	const char *arg_fname = NULL;
	const char *res_fname = NULL;
	void	   *arg_buffer = NULL;
#if PLCUDA_NUM_ARGS > 0
	cl_ulong	argval[PLCUDA_NUM_ARGS];	/* for immediate values */
	void	   *argptr[PLCUDA_NUM_ARGS];
#else
#define argval		NULL
#define argptr		NULL
#endif
	cudaError_t	rc;
	PLCUDA_RESULT_TYPE result;

	/* command line options */
	while ((c = getopt(argc, argv, "a:r:")) >= 0)
	{
		switch (c)
		{
			case 'a':
				if (arg_fname)
					EEXIT("argument buffer is specified twice");
				arg_fname = strdup(optarg);
				if (!arg_fname)
					EEXIT("out of memory");
				break;
			case 'r':
				if (res_fname)
					EEXIT("result buffer is specified twice");
				res_fname = strdup(optarg);
				if (!res_fname)
					EEXIT("out of memory");
				break;
			default:
				EEXIT("unknown option '%c'", c);
				break;
		}
	}
	if (!res_fname)
		EEXIT("no result buffer");
	if (arg_fname)
	{
		fdesc = shm_open(arg_fname, O_RDONLY, 0600);
		if (fdesc < 0)
			EEXIT("failed on open('%s'): %m", arg_fname);
		if (fstat(fdesc, &stbuf) != 0)
			EEXIT("failed on fstat('%s'): %m", arg_fname);
		rc = cudaMallocManaged(&arg_buffer, stbuf.st_size);
		if (rc != cudaSuccess)
			EEXIT("failed on cudaMallocManaged: %s", cudaGetErrorName(rc));
		nbytes = 0;
		do {
			s = read(fdesc, (char *)arg_buffer + nbytes,
					 stbuf.st_size - nbytes);
			if (s < 0)
			{
				if (errno == EINTR)
					continue;
				EEXIT("failed on read(%s): %m", arg_fname);
			}
			else if (s == 0)
				EEXIT("could not read entire PL/CUDA arguments");
			nbytes += s;
		} while (nbytes < stbuf.st_size);

		close(fdesc);
	}

	/* setup arguments */
	for (i=optind, j=0; i < argc; i++, j++)
	{
		const char *tok = argv[i];
		void	   *ptr = NULL;

		if (j >= PLCUDA_NUM_ARGS)
			EEXIT("too larget arguments for PL/CUDA function");
		if (strcmp(tok, "__null__") == 0)
			ptr = NULL;
		else if (strncmp(tok, "v:", 2) == 0)
		{
			/* immediate value */
			argval[i] = strtol(tok+2, NULL, 16);
			ptr = &argval[i];
		}
		else if (strncmp(tok, "r:", 2) == 0)
		{
			/* reference to argument buffer */
			cl_ulong	offset = strtol(tok+2, NULL, 16);
			ptr = (char *)arg_buffer + offset;
		}
		else if (strncmp(tok, "g:", 2) == 0)
		{
			/* IPC-handle of GPU memory store */
			for (k=0; k < j; k++)
			{
				/* reuse if same IPC handle is already opened */
				if (strcmp(tok, argv[optind+k]) == 0)
				{
					ptr = argptr[k];
					break;
				}
			}
			if (!ptr)
			{
				cudaIpcMemHandle_t ipc_mhandle = decode_ipcmhandle(tok+2);

				rc = cudaIpcOpenMemHandle(&ptr, ipc_mhandle,
										  cudaIpcMemLazyEnablePeerAccess);
				if (rc != cudaSuccess)
					EEXIT("failed on cudaIpcOpenMemHandle: %s",
						  cudaGetErrorName(rc));
			}
		}
		else
			EEXIT("Bug? unexpected PL/CUDA argument format: %s", tok);
		argptr[j] = ptr;
	}

	/* open the result buffer */
	fdesc = shm_open(res_fname, O_RDWR, 0600);
	if (fdesc < 0)
		EEXIT("failed on shm_open('%s'): %m", res_fname);
	if (fstat(fdesc, &stbuf) != 0)
		EEXIT("failed on fstat(2): %m");
	/* kick user defined portion */
	result = plcuda_main(argptr);
#if PLCUDA_RESULT_TYPLEN == -1
	if (!result)
		return 1;	/* returns NULL */
	buffer = result;
	nbytes = VARSIZE_ANY(buffer);
#elif PLCUDA_RESULT_TYPLEN > 0
#if PLCUDA_RESULT_TYPBYVAL
	buffer = &result;
#else
	if (!result)
		return 1;	/* returns NULL */
	buffer = result;
#endif
	nbytes = PLCUDA_RESULT_TYPLEN;
#else
#error "unexpected result type properties"
#endif
	/* write back the result of PL/CUDA */
	do {
		s = write(fdesc, buffer, nbytes);
		if (s < 0)
		{
			if (errno == EINTR)
				continue;
			EEXIT("failed on write('%s'): %m", res_fname);
		}
		buffer = (char *)buffer + s;
		nbytes -= s;
	} while (nbytes > 0);

	close(fdesc);

	return 0;
}
