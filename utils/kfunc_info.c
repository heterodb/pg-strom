#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <libgen.h>
#include <cuda.h>
#include <nvrtc.h>


#define lengthof(array) (sizeof (array) / sizeof ((array)[0]))

static const char  *cmdname;

static void usage(void)
{
	fprintf(stderr, 
			"usage: %s [options...] <kernel source>\n"
			"  options:\n"
			"    -k <function> : specifies the kernel name\n"
			"    -d <device>   : specifies the target device\n"
			"    -c x.y        : specifies the target capability\n"
			"                    NOTE: -c and -d are exclusive\n"
			"    -s <dynamic shmem usage per thread>\n"
			"    -S <dynamic shmem usage per block>\n"
			"    -v            : print NVRTC version info\n"
			"    -l            : print List of devices\n"
			"    -h            : print this message and exit\n",
			cmdname);
	exit(1);
}

static void cuda_error(CUresult rc, const char *apiname)
{
	const char *errName;
	const char *errStr;

	cuGetErrorName(rc, &errName);
	cuGetErrorString(rc, &errStr);

	fprintf(stderr, "failed on %s: %s - %s\n", apiname, errName, errStr);
	exit(1);
}

static void nvrtc_error(nvrtcResult rc, const char *apiname)
{
	fprintf(stderr, "failed on %s: %s\n", apiname, nvrtcGetErrorString(rc));
	exit(1);
}

static void print_nvrtc_version(void)
{
	int			major;
	int			minor;
	nvrtcResult	rc;

	rc = nvrtcVersion(&major, &minor);
    if (rc != NVRTC_SUCCESS)
	{
		fprintf(stderr, "failed on nvrtcVersion: %s\n",
				nvrtcGetErrorString(rc));
		exit(1);
	}
	printf("NVRTC Version: %d.%d\n", major, minor);
}

static inline char *
format_bytesz(size_t nbytes)
{
	char	buf[256];
	char   *result;

	if (nbytes > (size_t)(1UL << 43))
		snprintf(buf, sizeof(buf),
				 "%.2fTB", (double)nbytes / (double)(1UL << 40));
	else if (nbytes > (size_t)(1UL << 33))
		snprintf(buf, sizeof(buf),
				 "%.2fGB", (double)nbytes / (double)(1UL << 30));
	else if (nbytes > (size_t)(1UL << 23))
		snprintf(buf, sizeof(buf), "%zuMB", nbytes / (1UL << 20));
	else if (nbytes > (size_t)(1UL << 13))
		snprintf(buf, sizeof(buf), "%zuKB", nbytes / (1UL << 10));
	else
		snprintf(buf, sizeof(buf), "%zuB", nbytes);

	result = strdup(buf);
	if (!result)
	{
		fputs("out of memory", stderr);
		exit(1);
	}
	return result;
}

static inline char *
format_clock(size_t clock)
{
	char	buf[256];
	char   *result;

	if (clock > (size_t)(1UL << 31))
		snprintf(buf, sizeof(buf),
				 "%.2fGHz", (double)clock / (double)(1UL << 30));
	else if (clock > (size_t)(1UL << 21))
		snprintf(buf, sizeof(buf), "%zuMHz", clock / (1UL << 20));
	else if (clock > (size_t)(1UL << 11))
		snprintf(buf, sizeof(buf), "%zuKHz", clock / (1UL << 10));
	else
		snprintf(buf, sizeof(buf), "%zuHz", clock);

	result = strdup(buf);
	if (!result)
	{
		fputs("out of memory", stderr);
		exit(1);
	}
	return result;
}

static void print_cuda_devices(int num_devices)
{
	CUdevice	dev;
	CUresult	rc;
	char		dev_name[512];
	size_t		dev_total_mem;
	int			dev_mem_clk;
	int			dev_mem_width;
	int			dev_mpu_nums;
	int			dev_mpu_clk;
	int			dev_nregs_mpu;
	int			dev_nregs_blk;
	int			dev_l2_sz;
	int			dev_cap_major;
	int			dev_cap_minor;
	int			i, j;
	struct {
		int		attr;
		int	   *dptr;
	} catalog[] = {
		{ CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,        &dev_mem_clk },
		{ CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,  &dev_mem_width },
		{ CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,     &dev_mpu_nums },
		{ CU_DEVICE_ATTRIBUTE_CLOCK_RATE,               &dev_mpu_clk },
		{ CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,&dev_nregs_mpu},
		{ CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,  &dev_nregs_blk },
		{ CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,            &dev_l2_sz },
		{ CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, &dev_cap_major },
		{ CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, &dev_cap_minor },
	};

	for (i=0; i < num_devices; i++)
	{
		int		cores_per_mpu = -1;

		rc = cuDeviceGet(&dev, i);
		if (rc != CUDA_SUCCESS)
			cuda_error(rc, "cuDeviceGet");

		rc = cuDeviceGetName(dev_name, sizeof(dev_name), dev);
		if (rc != CUDA_SUCCESS)
			cuda_error(rc, "cuDeviceGetName");

		rc = cuDeviceTotalMem(&dev_total_mem, dev);
		if (rc != CUDA_SUCCESS)
			cuda_error(rc, "cuDeviceTotalMem");

		for (j=0; j < lengthof(catalog); j++)
		{
			rc = cuDeviceGetAttribute(catalog[j].dptr, catalog[j].attr, dev);
			if (rc != CUDA_SUCCESS)
				cuda_error(rc, "cuDeviceGetAttribute");
		}

		/* Number of CUDA cores */
		if (dev_cap_major == 1)
			cores_per_mpu = 8;
		else if (dev_cap_major == 2)
		{
			if (dev_cap_minor == 0)
				cores_per_mpu = 32;
			else if (dev_cap_minor == 1)
				cores_per_mpu = 48;
			else
				cores_per_mpu = -1;
		}
		else if (dev_cap_major == 3)
			cores_per_mpu = 192;
		else if (dev_cap_major == 5)
			cores_per_mpu = 128;

		printf("GPU%d - %s (capability: %d.%d), %d %s %s,"
			   " L2 %s, RAM %s (%dbits, %s), Regs=%d/%d\n",
			   i, dev_name, dev_cap_major, dev_cap_minor,
			   (cores_per_mpu > 0 ? cores_per_mpu : 1) * dev_mpu_nums,
			   (cores_per_mpu > 0 ? "CUDA cores" : "SMs"),
			   format_clock(dev_mpu_clk),
			   format_bytesz(dev_l2_sz),
			   format_bytesz(dev_total_mem),
			   dev_mem_width,
			   format_clock((size_t)dev_mem_clk * 1000),
			   dev_nregs_blk,
			   dev_nregs_mpu);
	}
}

static char *
load_kernel_source(const char *source_file, int *link_dev_runtime)
{
	int				fdesc;
	struct stat		st_buf;
	char		   *source;

	fdesc = open(source_file, O_RDONLY);
	if (fdesc < 0)
	{
		fprintf(stderr, "failed to open '%s': %m\n", source_file);
		exit(1);
	}
	if (fstat(fdesc, &st_buf) != 0)
	{
		fputs("failed on fstat(2)", stderr);
		exit(1);
	}
	source = malloc(st_buf.st_size);
	if (!source)
	{
		fputs("out of memory", stderr);
		exit(1);
	}
	if (read(fdesc, source, st_buf.st_size) != st_buf.st_size)
	{
		fputs("could not read entire source file", stderr);
		exit(1);
	}

	/* Does it need to link device runtime? */
	if (strstr(source, "#define CUDA_DYNPARA_H\n") != NULL)
		*link_dev_runtime = 1;
	else
		*link_dev_runtime = 0;


	close(fdesc);

	return source;
}

static void
link_device_libraries(void *ptx_image, size_t ptx_image_len,
					  void **p_bin_image, size_t *p_bin_image_len,
					  long target_capability)
{
	CUlinkState		lstate;
	CUjit_option	jit_options[10];
	void		   *jit_option_values[10];
	int				jit_index = 0;
	CUresult		rc;
	char			pathname[1024];

	/*
	 * JIT options
	 */
	jit_options[jit_index] = CU_JIT_TARGET;
	jit_option_values[jit_index] = (void *)target_capability;
	jit_index++;
#ifdef PGSTROM_DEBUG
	jit_options[jit_index] = CU_JIT_GENERATE_DEBUG_INFO;
	jit_option_values[jit_index] = (void *)1UL;
	jit_index++;

	jit_options[jit_index] = CU_JIT_GENERATE_LINE_INFO;
	jit_option_values[jit_index] = (void *)1UL;
	jit_index++;
#endif
	/* makes a linkage object */
	rc = cuLinkCreate(jit_index, jit_options, jit_option_values, &lstate);
	if (rc != CUDA_SUCCESS)
		cuda_error(rc, "cuLinkCreate");

    /* add the base PTX image */
	rc = cuLinkAddData(lstate, CU_JIT_INPUT_PTX,
					   ptx_image, ptx_image_len,
					   "PG-Strom", 0, NULL, NULL);
	if (rc != CUDA_SUCCESS)
		cuda_error(rc, "cuLinkAddData");

    /* libcudart.a, if any */
	snprintf(pathname, sizeof(pathname), "%s/libcudadevrt.a",
			 CUDA_LIBRARY_PATH);
	rc = cuLinkAddFile(lstate, CU_JIT_INPUT_LIBRARY, pathname,
					   0, NULL, NULL);
	if (rc != CUDA_SUCCESS)
		cuda_error(rc, "cuLinkAddFile");

	/* do the linkage */
	rc = cuLinkComplete(lstate, p_bin_image, p_bin_image_len);
	if (rc != CUDA_SUCCESS)
		cuda_error(rc, "cuLinkComplete");


}

static CUmodule
build_kernel_source(const char *source_file, long target_capability)
{
	char		   *source;
	int				link_dev_runtime;
	nvrtcProgram	program;
	nvrtcResult		rc;
	char			arch_buf[128];
	const char	   *options[10];
	int				opt_index = 0;
	int				build_failure = 0;
	char		   *build_log;
	size_t			build_log_len;
	char		   *ptx_image;
	size_t			ptx_image_len;
	void		   *bin_image;
	size_t			bin_image_len;
	CUmodule		cuda_module;
	CUresult		cuda_rc;

	source = load_kernel_source(source_file, &link_dev_runtime);
	rc = nvrtcCreateProgram(&program,
							source,
							NULL,
							0,
							NULL,
							NULL);
	if (rc != NVRTC_SUCCESS)
		nvrtc_error(rc, "nvrtcCreateProgram");

	/*
	 * Put command line options as cuda_program.c doing
	 */
	options[opt_index++] = "-I " CUDA_INCLUDE_PATH;
	snprintf(arch_buf, sizeof(arch_buf),
			 "--gpu-architecture=compute_%ld", target_capability);
	options[opt_index++] = arch_buf;
#ifdef PGSTROM_DEBUG
	options[opt_index++] = "--device-debug";
	options[opt_index++] = "--generate-line-info";
#endif
	options[opt_index++] = "--use_fast_math";
	if (link_dev_runtime)
		options[opt_index++] = "--relocatable-device-code=true";

	/*
	 * Kick runtime compiler
	 */
	rc = nvrtcCompileProgram(program, opt_index, options);
	if (rc != NVRTC_SUCCESS)
	{
		if (rc == NVRTC_ERROR_COMPILATION)
			build_failure = 1;
		else
			nvrtc_error(rc, "nvrtcCompileProgram");
	}

	/*
	 * Print build log
	 */
	rc = nvrtcGetProgramLogSize(program, &build_log_len);
	if (rc != NVRTC_SUCCESS)
		nvrtc_error(rc, "nvrtcGetProgramLogSize");
	build_log = malloc(build_log_len + 1);
	if (!build_log)
	{
		fputs("out of memory", stderr);
		exit(1);
	}
	rc = nvrtcGetProgramLog(program, build_log);
	if (rc != NVRTC_SUCCESS)
		nvrtc_error(rc, "nvrtcGetProgramLog");

	if (build_log_len > 1)
		printf("build log:\n%s\n", build_log);
	if (build_failure)
		exit(1);

	/*
	 * Get PTX Image
	 */
	rc = nvrtcGetPTXSize(program, &ptx_image_len);
	if (rc != NVRTC_SUCCESS)
		nvrtc_error(rc, "nvrtcGetPTXSize");
	ptx_image = malloc(ptx_image_len + 1);
	if (!ptx_image)
	{
		fputs("out of memory", stderr);
		exit(1);
	}
	rc = nvrtcGetPTX(program, ptx_image);
	if (rc != NVRTC_SUCCESS)
		nvrtc_error(rc, "nvrtcGetPTX");
	ptx_image[ptx_image_len] = '\0';

	/*
	 * Link device runtime if needed
	 */
	if (link_dev_runtime)
	{
		link_device_libraries(ptx_image, ptx_image_len,
							  &bin_image, &bin_image_len,
							  target_capability);
	}
	else
	{
		bin_image = ptx_image;
		bin_image_len = ptx_image_len;
	}

	cuda_rc = cuModuleLoadData(&cuda_module, bin_image);
	if (cuda_rc != CUDA_SUCCESS)
		cuda_error(rc, "cuModuleLoadData");
	return cuda_module;
}

static long dynamic_shmem_per_thread = sizeof(long);
static long dynamic_shmem_per_block  = 0;

static size_t cb_occupancy_shmem_size(int block_size)
{
	return dynamic_shmem_per_thread * block_size;
}

static void print_function_attrs(CUmodule cuda_module, const char *func_name)
{
	CUfunction	kernel;
	CUresult	rc;
	int			max_threads_per_block;
	int			shared_mem_sz;
	int			const_mem_sz;
	int			local_mem_sz;
	int			num_regs;
	int			ptx_version;
	int			binary_version;
	int			cache_mode_ca;
	int			min_grid_sz;
	int			max_block_sz;
	int			i;
	struct {
		CUfunction_attribute attr;
		int	   *vptr;
	} catalog[] = {
		{ CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, &max_threads_per_block },
		{ CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,     &shared_mem_sz },
		{ CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,      &const_mem_sz },
		{ CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,      &local_mem_sz },
		{ CU_FUNC_ATTRIBUTE_NUM_REGS,              &num_regs },
		{ CU_FUNC_ATTRIBUTE_PTX_VERSION,           &ptx_version },
		{ CU_FUNC_ATTRIBUTE_BINARY_VERSION,        &binary_version },
		{ CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,         &cache_mode_ca },
	};

	rc = cuModuleGetFunction(&kernel, cuda_module, func_name);
	if (rc != CUDA_SUCCESS)
		cuda_error(rc, "cuModuleGetFunction");

	for (i=0; i < lengthof(catalog); i++)
	{
		rc = cuFuncGetAttribute(catalog[i].vptr,
								catalog[i].attr,
								kernel);
		if (rc != CUDA_SUCCESS)
			cuda_error(rc, "cuFuncGetAttribute");
	}

	rc = cuOccupancyMaxPotentialBlockSize(&min_grid_sz,
										  &max_block_sz,
										  kernel,
										  cb_occupancy_shmem_size,
										  dynamic_shmem_per_block,
										  1024 * 1024);
	if (rc != CUDA_SUCCESS)
		cuda_error(rc, "cuOccupancyMaxPotentialBlockSize");

	printf("Kernel Function:    %s\n"
		   "  Max threads per block:    %d\n"
		   "  Shared memory usage:      %d\n"
		   "  Constant memory usage:    %d\n"
		   "  Local memory usage:       %d\n"
		   "  Number of registers:      %d\n"
		   "  PTX version:              %d\n"
		   "  Binary version:           %d\n"
		   "  Global memory caching:    %s\n"
		   "  Max potential block size: %u\n"
		   "  (shmem usage: %ld/thread + %ld/block)\n",
		   func_name,
		   max_threads_per_block,
		   shared_mem_sz,
		   const_mem_sz,
		   local_mem_sz,
		   num_regs,
		   ptx_version,
		   binary_version,
		   cache_mode_ca ? "enabled" : "disabled",
		   max_block_sz,
		   dynamic_shmem_per_thread,
		   dynamic_shmem_per_block);
}

#define MAX_KERNEL_FUNCTIONS	1024

int main(int argc, char *argv[])
{
	char	   *kernel_source;
	char	   *kfunc_names[MAX_KERNEL_FUNCTIONS];
	int			kfunc_index = 0;
	int			target_device = -1;
	long		target_capability = -1;
	int			print_version = 0;
	int			print_devices = 0;
	int			num_devices;
	int			i, opt;
	int			major;
	int			minor;
	CUdevice	device;
	CUcontext	context;
	CUmodule	cuda_module;
	CUresult	rc;

	/* misc initialization */
	cmdname = basename(strdup(argv[0]));
	cuInit(0);
	rc = cuDeviceGetCount(&num_devices);
	if (rc != CUDA_SUCCESS)
		cuda_error(rc, "cuDeviceGetCount");

	while ((opt = getopt(argc, argv, "k:d:c:s:S:vlh")) >= 0)
	{
		switch (opt)
		{
			case 'k':
				if (kfunc_index == MAX_KERNEL_FUNCTIONS)
				{
					fputs("Too much kernel function specified", stderr);
					return 1;
				}
				kfunc_names[kfunc_index++] = strdup(optarg);
				break;
			case 'd':
				if (target_device >= 0)
				{
					fputs("-d is specified twice or more", stderr);
					usage();
				}
				if (target_capability >= 0)
				{
					fputs("-d and -c are exclusive option", stderr);
					usage();
				}
				target_device = atoi(optarg);
				if (target_device < 0 || target_device >= num_devices)
				{
					fprintf(stderr, "invalid device: -d %d\n", target_device);
					usage();
				}
				break;
			case 'c':
				if (target_capability >= 0)
				{
					fputs("-c is specified twice or more", stderr);
					usage();
				}
				if (target_device >= 0)
				{
					fputs("-d and -c are exclusive option", stderr);
					usage();
				}
				if (sscanf(optarg, "%d.%d", &major, &minor) != 2)
				{
					fprintf(stderr, "invalid capability format: -c %s\n",
							optarg);
					usage();
				}
				target_capability = major * 10 + minor;
				break;
			case 's':
				dynamic_shmem_per_thread = atol(optarg);
				if (dynamic_shmem_per_thread < 0)
				{
					fprintf(stderr, "invalid dynamic shmem per thread: %ld\n",
							dynamic_shmem_per_thread);
					usage();
				}
				break;
			case 'S':
				dynamic_shmem_per_block = atol(optarg);
				if (dynamic_shmem_per_block < 0)
				{
					fprintf(stderr, "invalid dynamic shmem per block: %ld",
							dynamic_shmem_per_block);
					usage();
				}
				break;
			case 'v':
				print_version = 1;
				break;
			case 'l':
				print_devices = 1;
				break;
			case 'h':
			default:
				usage();
				break;
		}
	}

	if (optind + 1 != argc)
	{
		if (print_version || print_devices)
		{
			if (print_version)
				print_nvrtc_version();
			if (print_devices)
				print_cuda_devices(num_devices);
			return 0;
		}
		fputs("no kernel source is specified", stderr);
		usage();
	}
	kernel_source = argv[optind];

	if (target_capability < 0)
	{
		CUdevice	dev;

		if (target_device < 0)
			target_device = 0;	/* default device */

		rc = cuDeviceGet(&dev, target_device);
		if (rc != CUDA_SUCCESS)
			cuda_error(rc, "cuDeviceGet");

		rc = cuDeviceGetAttribute(&major,
					CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
		if (rc != CUDA_SUCCESS)
			cuda_error(rc, "cuDeviceGetAttribute");
		rc = cuDeviceGetAttribute(&minor,
					CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
		if (rc != CUDA_SUCCESS)
			cuda_error(rc, "cuDeviceGetAttribute");

		target_capability = 10 * major + minor;
	}

	if (print_version)
		print_nvrtc_version();
	if (print_devices)
		print_cuda_devices(num_devices);

	/* make a dummy context */
	rc = cuDeviceGet(&device, 0);
	if (rc != CUDA_SUCCESS)
		cuda_error(rc, "cuDeviceGet");
	rc = cuCtxCreate(&context, 0, device);
	if (rc != CUDA_SUCCESS)
		cuda_error(rc, "cuCtxCreate");

	cuda_module = build_kernel_source(kernel_source, target_capability);

	for (i=0; i < kfunc_index; i++)
	{
		if (i > 0)
			putchar('\n');
		print_function_attrs(cuda_module, kfunc_names[i]);
	}

	/* drop a cuda context */
	rc = cuCtxDestroy(context);
	if (rc != CUDA_SUCCESS)
		cuda_error(rc, "cuCtxDestroy");

	return 0;
}
