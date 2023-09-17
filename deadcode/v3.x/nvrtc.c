/*
 * nvrtc.c
 *
 * A thin wrapper to call NVRTC library functions.
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include <dlfcn.h>

/*
 * nvrtcGetErrorString
 */
static const char *(*p_nvrtcGetErrorString)(
	nvrtcResult result) = NULL;

const char *
nvrtcGetErrorString(nvrtcResult result)
{
	return p_nvrtcGetErrorString(result);
}

/*
 * nvrtcVersion
 */
static nvrtcResult (*p_nvrtcVersion)(
	int *major,
	int *minor) = NULL;

nvrtcResult
nvrtcVersion(int *major, int *minor)
{
	return p_nvrtcVersion(major, minor);
}

/*
 * nvrtcCreateProgram
 */
static nvrtcResult (*p_nvrtcCreateProgram)(
	nvrtcProgram *prog,
	const char *src,
	const char *name,
	int numHeaders,
	const char * const *headers,
	const char * const *includeNames) = NULL;

nvrtcResult
nvrtcCreateProgram(nvrtcProgram *prog,
				   const char *src,
				   const char *name,
				   int numHeaders,
				   const char * const *headers,
				   const char * const *includeNames)
{
	return p_nvrtcCreateProgram(prog, src, name,
								numHeaders, headers, includeNames);
}

/*
 * nvrtcDestroyProgram
 */
static nvrtcResult (*p_nvrtcDestroyProgram)(
	nvrtcProgram *prog) = NULL;

nvrtcResult
nvrtcDestroyProgram(nvrtcProgram *prog)
{
	return p_nvrtcDestroyProgram(prog);
}

/*
 * nvrtcCompileProgram
 */
static nvrtcResult (*p_nvrtcCompileProgram)(
	nvrtcProgram prog,
	int numOptions,
	const char * const *options) = NULL;

nvrtcResult
nvrtcCompileProgram(nvrtcProgram prog,
					int numOptions,
					const char * const *options)
{
	return p_nvrtcCompileProgram(prog, numOptions, options);
}

/*
 * nvrtcGetPTXSize
 */
static nvrtcResult (*p_nvrtcGetPTXSize)(
	nvrtcProgram prog,
	size_t *ptxSizeRet) = NULL;

nvrtcResult
nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet)
{
	return p_nvrtcGetPTXSize(prog, ptxSizeRet);
}

/*
 * nvrtcGetPTX
 */
static nvrtcResult (*p_nvrtcGetPTX)(
	nvrtcProgram prog,
	char *ptx) = NULL;

nvrtcResult
nvrtcGetPTX(nvrtcProgram prog, char *ptx)
{
	return p_nvrtcGetPTX(prog, ptx);
}

/*
 * nvrtcGetProgramLogSize
 */
static nvrtcResult (*p_nvrtcGetProgramLogSize)(
	nvrtcProgram prog,
	size_t *logSizeRet) = NULL;

nvrtcResult
nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet)
{
	return p_nvrtcGetProgramLogSize(prog, logSizeRet);
}

/*
 * nvrtcGetProgramLog
 */
static nvrtcResult (*p_nvrtcGetProgramLog)(
	nvrtcProgram prog,
	char *log) = NULL;

nvrtcResult
nvrtcGetProgramLog(nvrtcProgram prog, char *log)
{
	return p_nvrtcGetProgramLog(prog, log);
}

/*
 * nvrtcAddNameExpression
 */
static nvrtcResult (*p_nvrtcAddNameExpression)(
	nvrtcProgram prog,
	const char * const name_expression) = NULL;

nvrtcResult
nvrtcAddNameExpression(nvrtcProgram prog,
					   const char * const name_expression)
{
	return p_nvrtcAddNameExpression(prog, name_expression);
}

/*
 * nvrtcGetLoweredName
 */
static nvrtcResult (*p_nvrtcGetLoweredName)(
	nvrtcProgram prog,
	const char *const name_expression,
	const char** lowered_name) = NULL;

nvrtcResult
nvrtcGetLoweredName(nvrtcProgram prog,
					const char *const name_expression,
					const char** lowered_name)
{
	return p_nvrtcGetLoweredName(prog, name_expression, lowered_name);
}

/*
 * nvrtcGetCUBIN
 */
static nvrtcResult (*p_nvrtcGetCUBIN)(
	nvrtcProgram prog,
	char* cubin) = NULL;

nvrtcResult
nvrtcGetCUBIN(nvrtcProgram prog, char* cubin)
{
	return p_nvrtcGetCUBIN(prog, cubin);
}

/*
 * nvrtcGetCUBINSize
 */
static nvrtcResult (*p_nvrtcGetCUBINSize)(
	nvrtcProgram prog,
	size_t *cubinSizeRet) = NULL;

nvrtcResult
nvrtcGetCUBINSize(nvrtcProgram prog, size_t *cubinSizeRet)
{
	return p_nvrtcGetCUBINSize(prog, cubinSizeRet);
}

/*
 * nvrtcGetNumSupportedArchs
 */
static nvrtcResult (*p_nvrtcGetNumSupportedArchs)(int *numArchs) = NULL;

nvrtcResult
nvrtcGetNumSupportedArchs(int *numArchs)
{
	return p_nvrtcGetNumSupportedArchs(numArchs);
}

/*
 * nvrtcGetSupportedArchs
 */
static nvrtcResult (*p_nvrtcGetSupportedArchs)(int *supportedArchs) = NULL;

nvrtcResult
nvrtcGetSupportedArchs(int *supportedArchs)
{
	return p_nvrtcGetSupportedArchs(supportedArchs);
}

/*
 * lookup_nvrtc_function
 */
static void *
lookup_nvrtc_function(void *handle, const char *func_name)
{
	void   *func_addr = dlsym(handle, func_name);

	if (!func_addr)
		elog(ERROR, "could not find NVRTC symbol \"%s\" - %s",
			 func_name, dlerror());
	return func_addr;
}

#define LOOKUP_NVRTC_FUNCTION(func_name)		\
	p_##func_name = lookup_nvrtc_function(handle, #func_name)

/*
 * pgstrom_nvrtc_version - free from errors once loaded
 */
int
pgstrom_nvrtc_version(void)
{
	static int		nvrtc_version = -1;

	if (nvrtc_version < 0)
	{
		int			major, minor;
		nvrtcResult	rv;

		rv = nvrtcVersion(&major, &minor);
		if (rv != NVRTC_SUCCESS)
			elog(ERROR, "failed on nvrtcVersion: %d", (int)rv);
		nvrtc_version = major * 1000 + minor * 10;
	}
	return nvrtc_version;
}

/*
 * pgstrom_init_nvrtc
 */
void
pgstrom_init_nvrtc(void)
{
	CUresult	rc;
	int			cuda_version;
	int			nvrtc_version;
	char		namebuf[MAXPGPATH];
	void	   *handle;

	rc = cuDriverGetVersion(&cuda_version);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDriverGetVersion: %s", errorText(rc));

	snprintf(namebuf, sizeof(namebuf),
			 "libnvrtc.so.%d.%d",
			 (cuda_version / 1000),
			 (cuda_version % 1000) / 10);
	handle = dlopen(namebuf, RTLD_NOW | RTLD_LOCAL);
	if (!handle)
	{
		handle = dlopen("libnvrtc.so", RTLD_NOW | RTLD_LOCAL);
		if (!handle)
			elog(ERROR, "failed on open '%s' and 'libnvrtc.so': %m", namebuf);
	}
	LOOKUP_NVRTC_FUNCTION(nvrtcVersion);
	nvrtc_version = pgstrom_nvrtc_version();

	LOOKUP_NVRTC_FUNCTION(nvrtcGetErrorString);
	LOOKUP_NVRTC_FUNCTION(nvrtcCreateProgram);
	LOOKUP_NVRTC_FUNCTION(nvrtcDestroyProgram);
	LOOKUP_NVRTC_FUNCTION(nvrtcCompileProgram);
	LOOKUP_NVRTC_FUNCTION(nvrtcGetPTXSize);
	LOOKUP_NVRTC_FUNCTION(nvrtcGetPTX);
	LOOKUP_NVRTC_FUNCTION(nvrtcGetProgramLogSize);
	LOOKUP_NVRTC_FUNCTION(nvrtcGetProgramLog);
	if (nvrtc_version >= 10000)		/* CUDA 10.0 */
	{
		LOOKUP_NVRTC_FUNCTION(nvrtcAddNameExpression);
		LOOKUP_NVRTC_FUNCTION(nvrtcGetLoweredName);
	}
	if (nvrtc_version >= 11010)		/* CUDA 11.1 */
	{
		LOOKUP_NVRTC_FUNCTION(nvrtcGetCUBIN);
		LOOKUP_NVRTC_FUNCTION(nvrtcGetCUBINSize);
	}
	if (nvrtc_version >= 11020)		/* CUDA 11.2 */
	{
		LOOKUP_NVRTC_FUNCTION(nvrtcGetNumSupportedArchs);
		LOOKUP_NVRTC_FUNCTION(nvrtcGetSupportedArchs);
	}

	if (cuda_version == nvrtc_version)
		elog(LOG, "NVRTC %d.%d is successfully loaded.",
			 (nvrtc_version / 1000),
			 (nvrtc_version % 1000) / 10);
	else
		elog(LOG, "NVRTC %d.%d is successfully loaded, but CUDA driver expects %d.%d. Check /etc/ld.so.conf or LD_LIBRARY_PATH configuration.",
			 (nvrtc_version / 1000),
			 (nvrtc_version % 1000) / 10,
			 (cuda_version / 1000),
			 (cuda_version % 1000) / 10);
}
