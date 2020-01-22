/*
 * nvrtc.c
 *
 * A thin wrapper to call NVRTC library functions.
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
 * pgstrom_init_nvrtc
 */
void
pgstrom_init_nvrtc(void)
{
	CUresult	rc;
	nvrtcResult	rv;
	int			cuda_version;
	int			nvrtc_version;
	int			major, minor;
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
	rv = nvrtcVersion(&major, &minor);
	if (rv != NVRTC_SUCCESS)
		elog(ERROR, "failed on nvrtcVersion: %d", (int)rv);
	nvrtc_version = major * 1000 + minor * 10;

	LOOKUP_NVRTC_FUNCTION(nvrtcGetErrorString);
	LOOKUP_NVRTC_FUNCTION(nvrtcCreateProgram);
	LOOKUP_NVRTC_FUNCTION(nvrtcDestroyProgram);
	LOOKUP_NVRTC_FUNCTION(nvrtcCompileProgram);
	LOOKUP_NVRTC_FUNCTION(nvrtcGetPTXSize);
	LOOKUP_NVRTC_FUNCTION(nvrtcGetPTX);
	LOOKUP_NVRTC_FUNCTION(nvrtcGetProgramLogSize);
	LOOKUP_NVRTC_FUNCTION(nvrtcGetProgramLog);
	if (major >= 10)		/* CUDA10.0 */
	{
		LOOKUP_NVRTC_FUNCTION(nvrtcAddNameExpression);
		LOOKUP_NVRTC_FUNCTION(nvrtcGetLoweredName);
	}

	if (cuda_version == nvrtc_version)
		elog(LOG, "NVRTC %d.%d is successfully loaded.", major, minor);
	else
		elog(LOG, "NVRTC %d.%d is successfully loaded, but CUDA driver expects %d.%d. Check /etc/ld.so.conf or LD_LIBRARY_PATH configuration.",
			 major, minor,
			 (cuda_version / 1000),
			 (cuda_version % 1000) / 10);
}
