#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "pystrom.h"

static bool
ipcmem__parse_ident(const char *ident,
					int *p_device_id,
					cudaIpcMemHandle_t *p_ipc_mhandle,
					size_t *p_bytesize,
					char *p_type_code,
					int *p_nattrs,
					long *p_nitems)
{
	char	   *buffer = alloca(strlen(ident) + 1);
	char	   *tok, *save;
	int			device_id = -1;
	cudaIpcMemHandle_t ipc_mhandle;
	ssize_t		bytesize = -1;
	char		type_code = '\0';
	int			nattrs = -1;
	ssize_t		nitems = -1;
	uint32_t	mask = 0;
	ssize_t		unitsz = 1;

	strcpy(buffer, ident);
	for (tok = strtok_r(buffer, ",", &save);
		 tok != NULL;
		 tok = strtok_r(NULL, ",", &save))
	{
		char   *pos = strchr(tok, '=');

		if (!pos)
		{
			PyErr_Format(PyExc_ValueError,
						 "invalid GPU memory identifier");
			return false;
		}
		*pos++ = '\0';

		if (strcmp(tok, "device_id") == 0)
		{
			device_id = atoi(pos);
			mask |= 0x0001;
		}
		else if (strcmp(tok, "bytesize") == 0)
		{
			bytesize = atol(pos);
			mask |= 0x0002;
		}
		else if (strcmp(tok, "ipc_handle") == 0)
		{
			size_t		len = strlen(pos);

			if (len != 2 * sizeof(cudaIpcMemHandle_t) ||
				hex_decode(pos, len, (char *)&ipc_mhandle) < 0)
			{
				PyErr_Format(PyExc_ValueError,
							 "invalid cudaIpcMemHandle [%s]", pos);
				return false;
			}
			mask |= 0x0004;
		}
		else if (strcmp(tok, "format") == 0)
		{
			if (strcmp(pos, "cupy-int16") == 0)
			{
				type_code = 'h';
				unitsz = sizeof(int16_t);
			}
			else if (strcmp(pos, "cupy-int32") == 0)
			{
				type_code = 'i';
				unitsz = sizeof(int32_t);
			}
			else if (strcmp(pos, "cupy-int64") == 0)
			{
				type_code = 'l';
				unitsz = sizeof(int64_t);
			}
			else if (strcmp(pos, "cupy-float16") == 0)
			{
				type_code = 'e';
				unitsz = sizeof(int16_t);
			}
			else if (strcmp(pos, "cupy-float32") == 0)
			{
				type_code = 'f';
				unitsz = sizeof(int32_t);
			}
			else if (strcmp(pos, "cupy-float64") == 0)
			{
				type_code = 'd';
				unitsz = sizeof(int64_t);
			}
			else
			{
				PyErr_Format(PyExc_TypeError,
							 "unknown format [%s]", pos);
				return false;
			}
			mask |= 0x0008;
		}
		else if (strcmp(tok, "nitems") == 0)
		{
			nitems = atol(pos);
			mask |= 0x0010;
		}
		else if (strcmp(tok, "attnums") == 0)
		{
			char   *__tok, *__save;
			int		count = 0;

			for (__tok = strtok_r(pos, " ", &__save);
				 __tok != NULL;
				 __tok = strtok_r(NULL, " ", &__save))
			{
				if (atoi(__tok) <= 0)
				{
					PyErr_Format(PyExc_ValueError,
								 "unexpected attribute number [%s]", __tok);
					return false;
				}
				count++;
			}
			if (count == 0)
			{
				PyErr_Format(PyExc_ValueError,
							 "no attribute numbers are defined");
			}
			nattrs = count;
			mask |= 0x0020;
		}
		else if (strcmp(tok, "table_oid") == 0)
		{
			/* just ignore the attributes */
			mask |= 0x0040;
		}
		else
		{
			PyErr_Format(PyExc_ValueError, "unexpected token [%s]", tok);
			return false;
		}
	}

	if ((mask & 0x001f) != 0x001f)
	{
		PyErr_Format(PyExc_ValueError,
					 "identifier token has no attributes of:%s%s%s%s%s%s",
					 (mask & 0x0001) != 0 ? " device_id" : "",
					 (mask & 0x0002) != 0 ? " byte_size" : "",
					 (mask & 0x0004) != 0 ? " ipc_handle" : "",
					 (mask & 0x0008) != 0 ? " format" : "",
					 (mask & 0x0010) != 0 ? " nitems" : "",
					 (mask & 0x0020) != 0 ? " nattrs" : "");
		return false;
	}

	if (bytesize < unitsz * nitems)
	{
		PyErr_Format(PyExc_ValueError,
					 "bytesize [%zu] is too small for %d x %ld items",
					 bytesize, nattrs, nitems);
		return false;
	}

	if (nitems % nattrs != 0)
	{
		PyErr_Format(PyExc_ValueError,
					 "nitems=%ld does not fit to nattrs=%d",
					 nitems, nattrs);
		return false;
	}

	if (p_device_id)
		*p_device_id = device_id;
	if (p_ipc_mhandle)
		memcpy(p_ipc_mhandle, &ipc_mhandle, sizeof(ipc_mhandle));
	if (p_bytesize)
		*p_bytesize = bytesize;
	if (p_type_code)
		*p_type_code = type_code;
	if (p_nattrs)
		*p_nattrs = nattrs;
	if (p_nitems)
		*p_nitems = nitems / nattrs;

	return true;
}

uintptr_t
cupy_strom__ipcmem_open(const char *ident,
						int *p_device_id,
						size_t *p_bytesize,
						char *p_type_code,
						int *p_width,
						long *p_height)
{
	int			device_id = -1;
	cudaIpcMemHandle_t ipc_mhandle;
	cudaError_t	rc;
	void	   *result = NULL;

	if (!ipcmem__parse_ident(ident,
							 &device_id,
							 &ipc_mhandle,
							 p_bytesize,
							 p_type_code,
							 p_width,
							 p_height))
		return 0UL;

	rc = cudaSetDevice(device_id);
	if (rc != cudaSuccess)
	{
		PyErr_Format(PyExc_SystemError,
					 "failed on cudaSetDevice: %s",
					 cudaGetErrorString(rc));
		return 0UL;
	}
	rc = cudaIpcOpenMemHandle(&result, ipc_mhandle,
							  cudaIpcMemLazyEnablePeerAccess);
	if (rc == cudaErrorAlreadyMapped)
	{

	}
	else if (rc != cudaSuccess)
	{
		PyErr_Format(PyExc_SystemError,
					 "failed on cudaIpcOpenMemHandle: %s",
					 cudaGetErrorString(rc));
		return 0UL;
	}
	
	if (p_device_id)
		*p_device_id = device_id;
	return (uintptr_t)result;
}

int
cupy_strom__ipcmem_close(uintptr_t device_ptr)
{
	cudaError_t	rc;

	rc = cudaIpcCloseMemHandle((void *)device_ptr);
	if (rc != cudaSuccess)
	{
		PyErr_Format(PyExc_SystemError,
					 "failed on cudaIpcCloseMemHandle: %s",
					 cudaGetErrorString(rc));
		return -1;
	}
	return 0;
}
