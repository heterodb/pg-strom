#include <Python.h>
#include <cuda_runtime_api.h>
#include "cuda_common.h"

/* supported data types */
#define BOOLOID 16
#define INT2OID 21
#define INT4OID 23
#define INT8OID 20
#define FLOAT4OID 700
#define FLOAT8OID 701

static PyObject *
create_ndarray_normal(cl_uint type_oid, cl_ulong nitems, cl_ulong nattrs)
{
	PyObject   *mod_cupy = NULL;
	PyObject   *mod_core = NULL;
	PyObject   *mod_dtype = NULL;
	PyObject   *dtype = NULL;
	PyObject   *result = NULL;
	const char *type_code;

	switch (type_oid)
	{
		case BOOLOID:
			type_code = "?";
			break;
		case INT2OID:
			type_code = "h";
			break;
		case INT4OID:
			type_code = "i";
			break;
		case INT8OID:
			type_code = "l";
			break;
		case FLOAT4OID:
			type_code = "f";
			break;
		case FLOAT8OID:
			type_code = "d";
			break;
		default:
			PyErr_Format(PyExc_SystemError, "unknown element data type");
			return NULL;
	}

	mod_cupy = PyImport_ImportModule("cupy");
	if (!mod_cupy)
	{
		PyErr_Format(PyExc_SystemError, "could not import 'cupy' module");
		goto bailout;
	}
	mod_core = PyObject_GetAttrString(mod_cupy, "core");
	if (!mod_core)
	{
		PyErr_Format(PyExc_SystemError, "cupy.core was not found");
		goto bailout;
	}
	mod_dtype = PyObject_GetAttrString(mod_core, "_dtype");
	if (!mod_dtype)
	{
		PyErr_Format(PyExc_SystemError,
					 "cupy.core._dtype was not found");
		goto bailout;
	}
	dtype = PyObject_CallMethod(mod_dtype, "get_dtype",
								"(s)", type_code);
	if (!dtype)
		goto bailout;

	result = PyObject_CallMethod(mod_core, "ndarray",
								 "(k k) O O s",
								 nitems, nattrs,
								 dtype,
								 Py_None,
								 "F");
bailout:
	if (dtype)
		Py_DECREF(dtype);
	if (mod_core)
		Py_DECREF(mod_core);
	if (mod_cupy)
		Py_DECREF(mod_cupy);
	return result;
}

static int
setup_ndarray_from_kds(PyObject *ndarray,
					   const char *m_kds,		/* device pointer */
					   kern_data_store *kds,	/* host buffer */
					   cl_uint type_oid, cl_int nattrs, cl_int *attIndex)
{
	cl_long		nitems = kds->nitems;
	cl_long		ndarray_sz;
	PyObject   *devmem = NULL;
	PyObject   *devptr = NULL;
	PyObject   *temp;
	char	   *dest;
	size_t		type_len;
	int			i, retval = 1;

	/* check size of ndarray */
	temp = PyObject_GetAttrString(ndarray, "size");
	if (!temp)
	{
		PyErr_Format(PyExc_SystemError, "'size' not found in ndarray");
		goto bailout;
	}
	ndarray_sz = PyLong_AsLong(temp);
	Py_DECREF(temp);
	if (ndarray_sz != nattrs * nitems)
	{
		PyErr_Format(PyExc_SystemError, "Wrong ndarray size");
		goto bailout;
	}
	/* get device memory pointer */
	devmem = PyObject_GetAttrString(ndarray, "data");
	if (!devmem)
	{
		PyErr_Format(PyExc_SystemError, "'data' not found in ndarray");
		goto bailout;
	}
	devptr = PyObject_GetAttrString(devmem, "ptr");
	if (!devptr)
	{
		PyErr_Format(PyExc_SystemError, "'devptr' not found in MemoryPointer");
		goto bailout;
	}
	dest = (char *)PyLong_AsLong(devptr);

	/* copy from Gstore_fdw to ndarray buffer */
	switch (type_oid)
	{
		case BOOLOID:
			type_len = sizeof(cl_bool);
			break;
		case INT2OID:
			type_len = sizeof(cl_short);
			break;
		case INT4OID:
			type_len = sizeof(cl_int);
			break;
		case INT8OID:
			type_len = sizeof(cl_long);
			break;
		case FLOAT4OID:
			type_len = sizeof(cl_float);
			break;
		case FLOAT8OID:
			type_len = sizeof(cl_double);
			break;
		default:
			PyErr_Format(PyExc_ValueError, "unknown data type");
			goto bailout;
	}

	for (i=0; i < nattrs; i++)
	{
		kern_colmeta   *cmeta = &kds->colmeta[attIndex[i]];
		size_t			offset;
		size_t			length;
		size_t			nbytes;
		cudaError_t		rc;

		assert(cmeta->atttypid == type_oid &&
			   cmeta->attlen == type_len);
		offset = __kds_unpack(cmeta->va_offset);
		length = __kds_unpack(cmeta->va_length);
		nbytes = type_len * nitems;
		if (length < nbytes)
		{
			cudaStreamSynchronize(0);
			PyErr_Format(PyExc_SystemError,
						 "Bug? values array of Gstore_fdw is too short");
			goto bailout;
		}
		rc = cudaMemcpyAsync(dest, m_kds + offset, nbytes,
							 cudaMemcpyDeviceToDevice, 0);
		if (rc != cudaSuccess)
		{
			cudaStreamSynchronize(0);
			PyErr_Format(PyExc_SystemError,
						 "failed on cudaMemcpyAsync: %s",
						 cudaGetErrorString(rc));
			goto bailout;
		}
		dest += nbytes;
	}
	cudaStreamSynchronize(0);
	retval = 0;
bailout:
	if (devptr)
		Py_DECREF(devptr);
	if (devmem)
		Py_DECREF(devmem);
	return retval;
}

static int
check_ndarray_column(kern_colmeta *cmeta, size_t nitems,
					 NameData *cname, cl_uint *p_type_oid)
{
	cl_uint		type_oid = *p_type_oid;
	size_t		va_offset = __kds_unpack(cmeta->va_offset);
	size_t		va_length = __kds_unpack(cmeta->va_length);
	size_t		dataLen;

	/* check data type consistency */
	if (type_oid == 0)
		type_oid = cmeta->atttypid;
	else if (type_oid != cmeta->atttypid)
	{
		PyErr_Format(PyExc_ValueError, "Data types are not consistent");
		return 1;
	}
	if (type_oid != BOOLOID &&
		type_oid != INT2OID &&
		type_oid != INT4OID &&
		type_oid != INT8OID &&
		type_oid != FLOAT4OID &&
		type_oid != FLOAT8OID)
	{
		PyErr_Format(PyExc_ValueError,
					 "Data type (oid: %u) is not supported", type_oid);
		return 1;
	}
	*p_type_oid = type_oid;

	/* other sanity checks */
	if (va_offset == 0)
	{
		PyErr_Format(PyExc_ValueError,
					 "column '%s' contains no data", cname->data);
		return 1;
	}

	if (!cmeta->attbyval ||
		cmeta->attlen <= 0 ||
		cmeta->attnum <= 0)
	{
		PyErr_Format(PyExc_ValueError,
					 "column '%s' is not fixed length numeric values",
					 cname->data);
		return 1;
	}

	dataLen = MAXALIGN(cmeta->attlen * nitems);
	if (va_length < dataLen)
	{
		PyErr_Format(PyExc_ValueError,
					 "Bug? column '%s' is shorter than expected length",
					 cname->data);
		return 1;
	}
	if (va_length > dataLen)
	{
		PyErr_Format(PyExc_ValueError,
					 "column '%s' contains NULLs",
					 cname->data);
		return 1;
	}
	return 0;
}


static PyObject *
pystrom_ipc_import(PyObject *self, PyObject *args)
{
	Py_buffer		ipc_token;
	gstoreIpcHandle gs_handle;
	cudaIpcMemHandle_t ipc_handle;
	PyObject	   *attnameList = NULL;
	cudaError_t		rc;
	void		   *m_devptr;
	char			__buffer[KDS_LEAST_LENGTH];
	char		   *buffer = __buffer;
	size_t			length = KDS_LEAST_LENGTH;
	kern_data_store *kds;
	NameData	   *attNames;
	cl_int		   *attIndex = NULL;
	cl_uint			i, j, nattrs = 0;
	cl_uint			type_oid = 0;
	PyObject	   *ndarray = NULL;

	if(!PyArg_ParseTuple(args, "y*|O!",
						 &ipc_token,
						 &PyList_Type,
						 &attnameList))
		Py_RETURN_NONE;

	/*
	 * Import GPU device memory using IPC memory handle
	 */
	if (ipc_token.len != sizeof(gstoreIpcHandle))
	{
		PyErr_Format(PyExc_ValueError,
					 "IPC token length mismatch: %d of %d",
					 ipc_token.len, sizeof(gstoreIpcHandle));
		Py_RETURN_NONE;
	}
	memcpy(&gs_handle, ipc_token.buf, ipc_token.len);
	if (VARSIZE(&gs_handle) != sizeof(gs_handle) ||
		gs_handle.magic != GSTORE_IPC_HANDLE_MAGIC)
	{
		PyErr_Format(PyExc_ValueError,
					 "IPC token corruption (vl_len: %d, magic: %08x)",
					 VARSIZE(&gs_handle), gs_handle.magic);
		Py_RETURN_NONE;
	}
	memcpy(&ipc_handle, gs_handle.ipc_handle, sizeof(cudaIpcMemHandle_t));
	rc = cudaIpcOpenMemHandle(&m_devptr, ipc_handle,
							  cudaIpcMemLazyEnablePeerAccess);
	if (rc != cudaSuccess)
	{
		PyErr_Format(PyExc_SystemError,
					 "Failed on cudaIpcOpenMemHandle: %s",
					 cudaGetErrorString(rc));
		Py_RETURN_NONE;
	}

memcpy_retry:
	rc = cudaMemcpy(buffer, m_devptr, length,
					cudaMemcpyDeviceToHost);
	if (rc != cudaSuccess)
	{
		PyErr_Format(PyExc_SystemError,
					 "Failed on cudaMemcpy: %s",
					 cudaGetErrorString(rc));
		goto bailout;
	}

	kds = (kern_data_store *)buffer;
	if (length < KERN_DATA_STORE_HEAD_LENGTH(kds))
	{
		length = KERN_DATA_STORE_HEAD_LENGTH(kds);
		buffer = malloc(length);
		if (!buffer)
		{
			PyErr_Format(PyExc_SystemError, "Out of memory: %m");
			goto bailout;
		}
		goto memcpy_retry;
	}

	attNames = KERN_DATA_STORE_ATTNAMES(kds);
	if (!attNames)
	{
		PyErr_Format(PyExc_SystemError,
					 "Bug? Gstore_fdw has no attribute names");
		goto bailout;
	}

	if (attnameList)
	{
		nattrs = PyList_Size(attnameList);
		attIndex = calloc(nattrs, sizeof(cl_int));
		if (!attIndex)
		{
			PyErr_Format(PyExc_SystemError, "Out of memory: %m");
			goto bailout;
		}
		for (i=0; i < nattrs; i++)
		{
			PyObject	   *aname = PyList_GetItem(attnameList, i);
			kern_colmeta   *cmeta = NULL;

			for (j=0; j < kds->ncols; j++)
			{
				PyObject   *cname = PyUnicode_FromString(attNames[j].data);
				if (PyUnicode_Compare(aname, cname) == 0)
				{
					cmeta = &kds->colmeta[j];
					Py_DecRef(cname);
					break;
				}
				Py_DecRef(cname);
			}
			if (!cmeta)
			{
				PyErr_Format(PyExc_ValueError,
							 "Specified column '%s' was not found",
							 PyUnicode_AsUTF8(aname));
				goto bailout;
			}
			if (check_ndarray_column(cmeta, kds->nitems,
									 &attNames[j], &type_oid))
				goto bailout;

			attIndex[i] = j;
		}
	}
	else
	{
		attIndex = calloc(kds->ncols, sizeof(cl_int));
		if (!attIndex)
		{
			PyErr_Format(PyExc_SystemError, "Out of memory: %m");
			goto bailout;
		}
		for (i=0, j=0; j < kds->ncols; j++)
		{
			kern_colmeta   *cmeta = &kds->colmeta[j];

			/* skip system column */
			if (cmeta->attnum < 0)
				continue;

			if (check_ndarray_column(cmeta, kds->nitems,
									 &attNames[j], &type_oid))
				goto bailout;

			attIndex[i++] = j;
		}
		nattrs = i;
	}
	/* creation of ndarray */
	ndarray = create_ndarray_normal(type_oid, kds->nitems, nattrs);
	if (!ndarray)
		goto bailout;
	/* copy from the Gstore_fdw */
	if (setup_ndarray_from_kds(ndarray, m_devptr,
							   kds, type_oid, nattrs, attIndex))
	{
		Py_DECREF(ndarray);
		ndarray = NULL;
		goto bailout;
	}

	/* ok, release temporary resources */
bailout:
	if (attIndex)
		free(attIndex);
	if (buffer != __buffer)
		free(buffer);

	rc = cudaIpcCloseMemHandle(m_devptr);
	if (rc != cudaSuccess)
	{
		PyErr_Format(PyExc_SystemError, "failed on cudaIpcCloseMemHandle: %s",
					 cudaGetErrorString(rc));
	}
	if (!ndarray)
		Py_RETURN_NONE;
	return ndarray;
}

static PyMethodDef pystrom_methods[] = {
	{"ipc_import", (PyCFunction)pystrom_ipc_import, METH_VARARGS, "Import Gstore_fdw as cupy.core.ndarray"},
	{NULL, NULL, 0, NULL},
};

static struct PyModuleDef pystrom_module = {
	PyModuleDef_HEAD_INIT,
	"pystrom",
	NULL,
	-1,
	pystrom_methods,
};


PyMODINIT_FUNC PyInit_pystrom(void)
{
	return PyModule_Create(&pystrom_module);
}
