#include <Python.h>
#include <cuda_runtime_api.h>
#include "cuda_common.h"

static PyObject *
my_test(PyObject *self, PyObject *args)
{
	PyObject	   *cupy;
	Py_buffer		__ipc_handle;
	cudaIpcMemHandle_t ipc_handle;
	cudaError_t		rc;
	void		   *m_devptr;
	char			__buffer[KDS_LEAST_LENGTH];
	char		   *buffer = __buffer;
	size_t			length = KDS_LEAST_LENGTH;
	kern_data_store *kds;
	NameData	   *attNames;
	cl_uint			j;

	if(!PyArg_ParseTuple(args, "y*", &__ipc_handle))
		Py_RETURN_NONE;
	if (__ipc_handle.len != sizeof(cudaIpcMemHandle_t))
	{
		PyErr_Format(PyExc_ValueError, "IPC handle length mismatch: %d of %d",
					 __ipc_handle.len, sizeof(cudaIpcMemHandle_t));
		Py_RETURN_NONE;
	}
	memcpy(&ipc_handle, __ipc_handle.buf, __ipc_handle.len);

	rc = cudaIpcOpenMemHandle(&m_devptr, ipc_handle,
							  cudaIpcMemLazyEnablePeerAccess);
	if (rc != cudaSuccess)
	{
		PyErr_Format(PyExc_SystemError, "failed on cudaIpcOpenMemHandle: %s",
					 cudaGetErrorString(rc));
		Py_RETURN_NONE;
	}

memcpy_retry:
	rc = cudaMemcpy(buffer, m_devptr, length,
					cudaMemcpyDeviceToHost);
	if (rc != cudaSuccess)
	{
		PyErr_Format(PyExc_SystemError, "failed on cudaMemcpy: %s",
					 cudaGetErrorString(rc));
		goto error_1;
	}

	kds = (kern_data_store *)buffer;
	if (length < KERN_DATA_STORE_HEAD_LENGTH(kds))
	{
		length = KERN_DATA_STORE_HEAD_LENGTH(kds);
		buffer = malloc(length);
		if (!buffer)
		{
			PyErr_Format(PyExc_SystemError, "out of memory: %m");
			goto error_1;
		}
		goto memcpy_retry;
	}

	attNames = KERN_DATA_STORE_ATTNAMES(kds);
	printf("kds {length=%zu ncols=%d}\n", kds->length, kds->ncols);
	for (j=0; j < kds->ncols; j++)
	{
		kern_colmeta *cmeta = &kds->colmeta[j];

		printf("att %d (%s) {attbyval=%d attalign=%d attlen=%d attnum=%d atttypid=%u atttypmod=%d va_offset=%lu va_length=%lu}\n", j, attNames ? attNames[j].data : "(null)", cmeta->attbyval, cmeta->attalign, cmeta->attlen, cmeta->attnum, cmeta->atttypid, cmeta->atttypmod, __kds_unpack(cmeta->va_offset), __kds_unpack(cmeta->va_length));
	}

	/* release resources */
error_1:
	if (buffer != __buffer)
		free(buffer);

	rc = cudaIpcCloseMemHandle(m_devptr);
	if (rc != cudaSuccess)
	{
		PyErr_Format(PyExc_SystemError, "failed on cudaIpcCloseMemHandle: %s",
					 cudaGetErrorString(rc));
	}
	Py_RETURN_NONE;





#if 0
	sleep(10);

	cupy = PyImport_ImportModule("cupy");
	if (!cupy)
		Py_RETURN_NONE;

	printf("end of pystrom.my_test\n");
	Py_DECREF(cupy);

	Py_RETURN_NONE;
#endif
}

static PyMethodDef pystrom_methods[] = {
	{"my_test", (PyCFunction)my_test, METH_VARARGS, "my_test of pystrom"},
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
