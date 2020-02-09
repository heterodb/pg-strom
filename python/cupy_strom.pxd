from libc.stdint cimport uintptr_t
cdef extern from "pystrom.h":
	uintptr_t cupy_strom__ipcmem_open(const char *ident,
									  int *p_device_id,
									  size_t *p_bytesize,
									  char *p_type_code,
									  int *p_width,
									  long *p_height)
	int cupy_strom__ipcmem_close(uintptr_t device_ptr)
