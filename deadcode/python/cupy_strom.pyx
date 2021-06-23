import cython
import numpy
import cupy
	
class IpcMemory(cupy.cuda.BaseMemory):
	def __init__(self):
		self.device_id = 0
		self.ptr = 0
		self.size = 0
		super().__init__()

	def open(self, str ident):
		cdef uintptr_t c_device_ptr
		cdef int	c_device_id
		cdef size_t	c_bytesize
		cdef char	c_type_code
		cdef int	c_nattrs
		cdef long	c_nitems
		
		if (self.ptr != 0):
			self.close()
		c_device_ptr = cupy_strom__ipcmem_open(ident.encode('utf-8'),
											   &c_device_id,
											   &c_bytesize,
											   &c_type_code,
											   &c_nattrs,
											   &c_nitems)
		if (c_device_ptr == 0):
			raise SystemError("failed on cupy_strom__ipcmem_open")
		self.device_id = c_device_id
		self.ptr = c_device_ptr
		self.size = c_bytesize
		self.cupy_type_code = chr(c_type_code)
		self.cupy_nattrs = c_nattrs
		self.cupy_nitems = c_nitems

	def close(self):
		if (self.ptr != 0):
			if (cupy_strom__ipcmem_close(self.ptr) != 0):
				raise SystemError("failed on cupy_strom__ipcmem_close")
			self.device_id = 0
			self.ptr = 0
			self.size = 0

	def __del__(self):
		self.close()

def ipc_import(str token):
	ipcMem = IpcMemory()
	ipcMem.open(token)

	return cupy.ndarray([ipcMem.cupy_nattrs,ipcMem.cupy_nitems],
						numpy.dtype(ipcMem.cupy_type_code),
						cupy.cuda.memory.MemoryPointer(ipcMem, 0),
						None,
						'C')
