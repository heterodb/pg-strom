import cython
import numpy
import cupy
	
class IpcMemory(cupy.cuda.BaseMemory):
	def __init__(self):
		self.device_id = 0
		self.ptr = 0
		self.size = 0
		self.is_opened = False
		super().__init__()

	def open(self, str ident):
		cdef uintptr_t c_device_ptr
		cdef int	c_device_id
		cdef size_t	c_bytesize
		cdef char	c_type_code
		cdef int	c_width
		cdef long	c_height
		
		if (self.is_opened):
			self.close()
		c_device_ptr = cupy_strom__ipcmem_open(ident.encode('utf-8'),
											   &c_device_id,
											   &c_bytesize,
											   &c_type_code,
											   &c_width,
											   &c_height)
		self.device_id = c_device_id
		self.ptr = c_device_ptr
		self.size = c_bytesize
		self.cupy_type_code = chr(c_type_code)
		self.cupy_width = c_width
		self.cupy_height = c_height
		self.is_opened = True

	def close(self):
		if (self.is_opened):
			cupy_strom__ipcmem_close(self.ptr)
			self.ptr = 0
			self.size = 0
			self.is_opened = False;

	def __del__(self):
		self.close()

def ipc_import(str token):
	ipcMem = IpcMemory()
	ipcMem.open(token)

	return cupy.ndarray([ipcMem.cupy_height,ipcMem.cupy_width],
						numpy.dtype(ipcMem.cupy_type_code),
						cupy.cuda.memory.MemoryPointer(ipcMem, 0),
						None,
						'F')
