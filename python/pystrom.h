#include <stdint.h>

#ifndef PYSTROM_H
#define PYSTROM_H
/* boolean */
#ifndef bool
typedef char bool;
#endif
#ifndef true
#define true	((bool) 1)
#endif
#ifndef false
#define false	((bool) 0)
#endif

extern uintptr_t cupy_strom__ipcmem_open(const char *ident,
										 int *p_device_id,
										 size_t *p_bytesize,
										 char *p_type_code,
										 int *p_width,
										 long *p_height);
extern void		cupy_strom__ipcmem_close(uintptr_t device_ptr);

#endif	/* PYSTROM_H */
