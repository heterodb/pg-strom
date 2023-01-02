/*
 * xpu_textlib.h
 *
 * Misc definitions for text routines for both of GPU and DPU
 * --
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_TEXTLIB_H
#define XPU_TEXTLIB_H

#ifndef PG_BYTEAOID
#define PG_BYTEAOID		17
#endif
#ifndef PG_TEXTOID
#define PG_TEXTOID		25
#endif
PGSTROM_SQLTYPE_VARLENA_DECLARATION(bytea);
PGSTROM_SQLTYPE_VARLENA_DECLARATION(text);
PGSTROM_SQLTYPE_VARLENA_DECLARATION(bpchar);

/*
 * Database Encoding Info
 */
struct xpu_encode_info {
	char	encname[16];
	int		enc_maxlen;
	int	  (*enc_mblen)(const char *s);
};
typedef struct xpu_encode_info	xpu_encode_info;

EXTERN_DATA xpu_encode_info		xpu_encode_catalog[];


#endif  /* XPU_TEXTLIB_H */
