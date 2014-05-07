/*
 * opencl_textlib.h
 *
 * Collection of text functions for OpenCL devices
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef OPENCL_TEXTLIB_H
#define OPENCL_TEXTLIB_H
#ifdef OPENCL_DEVICE_CODE

static int
varlena_cmp(__global varlena *s1, cl_uint len1,
			__global varlena *s2, cl_uint len2)
{}






#ifdef PG_TYPE_BPCHAR
static pg_bool
pgfn_bpchareq(pg_bpchar arg1, pg_bpchar arg2)
{}

static pg_bool
pgfn_bpcharne(pg_bpchar arg1, pg_bpchar arg2)
{}

static pg_bool
pgfn_bpcharlt(pg_bpchar arg1, pg_bpchar arg2)
{}

static pg_bool
pgfn_bpcharle(pg_bpchar arg1, pg_bpchar arg2)
{}

static pg_bool
pgfn_bpchargt(pg_bpchar arg1, pg_bpchar arg2)
{}

static pg_bool
pgfn_bpcharge(pg_bpchar arg1, pg_bpchar arg2)
{}
#endif	/* PGTYPE_BPCHAR */


#ifdef PG_TYPE_TEXT
static pg_bool
pgfn_texteq(pg_text arg1, pg_text arg2)
{}

static pg_bool
pgfn_textne(pg_text arg1, pg_text arg2)
{}

static pg_bool
pgfn_textlt(pg_text arg1, pg_text arg2)
{}

static pg_bool
pgfn_textle(pg_text arg1, pg_text arg2)
{}

static pg_bool
pgfn_textgt(pg_text arg1, pg_text arg2)
{}

static pg_bool
pgfn_textge(pg_text arg1, pg_text arg2)
{}
#endif	/* PG_TYPE_TEXT */

#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_TEXTLIB_H */
