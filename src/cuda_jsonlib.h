/*
 * cuda_jsonlib.h
 *
 * Collection of text functions for CUDA GPU devices
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"

#ifndef CUDA_JSONLIB_H
#define CUDA_JSONLIB_H

#ifndef PG_JSONB_TYPE_DEFINED
#define PG_JSONB_TYPE_DEFINED
STROMCL_VARLENA_DATATYPE_TEMPLATE(jsonb)
STROMCL_VARLENA_VARREF_TEMPLATE(jsonb)
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(jsonb)
STROMCL_UNSUPPORTED_ARROW_TEMPLATE(jsonb)
#endif	/* PG_JSONB_TYPE_DEFINED */

#ifdef __CUDACC__
/* jsonb operator functions  */
DEVICE_FUNCTION(pg_jsonb_t)
pgfn_jsonb_object_field(kern_context *kcxt,
						pg_jsonb_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_jsonb_t)
pgfn_jsonb_array_element(kern_context *kcxt,
						 pg_jsonb_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_text_t)
pgfn_jsonb_object_field_text(kern_context *kcxt,
							 pg_jsonb_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_text_t)
pgfn_jsonb_array_element_text(kern_context *kcxt,
							  pg_jsonb_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_jsonb_exists(kern_context *kcxt,
				  pg_jsonb_t arg1, pg_text_t arg2);
/* special shortcut for CoerceViaIO; fetch jsonb element as numeric values */
DEVICE_FUNCTION(pg_numeric_t)
pgfn_jsonb_object_field_as_numeric(kern_context *kcxt,
								   pg_jsonb_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_int2_t)
pgfn_jsonb_object_field_as_int2(kern_context *kcxt,
								pg_jsonb_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_jsonb_object_field_as_int4(kern_context *kcxt,
								pg_jsonb_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_int8_t)
pgfn_jsonb_object_field_as_int8(kern_context *kcxt,
								pg_jsonb_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_float4_t)
pgfn_jsonb_object_field_as_float4(kern_context *kcxt,
								  pg_jsonb_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_float8_t)
pgfn_jsonb_object_field_as_float8(kern_context *kcxt,
								  pg_jsonb_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_jsonb_array_element_as_numeric(kern_context *kcxt,
									pg_jsonb_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_int2_t)
pgfn_jsonb_array_element_as_int2(kern_context *kcxt,
								 pg_jsonb_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_jsonb_array_element_as_int4(kern_context *kcxt,
								 pg_jsonb_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_int8_t)
pgfn_jsonb_array_element_as_int8(kern_context *kcxt,
								 pg_jsonb_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_float4_t)
pgfn_jsonb_array_element_as_float4(kern_context *kcxt,
								   pg_jsonb_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_float8_t)
pgfn_jsonb_array_element_as_float8(kern_context *kcxt,
								   pg_jsonb_t arg1, pg_int4_t arg2);
#endif	/* __CUDACC__ */
#endif	/* CUDA_JSONLIB_H */
