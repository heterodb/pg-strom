/*
 * xpu_basetype.c
 *
 * Collection of primitive Int/Float type support on XPU(GPU/DPU/SPU)
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 *
 */
#include "xpu_common.h"

PGSTROM_SIMPLE_BASETYPE_TEMPLATE(bool, int8_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(int1, int8_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(int2, int16_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(int4, int32_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(int8, int64_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(float2, float2_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(float4, float4_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(float8, float8_t);
