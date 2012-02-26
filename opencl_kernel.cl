/*
 * opencl_kernel.cl
 *
 * GPU kernel routine based on OpenCL
 * --
 * Copyright 2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "opencl_catalog.h"

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define VAR_IS_NULL(attno)												\
	((cs_rowmap[cs_isnull[(attno) + get_global_id(0)]] & bitmask) != 0)
#define VAR_REF(type, attno)											\
	*((type *)(&cs_rowmap[cs_values[(attno)] + sizeof(type) * index]))
#define REG_REF(type, regidx)					\
	*((type *)(&regs[(regidx)]))

__kernel void
opencl_qual(__constant int    commands[],
			__constant int    cs_isnull[],
			__constant int    cs_values[],
			__global   uchar *cs_rowmap)
{
	__constant int *cmd;
	int		index	= 8 * get_global_id(0);
	int		bitmask;
	uchar	result;
	uchar	errors;
	uint	regs[PGSTROM_GPU_NUM_REGS];

	/*
	 * commands[0] is used for deliver number of items
	 */
	if (index >= commands[0])
		return;

	result = cs_rowmap[get_global_id(0)];
	errors = 0;
	for (bitmask = 1; bitmask < 256; bitmask <<= 1, index++)
	{
		/*
		 * In case of removed tuple
		 */
		if ((result & bitmask) != 0)
			continue;

		cmd = &commands[1];
		while (*cmd != GPUCMD_TERMINAL_COMMAND)
		{
			switch (*cmd)
			{
				/*
				 * Reference to constant values
				 */
				case GPUCMD_CONREF_NULL:
					regs[*(cmd+1)] = 0;
					cmd += 2;
					break;

				case GPUCMD_CONREF_BOOL:
				case GPUCMD_CONREF_INT2:
				case GPUCMD_CONREF_INT4:
				case GPUCMD_CONREF_FLOAT4:
					/* 32bit constant value */
					regs[*(cmd+1)] = *(cmd+2);
					cmd += 3;
					break;

				case GPUCMD_CONREF_INT8:
				case GPUCMD_CONREF_FLOAT8:
					/* 64bit constant value */
					regs[*(cmd+1)    ] = *(cmd+2);
					regs[*(cmd+1) + 1] = *(cmd+3);
					cmd += 4;
					break;

				/*
				 * Reference to variables
				 */
				case GPUCMD_VARREF_BOOL:
					/* reference to 8bits-variable */
					if (VAR_IS_NULL(*(cmd+2)))
						errors |= bitmask;
					else
						regs[*(cmd+1)] = VAR_REF(uchar, *(cmd+2));
					cmd += 3;
					break;

				case GPUCMD_VARREF_INT2:
					/* reference to 16bits-variable */
					if (VAR_IS_NULL(*(cmd+2)))
						errors |= bitmask;
					else
						regs[*(cmd+1)] = VAR_REF(ushort, *(cmd+2));
					cmd += 3;
					break;

				case GPUCMD_VARREF_INT4:
				case GPUCMD_VARREF_FLOAT4:
					/* reference to 32bits-variable */
					if (VAR_IS_NULL(*(cmd+2)))
						errors |= bitmask;
					else
						regs[*(cmd+1)] = VAR_REF(uint, *(cmd+2));
					cmd += 3;
					break;

				case GPUCMD_VARREF_INT8:
				case GPUCMD_VARREF_FLOAT8:
					/* reference to 64bits-variable */
					if (VAR_IS_NULL(*(cmd+2)))
						errors |= bitmask;
					else
						REG_REF(ulong, *(cmd+1)) = VAR_REF(ulong, *(cmd+2));
					cmd += 3;
					break;

#if 0
				/*
				 * Cast of Data Types
				 */
				case GPUCMD_CAST_INT2_TO_INT4:
				case GPUCMD_CAST_INT2_TO_INT8:
				case GPUCMD_CAST_INT2_TO_FLOAT4:
				case GPUCMD_CAST_INT2_TO_FLOAT8:
				case GPUCMD_CAST_INT4_TO_INT2:
				case GPUCMD_CAST_INT4_TO_INT8:
				case GPUCMD_CAST_INT4_TO_FLOAT4:
				case GPUCMD_CAST_INT4_TO_FLOAT8:
				case GPUCMD_CAST_INT8_TO_INT2:
				case GPUCMD_CAST_INT8_TO_INT4:
				case GPUCMD_CAST_INT8_TO_FLOAT4:
				case GPUCMD_CAST_INT8_TO_FLOAT8:
				case GPUCMD_CAST_FLOAT4_TO_INT2:
				case GPUCMD_CAST_FLOAT4_TO_INT4:
				case GPUCMD_CAST_FLOAT4_TO_INT8:
				case GPUCMD_CAST_FLOAT4_TO_FLOAT8:
				case GPUCMD_CAST_FLOAT8_TO_INT2:
				case GPUCMD_CAST_FLOAT8_TO_INT4:
				case GPUCMD_CAST_FLOAT8_TO_INT8:
				case GPUCMD_CAST_FLOAT8_TO_FLOAT4:
					break;
#endif
				/*
				 * Boolean operations
				 */
				case GPUCMD_BOOLOP_AND:
					regs[*(cmd+1)] = regs[*(cmd+1)] & regs[*(cmd+2)];
					cmd += 3;
					break;

				case GPUCMD_BOOLOP_OR:
					regs[*(cmd+1)] = regs[*(cmd+1)] | regs[*(cmd+2)];
					cmd += 3;
					break;

				case GPUCMD_BOOLOP_NOT:
					regs[*(cmd+1)] = !regs[*(cmd+1)];
					cmd += 2;
					break;

				case GPUCMD_OPER_FLOAT8_LT:
					regs[*(cmd+1)] = (int)(REG_REF(double, *(cmd+2)) <
										   REG_REF(double, *(cmd+3)));
					cmd += 4;
					break;

				case GPUCMD_OPER_FLOAT8_MI:
					REG_REF(double, *(cmd+1))
						= (REG_REF(double, *(cmd+2)) -
						   REG_REF(double, *(cmd+3)));
					cmd += 4;
					break;

				case GPUCMD_OPER_FLOAT8_PL:
					REG_REF(double, *(cmd+1))
						= (REG_REF(double, *(cmd+2)) -
						   REG_REF(double, *(cmd+3)));
					cmd += 4;
					break;

				case GPUCMD_FUNC_POWER:
					REG_REF(double, *(cmd+1))
						= pow(REG_REF(double, *(cmd+2)),
							  REG_REF(double, *(cmd+3)));
					cmd += 4;
					break;

				default:
					errors |= bitmask;
					break;
			}
		}
		/*
		 * 
		 */
		if (regs[0] == 0)
			result |= bitmask;
	}
	cs_rowmap[get_global_id(0)] = (result | errors);
}
