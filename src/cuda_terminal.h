/*
 * cuda_terminal.h
 *
 * CUDA device code to be located on end of the code block
 * --
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#ifndef CUDA_TERMINAL_H
#define CUDA_TERMINAL_H

#ifdef CUDA_GPUSCAN_H
/*
 * If cuda_gpuscan.h is required by GpuJoin or GpuPreAgg, it does not define
 * gpuscan_projection of course. So, we need to have a stub in this case.
 */
#ifndef CUDA_GPUSCAN_HAS_PROJECTION
STATIC_FUNCTION(void)
gpuscan_projection(kern_context *kcxt,
                   kern_data_store *kds_src,
                   HeapTupleHeaderData *htup,
                   ItemPointerData *t_self,
                   Datum *tup_values,
                   cl_bool *tup_isnull,
                   cl_bool *tup_internal)
{
	STROM_SET_ERROR(&kcxt->e, StromError_WrongCodeGeneration);
	return;
}
#endif	/* CUDA_GPUSCAN_HAS_PROJECTION */
#endif	/* CUDA_GPUSCAN_H */

#ifndef CUDA_NUMERIC_H
/*
 * pg_numeric_to_varlena - has to be defined in cuda_numeric.h if NUMERIC
 * data type is required, however, kern_form_heaptuple need to call this
 * function to treat internal representation of NUMERIC type.
 * The function below is a dummy alternative not to be called.
 */
STATIC_FUNCTION(cl_uint)
pg_numeric_to_varlena(kern_context *kcxt, char *vl_buffer,
                      Datum value, cl_bool isnull)
{
	STROM_SET_ERROR(&kcxt->e, StromError_WrongCodeGeneration);
	return 0;
}
#endif

#endif	/* CUDA_TERMINAL_H */
