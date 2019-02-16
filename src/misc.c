/*
 * misc.c
 *
 * miscellaneous and uncategorized routines but usefull for multiple subsystems
 * of PG-Strom.
 * ----
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
#include "pg_strom.h"

/*
 * make_flat_ands_expr - similar to make_ands_explicit but it pulls up
 * underlying and-clause
 */
Expr *
make_flat_ands_explicit(List *andclauses)
{
	List	   *args = NIL;
	ListCell   *lc;

	if (andclauses == NIL)
		return (Expr *) makeBoolConst(true, false);
	else if (list_length(andclauses) == 1)
		return (Expr *) linitial(andclauses);

	foreach (lc, andclauses)
	{
		Expr   *expr = lfirst(lc);

		Assert(exprType((Node *)expr) == BOOLOID);
		if (IsA(expr, BoolExpr) &&
			((BoolExpr *)expr)->boolop == AND_EXPR)
			args = list_concat(args, ((BoolExpr *) expr)->args);
		else
			args = lappend(args, expr);
	}
	Assert(list_length(args) > 1);
	return make_andclause(args);
}

#if PG_VERSION_NUM < 100000
/*
 * compute_parallel_worker at optimizer/path/allpaths.c
 * was newly added at PG10.x
 *
 * Compute the number of parallel workers that should be used to scan a
 * relation.  We compute the parallel workers based on the size of the heap to
 * be scanned and the size of the index to be scanned, then choose a minimum
 * of those.
 *
 * "heap_pages" is the number of pages from the table that we expect to scan,
 *  or -1 if we don't expect to scan any.
 *
 * "index_pages" is the number of pages from the index that we expect to scan,
 *  or -1 if we don't expect to scan any.
 */
int
compute_parallel_worker(RelOptInfo *rel, double heap_pages, double index_pages)
{
	int			parallel_workers = 0;

	/*
	 * If the user has set the parallel_workers reloption, use that; otherwise
	 * select a default number of workers.
	 */
	if (rel->rel_parallel_workers != -1)
		parallel_workers = rel->rel_parallel_workers;
	else
	{
		/*
		 * If the number of pages being scanned is insufficient to justify a
		 * parallel scan, just return zero ... unless it's an inheritance
		 * child. In that case, we want to generate a parallel path here
		 * anyway.  It might not be worthwhile just for this relation, but
		 * when combined with all of its inheritance siblings it may well pay
		 * off.
		 */
		if (rel->reloptkind == RELOPT_BASEREL &&
			(heap_pages >= 0 && heap_pages < min_parallel_relation_size))
			return 0;

		if (heap_pages >= 0)
		{
			int			heap_parallel_threshold;
			int			heap_parallel_workers = 1;

			/*
			 * Select the number of workers based on the log of the size of
			 * the relation.  This probably needs to be a good deal more
			 * sophisticated, but we need something here for now.  Note that
			 * the upper limit of the min_parallel_relation_size GUC is
			 * chosen to prevent overflow here.
			 */
			heap_parallel_threshold = Max(min_parallel_relation_size, 1);
			while (heap_pages >= (BlockNumber) (heap_parallel_threshold * 3))
			{
				heap_parallel_workers++;
				heap_parallel_threshold *= 3;
				if (heap_parallel_threshold > INT_MAX / 3)
					break;		/* avoid overflow */
			}

			parallel_workers = heap_parallel_workers;
		}
		/*
		 * NOTE: PG9.6 does not pay attention for # of index pages
		 * for decision of parallel execution.
		 */
	}

	/*
	 * In no case use more than max_parallel_workers_per_gather workers.
	 */
	parallel_workers = Min(parallel_workers, max_parallel_workers_per_gather);

	return parallel_workers;
}
#endif		/* < PG10 */

#if PG_VERSION_NUM < 110000
char
get_func_prokind(Oid funcid)
{
	HeapTuple	tup;
	Form_pg_proc procForm;
	char		prokind;

	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(funcid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for function %u", funcid);
	procForm = (Form_pg_proc) GETSTRUCT(tup);
	if (procForm->proisagg)
	{
		Assert(!procForm->proiswindow);
		prokind = PROKIND_AGGREGATE;
	}
	else if (procForm->proiswindow)
	{
		Assert(!procForm->proisagg);
		prokind = PROKIND_WINDOW;
	}
	else
	{
		prokind = PROKIND_FUNCTION;
	}
	ReleaseSysCache(tup);

	return prokind;
}
#endif		/* <PG11 */

/*
 * errorText - string form of the error code
 */
const char *
errorText(int errcode)
{
	static __thread char buffer[800];
	const char	   *label;

	switch (errcode)
	{
		case StromError_Success:
			label = "Suceess";
			break;
		case StromError_CpuReCheck:
			label = "CPU ReCheck";
			break;
		case StromError_InvalidValue:
			label = "Invalid Value";
			break;
		case StromError_DataStoreNoSpace:
			label = "Data store no space";
			break;
		case StromError_WrongCodeGeneration:
			label = "Wrong code generation";
			break;
		case StromError_OutOfMemory:
			label = "Out of Memory";
			break;
		case StromError_DataCorruption:
			label = "Data corruption";
			break;
		default:
			if (errcode <= CUDA_ERROR_UNKNOWN)
			{
				const char *error_val;
				const char *error_str;

				/* Likely CUDA driver error */
				if (cuGetErrorName(errcode, &error_val) == CUDA_SUCCESS &&
					cuGetErrorString(errcode, &error_str) == CUDA_SUCCESS)
					snprintf(buffer, sizeof(buffer), "%s - %s",
							 error_val, error_str);
				else
					snprintf(buffer, sizeof(buffer), "%d - unknown",
							 errcode);
			}
			else
			{
				/* ??? Unknown PG-Strom error??? */
				snprintf(buffer, sizeof(buffer),
						 "Unexpected Error: %d",
						 errcode);
			}
			return buffer;
	}
	return label;
}

/*
 * errorTextKernel - string form of the kern_errorbuf
 */
const char *
errorTextKernel(kern_errorbuf *kerror)
{
	static __thread char buffer[1024];
	const char *kernel_name;

#define KERN_ENTRY(KERNEL)						\
	case StromKernel_##KERNEL: kernel_name = #KERNEL; break

	switch (kerror->kernel)
	{
		KERN_ENTRY(HostPGStrom);
		KERN_ENTRY(CudaRuntime);
		KERN_ENTRY(NVMeStrom);
		KERN_ENTRY(gpuscan_main_row);
		KERN_ENTRY(gpuscan_main_block);
		KERN_ENTRY(gpuscan_main_column);
		KERN_ENTRY(gpujoin_main);
		KERN_ENTRY(gpujoin_right_outer);
		KERN_ENTRY(gpupreagg_setup_row);
		KERN_ENTRY(gpupreagg_setup_block);
		KERN_ENTRY(gpupreagg_setup_column);
		KERN_ENTRY(gpupreagg_nogroup_reduction);
		KERN_ENTRY(gpupreagg_groupby_reduction);
		KERN_ENTRY(gpusort_setup_column);
		KERN_ENTRY(gpusort_bitonic_local);
		KERN_ENTRY(gpusort_bitonic_step);
		KERN_ENTRY(gpusort_bitonic_merge);
		default:
			kernel_name = "unknown kernel";
			break;
	}
#undef KERN_ENTRY
	snprintf(buffer, sizeof(buffer), "%s at %s:%d by %s",
			 errorText(kerror->errcode),
			 kerror->filename, kerror->lineno, kernel_name);
	return buffer;
}

/*
 * ----------------------------------------------------------------
 *
 * SQL functions to support regression test
 *
 * ----------------------------------------------------------------
 */
Datum pgstrom_random_int(PG_FUNCTION_ARGS);
Datum pgstrom_random_float(PG_FUNCTION_ARGS);
Datum pgstrom_random_date(PG_FUNCTION_ARGS);
Datum pgstrom_random_time(PG_FUNCTION_ARGS);
Datum pgstrom_random_timetz(PG_FUNCTION_ARGS);
Datum pgstrom_random_timestamp(PG_FUNCTION_ARGS);
Datum pgstrom_random_timestamptz(PG_FUNCTION_ARGS);
Datum pgstrom_random_interval(PG_FUNCTION_ARGS);
Datum pgstrom_random_macaddr(PG_FUNCTION_ARGS);
Datum pgstrom_random_inet(PG_FUNCTION_ARGS);
Datum pgstrom_random_text(PG_FUNCTION_ARGS);
Datum pgstrom_random_text_length(PG_FUNCTION_ARGS);
Datum pgstrom_random_int4range(PG_FUNCTION_ARGS);
Datum pgstrom_random_int8range(PG_FUNCTION_ARGS);
Datum pgstrom_random_tsrange(PG_FUNCTION_ARGS);
Datum pgstrom_random_tstzrange(PG_FUNCTION_ARGS);
Datum pgstrom_random_daterange(PG_FUNCTION_ARGS);

static inline bool
generate_null(double ratio)
{
	if (ratio <= 0.0)
		return false;
	if (100.0 * drand48() < ratio)
		return true;
	return false;
}

Datum
pgstrom_random_int(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	int64		lower = (!PG_ARGISNULL(1) ? PG_GETARG_INT64(1) : 0);
	int64		upper = (!PG_ARGISNULL(2) ? PG_GETARG_INT64(2) : INT_MAX);
	cl_ulong	v;

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_INT64(lower);
	v = ((cl_ulong)random() << 31) | (cl_ulong)random();

	PG_RETURN_INT64(lower + v % (upper - lower));
}
PG_FUNCTION_INFO_V1(pgstrom_random_int);

Datum
pgstrom_random_float(PG_FUNCTION_ARGS)
{
	float8	ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	float8	lower = (!PG_ARGISNULL(1) ? PG_GETARG_FLOAT8(1) : 0.0);
	float8	upper = (!PG_ARGISNULL(2) ? PG_GETARG_FLOAT8(2) : 1.0);

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_FLOAT8(lower);

	PG_RETURN_FLOAT8((upper - lower) * drand48() + lower);
}
PG_FUNCTION_INFO_V1(pgstrom_random_float);

Datum
pgstrom_random_date(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	DateADT		lower;
	DateADT		upper;
	cl_ulong	v;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_DATEADT(1);
	else
		lower = date2j(2015, 1, 1) - POSTGRES_EPOCH_JDATE;
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_DATEADT(2);
	else
		upper = date2j(2025, 12, 31) - POSTGRES_EPOCH_JDATE;

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_DATEADT(lower);
	v = ((cl_ulong)random() << 31) | (cl_ulong)random();

	PG_RETURN_DATEADT(lower + v % (upper - lower));
}
PG_FUNCTION_INFO_V1(pgstrom_random_date);

Datum
pgstrom_random_time(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	TimeADT		lower = 0;
	TimeADT		upper = HOURS_PER_DAY * USECS_PER_HOUR - 1;
	cl_ulong	v;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMEADT(1);
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMEADT(2);
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_TIMEADT(lower);
	v = ((cl_ulong)random() << 31) | (cl_ulong)random();

	PG_RETURN_TIMEADT(lower + v % (upper - lower));
}
PG_FUNCTION_INFO_V1(pgstrom_random_time);

Datum
pgstrom_random_timetz(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	TimeADT		lower = 0;
	TimeADT		upper = HOURS_PER_DAY * USECS_PER_HOUR - 1;
	TimeTzADT  *temp;
	cl_ulong	v;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMEADT(1);
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMEADT(2);
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	temp = palloc(sizeof(TimeTzADT));
	temp->zone = (random() % 23 - 11) * USECS_PER_HOUR;
	if (upper == lower)
		temp->time = lower;
	else
	{
		v = ((cl_ulong)random() << 31) | (cl_ulong)random();
		temp->time = lower + v % (upper - lower);
	}
	PG_RETURN_TIMETZADT_P(temp);
}
PG_FUNCTION_INFO_V1(pgstrom_random_timetz);

Datum
pgstrom_random_timestamp(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	Timestamp	lower;
	Timestamp	upper;
	cl_ulong	v;
	struct pg_tm tm;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMESTAMP(1);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 45;	/* '2015-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &lower) != 0)
			elog(ERROR, "timestamp out of range");
	}

	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMESTAMP(2);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 55;	/* '2025-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &upper) != 0)
			elog(ERROR, "timestamp out of range");
	}
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_TIMEADT(lower);
	v = ((cl_ulong)random() << 31) | (cl_ulong)random();

	PG_RETURN_TIMESTAMP(lower + v % (upper - lower));
}
PG_FUNCTION_INFO_V1(pgstrom_random_timestamp);

Datum
pgstrom_random_macaddr(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	macaddr	   *temp;
	cl_ulong	lower;
	cl_ulong	upper;
	cl_ulong	v, x;

	if (PG_ARGISNULL(1))
		lower = 0xabcd00000000UL;
	else
	{
		temp = PG_GETARG_MACADDR_P(1);
		lower = (((cl_ulong)temp->a << 40) | ((cl_ulong)temp->b << 32) |
				 ((cl_ulong)temp->c << 24) | ((cl_ulong)temp->d << 16) |
				 ((cl_ulong)temp->e <<  8) | ((cl_ulong)temp->f));
	}

	if (PG_ARGISNULL(2))
		upper = 0xabcdffffffffUL;
	else
	{
		temp = PG_GETARG_MACADDR_P(2);
		upper = (((cl_ulong)temp->a << 40) | ((cl_ulong)temp->b << 32) |
				 ((cl_ulong)temp->c << 24) | ((cl_ulong)temp->d << 16) |
				 ((cl_ulong)temp->e <<  8) | ((cl_ulong)temp->f));
	}

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		x = lower;
	else
	{
		v = ((cl_ulong)random() << 31) | (cl_ulong)random();
		x = lower + v % (upper - lower);
	}
	temp = palloc(sizeof(macaddr));
	temp->a = (x >> 40) & 0x00ff;
	temp->b = (x >> 32) & 0x00ff;
	temp->c = (x >> 24) & 0x00ff;
	temp->d = (x >> 16) & 0x00ff;
	temp->e = (x >>  8) & 0x00ff;
	temp->f = (x      ) & 0x00ff;
	PG_RETURN_MACADDR_P(temp);
}
PG_FUNCTION_INFO_V1(pgstrom_random_macaddr);

Datum
pgstrom_random_inet(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	inet	   *temp;
	int			i, j, bits;
	cl_ulong	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();

	if (!PG_ARGISNULL(1))
		temp = (inet *)PG_DETOAST_DATUM_COPY(PG_GETARG_DATUM(1));
	else
	{
		/* default template 192.168.0.1/16 */
		temp = palloc(sizeof(inet));
		temp->inet_data.family = PGSQL_AF_INET;
		temp->inet_data.bits = 16;
		temp->inet_data.ipaddr[0] = 0xc0;
		temp->inet_data.ipaddr[1] = 0xa8;
		temp->inet_data.ipaddr[2] = 0x01;
		temp->inet_data.ipaddr[3] = 0x00;
		SET_VARSIZE(temp, sizeof(inet));
	}
	bits = ip_bits(temp);
	i = ip_maxbits(temp) / 8 - 1;
	j = v = 0;
	while (bits > 0)
	{
		if (j < 8)
		{
			v |= (cl_ulong)random() << j;
			j += 31;	/* note: only 31b of random() are valid */
		}
		if (bits >= 8)
			temp->inet_data.ipaddr[i--] = (v & 0xff);
		else
		{
			cl_uint		mask = (1 << bits) - 1;

			temp->inet_data.ipaddr[i] &= ~(mask);
			temp->inet_data.ipaddr[i] |= (v & mask);
			i--;
		}
		bits -= 8;
		v >>= 8;
	}
	ip_bits(temp) = ip_maxbits(temp);
	PG_RETURN_INET_P(temp);
}
PG_FUNCTION_INFO_V1(pgstrom_random_inet);

Datum
pgstrom_random_text(PG_FUNCTION_ARGS)
{
	static const char *base32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	text	   *temp;
	char	   *pos;
	int			i, j, n;
	cl_ulong	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();

	if (PG_ARGISNULL(1))
		temp = cstring_to_text("test_**");
	else
		temp = PG_GETARG_TEXT_P_COPY(1);

	n = VARSIZE_ANY_EXHDR(temp);
	pos = VARDATA_ANY(temp);
	for (i=0, j=0, v=0; i < n; i++, pos++)
	{
		if (*pos == '*')
		{
			if (j < 5)
			{
				v |= (cl_ulong)random() << j;
				j += 31;
			}
			*pos = base32[v & 0x1f];
			v >>= 5;
			j -= 5;
		}
	}
	PG_RETURN_TEXT_P(temp);
}
PG_FUNCTION_INFO_V1(pgstrom_random_text);

Datum
pgstrom_random_text_length(PG_FUNCTION_ARGS)
{
	static const char *base32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	cl_int		maxlen;
	text	   *temp;
	char	   *pos;
	int			i, j, n;
	cl_ulong	v = 0;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	maxlen = (PG_ARGISNULL(1) ? 10 : PG_GETARG_INT32(1));
	if (maxlen < 1 || maxlen > BLCKSZ)
		elog(ERROR, "%s: max length too much", __FUNCTION__);
	n = 1 + random() % maxlen;

	temp = palloc(VARHDRSZ + n);
	SET_VARSIZE(temp, VARHDRSZ + n);
	pos = VARDATA(temp);
	for (i=0, j=0; i < n; i++, pos++)
	{
		if (j < 5)
		{
			v |= (cl_ulong)random() << j;
			j += 31;
		}
		*pos = base32[v & 0x1f];
		v >>= 5;
		j -= 5;
	}
	PG_RETURN_TEXT_P(temp);
}
PG_FUNCTION_INFO_V1(pgstrom_random_text_length);

static Datum
simple_make_range(TypeCacheEntry *typcache, Datum x_val, Datum y_val)
{
	RangeBound	x, y;
	RangeType  *range;

	memset(&x, 0, sizeof(RangeBound));
	x.val = x_val;
	x.infinite = generate_null(0.5);
	x.inclusive = generate_null(25.0);

	memset(&y, 0, sizeof(RangeBound));
	y.val = y_val;
	y.infinite = generate_null(0.5);
	y.inclusive = generate_null(25.0);

	if (x.infinite || y.infinite || x.val <= y.val)
	{
		x.lower = true;
		range = make_range(typcache, &x, &y, false);
	}
	else
	{
		y.lower = true;
		range = make_range(typcache, &y, &x, false);
	}
	return PointerGetDatum(range);
}

Datum
pgstrom_random_int4range(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	int32		lower = (!PG_ARGISNULL(1) ? PG_GETARG_INT32(1) : 0);
	int32		upper = (!PG_ARGISNULL(2) ? PG_GETARG_INT32(2) : INT_MAX);
	int32		x, y;
	Oid			type_oid;
	TypeCacheEntry *typcache;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   CStringGetDatum("int4range"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	typcache = range_get_typcache(fcinfo, type_oid);
	x = lower + random() % (upper - lower);
	y = lower + random() % (upper - lower);
	return simple_make_range(typcache,
							 Int32GetDatum(x),
							 Int32GetDatum(y));
}
PG_FUNCTION_INFO_V1(pgstrom_random_int4range);

Datum
pgstrom_random_int8range(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	int64		lower = (!PG_ARGISNULL(1) ? PG_GETARG_INT64(1) : 0);
	int64		upper = (!PG_ARGISNULL(2) ? PG_GETARG_INT64(2) : LONG_MAX);
	TypeCacheEntry *typcache;
	Oid			type_oid;
	int64		x, y, v;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   CStringGetDatum("int8range"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	typcache = range_get_typcache(fcinfo, type_oid);
	v = ((int64)random() << 31) | (int64)random();
	x = lower + v % (upper - lower);
	v = ((int64)random() << 31) | (int64)random();
	y = lower + v % (upper - lower);
	return simple_make_range(typcache,
							 Int64GetDatum(x),
							 Int64GetDatum(y));
}
PG_FUNCTION_INFO_V1(pgstrom_random_int8range);

Datum
pgstrom_random_tsrange(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	Timestamp	lower;
	Timestamp	upper;
	struct pg_tm tm;
	TypeCacheEntry *typcache;
	Oid			type_oid;
	Timestamp	x, y;
	cl_ulong	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMESTAMP(1);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 45;	/* '2015-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &lower) != 0)
			elog(ERROR, "timestamp out of range");
	}

	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMESTAMP(2);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 55;	/* '2025-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &upper) != 0)
			elog(ERROR, "timestamp out of range");
	}
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);

	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   CStringGetDatum("tsrange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	typcache = range_get_typcache(fcinfo, type_oid);
	v = ((cl_ulong)random() << 31) | random();
	x = lower + v % (upper - lower);
	v = ((cl_ulong)random() << 31) | random();
	y = lower + v % (upper - lower);
	return simple_make_range(typcache,
							 TimestampGetDatum(x),
							 TimestampGetDatum(y));	
}
PG_FUNCTION_INFO_V1(pgstrom_random_tsrange);

Datum
pgstrom_random_tstzrange(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	Timestamp	lower;
	Timestamp	upper;
	struct pg_tm tm;
	TypeCacheEntry *typcache;
	Oid			type_oid;
	Timestamp	x, y;
	cl_ulong	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMESTAMP(1);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 45;	/* '2015-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &lower) != 0)
			elog(ERROR, "timestamp out of range");
	}

	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMESTAMP(2);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 55;	/* '2025-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &upper) != 0)
			elog(ERROR, "timestamp out of range");
	}
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);

	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   CStringGetDatum("tstzrange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	typcache = range_get_typcache(fcinfo, type_oid);
	v = ((cl_ulong)random() << 31) | random();
	x = lower + v % (upper - lower);
	v = ((cl_ulong)random() << 31) | random();
	y = lower + v % (upper - lower);
	return simple_make_range(typcache,
							 TimestampTzGetDatum(x),
							 TimestampTzGetDatum(y));	
}
PG_FUNCTION_INFO_V1(pgstrom_random_tstzrange);

Datum
pgstrom_random_daterange(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	DateADT		lower;
	DateADT		upper;
	DateADT		x, y;
	TypeCacheEntry *typcache;
	Oid			type_oid;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_DATEADT(1);
	else
		lower = date2j(2015, 1, 1) - POSTGRES_EPOCH_JDATE;
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_DATEADT(2);
	else
		upper = date2j(2025, 12, 31) - POSTGRES_EPOCH_JDATE;
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);

	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   CStringGetDatum("daterange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	typcache = range_get_typcache(fcinfo, type_oid);
	x = lower + random() % (upper - lower);
	y = lower + random() % (upper - lower);
	return simple_make_range(typcache,
							 DateADTGetDatum(x),
							 DateADTGetDatum(y));
}
PG_FUNCTION_INFO_V1(pgstrom_random_daterange);
