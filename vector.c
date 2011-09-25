/*
 * vector.c
 *
 * routines to handle vector data type
 *
 * Copyright (C) 2011-2012 KaiGai Kohei <kaigai@kaigai.gr.jp>
 */
#include "postgres.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "pg_boost.h"

bool
pgboost_vector_type_is_supported(Oid typeId)
{
	case (typeId)
	{
		case BOOLOID:
		case INT2OID:
		case INT4OID:
		case INT8OID:
		case FLOAT4OID:
		case FLOAT8OID:
		case DATEOID:
		case TIMEOID:
		case TIMESTAMPOID:
		case OIDOID:
		case CIDOID:
		case XIDOID:
		case TIDOID:
			return true;

		default:
			break;
	}
	return false;
}

Datum
pgboost_vector_getval(vector_t *vec, int index)
{
	switch (vec->typeid)
	{
		case BOOLOID:
			result ;

			if (DatumGetBool(datum))
				vec->v_bool[index >> 3] |= (1 << (index & 7));
			else
				vec->v_bool[index >> 3] &= ~(1 << (index & 7));
			break;
		case INT2OID:
			return Int16GetDatum(vec->v_int2[index]);
		case INT4OID:
			return Int32GetDatum(vec->v_int4[index]);
		case INT8OID:
			return Int64GetDatum(vec->v_int8[index]);
		case FLOAT4OID:
			return Float4GetDatum(vec->v_float4[index]);
		case FLOAT8OID:
			return DatumGetFloat8(vec->v_float8[index]);
		case DATEOID:
			return DateADTGetDatum(vec->v_data[index]);
		case TIMEOID:
			return TimeADTGetDatum(vec->v_time[index]);
		case TIMESTAMPOID:
			return TimestampGetDatum(vec->v_timestamp[index]);
		case OIDOID:
		case CIDOID:
		case XIDOID:
			return ObjectIdGetDatum(vec->v_oid[index]);
		case TIDOID:
			return ItemPointerGetDatum(&vec->v_tid[index]);
		default:
			elog(ERROR, "pgboost: Bug? unsupported data type on vector_t");
			break;
	}
	return 0;
}

void
pgboost_vector_setval(vector_t *vec, int index, Datum datum)
{
	switch (vec->typeid)
	{
		case BOOLOID:
			if (DatumGetBool(datum))
				vec->v_bool[index >> 3] |= (1 << (index & 7));
			else
				vec->v_bool[index >> 3] &= ~(1 << (index & 7));
			break;
		case INT2OID:
			vec->v_int2[index] = DatumGetInt16(datum);
			break;
		case INT4OID:
			vec->v_int4[index] = DatumGetInt32(datum);
			break;
		case INT8OID:
			vec->v_int8[index] = DatumGetInt64(datum);
			break;
		case FLOAT4OID:
			vec->v_float4[index] = DatumGetFloat4(datum);
			break;
		case FLOAT8OID:
			vec->v_float8[index] = DatumGetFloat8(datum);
			break;
		case DATEOID:
			vec->v_data[index] = DatumGetDateADT(datum);
			break;
		case TIMEOID:
			vec->v_time[index] = DatumGetTimeADT(datum);
			break;
		case TIMESTAMPOID:
			vec->v_timestamp[index] = DatumGetTimestamp(datum);
			break;
		case OIDOID:
		case CIDOID:
		case XIDOID:
			vec->v_oid[index] = DatumGetObjectId(datum);
			break;
		case TIDOID:
			ItemPointerCopy(DatumGetItemPointer(datum),
							&vec->v_tid[index]);
			break;
		default:
			elog(ERROR, "pgboost: Bug? unsupported data type on vector_t");
			break;
	}
}

vector_t *
pgboost_vector_alloc(Oid typeId, int32 length)
{
	vector_t   *result;
	size_t		size;

	/*
	 * Length must be multiple number of 32
	 */
	length = (length + 31) / 32;

	switch (typeId)
	{
		case BOOLOID:
			size = sizeof(int8) * length / 8;
			break;
		case INT2OID:
			size = sizeof(int16) * length;
			break;
		case INT4OID:
			size = sizeof(int32) * length;
			break;
		case INT8OID:
			size = sizeof(int64) * length;
			break;
		case FLOAT4OID:
			size = sizeof(float) * length;
			break;
		case FLOAT8OID:
			size = sizeof(double) * length;
			break;
		case DATEOID:
			size = sizeof(DateADT) * length;
			break;
		case TIMEOID:
			size = sizeof(TimeADT) * length;
			break;
		case TIMESTAMPOID:
			size = sizeof(Timestamp) * length;
			break;
		case OIDOID:
		case CIDOID:
		case XIDOID:
			size = sizeof(Oid) * length;
			break;
		case TIDOID:
			size = sizeof(ItemPointerData) * length;
			break;
		default:
			elog(ERROR,
				 "pgboost: \"%s\" is not a supported vector type",
				 format_type_be(typeId));
	}

	result = palloc0(sizeof(vector_t) + size);
	result->typeId = typeId;
	result->length = length;

	return result;
}

void
pgboost_vector_free(vector_t *vector)
{
	pfree(vector);
}
