/*
 * cuda_postgis.cu
 *
 * Routines of basic PostGIS functions & operators for CUDA GPU devices
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"
#include "cuda_postgis.h"

/* ================================================================
 *
 * Internal Utility Routines
 *
 * ================================================================ */

/*
 * Floating point comparators. (see, liblwgeom_internal.h)
 */
#define FP_TOLERANCE 1e-12
#define FP_IS_ZERO(A) (fabs(A) <= FP_TOLERANCE)
#define FP_MAX(A, B) (((A) > (B)) ? (A) : (B))
#define FP_MIN(A, B) (((A) < (B)) ? (A) : (B))
#define FP_ABS(a)   ((a) <  (0) ? -(a) : (a))
#define FP_EQUALS(A, B) (fabs((A)-(B)) <= FP_TOLERANCE)
#define FP_NEQUALS(A, B) (fabs((A)-(B)) > FP_TOLERANCE)
#define FP_LT(A, B) (((A) + FP_TOLERANCE) < (B))
#define FP_LTEQ(A, B) (((A) - FP_TOLERANCE) <= (B))
#define FP_GT(A, B) (((A) - FP_TOLERANCE) > (B))
#define FP_GTEQ(A, B) (((A) + FP_TOLERANCE) >= (B))
#define FP_CONTAINS_TOP(A, X, B) (FP_LT(A, X) && FP_LTEQ(X, B))
#define FP_CONTAINS_BOTTOM(A, X, B) (FP_LTEQ(A, X) && FP_LT(X, B))
#define FP_CONTAINS_INCL(A, X, B) (FP_LTEQ(A, X) && FP_LTEQ(X, B))
#define FP_CONTAINS_EXCL(A, X, B) (FP_LT(A, X) && FP_LT(X, B))
#define FP_CONTAINS(A, X, B) FP_CONTAINS_EXCL(A, X, B)

STATIC_INLINE(cl_bool)
geometry_is_collection(const pg_geometry_t *geom)
{
	/* see lw_dist2d_is_collection */
	if (geom->type == GEOM_TINTYPE ||
		geom->type == GEOM_MULTIPOINTTYPE ||
		geom->type == GEOM_MULTILINETYPE ||
		geom->type == GEOM_MULTIPOLYGONTYPE ||
		geom->type == GEOM_COLLECTIONTYPE ||
		geom->type == GEOM_MULTICURVETYPE ||
		geom->type == GEOM_MULTISURFACETYPE ||
		geom->type == GEOM_COMPOUNDTYPE ||
		geom->type == GEOM_POLYHEDRALSURFACETYPE)
		return true;

	return false;
}

STATIC_FUNCTION(cl_bool)
setup_geometry_rawsize(pg_geometry_t *geom)
{
	switch (geom->type)
	{
		case GEOM_POINTTYPE:
		case GEOM_LINETYPE:
		case GEOM_TRIANGLETYPE:
		case GEOM_CIRCSTRINGTYPE:
			geom->rawsize = (sizeof(double) *
							 GEOM_FLAGS_NDIMS(geom->flags) * geom->nitems);
			return true;
		case GEOM_POLYGONTYPE:
			{
				size_t		rawsize = LONGALIGN(sizeof(cl_uint)*geom->nitems);
				const char *rawdata = geom->rawdata;
				cl_uint		__nitems;

				for (int i=0; i < geom->nitems; i++)
				{
					memcpy(&__nitems, rawdata, sizeof(cl_uint));
					rawdata += sizeof(cl_uint);
					rawsize += (sizeof(double) *
								GEOM_FLAGS_NDIMS(geom->flags) * __nitems);
				}
				geom->rawsize = rawsize;
			}
			return true;
		case GEOM_MULTIPOINTTYPE:
		case GEOM_MULTILINETYPE:
		case GEOM_MULTIPOLYGONTYPE:
		case GEOM_COLLECTIONTYPE:
		case GEOM_COMPOUNDTYPE:
		case GEOM_CURVEPOLYTYPE:
		case GEOM_MULTICURVETYPE:
		case GEOM_MULTISURFACETYPE:
		case GEOM_POLYHEDRALSURFACETYPE:
		case GEOM_TINTYPE:
			{
				const char *pos = geom->rawdata;
				for (int i=0; i < geom->nitems; i++)
				{
					pg_geometry_t __geom;

					__geom.type = __Fetch((cl_int *)pos);
					pos += sizeof(cl_int);
					__geom.flags = geom->flags;
					__geom.nitems = __Fetch((cl_uint *)pos);
					pos += sizeof(cl_uint);
					__geom.rawdata = pos;
					if (!setup_geometry_rawsize(&__geom))
						return false;
					pos += __geom.rawsize;
				}
				geom->rawsize = (pos - geom->rawdata);
			}
			return true;
		default:
			/* unknown geometry type */
			break;
	}
	return false;
}

STATIC_FUNCTION(const char *)
geometry_load_subitem(pg_geometry_t *gsub,
					  const pg_geometry_t *geom,
					  const char *pos, int index,
					  kern_context *kcxt = NULL)
{
	switch (geom->type)
	{
		case GEOM_POINTTYPE:
		case GEOM_LINETYPE:
		case GEOM_TRIANGLETYPE:
		case GEOM_CIRCSTRINGTYPE:
			/* no sub-geometry */
			if (kcxt)
				STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
							  "geometry data curruption (no sub-items)");
			break;
		case GEOM_POLYGONTYPE:
			if (index == 0)
				pos = geom->rawdata + LONGALIGN(sizeof(cl_int) * geom->nitems);
			else if (index >= geom->nitems)
				break;
			gsub->isnull = false;
			gsub->type   = GEOM_LINETYPE;
			gsub->flags  = geom->flags;
			gsub->srid   = geom->srid;
			gsub->nitems = __Fetch(((cl_uint *)geom->rawdata) + index);
			gsub->rawdata = pos;
			gsub->bbox   = geom->bbox;
			setup_geometry_rawsize(gsub);
			pos += gsub->rawsize;
			if (pos > geom->rawdata + geom->rawsize)
			{
				if (kcxt)
					STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
								  "geometry data curruption (polygon)");
				break;
			}
			return pos;

		case GEOM_MULTIPOINTTYPE:
        case GEOM_MULTILINETYPE:
        case GEOM_MULTIPOLYGONTYPE:
        case GEOM_COLLECTIONTYPE:
        case GEOM_COMPOUNDTYPE:
        case GEOM_CURVEPOLYTYPE:
        case GEOM_MULTICURVETYPE:
        case GEOM_MULTISURFACETYPE:
        case GEOM_POLYHEDRALSURFACETYPE:
        case GEOM_TINTYPE:
			if (index == 0)
				pos = geom->rawdata;
			else if (index >= geom->nitems)
				break;
			gsub->isnull = false;
			gsub->type = __Fetch((cl_int *)pos);
			pos += sizeof(cl_int);
			gsub->flags = geom->flags;
			gsub->srid = geom->srid;
			gsub->nitems = __Fetch((cl_uint *)pos);
			pos += sizeof(cl_uint);
			gsub->rawdata = pos;
			gsub->bbox = geom->bbox;
			setup_geometry_rawsize(gsub);
			pos += gsub->rawsize;
			if (pos > geom->rawdata + geom->rawsize)
			{
				if (kcxt)
					STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
								  "geometry data curruption (collection)");
				break;
			}
			return pos;

		default:
			/* unknown geometry type */
			break;
	}
	memset(gsub, 0, sizeof(pg_geometry_t));
	gsub->isnull = true;
	return NULL;
}

STATIC_FUNCTION(cl_bool)
geometry_is_empty(const pg_geometry_t *geom)
{
	if (geometry_is_collection(geom))
	{
		pg_geometry_t	__geom;
		const char	   *pos = NULL;

		for (int i=0; i < geom->nitems; i++)
		{
			pos = geometry_load_subitem(&__geom, geom, pos, i);
			if (!geometry_is_empty(&__geom))
				return false;
		}
		return true;	/* empty */
	}
	return (geom->nitems == 0 ? true : false);
}

/* ================================================================
 *
 * GSERIALIZED v1/v2 (see, gserialized1.h and gserialized2.h)
 *
 * ================================================================
 */
STATIC_INLINE(cl_int)
__gserialized_get_srid(const __GSERIALIZED *gs)
{
	cl_int		srid;

	/* Only the first 21 bits are set. */
	srid = (((cl_uint)gs->srid[0] << 16) |
			((cl_uint)gs->srid[1] <<  8) |
			((cl_uint)gs->srid[2])) & 0x001fffffU;
	/* 0 is our internal unknown value */
	return (srid == 0 ? SRID_UNKNOWN : srid);
}

STATIC_FUNCTION(pg_geometry_t)
__geometry_datum_ref_v1(kern_context *kcxt, void *addr, cl_int sz)
{
	/* see lwgeom_from_gserialized1() */
	__GSERIALIZED  *gs = (__GSERIALIZED *)addr;
	pg_geometry_t	geom;
	cl_uint			gs_type;
	cl_ushort		geom_flags = 0;
	char		   *rawdata = gs->data;

	memset(&geom, 0, sizeof(pg_geometry_t));
	if ((gs->gflags & G1FLAG_Z) != 0)
		geom_flags |= GEOM_FLAG__Z;
	if ((gs->gflags & G1FLAG_M) != 0)
		geom_flags |= GEOM_FLAG__M;
	if ((gs->gflags & G1FLAG_BBOX) != 0)
		geom_flags |= GEOM_FLAG__BBOX;
	if ((gs->gflags & G1FLAG_GEODETIC) != 0)
		geom_flags |= GEOM_FLAG__GEODETIC;
	if ((gs->gflags & G1FLAG_SOLID) != 0)
		geom_flags |= GEOM_FLAG__SOLID;

	geom.flags = geom_flags;
	geom.srid = __gserialized_get_srid(gs);
	if ((geom.flags & GEOM_FLAG__BBOX) != 0)
	{
		geom.bbox = (geom_bbox *)rawdata;
		rawdata += geometry_bbox_size(geom.flags);
	}
	memcpy(&gs_type, rawdata, sizeof(cl_uint));
	rawdata += sizeof(cl_uint);
	if (GEOM_TYPE_IS_VALID(gs_type))
		geom.type = gs_type;
	else
	{
		STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
					  "geometry data v1 has unsupported type");
		geom.isnull = true;
	}

	memcpy(&geom.nitems, rawdata, sizeof(cl_uint));
	rawdata += sizeof(cl_uint);
	geom.rawdata = rawdata;
	if (!setup_geometry_rawsize(&geom) ||
		geom.rawdata + geom.rawsize > gs->data + sz)
	{
		STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
					  "geometry data v1 corrupted");
		geom.isnull = true;
	}
	/*
	 * NOTE: Unlike CPU version of lwgeom_from_gserialized1(),
	 * we don't generate boundary box here, even if raw geometry
	 * datum has no boundary box unexpectedly. It is mandatorily
	 * generated by PostGIS, thus the code to construct bounday
	 * box just consumes device memory footpoint, and we have no
	 * proper way for code debugging.
	 */
	return geom;
}

/* flags of GSERIALIZED v2 */

STATIC_FUNCTION(pg_geometry_t)
__geometry_datum_ref_v2(kern_context *kcxt, void *addr, cl_int sz)
{
	/* see lwgeom_from_gserialized2() */
	__GSERIALIZED  *gs = (__GSERIALIZED *)addr;
	pg_geometry_t	geom;
	cl_uint			gs_type;
	cl_ushort		geom_flags = 0;
	char		   *rawdata = gs->data;

	memset(&geom, 0, sizeof(pg_geometry_t));

	/* parse version.2 flags */
	if ((gs->gflags & G2FLAG_Z) != 0)
		geom_flags |= GEOM_FLAG__Z;
	if ((gs->gflags & G2FLAG_M) != 0)
		geom_flags |= GEOM_FLAG__M;
	if ((gs->gflags & G2FLAG_BBOX) != 0)
		geom_flags |= GEOM_FLAG__BBOX;
	if ((gs->gflags & G2FLAG_GEODETIC) != 0)
		geom_flags |= G1FLAG_GEODETIC;
	if ((gs->gflags & G2FLAG_EXTENDED) != 0)
	{
		cl_ulong    ex_flags;

		memcpy(&ex_flags, rawdata, sizeof(cl_ulong));
		if ((ex_flags & G2FLAG_X_SOLID) != 0)
			geom_flags |= GEOM_FLAG__SOLID;
		rawdata += sizeof(cl_ulong);
	}
	geom.flags = geom_flags;
	geom.srid = __gserialized_get_srid(gs);
	if ((geom.flags & GEOM_FLAG__BBOX) != 0)
	{
		geom.bbox = (geom_bbox *)rawdata;
		rawdata += geometry_bbox_size(geom.flags);
	}

	memcpy(&gs_type, rawdata, sizeof(cl_uint));
	rawdata += sizeof(cl_uint);
	if (GEOM_TYPE_IS_VALID(gs_type))
		geom.type = gs_type;
	else
	{
		STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
					  "geometry data v2 has unsupported type");
		geom.isnull = true;
	}

	memcpy(&geom.nitems, rawdata, sizeof(cl_uint));
	rawdata += sizeof(cl_uint);
	geom.rawdata = rawdata;
	if (!setup_geometry_rawsize(&geom) ||
		geom.rawdata + geom.rawsize > gs->data + sz)
	{
		STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
					  "geometry data v2 corrupted");
		geom.isnull = true;
	}
	/*
	 * NOTE: Unlike CPU version of lwgeom_from_gserialized1(),
	 * we don't generate boundary box here, even if raw geometry
	 * datum has no boundary box unexpectedly. It is mandatorily
	 * generated by PostGIS, thus the code to construct bounday
	 * box just consumes device memory footpoint, and we have no
	 * proper way for code debugging.
	 */
	return geom;
}

DEVICE_FUNCTION(pg_geometry_t)
pg_geometry_datum_ref(kern_context *kcxt, void *addr)
{
	pg_geometry_t	result;

	if (!addr)
	{
		memset(&result, 0, sizeof(pg_geometry_t));
		result.isnull = true;
	}
	else if (VARATT_IS_1B_E(addr))
	{
		STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_VARLENA_UNSUPPORTED,
						   "varlena datum is compressed or external");
		memset(&result, 0, sizeof(pg_geometry_t));
		result.isnull = true;
	}
	else
	{
		__GSERIALIZED  *g = (__GSERIALIZED *)VARDATA_ANY(addr);
		cl_int			sz = VARSIZE_ANY_EXHDR(addr);

		if ((g->gflags & G2FLAG_VER_0) != 0)
			result = __geometry_datum_ref_v2(kcxt, g, sz);
		else
			result = __geometry_datum_ref_v1(kcxt, g, sz);
	}
	return result;
}

DEVICE_FUNCTION(void)
pg_datum_ref(kern_context *kcxt, pg_geometry_t &result, void *addr)
{
	result = pg_geometry_datum_ref(kcxt, addr);
}

DEVICE_FUNCTION(void)
pg_datum_ref_slot(kern_context *kcxt,
				  pg_geometry_t &result,
				  cl_char dclass, Datum datum)
{
	if (dclass == DATUM_CLASS__NULL)
		result = pg_geometry_datum_ref(kcxt, NULL);
	else if (dclass == DATUM_CLASS__GEOMETRY)
		memcpy(&result, DatumGetPointer(datum), sizeof(pg_geometry_t));
	else
	{
		assert(dclass == DATUM_CLASS__NORMAL);
		result = pg_geometry_datum_ref(kcxt, (char *)datum);
	}
}

DEVICE_FUNCTION(pg_geometry_t)
pg_geometry_param(kern_context *kcxt, cl_uint param_id)
{
	kern_parambuf *kparams = kcxt->kparams;
	pg_geometry_t result;

	if (param_id < kparams->nparams &&
		kparams->poffset[param_id] > 0)
	{
		void   *addr = ((char *)kparams +
						kparams->poffset[param_id]);
		result = pg_geometry_datum_ref(kcxt, addr);
	}
	else
	{
		memset(&result, 0, sizeof(pg_geometry_t));
		result.isnull = true;
	}
	return result;
}

DEVICE_FUNCTION(cl_int)
pg_datum_store(kern_context *kcxt,
			   pg_geometry_t datum,
			   cl_char &dclass,
			   Datum &value)
{
	if (datum.isnull)
		dclass = DATUM_CLASS__NULL;
	else
	{
		pg_geometry_t *temp;

		temp = (pg_geometry_t *)
			kern_context_alloc(kcxt, sizeof(pg_geometry_t));
		if (temp)
		{
			memcpy(temp, &datum, sizeof(pg_geometry_t));
			dclass = DATUM_CLASS__GEOMETRY;
			value = PointerGetDatum(temp);
			return sizeof(pg_geometry_t);
		}
		STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY,
					  "out of memory");
		dclass = DATUM_CLASS__NULL;
	}
	return 0;
}

/* ================================================================
 *
 * box2df (bounding-box) input/output functions
 *
 * ================================================================
 */
DEVICE_FUNCTION(pg_box2df_t)
pg_box2df_datum_ref(kern_context *kcxt, void *addr)
{
	pg_box2df_t   result;

	if (!addr)
		result.isnull = true;
	else
	{
		result.isnull = false;
		memcpy(&result.value, (geom_bbox_2d *)addr, sizeof(geom_bbox_2d));
	}
	return result;                                              
}                                                               

DEVICE_FUNCTION(void)
pg_datum_ref(kern_context *kcxt, pg_box2df_t &result, void *addr)
{
	result = pg_box2df_datum_ref(kcxt, addr);
}

DEVICE_FUNCTION(void)                                             
pg_datum_ref_slot(kern_context *kcxt,
				  pg_box2df_t &result,
				  cl_char dclass, Datum datum)
{
	if (dclass == DATUM_CLASS__NULL)
		result = pg_box2df_datum_ref(kcxt, NULL);             
	else
	{
		assert(dclass == DATUM_CLASS__NORMAL);
		result = pg_box2df_datum_ref(kcxt, (char *)datum);
	}                                                           
}

DEVICE_FUNCTION(pg_box2df_t)
pg_box2df_param(kern_context *kcxt,cl_uint param_id)
{
	kern_parambuf *kparams = kcxt->kparams;                     
	pg_box2df_t result;                                       

	if (param_id < kparams->nparams &&
		kparams->poffset[param_id] > 0)                         
	{
		void   *addr = ((char *)kparams +                       
						kparams->poffset[param_id]);            
		result = pg_box2df_datum_ref(kcxt, addr);             
	}                                                           
	else                                                        
		result.isnull = true;                                   
	return result;                                              
}                                                               

DEVICE_FUNCTION(cl_int)                                           
pg_datum_store(kern_context *kcxt,                              
			   pg_box2df_t datum,                             
			   cl_char &dclass,                                 
			   Datum &value)                                    
{
	void	   *res;

	if (datum.isnull)
	{
		dclass = DATUM_CLASS__NULL;
		return 0;
	}
	res = kern_context_alloc(kcxt, sizeof(geom_bbox_2d));
	if (!res)
	{
		dclass = DATUM_CLASS__NULL;
		STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY, "out of memory");
		return 0;                                               
	}
	memcpy(res, &datum.value, sizeof(geom_bbox_2d));
	dclass = DATUM_CLASS__NORMAL;
	value = PointerGetDatum(res);
	return sizeof(geom_bbox_2d);
}

/* ================================================================
 *
 * Basic geometry constructor and related
 *
 * ================================================================
 */
DEVICE_FUNCTION(pg_geometry_t)
pgfn_st_setsrid(kern_context *kcxt, const pg_geometry_t &arg1, pg_int4_t arg2)
{
	pg_geometry_t	result;

	if (arg1.isnull || arg2.isnull)
	{
		memset(&result, 0, sizeof(pg_geometry_t));
		result.isnull = true;
	}
	else
	{
		/* see clamp_srid */
		cl_int	srid = arg2.value;

		if (srid <= 0)
			srid = SRID_UNKNOWN;
		else if (srid > SRID_MAXIMUM)
			srid = SRID_USER_MAXIMUM + 1 +
				(srid % (SRID_MAXIMUM - SRID_USER_MAXIMUM - 1));
		memcpy(&result, &arg1, sizeof(pg_geometry_t));
		result.srid = srid;
	}
	return result;
}

DEVICE_FUNCTION(pg_geometry_t)
pgfn_st_makepoint2(kern_context *kcxt, pg_float8_t x, pg_float8_t y)
{
	pg_geometry_t	geom;

	memset(&geom, 0, sizeof(pg_geometry_t));
	if (x.isnull || y.isnull)
		geom.isnull = true;
	else
	{
		cl_double  *rawdata = (cl_double *)
			kern_context_alloc(kcxt, 2 * sizeof(cl_double));

		if (!rawdata)
		{
			STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,
							   "out of memory");
			geom.isnull = true;
		}
		else
		{
			rawdata[0] = x.value;
			rawdata[1] = y.value;

			geom.type = GEOM_POINTTYPE;
			geom.flags = 0;
			geom.srid = SRID_UNKNOWN;
			geom.nitems = 1;
			geom.rawsize = 2 * sizeof(cl_double);
			geom.rawdata = (char *)rawdata;
		}
	}
	return geom;
}

DEVICE_FUNCTION(pg_geometry_t)
pgfn_st_makepoint3(kern_context *kcxt,
				   pg_float8_t x, pg_float8_t y, pg_float8_t z)
{
	pg_geometry_t	geom;

	memset(&geom, 0, sizeof(pg_geometry_t));
	if (x.isnull || y.isnull || z.isnull)
		geom.isnull = true;
	else
	{
		cl_double  *rawdata = (cl_double *)
			kern_context_alloc(kcxt, 3 * sizeof(cl_double));

		if (!rawdata)
		{
			STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,
							   "out of memory");
			geom.isnull = true;
		}
		else
		{
			rawdata[0] = x.value;
			rawdata[1] = y.value;
			rawdata[2] = z.value;

			geom.type = GEOM_POINTTYPE;
			geom.flags = GEOM_FLAG__Z;
			geom.srid = SRID_UNKNOWN;
			geom.nitems = 1;
			geom.rawsize = 3 * sizeof(cl_double);
			geom.rawdata = (char *)rawdata;
		}
	}
	return geom;
}

DEVICE_FUNCTION(pg_geometry_t)
pgfn_st_makepoint4(kern_context *kcxt,
				   pg_float8_t x, pg_float8_t y,
				   pg_float8_t z, pg_float8_t m)
{
	pg_geometry_t	geom;

	memset(&geom, 0, sizeof(pg_geometry_t));
	if (x.isnull || y.isnull || z.isnull || m.isnull)
		geom.isnull = true;
	else
	{
		cl_double  *rawdata = (cl_double *)
			kern_context_alloc(kcxt, 4 * sizeof(cl_double));

		if (!rawdata)
		{
			STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,
							   "out of memory");
			geom.isnull = true;
		}
		else
		{
			rawdata[0] = x.value;
			rawdata[1] = y.value;
			rawdata[2] = z.value;
			rawdata[3] = m.value;

			geom.type = GEOM_POINTTYPE;
			geom.flags = GEOM_FLAG__Z | GEOM_FLAG__M;
			geom.srid = SRID_UNKNOWN;
			geom.nitems = 1;
			geom.rawsize = 2 * sizeof(cl_double);
			geom.rawdata = (char *)rawdata;
		}
	}
	return geom;
}

/* ================================================================
 *
 * Internal utility functions
 *
 * ================================================================
 */
#define PT_INSIDE		1
#define PT_BOUNDARY		0
#define PT_OUTSIDE		(-1)
#define PT_ERROR		9999

STATIC_INLINE(const char *)
__loadPoint2d(POINT2D *pt, const char *rawdata, cl_uint unitsz)
{
	memcpy(pt, rawdata, sizeof(POINT2D));
	return rawdata + unitsz;
}

STATIC_INLINE(const char *)
__loadPoint2dIndex(POINT2D *pt,
				   const char *rawdata, cl_uint unitsz, cl_uint index)
{
	rawdata += unitsz * index;
	memcpy(pt, rawdata, sizeof(POINT2D));
	return rawdata + unitsz;
}

STATIC_INLINE(int)
__geom_segment_side(const POINT2D *p1, const POINT2D *p2, const POINT2D *q)
{
	double	side = ((q->x - p1->x) * (p2->y - p1->y) -
					(p2->x - p1->x) * (q->y - p1->y));
	return (side < 0.0 ? -1 : (side > 0.0 ? 1 : 0));
}

STATIC_INLINE(cl_bool)
__geom_pt_in_arc(const POINT2D *P,
				 const POINT2D *A1,
				 const POINT2D *A2,
				 const POINT2D *A3)
{
	return __geom_segment_side(A1, A3, A2) == __geom_segment_side(A1, A3, P);
}

STATIC_INLINE(cl_bool)
__geom_pt_in_seg(const POINT2D *P,
				 const POINT2D *A1,
				 const POINT2D *A2)
{
	return ((A1->x <= P->x && P->x < A2->x) ||
			(A1->x >= P->x && P->x > A2->x) ||
			(A1->y <= P->y && P->y < A2->y) ||
			(A1->y >= P->y && P->y > A2->y));
}

STATIC_INLINE(cl_int)
__geom_pt_within_seg(const POINT2D *P,
					 const POINT2D *A1,
					 const POINT2D *A2)
{
	if (((A1->x <= P->x && P->x <= A2->x) ||
		 (A2->x <= P->x && P->x <= A1->x)) &&
		((A1->y <= P->y && P->y <= A2->y) ||
		 (A2->y <= P->y && P->y <= A1->y)))
	{
		if ((A1->x == P->x && A1->y == P->y) ||
			(A2->x == P->x && A2->y == P->y))
			return PT_BOUNDARY;
		return PT_INSIDE;
	}
	return PT_OUTSIDE;
}

/* ================================================================
 *
 * Basic Operators related to boundary-box
 *
 * ================================================================
 */
DEVICE_INLINE(cl_bool)
__geom_bbox_2d_is_empty(const geom_bbox_2d *bbox)
{
	return isnan(bbox->xmin);
}

/* see, gserialized_datum_get_box2df_p() */
STATIC_FUNCTION(cl_bool)
__geometry_get_bbox2d(kern_context *kcxt,
					  const pg_geometry_t *geom, geom_bbox_2d *bbox)
{
	POINT2D			pt;
	cl_uint			unitsz;
	const char	   *rawdata;
	double			xmin, xmax;
	double			ymin, ymax;
	pg_geometry_t	temp;
	const char	   *pos;

	/* geometry already has bounding-box? */
	if (geom->bbox)
	{
		memcpy(bbox, &geom->bbox->d2, sizeof(geom_bbox_2d));
		return true;
	}

	if ((geom->flags & GEOM_FLAG__GEODETIC) != 0)
		return false;
	if (geom->type == GEOM_POINTTYPE)
	{
		__loadPoint2d(&pt, geom->rawdata, 0);
		xmin = xmax = pt.x;
		ymin = ymax = pt.y;
	}
	else if (geom->type == GEOM_LINETYPE)
	{
		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(geom->flags);

		rawdata = __loadPoint2d(&pt, geom->rawdata, unitsz);
		xmin = xmax = pt.x;
		ymin = ymax = pt.y;
		for (int i = 1; i < geom->nitems; i++)
		{
			rawdata = __loadPoint2d(&pt, rawdata, unitsz);
			if (xmax < pt.x)
				xmax = pt.x;
			if (xmin > pt.x)
				xmin = pt.x;
			if (ymax < pt.y)
				ymax = pt.y;
			if (ymin > pt.y)
				ymin = pt.y;
		}
	}
	else if (geom->type == GEOM_MULTIPOINTTYPE)
	{
		pos = geometry_load_subitem(&temp, geom, NULL, 0, kcxt);
		if (!pos)
			return false;
		__loadPoint2d(&pt, temp.rawdata, 0);
		xmin = xmax = pt.x;
		ymin = ymax = pt.y;
		for (int i=1; i < geom->nitems; i++)
		{
			pos = geometry_load_subitem(&temp, geom, pos, i, kcxt);
			if (!pos)
				return false;
			__loadPoint2d(&pt, temp.rawdata, 0);
			if (xmax < pt.x)
				xmax = pt.x;
			if (xmin > pt.x)
				xmin = pt.x;
			if (ymax < pt.y)
				ymax = pt.y;
			if (ymin > pt.y)
				ymin = pt.y;
		}
	}
	else if (geom->type == GEOM_MULTILINETYPE)
	{
		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(geom->flags);

		pos = geometry_load_subitem(&temp, geom, NULL, 0, kcxt);
		if (!pos)
			return false;
		__loadPoint2d(&pt, temp.rawdata, 0);
		xmin = xmax = pt.x;
		ymin = ymax = pt.y;
		for (int i=1; i < geom->nitems; i++)
		{
			pos = geometry_load_subitem(&temp, geom, pos, i, kcxt);
			if (!pos)
				return false;
			rawdata = temp.rawdata;
			for (int j=0; j < temp.nitems; j++)
			{
				rawdata = __loadPoint2d(&pt, rawdata, unitsz);
				if (xmax < pt.x)
					xmax = pt.x;
				if (xmin > pt.x)
					xmin = pt.x;
				if (ymax < pt.y)
					ymax = pt.y;
				if (ymin > pt.y)
					ymin = pt.y;
			}
		}
	}
	else
	{
		return false;	/* not a supported type */
	}
	bbox->xmin = __double2float_rd(xmin);
	bbox->xmax = __double2float_ru(xmax);
	bbox->ymin = __double2float_rd(ymin);
	bbox->ymax = __double2float_ru(ymax);

	return true;
}

DEVICE_INLINE(cl_bool)
__geom_overlaps_bbox2d(const geom_bbox_2d *bbox1,
					   const geom_bbox_2d *bbox2)
{
	if (!__geom_bbox_2d_is_empty(bbox1) &&
		!__geom_bbox_2d_is_empty(bbox2) &&
		bbox1->xmin <= bbox2->xmax &&
		bbox1->xmax >= bbox2->xmin &&
		bbox1->ymin <= bbox2->ymax &&
		bbox1->ymax >= bbox2->ymin)
		return true;
	return false;
}

/* see, gserialized_overlaps_2d() */
DEVICE_FUNCTION(pg_bool_t)
pgfn_geometry_overlaps(kern_context *kcxt,
					   const pg_geometry_t &arg1,
					   const pg_geometry_t &arg2)
{
	pg_bool_t	result;
	geom_bbox_2d bbox1;
	geom_bbox_2d bbox2;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		/* see box2df_overlaps() */
		if (__geometry_get_bbox2d(kcxt, &arg1, &bbox1) &&
			__geometry_get_bbox2d(kcxt, &arg2, &bbox2))
			result.value = __geom_overlaps_bbox2d(&bbox1, &bbox2);
		else
			result.value = false;
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_box2df_geometry_overlaps(kern_context *kcxt,
							  const pg_box2df_t &arg1,
							  const pg_geometry_t &arg2)
{
	pg_bool_t	result;
	geom_bbox_2d bbox2;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		if (__geometry_get_bbox2d(kcxt, &arg2, &bbox2))
			result.value = __geom_overlaps_bbox2d(&arg1.value, &bbox2);
		else
			result.value = false;
	}
	return result;
}

DEVICE_INLINE(cl_bool)
__geom_contains_bbox2d(const geom_bbox_2d *bbox1,
					   const geom_bbox_2d *bbox2)
{
	if (!__geom_bbox_2d_is_empty(bbox1) &&
		__geom_bbox_2d_is_empty(bbox2))
		return true;
	if (bbox1->xmin <= bbox2->xmin &&
		bbox1->xmax >= bbox2->xmax &&
		bbox1->ymin <= bbox2->ymin &&
		bbox1->ymax >= bbox2->ymax)
		return true;

	return false;
}

/* see, gserialized_contains_2d() */
DEVICE_FUNCTION(pg_bool_t)
pgfn_geometry_contains(kern_context *kcxt,
					   const pg_geometry_t &arg1,
					   const pg_geometry_t &arg2)
{
	pg_bool_t	result;
	geom_bbox_2d bbox1;
	geom_bbox_2d bbox2;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		if (!__geometry_get_bbox2d(kcxt, &arg1, &bbox1) ||
			!__geometry_get_bbox2d(kcxt, &arg2, &bbox2))
			result.value = false;
		else
			result.value = __geom_contains_bbox2d(&bbox1, &bbox2);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_box2df_geometry_contains(kern_context *kcxt,
                              const pg_box2df_t &arg1,
                              const pg_geometry_t &arg2)
{
	pg_bool_t	result;
	geom_bbox_2d bbox2;

	result.isnull = (arg1.isnull | arg2.isnull);
    if (!result.isnull)
	{
		if (!__geometry_get_bbox2d(kcxt, &arg2, &bbox2))
			result.value = false;
		else
			result.value = __geom_contains_bbox2d(&arg1.value, &bbox2);
	}
	return result;
}

DEVICE_INLINE(cl_bool)
__geom_within_bbox2d(const geom_bbox_2d *bbox1,
					 const geom_bbox_2d *bbox2)
{
	if (__geom_bbox_2d_is_empty(bbox1) &&
		!__geom_bbox_2d_is_empty(bbox2))
		return true;
	if (bbox1->xmin >= bbox2->xmin &&
		bbox1->xmax <= bbox2->xmax &&
		bbox1->ymin >= bbox2->ymin &&
		bbox1->ymax <= bbox2->ymax)
		return true;
	return false;
}

/* see, gserialized_within_2d() */
DEVICE_FUNCTION(pg_bool_t)
pgfn_geometry_within(kern_context *kcxt,
					 const pg_geometry_t &arg1,
					 const pg_geometry_t &arg2)
{
	pg_bool_t	result;
	geom_bbox_2d bbox1;
	geom_bbox_2d bbox2;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		/* see box2df_within() */
		if (!__geometry_get_bbox2d(kcxt, &arg1, &bbox1) ||
			!__geometry_get_bbox2d(kcxt, &arg2, &bbox2))
			result.value = false;
		else
			result.value = __geom_within_bbox2d(&bbox1, &bbox2);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_box2df_geometry_within(kern_context *kcxt,
							const pg_box2df_t &arg1,
							const pg_geometry_t &arg2)
{
	pg_bool_t	result;
	geom_bbox_2d bbox2;

	result.isnull = (arg1.isnull | arg2.isnull);
    if (!result.isnull)
    {
		if (!__geometry_get_bbox2d(kcxt, &arg2, &bbox2))
			result.value = false;
		else
			result.value = __geom_within_bbox2d(&arg1.value, &bbox2);
	}
	return result;
}

/* see, LWGEOM_expand() */
DEVICE_FUNCTION(pg_geometry_t)
pgfn_st_expand(kern_context *kcxt,
			   const pg_geometry_t &arg1, pg_float8_t arg2)
{
	pg_geometry_t	geom;
	geom_bbox_2d	bbox;
	char		   *pos;

	memset(&geom, 0, sizeof(pg_geometry_t));
	if (arg1.isnull || arg2.isnull)
	{
		geom.isnull = true;
		return geom;
	}

	/* cannot expand an empty */
	if (arg1.nitems == 0)
		return arg1;
	/* cannot expand something with no boundary-box */
	if (!__geometry_get_bbox2d(kcxt, &arg1, &bbox))
		return arg1;
	/* expand bbox */
	bbox.xmin -= arg2.value;
	bbox.xmax += arg2.value;
	bbox.ymin -= arg2.value;
	bbox.ymax += arg2.value;

	pos = (char *)
		kern_context_alloc(kcxt, (sizeof(geom_bbox_2d) +	/* bounding box */
								  2 * sizeof(cl_uint) +		/* nitems + padding */
								  10 * sizeof(double)));	/* 5 x Points */
	if (!pos)
	{
		STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY,
					  "out of memory");
		geom.isnull = true;
		return geom;
	}
	geom.type = GEOM_POLYGONTYPE;
	geom.flags = GEOM_FLAG__BBOX;
	geom.srid = SRID_UNKNOWN;
	geom.nitems = 1;
	geom.rawsize = (2 * sizeof(cl_uint) + 10 * sizeof(double));
	geom.bbox = (geom_bbox *)pos;
	memcpy(&geom.bbox->d2, &bbox, sizeof(geom_bbox_2d));
	pos += sizeof(geom_bbox_2d);

	geom.rawdata = pos;
	((cl_uint *)pos)[0] = 5;		/* # of Points */
	((cl_uint *)pos)[1] = 0;		/* padding */
	pos += 2 * sizeof(cl_uint);

	((double *)pos)[0] = bbox.xmin;
	((double *)pos)[1] = bbox.ymin;
	((double *)pos)[2] = bbox.xmin;
	((double *)pos)[3] = bbox.ymax;
	((double *)pos)[4] = bbox.xmax;
	((double *)pos)[5] = bbox.ymax;
	((double *)pos)[6] = bbox.xmax;
	((double *)pos)[7] = bbox.ymin;
	((double *)pos)[8] = bbox.xmin;
	((double *)pos)[9] = bbox.ymin;
	pos += 10 * sizeof(double);

	return geom;
}

/* ================================================================
 *
 * GiST Index Handlers
 *
 * ================================================================
 */
DEVICE_FUNCTION(cl_bool)
pgindex_gist_geometry_overlap(kern_context *kcxt,
							  PageHeaderData *i_page,
							  const pg_box2df_t &i_var,
							  const pg_geometry_t &i_arg)
{
	geom_bbox_2d __bbox;

	if (!i_var.isnull &&
		!i_arg.isnull && __geometry_get_bbox2d(kcxt, &i_arg, &__bbox))
		return __geom_overlaps_bbox2d(&i_var.value, &__bbox);
	return false;
}

DEVICE_FUNCTION(cl_bool)
pgindex_gist_box2df_overlap(kern_context *kcxt,
							PageHeaderData *i_page,
							const pg_box2df_t &i_var,
							const pg_box2df_t &i_arg)
{
	if (!i_var.isnull && !i_arg.isnull)
		return __geom_overlaps_bbox2d(&i_var.value, &i_arg.value);
	return false;
}

DEVICE_FUNCTION(cl_bool)
pgindex_gist_geometry_contains(kern_context *kcxt,
							   PageHeaderData *i_page,
							   const pg_box2df_t &i_var,
							   const pg_geometry_t &i_arg)
{
	geom_bbox_2d __bbox;

	if (!i_var.isnull &&
		!i_arg.isnull && __geometry_get_bbox2d(kcxt, &i_arg, &__bbox))
		return __geom_contains_bbox2d(&i_var.value, &__bbox);
	return false;
}

DEVICE_FUNCTION(cl_bool)
pgindex_gist_box2df_contains(kern_context *kcxt,
							 PageHeaderData *i_page,
							 const pg_box2df_t &i_var,
							 const pg_box2df_t &i_arg)
{
	if (!i_var.isnull && !i_arg.isnull)
		return __geom_contains_bbox2d(&i_var.value, &i_arg.value);
	return false;
}

DEVICE_FUNCTION(cl_bool)
pgindex_gist_geometry_contained(kern_context *kcxt,
								PageHeaderData *i_page,
								const pg_box2df_t &i_var,
								const pg_geometry_t &i_arg)
{
	geom_bbox_2d __bbox;

	if (!i_var.isnull &&
		!i_arg.isnull && __geometry_get_bbox2d(kcxt, &i_arg, &__bbox))
	{
		if (!GistPageIsLeaf(i_page))
			return __geom_overlaps_bbox2d(&i_var.value, &__bbox);
		else
			return __geom_within_bbox2d(&i_var.value, &__bbox);
	}
	return false;
}

DEVICE_FUNCTION(cl_bool)
pgindex_gist_box2df_contained(kern_context *kcxt,
							  PageHeaderData *i_page,
							  const pg_box2df_t &i_var,
							  const pg_box2df_t &i_arg)
{
	if (!i_var.isnull && !i_arg.isnull)
	{
		if (!GistPageIsLeaf(i_page))
			return __geom_overlaps_bbox2d(&i_var.value, &i_arg.value);
		else
			return __geom_within_bbox2d(&i_var.value, &i_arg.value);
	}
	return false;
}

/* ================================================================
 *
 * St_Distance(geometry,geometry)
 *
 * ================================================================
 */

/*
 * state object used in distance-calculations
 */
typedef struct
{
	kern_context *kcxt;		/* for error reporting */
	double		distance;
	POINT2D		p1;
	POINT2D		p2;
	int			twisted;	/* to preserve the order of incoming points */
	double		tolerance;	/* the tolerance for dwithin and dfullywithin */
} DISTPTS;

STATIC_FUNCTION(int)
__geom_contains_point(const pg_geometry_t *ring,
					  const POINT2D *pt,
					  kern_context *kcxt)
{
	/* see ptarray_contains_point_partial */
	cl_uint		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(ring->flags);
	POINT2D		seg1, seg2;
	int			side, wn = 0;

	__loadPoint2d(&seg1, ring->rawdata, unitsz);
	__loadPoint2d(&seg2, ring->rawdata + unitsz * (ring->nitems-1), unitsz);
	if (!FP_EQUALS(seg1.x, seg2.x) || !FP_EQUALS(seg1.y, seg2.y))
	{
		STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
					  "__geom_contains_point called on unclosed ring");
		return PT_ERROR;
	}

	for (int i=1; i < ring->nitems; i++, seg1 = seg2)
	{
		__loadPoint2d(&seg2, ring->rawdata + i * unitsz, unitsz);
		/* zero length segments are ignored. */
		if (seg1.x == seg2.x && seg1.y == seg2.y)
			continue;

		/* only test segments in our vertical range */
		if (pt->y > Max(seg1.y, seg2.y) ||
			pt->y < Min(seg1.y, seg2.y))
			continue;

		side = __geom_segment_side(&seg1, &seg2, pt);
		if (side == 0 && __geom_pt_in_seg(pt, &seg1, &seg2))
			return PT_BOUNDARY;

		if (side < 0 && seg1.y <= pt->y && pt->y < seg2.y)
		{
			/*
			 * If the point is to the left of the line, and it's rising,
			 * then the line is to the right of the point and
			 * circling counter-clockwise, so increment.
			 */
			wn++;
		}
		else if (side > 0 && seg2.y <= pt->y && pt->y < seg1.y)
		{
			/*
			 * If the point is to the right of the line, and it's falling,
			 * then the line is to the right of the point and circling
			 * clockwise, so decrement.
			 */
			wn--;
		}
	}
	return (wn == 0 ? PT_OUTSIDE : PT_INSIDE);
}

STATIC_INLINE(cl_bool)
__geom_dist2d_pt_pt(const POINT2D *p1,
					const POINT2D *p2,
					DISTPTS *dl)
{
	/* see lw_dist2d_pt_pt */
	double	dist = hypot(p1->x - p2->x, p1->y - p2->y);

	if (dl->distance > dist)
	{
		dl->distance = dist;
		if (dl->twisted > 0)
		{
			dl->p1 = *p1;
			dl->p2 = *p2;
		}
		else
		{
			dl->p1 = *p2;
			dl->p2 = *p1;
		}
	}
	return true;
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_pt_seg(const POINT2D *pt,
					 const POINT2D *A,
					 const POINT2D *B,
					 DISTPTS *dl)
{
	/* see lw_dist2d_pt_seg */
	POINT2D		c;
	double		r;

	/* if start==end, then use pt distance  */
	if (FP_EQUALS(A->x, B->x) && FP_EQUALS(A->y, B->y))
		return __geom_dist2d_pt_pt(pt, A, dl);

	/*
	 * otherwise, we use comp.graphics.algorithms
	 * Frequently Asked Questions method
	 *
	 *  (1)        AC dot AB
	 *         r = ---------
	 *              ||AB||^2
	 *  r has the following meaning:
	 *  r=0 P = A
	 *  r=1 P = B
	 *  r<0 P is on the backward extension of AB
	 *  r>1 P is on the forward extension of AB
	 *  0<r<1 P is interior to AB
	 */
	r = ((pt->x - A->x) * (B->x - A->x) + (pt->y - A->y) * (B->y - A->y)) /
		((B->x - A->x) * (B->x - A->x) + (B->y - A->y) * (B->y - A->y));

	/* If p projected on the line is outside point A */
	if (r < 0.0)
		return __geom_dist2d_pt_pt(pt, A, dl);
	/* If p projected on the line is outside point B or on point B */
	if (r >= 1.0)
		return __geom_dist2d_pt_pt(pt, B, dl);
	/*
	 * If the point pt is on the segment this is a more robust way
	 * to find out that
	 */
	if ((A->y - pt->y) * (B->x - A->x) == (A->x - pt->x) * (B->y - A->y))
	{
		dl->distance = 0.0;
		dl->p1 = *pt;
		dl->p2 = *pt;
	}
	/*
	 * If the projection of point p on the segment is between A and B
	 * then we find that "point on segment" and send it to __geom_dist2d_pt_pt
	 */
	c.x = A->x + r * (B->x - A->x);
	c.y = A->y + r * (B->y - A->y);

	return __geom_dist2d_pt_pt(pt, &c, dl);
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_seg_seg(const POINT2D *A,
					  const POINT2D *B,
					  const POINT2D *C,
					  const POINT2D *D,
					  DISTPTS *dl)
{
	/* see, lw_dist2d_seg_seg */
	double	s_top, s_bot, s;
	double	r_top, r_bot, r;

	/* A and B are the same point */
	if (A->x == B->x && A->y == B->y)
		return __geom_dist2d_pt_seg(A, C, D, dl);
	/* C and D are the same point */
	if (C->x == D->x && C->y == D->y)
		return __geom_dist2d_pt_seg(D, A, B, dl);

	/* AB and CD are line segments
	 * from comp.graphics.algo
	 *
	 * Solving the above for r and s yields
	 *           (Ay-Cy)(Dx-Cx)-(Ax-Cx)(Dy-Cy)
	 *      r = ----------------------------- (eqn 1)
	 *           (Bx-Ax)(Dy-Cy)-(By-Ay)(Dx-Cx)
	 *
	 *           (Ay-Cy)(Bx-Ax)-(Ax-Cx)(By-Ay)
	 *      s = ----------------------------- (eqn 2)
	 *           (Bx-Ax)(Dy-Cy)-(By-Ay)(Dx-Cx)
	 * Let P be the position vector of the intersection point, then
	 *     P=A+r(B-A) or
	 *     Px=Ax+r(Bx-Ax)
	 *     Py=Ay+r(By-Ay)
	 *
	 * By examining the values of r & s, you can also determine
	 * some other limiting conditions:
	 * If 0<=r<=1 & 0<=s<=1, intersection exists
	 * r<0 or r>1 or s<0 or s>1 line segments do not intersect
	 * If the denominator in eqn 1 is zero, AB & CD are parallel
	 * If the numerator in eqn 1 is also zero, AB & CD are collinear.
	 */
	r_top = (A->y - C->y) * (D->x - C->x) - (A->x - C->x) * (D->y - C->y);
	r_bot = (B->x - A->x) * (D->y - C->y) - (B->y - A->y) * (D->x - C->x);

	s_top = (A->y - C->y) * (B->x - A->x) - (A->x - C->x) * (B->y - A->y);
	s_bot = (B->x - A->x) * (D->y - C->y) - (B->y - A->y) * (D->x - C->x);

	if (r_bot == 0 || s_bot == 0)
	{
		if (__geom_dist2d_pt_seg(A, C, D, dl) &&
			__geom_dist2d_pt_seg(B, C, D, dl))
		{
			dl->twisted *= -1;	/* the order was changed */
			return (__geom_dist2d_pt_seg(C, A, B, dl) &&
					__geom_dist2d_pt_seg(D, A, D, dl));
		}
		return false;
	}

	s = s_top / s_bot;
	r = r_top / r_bot;
	if ((r < 0) || (r > 1) || (s < 0) || (s > 1))
	{
		if (__geom_dist2d_pt_seg(A, C, D, dl) &&
			__geom_dist2d_pt_seg(B, C, D, dl))
		{
			dl->twisted *= -1;	/* the order was changed */
			return (__geom_dist2d_pt_seg(C, A, B, dl) &&
					__geom_dist2d_pt_seg(D, A, B, dl));
		}
		return false;
	}
	else
	{
		/* If there is intersection we identify the intersection point */
		POINT2D		P;

		if ((A->x == C->x && A->y == C->y) ||
			(A->x == D->x && A->y == D->y))
		{
			P.x = A->x;
			P.y = A->y;
		}
		else if ((B->x == C->x && B->y == C->y) ||
				 (B->x == D->x && B->y == D->y))
		{
			P.x = B->x;
			P.y = B->y;
		}
		else
		{
			P.x = A->x + r * (B->x - A->x);
			P.y = A->y + r * (B->y - A->y);
		}
		dl->distance = 0.0;
		dl->p1 = P;
		dl->p2 = P;
	}
	return true;
}

STATIC_FUNCTION(double)
__geom_arc_center(const POINT2D *p1,
				  const POINT2D *p2,
				  const POINT2D *p3,
				  POINT2D *result)
{
	/* see, lw_arc_center */
	POINT2D		c;
	double		dx21, dy21, dx31, dy31, h21, h31, d;

	c.x = c.y = 0.0;

	/* closed circle? */
	if (FP_EQUALS(p1->x, p3->x) && FP_EQUALS(p1->y, p3->y))
	{
		c.x = p1->x + (p2->x - p1->x) / 2.0;
		c.y = p1->y + (p2->y - p1->y) / 2.0;
		*result = c;
		return hypot(c.x - p1->x, c.y - p1->y);
    }

	/*
	 * Using cartesian eguations from the page
	 * https://en.wikipedia.org/wiki/Circumscribed_circle
	 */
	dx21 = p2->x - p1->x;
	dy21 = p2->y - p1->y;
	dx31 = p3->x - p1->x;
	dy31 = p3->y - p1->y;

	h21 = pow(dx21, 2.0) + pow(dy21, 2.0);
	h31 = pow(dx31, 2.0) + pow(dy31, 2.0);

	 /*
	  * 2 * |Cross product|, d<0 means clockwise
	  * and d>0 counterclockwise sweeping angle
	  */
	d = 2 * (dx21 * dy31 - dx31 * dy21);

	/* Check colinearity, |Cross product| = 0 */
	if (fabs(d) < FP_TOLERANCE)
		return -1.0;

	/* Calculate centroid coordinates and radius */
	c.x = p1->x + (h21 * dy31 - h31 * dy21) / d;
	c.y = p1->y - (h21 * dx31 - h31 * dx21) / d;
	*result = c;
	return hypot(c.x - p1->x, c.y - p1->y);
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_pt_arc(const POINT2D *P,
					 const POINT2D *A1,
					 const POINT2D *A2,
					 const POINT2D *A3,
					 DISTPTS *dl)
{
	/* see, lw_dist2d_pt_arc */
	POINT2D		C;	/* center of circle defined by arc A */
	POINT2D		X;	/* point circle(A) where line from C to P crosses */
	double		radius_A, d;

	/* What if the arc is a point? */
	if (A1->x == A2->x && A2->x == A3->x &&
		A1->y == A2->y && A2->y == A3->y )
		return __geom_dist2d_pt_pt(P, A1, dl);

	/* Calculate centers and radii of circles. */
	radius_A = __geom_arc_center(A1, A2, A3, &C);

	/* This "arc" is actually a line (A2 is colinear with A1,A3) */
	if (radius_A < 0.0)
		return __geom_dist2d_pt_seg(P, A1, A3, dl);

	/* Distance from point to center */
	d = hypot(C.x - P->x, C.y - P->y);

	/* P is the center of the circle */
	if (FP_EQUALS(d, 0.0))
	{
		dl->distance = radius_A;
		dl->p1 = *A1;
		dl->p2 = *P;
		return true;
	}

	/* X is the point on the circle where the line from P to C crosses */
	X.x = C.x + (P->x - C.x) * radius_A / d;
	X.y = C.y + (P->y - C.y) * radius_A / d;

	/* Is crossing point inside the arc? Or arc is actually circle? */
	if ((FP_EQUALS(A1->x, A3->x) &&
		 FP_EQUALS(A1->y, A3->y)) || __geom_pt_in_arc(&X, A1, A2, A3))
	{
		__geom_dist2d_pt_pt(P, &X, dl);
	}
	else
	{
		/* Distance is the minimum of the distances to the arc end points */
		__geom_dist2d_pt_pt(A1, P, dl);
		__geom_dist2d_pt_pt(A3, P, dl);
	}
	return true;
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_seg_arc(const POINT2D *A1,
					  const POINT2D *A2,
					  const POINT2D *B1,
					  const POINT2D *B2,
					  const POINT2D *B3,
					  DISTPTS *dl)
{
	/* see lw_dist2d_seg_arc */
	POINT2D		C;			/* center of arc circle */
	POINT2D		D;			/* point on A closest to C */
	double		radius_C;	/* radius of arc circle */
	double		dist_C_D;	/* distance from C to D */
	DISTPTS		dltemp;
	cl_bool		pt_in_arc;
	cl_bool		pt_in_seg;

	/* What if the "arc" is a point? */
	if (B1->x == B2->x && B1->y == B2->y &&
		B2->x == B3->x && B2->y == B3->y)
		return __geom_dist2d_pt_seg(B1, A1, A2, dl);

	/* Calculate center and radius of the circle. */
	radius_C = __geom_arc_center(B1, B2, B3, &C);

	/* This "arc" is actually a line (B2 is collinear with B1,B3) */
	if (radius_C < 0.0)
		return __geom_dist2d_seg_seg(A1, A2, B1, B3, dl);

	/* Calculate distance between the line and circle center */
	memset(&dltemp, 0, sizeof(DISTPTS));
	dltemp.distance = DBL_MAX;
	dltemp.twisted = -1;
	if (!__geom_dist2d_pt_seg(&C, A1, A2, &dltemp))
		return false;
	D = dltemp.p1;
	dist_C_D = dltemp.distance;

	if (dist_C_D < radius_C)
	{
		POINT2D	E, F;	/* points of intersection of edge A and circle(B) */
		double	length_A;	/* length of the segment A */
		double	dist_D_EF;	/* distance from D to E or F */

		dist_D_EF = sqrt(radius_C * radius_C - dist_C_D * dist_C_D);
		length_A = hypot(A2->x - A1->x, A2->y - A1->y);

		/* Point of intersection E */
		E.x = D.x - (A2->x - A1->x) * dist_D_EF / length_A;
		E.y = D.y - (A2->y - A1->y) * dist_D_EF / length_A;
		/* Point of intersection F */
		F.x = D.x + (A2->x - A1->x) * dist_D_EF / length_A;
		F.y = D.y + (A2->y - A1->y) * dist_D_EF / length_A;

		/* If E is within A and within B then it's an intersection point */
		pt_in_arc = __geom_pt_in_arc(&E, B1, B2, B3);
		pt_in_seg = __geom_pt_in_seg(&E, A1, A2);
		if (pt_in_arc && pt_in_seg)
		{
			dl->distance = 0.0;
			dl->p1 = E;
			dl->p2 = E;
			return true;
		}

		 /* If F is within A and within B then it's an intersection point */
		pt_in_arc = __geom_pt_in_arc(&F, B1, B2, B3);
		pt_in_seg = __geom_pt_in_seg(&F, A1, A2);
		if (pt_in_arc && pt_in_seg)
		{
			dl->distance = 0.0;
			dl->p1 = F;
			dl->p2 = F;
			return true;
		}
	}
	else if (dist_C_D == radius_C)
	{
		/* Is D contained in both A and B? */
		pt_in_arc = __geom_pt_in_arc(&D, B1, B2, B3);
		pt_in_seg = __geom_pt_in_seg(&D, A1, A2);
		if (pt_in_arc && pt_in_seg)
		{
			dl->distance = 0.0;
			dl->p1 = D;
			dl->p2 = D;
			return true;
		}
	}
	else
	{
		POINT2D		G;	/* Point on circle closest to A */

		G.x = C.x + (D.x - C.x) * radius_C / dist_C_D;
		G.y = C.y + (D.y - C.y) * radius_C / dist_C_D;

		pt_in_arc = __geom_pt_in_arc(&G, B1, B2, B3);
		pt_in_seg = __geom_pt_in_seg(&D, A1, A2);
		if (pt_in_arc && pt_in_seg)
			return __geom_dist2d_pt_pt(&D, &G, dl);
	}

	if (pt_in_arc && !pt_in_seg)
	{
		/* closest point is in the arc, but not in the segment,
		 * so one of the segment end points must be the closest.
		 */
		__geom_dist2d_pt_arc(A1, B1, B2, B3, dl);
		__geom_dist2d_pt_arc(A2, B1, B2, B3, dl);
		return true;
	}

	if (pt_in_seg && !pt_in_arc)
	{
		/* or, one of the arc end points is the closest */
		__geom_dist2d_pt_seg(B1, A1, A2, dl);
		__geom_dist2d_pt_seg(B3, A1, A2, dl);
		return true;
	}
	/* finally, one of the end-point to end-point combos is the closest. */
	__geom_dist2d_pt_pt(A1, B1, dl);
	__geom_dist2d_pt_pt(A1, B3, dl);
	__geom_dist2d_pt_pt(A2, B1, dl);
	__geom_dist2d_pt_pt(A2, B3, dl);
	return true;
}

STATIC_INLINE(double)
distance2d_sqr_pt_pt(const POINT2D *p1,
					 const POINT2D *p2)
{
	double	hside = p2->x - p1->x;
	double	vside = p2->y - p1->y;

	return hside * hside + vside * vside;
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_arc_arc_concentric(const POINT2D *A1,
								 const POINT2D *A2,
								 const POINT2D *A3,
								 double radius_A,
								 const POINT2D *B1,
								 const POINT2D *B2,
								 const POINT2D *B3,
								 double radius_B,
								 const POINT2D *CENTER,
								 DISTPTS *dl)
{
	int			seg_side;
	double		dist_sqr;
	double		shortest_sqr;
	const POINT2D *P1;
	const POINT2D *P2;

	if (radius_A == radius_B)
    {
		/* Check if B1 or B3 are in the same side as A2 in the A1-A3 arc */
		seg_side = __geom_segment_side(A1, A3, A2);
		if (seg_side == __geom_segment_side(A1, A3, B1))
		{
			dl->distance = 0;
			dl->p1 = *B1;
			dl->p2 = *B1;
			return true;
		}
		if (seg_side == __geom_segment_side(A1, A3, B3))
		{
			dl->distance = 0;
			dl->p1 = *B3;
			dl->p2 = *B3;
			return true;
		}
		/* Check if A1 or A3 are in the same side as B2 in the B1-B3 arc */
		seg_side = __geom_segment_side(B1, B3, B2);
		if (seg_side == __geom_segment_side(B1, B3, A1))
		{
			dl->distance = 0;
			dl->p1 = *A1;
			dl->p2 = *A1;
			return true;
		}
		if (seg_side == __geom_segment_side(B1, B3, A3))
		{
			dl->distance = 0;
			dl->p1 = *A3;
			dl->p2 = *A3;
			return true;
		}
	}
	else
	{
		/* Check if any projection of B ends are in A*/
		POINT2D		proj;

		seg_side = __geom_segment_side(A1, A3, A2);
		/* B1 */
        proj.x = CENTER->x + (B1->x - CENTER->x) * radius_A / radius_B;
        proj.y = CENTER->y + (B1->y - CENTER->y) * radius_A / radius_B;
		if (seg_side == __geom_segment_side(A1, A3, &proj))
		{
			dl->distance = fabs(radius_A - radius_B);
			dl->p1 = proj;
			dl->p2 = *B1;
			return true;
		}
		/* B3 */
		proj.x = CENTER->x + (B3->x - CENTER->x) * radius_A / radius_B;
		proj.y = CENTER->y + (B3->y - CENTER->y) * radius_A / radius_B;
		if (seg_side == __geom_segment_side(A1, A3, &proj))
		{
			dl->distance = fabs(radius_A - radius_B);
			dl->p1 = proj;
			dl->p2 = *B3;
			return true;
		}

		/* Now check projections of A in B */
		seg_side = __geom_segment_side(B1, B3, B2);
		/* A1 */
		proj.x = CENTER->x + (A1->x - CENTER->x) * radius_B / radius_A;
		proj.y = CENTER->y + (A1->y - CENTER->y) * radius_B / radius_A;
		if (seg_side == __geom_segment_side(B1, B3, &proj))
		{
			dl->distance = fabs(radius_A - radius_B);
			dl->p1 = proj;
			dl->p2 = *A1;
			return true;
		}

		/* A3 */
		proj.x = CENTER->x + (A3->x - CENTER->x) * radius_B / radius_A;
		proj.y = CENTER->y + (A3->y - CENTER->y) * radius_B / radius_A;
		if (seg_side == __geom_segment_side(B1, B3, &proj))
		{
			dl->distance = fabs(radius_A - radius_B);
			dl->p1 = proj;
			dl->p2 = *A3;
			return true;
		}
	}

	/* check the shortest between the distances of the 4 ends */
	shortest_sqr = dist_sqr = distance2d_sqr_pt_pt(A1, B1);
	P1 = A1;
	P2 = B1;

	dist_sqr = distance2d_sqr_pt_pt(A1, B3);
	if (dist_sqr < shortest_sqr)
	{
		shortest_sqr = dist_sqr;
		P1 = A1;
		P2 = B3;
	}

	dist_sqr = distance2d_sqr_pt_pt(A3, B1);
	if (dist_sqr < shortest_sqr)
	{
		shortest_sqr = dist_sqr;
		P1 = A3;
		P2 = B1;
	}

	dist_sqr = distance2d_sqr_pt_pt(A3, B3);
	if (dist_sqr < shortest_sqr)
	{
		shortest_sqr = dist_sqr;
		P1 = A3;
		P2 = B3;
	}
	dl->distance = sqrt(shortest_sqr);
	dl->p1 = *P1;
	dl->p2 = *P2;

	return true;
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_arc_arc(const POINT2D *A1,
					  const POINT2D *A2,
					  const POINT2D *A3,
					  const POINT2D *B1,
					  const POINT2D *B2,
					  const POINT2D *B3,
					  DISTPTS *dl)
{
	/* lw_dist2d_arc_arc */
	POINT2D		CA, CB;		/* Center points of arcs A and B */
	double		radius_A;	/* Radii of arcs A and B */
	double		radius_B;
	double		d;
	cl_bool		pt_in_arc_A;
	cl_bool		pt_in_arc_B;

	/* What if one or both of our "arcs" is actually a point? */
	if (B1->x == B2->x && B2->x == B3->x &&
		B1->y == B2->y && B2->y == B3->y)
	{
		if (A1->x == A2->x && A2->x == A3->x &&
			A1->y == A2->y && A2->y == A3->y)
			return __geom_dist2d_pt_pt(B1, A1, dl);
		else
			return __geom_dist2d_pt_arc(B1, A1, A2, A3, dl);
	}
	else if (A1->x == A2->x && A2->x == A3->x &&
			 A1->y == A2->y && A2->y == A3->y)
		return __geom_dist2d_pt_arc(A1, B1, B2, B3, dl);

	/*c alculate centers and radii of circles. */
	radius_A = __geom_arc_center(A1, A2, A3, &CA);
	radius_B = __geom_arc_center(B1, B2, B3, &CB);

	/* two co-linear arcs?!? That's two segments. */
	if (radius_A < 0 && radius_B < 0)
		return __geom_dist2d_seg_seg(A1, A3, B1, B3, dl);
	/* A is co-linear, delegate to lw_dist_seg_arc here. */
	if (radius_A < 0)
		return __geom_dist2d_seg_arc(A1, A3, B1, B2, B3, dl);
	/* B is co-linear, delegate to lw_dist_seg_arc here. */
	if (radius_B < 0)
		return __geom_dist2d_seg_arc(B1, B3, A1, A2, A3, dl);
	/* center-center distance */
	d = hypot(CA.x - CB.x, CA.y - CB.y);

	/* concentric arcs */
	if (FP_EQUALS(d, 0.0))
		return __geom_dist2d_arc_arc_concentric(A1, A2, A3, radius_A,
												B1, B2, B3, radius_B,
												&CA, dl);
	/* make sure that arc "A" has the bigger radius */
	if (radius_B > radius_A)
	{
		__swap(A1, B1);
		__swap(A2, B2);
		__swap(A3, B3);
		__swap(CA, CB);
		__swap(radius_A, radius_B);
	}

	/* circles touch at a point. Is that point within the arcs? */
	if (d == (radius_A + radius_B))
	{
		POINT2D		D;			/* Mid-point between the centers CA and CB */

		D.x = CA.x + (CB.x - CA.x) * radius_A / d;
		D.y = CA.y + (CB.y - CA.y) * radius_A / d;
		pt_in_arc_A = __geom_pt_in_arc(&D, A1, A2, A3);
		pt_in_arc_B = __geom_pt_in_arc(&D, B1, B2, B3);
		if (pt_in_arc_A && pt_in_arc_B)
		{
			dl->distance = 0.0;
			dl->p1 = D;
			dl->p2 = D;
			return true;
		}
	}
	else if (d > (radius_A + radius_B) ||	/* Disjoint */
			 d < (radius_A - radius_B))		/* Contained */
	{
		/* Points where the line from CA to CB cross their circle bounds */
		POINT2D		XA, XB;

		XA.x = CA.x + (CB.x - CA.x) * radius_A / d;
		XA.y = CA.y + (CB.y - CA.y) * radius_A / d;
		XB.x = CB.x + (CA.x - CB.x) * radius_B / d;
		XB.y = CB.y + (CA.y - CB.y) * radius_B / d;

		pt_in_arc_A = __geom_pt_in_arc(&XA, A1, A2, A3);
		pt_in_arc_B = __geom_pt_in_arc(&XB, B1, B2, B3);
		if (pt_in_arc_A && pt_in_arc_B)
			return __geom_dist2d_pt_pt(&XA, &XB, dl);
	}
	else if (d < (radius_A + radius_B))		/* crosses at two points */
	{
		POINT2D		D, E, F;
		double		a = (radius_A * radius_A -
						 radius_B * radius_B + d * d) / (2 * d);
		double		h = sqrt(radius_A * radius_A - a * a);

		/* Location of D */
		D.x = CA.x + (CB.x - CA.x) * a / d;
		D.y = CA.y + (CB.y - CA.y) * a / d;

		/* Start from D and project h units perpendicular to CA-D to get E */
		E.x = D.x + (D.y - CA.y) * h / a;
		E.y = D.y + (D.x - CA.x) * h / a;

		/* Crossing point E contained in arcs? */
		pt_in_arc_A = __geom_pt_in_arc(&E, A1, A2, A3);
		pt_in_arc_B = __geom_pt_in_arc(&E, B1, B2, B3);
		if (pt_in_arc_A && pt_in_arc_B)
		{
			dl->distance = 0.0;
			dl->p1 = E;
			dl->p2 = E;
			return true;
		}

		/* Start from D and project h units perpendicular to CA-D to get F */
		F.x = D.x - (D.y - CA.y) * h / a;
		F.y = D.y - (D.x - CA.x) * h / a;

		/* Crossing point F contained in arcs? */
		pt_in_arc_A = __geom_pt_in_arc(&F, A1, A2, A3);
		pt_in_arc_B = __geom_pt_in_arc(&F, B1, B2, B3);
		if (pt_in_arc_A && pt_in_arc_B)
		{
			dl->distance = 0.0;
			dl->p1 = F;
			dl->p2 = F;
			return true;
		}
	}
	else
	{
		STROM_ELOG(dl->kcxt, "arcs neither touch, intersect nor disjoint");
		return false;
	}

	if (pt_in_arc_A && !pt_in_arc_B)
	{
		/*
		 * closest point is in the arc A, but not in the arc B,
		 * so one of the B end points must be the closest.
		 */
		__geom_dist2d_pt_arc(B1, A1, A2, A3, dl);
		__geom_dist2d_pt_arc(B3, A1, A2, A3, dl);
	}
	else if (!pt_in_arc_A && pt_in_arc_B)
	{
		/*
		 * closest point is in the arc B, but not in the arc A,
		 * so one of the A end points must be the closest.
		 */
		__geom_dist2d_pt_arc(A1, B1, B2, B3, dl);
		__geom_dist2d_pt_arc(A3, B1, B2, B3, dl);
	}
	else
	{
		/* finally, one of the end-point to end-point combos is the closest. */
		__geom_dist2d_pt_pt(A1, B1, dl);
        __geom_dist2d_pt_pt(A1, B3, dl);
        __geom_dist2d_pt_pt(A3, B1, dl);
        __geom_dist2d_pt_pt(A3, B3, dl);
	}
	return true;
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_pt_ptarray(const POINT2D *pt,
						 const pg_geometry_t *geom,
						 DISTPTS *dl)
{
	size_t		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(geom->flags);
	const char *rawdata = geom->rawdata;
	POINT2D		start, end;

	if (geom->nitems == 0)
		return false;
	rawdata = __loadPoint2d(&start, rawdata, unitsz);

	if (!__geom_dist2d_pt_pt(pt, &start, dl))
		return false;
	for (int i = 1; i < geom->nitems; i++)
	{
		rawdata = __loadPoint2d(&end, rawdata, unitsz);

		if (!__geom_dist2d_pt_seg(pt, &start, &end, dl))
			return false;
		/* just a check if the answer is already given */
		if (dl->distance <= dl->tolerance)
			return true;
		start = end;
	}
	return true;
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_ptarray_ptarray(const pg_geometry_t *geom1,
							  const pg_geometry_t *geom2,
							  DISTPTS *dl)
{
	/* see lw_dist2d_ptarray_ptarray */
	cl_uint		unitsz1 = sizeof(double) * GEOM_FLAGS_NDIMS(geom1->flags);
	cl_uint		unitsz2 = sizeof(double) * GEOM_FLAGS_NDIMS(geom2->flags);
	const char *rawdata1;
	const char *rawdata2;
	POINT2D		start, end;
	POINT2D		start2, end2;

	rawdata1 = __loadPoint2d(&start, geom1->rawdata, unitsz1);
	for (int i=1; i < geom1->nitems; i++, start=end)
	{
		rawdata1 = __loadPoint2d(&end, rawdata1, unitsz1);
		rawdata2 = __loadPoint2d(&start2, geom2->rawdata, unitsz2);
		for (int j=1; j < geom2->nitems; j++, start2=end2)
		{
			rawdata2 = __loadPoint2d(&end2, rawdata2, unitsz2);
			__geom_dist2d_seg_seg(&start, &end, &start2, &end2, dl);
		}
	}
	return true;
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_pt_ptarrayarc(const POINT2D *pt,
							const pg_geometry_t *geom,
							DISTPTS *dl)
{
	POINT2D		A1, A2, A3;
	size_t		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(geom->flags);
	const char *rawdata;

	if (geom->nitems % 2 == 0 || geom->nitems < 3)
		return false;
	rawdata = __loadPoint2d(&A1, geom->rawdata, unitsz);
	if (!__geom_dist2d_pt_pt(pt, &A1, dl))
		return false;
	for (int i=1; i < geom->nitems; i+=2)
	{
		rawdata = __loadPoint2d(&A2, rawdata, unitsz);
		rawdata = __loadPoint2d(&A3, rawdata, unitsz);
		if (!__geom_dist2d_pt_arc(pt, &A1, &A2, &A3, dl))
			return false;
		A1 = A3;
	}
	return true;
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_ptarray_ptarrayarc(const pg_geometry_t *geom,
								 const pg_geometry_t *garc,
								 DISTPTS *dl)
{
	/* see, lw_dist2d_ptarray_ptarrayarc */
	POINT2D		A1, A2;
	POINT2D		B1, B2, B3;
	cl_uint		unitsz_a = sizeof(double) * GEOM_FLAGS_NDIMS(geom->flags);
	cl_uint		unitsz_b = sizeof(double) * GEOM_FLAGS_NDIMS(garc->flags);
	const char *rawdata_a;
	const char *rawdata_b;

	if (garc->nitems % 2 == 0 || garc->nitems < 3)
		return false;

	rawdata_a = __loadPoint2d(&A1, geom->rawdata, unitsz_a);
	for (int t = 1; t < geom->nitems; t++, A1=A2)
	{
		rawdata_a = __loadPoint2d(&A2, rawdata_a, unitsz_a);
		rawdata_b = __loadPoint2d(&B1, garc->rawdata, unitsz_b);
		for (int u = 1; u < garc->nitems; u+=2, B1=B3)
		{
			rawdata_b = __loadPoint2d(&B2, rawdata_b, unitsz_b);
			rawdata_b = __loadPoint2d(&B3, rawdata_b, unitsz_b);
			__geom_dist2d_seg_arc(&A1, &A2, &B1, &B2, &B3, dl);
		}
	}
	return true;
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_ptarrayarc_ptarrayarc(const pg_geometry_t *geom1,
									const pg_geometry_t *geom2,
									DISTPTS *dl)
{
	/* see lw_dist2d_ptarrayarc_ptarrayarc */
	POINT2D		A1, A2, A3;
	POINT2D		B1, B2, B3;
	cl_uint		unitsz1 = sizeof(double) * GEOM_FLAGS_NDIMS(geom1->flags);
	cl_uint		unitsz2 = sizeof(double) * GEOM_FLAGS_NDIMS(geom2->flags);
	const char *rawdata1;
	const char *rawdata2;
	int			twist = dl->twisted;

	rawdata1 = __loadPoint2d(&A1, geom1->rawdata, unitsz1);
	for (int t = 1; t < geom1->nitems; t += 2, A1 = A3)
	{
		rawdata1 = __loadPoint2d(&A2, rawdata1, unitsz1);
		rawdata1 = __loadPoint2d(&A3, rawdata1, unitsz1);

		rawdata2 = __loadPoint2d(&B1, geom2->rawdata, unitsz2);
		for (int u = 1; u < geom2->nitems; u += 2, B1 = B3)
		{
			rawdata2 = __loadPoint2d(&B2, rawdata2, unitsz2);
			rawdata2 = __loadPoint2d(&B3, rawdata2, unitsz2);
			dl->twisted = twist;
			if (!__geom_dist2d_arc_arc(&A1, &A2, &A3, &B1, &B2, &B3, dl))
				return false;
		}
	}
	return true;
}

/* forward declaration */
STATIC_FUNCTION(cl_bool)
geom_dist2d_recursive(const pg_geometry_t *geom1,
					  const pg_geometry_t *geom2,
					  DISTPTS *dl);

STATIC_FUNCTION(cl_bool)
geom_dist2d_point_point(const pg_geometry_t *geom1,
						const pg_geometry_t *geom2,
						DISTPTS *dl)
{
	/* see, lw_dist2d_point_point() */
	POINT2D		p1, p2;

	assert(geom1->type == GEOM_POINTTYPE &&
		   geom2->type == GEOM_POINTTYPE);
	__loadPoint2d(&p1, geom1->rawdata, 0);
	__loadPoint2d(&p2, geom2->rawdata, 0);
	return __geom_dist2d_pt_pt(&p1, &p2, dl);
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_point_line(const pg_geometry_t *geom1,
                       const pg_geometry_t *geom2,
                       DISTPTS *dl)
{
	/* see, lw_dist2d_point_line */
	POINT2D		pt;

	assert(geom1->type == GEOM_POINTTYPE &&
		   geom2->type == GEOM_LINETYPE);
	__loadPoint2d(&pt, geom1->rawdata, 0);
	return __geom_dist2d_pt_ptarray(&pt, geom2, dl);
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_point_tri(const pg_geometry_t *geom1,
					  const pg_geometry_t *geom2,
					  DISTPTS *dl)
{
	/* see, lw_dist2d_point_tri */
	POINT2D		pt;
	int			status;

	assert(geom1->type == GEOM_POINTTYPE &&
		   geom2->type == GEOM_TRIANGLETYPE);
	__loadPoint2d(&pt, geom1->rawdata, 0);
	status = __geom_contains_point(geom2, &pt, dl->kcxt);
	if (status == PT_ERROR)
		return false;
	else if (status != PT_OUTSIDE)
	{
		dl->distance = 0.0;
		dl->p1 = pt;
		dl->p2 = pt;
		return true;
	}
	assert(status == PT_OUTSIDE);
	return __geom_dist2d_pt_ptarray(&pt, geom2, dl);
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_point_poly(const pg_geometry_t *geom1,
					   const pg_geometry_t *geom2,
					   DISTPTS *dl)
{
	/* see, lw_dist2d_point_poly */
	POINT2D		pt;
	const char *pos = NULL;

	assert(geom1->type == GEOM_POINTTYPE &&
		   geom2->type == GEOM_POLYGONTYPE);
	__loadPoint2d(&pt, geom1->rawdata, 0);
	for (int i=0; i < geom2->nitems; i++)
	{
		pg_geometry_t __geom;
		int		status;

		pos = geometry_load_subitem(&__geom, geom2, pos, i, dl->kcxt);
		if (!pos)
			return false;
		status = __geom_contains_point(&__geom, &pt, dl->kcxt);
		if (status == PT_ERROR)
			return false;
		if (i == 0)
		{
			/* Return distance to outer ring if not inside it */
			if (status == PT_OUTSIDE)
				return __geom_dist2d_pt_ptarray(&pt, &__geom, dl);
		}
		else
		{
			 /*
			  * Inside the outer ring.
			  * Scan though each of the inner rings looking to
			  * see if its inside.  If not, distance==0.
			  * Otherwise, distance = pt to ring distance
			  */
			if (status == PT_BOUNDARY || status == PT_INSIDE)
				return __geom_dist2d_pt_ptarray(&pt, &__geom, dl);
		}
	}
	/* It is inside of the polygon  */
	dl->distance = 0.0;
	dl->p1 = pt;
	dl->p2 = pt;
	return true;
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_point_circstring(const pg_geometry_t *geom1,
							 const pg_geometry_t *geom2,
							 DISTPTS *dl)
{
	/* see, lw_dist2d_point_circstring() */
	POINT2D		pt;

	assert(geom1->type == GEOM_POINTTYPE &&
		   geom2->type == GEOM_CIRCSTRINGTYPE);
	__loadPoint2d(&pt, geom1->rawdata, 0);
	return __geom_dist2d_pt_ptarrayarc(&pt, geom2, dl);
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_point_curvepoly(const pg_geometry_t *geom1,
							const pg_geometry_t *geom2,
							DISTPTS *dl)
{
	/* see, lw_dist2d_point_curvepoly */
	POINT2D		pt;
	const char *pos = NULL;

	assert(geom1->type == GEOM_POINTTYPE &&
		   geom2->type == GEOM_CURVEPOLYTYPE);
	__loadPoint2d(&pt, geom1->rawdata, 0);
	for (int i=0; i < geom2->nitems; i++)
	{
		pg_geometry_t __geom;
		int		status;

		pos = geometry_load_subitem(&__geom, geom2, pos, i, dl->kcxt);
		if (!pos)
			return false;
		status = __geom_contains_point(&__geom, &pt, dl->kcxt);
		if (status == PT_ERROR)
			return false;
		if (i == 0)
		{
			if (status == PT_OUTSIDE)
				return geom_dist2d_recursive(geom1, &__geom, dl);
		}
		else
		{
			if (status == PT_INSIDE)
				return geom_dist2d_recursive(geom1, &__geom, dl);
		}
	}
	/* Is inside the polygon */
	dl->distance = 0.0;
	dl->p1 = pt;
	dl->p2 = pt;
	return true;
}

STATIC_INLINE(cl_bool)
geom_dist2d_line_line(const pg_geometry_t *geom1,
					  const pg_geometry_t *geom2,
					  DISTPTS *dl)
{
	/* see lw_dist2d_line_line */
	assert(geom1->type == GEOM_LINETYPE &&
		   geom2->type == GEOM_LINETYPE);
	return __geom_dist2d_ptarray_ptarray(geom1, geom2, dl);
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_line_tri(const pg_geometry_t *geom1,
					 const pg_geometry_t *geom2,
					 DISTPTS *dl)
{
	/* see lw_dist2d_line_tri */
	POINT2D		pt;

	assert(geom1->type == GEOM_LINETYPE &&
		   geom2->type == GEOM_TRIANGLETYPE);
	/* XXX why only point-0? */
	__loadPoint2d(&pt, geom1->rawdata, 0);
	if (__geom_contains_point(geom2, &pt, dl->kcxt))
	{
		dl->distance = 0.0;
		dl->p1 = pt;
		dl->p2 = pt;
		return true;
	}
	return __geom_dist2d_ptarray_ptarray(geom1, geom2, dl);
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_line_poly(const pg_geometry_t *geom1,
					  const pg_geometry_t *geom2,
					  DISTPTS *dl)
{
	/* see, lw_dist2d_line_poly */
	const char *pos = NULL;
	POINT2D		pt0;
	cl_bool		meet_inside = false;

	assert((geom1->type == GEOM_LINETYPE ||
			geom1->type == GEOM_CIRCSTRINGTYPE) &&
		   geom2->type == GEOM_POLYGONTYPE);
	__loadPoint2d(&pt0, geom1->rawdata, 0);
	for (int i=0; i < geom2->nitems; i++)
	{
		pg_geometry_t __geom;
		int		status;

		pos = geometry_load_subitem(&__geom, geom2, pos, i, dl->kcxt);
		if (!pos)
			return false;
		if (i == 0)
		{
			/* Line has a point outside of the polygon.
			 * Check distance to outer ring only.
			 */
			status = __geom_contains_point(&__geom, &pt0, dl->kcxt);
			if (status == PT_ERROR)
				return false;
			if (status == PT_OUTSIDE)
				return __geom_dist2d_ptarray_ptarray(geom1, &__geom, dl);
		}
		else
		{
			if (!__geom_dist2d_ptarray_ptarray(geom1, &__geom, dl))
				return false;
			/* just a check if the answer is already given */
			if (dl->distance <= dl->tolerance)
				return true;
			if (!meet_inside)
			{
				status = __geom_contains_point(&__geom, &pt0, dl->kcxt);
				if (status == PT_ERROR)
					return false;
				if (status != PT_OUTSIDE)
					meet_inside = true;
			}
		}
	}
	if (!meet_inside)
	{
		dl->distance = 0.0;
		dl->p1 = pt0;
		dl->p2 = pt0;
	}
	return true;
}

STATIC_INLINE(cl_bool)
geom_dist2d_line_circstring(const pg_geometry_t *geom1,
							const pg_geometry_t *geom2,
							DISTPTS *dl)
{
	/* see, lw_dist2d_line_circstring */
	assert(geom1->type == GEOM_LINETYPE &&
		   geom2->type == GEOM_CIRCSTRINGTYPE);
	return __geom_dist2d_ptarray_ptarrayarc(geom1, geom2, dl);
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_line_curvepoly(const pg_geometry_t *geom1,
						   const pg_geometry_t *geom2,
						   DISTPTS *dl)
{
	/* see, lw_dist2d_line_curvepoly */
	const char *pos = NULL;
	POINT2D		pt0;
	cl_bool		meet_inside = false;

	/* note that geom_dist2d_circstring_curvepoly() may call this function */
	assert((geom1->type == GEOM_LINETYPE ||
			geom1->type == GEOM_CIRCSTRINGTYPE) &&
		   geom2->type == GEOM_CURVEPOLYTYPE);

	__loadPoint2d(&pt0, geom1->rawdata, 0);
	for (int i=0; i < geom2->nitems; i++)
	{
		pg_geometry_t __geom;
		int		status;

		pos = geometry_load_subitem(&__geom, geom2, pos, i, dl->kcxt);
		if (!pos)
			return false;
		if (i == 0)
		{
			status = __geom_contains_point(&__geom, &pt0, dl->kcxt);
			if (status == PT_ERROR)
				return false;
			if (status == PT_OUTSIDE)
				return geom_dist2d_recursive(geom1, &__geom, dl);
		}
		else
		{
			if (!geom_dist2d_recursive(geom1, &__geom, dl))
				return false;
			/* just a check if the answer is already given */
			if (dl->distance <= dl->tolerance)
				return true;
			if (!meet_inside)
			{
				status = __geom_contains_point(&__geom, &pt0, dl->kcxt);
				if (status == PT_ERROR)
					return false;
				if (status != PT_OUTSIDE)
					meet_inside = true;
			}
		}
	}

	if (!meet_inside)
	{
		dl->distance = 0.0;
		dl->p1 = pt0;
		dl->p2 = pt0;
	}
	return true;
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_tri_tri(const pg_geometry_t *geom1,
					const pg_geometry_t *geom2,
					DISTPTS *dl)
{
	/* see lw_dist2d_tri_tri */
	POINT2D		pt;
	int			status;

	assert(geom1->type == GEOM_TRIANGLETYPE &&
		   geom2->type == GEOM_TRIANGLETYPE);

	__loadPoint2d(&pt, geom2->rawdata, 0);
	status = __geom_contains_point(geom1, &pt, dl->kcxt);
	if (status == PT_ERROR)
		return false;
	else if (status != PT_OUTSIDE)
	{
		dl->distance = 0.0;
		dl->p1 = pt;
		dl->p2 = pt;
		return true;
	}

	__loadPoint2d(&pt, geom1->rawdata, 0);
	status = __geom_contains_point(geom2, &pt, dl->kcxt);
	if (status == PT_ERROR)
		return false;
	else if (status != PT_OUTSIDE)
	{
		dl->distance = 0.0;
		dl->p1 = pt;
		dl->p2 = pt;
		return true;
	}
	return __geom_dist2d_ptarray_ptarray(geom1, geom2, dl);
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_tri_poly(const pg_geometry_t *geom1,
					 const pg_geometry_t *geom2,
					 DISTPTS *dl)
{
	/* lw_dist2d_tri_poly */
	const char *pos = NULL;
	POINT2D		pt;
	cl_bool		meet_inside = false;

	assert(geom1->type == GEOM_TRIANGLETYPE &&
		   geom2->type == GEOM_POLYGONTYPE);
	__loadPoint2d(&pt, geom1->rawdata, 0);
	for (int i=0; i < geom2->nitems; i++)
	{
		pg_geometry_t __geom;
		int		status;

		pos = geometry_load_subitem(&__geom, geom2, pos, i, dl->kcxt);
		if (!pos)
			return false;
		if (i == 0)
		{
			status = __geom_contains_point(&__geom, &pt, dl->kcxt);
			if (status == PT_ERROR)
				return false;
			if (status == PT_OUTSIDE)
			{
				POINT2D		pt2;

				if (!__geom_dist2d_ptarray_ptarray(geom1, &__geom, dl))
					return false;
				/* just a check if the answer is already given */
				if (dl->distance <= dl->tolerance)
					return true;
				__loadPoint2d(&pt2, __geom.rawdata, 0);
				status = __geom_contains_point(geom1, &pt2, dl->kcxt);
				if (status == PT_ERROR)
					return false;
				if (status != PT_OUTSIDE)
				{
					dl->distance = 0.0;
					dl->p1 = pt2;
					dl->p2 = pt2;
					return true;
				}
			}
		}
		else
		{
			if (!__geom_dist2d_ptarray_ptarray(geom1, &__geom, dl))
				return false;
			/* just a check if the answer is already given */
			if (dl->distance <= dl->tolerance)
				return true;
			if (!meet_inside)
			{
				status = __geom_contains_point(&__geom, &pt, dl->kcxt);
				if (status == PT_ERROR)
					return false;
				if (status != PT_OUTSIDE)
					meet_inside = true;
			}
		}
	}

	if (!meet_inside)
	{
		dl->distance = 0.0;
		dl->p1 = pt;
		dl->p2 = pt;
	}
	return true;
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_tri_circstring(const pg_geometry_t *geom1,
						   const pg_geometry_t *geom2,
						   DISTPTS *dl)
{
	/* see lw_dist2d_tri_circstring */
	POINT2D		pt;
	int			status;

	assert(geom1->type == GEOM_TRIANGLETYPE &&
		   geom2->type == GEOM_CIRCSTRINGTYPE);

	__loadPoint2d(&pt, geom2->rawdata, 0);
	status = __geom_contains_point(geom1, &pt, dl->kcxt);
	if (status == PT_ERROR)
		return false;
	if (status != PT_OUTSIDE)
	{
		dl->distance = 0.0;
		dl->p1 = pt;
		dl->p2 = pt;
		return true;
	}
	return __geom_dist2d_ptarray_ptarrayarc(geom1, geom2, dl);
}

STATIC_INLINE(cl_bool)
__geom_curvering_getfirstpoint2d(POINT2D *pt,
								 const pg_geometry_t *geom,
								 DISTPTS *dl)
{
	/* see lw_curvering_getfirstpoint2d_cp */
	if (geom->type == GEOM_LINETYPE ||
		geom->type == GEOM_CIRCSTRINGTYPE)
	{
		memcpy(pt, &geom->rawdata, sizeof(POINT2D));
		return true;
	}
	else if (geom->type == GEOM_COMPOUNDTYPE)
	{
		//XXX compound has inline types, right assumption?
		const char *dataptr
			= (geom->rawdata + LONGALIGN(sizeof(cl_uint) * geom->nitems));
		__loadPoint2d(pt, dataptr, 0);
		return true;
	}
	return false;
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_tri_curvepoly(const pg_geometry_t *geom1,
						  const pg_geometry_t *geom2,
						  DISTPTS *dl)
{
	/* see lw_dist2d_tri_curvepoly */
	const char *pos = NULL;
	POINT2D		pt;
	cl_bool		meet_inside = false;

	assert(geom1->type == GEOM_TRIANGLETYPE &&
		   geom2->type == GEOM_CURVEPOLYTYPE);

	__loadPoint2d(&pt, geom1->rawdata, 0);
	for (int i=0; i < geom2->nitems; i++)
	{
		pg_geometry_t __geom;
		cl_int		status;

		pos = geometry_load_subitem(&__geom, geom2, pos, i, dl->kcxt);
		if (!pos)
			return false;
		if (i == 0)
		{
			status = __geom_contains_point(&__geom, &pt, dl->kcxt);
			if (status == PT_ERROR)
				return false;
			if (status == PT_OUTSIDE)
			{
				POINT2D		pt2;

				if (!geom_dist2d_recursive(geom1, &__geom, dl))
					return false;
				if (!__geom_curvering_getfirstpoint2d(&pt2, &__geom, dl))
					return false;
				/* maybe poly is inside triangle? */
				status = __geom_contains_point(geom1, &pt2, dl->kcxt);
				if (status == PT_ERROR)
					return false;
				if (status != PT_OUTSIDE)
				{
					dl->distance = 0.0;
					dl->p1 = pt;
					dl->p2 = pt;
					return true;
				}
			}
		}
		else
		{
			if (!geom_dist2d_recursive(geom1, &__geom, dl))
				return false;
			/* just a check if the answer is already given */
			if (dl->distance <= dl->tolerance)
				return true;
			if (!meet_inside)
			{
				status = __geom_contains_point(&__geom, &pt, dl->kcxt);
				if (status == PT_ERROR)
					return false;
				if (status != PT_OUTSIDE)
					meet_inside = true;
			}
		}
	}

	if (!meet_inside)
	{
		dl->distance = 0.0;
		dl->p1 = pt;
		dl->p2 = pt;
	}
	return true;
}

STATIC_INLINE(cl_bool)
geom_dist2d_circstring_poly(const pg_geometry_t *geom1,
							const pg_geometry_t *geom2,
							DISTPTS *dl)
{
	/* see, lw_dist2d_circstring_poly() */
	return geom_dist2d_line_poly(geom1, geom2, dl);
}

STATIC_INLINE(cl_bool)
geom_dist2d_circstring_circstring(const pg_geometry_t *geom1,
								  const pg_geometry_t *geom2,
								  DISTPTS *dl)
{
	/* see, lw_dist2d_circstring_circstring */
	assert(geom1->type == GEOM_CIRCSTRINGTYPE &&
		   geom2->type == GEOM_CIRCSTRINGTYPE);
	return __geom_dist2d_ptarrayarc_ptarrayarc(geom1, geom2, dl);
}

STATIC_INLINE(cl_bool)
geom_dist2d_circstring_curvepoly(const pg_geometry_t *geom1,
								 const pg_geometry_t *geom2,
								 DISTPTS *dl)
{
	/* see, lw_dist2d_circstring_curvepoly */
	assert(geom1->type == GEOM_CIRCSTRINGTYPE &&
		   geom2->type == GEOM_CURVEPOLYTYPE);
	return geom_dist2d_line_curvepoly(geom1, geom2, dl);
}

STATIC_FUNCTION(cl_bool)
__geom_dist2d_xpoly_xpoly(const pg_geometry_t *geom1,
						  const pg_geometry_t *geom2,
						  DISTPTS *dl)
{
	/* see, lw_dist2d_poly_poly */
	pg_geometry_t __geom1;
	pg_geometry_t __geom2;
	pg_geometry_t __gtemp;
	const char *pos1 = NULL;
	const char *pos2 = NULL;
	POINT2D		pt;
	int			status;

 	assert((geom1->type == GEOM_POLYGONTYPE ||
			geom1->type == GEOM_CURVEPOLYTYPE) &&
		   (geom2->type == GEOM_POLYGONTYPE ||
			geom2->type == GEOM_CURVEPOLYTYPE));
	pos1 = geometry_load_subitem(&__geom1, geom1, pos1, 0, dl->kcxt);
	pos2 = geometry_load_subitem(&__geom2, geom2, pos2, 0, dl->kcxt);
	if (!pos1 || !pos2)
		return false;

	/* 2. check if poly1 has first point outside poly2 and vice versa,
	 * if so, just check outer rings here it would be possible to handle
	 * the information about which one is inside which one and only search
	 * for the smaller ones in the bigger ones holes.
	 */
	__loadPoint2d(&pt, __geom1.rawdata, 0);
	status = __geom_contains_point(&__geom2, &pt, dl->kcxt);
	if (status == PT_ERROR)
		return false;
	if (status == PT_OUTSIDE)
	{
		__loadPoint2d(&pt, __geom2.rawdata, 0);
		status = __geom_contains_point(&__geom1, &pt, dl->kcxt);
		if (status == PT_ERROR)
			return false;
		if (status == PT_OUTSIDE)
			return __geom_dist2d_ptarray_ptarray(&__geom1, &__geom2, dl);
	}

	/* 3. check if first point of poly2 is in a hole of poly1.
	 * If so check outer ring of poly2 against that hole of poly1
	 */
	__loadPoint2d(&pt, __geom2.rawdata, 0);
	for (int i = 1; i < geom1->nitems; i++)
	{
		pos1 = geometry_load_subitem(&__gtemp, geom1, pos1, i, dl->kcxt);
		if (!pos1)
			return false;
		status = __geom_contains_point(&__gtemp, &pt, dl->kcxt);
		if (status == PT_ERROR)
			return false;
		if (status != PT_OUTSIDE)
			return __geom_dist2d_ptarray_ptarray(&__gtemp, &__geom2, dl);
	}

	/* 4. check if first point of poly1 is in a hole of poly2.
	 * If so check outer ring of poly1 against that hole of poly2
	 */
	 __loadPoint2d(&pt, __geom1.rawdata, 0);
	 for (int i = 1; i < geom2->nitems; i++)
	 {
		 pos2 = geometry_load_subitem(&__gtemp, geom2, pos2, i, dl->kcxt);
		 if (!pos2)
			 return false;
		 status = __geom_contains_point(&__gtemp, &pt, dl->kcxt);
		 if (status == PT_ERROR)
			 return false;
		 if (status != PT_OUTSIDE)
			 return __geom_dist2d_ptarray_ptarray(&__geom1, &__gtemp, dl);
	 }

	 /* 5. If we have come all the way here we know that the first
	  * point of one of them is inside the other ones outer ring and
	  * not in holes so we check wich one is inside.
	  */
	 __loadPoint2d(&pt, __geom1.rawdata, 0);
	 status = __geom_contains_point(&__geom2, &pt, dl->kcxt);
	 if (status == PT_ERROR)
		 return false;
	 if (status != PT_OUTSIDE)
	 {
		 dl->distance = 0.0;
		 dl->p1 = pt;
		 dl->p2 = pt;
		 return true;
	 }

	 __loadPoint2d(&pt, __geom2.rawdata, 0);
	 status = __geom_contains_point(&__geom1, &pt, dl->kcxt);
	 if (status == PT_ERROR)
		 return false;
	 if (status != PT_OUTSIDE)
	 {
		 dl->distance = 0.0;
		 dl->p1 = pt;
		 dl->p2 = pt;
		 return true;
	 }
	 return false;
}

STATIC_INLINE(cl_bool)
geom_dist2d_poly_poly(const pg_geometry_t *geom1,
					  const pg_geometry_t *geom2,
					  DISTPTS *dl)
{
	/* see lw_dist2d_poly_poly */
	assert(geom1->type == GEOM_POLYGONTYPE &&
		   geom2->type == GEOM_POLYGONTYPE);
	return __geom_dist2d_xpoly_xpoly(geom1, geom2, dl);
}

STATIC_INLINE(cl_bool)
geom_dist2d_poly_curvepoly(const pg_geometry_t *geom1,
						   const pg_geometry_t *geom2,
						   DISTPTS *dl)
{
	/* see lw_dist2d_poly_curvepoly */
	assert(geom1->type == GEOM_POLYGONTYPE &&
		   geom2->type == GEOM_CURVEPOLYTYPE);
	return __geom_dist2d_xpoly_xpoly(geom1, geom2, dl);
}

STATIC_INLINE(cl_bool)
geom_dist2d_curvepoly_curvepoly(const pg_geometry_t *geom1,
								const pg_geometry_t *geom2,
								DISTPTS *dl)
{
	/* see lw_dist2d_curvepoly_curvepoly */
	assert(geom1->type == GEOM_CURVEPOLYTYPE &&
		   geom2->type == GEOM_CURVEPOLYTYPE);
	return __geom_dist2d_xpoly_xpoly(geom1, geom2, dl);
}

STATIC_FUNCTION(cl_bool)
geom_dist2d_recursive(const pg_geometry_t *geom1,
					  const pg_geometry_t *geom2,
					  DISTPTS *dl)
{
	/* see lw_dist2d_recursive() */
	pg_geometry_t	__geom;
	const char	   *pos = NULL;

	assert(!geom1->isnull && !geom2->isnull);
	if (geometry_is_collection(geom1))
	{
		for (int i=0; i < geom1->nitems; i++)
		{
			pos = geometry_load_subitem(&__geom, geom1, pos, i, dl->kcxt);
			if (!pos || !geom_dist2d_recursive(&__geom, geom2, dl))
				return false;
			if (dl->distance <= dl->tolerance)
				return true; /* just a check if the answer is already given */
		}
	}
	else if (geometry_is_collection(geom2))
	{
		for (int i=0; i < geom2->nitems; i++)
		{
			pos = geometry_load_subitem(&__geom, geom2, pos, i, dl->kcxt);
			if (!pos || !geom_dist2d_recursive(geom1, &__geom, dl))
				return false;
			if (dl->distance <= dl->tolerance)
				return true; /* just a check if the answer is already given */
		}
	}
	else if (geom1->nitems > 0 && geom2->nitems)
	{
		/*
		 * see lw_dist2d_distribute_bruteforce()
		 *
		 * NOTE that we don't use the logic of lw_dist2d_distribute_fast()
		 * here, even if both geometries are line, polygon, or triangle,
		 * because it internally allocates variable length buffer to sort
		 * the points. It is not easy to implement on GPU device.
		 */
		switch (geom1->type)
		{
			case GEOM_POINTTYPE:
				dl->twisted = 1;
				switch (geom2->type)
				{
					case GEOM_POINTTYPE:
						return geom_dist2d_point_point(geom1,geom2,dl);
					case GEOM_LINETYPE:
						return geom_dist2d_point_line(geom1,geom2,dl);
					case GEOM_TRIANGLETYPE:
						return geom_dist2d_point_tri(geom1,geom2,dl);
					case GEOM_POLYGONTYPE:
						return geom_dist2d_point_poly(geom1,geom2,dl);
					case GEOM_CIRCSTRINGTYPE:
						return geom_dist2d_point_circstring(geom1,geom2,dl);
					case GEOM_CURVEPOLYTYPE:
						return geom_dist2d_point_curvepoly(geom1,geom2,dl);
					default:
						STROM_ELOG(dl->kcxt, "unknown geometry data type");
						return false;
				}
			case GEOM_LINETYPE:
				dl->twisted = 1;
				switch (geom2->type)
				{
					case GEOM_POINTTYPE:
						dl->twisted = -1;
						return geom_dist2d_point_line(geom2,geom1,dl);
					case GEOM_LINETYPE:
						return geom_dist2d_line_line(geom1,geom2,dl);
					case GEOM_TRIANGLETYPE:
						return geom_dist2d_line_tri(geom1,geom2,dl);
					case GEOM_POLYGONTYPE:
						return geom_dist2d_line_poly(geom1,geom2,dl);
					case GEOM_CIRCSTRINGTYPE:
						return geom_dist2d_line_circstring(geom1,geom2,dl);
					case GEOM_CURVEPOLYTYPE:
						return geom_dist2d_line_curvepoly(geom1,geom2,dl);
					default:
						STROM_ELOG(dl->kcxt, "unknown geometry data type");
						return false;
				}
			case GEOM_TRIANGLETYPE:
				dl->twisted = 1;
				switch (geom2->type)
				{
					case GEOM_POINTTYPE:
						dl->twisted = -1;
						return geom_dist2d_point_tri(geom2,geom1,dl);
					case GEOM_LINETYPE:
						dl->twisted = -1;
						return geom_dist2d_line_tri(geom2,geom1,dl);
					case GEOM_TRIANGLETYPE:
						return geom_dist2d_tri_tri(geom1,geom2,dl);
					case GEOM_POLYGONTYPE:
						return geom_dist2d_tri_poly(geom1,geom2,dl);
					case GEOM_CIRCSTRINGTYPE:
						return geom_dist2d_tri_circstring(geom1,geom2,dl);
					case GEOM_CURVEPOLYTYPE:
						return geom_dist2d_tri_curvepoly(geom1,geom2,dl);
					default:
						STROM_ELOG(dl->kcxt, "unknown geometry data type");
						return false;
				}
			case GEOM_CIRCSTRINGTYPE:
				dl->twisted = 1;
				switch (geom2->type)
				{
					case GEOM_POINTTYPE:
						dl->twisted = -1;
						return geom_dist2d_point_circstring(geom2,geom1,dl);
					case GEOM_LINETYPE:
						dl->twisted = -1;
						return geom_dist2d_line_circstring(geom2,geom1,dl);
					case GEOM_TRIANGLETYPE:
						dl->twisted = -1;
						return geom_dist2d_tri_circstring(geom2,geom1,dl);
					case GEOM_POLYGONTYPE:
						return geom_dist2d_circstring_poly(geom1,geom2,dl);
					case GEOM_CIRCSTRINGTYPE:
						return geom_dist2d_circstring_circstring(geom1,
																 geom2,dl);
					case GEOM_CURVEPOLYTYPE:
						return geom_dist2d_circstring_curvepoly(geom1,
																geom2,dl);
					default:
						STROM_ELOG(dl->kcxt, "unknown geometry data type");
						return false;
				}
			case GEOM_POLYGONTYPE:
				dl->twisted = -1;
				switch (geom2->type)
				{
					case GEOM_POINTTYPE:
						return geom_dist2d_point_poly(geom2,geom1,dl);
					case GEOM_LINETYPE:
						return geom_dist2d_line_poly(geom2,geom1,dl);
					case GEOM_TRIANGLETYPE:
						return geom_dist2d_tri_poly(geom2,geom1,dl);
					case GEOM_POLYGONTYPE:
						dl->twisted = 1;
						return geom_dist2d_poly_poly(geom1,geom2,dl);
					case GEOM_CIRCSTRINGTYPE:
						return geom_dist2d_circstring_poly(geom2,geom1,dl);
					case GEOM_CURVEPOLYTYPE:
						dl->twisted = 1;
						return geom_dist2d_poly_curvepoly(geom1,geom2,dl);
					default:
						STROM_ELOG(dl->kcxt, "unknown geometry data type");
						return false;
				}
			case GEOM_CURVEPOLYTYPE:
				dl->twisted = -1;
				switch (geom2->type)
				{
					case GEOM_POINTTYPE:
						return geom_dist2d_point_curvepoly(geom2,geom1,dl);
					case GEOM_LINETYPE:
						return geom_dist2d_line_curvepoly(geom2,geom1,dl);
					case GEOM_TRIANGLETYPE:
						return geom_dist2d_tri_curvepoly(geom2,geom1,dl);
					case GEOM_POLYGONTYPE:
						return geom_dist2d_poly_curvepoly(geom2,geom1,dl);
					case GEOM_CIRCSTRINGTYPE:
						return geom_dist2d_circstring_curvepoly(geom2,
																geom1,dl);
					case GEOM_CURVEPOLYTYPE:
						dl->twisted = 1;
						return geom_dist2d_curvepoly_curvepoly(geom1,
															   geom2,dl);
					default:
						STROM_ELOG(dl->kcxt, "unknown geometry data type");
						return false;
				}
			default:
				STROM_ELOG(dl->kcxt, "unknown geometry data type");
				return false;
		}
	}
	return true;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_st_distance(kern_context *kcxt,
				 const pg_geometry_t &geom1,
				 const pg_geometry_t &geom2)
{
	pg_float8_t	result;
	DISTPTS		dl;

	result.isnull = geom1.isnull | geom2.isnull;
	if (!result.isnull)
	{
		if (geom1.srid != geom2.srid)
		{
			STROM_EREPORT(kcxt, ERRCODE_INVALID_PARAMETER_VALUE,
						  "Operation on mixed SRID geometries");
			result.isnull = true;
		}
		else
		{
			memset(&dl, 0, sizeof(DISTPTS));
			dl.kcxt = kcxt;
			dl.distance = DBL_MAX;
			dl.tolerance = 0.0;
			if (!geom_dist2d_recursive(&geom1, &geom2, &dl))
				result.isnull = true;
			else
				result.value = dl.distance;
		}
	}
	return result;
}

/*
 * St_Dwithin - Returns true if the geometries are within the specified
 *              distance of one another
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_st_dwithin(kern_context *kcxt,
				const pg_geometry_t &geom1,
				const pg_geometry_t &geom2,
				pg_float8_t arg3)
{
	/* see, LWGEOM_dwithin */
	pg_bool_t	result;
	DISTPTS		dl;

	result.isnull = geom1.isnull | geom2.isnull | arg3.isnull;
	if (!result.isnull)
	{
		if (geom1.srid != geom2.srid)
		{
			STROM_EREPORT(kcxt, ERRCODE_INVALID_PARAMETER_VALUE,
						  "Operation on mixed SRID geometries");
			result.isnull = true;
		}
		else if (arg3.value < 0.0)
		{
			STROM_EREPORT(kcxt, ERRCODE_INVALID_PARAMETER_VALUE,
						  "Tolerance cannot be less than zero");
			result.isnull = true;
		}
		else
		{
			memset(&dl, 0, sizeof(DISTPTS));
			dl.kcxt = kcxt;
			dl.distance = DBL_MAX;
			dl.tolerance = arg3.value;
			if (!geom_dist2d_recursive(&geom1, &geom2, &dl))
				result.isnull = true;
			else
				result.value = (dl.distance <= arg3.value);
		}
	}
	return result;
}

/* ================================================================
 *
 * ST_LineCrossingDirection
 *
 * ================================================================
 */
#define SEG_ERROR		   -1
#define SEG_NO_INTERSECTION	0
#define SEG_COLINEAR		1
#define SEG_CROSS_LEFT		2
#define SEG_CROSS_RIGHT		3

STATIC_FUNCTION(cl_int)
__geom_segment_intersects(const POINT2D *p1, const POINT2D *p2,
						  const POINT2D *q1, const POINT2D *q2)
{
	/* see, lw_segment_intersects */
	int		pq1, pq2, qp1, qp2;

	/* No envelope interaction => we are done. */
	if (Min(p1->x, p2->x) > Max(q1->x, q2->x) ||
		Max(p1->x, p2->x) < Min(q1->x, q2->x) ||
		Min(p1->y, p2->y) > Max(q1->y, q2->y) ||
		Max(p1->y, p2->y) < Min(q1->y, q2->y))
		return SEG_NO_INTERSECTION;

	/* Are the start and end points of q on the same side of p? */
	pq1 = __geom_segment_side(p1, p2, q1);
	pq2 = __geom_segment_side(p1, p2, q2);
	if ((pq1 > 0 && pq2 > 0) || (pq1 < 0 && pq2 < 0))
		return SEG_NO_INTERSECTION;

	 /* Are the start and end points of p on the same side of q? */
	qp1 = __geom_segment_side(q1, q2, p1);
	qp2 = __geom_segment_side(q1, q2, p2);
	if ((qp1 > 0 && qp2 > 0) || (qp1 < 0 && qp2 < 0))
		return SEG_NO_INTERSECTION;

	/* Nobody is on one side or another? Must be colinear. */
	if (pq1 == 0 && pq2 == 0 && qp1 == 0 && qp2 == 0)
		return SEG_COLINEAR;

	/* Second point of p or q touches, it's not a crossing. */
	if (pq2 == 0 || qp2 == 0)
		return SEG_NO_INTERSECTION;

	/* First point of p touches, it's a "crossing". */
	if (pq1 == 0)
		return (pq2 > 0 ? SEG_CROSS_RIGHT : SEG_CROSS_LEFT);

	/* The segments cross, what direction is the crossing? */
	return (pq1 < pq2 ? SEG_CROSS_RIGHT : SEG_CROSS_LEFT);
}

#define LINE_NO_CROSS				0
#define LINE_CROSS_LEFT			   -1
#define LINE_CROSS_RIGHT			1
#define LINE_MULTICROSS_END_LEFT   -2
#define LINE_MULTICROSS_END_RIGHT	2
#define LINE_MULTICROSS_END_SAME_FIRST_LEFT	   -3
#define LINE_MULTICROSS_END_SAME_FIRST_RIGHT	3

STATIC_FUNCTION(cl_int)
__geom_crossing_direction(kern_context *kcxt,
						  const pg_geometry_t *geom1,
						  const pg_geometry_t *geom2)
{
	/* see, lwline_crossing_direction */
	POINT2D		p1, p2;
	POINT2D		q1, q2;
	int			unitsz1 = sizeof(double) * GEOM_FLAGS_NDIMS(geom1->flags);
	int			unitsz2 = sizeof(double) * GEOM_FLAGS_NDIMS(geom2->flags);
	const char *pos1;
	const char *pos2;
	int			cross_left = 0;
	int			cross_right = 0;
	int			first_cross = 0;
	int			this_cross = 0;

	/* one-point lines can't intersect (and shouldn't exist). */
	if (geom1->nitems < 2 || geom2->nitems < 2)
		return LINE_NO_CROSS;

	pos2 = __loadPoint2d(&q1, geom2->rawdata, unitsz2);
	for (int i=1; i < geom2->nitems; i++, q1=q2)
	{
		pos2 = __loadPoint2d(&q2, pos2, unitsz2);

		pos1 = __loadPoint2d(&p1, geom1->rawdata, unitsz1);
		for (int j=1; j < geom1->nitems; j++, p1=p2)
		{
			pos1 = __loadPoint2d(&p2, pos1, unitsz1);

			this_cross = __geom_segment_intersects(&p1, &p2, &q1, &q2);
			if (this_cross == SEG_CROSS_LEFT)
			{
				cross_left++;
				if (!first_cross)
					first_cross = SEG_CROSS_LEFT;
			}
			if (this_cross == SEG_CROSS_RIGHT)
			{
				cross_right++;
				if (!first_cross)
					first_cross = SEG_CROSS_RIGHT;
			}
		}
	}

    if (!cross_left && !cross_right)
		return LINE_NO_CROSS;
	if (!cross_left && cross_right == 1)
        return LINE_CROSS_RIGHT;
    if (!cross_right && cross_left == 1)
        return LINE_CROSS_LEFT;
	if (cross_left - cross_right == 1)
        return LINE_MULTICROSS_END_LEFT;
	if (cross_left - cross_right == -1)
        return LINE_MULTICROSS_END_RIGHT;
	if (cross_left - cross_right == 0 &&
		first_cross == SEG_CROSS_LEFT)
        return LINE_MULTICROSS_END_SAME_FIRST_LEFT;
	if (cross_left - cross_right == 0 &&
		first_cross == SEG_CROSS_RIGHT)
		return LINE_MULTICROSS_END_SAME_FIRST_RIGHT;
	return LINE_NO_CROSS;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_st_linecrossingdirection(kern_context *kcxt,
							  const pg_geometry_t &geom1,
							  const pg_geometry_t &geom2)
{
	pg_int4_t	result;

	result.isnull = geom1.isnull | geom2.isnull;
	if (!result.isnull)
	{
		if (geom1.srid != geom2.srid)
		{
			STROM_ELOG(kcxt, "Operation on mixed SRID geometries");
			result.isnull = true;
		}
		else if (geom1.type != GEOM_LINETYPE ||
				 geom2.type != GEOM_LINETYPE)
		{
			STROM_ELOG(kcxt,
					   "St_LineCrossingDirection only accepts LINESTRING");
			result.isnull = true;
		}
		else
		{
			result.value = __geom_crossing_direction(kcxt, &geom1, &geom2);
		}
	}
	return result;
}

/* ================================================================
 *
 * St_Relate(geometry,geometry)
 *
 * ================================================================
 */
STATIC_INLINE(double)
determineSide(const POINT2D *seg1,
			  const POINT2D *seg2,
			  const POINT2D *point)
{
	return ((seg2->x - seg1->x) * (point->y - seg1->y) -
			(point->x - seg1->x) * (seg2->y - seg1->y));
}

STATIC_INLINE(cl_bool)
isOnSegment(const POINT2D *seg1,
			const POINT2D *seg2,
			const POINT2D *point)
{
	if (Max(seg1->x, seg2->x) < point->x ||
		Min(seg1->x, seg2->x) > point->x)
		return false;
	if (Max(seg1->y, seg2->y) < point->y ||
		Min(seg1->y, seg2->y) > point->y)
		return false;
	return true;
}

STATIC_FUNCTION(cl_int)
__geom_point_in_ring(const pg_geometry_t *geom, const POINT2D *pt,
					 kern_context *kcxt)
{
	/* see, point_in_ring */
	cl_uint		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(geom->flags);
	POINT2D		seg1;
	POINT2D		seg2;
	const char *pos;
	int			wn = 0;
	double		side;

	pos = __loadPoint2d(&seg1, geom->rawdata, unitsz);
	for (int i=1; i < geom->nitems; i++, seg1 = seg2)
	{
		pos = __loadPoint2d(&seg2, pos, unitsz);

		/* zero length segments are ignored. */
		if (seg1.x == seg2.x && seg1.y == seg2.y)
			continue;

		side = determineSide(&seg1, &seg2, pt);
		if (side == 0.0)
		{
			if (isOnSegment(&seg1, &seg2, pt))
				return PT_BOUNDARY;		/* on boundary */
		}

		if (seg1.y <= pt->y && pt->y < seg2.y && side > 0.0)
		{
			/*
			 * If the point is to the left of the line, and it's rising,
			 * then the line is to the right of the point and
			 * circling counter-clockwise, so increment.
			 */
			wn++;
		}
		else if (seg2.y <= pt->y && pt->y < seg1.y && side < 0.0)
		{
			/*
			 * If the point is to the right of the line, and it's falling,
			 * then the line is to the right of the point and circling
			 * clockwise, so decrement.
			 */
			wn--;
		}
	}
	if (wn == 0)
		return PT_OUTSIDE;
	return PT_INSIDE;
}

STATIC_FUNCTION(cl_int)
__geom_point_in_polygon(const pg_geometry_t *poly, const POINT2D *pt,
						kern_context *kcxt)
{
	/* see, point_in_polygon */
	pg_geometry_t	ring;
	const char	   *pos = NULL;
	cl_int			status;
	cl_int			retval = PT_OUTSIDE;

	assert(poly->type == GEOM_POLYGONTYPE);
	for (int i=0; i < poly->nitems; i++)
	{
		pos = geometry_load_subitem(&ring, poly, pos, i, kcxt);
		if (!pos)
			return PT_ERROR;
		status = __geom_point_in_ring(&ring, pt, kcxt);
		if (status == PT_ERROR)
			return PT_ERROR;
		if (i == 0)
		{
			if (status == PT_OUTSIDE)
				return PT_OUTSIDE;
			retval = status;
		}
		else
		{
			/* inside a hole => outside the polygon */
			if (status == PT_INSIDE)
				return PT_OUTSIDE;
			/* on the edge of a hole */
			if (status == PT_BOUNDARY)
				return PT_BOUNDARY;
		}
	}
	return retval;
}

STATIC_FUNCTION(cl_int)
__geom_point_in_multipolygon(const pg_geometry_t *geom, const POINT2D *pt,
							 kern_context *kcxt)
{
	/* see, point_in_multipolygon */
	pg_geometry_t poly;
	pg_geometry_t ring;
	const char *pos1 = NULL;
	const char *pos2;
	cl_int		status;
	cl_int		retval = PT_OUTSIDE;

	assert(geom->type == GEOM_MULTIPOLYGONTYPE);
	for (int j=0; j < geom->nitems; j++)
	{
		pos1 = geometry_load_subitem(&poly, geom, pos1, j, kcxt);
		if (!pos1)
			return PT_ERROR;
		if (poly.nitems == 0)
			continue;	/* skip empty polygon */

		pos2 = geometry_load_subitem(&ring, &poly, NULL, 0, kcxt);
		if (!pos2)
			return PT_ERROR;
		status = __geom_point_in_ring(&ring, pt, kcxt);
		if (status == PT_ERROR)
			return PT_ERROR;
		if (status == PT_OUTSIDE)
			continue;	/* outside the exterior ring */
		if (status == PT_BOUNDARY)
			return PT_BOUNDARY;

		retval = status;
		for (int i=1; i < poly.nitems; i++)
		{
			pos2 = geometry_load_subitem(&ring, &poly, pos2, i, kcxt);
			if (!pos2)
				return PT_ERROR;
			status = __geom_point_in_ring(&ring, pt, kcxt);
			if (status == PT_ERROR)
				return PT_ERROR;
			/* inside a hole => outside the polygon */
			if (status == PT_INSIDE)
			{
				retval = PT_OUTSIDE;
				break;
			}
			/* on the edge of a hole */
			if (status == PT_BOUNDARY)
				return PT_BOUNDARY;
		}
		if (retval != PT_OUTSIDE)
			return retval;
	}
	return retval;
}

/* ================================================================
 *
 * Routines to generate intersection-matrix
 *
 * ================================================================
 */
#define IM__INTER_INTER_0D		0000000001U
#define IM__INTER_INTER_1D		0000000003U
#define IM__INTER_INTER_2D		0000000007U
#define IM__INTER_BOUND_0D		0000000010U
#define IM__INTER_BOUND_1D		0000000030U
#define IM__INTER_BOUND_2D		0000000070U
#define IM__INTER_EXTER_0D		0000000100U
#define IM__INTER_EXTER_1D		0000000300U
#define IM__INTER_EXTER_2D		0000000700U
#define IM__BOUND_INTER_0D		0000001000U
#define IM__BOUND_INTER_1D		0000003000U
#define IM__BOUND_INTER_2D		0000007000U
#define IM__BOUND_BOUND_0D		0000010000U
#define IM__BOUND_BOUND_1D		0000030000U
#define IM__BOUND_BOUND_2D		0000070000U
#define IM__BOUND_EXTER_0D		0000100000U
#define IM__BOUND_EXTER_1D		0000300000U
#define IM__BOUND_EXTER_2D		0000700000U
#define IM__EXTER_INTER_0D		0001000000U
#define IM__EXTER_INTER_1D		0003000000U
#define IM__EXTER_INTER_2D		0007000000U
#define IM__EXTER_BOUND_0D		0010000000U
#define IM__EXTER_BOUND_1D		0030000000U
#define IM__EXTER_BOUND_2D		0070000000U
#define IM__EXTER_EXTER_0D		0100000000U
#define IM__EXTER_EXTER_1D		0300000000U
#define IM__EXTER_EXTER_2D		0700000000U
#define IM__MASK_FULL			0777777777U

STATIC_INLINE(cl_int)
IM__TWIST(cl_int status)
{
	if (status < 0)
		return status;		/* error */
	return (((status & IM__INTER_INTER_2D))       |
			((status & IM__INTER_BOUND_2D) <<  6) |
			((status & IM__INTER_EXTER_2D) << 12) |
			((status & IM__BOUND_INTER_2D) >>  6) |
			((status & IM__BOUND_BOUND_2D))       |
			((status & IM__BOUND_EXTER_2D) <<  6) |
			((status & IM__EXTER_INTER_2D) >> 12) |
			((status & IM__EXTER_BOUND_2D) >>  6) |
			((status & IM__EXTER_EXTER_2D)));
}

STATIC_INLINE(void)
IM__PRINTF(const char *label, cl_int status)
{
	printf("[%s]\n%d %d %d\n%d %d %d\n%d %d %d\n",
		   label,
		   (status & IM__INTER_INTER_2D) != 0 ? 1 : 0,
		   (status & IM__INTER_BOUND_2D) != 0 ? 1 : 0,
		   (status & IM__INTER_EXTER_2D) != 0 ? 1 : 0,
		   (status & IM__BOUND_INTER_2D) != 0 ? 1 : 0,
		   (status & IM__BOUND_BOUND_2D) != 0 ? 1 : 0,
		   (status & IM__BOUND_EXTER_2D) != 0 ? 1 : 0,
		   (status & IM__EXTER_INTER_2D) != 0 ? 1 : 0,
		   (status & IM__EXTER_BOUND_2D) != 0 ? 1 : 0,
		   (status & IM__EXTER_EXTER_2D) != 0 ? 1 : 0);
}

#define PT_EQ(A,B)		(FP_EQUALS((A).x,(B).x) && FP_EQUALS((A).y,(B).y))
#define PT_NE(A,B)		(FP_NEQUALS((A).x,(B).x) || FP_NEQUALS((A).y,(B).y))

STATIC_FUNCTION(cl_int)
geom_relate_point_point(kern_context *kcxt,
						const pg_geometry_t *geom1,
						const pg_geometry_t *geom2)
{
	const char *pos1 = NULL;
	const char *pos2 = NULL;
	cl_int		nloops1;
	cl_int		nloops2;
	cl_int		retval;
	cl_bool		twist_retval = false;

	assert((geom1->type == GEOM_POINTTYPE ||
			geom1->type == GEOM_MULTIPOINTTYPE) &&
		   (geom2->type == GEOM_POINTTYPE ||
			geom2->type == GEOM_MULTIPOINTTYPE));

	/* shortcut if either-geometry is empty */
	if (geom1->nitems == 0 && geom2->nitems == 0)
		return IM__EXTER_EXTER_2D;
	if (geom1->nitems == 0)
		return IM__EXTER_INTER_0D | IM__EXTER_EXTER_2D;
	if (geom2->nitems == 0)
		return IM__INTER_EXTER_0D | IM__EXTER_EXTER_2D;

	/*
	 * micro optimization: geom2 should have smaller number of items
	 */
	if (geom2->type != GEOM_POINTTYPE)
	{
		if (geom1->type == GEOM_POINTTYPE ||
			geom1->nitems < geom2->nitems)
		{
			__swap(geom1, geom2);
			twist_retval = true;
		}
	}
	retval = IM__EXTER_EXTER_2D;

	nloops1 = (geom1->type == GEOM_POINTTYPE ? 1 : geom1->nitems);
	nloops2 = (geom2->type == GEOM_POINTTYPE ? 1 : geom2->nitems);
	for (int base=0; base < nloops2; base += CL_LONG_NBITS)
	{
		cl_ulong	matched2 = 0;
		cl_ulong	__mask;
		pg_geometry_t temp;

		for (int i=0; i < nloops1; i++)
		{
			POINT2D		pt1;
			bool		matched1 = false;

			if (geom1->type == GEOM_POINTTYPE)
				__loadPoint2d(&pt1, geom1->rawdata, 0);
			else
			{
				pos1 = geometry_load_subitem(&temp, geom1, pos1, i, kcxt);
				if (!pos1)
					return -1;
				__loadPoint2d(&pt1, temp.rawdata, 0);
			}

			for (int j=0; j < nloops2; j++)
			{
				POINT2D		pt2;

				if (geom2->type == GEOM_POINTTYPE)
					__loadPoint2d(&pt2, geom2->rawdata, 0);
				else
				{
					pos2 = geometry_load_subitem(&temp, geom2, pos2, j, kcxt);
					if (!pos2)
						return -1;
					__loadPoint2d(&pt2, temp.rawdata, 0);
				}

				if (pt1.x == pt2.x && pt1.y == pt2.y)
				{
					retval |= IM__INTER_INTER_0D;
					matched1 = true;
					if (j >= base && j < base + CL_LONG_NBITS)
						matched2 |= (1UL << (j - base));
				}
			}
			if (!matched1)
				retval |= IM__INTER_EXTER_0D;
		}
		if (base + CL_LONG_NBITS >= nloops2)
			__mask = (1UL << (nloops2 - base)) - 1;
		else
			__mask = ~0UL;

		if (__mask != matched2)
		{
			retval |= IM__EXTER_INTER_0D;
			break;
		}
	}
	return (twist_retval ? IM__TWIST(retval) : retval);
}

STATIC_FUNCTION(cl_int)
geom_relate_point_line(kern_context *kcxt,
					   const pg_geometry_t *geom1,
					   const pg_geometry_t *geom2)
{
	const char *pos1 = NULL;
	const char *pos2 = NULL;
	cl_uint		nloops1;
	cl_uint		nloops2;
	cl_int		retval;
	cl_uint		unitsz;

	assert((geom1->type == GEOM_POINTTYPE ||
			geom1->type == GEOM_MULTIPOINTTYPE) &&
		   (geom2->type == GEOM_LINETYPE ||
			geom2->type == GEOM_MULTILINETYPE));
	/* shortcut if either-geometry is empty */
	if (geom1->nitems == 0 && geom2->nitems == 0)
		return IM__EXTER_EXTER_2D;
    if (geom1->nitems == 0)
		return IM__EXTER_INTER_1D | IM__EXTER_BOUND_0D | IM__EXTER_EXTER_2D;
	if (geom2->nitems == 0)
		return IM__INTER_EXTER_0D | IM__EXTER_EXTER_2D;

	nloops1 = (geom1->type == GEOM_POINTTYPE ? 1 : geom1->nitems);
	nloops2 = (geom2->type == GEOM_LINETYPE  ? 1 : geom2->nitems);

	retval = IM__EXTER_INTER_1D | IM__EXTER_EXTER_2D;
	unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(geom2->flags);
	for (int base=0; base < nloops2; base += CL_LONG_NBITS)
	{
		cl_ulong	head_matched = 0UL;
		cl_ulong	tail_matched = 0UL;
		cl_ulong	boundary_mask = 0UL;

		/* walks on for each points */
		for (int i=0; i < nloops1; i++)
		{
			POINT2D		P;
			cl_bool		matched = false;

			/* fetch a point */
			if (geom1->type == GEOM_POINTTYPE)
				__loadPoint2d(&P, geom1->rawdata, 0);
			else
			{
				pg_geometry_t   __temp1;

				pos1 = geometry_load_subitem(&__temp1, geom1, pos1, i);
				if (!pos1)
					return -1;
				__loadPoint2d(&P, __temp1.rawdata, 0);
			}

			/* walks on for each linestrings */
			for (int j=0; j < nloops2; j++)
			{
				pg_geometry_t	__line_data;
				const pg_geometry_t *line;
				const char	   *lpos;
				POINT2D			Q1, Q2;
				cl_bool			has_boundary;

				/* fetch a linestring */
				if (geom2->type == GEOM_LINETYPE)
					line = geom2;
				else
				{
					pos2 = geometry_load_subitem(&__line_data, geom2, pos2, j);
					if (!pos2)
						return -1;
					line = &__line_data;
				}
				/* walks on vertex of the line edges */
				__loadPoint2dIndex(&Q2, line->rawdata, unitsz, line->nitems-1);
				lpos = __loadPoint2d(&Q1, line->rawdata, unitsz);
				has_boundary = PT_NE(Q1,Q2);
				if (has_boundary && (j >= base && j < base + CL_LONG_NBITS))
					boundary_mask |= (1UL << (j - base));
				for (int k=2; k <= line->nitems; k++, Q1=Q2)
				{
					lpos = __loadPoint2d(&Q2, lpos, unitsz);

					if (has_boundary)
					{
						if (k==2 && PT_EQ(P,Q1))
						{
							/* boundary case handling (head) */
							retval |= IM__INTER_BOUND_0D;
							matched = true;
							if (j >= base && j < base + CL_LONG_NBITS)
								head_matched |= (1UL << (j - base));
							continue;
						}
						else if (k == line->nitems && PT_EQ(P,Q2))
						{
							/* boundary case handling (tail) */
							retval |= IM__INTER_BOUND_0D;
							matched = true;
							if (j >= base && j < base + CL_LONG_NBITS)
								tail_matched |= (1UL << (j - base));
							continue;
						}
					}
					if (__geom_segment_side(&Q1, &Q2, &P) == 0 &&
						__geom_pt_in_seg(&P, &Q1, &Q2))
					{
						retval |= IM__INTER_INTER_0D;
						matched = true;
					}
				}
			}
			/*
			 * This point is neither interior nor boundary of linestrings
			 */
			if (!matched)
				retval |= IM__INTER_EXTER_0D;
		}
		/*
		 * If herea are any linestring-edges not referenced by the points,
		 * it needs to set EXTER-BOUND item.
		 */
		if (head_matched != boundary_mask || tail_matched != boundary_mask)
		{
			retval |= IM__EXTER_BOUND_0D;
			break;
		}
	}
	return retval;
}

STATIC_FUNCTION(cl_int)
geom_relate_point_triangle(kern_context *kcxt,
						   const pg_geometry_t *geom1,
						   const pg_geometry_t *geom2)
{
	cl_uint			nloops;
	cl_int			retval;
	const char	   *pos = NULL;
	geom_bbox_2d	bbox;

	assert((geom1->type == GEOM_POINTTYPE ||
			geom1->type == GEOM_MULTIPOINTTYPE) &&
		   (geom2->type == GEOM_TRIANGLETYPE));
	if (geom1->nitems == 0)
	{
		if (geom2->nitems == 0)
			return IM__EXTER_EXTER_2D;
		return IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;
	}
	else if (geom2->nitems == 0)
		return IM__INTER_EXTER_0D | IM__EXTER_EXTER_2D;

	nloops = (geom1->type == GEOM_POINTTYPE ? 1 : geom1->nitems);
	retval = IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;

	if (geom2->bbox)
		memcpy(&bbox, geom2->bbox, sizeof(geom_bbox_2d));
	for (int i=0; i < nloops; i++)
	{
		POINT2D		pt;
		cl_int		status;

		if (geom1->type == GEOM_POINTTYPE)
			__loadPoint2d(&pt, geom1->rawdata, 0);
		else
		{
			pg_geometry_t __temp;

			pos = geometry_load_subitem(&__temp, geom1, pos, i, kcxt);
			if (!pos)
				return -1;
			__loadPoint2d(&pt, __temp.rawdata, 0);
		}

		/* shortcut if boundary-box is available */
		if (geom2->bbox && (pt.x < bbox.xmin || pt.x > bbox.xmax ||
							pt.y < bbox.ymin || pt.y > bbox.ymax))
			status = PT_OUTSIDE;
		else
			status = __geom_contains_point(geom2, &pt, kcxt);

		if (status == PT_INSIDE)
			retval |= IM__INTER_INTER_0D;
		else if (status == PT_BOUNDARY)
			retval |= IM__INTER_BOUND_0D;
		else if (status == PT_OUTSIDE)
			retval |= IM__INTER_EXTER_0D;
		else
			return -1;
	}
	if (retval & IM__INTER_INTER_0D)
		retval |= IM__EXTER_INTER_0D;
	if (retval & IM__INTER_BOUND_0D)
		retval |= IM__EXTER_BOUND_0D;

	return retval;
}

STATIC_FUNCTION(cl_int)
geom_relate_point_poly(kern_context *kcxt,
					   const pg_geometry_t *geom1,
					   const pg_geometry_t *geom2)
{
	const char *pos1 = NULL;
	const char *pos2 = NULL;
	cl_uint		nloops1;
	cl_uint		nloops2;
	cl_int		retval = IM__EXTER_EXTER_2D;

	assert((geom1->type == GEOM_POINTTYPE ||
			geom1->type == GEOM_MULTIPOINTTYPE) &&
           (geom2->type == GEOM_POLYGONTYPE ||
			geom2->type == GEOM_MULTIPOLYGONTYPE));
	if (geom1->nitems == 0)
	{
		if (geom2->nitems == 0)
			return IM__EXTER_EXTER_2D;
		return IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;
	}
	else if (geom2->nitems == 0)
		return IM__INTER_EXTER_0D | IM__EXTER_EXTER_2D;

	nloops1 = (geom1->type == GEOM_POINTTYPE   ? 1 : geom1->nitems);
	nloops2 = (geom2->type == GEOM_POLYGONTYPE ? 1 : geom2->nitems);

	retval = IM__EXTER_EXTER_2D;
	for (int i=0; i < nloops1; i++)
	{
		pg_geometry_t temp;
		POINT2D		pt;
		cl_bool		matched = false;

		if (geom1->type == GEOM_POINTTYPE)
			__loadPoint2d(&pt, geom1->rawdata, 0);
		else
		{
			pos1 = geometry_load_subitem(&temp, geom1, pos1, i, kcxt);
			if (!pos1)
				return -1;
			__loadPoint2d(&pt, temp.rawdata, 0);
		}

		for (int j=0; j < nloops2; j++)
		{
			const pg_geometry_t *poly;
			cl_int			status;

			if (geom2->type == GEOM_POLYGONTYPE)
				poly = geom2;
			else
			{
				pos2 = geometry_load_subitem(&temp, geom2, pos2, j, kcxt);
				if (!pos2)
					return -1;
				poly = &temp;
			}
			/* skip empty polygon */
			if (poly->nitems == 0)
				continue;
			retval |= IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D;

			/* shortcut by boundary-box, if any */
			if (poly->bbox)
			{
				geom_bbox_2d	bbox;

				memcpy(&bbox, poly->bbox, sizeof(geom_bbox_2d));
				if (pt.x < bbox.xmin || pt.x > bbox.xmax ||
					pt.y < bbox.ymin || pt.y > bbox.ymax)
					continue;
			}

			/* dive into the polygon */
			status = __geom_point_in_polygon(poly, &pt, kcxt);
			if (status == PT_INSIDE)
			{
				matched = true;
				retval |= IM__INTER_INTER_0D;
			}
			else if (status == PT_BOUNDARY)
			{
				matched = true;
				retval |= IM__INTER_BOUND_0D;
			}
			else if (status != PT_OUTSIDE)
				return -1;	/* error */
		}
		if (!matched)
			retval |= IM__INTER_EXTER_0D;
	}
	return retval;
}

STATIC_FUNCTION(cl_int)
__geom_relate_seg_line(kern_context *kcxt,
					   POINT2D P1, cl_bool p1_is_head,
					   POINT2D P2, cl_bool p2_is_tail,
					   const pg_geometry_t *geom, cl_uint start)
{
	const char *gpos = NULL;
	cl_int		retval = IM__EXTER_EXTER_2D;
	cl_bool		p1_contained = false;
	cl_bool		p2_contained = false;
	cl_uint		nloops;
	cl_uint		index = start;

	if (CHECK_KERNEL_STACK_DEPTH(kcxt))
	{
		STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_RECURSION_TOO_DEEP,
						   "too deep recursive calls");
		return -1;
	}

	nloops = (geom->type == GEOM_LINETYPE ? 1 : geom->nitems);
	for (int k=0; k < nloops; k++)
	{
		pg_geometry_t __temp;
		const pg_geometry_t *line;
		const char *ppos = NULL;
		POINT2D		Q1, Q2;
		cl_uint		unitsz;
		cl_int		__j = 2;

		if (geom->type == GEOM_LINETYPE)
			line = geom;
		else
		{
			gpos = geometry_load_subitem(&__temp, geom, gpos, k, kcxt);
			if (!gpos)
				return -1;
			line = &__temp;
		}
		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(line->flags);

		if (start == 0)
		{
			ppos = __loadPoint2d(&Q1, line->rawdata, unitsz);
			index++;
		}
		else if (index + line->nitems <= start)
		{
			index += line->nitems;
			continue;	/* skip this sub-line */
		}
		else
		{
			assert(index - start < line->nitems);
			ppos = __loadPoint2dIndex(&Q1, line->rawdata, unitsz,
									  index - start);
			index++;
			__j = index - start + 2;
			start = 0;
		}

		for (int j=__j; j <= line->nitems; j++, index++, Q1=Q2)
		{
			cl_bool		q1_is_head = (j==2);
			cl_bool		q2_is_tail = (j==line->nitems);
			cl_int		qp1, qp2;
			cl_int		p1_in_qq, p2_in_qq;
			cl_int		status;

			ppos = __loadPoint2d(&Q2, ppos, unitsz);
			qp1 = __geom_segment_side(&Q1, &Q2, &P1);
			qp2 = __geom_segment_side(&Q1, &Q2, &P2);
			if ((qp1 > 0 && qp2 > 0) || (qp1 < 0 && qp2 < 0))
				continue;	/* no intersection */

			p1_in_qq = __geom_pt_within_seg(&P1,&Q1,&Q2);
			p2_in_qq = __geom_pt_within_seg(&P2,&Q1,&Q2);

			/* P1 is on Q1-Q2 */
			if (qp1==0 && p1_in_qq != PT_OUTSIDE)
			{
				p1_contained = true;
				if (p1_is_head)
				{
					if ((q1_is_head && PT_EQ(P1,Q1)) ||
						(q2_is_tail && PT_EQ(P1,Q2)))
						retval |= IM__BOUND_BOUND_0D;
					else
						retval |= IM__BOUND_INTER_0D;
				}
				else
					retval |= IM__INTER_INTER_0D;
			}

			/* P2 is on Q1-Q2 */
			if (qp2==0 && p2_in_qq != PT_OUTSIDE)
			{
				p2_contained = true;
				if (p2_is_tail)
				{
					if ((q1_is_head && PT_EQ(P2,Q1)) ||
						(q2_is_tail && PT_EQ(P2,Q2)))
						retval |= IM__BOUND_BOUND_0D;
					else
						retval |= IM__BOUND_INTER_0D;
				}
				else
					retval |= IM__INTER_INTER_0D;
			}

			/* P1-P2 and Q1-Q2 are colinear */
			if (qp1 == 0 && qp2 == 0)
			{
				if (p1_in_qq != PT_OUTSIDE &&
					p2_in_qq != PT_OUTSIDE)
				{
					/* P1-P2 is fully contained by Q1-Q2 */
					p1_contained = p2_contained = true;
					if (PT_EQ(P1,P2))
						retval |= IM__INTER_INTER_0D;
					else
						retval |= IM__INTER_INTER_1D;
					goto out;
				}
				else if (p1_in_qq != PT_OUTSIDE &&
						 p2_in_qq == PT_OUTSIDE)
				{
					/* P1 is in Q1-Q2, but P2 is not, so Qx-P2 shall remain */
					p1_contained = true;
					if (__geom_pt_within_seg(&Q1,&P1,&P2) == PT_INSIDE)
					{
						P1 = Q1;
						p1_is_head = false;
						retval |= IM__INTER_INTER_1D;
					}
					else if (__geom_pt_within_seg(&Q2,&P1,&P2) == PT_INSIDE)
					{
						P1 = Q2;
						p1_is_head = false;
						retval |= IM__INTER_INTER_1D;
					}
				}
				else if (p1_in_qq == PT_OUTSIDE &&
						 p2_in_qq != PT_OUTSIDE)
				{
					/* P2 is in Q1-Q2, but P1 is not, so Qx-P1 shall remain */
					p2_contained = true;
					if (__geom_pt_within_seg(&Q1,&P1,&P2) == PT_INSIDE)
					{
						P2 = Q1;
						p2_is_tail = false;
						retval |= IM__INTER_INTER_1D;
					}
					else if (__geom_pt_within_seg(&Q2,&P1,&P2) == PT_INSIDE)
					{
						P2 = Q2;
						p2_is_tail = false;
						retval |= IM__INTER_INTER_1D;
					}
				}
				else if (__geom_pt_within_seg(&Q1,&P1,&Q2) != PT_OUTSIDE &&
						 __geom_pt_within_seg(&Q2,&Q1,&P2) != PT_OUTSIDE)
				{
					/* P1-Q1-Q2-P2 */
					if (PT_NE(P1,Q1))
					{
						status = __geom_relate_seg_line(kcxt,
														P1, p1_is_head,
														Q1, false,
														geom, index+1);
						if (status < 0)
							return -1;
						retval |= status;
					}
					if (PT_NE(Q2,P2))
					{
						status = __geom_relate_seg_line(kcxt,
														Q2, false,
														P2, p2_is_tail,
														geom, index+1);
						if (status < 0)
							return -1;
						retval |= status;
					}
					goto out;
				}
				else if (__geom_pt_within_seg(&Q2,&P1,&Q1) != PT_OUTSIDE &&
						 __geom_pt_within_seg(&Q1,&Q2,&P2) != PT_OUTSIDE)
				{
					/* P1-Q2-Q1-P2 */
					if (PT_NE(P1,Q2))
					{
						status = __geom_relate_seg_line(kcxt,
														P1, p1_is_head,
														Q2, false,
														geom, index+1);
						if (status < 0)
							return -1;
						retval |= status;
					}
					if (PT_NE(Q1,P2))
					{
						status = __geom_relate_seg_line(kcxt,
														Q1, false,
														P2, p2_is_tail,
														geom, index+1);
						if (status < 0)
							return -1;
						retval |= status;
					}
					goto out;
				}
				else
				{
					/* elsewhere P1-P2 and Q1-Q2 have no intersection */
				}
			}
			else
			{
				cl_int	pq1 = __geom_segment_side(&P1,&P2,&Q1);
				cl_int	pq2 = __geom_segment_side(&P1,&P2,&Q2);

				/* P1-P2 and Q1-Q2 crosses mutually */
				if (((pq1 > 0 && pq2 < 0) || (pq1 < 0 && pq2 > 0)) &&
					((qp1 > 0 && qp2 < 0) || (qp1 < 0 && qp2 > 0)))
				{
					retval |= IM__INTER_INTER_0D;
				}
			}
		}
	}
	if (PT_NE(P1,P2))
		retval |= IM__INTER_EXTER_1D;
out:
	if (p1_is_head && !p1_contained)
		retval |= IM__BOUND_EXTER_0D;
	if (p2_is_tail && !p2_contained)
		retval |= IM__BOUND_EXTER_0D;
	return retval;
}

STATIC_FUNCTION(cl_int)
geom_relate_line_line(kern_context *kcxt,
					  const pg_geometry_t *geom1,
					  const pg_geometry_t *geom2)
{
	const char *gpos = NULL;
	const char *ppos = NULL;
	const pg_geometry_t *line;
	pg_geometry_t __temp;
	POINT2D		P1, P2;
	cl_uint		nloops;
	cl_uint		unitsz;
	cl_int		retval1 = IM__EXTER_EXTER_2D;
	cl_int		retval2 = IM__EXTER_EXTER_2D;

	assert((geom1->type == GEOM_LINETYPE ||
			geom1->type == GEOM_MULTILINETYPE) &&
		   (geom2->type == GEOM_LINETYPE ||
			geom2->type == GEOM_MULTILINETYPE));
	/* special empty cases */
	if (geom1->nitems == 0)
	{
		if (geom2->nitems == 0)
			return IM__EXTER_EXTER_2D;
		return IM__EXTER_INTER_1D | IM__EXTER_BOUND_0D | IM__EXTER_EXTER_2D;
	}
	else if (geom2->nitems == 0)
		return IM__INTER_EXTER_1D | IM__BOUND_EXTER_0D | IM__EXTER_EXTER_2D;

	/* 1st loop */
	nloops = (geom1->type == GEOM_LINETYPE ? 1 : geom1->nitems);
	for (int k=0; k < nloops; k++)
	{
		if (geom1->type == GEOM_LINETYPE)
			line = geom1;
		else
		{
			gpos = geometry_load_subitem(&__temp, geom1, gpos, k, kcxt);
			if (!gpos)
				return -1;
			line = &__temp;
		}
		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(line->flags);
		ppos = __loadPoint2d(&P1, line->rawdata, unitsz);
		for (int i=2; i <= line->nitems; i++, P1=P2)
		{
			ppos = __loadPoint2d(&P2, ppos, unitsz);
			retval1 |= __geom_relate_seg_line(kcxt,
											  P1, i==2,
											  P2, i==line->nitems,
											  geom2, 0);
		}
	}
	/* 2nd loop (twisted) */
	nloops = (geom2->type == GEOM_LINETYPE ? 1 : geom2->nitems);
	for (int k=0; k < nloops; k++)
	{
		if (geom2->type == GEOM_LINETYPE)
			line = geom2;
		else
		{
			gpos = geometry_load_subitem(&__temp, geom2, gpos, k, kcxt);
			if (!gpos)
				return -1;
			line = &__temp;
		}
		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(line->flags);
		ppos = __loadPoint2d(&P1, line->rawdata, unitsz);
		for (int j=2; j <= line->nitems; j++, P1=P2)
		{
			ppos = __loadPoint2d(&P2, ppos, unitsz);
			retval2 |= __geom_relate_seg_line(kcxt,
											  P1, j==2,
											  P2, j==line->nitems,
											  geom1, 0);
		}
	}
	return retval1 | IM__TWIST(retval2);
}

#define IM__LINE_HEAD_CONTAINED		01000000000U
#define IM__LINE_TAIL_CONTAINED		02000000000U

STATIC_FUNCTION(cl_int)
__geom_relate_seg_triangle(kern_context *kcxt,
						   const POINT2D &P1, cl_bool p1_is_head,
						   const POINT2D &P2, cl_bool p2_is_tail,
						   const pg_geometry_t *ring)
{
	POINT2D		Q1,Q2, Pc;
	const char *pos;
	cl_uint		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(ring->flags);
	cl_char		p1_location = '?';
	cl_char		p2_location = '?';
	cl_char		pc_location = '?';
	cl_int		wn1 = 0;
	cl_int		wn2 = 0;
	cl_int		wnc = 0;
	cl_int		ncrosses = 0;
	cl_int		retval = 0;

	Pc.x = (P1.x + P2.x) / 2.0;
	Pc.y = (P2.y + P2.y) / 2.0;

	pos = __loadPoint2d(&Q1, ring->rawdata, unitsz);
	__loadPoint2dIndex(&Q2, ring->rawdata, unitsz, ring->nitems-1);
	if (PT_NE(Q1,Q2))
	{
		STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
					  "triangle corruption: unclosed ring");
		return -1;
	}

	for (int j=2; j <= ring->nitems; j++, Q1=Q2)
	{
		cl_int		qp1, qp2, qpc;

		pos = __loadPoint2d(&Q2, pos, unitsz);
		/*
		 * Update the state of winding number algorithm to determine
		 * the location of P1/P2 whether they are inside or outside
		 * of the Q1-Q2 edge.
		 */
		qp1 = __geom_segment_side(&Q1, &Q2, &P1);
		if (qp1 < 0 && Q1.y <= P1.y && P1.y < Q2.y)
			wn1++;
		else if (qp1 > 0 && Q2.y <= P1.y && P1.y < Q1.y)
			wn1--;
		if (qp1 == 0 && __geom_pt_within_seg(&P1,&Q1,&Q2) != PT_OUTSIDE)
			p1_location = 'B';

		qp2 = __geom_segment_side(&Q1, &Q2, &P2);
		if (qp2 < 0 && Q1.y <= P2.y && P2.y < Q2.y)
			wn2++;
		else if (qp2 > 0 && Q2.y <= P2.y && P2.y < Q1.y)
			wn2--;
		if (qp2 == 0 && __geom_pt_within_seg(&P2,&Q1,&Q2) != PT_OUTSIDE)
			p2_location = 'B';

		qpc = __geom_segment_side(&Q1, &Q2, &Pc);
		if (qpc < 0 && Q1.y <= Pc.y && Pc.y < Q2.y)
			wnc++;
		else if (qpc > 0 && Q2.y <= Pc.y && Pc.y < Q1.y)
			wnc--;
		if (qpc == 0 && __geom_pt_within_seg(&Pc,&Q1,&Q2) != PT_OUTSIDE)
			pc_location = 'B';

		if ((qp1 > 0 && qp2 > 0) || (qp1 < 0 && qp2 < 0))
		{
			/* P1-P2 and Q1-Q2 are not crosses/touched */
		}
		else if (!qp1 && !qp2)
		{
			/* P1-P2 and Q1-Q2 are colinear */
			cl_int	p1_in_qq = __geom_pt_within_seg(&P1,&Q1,&Q2);
			cl_int	p2_in_qq = __geom_pt_within_seg(&P2,&Q1,&Q2);
			cl_int	q1_in_pp = __geom_pt_within_seg(&Q1,&P1,&P2);
			cl_int	q2_in_pp = __geom_pt_within_seg(&Q2,&P1,&P2);

			if (p1_in_qq != PT_OUTSIDE &&
				p2_in_qq != PT_OUTSIDE)
			{
				/* P1-P2 is fully contained by Q1-Q2 */
				if (p1_is_head)
					retval |= IM__BOUND_BOUND_0D | IM__LINE_HEAD_CONTAINED;
				if (p2_is_tail)
					retval |= IM__BOUND_BOUND_0D | IM__LINE_TAIL_CONTAINED;
				if (PT_EQ(P1,P2))
				{
					if (!p1_is_head && !p2_is_tail)
						retval |= IM__INTER_BOUND_0D;
				}
				else
					retval |= IM__INTER_BOUND_1D;
				return retval;
			}
			else if (p1_in_qq != PT_OUTSIDE &&
					 p2_in_qq == PT_OUTSIDE)
			{
				/* P1 is contained by Q1-Q2, but P2 is not */
				if (p1_is_head)
					retval |= IM__BOUND_BOUND_0D | IM__LINE_HEAD_CONTAINED;
				if (p1_in_qq == PT_INSIDE)
					retval |= IM__INTER_BOUND_1D;
				else if ((PT_EQ(P1,Q1) && q2_in_pp != PT_OUTSIDE) ||
						 (PT_EQ(P1,Q2) && q1_in_pp != PT_OUTSIDE))
					retval |= IM__INTER_BOUND_1D;
			}
			else if (p1_in_qq == PT_OUTSIDE &&
					 p2_in_qq != PT_OUTSIDE)
			{
				/* P2 is contained by Q1-Q2, but P1 is not */
				if (p2_is_tail)
					retval |= IM__BOUND_BOUND_0D | IM__LINE_TAIL_CONTAINED;
				if (p2_in_qq == PT_INSIDE)
					retval |= IM__INTER_BOUND_1D;
				else if ((PT_EQ(P2,Q1) && q2_in_pp != PT_OUTSIDE) ||
						 (PT_EQ(P2,Q2) && q1_in_pp != PT_OUTSIDE))
					retval |= IM__INTER_BOUND_1D;
			}
			else if (q1_in_pp != PT_OUTSIDE &&
					 q2_in_pp != PT_OUTSIDE)
			{
				/* Q1-Q2 is fully contained by P1-P2 */
				retval |= IM__INTER_BOUND_1D;
			}
		}
		else
		{
			cl_int	pq1 = __geom_segment_side(&Q1,&P1,&P2);
			cl_int	pq2 = __geom_segment_side(&Q2,&P1,&P2);
#if 0
			printf("P1(%d,%d)[%d]-P2(%d,%d)[%d] Q1(%d,%d)[%d]-Q2(%d,%d)[%d]\n",
				   (int)P1.x, (int)P1.y, qp1,
				   (int)P2.x, (int)P2.y, qp2,
				   (int)Q1.x, (int)Q1.y, pq1,
				   (int)Q2.x, (int)Q2.y, pq2);
#endif
			if (qp1==0 && ((pq1 >= 0 && pq2 <= 0) ||
						   (pq1 <= 0 && pq2 >= 0)))
			{
				/* P1 is on Q1-Q2 */
				if (p1_is_head)
					retval |= IM__BOUND_BOUND_0D | IM__LINE_HEAD_CONTAINED;
				else
					retval |= IM__INTER_BOUND_0D;
			}
			else if (qp2==0 && ((pq1 >= 0 && pq2 <= 0) ||
								(pq1 <= 0 && pq2 >= 0)))
			{
				/* P2 is on Q1-Q2 */
				if (p2_is_tail)
					retval |= IM__BOUND_BOUND_0D | IM__LINE_TAIL_CONTAINED;
				else
					retval |= IM__INTER_BOUND_0D;
			}
			else if (pq1==0 && ((qp1 >= 0 && qp2 <= 0) ||
								(qp1 <= 0 && qp2 >= 0)))
			{
				/* Q1 on P1-P2 */
				retval |= IM__INTER_BOUND_0D;
			}
			else if (pq2==0 && ((qp1 >= 0 && qp2 <= 0) ||
								(qp1 <= 0 && qp2 >= 0)))
			{
				/* Q2 on P1-P2 */
				retval |= IM__INTER_BOUND_0D;
			}
			else if (((qp1 > 0 && qp2 < 0) || (qp1 < 0 && qp2 > 0)) &&
					 ((pq1 > 0 && pq2 < 0) || (pq1 < 0 && pq2 > 0)))
			{
				/* P1-P2 and Q1-Q2 crosses */
				retval |= IM__INTER_BOUND_0D;
				ncrosses++;
			}
		}
	}
	/* check P1,P2 locations */
	if (p1_location == '?')
		p1_location = (wn1 == 0 ? 'E' : 'I');
	if (p2_location == '?')
		p2_location = (wn2 == 0 ? 'E' : 'I');
	if (pc_location == '?')
		pc_location = (wnc == 0 ? 'E' : 'I');

	if (p1_location == 'I' || p2_location == 'I')
	{
		if (PT_EQ(P1,P2))
			retval |= IM__INTER_INTER_0D;
		else
			retval |= IM__INTER_INTER_1D;
	}
	if (p1_location == 'E' || p2_location == 'E')
	{
		if (PT_EQ(P1,P2))
			retval |= IM__INTER_EXTER_0D;
		else
			retval |= IM__INTER_EXTER_1D;
	}
	if (p1_location == 'B' && p2_location == 'B')
	{
		if (pc_location == 'I')
			retval |= IM__INTER_INTER_1D;
	}
	else if ((p1_location == 'E' && p2_location == 'B') ||
			 (p1_location == 'B' && p2_location == 'E') ||
			 (p1_location == 'E' && p2_location == 'E'))
	{
		if (ncrosses > 0)
			retval |= IM__INTER_INTER_1D;
		retval |= IM__INTER_EXTER_1D;
	}
	if (p1_is_head && p1_location == 'I')
		retval |= IM__BOUND_INTER_0D | IM__LINE_HEAD_CONTAINED;
	if (p2_is_tail && p2_location == 'I')
		retval |= IM__BOUND_INTER_0D | IM__LINE_TAIL_CONTAINED;
	if ((p1_is_head && p1_location == 'E') ||
		(p2_is_tail && p2_location == 'E'))
		retval |= IM__BOUND_EXTER_0D;
#if 0
	printf("P1(%d,%d)[%c]-P2(%d,%d)[%c]\n",
		   (int)P1.x, (int)P1.y, p1_location,
		   (int)P2.x, (int)P2.y, p2_location);
#endif
	return retval;
}

STATIC_FUNCTION(cl_int)
geom_relate_line_triangle(kern_context *kcxt,
						  const pg_geometry_t *geom1,
						  const pg_geometry_t *geom2)
{
	const char *gpos = NULL;
	const char *ppos = NULL;
	cl_int		nloops;
	cl_uint		unitsz;
	cl_int		retval, __mask;
	geom_bbox_2d bbox2;

	assert((geom1->type == GEOM_LINETYPE ||
			geom1->type == GEOM_MULTILINETYPE) &&
		   (geom2->type == GEOM_TRIANGLETYPE));
	/* special empty cases */
	if (geom1->nitems == 0)
	{
		if (geom2->nitems == 0)
			return IM__EXTER_EXTER_2D;
		return IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;
	}
	else if (geom2->nitems == 0)
		return IM__INTER_EXTER_1D | IM__BOUND_EXTER_0D | IM__EXTER_EXTER_2D;

	/* sanity check of the triangle */
	if (geom2->nitems != 4)
	{
		STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
					  "triangle must have exactly 4 points");
		return -1;
	}
	retval = IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;

	/* shortcut if both of geometry has bounding box */
	if (geom2->bbox)
	{
		memcpy(&bbox2, geom2->bbox, sizeof(geom_bbox_2d));
		if (geom1->bbox)
		{
			geom_bbox_2d	bbox1;

			memcpy(&bbox1, geom1->bbox, sizeof(geom_bbox_2d));

			if (bbox1.xmax < bbox2.xmin || bbox1.xmin > bbox2.xmax ||
				bbox1.ymax < bbox2.ymin || bbox2.ymin > bbox2.ymax)
				return (retval | IM__INTER_EXTER_1D | IM__BOUND_EXTER_0D);
		}
	}

	nloops = (geom1->type == GEOM_LINETYPE ? 1 : geom1->nitems);
    for (int k=0; k < nloops; k++)
	{
		const pg_geometry_t *line;
		cl_uint		nitems;
		cl_bool		p1_is_head = true;
		POINT2D		P1, P2;

		if (geom1->type == GEOM_LINETYPE)
            line = geom1;
		else
		{
			pg_geometry_t __temp;

			gpos = geometry_load_subitem(&__temp, geom1, gpos, k, kcxt);
			if (!gpos)
				return -1;
			line = &__temp;
		}
		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(line->flags);
		if (line->nitems == 0)
			continue;
		/* decrement nitems if tail items are duplicated */
		__loadPoint2dIndex(&P2, line->rawdata, unitsz, line->nitems-1);
		for (nitems = line->nitems; nitems >= 2; nitems--)
		{
			__loadPoint2dIndex(&P1, line->rawdata, unitsz, nitems-2);
			if (PT_EQ(P1, P2))
				break;
		}
		/* checks for each edge */
		ppos = __loadPoint2d(&P1, line->rawdata, unitsz);
		for (int i=2; i <= line->nitems; i++)
		{
			ppos = __loadPoint2d(&P2, ppos, unitsz);
			if (PT_EQ(P1,P2))
				continue;
			/* shortcut, if this edge is obviously disjoint */
			if (geom2->bbox && (Max(P1.x,P2.x) < bbox2.xmin ||
								Min(P1.x,P2.x) > bbox2.xmax ||
								Max(P1.y,P2.y) < bbox2.ymin ||
								Min(P1.y,P2.y) > bbox2.ymax))
			{
				retval |= IM__INTER_EXTER_1D;
			}
			else
			{
				retval |= __geom_relate_seg_triangle(kcxt,
													 P1, p1_is_head,
													 P2, i==line->nitems,
													 geom2);
			}
			p1_is_head = false;
			P1=P2;
		}
	}
	__mask = (IM__LINE_HEAD_CONTAINED | IM__LINE_TAIL_CONTAINED);
	if ((retval & __mask) != __mask)
		retval |= IM__BOUND_EXTER_0D;

	return (retval & IM__MASK_FULL);
}

STATIC_FUNCTION(cl_int)
__geom_relate_seg_polygon(kern_context *kcxt,
						  const POINT2D &P1, cl_bool p1_is_head,
						  const POINT2D &P2, cl_bool p2_is_tail,
						  const pg_geometry_t *geom,
						  cl_int nskips, cl_bool last_polygons)
{
	const char *pos1 = NULL;
	const char *pos2 = NULL;
	pg_geometry_t __polyData;
	POINT2D		Pc;
	cl_int		nloops;
	cl_int		retval = 0;
	cl_int		status;
	cl_int		nrings = 0;
	cl_uint		__nrings_next;

	if (CHECK_KERNEL_STACK_DEPTH(kcxt))
	{
		STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_RECURSION_TOO_DEEP,
						   "too deep recursive calls");
		return -1;
	}
	/* centroid of P1-P2 */
	Pc.x = (P1.x + P2.x) / 2.0;
	Pc.y = (P1.y + P2.y) / 2.0;

	nloops = (geom->type == GEOM_POLYGONTYPE ? 1 : geom->nitems);
	for (int k=0; k < nloops; k++, nrings = __nrings_next)
	{
		const pg_geometry_t *poly;
		pg_geometry_t ring;
		cl_char		p1_location = '?';
		cl_char		p2_location = '?';
		cl_char		pc_location = '?';

		if (geom->type == GEOM_POLYGONTYPE)
			poly = geom;
		else
		{
			pos1 = geometry_load_subitem(&__polyData, geom, pos1, k, kcxt);
			if (!pos1)
				return -1;
			poly = &__polyData;
		}
		if (poly->nitems == 0)
			continue;
		/* rewind to the point where recursive call is invoked */
		__nrings_next = nrings + poly->nitems;
		if (__nrings_next < nskips)
			continue;

		/* check for each ring/hole */
		for (int i=0; i < poly->nitems; i++, nrings++)
		{
			POINT2D		Q1, Q2;
			const char *pos;
			cl_uint		unitsz;
			cl_int		wn1 = 0;
			cl_int		wn2 = 0;
			cl_int		wnc = 0;
			cl_int		pq1, pq2;

			p1_location = p2_location = pc_location = '?';

			pos2 = geometry_load_subitem(&ring, poly, pos2, i, kcxt);
			if (!pos2)
				return -1;
			if (ring.nitems < 4)
			{
				STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
							  "polygon corruption: too small vertex");
				return -1;
			}
			if (nrings < nskips)
				continue;

			unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(ring.flags);

			/* ring/hole must be closed. */
			pos = __loadPoint2d(&Q1, ring.rawdata, unitsz);
			__loadPoint2dIndex(&Q2, ring.rawdata, unitsz, ring.nitems - 1);
			if (PT_NE(Q1,Q2))
			{
				STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
							  "polygon corruption: unclosed ring/hole");
				return -1;
			}

			pq1 = __geom_segment_side(&P1,&P2,&Q1);
			for (int j=1; j < ring.nitems; j++)
			{
				cl_int	qp1, qp2, qpc;

				pos = __loadPoint2d(&Q2, pos, unitsz);
				if (PT_EQ(Q1, Q2))
					continue;	/* ignore zero length edge */
				pq2 = __geom_segment_side(&P1,&P2,&Q2);

				/*
				 * Update the state of winding number algorithm to determine
				 * the location of P1/P2 whether they are inside or outside
				 * of the Q1-Q2 edge.
				 */
				qp1 = __geom_segment_side(&Q1, &Q2, &P1);
				if (qp1 < 0 && Q1.y <= P1.y && P1.y < Q2.y)
					wn1++;
				else if (qp1 > 0 && Q2.y <= P1.y && P1.y < Q1.y)
					wn1--;

				qp2 = __geom_segment_side(&Q1, &Q2, &P2);
				if (qp2 < 0 && Q1.y <= P2.y && P2.y < Q2.y)
					wn2++;
				else if (qp2 > 0 && Q2.y <= P2.y && P2.y < Q1.y)
					wn2--;

				qpc = __geom_segment_side(&Q1, &Q2, &Pc);
				if (qpc < 0 && Q1.y <= Pc.y && Pc.y < Q2.y)
					wnc++;
				else if (qpc > 0 && Q2.y <= Pc.y && Pc.y < Q1.y)
					wnc--;
				if (qpc == 0 && __geom_pt_within_seg(&Pc,&Q1,&Q2))
					pc_location = 'B';
#if 0
				printf("P1(%d,%d)-P2(%d,%d) Q1(%d,%d)-Q2(%d,%d) qp1=%d qp2=%d pq1=%d pq2=%d\n",
					   (int)P1.x, (int)P1.y, (int)P2.x, (int)P2.y,
					   (int)Q1.x, (int)Q1.y, (int)Q2.x, (int)Q2.y,
					   qp1, qp2, pq1, pq2);
#endif
				if (!qp1 && !qp2)
				{
					/* P1-P2 and Q1-Q2 are colinear */
					cl_int	p1_in_qq = __geom_pt_within_seg(&P1,&Q1,&Q2);
					cl_int	p2_in_qq = __geom_pt_within_seg(&P2,&Q1,&Q2);
					cl_int	q1_in_pp, q2_in_pp;

					if (p1_in_qq != PT_OUTSIDE &&
						p2_in_qq != PT_OUTSIDE)
					{
						/* P1-P2 is fully contained by Q1-Q2 */
						if (p1_is_head)
							retval |= (IM__BOUND_BOUND_0D |
									   IM__LINE_HEAD_CONTAINED);
						if (p2_is_tail )
							retval |= (IM__BOUND_BOUND_0D |
									   IM__LINE_TAIL_CONTAINED);
						if (PT_EQ(P1,P2))
						{
							if (!p1_is_head && !p2_is_tail)
								retval |= IM__INTER_BOUND_0D;
						}
						else
							retval |= IM__INTER_BOUND_1D;
						return retval;
					}

					q1_in_pp = __geom_pt_within_seg(&Q1,&P1,&P2);
					q2_in_pp = __geom_pt_within_seg(&Q2,&P1,&P2);
					if (p1_in_qq != PT_OUTSIDE &&
						p2_in_qq == PT_OUTSIDE)
					{
						/* P1 is contained by Q1-Q2, but P2 is not */
						if (p1_is_head)
							retval |= (IM__BOUND_BOUND_0D |
									   IM__LINE_HEAD_CONTAINED);
						else
							retval |= IM__INTER_BOUND_0D;

						if (q1_in_pp == PT_INSIDE)
						{
							/* case of Q2-P1-Q1-P2; Q1-P2 is out of bounds */
							assert(q2_in_pp != PT_INSIDE);
							status = __geom_relate_seg_polygon(kcxt,
															   Q1, false,
															   P2, p2_is_tail,
															   geom,
															   nrings,
															   last_polygons);
							if (status < 0)
								return -1;
							return (retval | status | IM__INTER_BOUND_1D);
						}
						else if (q2_in_pp == PT_INSIDE)
						{
							/* case of Q1-P1-Q2-P2; Q2-P2 is out of bounds */
							assert(q1_in_pp != PT_INSIDE);
							status = __geom_relate_seg_polygon(kcxt,
															   Q2, false,
															   P2, p2_is_tail,
															   geom,
															   nrings,
															   last_polygons);
							if (status < 0)
								return -1;
							return (retval | status | IM__INTER_BOUND_1D);
						}
						else
						{
							assert(q1_in_pp == PT_BOUNDARY ||
								   q2_in_pp == PT_BOUNDARY);
						}
					}
					else if (p1_in_qq == PT_OUTSIDE &&
							 p2_in_qq != PT_OUTSIDE)
					{
						/* P2 is contained by Q1-Q2, but P2 is not */
						if (p2_is_tail)
							retval |= (IM__BOUND_BOUND_0D |
									   IM__LINE_TAIL_CONTAINED);
						else
							retval |= IM__INTER_BOUND_0D;

						if (q1_in_pp == PT_INSIDE)
						{
							/* P1-Q1-P2-Q2; P1-Q1 is out of bounds */
							status = __geom_relate_seg_polygon(kcxt,
															   P1, p1_is_head,
															   Q1, false,
															   geom,
															   nrings,
															   last_polygons);
							if (status < 0)
								return -1;
							return (retval | status | IM__INTER_BOUND_1D);
						}
						else if (q2_in_pp == PT_INSIDE)
						{
							/* P1-Q2-P2-Q1; P1-Q2 is out of bounds */
							status = __geom_relate_seg_polygon(kcxt,
															   P1, p1_is_head,
															   Q2, false,
															   geom,
															   nrings,
															   last_polygons);
							if (status < 0)
								return -1;
							return (retval | status | IM__INTER_BOUND_1D);
						}
					}
					else if (__geom_pt_within_seg(&Q1,&P1,&Q2) != PT_OUTSIDE &&
							 __geom_pt_within_seg(&Q2,&Q1,&P2) != PT_OUTSIDE)
					{
						/* case of P1-Q1-Q2-P2 */
						if (PT_NE(P1,Q1))
						{
							status = __geom_relate_seg_polygon(kcxt,
															   P1, p1_is_head,
															   Q1, false,
															   geom,
															   nrings,
															   last_polygons);
							if (status < 0)
								return -1;
							retval |= status;
						}
						if (PT_NE(Q2,P2))
						{
							status = __geom_relate_seg_polygon(kcxt,
															   Q2, false,
															   P2, p2_is_tail,
															   geom,
															   nrings,
															   last_polygons);
							if (status < 0)
								return -1;
							retval |= status;
						}
						return (retval | IM__INTER_BOUND_1D);
					}
					else if (__geom_pt_within_seg(&Q2,&P1,&Q1) != PT_OUTSIDE &&
							 __geom_pt_within_seg(&Q1,&Q2,&P1) != PT_OUTSIDE)
					{
						/* case of P1-Q2-Q1-P2 */
						if (PT_NE(P1,Q2))
						{
							status = __geom_relate_seg_polygon(kcxt,
															   P1, p1_is_head,
															   Q2, false,
															   geom,
															   nrings,
															   last_polygons);
							if (status < 0)
								return -1;
							retval |= status;
						}
						if (PT_NE(Q1,P2))
						{
							status = __geom_relate_seg_polygon(kcxt,
															   Q1, false,
															   P2, p2_is_tail,
															   geom,
															   nrings,
															   last_polygons);
							if (status < 0)
								return -1;
							retval |= status;
						}
						return (retval | IM__INTER_BOUND_1D);
					}
				}
				else if (qp1 == 0 && ((pq1 >= 0 && pq2 <= 0) ||
									  (pq1 <= 0 && pq2 >= 0)))
				{
					/* P1 touched Q1-Q2 */
					if (p1_is_head)
						retval |= (IM__BOUND_BOUND_0D |
								   IM__LINE_HEAD_CONTAINED);
					else
						retval |= IM__INTER_BOUND_0D;
					p1_location = 'B';
				}
				else if (qp2 == 0 && ((pq1 >= 0 && pq2 <= 0) ||
									  (pq1 <= 0 && pq2 >= 0)))
				{
					/* P2 touched Q1-Q2 */
					if (p2_is_tail)
						retval |= (IM__BOUND_BOUND_0D |
								   IM__LINE_TAIL_CONTAINED);
					else
						retval |= IM__INTER_BOUND_0D;
					p2_location = 'B';
				}
				else if (((qp1 >= 0 && qp2 <= 0) || (qp1 <= 0 && qp2 >= 0)) &&
						 ((pq1 >= 0 && pq2 <= 0) || (pq1 <= 0 && pq2 >= 0)))
				{
					/*
					 * P1-P2 and Q1-Q2 crosses.
					 *
					 * The point where crosses is:
					 *   P1 + r * (P2-P1) = Q1 + s * (Q2 - Q1)
					 *   [0 < s,r < 1]
					 *
					 * frac = (P2.x-P1.x)(Q2.y-Q1.y)-(P2.y-P1.y)(Q2.x-Q1.x)
					 * r = ((Q2.y - Q1.y) * (Q1.x-P1.x) -
					 *      (Q1.x - Q1.x) * (Q1.y-P1.y)) / frac
					 * s = ((P2.y - P1.y) * (Q1.x-P1.x) -
					 *      (P2.x - P1.x) * (Q1.y-P1.y)) / frac
					 *
					 * C = P1 + r * (P2-P1)
					 */
					cl_double	r, frac;
					POINT2D		C;

					frac = ((P2.x - P1.x) * (Q2.y - Q1.y) -
							(P2.y - P1.y) * (Q2.x - Q1.x));
					assert(frac != 0.0);
					r = ((Q2.y - Q1.y) * (Q1.x - P1.x) -
						 (Q2.x - Q1.x) * (Q1.y - P1.y)) / frac;
					C.x = P1.x + r * (P2.x - P1.x);
					C.y = P1.y + r * (P2.y - P1.y);
#if 0
					printf("P1(%d,%d)-P2(%d,%d) x Q1(%d,%d)-Q2(%d,%d) crosses at C(%f,%f) %d %d\n",
						   (int)P1.x, (int)P1.y, (int)P2.x, (int)P2.y,
						   (int)Q1.x, (int)Q1.y, (int)Q2.x, (int)Q2.y,
						   C.x, C.y,
						   (int)(FP_NEQUALS(P1.x,C.x) || FP_NEQUALS(P1.y,C.y)),
						   (int)(FP_NEQUALS(P2.x,C.x) || FP_NEQUALS(P2.y,C.y)));
#endif
					if (PT_EQ(P1,C))
					{
						if (p1_is_head)
							retval |= (IM__BOUND_BOUND_0D |
									   IM__LINE_HEAD_CONTAINED);
						else
							retval |= IM__INTER_BOUND_0D;
						p1_location = 'B';
					}
					else if (PT_EQ(P2,C))
					{
						if (p2_is_tail)
							retval |= (IM__BOUND_BOUND_0D |
									   IM__LINE_TAIL_CONTAINED);
						else
							retval |= IM__INTER_BOUND_0D;
						p2_location = 'B';
					}
					else
					{
						/* try P1-C recursively */
						status = __geom_relate_seg_polygon(kcxt,
														   P1, p1_is_head,
														   C, false,
														   geom,
														   nrings,
														   last_polygons);
						if (status < 0)
							return -1;
						retval |= status;
						/* try C-P2 recursively */
						status = __geom_relate_seg_polygon(kcxt,
														   C, false,
														   P2, p2_is_tail,
														   geom,
														   nrings,
														   last_polygons);
						if (status < 0)
							return -1;
						retval |= status;
						return (retval | IM__INTER_BOUND_0D);
					}
				}
				/* move to the next edge */
				pq1 = pq2;
				Q1 = Q2;
			}
			/* location of P1,P2 and Pc */
			if (p1_location == '?')
				p1_location = (wn1 == 0 ? 'E' : 'I');
			if (p2_location == '?')
				p2_location = (wn2 == 0 ? 'E' : 'I');
			if (pc_location == '?')
				pc_location = (wnc == 0 ? 'E' : 'I');
#if 0
			printf("Poly(%d)/Ring(%d) P1(%d,%d)[%c]-P2(%d,%d)[%c] (Pc(%d,%d)[%c])\n",
				   k, i,
				   (int)P1.x, (int)P1.y, p1_location,
				   (int)P2.x, (int)P2.y, p2_location,
				   (int)Pc.x, (int)Pc.y, pc_location);
#endif
			if (i == 0)
			{
				/* case of ring-0 */
				if ((p1_location == 'I' && p2_location == 'I') ||
					(p1_location == 'I' && p2_location == 'B') ||
					(p1_location == 'B' && p2_location == 'I'))
				{
					/*
					 * P1-P2 goes through inside of the polygon,
					 * so don't need to check other polygons any more.
					 */
					last_polygons = true;
				}
				else if (p1_location == 'B' && p2_location == 'B')
				{
					if (pc_location == 'B')
						return retval;	/* P1-P2 exactly goes on boundary */
					if (pc_location == 'I')
						last_polygons = true;
					if (pc_location == 'E')
						break;
				}
				else if ((p1_location == 'B' && p2_location == 'E') ||
						 (p1_location == 'E' && p2_location == 'B') ||
						 (p1_location == 'E' && p2_location == 'E'))
				{
					/*
					 * P1-P2 goes outside of the polygon, so don't need
					 * to check holes of this polygon.
					 */
					break;
				}
				else
				{
					/*
					 * If P1-P2 would be I-E or E-I, it obviously goes
					 * across the boundary line; should not happen.
					 */
					printf("P1 [%c] (%.2f,%.2f) P2 [%c] (%.2f,%.2f)\n",
						   p1_location, P1.x, P1.y,
						   p2_location, P2.x, P2.y);
					return -1;
				}
			}
			else
			{
				if ((p1_location == 'I' && p2_location == 'I') ||
					(p1_location == 'I' && p2_location == 'B') ||
					(p1_location == 'B' && p2_location == 'I') ||
					(p1_location == 'B' && p2_location == 'B' &&
					 pc_location == 'I'))
				{
					/*
					 * P1-P2 goes throught inside of the hole.
					 */
					return (retval | IM__INTER_EXTER_1D);
				}
			}
		}

		/*
		 * 'last_polygons == true' means P1-P2 goes inside of the polygon
		 * and didn't touch any holes.
		 */
		if (last_polygons)
		{
			if (p1_is_head && p1_location != 'B')
				retval |= (IM__BOUND_INTER_0D | IM__LINE_HEAD_CONTAINED);
			if (p2_is_tail && p2_location != 'B')
				retval |= (IM__BOUND_INTER_0D | IM__LINE_TAIL_CONTAINED);
			return (retval | IM__INTER_INTER_1D);
		}
	}
	/*
	 * Once the control reached here, it means P1-P2 never goes inside
	 * of the polygons.
	 */
	return (retval | IM__INTER_EXTER_1D);
}

STATIC_FUNCTION(cl_int)
geom_relate_line_polygon(kern_context *kcxt,
						 const pg_geometry_t *geom1,
						 const pg_geometry_t *geom2)
{
	const char *gpos = NULL;
	const char *ppos = NULL;
	cl_int		nloops;
	cl_uint		unitsz;
	cl_int		retval = IM__EXTER_EXTER_2D;
	cl_int		temp;
	geom_bbox_2d bbox2;

	assert((geom1->type == GEOM_LINETYPE ||
			geom1->type == GEOM_MULTILINETYPE) &&
		   (geom2->type == GEOM_POLYGONTYPE ||
			geom2->type == GEOM_MULTIPOLYGONTYPE));
    /* special empty cases */
    if (geom1->nitems == 0)
    {
        if (geom2->nitems == 0)
            return IM__EXTER_EXTER_2D;
        return IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;
    }
    else if (geom2->nitems == 0)
		return IM__INTER_EXTER_1D | IM__BOUND_EXTER_0D | IM__EXTER_EXTER_2D;

	retval = IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;

	/* shortcut if both of geometry has bounding box */
	if (geom2->bbox)
	{
		memcpy(&bbox2, geom2->bbox, sizeof(geom_bbox_2d));
		if (geom1->bbox)
		{
			geom_bbox_2d	bbox1;

			memcpy(&bbox1, geom1->bbox, sizeof(geom_bbox_2d));

			if (bbox1.xmax < bbox2.xmin || bbox1.xmin > bbox2.xmax ||
				bbox1.ymax < bbox2.ymin || bbox2.ymin > bbox2.ymax)
				return (retval | IM__INTER_EXTER_1D | IM__BOUND_EXTER_0D);
		}
	}

	nloops = (geom1->type == GEOM_LINETYPE ? 1 : geom1->nitems);
	for (int k=0; k < nloops; k++)
	{
		const pg_geometry_t *line;
		cl_uint		nitems;
		cl_bool		has_boundary;
		cl_bool		p1_is_head = true;
		POINT2D		P1, P2;

		if (geom1->type == GEOM_LINETYPE)
			line = geom1;
		else
		{
			pg_geometry_t __lineData;

			gpos = geometry_load_subitem(&__lineData, geom1, gpos, k, kcxt);
			if (!gpos)
				return -1;
			line = &__lineData;
		}
		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(line->flags);
		if (line->nitems == 0)
			continue;	/* empty */
		/* decrement nitems if tail items are duplicated */
		__loadPoint2dIndex(&P2, line->rawdata, unitsz, line->nitems-1);
		for (nitems = line->nitems; nitems >= 2; nitems--)
		{
			__loadPoint2dIndex(&P1, line->rawdata, unitsz, nitems-2);
			if (PT_NE(P1, P2))
				break;
		}
		/* checks for each edge */
		ppos = __loadPoint2d(&P1, line->rawdata, unitsz);
		has_boundary = PT_NE(P1, P2);
		for (int i=2; i <= nitems; i++)
		{
			ppos = __loadPoint2d(&P2, ppos, unitsz);
			if (PT_EQ(P1, P2))
				continue;

			/* shortcut, if this edge is obviously disjoint */
			if (geom2->bbox && (Max(P1.x,P2.x) < bbox2.xmin ||
								Min(P1.x,P2.x) > bbox2.xmax ||
								Max(P1.y,P2.y) < bbox2.ymin ||
								Min(P1.y,P2.y) > bbox2.ymax))
			{
				retval |= (IM__INTER_EXTER_1D | IM__BOUND_EXTER_0D);
			}
			else
			{
				temp = __geom_relate_seg_polygon(kcxt,
												 P1, (has_boundary && p1_is_head),
												 P2, (has_boundary && i==nitems),
												 geom2, 0, false);
				if (temp < 0)
					return -1;
				retval |= temp;
			}
			P1=P2;
			p1_is_head = false;
		}

		if (has_boundary)
		{
			temp = (IM__LINE_HEAD_CONTAINED | IM__LINE_TAIL_CONTAINED);
			if ((retval & temp) != temp)
				retval |= IM__BOUND_EXTER_0D;
		}
	}
	return (retval & IM__MASK_FULL);
}

STATIC_FUNCTION(cl_int)
__geom_relate_ring_polygon(kern_context *kcxt,
						   const pg_geometry_t *ring,
						   const pg_geometry_t *geom)
{
	const char *gpos;
	const char *ppos;
	cl_uint		unitsz;
	cl_uint		nitems;
	cl_int		nloops;
	cl_bool		poly_has_inside = false;
	cl_bool		poly_has_outside = false;
	cl_int		rflags = 0;
	cl_int		boundary = 0;
	cl_int		status;
	POINT2D		P1, P2;
	geom_bbox_2d bbox;

	if (ring->nitems == 0)
		return 0;		/* empty */
	unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(geom->flags);
	if (geom->bbox)
		memcpy(&bbox, geom->bbox, sizeof(geom_bbox_2d));

	/* decrement nitems if tail items are duplicated */
	__loadPoint2dIndex(&P1, ring->rawdata, unitsz, ring->nitems-1);
	for (nitems = ring->nitems; nitems >= 2; nitems--)
	{
		__loadPoint2dIndex(&P2, ring->rawdata, unitsz, nitems-2);
		if (PT_NE(P1,P2))
			break;
	}
	/* checks for each edge */
	ppos = __loadPoint2d(&P1, ring->rawdata, unitsz);
	for (int i=2; i <= nitems; i++)
	{
		ppos = __loadPoint2d(&P2, ppos, unitsz);
		if (PT_EQ(P1,P2))
			continue;
		if (geom->bbox && (Max(P1.x,P2.x) < bbox.xmin ||
						   Min(P1.x,P2.x) > bbox.xmax ||
						   Max(P1.y,P2.y) < bbox.ymin ||
						   Min(P1.y,P2.y) > bbox.ymax))
		{
			status = (IM__INTER_EXTER_1D |
					  IM__BOUND_EXTER_0D |
					  IM__EXTER_INTER_2D |
					  IM__EXTER_BOUND_1D |
					  IM__EXTER_EXTER_2D);
        }
        else
        {
            status = __geom_relate_seg_polygon(kcxt,
                                               P1, false,
                                               P2, false,
                                               geom, 0, false);
			if (status < 0)
				return -1;
		}
		rflags |= status;
		P1=P2;
	}
	/*
	 * Simple check whether polygon is fully contained by the ring
	 */
	nloops = (geom->type == GEOM_POLYGONTYPE ? 1 : geom->nitems);
	for (int k=0; k < nloops; k++)
	{
		pg_geometry_t	poly0;
		cl_int			location;

		if (geom->type == GEOM_POLYGONTYPE)
		{
			if (!geometry_load_subitem(&poly0, geom, NULL, 0, kcxt))
				return -1;
		}
		else
		{
			pg_geometry_t	__temp;

			gpos = geometry_load_subitem(&__temp, geom, gpos, k, kcxt);
			if (!gpos)
				return -1;
			if (!geometry_load_subitem(&poly0, &__temp, NULL, 0, kcxt))
				return -1;
		}
		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(poly0.flags);
		ppos = poly0.rawdata;
		for (int i=0; i < poly0.nitems; i++)
		{
			ppos = __loadPoint2d(&P1, ppos, unitsz);
			location = __geom_point_in_ring(ring, &P1, kcxt);
			if (location == PT_INSIDE)
				poly_has_inside = true;
			else if (location == PT_OUTSIDE)
				poly_has_outside = true;
			else if (location != PT_BOUNDARY)
				return -1;
		}
		if (poly_has_inside && poly_has_outside)
			break;
	}

	/*
	 * transform rflags to ring-polygon relationship
	 */
	if ((rflags & IM__INTER_BOUND_2D) == IM__INTER_BOUND_1D)
		boundary = IM__BOUND_BOUND_1D;
	else if ((rflags & IM__INTER_BOUND_2D) == IM__INTER_BOUND_0D)
		boundary = IM__BOUND_BOUND_0D;

	if ((rflags & IM__INTER_INTER_2D) == 0 &&
		(rflags & IM__INTER_BOUND_2D) != 0 &&
		(rflags & IM__INTER_EXTER_2D) == 0)
	{
		/* ring equals to the polygon */
		return (IM__INTER_INTER_2D |
				IM__BOUND_BOUND_1D |
				IM__EXTER_EXTER_2D);
	}
	else if ((rflags & IM__INTER_INTER_2D) == 0 &&
			 (rflags & IM__INTER_BOUND_2D) == 0 &&
			 (rflags & IM__INTER_EXTER_2D) != 0)
	{
		if (poly_has_outside)
		{
			/* disjoint */
			return (IM__INTER_EXTER_2D |
					IM__BOUND_EXTER_1D |
					IM__EXTER_INTER_2D |
					IM__EXTER_BOUND_1D |
					IM__EXTER_EXTER_2D);
		}
		else
		{
			/* ring fully contains the polygons */
			return (IM__INTER_INTER_2D |
					IM__INTER_BOUND_1D |
					IM__INTER_EXTER_2D |
					IM__BOUND_EXTER_1D |
					IM__EXTER_EXTER_2D);
		}
	}
	else if ((rflags & IM__INTER_INTER_2D) != 0 &&
			 (rflags & IM__INTER_BOUND_2D) != 0 &&
			 (rflags & IM__INTER_EXTER_2D) != 0)
	{
		/* ring has intersection to the polygon */
		assert(boundary != 0);
		return boundary | (IM__INTER_INTER_2D |
						   IM__INTER_BOUND_1D |
						   IM__INTER_EXTER_2D |
						   IM__BOUND_INTER_1D |
						   IM__BOUND_EXTER_1D |
						   IM__EXTER_INTER_2D |
						   IM__EXTER_BOUND_1D |
						   IM__EXTER_EXTER_2D);
	}
	else if ((rflags & IM__INTER_INTER_2D) == 0 &&
			 (rflags & IM__INTER_BOUND_2D) != 0 &&
			 (rflags & IM__INTER_EXTER_2D) != 0)
	{
		if (poly_has_outside)
		{
			/* ring touched the polygon at a boundary, but no intersection */
			assert(boundary != 0);
			return boundary | (IM__INTER_EXTER_2D |
							   IM__BOUND_EXTER_1D |
							   IM__EXTER_INTER_2D |
							   IM__EXTER_BOUND_1D |
							   IM__EXTER_EXTER_2D);
		}
		else
		{
			/* ring fully contains the polygon touched at boundaries */
			assert(boundary != 0);
			return boundary | (IM__INTER_INTER_2D |
							   IM__INTER_BOUND_1D |
							   IM__INTER_EXTER_2D |
							   IM__BOUND_EXTER_1D |
							   IM__EXTER_EXTER_2D);
		}
	}
	else if ((rflags & IM__INTER_INTER_2D) != 0 &&
             (rflags & IM__INTER_EXTER_2D) == 0)
	{
		/* ring is fully contained by the polygon; might be touched */
		return boundary | (IM__INTER_INTER_2D |
						   IM__BOUND_INTER_1D |
						   IM__EXTER_INTER_2D |
						   IM__EXTER_BOUND_1D |
						   IM__EXTER_EXTER_2D);
	}
	return -1;		/* unknown intersection */
}

STATIC_FUNCTION(cl_int)
geom_relate_triangle_triangle(kern_context *kcxt,
							  const pg_geometry_t *geom1,
							  const pg_geometry_t *geom2)
{
	pg_geometry_t poly;
	cl_double	polyData[9];
	cl_uint		unitsz = sizeof(double) * GEOM_FLAGS_NDIMS(geom2->flags);
	const char *pos;
	POINT2D		P;

	assert(geom1->type == GEOM_TRIANGLETYPE &&
		   geom2->type == GEOM_TRIANGLETYPE);
	/* special empty cases */
	if (geom1->nitems == 0)
	{
		if (geom2->nitems == 0)
			return IM__EXTER_EXTER_2D;
		return IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;
	}
	else if (geom2->nitems == 0)
		return IM__INTER_EXTER_2D | IM__BOUND_EXTER_1D | IM__EXTER_EXTER_2D;

	/* sanity checks */
	if (geom1->nitems != 4 || geom2->nitems != 4)
	{
		STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
					  "triangle must have exactly 4 points");
		return -1;
	}

	/* shortcut if both of geometry has bounding box */
	if (geom1->bbox && geom2->bbox)
	{
		geom_bbox_2d	bbox1;
		geom_bbox_2d	bbox2;

		memcpy(&bbox1, geom1->bbox, sizeof(geom_bbox_2d));
		memcpy(&bbox2, geom2->bbox, sizeof(geom_bbox_2d));
		if (bbox1.xmax < bbox2.xmin || bbox1.xmin > bbox2.xmax ||
			bbox1.ymax < bbox2.ymin || bbox2.ymin > bbox2.ymax)
			return (IM__INTER_EXTER_2D |
					IM__BOUND_EXTER_1D |
					IM__EXTER_INTER_2D |
					IM__EXTER_BOUND_1D |
					IM__EXTER_EXTER_2D);
	}

	/* setup pseudo polygon for code reuse */
	pos = geom2->rawdata;

	memset(polyData, 0, sizeof(polyData));
	((cl_uint *)polyData)[0] = 4;
	for (int i=0; i < 4; i++)
	{
		pos = __loadPoint2d(&P, pos, unitsz);
		if (!pos)
			return -1;
		polyData[2*i+1] = P.x;
		polyData[2*i+2] = P.y;
	}

	poly.isnull = false;
	poly.type = GEOM_POLYGONTYPE;
	poly.flags = GEOM_FLAG__READONLY;
	poly.srid = geom2->srid;
	poly.nitems = 1;	/* only ring-0 */
	poly.rawsize = sizeof(polyData);
	poly.rawdata = (const char *)polyData;
	poly.bbox = NULL;
	
	return __geom_relate_ring_polygon(kcxt, geom1, &poly);
}

STATIC_FUNCTION(cl_int)
geom_relate_triangle_polygon(kern_context *kcxt,
							 const pg_geometry_t *geom1,
							 const pg_geometry_t *geom2)
{
	assert((geom1->type == GEOM_TRIANGLETYPE) &&
		   (geom2->type == GEOM_POLYGONTYPE ||
			geom2->type == GEOM_MULTIPOLYGONTYPE));
	/* special empty cases */
	if (geom1->nitems == 0)
	{
		if (geom2->nitems == 0)
			return IM__EXTER_EXTER_2D;
		return IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;
	}
	else if (geom2->nitems == 0)
		return IM__INTER_EXTER_2D | IM__BOUND_EXTER_1D | IM__EXTER_EXTER_2D;

	/* sanity checks */
	if (geom1->nitems != 4)
	{
		STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
					  "triangle must have exactly 4 points");
		return -1;
	}

	/* shortcut if both of geometry has bounding box */
	if (geom1->bbox && geom2->bbox)
	{
		geom_bbox_2d	bbox1;
		geom_bbox_2d	bbox2;

		memcpy(&bbox1, geom1->bbox, sizeof(geom_bbox_2d));
		memcpy(&bbox2, geom2->bbox, sizeof(geom_bbox_2d));
		if (bbox1.xmax < bbox2.xmin || bbox1.xmin > bbox2.xmax ||
			bbox1.ymax < bbox2.ymin || bbox2.ymin > bbox2.ymax)
			return (IM__INTER_EXTER_2D |
					IM__BOUND_EXTER_1D |
					IM__EXTER_INTER_2D |
					IM__EXTER_BOUND_1D |
					IM__EXTER_EXTER_2D);
	}
	return __geom_relate_ring_polygon(kcxt, geom1, geom2);
}

STATIC_FUNCTION(cl_int)
geom_relate_polygon_polygon(kern_context *kcxt,
							const pg_geometry_t *geom1,
							const pg_geometry_t *geom2)
{
	pg_geometry_t __polyData;
	const pg_geometry_t *poly;
	const char *gpos = NULL;
	const char *rpos = NULL;
	cl_int		nloops;
	cl_int		retval = IM__EXTER_EXTER_2D;

	assert((geom1->type == GEOM_POLYGONTYPE ||
			geom1->type == GEOM_MULTIPOLYGONTYPE) &&
		   (geom2->type == GEOM_POLYGONTYPE ||
			geom2->type == GEOM_MULTIPOLYGONTYPE));
	/* special empty cases */
	if (geom1->nitems == 0)
	{
		if (geom2->nitems == 0)
			return IM__EXTER_EXTER_2D;
		return IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;
	}
	else if (geom2->nitems == 0)
		return IM__INTER_EXTER_2D | IM__BOUND_EXTER_1D | IM__EXTER_EXTER_2D;

	/* shortcut if both of geometry has bounding box */
	if (geom1->bbox && geom2->bbox)
	{
		geom_bbox_2d bbox1;
		geom_bbox_2d bbox2;

		memcpy(&bbox1, geom1->bbox, sizeof(geom_bbox_2d));
		memcpy(&bbox2, geom2->bbox, sizeof(geom_bbox_2d));
		if (bbox1.xmax < bbox2.xmin || bbox1.xmin > bbox2.xmax ||
			bbox1.ymax < bbox2.ymin || bbox2.ymin > bbox2.ymax)
			return (IM__INTER_EXTER_2D |
					IM__BOUND_EXTER_1D |
					IM__EXTER_INTER_2D |
					IM__EXTER_BOUND_1D |
					IM__EXTER_EXTER_2D);
	}

	nloops = (geom1->type == GEOM_POLYGONTYPE ? 1 : geom1->nitems);
	for (int k=0; k < nloops; k++)
	{
		cl_int		__retval = 0;	/* pending result for each polygon */

		if (geom1->type == GEOM_POLYGONTYPE)
			poly = geom1;
		else
		{
			gpos = geometry_load_subitem(&__polyData, geom1, gpos, k, kcxt);
			if (!gpos)
				return -1;
			poly = &__polyData;
		}

		for (int i=0; i < poly->nitems; i++)
		{
			pg_geometry_t ring;
			cl_int		status;

			rpos = geometry_load_subitem(&ring, poly, rpos, i, kcxt);
			if (!rpos)
				return -1;

			status = __geom_relate_ring_polygon(kcxt, &ring, geom2);
			if (status < 0)
				return -1;
			if (i == 0)
			{
				__retval = status;
				if ((__retval & IM__INTER_INTER_2D) == 0)
					break;	/* disjoint, so we can skip holes */
			}
			else
			{
				/* add boundaries, if touched/crossed */
				__retval |= (status & IM__BOUND_BOUND_2D);

				/* geom2 is disjoint from the hole? */
				if ((status & IM__INTER_INTER_2D) == 0)
					continue;
				/*
				 * geom2 is fully contained by the hole, so reconstruct
				 * the DE9-IM as disjointed polygon.
				 */
				if ((status & IM__INTER_EXTER_2D) != 0 &&
					(status & IM__EXTER_INTER_2D) == 0)
				{
					__retval = ((status & IM__BOUND_BOUND_2D) |
								IM__INTER_EXTER_2D |
								IM__BOUND_EXTER_1D |
								IM__EXTER_INTER_2D |
								IM__EXTER_BOUND_1D |
								IM__EXTER_EXTER_2D);
					break;
				}

				/*
				 * geom2 has a valid intersection with the hole, add it.
				 */
				if ((status & IM__INTER_INTER_2D) != 0)
				{
					__retval |= (IM__BOUND_INTER_1D |
								 IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D);
					break;
				}
			}
		}
		retval |= __retval;
	}
	return retval;
}

STATIC_FUNCTION(cl_int)
geom_relate_internal(kern_context *kcxt,
					 const pg_geometry_t *geom1,
					 const pg_geometry_t *geom2)
{
	assert(!geom1->isnull && !geom2->isnull);
	if (geom1->type == GEOM_POINTTYPE ||
		geom1->type == GEOM_MULTIPOINTTYPE)
	{
		if (geom2->type == GEOM_POINTTYPE ||
			geom2->type == GEOM_MULTIPOINTTYPE)
			return geom_relate_point_point(kcxt, geom1, geom2);
		else if (geom2->type == GEOM_LINETYPE ||
				 geom2->type == GEOM_MULTILINETYPE)
			return geom_relate_point_line(kcxt, geom1, geom2);
		else if (geom2->type == GEOM_TRIANGLETYPE)
			return geom_relate_point_triangle(kcxt, geom1, geom2);
		else if (geom2->type == GEOM_POLYGONTYPE ||
				 geom2->type == GEOM_MULTIPOLYGONTYPE)
			return geom_relate_point_poly(kcxt, geom1, geom2);
	}
	else if (geom1->type == GEOM_LINETYPE ||
			 geom1->type == GEOM_MULTILINETYPE)
	{
		if (geom2->type == GEOM_POINTTYPE ||
			geom2->type == GEOM_MULTIPOINTTYPE)
			return IM__TWIST(geom_relate_point_line(kcxt, geom2, geom1));
		else if (geom2->type == GEOM_LINETYPE ||
				 geom2->type == GEOM_MULTILINETYPE)
			return geom_relate_line_line(kcxt, geom1, geom2);
		else if (geom2->type == GEOM_TRIANGLETYPE)
			return geom_relate_line_triangle(kcxt, geom1, geom2);
		else if (geom2->type == GEOM_POLYGONTYPE ||
				 geom2->type == GEOM_MULTIPOLYGONTYPE)
			return geom_relate_line_polygon(kcxt, geom1, geom2);
	}
	else if (geom1->type == GEOM_TRIANGLETYPE)
	{
		if (geom2->type == GEOM_POINTTYPE ||
			geom2->type == GEOM_MULTIPOINTTYPE)
			return IM__TWIST(geom_relate_point_triangle(kcxt,geom2,geom1));
		else if (geom2->type == GEOM_LINETYPE ||
				 geom2->type == GEOM_MULTILINETYPE)
			return IM__TWIST(geom_relate_line_triangle(kcxt,geom2,geom1));
		else if (geom2->type == GEOM_TRIANGLETYPE)
			return geom_relate_triangle_triangle(kcxt,geom1,geom2);
		else if (geom2->type == GEOM_POLYGONTYPE ||
				 geom2->type == GEOM_MULTIPOLYGONTYPE)
			return geom_relate_triangle_polygon(kcxt,geom1,geom2);
	}
	else if (geom1->type == GEOM_POLYGONTYPE ||
			 geom1->type == GEOM_MULTIPOLYGONTYPE)
	{
		if (geom2->type == GEOM_POINTTYPE ||
			geom2->type == GEOM_MULTIPOINTTYPE)
			return IM__TWIST(geom_relate_point_poly(kcxt,geom2,geom1));
		else if (geom2->type == GEOM_LINETYPE ||
				 geom2->type == GEOM_MULTILINETYPE)
			return IM__TWIST(geom_relate_line_polygon(kcxt,geom2,geom1));
		else if (geom2->type == GEOM_TRIANGLETYPE)
			return IM__TWIST(geom_relate_triangle_polygon(kcxt,geom2,geom1));
		else if (geom2->type == GEOM_POLYGONTYPE ||
				 geom2->type == GEOM_MULTIPOLYGONTYPE)
			return geom_relate_polygon_polygon(kcxt,geom1,geom2);
	}
	STROM_CPU_FALLBACK(kcxt, ERRCODE_FEATURE_NOT_SUPPORTED,
					   "unsupported geometry type");
	return -1;	/* error */
}

/* ================================================================
 *
 * St_Relate(geometry,geometry)
 *
 * ================================================================
 */
DEVICE_FUNCTION(pg_text_t)
pgfn_st_relate(kern_context *kcxt,
			   const pg_geometry_t &geom1,
			   const pg_geometry_t &geom2)
{
	pg_text_t	result;
	cl_int		status;
	char	   *pos;

	memset(&result, 0, sizeof(pg_bool_t));
	result.isnull = geom1.isnull | geom2.isnull;
	if (result.isnull)
		return result;

	status = geom_relate_internal(kcxt, &geom1, &geom2);
	if (status < 0)
	{
		result.isnull = true;
		return result;
	}

	pos = (char *)kern_context_alloc(kcxt, VARHDRSZ + 9);
	if (!pos)
	{
		STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,
						   "out of memory");
		result.isnull = true;
		return result;
	}
	result.value = pos;
	result.length = -1;
	SET_VARSIZE(pos, VARHDRSZ + 9);

	pos += VARHDRSZ;
	for (int i=0; i < 9; i++, status >>= 3)
	{
		int		mask = (status & 0x7);

		*pos++ = ((mask & 0x04) != 0 ? '2' :
				  (mask & 0x02) != 0 ? '1' :
				  (mask & 0x01) != 0 ? '0' : 'F');
	}
	return result;
}

/* ================================================================
 *
 * St_Contains(geometry,geometry)
 *
 * ================================================================
 */
STATIC_FUNCTION(pg_bool_t)
fast_geom_contains_polygon_point(kern_context *kcxt,
								 const pg_geometry_t *geom1,
								 const pg_geometry_t *geom2)
{
	POINT2D		pt;
	cl_int		status = PT_ERROR;
	pg_bool_t	result;
	geom_bbox_2d bbox;

	assert((geom1->type == GEOM_POLYGONTYPE ||
			geom1->type == GEOM_MULTIPOLYGONTYPE) &&
		   (geom2->type == GEOM_POINTTYPE ||
			geom2->type == GEOM_MULTIPOINTTYPE));
	memset(&result, 0, sizeof(pg_bool_t));
	if (geom1->bbox)
		memcpy(&bbox, &geom1->bbox->d2, sizeof(geom_bbox_2d));

	if (geom2->type == GEOM_POINTTYPE)
	{
		__loadPoint2d(&pt, geom2->rawdata, 0);
		if (geom1->bbox)
		{
			if (pt.x < bbox.xmin || pt.x > bbox.xmax ||
				pt.y < bbox.ymin || pt.y > bbox.ymax)
				return result;
		}
		if (geom1->type == GEOM_POLYGONTYPE)
			status = __geom_point_in_polygon(geom1, &pt, kcxt);
		else
			status = __geom_point_in_multipolygon(geom1, &pt, kcxt);
		if (status == PT_ERROR)
			goto error;
		result.value = (status == PT_INSIDE);
	}
	else
	{
		pg_geometry_t __geom;
		const char *pos = NULL;
		cl_bool		meet_inside = false;

		for (int i=0; i < geom2->nitems; i++)
		{
			pos = geometry_load_subitem(&__geom, geom2, pos, i);
			if (!pos)
				goto error;
			__loadPoint2d(&pt, __geom.rawdata, 0);
			if (geom1->bbox && (pt.x < bbox.xmin || pt.x > bbox.xmax ||
								pt.y < bbox.ymin || pt.y > bbox.ymax))
			{
				status = PT_OUTSIDE;
				break;
			}
			if (geom1->type == GEOM_POLYGONTYPE)
				status = __geom_point_in_polygon(geom1, &pt, kcxt);
			else
				status = __geom_point_in_multipolygon(geom1, &pt, kcxt);
//			printf("PT(%d,%d) %d\n", (int)pt.x, (int)pt.y, status);
			if (status == PT_ERROR)
				goto error;
			else if (status == PT_INSIDE)
				meet_inside = true;
			else if (status == PT_OUTSIDE)
				break;
		}
//		printf("meet_inside = %d status = %d\n", (int)meet_inside, status);
		result.value = (meet_inside && status != PT_OUTSIDE);
	}
	return result;
error:
	result.isnull = true;
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_st_contains(kern_context *kcxt,
                 const pg_geometry_t &geom1,
                 const pg_geometry_t &geom2)
{
	pg_bool_t	result;
	cl_int		status;

	result.isnull = geom1.isnull | geom2.isnull;
	result.value  = false;
	if (result.isnull)
		return result;

	if (geometry_is_empty(&geom1) || geometry_is_empty(&geom2))
		return result;

	/*
	 * shortcut-1: if geom1 is a polygon and geom2 is a point type,
	 * call the fast point-in-polygon function.
	 */
	if ((geom1.type == GEOM_POLYGONTYPE ||
		 geom1.type == GEOM_MULTIPOLYGONTYPE) &&
		(geom2.type == GEOM_POINTTYPE ||
		 geom2.type == GEOM_MULTIPOINTTYPE))
		return fast_geom_contains_polygon_point(kcxt, &geom1, &geom2);

	/*
	 * shortcut-2: if geom2 bounding box is not completely inside
	 * geom1 bounding box we can return FALSE.
	 */
	if (geom1.bbox && geom2.bbox)
	{
		geom_bbox_2d	bbox1, bbox2;

		memcpy(&bbox1, &geom1.bbox->d2, sizeof(geom_bbox_2d));
		memcpy(&bbox2, &geom2.bbox->d2, sizeof(geom_bbox_2d));

		if (bbox2.xmin < bbox1.xmin || bbox1.xmax < bbox2.xmax ||
			bbox2.ymin < bbox1.ymin || bbox1.ymax < bbox2.ymax)
			return result;
	}

	/* elsewhere, use st_relate and DE9-IM */
	status = geom_relate_internal(kcxt, &geom1, &geom2);
	if (status < 0)
		result.isnull = true;
	else
	{
		result.value = ((status & IM__INTER_INTER_2D) != 0 &&
						(status & IM__EXTER_INTER_2D) == 0 &&
						(status & IM__EXTER_BOUND_2D) == 0);
	}
	return result;
}

/* ================================================================
 *
 * St_Crosses(geometry,geometry)
 *
 * ================================================================
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_st_crosses(kern_context *kcxt,
				const pg_geometry_t &geom1,
				const pg_geometry_t &geom2)
{
	pg_bool_t	result;
	cl_int		status;

	memset(&result, 0, sizeof(pg_bool_t));
	result.isnull = geom1.isnull | geom2.isnull;
	if (result.isnull)
		return result;

	if (geometry_is_empty(&geom1) || geometry_is_empty(&geom2))
		return result;

	/*
	 * shortcut: if bounding boxes are disjoint obviously,
	 * we can return FALSE immediately.
	 */
	if (geom1.bbox && geom2.bbox)
	{
		geom_bbox_2d	bbox1, bbox2;

		memcpy(&bbox1, &geom1.bbox->d2, sizeof(geom_bbox_2d));
		memcpy(&bbox2, &geom2.bbox->d2, sizeof(geom_bbox_2d));

		if (bbox1.xmax < bbox2.xmin ||
			bbox1.xmin > bbox2.xmax ||
			bbox1.ymax < bbox2.ymin ||
			bbox1.ymin > bbox2.ymax)
			return result;
	}

#if 0
	/*
	 * TODO: fast path for line-polygon situation
	 */
	if ((geom1.type == GEOM_LINETYPE ||
		 geom1.type == GEOM_MULTILINETYPE) &&
		(geom2.type == GEOM_POLYGONTYPE ||
		 geom2.type == GEOM_MULTIPOLYGONTYPE))
		return fast_geom_crosses_line_polygon(kcxt, &geom1, &geom2);
	if ((geom1.type == GEOM_POLYGONTYPE ||
		 geom1.type == GEOM_MULTIPOLYGONTYPE) &&
		(geom2.type == GEOM_LINETYPE ||
		 geom2.type == GEOM_MULTILINETYPE))
		return fast_geom_crosses_line_polygon(kcxt, &geom2, &geom2);
#endif
	status = geom_relate_internal(kcxt, &geom1, &geom2);
	if (status < 0)
		result.isnull = true;
	else if (geom1.type == GEOM_POINTTYPE ||
			 geom1.type == GEOM_MULTIPOINTTYPE)
	{
		if (geom2.type == GEOM_LINETYPE ||
			geom2.type == GEOM_MULTILINETYPE ||
			geom2.type == GEOM_TRIANGLETYPE ||
            geom2.type == GEOM_POLYGONTYPE ||
            geom2.type == GEOM_MULTIPOLYGONTYPE)
		{
			result.value = ((status & IM__INTER_INTER_2D) != 0 &&
							(status & IM__INTER_EXTER_2D) != 0);
		}
	}
	else if (geom1.type == GEOM_LINETYPE ||
			 geom1.type == GEOM_MULTILINETYPE)
	{
		if (geom2.type == GEOM_POINTTYPE ||
			geom2.type == GEOM_MULTIPOINTTYPE)
		{
			result.value = ((status & IM__INTER_INTER_2D) != 0 &&
							(status & IM__EXTER_INTER_2D) != 0);
		}
		else if (geom2.type == GEOM_LINETYPE ||
				 geom2.type == GEOM_MULTILINETYPE)
		{
			result.value = ((status & IM__INTER_INTER_2D)==IM__INTER_INTER_0D);
		}
		else if (geom2.type == GEOM_TRIANGLETYPE ||
				 geom2.type == GEOM_POLYGONTYPE ||
				 geom2.type == GEOM_MULTIPOLYGONTYPE)
		{
			result.value = ((status & IM__INTER_INTER_2D) != 0 &&
							(status & IM__INTER_EXTER_2D) != 0);
		}
	}
	else if (geom1.type == GEOM_TRIANGLETYPE ||
			 geom1.type == GEOM_POLYGONTYPE ||
			 geom1.type == GEOM_MULTIPOLYGONTYPE)
	{
		if (geom2.type == GEOM_POINTTYPE ||
			geom2.type == GEOM_MULTIPOINTTYPE ||
			geom2.type == GEOM_LINETYPE ||
			geom2.type == GEOM_MULTILINETYPE)
		{
			result.value = ((status & IM__INTER_INTER_2D) != 0 &&
							(status & IM__EXTER_INTER_2D) != 0);
		}
	}
	return result;
}
