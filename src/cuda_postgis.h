/*
 * cuda_postgis.h
 *
 * Collection of PostGIS functions & operators for CUDA GPU devices.
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"

#ifndef CUDA_POSTGIS_H
#define CUDA_POSTGIS_H

/*
 * GSERIALIZED; on-disk data layout of geometry values
 */
#define G1FLAG_Z			0x01
#define G1FLAG_M			0x02
#define G1FLAG_BBOX			0x04
#define G1FLAG_GEODETIC		0x08
#define G1FLAG_READONLY		0x10
#define G1FLAG_SOLID		0x20

#define G2FLAG_Z			0x01
#define G2FLAG_M			0x02
#define G2FLAG_BBOX			0x04
#define G2FLAG_GEODETIC		0x08
#define G2FLAG_EXTENDED		0x10
#define G2FLAG_RESERVED1	0x20 /* RESERVED FOR FUTURE USES */
#define G2FLAG_VER_0		0x40
#define G2FLAG_RESERVED2	0x80 /* RESERVED FOR FUTURE VERSIONS */

#define G2FLAG_X_SOLID			0x00000001
#define G2FLAG_X_CHECKED_VALID	0x00000002	/* To Be Implemented? */
#define G2FLAG_X_IS_VALID		0x00000004	/* To Be Implemented? */
#define G2FLAG_X_HAS_HASH		0x00000008	/* To Be Implemented? */

typedef struct
{
	cl_uchar	srid[3];	/* 24bits SRID */
	cl_uchar	gflags;		/* GxFLAG_* above */
	char		data[FLEXIBLE_ARRAY_MEMBER];
} __GSERIALIZED;

typedef struct
{
	cl_int		vl_len_;	/* varlena header */
	__GSERIALIZED body;
} GSERIALIZED;

/* see LWTYPE definitions; at liblwgeom.h */
#define GEOM_POINTTYPE				1
#define GEOM_LINETYPE				2
#define GEOM_POLYGONTYPE			3
#define GEOM_MULTIPOINTTYPE			4
#define GEOM_MULTILINETYPE			5
#define GEOM_MULTIPOLYGONTYPE		6
#define GEOM_COLLECTIONTYPE			7
#define GEOM_CIRCSTRINGTYPE			8
#define GEOM_COMPOUNDTYPE			9
#define GEOM_CURVEPOLYTYPE			10
#define GEOM_MULTICURVETYPE			11
#define GEOM_MULTISURFACETYPE		12
#define GEOM_POLYHEDRALSURFACETYPE	13
#define GEOM_TRIANGLETYPE			14
#define GEOM_TINTYPE				15
#define GEOM_NUMTYPES				16
#define GEOM_TYPE_IS_VALID(gs_type)	((gs_type) >= 1 && (gs_type) <= GEOM_NUMTYPES)

/* see LWFLAG_* in CPU code; at liblwgeom.h */
#define GEOM_FLAG__Z			0x01
#define GEOM_FLAG__M			0x02
#define GEOM_FLAG__ZM			0x03	/* == GEOM_FLAG__Z | GEOM_FLAG__M */
#define GEOM_FLAG__BBOX			0x04
#define GEOM_FLAG__GEODETIC		0x08
#define GEOM_FLAG__READONLY		0x10
#define GEOM_FLAG__SOLID		0x20
#define GEOM_FLAGS_NDIMS(flags)		\
	(2 + ((flags) & GEOM_FLAG__Z) + (((flags) & GEOM_FLAG__M) >> 1))

/*
 * boundary box
 */
typedef struct
{
	cl_float	xmin, xmax;
	cl_float	ymin, ymax;
} geom_bbox_2d;

typedef struct
{
	cl_float	xmin, xmax;
	cl_float	ymin, ymax;
	cl_float	zmin, zmax;
} geom_bbox_3d;

typedef struct
{
	cl_float	xmin, xmax;
	cl_float	ymin, ymax;
	cl_float	mmin, mmax;
} geom_bbox_3dm;

typedef struct
{
	cl_float	xmin, xmax;
	cl_float	ymin, ymax;
	cl_float	zmin, zmax;
	cl_float	mmin, mmax;
} geom_bbox_4d;

typedef union
{
	geom_bbox_2d	d2;
	geom_bbox_3d	d3;
	geom_bbox_3dm	d3m;
	geom_bbox_4d	d4;
} geom_bbox;

STATIC_INLINE(size_t)
geometry_bbox_size(cl_uint geom_flags)
{
	if ((geom_flags & GEOM_FLAG__GEODETIC) != 0)
		return sizeof(float) * 6;

	return sizeof(float) * 2 * GEOM_FLAGS_NDIMS(geom_flags);
}

/* some SRID definitions */
#define SRID_UNKNOWN		0
#define SRID_MAXIMUM		999999
#define SRID_USER_MAXIMUM	998999

#ifndef PG_GEOMETRY_TYPE_DEFINED
#define PG_GEOMETRY_TYPE_DEFINED
typedef struct
{
	cl_bool		isnull;
	cl_uchar	type;		/* one of GEOM_*TYPE */
	cl_ushort	flags;		/* combination of GEOM_FLAG__* */
	cl_int		srid;		/* SRID of this geometry */
	cl_uint		nitems;		/* # of items; what it exactly means depends on
							 * the geometry type */
	cl_uint		rawsize;	/* length of the rawdata buffer */
	const char *rawdata;	/* pointer to the raw-data; it may not be aligned,
							 * so needs to copy local buffer once */
	geom_bbox  *bbox;		/* boundary box if any */
} pg_geometry_t;
STROMCL_EXTERNAL_VARREF_TEMPLATE(geometry);
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(geometry);
STROMCL_UNSUPPORTED_ARROW_TEMPLATE(geometry);
#endif /* PG_GEOMETRY_TYPE_DEFINED */

#ifndef PG_BOX2DF_TYPE_DEFINED
#define PG_BOX2DF_TYPE_DEFINED
STROMCL_SIMPLE_DATATYPE_TEMPLATE(box2df, geom_bbox_2d)
STROMCL_EXTERNAL_VARREF_TEMPLATE(box2df)
STROMCL_UNSUPPORTED_ARROW_TEMPLATE(box2df)
#endif	/* PG_BOX2DF_TYPE_DEFINED */

/*
 * POINT2D, POINT3D, POINT3DM, POINT4D
 */
typedef struct
{
	double	x, y;
} POINT2D;

typedef struct
{
	double	x, y, z;
} POINT3D;

typedef struct
{
	double	x, y, z;
} POINT3DZ;

typedef struct
{
	double	x, y, m;
} POINT3DM;

typedef struct
{
	double	x, y, z, m;
} POINT4D;

#ifdef __CUDACC__
/* for DATUM_CLASS__GEOMETRY */
DEVICE_FUNCTION(cl_uint)
pg_geometry_datum_length(kern_context *kcxt, Datum datum);
DEVICE_FUNCTION(cl_uint)
pg_geometry_datum_write(kern_context *kcxt, char *dest, Datum datum);

/*
 * box2df operators & functions
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_geometry_overlaps(kern_context *kcxt,
					   const pg_geometry_t &arg1,
					   const pg_geometry_t &arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_box2df_geometry_overlaps(kern_context *kcxt,
							  const pg_box2df_t &arg1,
							  const pg_geometry_t &arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_geometry_contains(kern_context *kcxt,
					   const pg_geometry_t &arg1,
					   const pg_geometry_t &arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_box2df_geometry_contains(kern_context *kcxt,
							  const pg_box2df_t &arg1,
							  const pg_geometry_t &arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_geometry_within(kern_context *kcxt,
					 const pg_geometry_t &arg1,
					 const pg_geometry_t &arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_box2df_geometry_within(kern_context *kcxt,
							const pg_box2df_t &arg1,
							const pg_geometry_t &arg2);

DEVICE_FUNCTION(pg_geometry_t)
pgfn_st_expand(kern_context *kcxt,
			   const pg_geometry_t &arg1, pg_float8_t arg2);

/*
 * GiST index handlers
 */
DEVICE_FUNCTION(cl_bool)
pgindex_gist_geometry_overlap(kern_context *kcxt,
							  PageHeaderData *i_page,
							  const pg_box2df_t &i_var,
							  const pg_geometry_t &i_arg);
DEVICE_FUNCTION(cl_bool)
pgindex_gist_box2df_overlap(kern_context *kcxt,
							PageHeaderData *i_page,
							const pg_box2df_t &i_var,
							const pg_box2df_t &i_arg);
DEVICE_FUNCTION(cl_bool)
pgindex_gist_geometry_contains(kern_context *kcxt,
							   PageHeaderData *i_page,
							   const pg_box2df_t &i_var,
							   const pg_geometry_t &i_arg);
DEVICE_FUNCTION(cl_bool)
pgindex_gist_box2df_contains(kern_context *kcxt,
							 PageHeaderData *i_page,
							 const pg_box2df_t &i_var,
							 const pg_box2df_t &i_arg);
DEVICE_FUNCTION(cl_bool)
pgindex_gist_geometry_contained(kern_context *kcxt,
								PageHeaderData *i_page,
								const pg_box2df_t &i_var,
								const pg_geometry_t &i_arg);
DEVICE_FUNCTION(cl_bool)
pgindex_gist_box2df_contained(kern_context *kcxt,
							  PageHeaderData *i_page,
							  const pg_box2df_t &i_var,
							  const pg_box2df_t &i_arg);

/*
 * PostGIS functions
 */
DEVICE_FUNCTION(pg_geometry_t)
pgfn_st_setsrid(kern_context *kcxt,
				const pg_geometry_t &arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_geometry_t)
pgfn_st_makepoint2(kern_context *kcxt,
				   pg_float8_t x, pg_float8_t y);
DEVICE_FUNCTION(pg_geometry_t)
pgfn_st_makepoint3(kern_context *kcxt,
				   pg_float8_t x, pg_float8_t y, pg_float8_t z);
DEVICE_FUNCTION(pg_geometry_t)
pgfn_st_makepoint4(kern_context *kcxt,
				   pg_float8_t x, pg_float8_t y,
				   pg_float8_t z, pg_float8_t m);
DEVICE_FUNCTION(pg_float8_t)
pgfn_st_distance(kern_context *kcxt,
				 const pg_geometry_t &arg1,
				 const pg_geometry_t &arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_st_dwithin(kern_context *kcxt,
				const pg_geometry_t &arg1,
				const pg_geometry_t &arg2,
				pg_float8_t arg3);
DEVICE_FUNCTION(pg_int4_t)
pgfn_st_linecrossingdirection(kern_context *kcxt,
							  const pg_geometry_t &arg1,
							  const pg_geometry_t &arg2);
DEVICE_FUNCTION(pg_text_t)
pgfn_st_relate(kern_context *kcxt,
			   const pg_geometry_t &arg1,
			   const pg_geometry_t &arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_st_contains(kern_context *kcxt,
				 const pg_geometry_t &arg1,
				 const pg_geometry_t &arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_st_crosses(kern_context *kcxt,
				const pg_geometry_t &arg1,
				const pg_geometry_t &arg2);

#endif /* __CUDACC__ */

#endif /* CUDA_POSTGIS_H */
