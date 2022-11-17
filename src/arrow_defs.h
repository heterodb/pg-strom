/*
 * arrow_defs.h
 *
 * definitions for apache arrow format
 */
#ifndef _ARROW_DEFS_H_
#define _ARROW_DEFS_H_

#ifdef __cplusplus
typedef bool			__boolean;
#else
typedef unsigned char	__boolean;
#ifndef true
#define true	((__boolean) 1)
#endif
#ifndef false
#define false	((__boolean) 0)
#endif
#endif	/* !__CUDACC__ */

/*
 * MetadataVersion : short
 */
typedef enum
{
	ArrowMetadataVersion__V1 = 0,		/* not supported */
	ArrowMetadataVersion__V2 = 1,		/* not supported */
	ArrowMetadataVersion__V3 = 2,		/* not supported */
	ArrowMetadataVersion__V4 = 3,
	ArrowMetadataVersion__V5 = 4,
} ArrowMetadataVersion;

/*
 * Feature : long
 */
typedef enum
{
	/* Makes flatbuffers happy */
	ArrowFeature__Unused = 0,
	/*
	 * The stream makes use of multiple full dictionaries with the
	 * same ID and assumes clients implement dictionary replacement
	 * correctly.
	 */
	ArrowFeature__DictionaryReplacement = 1,
	/*
	 * The stream makes use of compressed bodies as described
	 * in the Message
	 */
	ArrowFeature__CompressedBody = 2,
} ArrowFeature;

/*
 * MessageHeader : byte
 */
typedef enum
{
	ArrowMessageHeader__Schema			= 1,
	ArrowMessageHeader__DictionaryBatch	= 2,
	ArrowMessageHeader__RecordBatch		= 3,
	ArrowMessageHeader__Tensor			= 4,
	ArrowMessageHeader__SparseTensor	= 5,
} ArrowMessageHeader;

/*
 * Endianness : short
 */
typedef enum
{
	ArrowEndianness__Little		= 0,
	ArrowEndianness__Big		= 1,
} ArrowEndianness;

/*
 * Type : byte
 */
typedef enum
{
	ArrowType__Null				= 1,
	ArrowType__Int				= 2,
	ArrowType__FloatingPoint	= 3,
	ArrowType__Binary			= 4,
	ArrowType__Utf8				= 5,
	ArrowType__Bool				= 6,
	ArrowType__Decimal			= 7,
	ArrowType__Date				= 8,
	ArrowType__Time				= 9,
	ArrowType__Timestamp		= 10,
	ArrowType__Interval			= 11,
	ArrowType__List				= 12,
	ArrowType__Struct			= 13,
	ArrowType__Union			= 14,
	ArrowType__FixedSizeBinary	= 15,
	ArrowType__FixedSizeList	= 16,
	ArrowType__Map				= 17,
	ArrowType__Duration			= 18,
	ArrowType__LargeBinary		= 19,
	ArrowType__LargeUtf8		= 20,
	ArrowType__LargeList		= 21,
} ArrowTypeTag;

/*
 * DateUnit : short
 */
typedef enum
{
	ArrowDateUnit__Day			= 0,
	ArrowDateUnit__MilliSecond	= 1,
} ArrowDateUnit;

/*
 * TimeUnit : short
 */
typedef enum
{
	ArrowTimeUnit__Second		= 0,
	ArrowTimeUnit__MilliSecond	= 1,
	ArrowTimeUnit__MicroSecond	= 2,
	ArrowTimeUnit__NanoSecond	= 3,
} ArrowTimeUnit;

/*
 * IntervalUnit : short
 */
typedef enum
{
	ArrowIntervalUnit__Year_Month	= 0,
	ArrowIntervalUnit__Day_Time		= 1,
} ArrowIntervalUnit;

/*
 * Precision : short
 */
typedef enum
{
	ArrowPrecision__Half		= 0,
	ArrowPrecision__Single		= 1,
	ArrowPrecision__Double		= 2,
} ArrowPrecision;

/*
 * UnionMode : short
 */
typedef enum
{
	ArrowUnionMode__Sparse		= 0,
	ArrowUnionMode__Dense		= 1,
} ArrowUnionMode;

/*
 * ArrowNodeTag
 */
typedef enum
{
	/* types */
	ArrowNodeTag__Null,
	ArrowNodeTag__Int,
	ArrowNodeTag__FloatingPoint,
	ArrowNodeTag__Utf8,
	ArrowNodeTag__Binary,
	ArrowNodeTag__Bool,
	ArrowNodeTag__Decimal,
	ArrowNodeTag__Date,
	ArrowNodeTag__Time,
	ArrowNodeTag__Timestamp,
	ArrowNodeTag__Interval,
	ArrowNodeTag__List,
	ArrowNodeTag__Struct,
	ArrowNodeTag__Union,
	ArrowNodeTag__FixedSizeBinary,
	ArrowNodeTag__FixedSizeList,
	ArrowNodeTag__Map,
	ArrowNodeTag__Duration,
	ArrowNodeTag__LargeBinary,
	ArrowNodeTag__LargeUtf8,
	ArrowNodeTag__LargeList,
	/* others */
	ArrowNodeTag__KeyValue,
	ArrowNodeTag__DictionaryEncoding,
	ArrowNodeTag__Field,
	ArrowNodeTag__FieldNode,
	ArrowNodeTag__Buffer,
	ArrowNodeTag__Schema,
	ArrowNodeTag__RecordBatch,
	ArrowNodeTag__DictionaryBatch,
	ArrowNodeTag__Message,
	ArrowNodeTag__Block,
	ArrowNodeTag__Footer,
	ArrowNodeTag__BodyCompression,
} ArrowNodeTag;

/*
 * ArrowTypeOptions - our own definition
 */
#define ARROW_TYPE_OPTIONS_COMMON_FIELDS		\
	ArrowTypeTag			tag;				\
	unsigned short			unitsz

typedef union		ArrowTypeOptions
{
	struct {
		ARROW_TYPE_OPTIONS_COMMON_FIELDS;
	} common;
	struct {
		ARROW_TYPE_OPTIONS_COMMON_FIELDS;
		unsigned short		bitWidth;
		__boolean			is_signed;
	} integer;
	struct {
		ARROW_TYPE_OPTIONS_COMMON_FIELDS;
		ArrowPrecision		precision;
	} floating_point;
	struct {
		ARROW_TYPE_OPTIONS_COMMON_FIELDS;
		unsigned short		precision;
		unsigned short		scale;
		unsigned short		bitWidth;
	} decimal;
	struct {
		ARROW_TYPE_OPTIONS_COMMON_FIELDS;
		ArrowDateUnit		unit;
	} date;
	struct {
		ARROW_TYPE_OPTIONS_COMMON_FIELDS;
		ArrowTimeUnit		unit;
	} time;
	struct {
		ARROW_TYPE_OPTIONS_COMMON_FIELDS;
		ArrowTimeUnit		unit;
	} timestamp;
	struct {
		ARROW_TYPE_OPTIONS_COMMON_FIELDS;
		ArrowIntervalUnit	unit;
	} interval;
	struct {
		ARROW_TYPE_OPTIONS_COMMON_FIELDS;
		int					byteWidth;
	} fixed_size_binary;
} ArrowTypeOptions;

#undef ARROW_TYPE_OPTIONS_COMMON_FIELDS

#ifndef __CUDACC__
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
/*
 * ArrowNode
 */
struct SQLbuffer;

struct ArrowNode
{
	ArrowNodeTag	tag;
	const char	   *tagName;
	void		  (*dumpArrowNode)(struct SQLbuffer *buf,
								   struct ArrowNode *node);
	void		  (*copyArrowNode)(struct ArrowNode *dest,
								   const struct ArrowNode *source);
};
typedef struct ArrowNode		ArrowNode;

#define ArrowNodeTag(PTR)		(((ArrowNode *)(PTR))->tag)
#define ArrowNodeIs(PTR,NAME)	(ArrowNodeTag(PTR) == ArrowNodeTag__##NAME)

/* Null */
typedef ArrowNode	ArrowTypeNull;

/* Int */
typedef struct		ArrowTypeInt
{
	ArrowNode		node;
	int32_t			bitWidth;
	__boolean		is_signed;
} ArrowTypeInt;

/* FloatingPoint */
typedef struct		ArrowTypeFloatingPoint
{
	ArrowNode		node;
	ArrowPrecision	precision;
} ArrowTypeFloatingPoint;

/* Utf8 */
typedef ArrowNode	ArrowTypeUtf8;

/* Binary  */
typedef ArrowNode	ArrowTypeBinary;

/* Bool */
typedef ArrowNode	ArrowTypeBool;

/* Decimal */
typedef struct		ArrowTypeDecimal
{
	ArrowNode		node;
	int32_t			precision;
	int32_t			scale;
	int32_t			bitWidth;		/* 128 [default] or 256 */
} ArrowTypeDecimal;

/* Date */
typedef struct		ArrowTypeDate
{
	ArrowNode		node;
	ArrowDateUnit	unit;
} ArrowTypeDate;

/* Time */
typedef struct		ArrowTypeTime
{
	ArrowNode		node;
	ArrowTimeUnit	unit;
	int32_t			bitWidth;
} ArrowTypeTime;

/* Timestamp */
typedef struct		ArrowTypeTimestamp
{
	ArrowNode		node;
	ArrowTimeUnit	unit;
	const char	   *timezone;
	int32_t			_timezone_len;
} ArrowTypeTimestamp;

/* Interval */
typedef struct		ArrowTypeInterval
{
	ArrowNode		node;
	ArrowIntervalUnit unit;
} ArrowTypeInterval;

/* List */
typedef ArrowNode	ArrowTypeList;

/* Struct */
typedef ArrowNode	ArrowTypeStruct;

/* Union */
typedef struct		ArrowTypeUnion
{
	ArrowNode		node;
	ArrowUnionMode	mode;
	int32_t		   *typeIds;
	int32_t			_num_typeIds;
} ArrowTypeUnion;

/* FixedSizeBinary */
typedef struct		ArrowTypeFixedSizeBinary
{
	ArrowNode		node;
	int32_t			byteWidth;
} ArrowTypeFixedSizeBinary;

/* FixedSizeList */
typedef struct		ArrowTypeFixedSizeList
{
	ArrowNode		node;
	int32_t			listSize;
} ArrowTypeFixedSizeList;

/* Map */
typedef struct		ArrowTypeMap
{
	ArrowNode		node;
	__boolean		keysSorted;
} ArrowTypeMap;

/* Duration */
typedef struct		ArrowTypeDuration
{
	ArrowNode		node;
	ArrowTimeUnit	unit;
} ArrowTypeDuration;

/* LargeBinary */
typedef ArrowNode	ArrowTypeLargeBinary;

/* LargeUtf8 */
typedef ArrowNode	ArrowTypeLargeUtf8;

/* LargeList */
typedef ArrowNode	ArrowTypeLargeList;

/*
 * ArrowType
 */
typedef union		ArrowType
{
	ArrowNode				node;
	ArrowTypeNull			Null;
	ArrowTypeInt			Int;
	ArrowTypeFloatingPoint	FloatingPoint;
	ArrowTypeUtf8			Utf8;
	ArrowTypeBinary			Binary;
	ArrowTypeBool			Bool;
	ArrowTypeDecimal		Decimal;
	ArrowTypeDate			Date;
	ArrowTypeTime			Time;
	ArrowTypeTimestamp		Timestamp;
	ArrowTypeInterval		Interval;
	ArrowTypeList			List;
	ArrowTypeStruct			Struct;
	ArrowTypeUnion			Union;
	ArrowTypeFixedSizeBinary FixedSizeBinary;
	ArrowTypeFixedSizeList	FixedSizeList;
	ArrowTypeMap			Map;
	ArrowTypeDuration		Duration;
	ArrowTypeLargeBinary	LargeBinary;
	ArrowTypeLargeUtf8		LargeUtf8;
	ArrowTypeLargeList		LargeList;
} ArrowType;

/*
 * Buffer
 */
typedef struct		ArrowBuffer
{
	ArrowNode		node;
	int64_t			offset;
	int64_t			length;
} ArrowBuffer;

/*
 * KeyValue
 */
typedef struct		ArrowKeyValue
{
	ArrowNode		node;
	const char	   *key;
	const char	   *value;
	int				_key_len;
	int				_value_len;
} ArrowKeyValue;

/*
 * DictionaryEncoding
 */
typedef struct		ArrowDictionaryEncoding
{
	ArrowNode		node;
	int64_t			id;
	ArrowTypeInt	indexType;
	__boolean		isOrdered;
} ArrowDictionaryEncoding;

/*
 * Field
 */
typedef struct		ArrowField
{
	ArrowNode		node;
	const char	   *name;
	int				_name_len;
	__boolean		nullable;
	ArrowType		type;
	ArrowDictionaryEncoding *dictionary;
	/* vector of nested data types */
	struct ArrowField *children;
	int				_num_children;
	/* vector of user defined metadata */
	ArrowKeyValue  *custom_metadata;
	int				_num_custom_metadata;
} ArrowField;

/*
 * FieldNode
 */
typedef struct		ArrowFieldNode
{
	ArrowNode		node;
	uint64_t		length;
	uint64_t		null_count;
} ArrowFieldNode;

/*
 * CompressionType : byte
 */
typedef enum		ArrowCompressionType
{
	ArrowCompressionType__LZ4_FRAME = 0,
	ArrowCompressionType__ZSTD = 1,
} ArrowCompressionType;

/*
 * BodyCompressionMethod : byte
 */
typedef enum		ArrowBodyCompressionMethod
{
	ArrowBodyCompressionMethod__BUFFER = 0,
} ArrowBodyCompressionMethod;

/*
 * BodyCompression
 */
typedef struct		ArrowBodyCompression
{
	ArrowNode		node;
	ArrowCompressionType		codec;
	ArrowBodyCompressionMethod	method;
} ArrowBodyCompression;

/*
 * Schema
 */
typedef struct		ArrowSchema
{
	ArrowNode		node;
	ArrowEndianness	endianness;
	/* vector of Field */
	ArrowField	   *fields;
	int				_num_fields;
	/* List of KeyValue */
	ArrowKeyValue  *custom_metadata;
	int				_num_custom_metadata;
	/* List of Features */
	ArrowFeature   *features;
	int				_num_features;
} ArrowSchema;

/*
 * RecordBatch
 */
typedef struct		ArrowRecordBatch
{
	ArrowNode		node;
	int64_t			length;
	/* vector of FieldNode */
	ArrowFieldNode  *nodes;
	int				_num_nodes;
	/* vector of Buffer */
	ArrowBuffer	    *buffers;
	int				_num_buffers;
	/* optional compression of the message body */
	ArrowBodyCompression *compression;
} ArrowRecordBatch;

/*
 * DictionaryBatch 
 */
typedef struct		ArrowDictionaryBatch
{
	ArrowNode		node;
	int64_t			id;
	ArrowRecordBatch data;
	__boolean		isDelta;
} ArrowDictionaryBatch;

/*
 * ArrowMessageBody
 */
typedef union
{
	ArrowNode		node;
	ArrowSchema		schema;
	ArrowDictionaryBatch dictionaryBatch;
	ArrowRecordBatch recordBatch;
} ArrowMessageBody;

/*
 * Message
 */
typedef struct		ArrowMessage
{
	ArrowNode		node;
	ArrowMetadataVersion version;
	ArrowMessageBody body;
	uint64_t		bodyLength;
} ArrowMessage;

/*
 * Block
 */
typedef struct		ArrowBlock
{
	ArrowNode		node;
	int64_t			offset;
	int32_t			metaDataLength;
	int64_t			bodyLength;
} ArrowBlock;

/*
 * Footer
 */
typedef struct		ArrowFooter
{
	ArrowNode		node;
	ArrowMetadataVersion version;
	ArrowSchema		schema;
	ArrowBlock	   *dictionaries;
	int				_num_dictionaries;
	ArrowBlock	   *recordBatches;
	int				_num_recordBatches;
} ArrowFooter;

/*
 * ArrowFileInfo - state information of readArrowFileDesc()
 */
typedef struct
{
	const char	   *filename;
	struct stat		stat_buf;
	ArrowFooter		footer;
	ArrowMessage   *dictionaries;	/* array of ArrowDictionaryBatch */
	ArrowMessage   *recordBatches;	/* array of ArrowRecordBatch */
} ArrowFileInfo;

#endif		/* !__CUDACC__ */
#endif		/* _ARROW_DEFS_H_ */
