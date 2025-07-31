/*
 * arrow-tools.h
 *
 * common definitions for arrow/parquet tools
 */
#ifndef _ARROW_TOOLS_H_
#define _ARROW_TOOLS_H_
#include <arrow/api.h>				/* dnf install libarrow-devel */
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/util/value_parsing.h>
#ifdef HAS_PARQUET
#include <parquet/arrow/reader.h>	/* dnf install parquet-libs-devel */
#include <parquet/file_reader.h>
#include <parquet/schema.h>
#include <parquet/stream_writer.h>
#endif

#define Elog(fmt,...)									\
	do {												\
		fprintf(stderr, "[ERROR %s:%d] " fmt "\n",		\
				__FILE__,__LINE__, ##__VA_ARGS__);		\
		exit(1);										\
	} while(0)
#define Info(fmt,...)									\
	do {												\
		if (verbose > 0)								\
			fprintf(stderr, "[INFO %s:%d] " fmt "\n",	\
					__FILE__,__LINE__, ##__VA_ARGS__);	\
	} while(0)

#define Debug(fmt,...)									\
	do {												\
		if (verbose > 1)								\
			fprintf(stderr, "[DEBUG %s:%d] " fmt "\n",	\
					__FILE__,__LINE__, ##__VA_ARGS__);	\
	} while(0)


#ifdef __cplusplus
//
// Useful definitions
//
typedef std::string							cppString;
typedef std::vector<cppString>				cppStringVec;

//
// Routines only valid for C++
//
extern "C" void
dump_arrow_metadata(std::shared_ptr<arrow::io::ReadableFile> arrow_file, const char *filename);

#endif
#endif	/* _ARROW_TOOLS_H_ */
