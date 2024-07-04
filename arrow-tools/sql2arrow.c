/*
 * sql2arrow.c - main logic of xxx2arrow command
 *
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "sql2arrow.h"
#include <assert.h>
#include <ctype.h>
#include <getopt.h>
#include <limits.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <time.h>

/* command options */
static char	   *simple_table_name = NULL;
static char	   *sqldb_command = NULL;
static char	   *output_filename = NULL;
static char	   *append_filename = NULL;
static size_t	batch_segment_sz = 0;
static char	   *sqldb_hostname = NULL;
static char	   *sqldb_port_num = NULL;
static char	   *sqldb_username = NULL;
static char	   *sqldb_password = NULL;
static char	   *sqldb_database = NULL;
static char	   *dump_arrow_filename = NULL;
static char	   *schema_arrow_filename = NULL;
static char	   *schema_arrow_tablename = NULL;
static char	   *stat_embedded_columns = NULL;
static int		num_worker_threads = 0;
static char	   *parallel_dist_keys = NULL;
static int		shows_progress = 0;
static userConfigOption *sqldb_session_configs = NULL;
static nestLoopOption *sqldb_nestloop_options = NULL;

/*
 * Per-worker state variables
 */
static volatile bool	worker_setup_done  = false;
static pthread_mutex_t	worker_setup_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	worker_setup_cond  = PTHREAD_COND_INITIALIZER;
static pthread_t	   *worker_threads;
static SQLtable		  **worker_tables;
static const char	  **worker_dist_keys = NULL;
static pthread_mutex_t	main_table_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * __trim
 */
static inline char *
__trim(char *token)
{
	char   *tail = token + strlen(token) - 1;

	while (*token == ' ' || *token == '\t')
		token++;
	while (tail >= token && (*tail == ' ' || *tail == '\t'))
		*tail-- = '\0';
	return token;
}

/*
 * loadArrowDictionaryBatches
 */
static SQLdictionary *
__loadArrowDictionaryBatchOne(const char *message_head,
							  ArrowDictionaryBatch *dbatch)
{
	SQLdictionary  *dict;
	ArrowBuffer	   *v_buffer = &dbatch->data.buffers[1];
	ArrowBuffer	   *e_buffer = &dbatch->data.buffers[2];
	uint32_t		   *values = (uint32_t *)(message_head + v_buffer->offset);
	char		   *extra = (char *)(message_head + e_buffer->offset);
	int				i;

	dict = palloc0(offsetof(SQLdictionary, hslots[1024]));
	dict->dict_id = dbatch->id;
	sql_buffer_init(&dict->values);
	sql_buffer_init(&dict->extra);
	dict->nloaded = dbatch->data.length;
	dict->nitems  = dbatch->data.length;
	dict->nslots = 1024;

	for (i=0; i < dict->nloaded; i++)
	{
		hashItem   *hitem;
		uint32_t		len = values[i+1] - values[i];
		char	   *pos = extra + values[i];
		uint32_t		hindex;

		hitem = palloc(offsetof(hashItem, label[len+1]));
		hitem->hash = hash_any((unsigned char *)pos, len);
		hitem->index = i;
		hitem->label_sz = len;
		memcpy(hitem->label, pos, len);
		hitem->label[len] = '\0';
		
		hindex = hitem->hash % dict->nslots;
		hitem->next = dict->hslots[hindex];
		dict->hslots[hindex] = hitem;
	}
	assert(dict->nitems == dict->nloaded);
	return dict;
}

static SQLdictionary *
loadArrowDictionaryBatches(int fdesc, ArrowFileInfo *af_info)
{
	SQLdictionary *dictionary_list = NULL;
	size_t		mmap_sz;
	char	   *mmap_head = NULL;
	int			i;

	if (af_info->footer._num_dictionaries == 0)
		return NULL;	/* no dictionaries */
	mmap_sz = TYPEALIGN(sysconf(_SC_PAGESIZE), af_info->stat_buf.st_size);
	mmap_head = mmap(NULL, mmap_sz,
					 PROT_READ, MAP_SHARED,
					 fdesc, 0);
	if (mmap_head == MAP_FAILED)
		Elog("failed on mmap(2): %m");
	for (i=0; i < af_info->footer._num_dictionaries; i++)
	{
		ArrowBlock	   *block = &af_info->footer.dictionaries[i];
		ArrowDictionaryBatch *dbatch;
		SQLdictionary  *dict;
		char		   *message_head;
		char		   *message_body;

		dbatch = &af_info->dictionaries[i].body.dictionaryBatch;
		if (dbatch->node.tag != ArrowNodeTag__DictionaryBatch ||
			dbatch->data._num_nodes != 1 ||
			dbatch->data._num_buffers != 3)
			Elog("DictionaryBatch (dictionary_id=%ld) has unexpected format",
				 dbatch->id);
		message_head = mmap_head + block->offset;
		message_body = message_head + block->metaDataLength;
		dict = __loadArrowDictionaryBatchOne(message_body, dbatch);
		dict->next = dictionary_list;
		dictionary_list = dict;
	}
	munmap(mmap_head, mmap_sz);

	return dictionary_list;
}

/*
 * MEMO: Program termision may lead file corruption if arrow file is
 * already overwritten with --append mode. The callback below tries to
 * revert the arrow file using UNDO log; that is Footer portion saved
 * before any writing stuff.
 */
typedef struct
{
	int			append_fdesc;
	off_t		footer_offset;
	size_t		footer_length;
	char		footer_backup[FLEXIBLE_ARRAY_MEMBER];
} arrowFileUndoLog;

static arrowFileUndoLog	   *arrow_file_undo_log = NULL;

/*
 * fixup_append_file_on_exit
 */
static void
fixup_append_file_on_exit(int status, void *p)
{
	arrowFileUndoLog *undo = arrow_file_undo_log;
	ssize_t		count = 0;
	ssize_t		nbytes;

	if (status == 0 || !undo)
		return;
	/* avoid infinite recursion */
	arrow_file_undo_log = NULL;

	if (lseek(undo->append_fdesc,
			  undo->footer_offset, SEEK_SET) != undo->footer_offset)
	{
		fprintf(stderr, "failed on lseek(2) on applying undo log.\n");
		return;
	}

	while (count < undo->footer_length)
	{
		nbytes = write(undo->append_fdesc,
					   undo->footer_backup + count,
					   undo->footer_length - count);
		if (nbytes > 0)
			count += nbytes;
		else if (errno != EINTR)
		{
			fprintf(stderr, "failed on write(2) on applying undo log.\n");
			return;
		}
	}

	if (ftruncate(undo->append_fdesc,
				  undo->footer_offset + undo->footer_length) != 0)
	{
		fprintf(stderr, "failed on ftruncate(2) on applying undo log.\n");
		return;
	}
}

/*
 * fixup_append_file_on_signal
 */
static void
fixup_append_file_on_signal(int signum)
{
	fixup_append_file_on_exit(0x80 + signum, NULL);
	switch (signum)
	{
		/* SIG_DFL == Term */
		case SIGHUP:
		case SIGINT:
		case SIGTERM:
			exit(0x80 + signum);

		/* SIG_DFL == Core */
		default:
			fprintf(stderr, "unexpected signal (%d) was caught by %s\n",
					signum, __FUNCTION__);
		case SIGSEGV:
		case SIGBUS:
			abort();
	}
}

/*
 * setup_append_file
 */
static void
setup_append_file(SQLtable *table, ArrowFileInfo *af_info)
{
	arrowFileUndoLog *undo;
	uint32_t		nitems;
	ssize_t			nbytes;
	ssize_t			length;
	loff_t			offset;
	char		   *pos;
	char			buffer[64];

	/* restore DictionaryBatches already in the file */
	nitems = af_info->footer._num_dictionaries;
	table->numDictionaries = nitems;
	table->dictionaries = palloc0(sizeof(ArrowBlock) * nitems);
	memcpy(table->dictionaries,
		   af_info->footer.dictionaries,
		   sizeof(ArrowBlock) * nitems);

	/* restore RecordBatches already in the file */
	nitems = af_info->footer._num_recordBatches;
	table->numRecordBatches = nitems;
	table->recordBatches = palloc0(sizeof(ArrowBlock) * nitems);
	memcpy(table->recordBatches,
		   af_info->footer.recordBatches,
		   sizeof(ArrowBlock) * nitems);

	/* move to the file offset in front of the Footer portion */
	nbytes = sizeof(int32_t) + 6;		/* strlen("ARROW1") */
	offset = af_info->stat_buf.st_size - nbytes;
	if (pread(table->fdesc, buffer, nbytes, offset) != nbytes)
		Elog("failed on pread(2): %m");
	offset -= *((uint32_t *)buffer);
	if (lseek(table->fdesc, offset, SEEK_SET) < 0)
		Elog("failed on lseek(%d, %zu, SEEK_SET): %m",
			 table->fdesc, offset);
	table->f_pos = offset;
	length = af_info->stat_buf.st_size - offset;

	/* makes Undo log to recover process termination */
	undo = palloc(offsetof(arrowFileUndoLog,
						   footer_backup[length]));
	undo->append_fdesc = table->fdesc;
	undo->footer_offset = offset;
	undo->footer_length = length;
	pos = undo->footer_backup;
	while (length > 0)
	{
		nbytes = pread(undo->append_fdesc,
					   pos, length, offset);
		if (nbytes > 0)
		{
			pos += nbytes;
			length -= nbytes;
			offset += nbytes;
		}
		else if (nbytes == 0)
			Elog("pread: unexpected EOF; arrow file corruption?");
		else if (nbytes < 0 && errno != EINTR)
			Elog("failed on pread(2): %m");
	}
	arrow_file_undo_log = undo;
	/* register callbacks for unexpected process termination */
	on_exit(fixup_append_file_on_exit, NULL);
	signal(SIGHUP,  fixup_append_file_on_signal);
	signal(SIGINT,  fixup_append_file_on_signal);
	signal(SIGTERM, fixup_append_file_on_signal);
	signal(SIGSEGV, fixup_append_file_on_signal);
	signal(SIGBUS,  fixup_append_file_on_signal);
}

/*
 * setup_output_file
 */
static void
setup_output_file(SQLtable *table, const char *output_filename)
{
	int		fdesc;

	if (output_filename)
	{
		fdesc = open(output_filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
		if (fdesc < 0)
			Elog("failed on open('%s'): %m", output_filename);
		table->fdesc = fdesc;
		table->filename = output_filename;
	}
	else
	{
		char	temp[200];

		strcpy(temp, "/tmp/XXXXXX.arrow");
		fdesc = mkostemps(temp, 6, O_RDWR | O_CREAT | O_TRUNC);
		if (fdesc < 0)
			Elog("failed on mkostemps('%s'): %m", temp);
		table->fdesc = fdesc;
		table->filename = pstrdup(temp);
		fprintf(stderr,
				"NOTICE: -o, --output=FILENAME option was not given,\n"
				"        so a temporary file '%s' was built instead.\n", temp);
	}
	/* write out header stuff */
	arrowFileWrite(table, "ARROW1\0\0", 8);
	writeArrowSchema(table);
}

static int
dumpArrowFile(const char *filename)
{
	ArrowFileInfo	af_info;
	int				i, fdesc;

	memset(&af_info, 0, sizeof(ArrowFileInfo));
	fdesc = open(filename, O_RDONLY);
	if (fdesc < 0)
		Elog("unable to open '%s': %m", filename);
	readArrowFileDesc(fdesc, &af_info);

	printf("[Footer]\n%s\n",
		   dumpArrowNode((ArrowNode *)&af_info.footer));

	for (i=0; i < af_info.footer._num_dictionaries; i++)
	{
		printf("[Dictionary Batch %d]\n%s\n%s\n", i,
			   dumpArrowNode((ArrowNode *)&af_info.footer.dictionaries[i]),
			   dumpArrowNode((ArrowNode *)&af_info.dictionaries[i]));
	}

	for (i=0; i < af_info.footer._num_recordBatches; i++)
	{
		printf("[Record Batch %d]\n%s\n%s\n", i,
			   dumpArrowNode((ArrowNode *)&af_info.footer.recordBatches[i]),
			   dumpArrowNode((ArrowNode *)&af_info.recordBatches[i]));
	}
	close(fdesc);
	return 0;
}

/*
 * --schema option support routines
 */
static const char *
__field_name_escaped(const char *field_name)
{
	char   *__namebuf;
	int		i = 0;

	for (const char *c = field_name; *c != '\0'; c++)
	{
		if (!isalnum(*c) && *c != '_')
			goto escaped;
	}
	return field_name;
escaped:
	__namebuf = malloc(strlen(field_name) * 2 + 100);
	if (!__namebuf)
		Elog("out of memory");
	__namebuf[i++] = '\"';
	for (const char *c = field_name; *c != '\0'; c++)
	{
		if (*c == '\"')
			__namebuf[i++] = '\\';
		__namebuf[i++] = *c;
	}
	__namebuf[i++] = '\"';
	__namebuf[i++] = '\0';

	return __namebuf;
}

static const char *
__field_type_label(ArrowField *field)
{
	const char *pg_type = NULL;
	const char *label;
	char	   *namebuf, *pos;

	/* fetch "pg_type" custom-metadata */
	for (int i=0; i < field->_num_custom_metadata; i++)
	{
		ArrowKeyValue *kv = &field->custom_metadata[i];

		if (strcmp(kv->key, "pg_type") == 0)
		{
			pg_type = kv->value;
			if (strcmp(pg_type, "pg_catalog.macaddr") == 0 ||
				strcmp(pg_type, "macaddr") == 0)
			{
				if (field->type.node.tag == ArrowNodeTag__FixedSizeBinary &&
					field->type.FixedSizeBinary.byteWidth == 6)
					return "macaddr";
			}
			else if (strcmp(pg_type, "pg_catalog.inet") == 0 ||
					 strcmp(pg_type, "inet") == 0)
			{
				if (field->type.node.tag == ArrowNodeTag__FixedSizeBinary &&
					(field->type.FixedSizeBinary.byteWidth == 4 ||
					 field->type.FixedSizeBinary.byteWidth == 16))
					return "inet";
			}
			else if (strcmp(pg_type, "cube@cube") == 0)
			{
				if (field->type.node.tag == ArrowNodeTag__Binary ||
					field->type.node.tag == ArrowNodeTag__LargeBinary)
					return "cube";
			}
			break;
		}
	}

	switch (field->type.node.tag)
	{
		case ArrowNodeTag__Int:
			switch (field->type.Int.bitWidth)
			{
				case 8:	 return "int1";
				case 16: return "int2";
				case 32: return "int4";
				case 64: return "int8";
				default:
					Elog("Arrow::Int has unknown bitWidth (%d)",
						 field->type.Int.bitWidth);
			}
			break;

		case ArrowNodeTag__FloatingPoint:
			switch (field->type.FloatingPoint.precision)
			{
				case ArrowPrecision__Half:   return "float2";
				case ArrowPrecision__Single: return "float4";
				case ArrowPrecision__Double: return "float8";
				default:
					Elog("Arrow::FloatingPoint has unknown precision");
			}
			break;
		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__LargeUtf8:
			return "text";
		case ArrowNodeTag__Binary:
		case ArrowNodeTag__LargeBinary:
			return "bytea";
		case ArrowNodeTag__Bool:
			return "bool";
		case ArrowNodeTag__Decimal:
			return "numeric";
		case ArrowNodeTag__Date:
			return "date";
		case ArrowNodeTag__Time:
			return "time";
		case ArrowNodeTag__Timestamp:
			return "timestamp";
		case ArrowNodeTag__Interval:
			return "interval";
		case ArrowNodeTag__List:
		case ArrowNodeTag__LargeList:
			assert(field->_num_children == 1);
			label = __field_type_label(field->children);
			namebuf = alloca(strlen(label) + 10);
			sprintf(namebuf, "%s[]", label);
			return pstrdup(namebuf);
		case ArrowNodeTag__Struct:
			if (pg_type)
			{
				namebuf = alloca(strlen(pg_type) + 1);

				strcpy(namebuf, pg_type);
				/* strip earlier than '.' */
				pos = strchr(namebuf, '.');
				if (pos && pos[1] != '\0')
					namebuf = pos+1;
			}
			else
			{
				namebuf = alloca(strlen(field->name) + 10);

				sprintf(namebuf, "__%s_comp", field->name);
			}
			return pstrdup(namebuf);

		case ArrowNodeTag__FixedSizeBinary:
		case ArrowNodeTag__FixedSizeList:
			namebuf = alloca(32);
			sprintf(namebuf, "char(%d)",
					field->type.FixedSizeBinary.byteWidth);
			return pstrdup(namebuf);

		default:
			Elog("unknown Arrow type at field: %s", field->name);
			break;
	}
	Elog("unknown Arrow type");
}

static void
__schemaArrowComposite(ArrowField *field)
{
	const char *comp_name;

	if (field->type.node.tag != ArrowNodeTag__Struct)
		return;

	for (int j=0; j < field->_num_children; j++)
		__schemaArrowComposite(&field->children[j]);

	comp_name = __field_type_label(field);
	printf("---\n"
		   "--- composite type definition of %s\n"
		   "---\n"
		   "CREATE TYPE %s AS (\n",
		   field->name, __field_name_escaped(comp_name));
	for (int j=0; j < field->_num_children; j++)
	{
		if (j > 0)
			printf(",\n");
		printf("  %s %s",
			   __field_name_escaped(field->children[j].name),
			   __field_type_label(&field->children[j]));
	}
	printf("\n);\n\n");
}

static int
schemaArrowFile(const char *filename, const char *table_name)
{
	ArrowFileInfo af_info;
	ArrowSchema *schema;
	int		fdesc;

	memset(&af_info, 0, sizeof(ArrowFileInfo));
	fdesc = open(filename, O_RDONLY);
	if (fdesc < 0)
		Elog("unable to open '%s': %m", filename);
	readArrowFileDesc(fdesc, &af_info);
	schema = &af_info.footer.schema;
	/* TODO: dump enum types if any */

	/* dump composite types if any */
	for (int j=0; j < schema->_num_fields; j++)
		__schemaArrowComposite(&schema->fields[j]);

	/* CREATE TABLE command */
	if (!table_name)
	{
		char   *namebuf = alloca(strlen(filename) + 1);
		char   *pos;

		strcpy(namebuf, filename);
		namebuf = basename(namebuf);
		pos = strrchr(namebuf, '.');
		if (pos && pos != namebuf)
			*pos = '\0';
		table_name = namebuf;
	}
	printf("---\n"
		   "--- Definition of %s\n"
		   "--- (generated from '%s')\n"
		   "---\n"
		   "CREATE TABLE %s (\n",
		   table_name,
		   filename,
		   __field_name_escaped(table_name));
	for (int j=0; j < schema->_num_fields; j++)
	{
		if (j > 0)
			printf(",\n");
		printf("  %s %s",
			   __field_name_escaped(schema->fields[j].name),
			   __field_type_label(&schema->fields[j]));
		if (!schema->fields[j].nullable)
			printf(" not null");
	}
	printf("\n);\n");
	close(fdesc);
	return 0;
}

static char *
read_sql_command_from_file(const char *filename)
{
	struct stat stat_buf;
	int			fdesc;
	loff_t		off;
	ssize_t		nbytes;
	char	   *buffer;

	fdesc = open(filename, O_RDONLY);
	if (fdesc < 0)
		Elog("failed on open('%s'): %m", filename);
	if (fstat(fdesc, &stat_buf) != 0)
		Elog("failed on fstat('%s'): %m", filename);
	buffer = malloc(stat_buf.st_size + 1);
	if (!buffer)
		Elog("out of memory");

	off = 0;
	while (off < stat_buf.st_size)
	{
		nbytes = read(fdesc, buffer + off, stat_buf.st_size - off);
		if (nbytes < 0)
		{
			if (errno == EINTR)
				continue;
			Elog("failed on read('%s'): %m", filename);
		}
		else if (nbytes == 0)
		{
			Elog("unexpected EOF at '%s'", filename);
		}
		off += nbytes;
	}
	buffer[stat_buf.st_size] = '\0';
	close(fdesc);

	return buffer;
}

#ifdef __PG2ARROW__
static nestLoopOption *
parseNestLoopOption(const char *command, bool outer_join)
{
	nestLoopOption *nlopt = palloc0(offsetof(nestLoopOption, pnames[10]));
	const char *pos;
	const char *end;
	const char *sub_command;
	char	   *dest;

	if (strncmp(command, "file://", 7) == 0)
		command = read_sql_command_from_file(command + 7);

	sub_command = dest = palloc0(strlen(command) + 100);

	for (pos = command; *pos != '\0'; pos++)
	{
		if (*pos == '\\')
		{
			pos++;
			*dest++ = *pos;
			if (*pos == '\0')
				Elog("syntax error in: %s", command);
		}
		else if (*pos != '$')
		{
			*dest++ = *pos;
		}
		else
		{
			char	   *pname, *c;
			size_t		sz;

			pos++;
			if (*pos != '(')
				Elog("syntax error in: %s", command);
			end = strchr(pos, ')');
			if (!end)
				Elog("syntax error in: %s", command);
			pname = strndup(pos+1, end - (pos + 1));
			for (c = pname; *c != '\0'; c++)
			{
				if (!isalnum(*c) && *c != '_')
					Elog("--nestloop: field reference name should be '[0-9a-zA-Z_]+'");
			}
			sz = offsetof(nestLoopOption, pnames[nlopt->n_params + 1]);
			nlopt = repalloc(nlopt, sz);
			nlopt->pnames[nlopt->n_params++] = pname;

			dest += sprintf(dest, "$%d", nlopt->n_params);
			pos = end;
		}
	}
	*dest = '\0';

	nlopt->sub_command = sub_command;
	nlopt->outer_join = outer_join;

	return nlopt;
}
#endif	/* __PG2ARROW__ */

int
parseParallelDistKeys(const char *parallel_dist_keys, const char *delim)
{
	char   *temp = pstrdup(parallel_dist_keys);
	char   *tok, *pos;
	int		nitems = 0;
	int		nrooms = 25;

	worker_dist_keys = palloc0(sizeof(const char *) * nrooms);
	for (tok = strtok_r(temp, delim, &pos);
		 tok != NULL;
		 tok = strtok_r(NULL, delim, &pos))
	{
		tok = __trim(tok);

		if (nitems >= nrooms)
		{
			nrooms += nrooms;
			worker_dist_keys = repalloc(worker_dist_keys,
										sizeof(const char *) * nrooms);
		}
		worker_dist_keys[nitems++] = tok;
	}
	return nitems;
}

static bool
__enable_field_stats(SQLfield *field)
{
	bool	retval = (field->write_stat != NULL);
	int		j;

	field->stat_enabled = retval;
	memset(&field->stat_datum, 0, sizeof(SQLstat));
	field->stat_list = NULL;

	if (field->element)
	{
		if (__enable_field_stats(field->element))
			retval = true;
	}
	for (j=0; j < field->nfields; j++)
	{
		if (__enable_field_stats(&field->subfields[j]))
			retval = true;
	}
	return retval;
}

static void
enable_embedded_stats(SQLtable *table)
{
	char	   *buffer;
	char	   *name, *pos;
	int			j;

	/* disabled? */
	if (!stat_embedded_columns)
		return;

	/* special case - all available columns? */
	if (strcmp(stat_embedded_columns, "*") == 0)
	{
		for (j=0; j < table->nfields; j++)
		{
			if (__enable_field_stats(&table->columns[j]))
				table->has_statistics = true;
		}
		return;
	}

	/* elsewhere, enables stat for each column specified */
	buffer = alloca(strlen(stat_embedded_columns) + 1);
	strcpy(buffer, stat_embedded_columns);
	for (name = strtok_r(buffer, ",", &pos);
		 name != NULL;
		 name = strtok_r(NULL, ",", &pos))
	{
		bool	found = false;

		name = __trim(name);
		for (j=0; j < table->nfields; j++)
		{
			SQLfield   *field = &table->columns[j];

			if (strcmp(field->field_name, name) == 0)
			{
				if (__enable_field_stats(field))
				{
					table->has_statistics = found = true;
				}
				else
				{
					Elog("field [%s; %s] does not support min/max statistics",
						 name, field->arrow_type.node.tagName);
				}
			}
		}

		if (!found)
			Elog("field name [%s], specified by --stat option, was not found",
				 name);
	}
}

static void
usage(void)
{
	fputs("Usage:\n"
#ifdef __PG2ARROW__
		  "  pg2arrow [OPTION] [database] [username]\n\n"
#else
		  "  mysql2arrow [OPTION] [database] [username]\n\n"
#endif
		  "General options:\n"
		  "  -d, --dbname=DBNAME   Database name to connect to\n"
		  "  -c, --command=COMMAND SQL command to run\n"
		  "  -t, --table=TABLENAME Equivalent to '-c SELECT * FROM TABLENAME'\n"
		  "      (-c and -t are exclusive, either of them must be given)\n"
		  "  -n, --num-workers=N_WORKERS    Enables parallel dump mode.\n"
		  "                        It requires the SQL command contains $(WORKER_ID)\n"
		  "                        and $(N_WORKERS), to be replaced by the numeric\n"
		  "                        worker-id and number of workers.\n"
		  "  -k, --parallel-keys=PARALLEL_KEYS Enables yet another parallel dump.\n"
		  "                        It requires the SQL command contains $(PARALLEL_KEY)\n"
		  "                        to be replaced by the comma separated token in the\n"
		  "                        PARALLEL_KEYS.\n"
		  "      (-n and -k are exclusive, either of them can be give if parallel dump.\n"
		  "       It is user's responsibility to avoid data duplication.)\n"
#ifdef __PG2ARROW__
		  "      --inner-join=SUB_COMMAND\n"
		  "      --outer-join=SUB_COMMAND\n"
#endif
		  "  -o, --output=FILENAME result file in Apache Arrow format\n"
		  "      --append=FILENAME result Apache Arrow file to be appended\n"
		  "      (--output and --append are exclusive. If neither of them\n"
		  "       are given, it creates a temporary file.)\n"
		  "  -S, --stat[=COLUMNS] embeds min/max statistics for each record batch\n"
		  "                       COLUMNS is a comma-separated list of the target\n"
		  "                       columns if partially enabled.\n"
		  "\n"
		  "Arrow format options:\n"
		  "  -s, --segment-size=SIZE size of record batch for each\n"
		  "\n"
		  "Connection options:\n"
		  "  -h, --host=HOSTNAME  database server host\n"
		  "  -p, --port=PORT      database server port\n"
		  "  -u, --user=USERNAME  database user name\n"
#ifdef __PG2ARROW__
		  "  -w, --no-password    never prompt for password\n"
		  "  -W, --password       force password prompt\n"
#endif
#ifdef __MYSQL2ARROW__
		  "  -P, --password=PASS  Password to use when connecting to server\n"
#endif
		  "\n"
		  "Other options:\n"
		  "      --dump=FILENAME  dump information of arrow file\n"
		  "      --schema=FILENAME dump schema definition as CREATE TABLE statement\n"
		  "      --schema-name=NAME table name in the CREATE TABLE statement\n"
		  "      --progress       shows progress of the job\n"
		  "      --set=NAME:VALUE config option to set before SQL execution\n"
		  "      --help           shows this message\n"
		  "\n"
		  "Report bugs to <pgstrom@heterodb.com>.\n",
		  stderr);
	exit(1);
}

static void
parse_options(int argc, char * const argv[])
{
	static struct option long_options[] = {
		{"dbname",       required_argument, NULL, 'd'},
		{"command",      required_argument, NULL, 'c'},
		{"table",        required_argument, NULL, 't'},
		{"output",       required_argument, NULL, 'o'},
		{"append",       required_argument, NULL, 1000},
		{"segment-size", required_argument, NULL, 's'},
		{"host",         required_argument, NULL, 'h'},
		{"port",         required_argument, NULL, 'p'},
		{"user",         required_argument, NULL, 'u'},
#ifdef __PG2ARROW__
		{"no-password",  no_argument,       NULL, 'w'},
		{"password",     no_argument,       NULL, 'W'},
#endif /* __PG2ARROW__ */
#ifdef __MYSQL2ARROW__
		{"password",     required_argument, NULL, 'P'},
#endif /* __MYSQL2ARROW__ */
		{"dump",         required_argument, NULL, 1001},
		{"progress",     no_argument,       NULL, 1002},
		{"set",          required_argument, NULL, 1003},
		{"inner-join",   required_argument, NULL, 1004},
		{"outer-join",   required_argument, NULL, 1005},
		{"schema",       required_argument, NULL, 1006},
		{"schema-name",  required_argument, NULL, 1007},
		{"stat",         optional_argument, NULL, 'S'},
		{"num-workers",  required_argument, NULL, 'n'},
		{"parallel-keys",required_argument, NULL, 'k'},
		{"help",         no_argument,       NULL, 9999},
		{NULL, 0, NULL, 0},
	};
	int			c;
	int			password_prompt = 0;
	userConfigOption *last_user_config = NULL;
	nestLoopOption *last_nest_loop __attribute__((unused)) = NULL;

	while ((c = getopt_long(argc, argv,
							"d:c:t:o:s:h:p:u:"
#ifdef __PG2ARROW__
							"wW"
#endif
#ifdef __MYSQL2ARROW__
							"P:"
#endif
							"n:k:S::", long_options, NULL)) >= 0)
	{
		switch (c)
		{
			case 'd':
				if (sqldb_database)
					Elog("-d option was supplied twice");
				sqldb_database = optarg;
				break;

			case 'c':
				if (sqldb_command)
					Elog("-c option was supplied twice");
				if (simple_table_name)
					Elog("-c and -t options are exclusive");
				if (strncmp(optarg, "file://", 7) == 0)
					sqldb_command = read_sql_command_from_file(optarg + 7);
				else
					sqldb_command = __trim(optarg);
				break;

			case 't':
				if (simple_table_name)
					Elog("-t option was supplied twice");
				if (sqldb_command)
					Elog("-c and -t options are exclusive");
				simple_table_name = __trim(optarg);
				break;

			case 'o':
				if (output_filename)
					Elog("-o option was supplied twice");
				if (append_filename)
					Elog("-o and --append are exclusive");
				output_filename = optarg;
				break;

			case 1000:		/* --append */
				if (append_filename)
					Elog("--append option was supplied twice");
				if (output_filename)
					Elog("-o and --append are exclusive");
				append_filename = optarg;
				break;
				
			case 's':
				if (batch_segment_sz != 0)
					Elog("-s option was supplied twice");
				else
				{
					char   *end;
					long	sz = strtoul(optarg, &end, 10);

					if (sz == 0)
						Elog("not a valid segment size: %s", optarg);
					else if (*end == '\0')
						batch_segment_sz = sz;
					else if (strcasecmp(end, "k") == 0 ||
							 strcasecmp(end, "kb") == 0)
						batch_segment_sz = (sz << 10);
					else if (strcasecmp(end, "m") == 0 ||
							 strcasecmp(end, "mb") == 0)
						batch_segment_sz = (sz << 20);
					else if (strcasecmp(end, "g") == 0 ||
							 strcasecmp(end, "gb") == 0)
						batch_segment_sz = (sz << 30);
					else
						Elog("not a valid segment size: %s", optarg);
				}
				break;

			case 'n':
				if (parallel_dist_keys != 0)
					Elog("-n and -k are exclusive");
				else if (num_worker_threads != 0)
					Elog("-n option was supplied twice");
				else
				{
					char   *end;
					long	num = strtoul(optarg, &end, 10);

					if (*end != '\0' || num < 1 || num > 9999)
						Elog("not a valid -n|--num-workers option: %s", optarg);
					num_worker_threads = num;
				}
				break;

			case 'k':
				if (parallel_dist_keys != 0)
					Elog("-k option was supplied twice");
				else if (num_worker_threads != 0)
					Elog("-n and -k are exclusive");
				else
					parallel_dist_keys = pstrdup(optarg);
				break;

			case 'h':
				if (sqldb_hostname)
					Elog("-h option was supplied twice");
				sqldb_hostname = optarg;
				break;

			case 'p':
				if (sqldb_port_num)
					Elog("-p option was supplied twice");
				sqldb_port_num = optarg;
				break;

			case 'u':
				if (sqldb_username)
					Elog("-u option was supplied twice");
				sqldb_username = optarg;
				break;
#ifdef __PG2ARROW__
			case 'w':
				if (password_prompt > 0)
					Elog("-w and -W options are exclusive");
				password_prompt = -1;
				break;
			case 'W':
				if (password_prompt < 0)
					Elog("-w and -W options are exclusive");
				password_prompt = 1;
				break;
#endif	/* __PG2ARROW__ */
#ifdef __MYSQL2ARROW__
			case 'P':
				if (sqldb_password)
					Elog("-p option was supplied twice");
				sqldb_password = optarg;
				break;
#endif /* __MYSQL2ARROW__ */
			case 1001:		/* --dump */
				if (schema_arrow_filename)
					Elog("--dump and --schema are exclusive");
				if (dump_arrow_filename)
					Elog("--dump option was supplied twice");
				dump_arrow_filename = optarg;
				break;

			case 1002:		/* --progress */
				if (shows_progress)
					Elog("--progress option was supplied twice");
				shows_progress = 1;
				break;

			case 1003:		/* --set */
				{
					userConfigOption *conf;
					char   *pos, *tail;

					pos = strchr(optarg, ':');
					if (!pos)
						Elog("--set option should take NAME:VALUE");
					*pos++ = '\0';
					while (isspace(*pos))
                        pos++;
					while (tail > pos && isspace(*tail))
						tail--;
					conf = palloc0(sizeof(userConfigOption) +
								   strlen(optarg) + strlen(pos) + 40);
					sprintf(conf->query, "SET %s = '%s'", optarg, pos);
					if (last_user_config)
						last_user_config->next = conf;
					else
						sqldb_session_configs = conf;
					last_user_config = conf;
				}
				break;
#ifdef __PG2ARROW__
			case 1004:		/* --inner-join */
			case 1005:		/* --outer-join */
				{
					nestLoopOption *nlopt = parseNestLoopOption(optarg, c == 1005);

					if (last_nest_loop)
						last_nest_loop->next = nlopt;
					else
						sqldb_nestloop_options = nlopt;
					last_nest_loop = nlopt;
				}
				break;
#endif	/* __PG2ARROW__ */
			case 1006:		/* --schema */
				if (dump_arrow_filename)
					Elog("--dump and --schema are exclusive");
				if (schema_arrow_filename)
					Elog("--schema was supplied twice");
				schema_arrow_filename = optarg;
				break;
			case 1007:		/* --schema-name */
				if (schema_arrow_tablename)
					Elog("--schema-name was supplied twice");
				schema_arrow_tablename = optarg;
				break;
			case 'S':		/* --stat */
				{
					if (stat_embedded_columns)
						Elog("--stat option was supplied twice");
					if (optarg)
						stat_embedded_columns = optarg;
					else
						stat_embedded_columns = "*";
				}
				break;
			case 9999:		/* --help */
			default:
				usage();
				break;
		}
	}

	if (optind + 1 == argc)
	{
		if (sqldb_database)
			Elog("database name was supplied twice");
		sqldb_database = argv[optind];
	}
	else if (optind + 2 == argc)
	{
		if (sqldb_database)
			Elog("database name was supplied twice");
		if (sqldb_username)
			Elog("database user was specified twice");
		sqldb_database = argv[optind];
		sqldb_username = argv[optind + 1];
	}
	else if (optind != argc)
		Elog("too much command line arguments");
	/* password prompt, if -W option is supplied */
	if (password_prompt > 0)
	{
		assert(!sqldb_password);
		sqldb_password = pstrdup(getpass("Password: "));
	}
	/* --dump is exclusive other SQL options */
	if (dump_arrow_filename)
	{
		if (sqldb_command || output_filename || append_filename)
			Elog("--dump option is exclusive with -c, -t, -o and --append");
		return;
	}
	/* --schema is exclusive other SQL options */
	if (schema_arrow_filename)
	{
		if (sqldb_command || output_filename || append_filename)
			Elog("--schema option is exclusive with -c, -t, -o and --append");
		return;
	}
	else if (schema_arrow_tablename)
	{
		Elog("--schema-name option must be used with --schema");
	}
	/* SQL command options */
	if (!simple_table_name &&!sqldb_command)
		Elog("Neither -c nor -t options are supplied");
	assert((simple_table_name && !sqldb_command) ||
		   (!simple_table_name && sqldb_command));
	if (parallel_dist_keys)
	{
		if (simple_table_name)
			Elog("Unable to use -k|--parallel-keys with -t|--table, adopt -c|--command instead");
		if (strstr(sqldb_command, "$(PARALLEL_KEY)") == NULL)
			Elog("SQL command must contain $(PARALLEL_KEY) token");

		num_worker_threads = parseParallelDistKeys(parallel_dist_keys, ",");
	}
	else if (sqldb_command)
	{
		assert(!simple_table_name);
		if (num_worker_threads == 0 || num_worker_threads == 1)
		{
			if (strstr(sqldb_command, "$(WORKER_ID)") != NULL ||
				strstr(sqldb_command, "$(N_WORKERS)") != NULL ||
				strstr(sqldb_command, "$(PARALLEL_KEY)") != NULL)
				Elog("Non-parallel SQL command should not use the reserved keywords: $(WORKER_ID), $(N_WORKERS) and $(PARALLEL_KEY)");
			num_worker_threads = 1;
		}
		else
		{
			assert(num_worker_threads > 1);
			if (strstr(sqldb_command, "$(WORKER_ID)") == NULL ||
				strstr(sqldb_command, "$(N_WORKERS)") == NULL)
				Elog("The custom SQL command has to contains $(WORKER_ID) and $(N_WORKERS) to avoid data duplications.\n"
					 "example) SELECT * FROM my_table WHERE my_id %% $(N_WORKERS) = $(WORKER_ID)");
			if (strstr(sqldb_command, "$(PARALLEL_KEY)") != NULL)
				Elog("-n|--num-workers does not support $(PARALLEL_KEY) token.");
		}
	}
	else
	{
		assert(simple_table_name);
		if (num_worker_threads == 0)
		{
			/* -t TABLE without -n option */
			num_worker_threads = 1;
		}
#ifndef __PG2ARROW__
		else if (num_worker_threads > 1)
		{
			Elog("-t|--table with parallel execution is not supported");
		}
#endif
	}
	/* default segment size */
	if (batch_segment_sz == 0)
		batch_segment_sz = (1UL << 28);		/* 256MB in default */
}

/*
 * sqldb_command_apply_worker_id
 */
const char *
sqldb_command_apply_worker_id(const char *command, int worker_id)
{
	const char *src = command;
	size_t	len = strlen(command) + 100;
	size_t	off = 0;
	char   *buf = palloc(len);
	int		c;

	while ((c = *src++) != '\0')
	{
		if (c == '$')
		{
			char	temp[100];
			const char *tok = NULL;

			if (strncmp(src, "(WORKER_ID)", 11) == 0)
			{
				sprintf(temp, "%d", worker_id);
				src += 11;
				tok = temp;
			}
			else if (strncmp(src, "(N_WORKERS)", 11) == 0)
			{
				sprintf(temp, "%d", num_worker_threads);
				src += 11;
				tok = temp;
			}
			else if (strncmp(src, "(PARALLEL_KEY)", 14) == 0)
			{
				if (worker_dist_keys && worker_dist_keys[worker_id])
				{
					src += 14;
					tok = worker_dist_keys[worker_id];
				}
			}

			if (tok)
			{
				size_t	sz = strlen(tok);

				if (off + sz + 1 >= len)
				{
					len = (off + sz) + len;
					buf = repalloc(buf, len);
				}
				strcpy(buf + off, tok);
				off += sz;
				continue;
			}
		}
		if (off + 1 == len)
		{
			len += len;
			buf = repalloc(buf, len);
		}
		buf[off++] = c;
	}
	buf[off++] = '\0';

	return buf;
}

/*
 * sql_table_merge_one_row
 */
static void
mergeArrowChunkOneRow(SQLtable *dst_table,
					  SQLtable *src_table, size_t src_index)
{
	size_t	usage = 0;

	for (int j=0; j < src_table->nfields; j++)
	{
		usage += sql_field_move_value(&dst_table->columns[j],
									  &src_table->columns[j], src_index);
	}
	dst_table->nitems++;
	dst_table->usage = usage;
}

/*
 * shows_record_batch_progress
 */
static void
shows_record_batch_progress(const ArrowBlock *block,
							int rb_index, size_t nitems,
							int worker_id)
{
	time_t		tv = time(NULL);
	struct tm	tm;
	char		namebuf[100];

	if (num_worker_threads == 1)
		namebuf[0] = '\0';
	else
		sprintf(namebuf, " by worker:%d", worker_id);

	localtime_r(&tv, &tm);
	printf("%04d-%02d-%02d %02d:%02d:%02d "
		   "RecordBatch[%d]: "
		   "offset=%lu length=%lu (meta=%u, body=%lu) nitems=%zu%s\n",
		   tm.tm_year + 1900,
		   tm.tm_mon + 1,
		   tm.tm_mday,
		   tm.tm_hour,
		   tm.tm_min,
		   tm.tm_sec,
		   rb_index,
		   block->offset,
		   block->metaDataLength + block->bodyLength,
		   block->metaDataLength,
		   block->bodyLength,
		   nitems,
		   namebuf);
}

/*
 * execute_sql2arrow
 */
static void
sql2arrow_common(void *sqldb_conn, uint32_t worker_id)
{
	SQLtable   *main_table = worker_tables[0];
	SQLtable   *data_table = worker_tables[worker_id];
	ArrowBlock	__block;
	int			__rb_index;

	/* fetch results and write record batches */
	while (sqldb_fetch_results(sqldb_conn, data_table))
	{
		if (data_table->usage >= batch_segment_sz)
		{
			__rb_index = writeArrowRecordBatchMT(main_table,
												 data_table,
												 &main_table_mutex,
												 &__block);
			if (shows_progress)
				shows_record_batch_progress(&__block,
											__rb_index,
											data_table->nitems,
											worker_id);
			sql_table_clear(data_table);
		}
	}
	/* wait and merge results */
	for (uint32_t k=1; (worker_id & k) == 0; k <<= 1)
	{
		uint32_t	buddy = (worker_id | k);
		SQLtable   *buddy_table;

		if (buddy >= num_worker_threads)
			break;
		if ((errno = pthread_join(worker_threads[buddy], NULL)) != 0)
			Elog("failed on pthread_join[%u]: %m", buddy);

		buddy_table = worker_tables[buddy];
		for (size_t i=0; i < buddy_table->nitems; i++)
		{
			/* merge one row */
			mergeArrowChunkOneRow(data_table, buddy_table, i);
			/* write out buffer */
			if (data_table->usage >= batch_segment_sz)
			{
				__rb_index = writeArrowRecordBatchMT(main_table,
													 data_table,
													 &main_table_mutex,
													 &__block);
				if (shows_progress)
					shows_record_batch_progress(&__block,
												__rb_index,
												data_table->nitems,
												worker_id);
				sql_table_clear(data_table);
			}
		}
		if (shows_progress && buddy_table->nitems > 0)
			printf("worker:%u merged pending results by worker:%u\n",
				   worker_id, buddy);
	}
}

/*
 * worker_main
 */
static void *
worker_main(void *__worker_id)
{
	uintptr_t	worker_id = (uintptr_t)__worker_id;
	SQLtable   *main_table;
	SQLtable   *data_table;
	void	   *sqldb_conn;
	const char *worker_command;

	/* wait for the initial setup */
	pthread_mutex_lock(&worker_setup_mutex);
	while (!worker_setup_done)
	{
		pthread_cond_wait(&worker_setup_cond,
						  &worker_setup_mutex);
	}
	pthread_mutex_unlock(&worker_setup_mutex);

	/*
	 * OK, now worker:0 already has started.
	 */
	main_table = worker_tables[0];
	sqldb_conn = sqldb_server_connect(sqldb_hostname,
									  sqldb_port_num,
									  sqldb_username,
									  sqldb_password,
									  sqldb_database,
									  sqldb_session_configs,
									  sqldb_nestloop_options);
	worker_command = sqldb_command_apply_worker_id(sqldb_command, worker_id);
	if (shows_progress)
		printf("worker:%lu SQL=[%s]\n", worker_id, worker_command);
	data_table = sqldb_begin_query(sqldb_conn,
								   worker_command,
								   NULL,
								   main_table->sql_dict_list);
	if (!data_table)
		Elog("Empty results by the query: %s", worker_command);
	data_table->segment_sz = batch_segment_sz;
	/* enables embedded min/max statistics, if any */
	enable_embedded_stats(data_table);
	/* check compatibility */
	if (!IsSQLtableCompatible(main_table, data_table))
		Elog("Schema definition by the query in worker:%lu is not compatible: %s",
			 worker_id, worker_command);
	worker_tables[worker_id] = data_table;
	/* main loop to fetch and write results */
	sql2arrow_common(sqldb_conn, worker_id);
	/* close the connection */
	sqldb_close_connection(sqldb_conn);

	if (shows_progress)
		printf("worker:%lu terminated\n", worker_id);

	return NULL;
}

/*
 * Entrypoint of pg2arrow / mysql2arrow
 */
int main(int argc, char * const argv[])
{
	int				append_fdesc = -1;
	ArrowFileInfo	af_info;
	void		   *sqldb_conn;
	const char	   *main_command;
	SQLtable	   *table;
	ArrowKeyValue  *kv;
	SQLdictionary  *sql_dict_list = NULL;
	time_t			tv1 = time(NULL);

	parse_options(argc, argv);

	/* special case if --dump=FILENAME */
	if (dump_arrow_filename)
		return dumpArrowFile(dump_arrow_filename);
	/* special case if --schema=FILENAME */
	if (schema_arrow_filename)
		return schemaArrowFile(schema_arrow_filename,
							   schema_arrow_tablename);
	/* setup workers */
	assert(num_worker_threads > 0);
	worker_threads = palloc0(sizeof(pthread_t)  * num_worker_threads);
	worker_tables  = palloc0(sizeof(SQLtable *) * num_worker_threads);
	for (uintptr_t i = 1; i < num_worker_threads; i++)
	{
		if ((errno = pthread_create(&worker_threads[i],
									NULL,
									worker_main,
									(void *)i)) != 0)
			Elog("failed on pthread_create: %m");
	}
	/* open connection */
	sqldb_conn = sqldb_server_connect(sqldb_hostname,
									  sqldb_port_num,
									  sqldb_username,
									  sqldb_password,
									  sqldb_database,
									  sqldb_session_configs,
									  sqldb_nestloop_options);
	/* read the original arrow file, if --append mode */
	if (append_filename)
	{
		append_fdesc = open(append_filename, O_RDWR, 0644);
		if (append_fdesc < 0)
			Elog("failed on open('%s'): %m", append_filename);
		readArrowFileDesc(append_fdesc, &af_info);
		sql_dict_list = loadArrowDictionaryBatches(append_fdesc, &af_info);
	}
	/* build simple table-scan query command */
	if (simple_table_name)
	{
		assert(!sqldb_command);
		sqldb_command = sqldb_build_simple_command(sqldb_conn,
												   simple_table_name,
												   num_worker_threads,
												   batch_segment_sz);
		if (!sqldb_command)
			Elog("out of memory");
	}
	/* begin SQL command execution */
	main_command = sqldb_command_apply_worker_id(sqldb_command, 0);
	if (shows_progress)
		printf("worker:0 SQL=[%s]\n", main_command);
	table = sqldb_begin_query(sqldb_conn,
							  main_command,
							  append_filename ? &af_info : NULL,
							  sql_dict_list);
	if (!table)
		Elog("Empty results by the query: %s", sqldb_command);
	table->segment_sz = batch_segment_sz;
	/* enables embedded min/max statistics, if any */
	enable_embedded_stats(table);

	/* save the SQL command as custom metadata */
	kv = palloc0(sizeof(ArrowKeyValue));
	initArrowNode(kv, KeyValue);
	kv->key = "sql_command";
	kv->_key_len = 11;
	kv->value = sqldb_command;
	kv->_value_len = strlen(sqldb_command);
	table->customMetadata = kv;
	table->numCustomMetadata = 1;

	/* open & setup result file */
	if (!append_filename)
		setup_output_file(table, output_filename);
	else
	{
		table->fdesc = append_fdesc;
		table->filename = append_filename;
		setup_append_file(table, &af_info);
	}
	/* write out dictionary batch, if any */
	writeArrowDictionaryBatches(table);
	/* the primary SQLtable become visible to other workers */
	pthread_mutex_lock(&worker_setup_mutex);
	worker_tables[0] = table;
	worker_setup_done = true;
	pthread_cond_broadcast(&worker_setup_cond);
	pthread_mutex_unlock(&worker_setup_mutex);
	/* main loop to fetch and write results*/
	sql2arrow_common(sqldb_conn, 0);
	if (table->nitems > 0)
	{
		ArrowBlock	__block;
		int			__rb_index;

		__rb_index = writeArrowRecordBatch(table, &__block);
		if (shows_progress)
			shows_record_batch_progress(&__block,
										__rb_index,
										table->nitems,
										0);
		sql_table_clear(table);
	}
	/* write out footer portion */
	writeArrowFooter(table);

	/* cleanup */
	sqldb_close_connection(sqldb_conn);
	close(table->fdesc);

	if (shows_progress)
	{
		time_t	elapsed = (time(NULL) - tv1);

		if (elapsed > 2 * 86400)	/* > 2days */
			printf("Total elapsed time: %ld days %02ld:%02ld:%02ld\n",
				   elapsed / 86400,
				   (elapsed % 86400) / 3600,
				   (elapsed % 3600) / 60,
				   (elapsed % 60));
		else
			printf("Total elapsed time: %02ld:%02ld:%02ld\n",
				   (elapsed % 86400) / 3600,
				   (elapsed % 3600) / 60,
				   (elapsed % 60));
	}
	return 0;
}

/*
 * This hash function was written by Bob Jenkins
 * (bob_jenkins@burtleburtle.net), and superficially adapted
 * for PostgreSQL by Neil Conway. For more information on this
 * hash function, see http://burtleburtle.net/bob/hash/doobs.html,
 * or Bob's article in Dr. Dobb's Journal, Sept. 1997.
 *
 * In the current code, we have adopted Bob's 2006 update of his hash
 * function to fetch the data a word at a time when it is suitably aligned.
 * This makes for a useful speedup, at the cost of having to maintain
 * four code paths (aligned vs unaligned, and little-endian vs big-endian).
 * It also uses two separate mixing functions mix() and final(), instead
 * of a slower multi-purpose function.
 */

/* Get a bit mask of the bits set in non-uint32 aligned addresses */
#define UINT32_ALIGN_MASK (sizeof(uint32_t) - 1)

/* Rotate a uint32 value left by k bits - note multiple evaluation! */
#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

/*----------
 * mix -- mix 3 32-bit values reversibly.
 *
 * This is reversible, so any information in (a,b,c) before mix() is
 * still in (a,b,c) after mix().
 *
 * If four pairs of (a,b,c) inputs are run through mix(), or through
 * mix() in reverse, there are at least 32 bits of the output that
 * are sometimes the same for one pair and different for another pair.
 * This was tested for:
 * * pairs that differed by one bit, by two bits, in any combination
 *	 of top bits of (a,b,c), or in any combination of bottom bits of
 *	 (a,b,c).
 * * "differ" is defined as +, -, ^, or ~^.  For + and -, I transformed
 *	 the output delta to a Gray code (a^(a>>1)) so a string of 1's (as
 *	 is commonly produced by subtraction) look like a single 1-bit
 *	 difference.
 * * the base values were pseudorandom, all zero but one bit set, or
 *	 all zero plus a counter that starts at zero.
 *
 * This does not achieve avalanche.  There are input bits of (a,b,c)
 * that fail to affect some output bits of (a,b,c), especially of a.  The
 * most thoroughly mixed value is c, but it doesn't really even achieve
 * avalanche in c.
 *
 * This allows some parallelism.  Read-after-writes are good at doubling
 * the number of bits affected, so the goal of mixing pulls in the opposite
 * direction from the goal of parallelism.  I did what I could.  Rotates
 * seem to cost as much as shifts on every machine I could lay my hands on,
 * and rotates are much kinder to the top and bottom bits, so I used rotates.
 *----------
 */
#define mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);	c += b; \
  b -= a;  b ^= rot(a, 6);	a += c; \
  c -= b;  c ^= rot(b, 8);	b += a; \
  a -= c;  a ^= rot(c,16);	c += b; \
  b -= a;  b ^= rot(a,19);	a += c; \
  c -= b;  c ^= rot(b, 4);	b += a; \
}

/*----------
 * final -- final mixing of 3 32-bit values (a,b,c) into c
 *
 * Pairs of (a,b,c) values differing in only a few bits will usually
 * produce values of c that look totally different.  This was tested for
 * * pairs that differed by one bit, by two bits, in any combination
 *	 of top bits of (a,b,c), or in any combination of bottom bits of
 *	 (a,b,c).
 * * "differ" is defined as +, -, ^, or ~^.  For + and -, I transformed
 *	 the output delta to a Gray code (a^(a>>1)) so a string of 1's (as
 *	 is commonly produced by subtraction) look like a single 1-bit
 *	 difference.
 * * the base values were pseudorandom, all zero but one bit set, or
 *	 all zero plus a counter that starts at zero.
 *
 * The use of separate functions for mix() and final() allow for a
 * substantial performance increase since final() does not need to
 * do well in reverse, but is does need to affect all output bits.
 * mix(), on the other hand, does not need to affect all output
 * bits (affecting 32 bits is enough).  The original hash function had
 * a single mixing operation that had to satisfy both sets of requirements
 * and was slower as a result.
 *----------
 */
#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c, 4); \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}

/*
 * hash_any() -- hash a variable-length key into a 32-bit value
 *		k		: the key (the unaligned variable-length array of bytes)
 *		len		: the length of the key, counting by bytes
 *
 * Returns a uint32 value.  Every bit of the key affects every bit of
 * the return value.  Every 1-bit and 2-bit delta achieves avalanche.
 * About 6*len+35 instructions. The best hash table sizes are powers
 * of 2.  There is no need to do mod a prime (mod is sooo slow!).
 * If you need less than 32 bits, use a bitmask.
 *
 * This procedure must never throw elog(ERROR); the ResourceOwner code
 * relies on this not to fail.
 *
 * Note: we could easily change this function to return a 64-bit hash value
 * by using the final values of both b and c.  b is perhaps a little less
 * well mixed than c, however.
 */
uint32_t
hash_any(const unsigned char *k, int keylen)
{
	register uint32_t a, b, c, len;

	/* Set up the internal state */
	len = keylen;
	a = b = c = 0x9e3779b9 + len + 3923095;

	/* If the source pointer is word-aligned, we use word-wide fetches */
	if (((uintptr_t) k & UINT32_ALIGN_MASK) == 0)
	{
		/* Code path for aligned source data */
		register const uint32_t *ka = (const uint32_t *) k;

		/* handle most of the key */
		while (len >= 12)
		{
			a += ka[0];
			b += ka[1];
			c += ka[2];
			mix(a, b, c);
			ka += 3;
			len -= 12;
		}

		/* handle the last 11 bytes */
		k = (const unsigned char *) ka;
#ifdef WORDS_BIGENDIAN
		switch (len)
		{
			case 11:
				c += ((uint32_t) k[10] << 8);
				/* fall through */
			case 10:
				c += ((uint32_t) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32_t) k[8] << 24);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += ka[1];
				a += ka[0];
				break;
			case 7:
				b += ((uint32_t) k[6] << 8);
				/* fall through */
			case 6:
				b += ((uint32_t) k[5] << 16);
				/* fall through */
			case 5:
				b += ((uint32_t) k[4] << 24);
				/* fall through */
			case 4:
				a += ka[0];
				break;
			case 3:
				a += ((uint32_t) k[2] << 8);
				/* fall through */
			case 2:
				a += ((uint32_t) k[1] << 16);
				/* fall through */
			case 1:
				a += ((uint32_t) k[0] << 24);
				/* case 0: nothing left to add */
		}
#else							/* !WORDS_BIGENDIAN */
		switch (len)
		{
			case 11:
				c += ((uint32_t) k[10] << 24);
				/* fall through */
			case 10:
				c += ((uint32_t) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32_t) k[8] << 8);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += ka[1];
				a += ka[0];
				break;
			case 7:
				b += ((uint32_t) k[6] << 16);
				/* fall through */
			case 6:
				b += ((uint32_t) k[5] << 8);
				/* fall through */
			case 5:
				b += k[4];
				/* fall through */
			case 4:
				a += ka[0];
				break;
			case 3:
				a += ((uint32_t) k[2] << 16);
				/* fall through */
			case 2:
				a += ((uint32_t) k[1] << 8);
				/* fall through */
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
#endif							/* WORDS_BIGENDIAN */
	}
	else
	{
		/* Code path for non-aligned source data */

		/* handle most of the key */
		while (len >= 12)
		{
#ifdef WORDS_BIGENDIAN
			a += (k[3] +
				  ((uint32_t) k[2] << 8) +
				  ((uint32_t) k[1] << 16) +
				  ((uint32_t) k[0] << 24));
			b += (k[7] +
				  ((uint32_t) k[6] << 8) +
				  ((uint32_t) k[5] << 16) +
				  ((uint32_t) k[4] << 24));
			c += (k[11] +
				  ((uint32_t) k[10] << 8) +
				  ((uint32_t) k[9] << 16) +
				  ((uint32_t) k[8] << 24));
#else					/* !WORDS_BIGENDIAN */
			a += (k[0] +
				  ((uint32_t) k[1] << 8) +
				  ((uint32_t) k[2] << 16) +
				  ((uint32_t) k[3] << 24));
			b += (k[4] +
				  ((uint32_t) k[5] << 8) +
				  ((uint32_t) k[6] << 16) +
				  ((uint32_t) k[7] << 24));
			c += (k[8] +
				  ((uint32_t) k[9] << 8) +
				  ((uint32_t) k[10] << 16) +
				  ((uint32_t) k[11] << 24));
#endif					/* WORDS_BIGENDIAN */
			mix(a, b, c);
			k += 12;
			len -= 12;
		}

		/* handle the last 11 bytes */
#ifdef WORDS_BIGENDIAN
		switch (len)
		{
			case 11:
				c += ((uint32_t) k[10] << 8);
				/* fall through */
			case 10:
				c += ((uint32_t) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32_t) k[8] << 24);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += k[7];
				/* fall through */
			case 7:
				b += ((uint32_t) k[6] << 8);
				/* fall through */
			case 6:
				b += ((uint32_t) k[5] << 16);
				/* fall through */
			case 5:
				b += ((uint32_t) k[4] << 24);
				/* fall through */
			case 4:
				a += k[3];
				/* fall through */
			case 3:
				a += ((uint32_t) k[2] << 8);
				/* fall through */
			case 2:
				a += ((uint32_t) k[1] << 16);
				/* fall through */
			case 1:
				a += ((uint32_t) k[0] << 24);
				/* case 0: nothing left to add */
		}
#else							/* !WORDS_BIGENDIAN */
		switch (len)
		{
			case 11:
				c += ((uint32_t) k[10] << 24);
				/* fall through */
			case 10:
				c += ((uint32_t) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32_t) k[8] << 8);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += ((uint32_t) k[7] << 24);
				/* fall through */
			case 7:
				b += ((uint32_t) k[6] << 16);
				/* fall through */
			case 6:
				b += ((uint32_t) k[5] << 8);
				/* fall through */
			case 5:
				b += k[4];
				/* fall through */
			case 4:
				a += ((uint32_t) k[3] << 24);
				/* fall through */
			case 3:
				a += ((uint32_t) k[2] << 16);
				/* fall through */
			case 2:
				a += ((uint32_t) k[1] << 8);
				/* fall through */
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
#endif							/* WORDS_BIGENDIAN */
	}

	final(a, b, c);

	/* report the result */
	return c;
}

/*
 * misc functions
 */
void *
palloc(size_t sz)
{
	void   *ptr = malloc(sz);

	if (!ptr)
		Elog("out of memory");
	return ptr;
}

void *
palloc0(size_t sz)
{
	void   *ptr = malloc(sz);

	if (!ptr)
		Elog("out of memory");
	memset(ptr, 0, sz);
	return ptr;
}

char *
pstrdup(const char *str)
{
	char   *ptr = strdup(str);

	if (!ptr)
		Elog("out of memory");
	return ptr;
}

void *
repalloc(void *old, size_t sz)
{
	char   *ptr = realloc(old, sz);

	if (!ptr)
		Elog("out of memory");
	return ptr;
}

void
pfree(void *ptr)
{
	free(ptr);
}





