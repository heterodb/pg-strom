#ifndef SQL2ARROW_H
#define SQL2ARROW_H
#include "arrow_ipc.h"
#include <limits.h>
#include <pthread.h>

typedef struct userConfigOption     userConfigOption;
struct userConfigOption
{
	userConfigOption *next;
	char		query[1];		/* SET xxx='xxx' command */
};

extern void *
sqldb_server_connect(const char *sqldb_hostname,
					 const char *sqldb_port_num,
					 const char *sqldb_username,
					 const char *sqldb_password,
					 const char *sqldb_database,
					 userConfigOption *session_config_list);
extern SQLtable *
sqldb_begin_query(void *sqldb_state,
				  const char *sqldb_command,
				  SQLdictionary *dictionary_list,
				  const char *flatten_composite_columns);
extern bool
sqldb_fetch_results(void *sqldb_state, SQLtable *table);

extern void
sqldb_close_connection(void *sqldb_state);

extern char *
sqldb_build_simple_command(void *sqldb_state,
						   const char *simple_table_name,
						   int num_worker_threads,
						   size_t batch_segment_sz);
/* misc functions */
extern void	   *palloc(size_t sz);
extern void	   *palloc0(size_t sz);
extern char	   *pstrdup(const char *str);
extern void	   *repalloc(void *ptr, size_t sz);
extern uint32_t	hash_any(const unsigned char *k, int keylen);

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
#endif	/* SQL2ARROW_H */
