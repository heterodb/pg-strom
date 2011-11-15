/*
 * utilcmds.c
 *
 * Routines to handle utility command queries
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"

/*
 * Saved hook entries
 */
static ProcessUtility_hook_type next_process_utility_hook = NULL;








static void
pgrapid_process_utility_command(Node *stmt,
								const char *queryString,
								ParamListInfo params,
								bool isTopLevel,
								DestReceiver *dest,
								char *completionTag)
{
	if (next_process_utility_hook)
		(*next_process_utility_hook)(stmt, queryString, params,
									 isTopLevel, dest, completionTag);
	else
		standard_ProcessUtility(stmt, queryString, params,
								isTopLevel, dest, completionTag);
	/*
	 * Do we need post ddl works?
	 */
	if (IsA(stmt, CreateForeignTableStmt))
	{

	}
	else if (IsA(stmt, AlterObjectSchemaStmt))
	{}
	else if (IsA(stmt, RenameStmt))
	{}
	else if (IsA(stmt, AlterOwnerStmt))
	{}
}

/*
 * pgrapid_utilcmds_init
 *
 * Registers ProcessUtility hook
 */
void
pgrapid_utilcmds_init(void)
{
	next_process_utility_hook = ProcessUtility_hook;
	ProcessUtility_hook = pgrapid_process_utility_command;
}
