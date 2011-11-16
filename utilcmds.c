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
#include "tcop/utility.h"
#include "pg_rapid.h"

/*
 * Saved hook entries
 */
static ProcessUtility_hook_type next_process_utility_hook = NULL;

/*
 * pgrapid_create_rowid_index
 *
 * create "<base_schema>.<base_rel>.(<column>.)idx" index of pg_rapid
 * schema to find-up a tuple that contains a particular rowid.
 */
static void
pgrapid_create_rowid_index(Relation base_rel, const char *attname,
						   Relation behind_rel, AttrNumber indexed)
{}

/*
 * pgrapid_create_usemap_store
 *
 * create "<base_schema>.<base_rel>.usemap" table of pg_rapid schema
 * that has "rowid(int8)" and "usemap(VarBit)" to track which rowid
 * has been in use.
 */
static void
pgrapid_create_usemap_store(Oid namespaceId, Relation base_rel)
{}

/*
 * pgrapid_create_column_store
 *
 * create "<base_schema>.<base_rel>.<column>.col" table of pg_rapid
 * schema that has "rowid(int8)", "nulls(VarBit)" and "values(bytes)"
 * to store the values of original columns. The maximum number of
 * values being stored in a tuple depends on length of unit data.
 */
static void
pgrapid_create_column_store(Oid namespaceId, Relation base_rel,
							const char *attname)
{}

/*
 * pgrapid_create_usemap_seq
 *
 * create "<base_schema>.<base_rel>.seq" sequence of pg_rapid schema
 * that enables to generate unique number between 0 to 2^48-1 by
 * PGRAPID_USEMAP_UNITSZ.
 */
static void
pgrapid_create_usemap_seq(Oid namespaceId, base_rel)
{}

static void
pgrapid_process_post_create(RangeVar *base_range)
{
	Relation	base_rel;
	Oid			namespaceId;
	AttrNumber	attnum;
	Oid			save_userid;
	int			save_sec_context;

	/* Ensure the base relation being visible */
	CommandCounterIncrement();

	/*
	 * Ensure existence of the schema that shall stores all the
	 * corresponding stuff. If not found, create it anyway.
	 */
	namespaceId = get_namespace_oid(PGRAPID_SCHEMA_NAME, true);
	if (!OidIsValid(namespaceId))
	{
		namespaceId = NamespaceCreate(PGBOOST_SCHEMA_NAME,
									  BOOTSTRAP_SUPERUSERID);
		CommandCounterIncrement();
	}

	/*
	 * Open the base relation; exclusive lock should be already
	 * acquired, so we use NoLock instead.
	 */
	base_rel = heap_openrv(base_range, NoLock);

	/* switch current credential of database users */
	GetUserIdAndSecContext(&save_userid, &save_sec_context);
	SetUserIdAndSecContext(BOOTSTRAP_SUPERUSERID, save_sec_context);

	/* create pg_rapid.<base_schema>_<base_rel>.map */
	pgrapid_create_usemap_store(namespaceId, base_rel);

	/* create pg_rapid.<base_schema>_<base_rel>_<column> */
	for (attnum = 0;
		 attnum < RelationGetNumberOfAttributes(base_rel);
		 attnum++)
	{
		const char *attname =
			NameStr(RelationGetDescr(base_rel)->attrs[attnum]->attname);

		pgrapid_create_column_store(namespaceId, base_rel, attname);
	}

	/* create pg_rapid.<base_schema>_<base_rel>.seq */
	pgrapid_create_usemap_seq(namespaceId, base_rel);


	/* restore security setting and close the base relation */
	SetUserIdAndSecContext(save_userid, save_sec_context);

	heap_close(base_rel, NoLock);
}

static void
pgrapid_process_post_alter_schema(RangeVar *base_range)
{}

static void
pgrapid_process_post_alter_rename(RangeVar *base_range)
{}

static void
pgrapid_process_post_alter_owner(RangeVar *base_range)
{}

/*
 * pgrapid_process_utility_command
 *
 * Entrypoint of the ProcessUtility hook; that handles post DDL operations.
 */
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
		CreateForeignTableStmt *cfts = (CreateForeignTableStmt *)stmt;
		ForeignDataWrapper	   *fdw;
		FdwRoutine			   *fdwfn;
		Oid		fservId;

		fservId = get_foreign_server_oid(cfts->servername, false);
		fdw = GetForeignDataWrapper(fservId);
		
		pgrapid_fdw_handler_is_called = false;
		fdwfn = GetFdwRoutine(fdw);
		if (pgrapid_fdw_handler_is_called)
			pgrapid_process_post_create(cfts->base.relation);
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
