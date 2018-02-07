#include <libpq-fe.h>
#include <signal.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
	cudaError_t	rc;
	size_t		length = (64U << 20);	/* 64MB */
	size_t		offset = 12345678;		/* 12MB */
	void	   *lbuffer;
	void	   *hbuffer;
	void	   *dbuffer;
	char	   *sql;
	cudaIpcMemHandle_t ipc_mhandle;
	int			device_nr;
	int			i, n;
	unsigned int loid;
	long		lo_size;
	PGconn	   *conn;
	PGresult   *res;
	char	   *datum;
	unsigned char *bytea_hex;
	size_t		bytea_sz;
	char		buf[2048];

	if (argc != 1)
	{
		fprintf(stderr, "usage: %s\n", basename(argv[0]));
		return 1;
	}

	rc = cudaGetDevice(&device_nr);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "failed on cudaGetDevice: %s",
				cudaGetErrorString(rc));
		return 1;
	}

	rc = cudaHostAlloc(&hbuffer, length,
					   cudaHostAllocDefault);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "failed on cudaHostAlloc: %s",
				cudaGetErrorString(rc));
		return 1;
	}

	rc = cudaMalloc(&dbuffer, offset + length);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "failed on cudaMalloc: %s",
				cudaGetErrorString(rc));
		return 1;
	}

	rc = cudaIpcGetMemHandle(&ipc_mhandle, dbuffer);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "failed on cudaIpcGetMemHandle: %s",
				cudaGetErrorString(rc));
		return 1;
	}

	/*
	 * Connect to PostgreSQL; note that all the connection info are passed
	 * by environment variables - PGDATABASE, PGHOST, PGPORT, PGUSER
	 */
	conn = PQconnectdb("");
	if (PQstatus(conn) != CONNECTION_OK)
	{
		fprintf(stderr, "%s\n", PQerrorMessage(conn));
		return 1;
	}

	/*
	 * Test for lo_export_gpu
	 */
	snprintf(buf, sizeof(buf),
			 "SELECT lo_from_bytea(0, string_agg(md5(x::text),'|')::bytea)\n"
			 "  FROM generate_series(1,100000) x");
	res = PQexec(conn, buf);
	if (!res || PQresultStatus(res) != PGRES_TUPLES_OK)
	{
		fprintf(stderr, "SQL failed: %s\n%s\n", PQerrorMessage(conn), buf);
		return 1;
	}
	loid = atol(PQgetvalue(res, 0, 0));
    PQclear(res);

	bytea_hex = PQescapeByteaConn(conn,
								  (const unsigned char *)(&ipc_mhandle),
								  sizeof(ipc_mhandle),
								  &bytea_sz);
	snprintf(buf, sizeof(buf),
			 "SELECT lo_export_gpu(%u,%d,E'\\%s'::bytea,%lu,%lu)",
			 loid, device_nr, bytea_hex, offset, length);
	res = PQexecParams(conn, buf, 0, NULL, NULL, NULL, NULL, 1);
	res = PQexec(conn, buf);
	if (!res || PQresultStatus(res) != PGRES_TUPLES_OK)
	{
		fprintf(stderr, "SQL failed: %s\n%s\n", PQerrorMessage(conn), buf);
		return 1;
	}
	lo_size = atol(PQgetvalue(res, 0, 0));
	PQclear(res);

	snprintf(buf, sizeof(buf), "SELECT lo_get(%u)", loid);
	res = PQexecParams(conn, buf, 0, NULL, NULL, NULL, NULL, 1);
	if (!res || PQresultStatus(res) != PGRES_TUPLES_OK )
	{
		fprintf(stderr, "SQL failed: %s\n%s\n", PQerrorMessage(conn), buf);
		return 1;
	}
	lbuffer = PQgetvalue(res, 0, 0);
	if (lo_size != PQgetlength(res, 0, 0))
	{
		fprintf(stderr, "lo_get() didn't return entire largeobject\n");
		return 1;
	}

	rc = cudaMemcpy(hbuffer, (char *)dbuffer + offset, length,
					cudaMemcpyDeviceToHost);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "failed on cudaMemcpy(DtoH): %s",
				cudaGetErrorName(rc),
				cudaGetErrorString(rc));
		return 1;
	}

	printf("lo_export_gpu\t%s\n",
		   memcmp(lbuffer, hbuffer, lo_size) == 0 ? "OK" : "FAIL");
	PQclear(res);

	snprintf(buf, sizeof(buf), "SELECT lo_unlink(%u)", loid);
	res = PQexec(conn, buf);
	if (!res || PQresultStatus(res) != PGRES_TUPLES_OK )
	{
		fprintf(stderr, "SQL failed: %s\n%s\n", PQerrorMessage(conn), buf);
		return 1;
	}

	/*
	 * Test for lo_import_gpu
	 */
	srand((unsigned int)time(NULL));
	for (i=0, n=length/sizeof(int); i < n; i++)
		((int *)hbuffer)[i] = rand();

	rc = cudaMemcpy((char *)dbuffer + offset, hbuffer, length,
					cudaMemcpyHostToDevice);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "failed on cudaMemcpy(HtoD): %s",
				cudaGetErrorName(rc),
				cudaGetErrorString(rc));
		return 1;
	}

	snprintf(buf, sizeof(buf),
			 "SELECT lo_import_gpu(%d,E'\\%s'::bytea,%lu,%lu)",
			 device_nr, bytea_hex, offset, length);
	res = PQexec(conn, buf);
	if (!res || PQresultStatus(res) != PGRES_TUPLES_OK)
	{
		fprintf(stderr, "SQL failed: %s\n%s\n", PQerrorMessage(conn), buf);
		return 1;
	}
	loid = atol(PQgetvalue(res, 0, 0));
	PQclear(res);

	bytea_hex = PQescapeByteaConn(conn,
								  (const unsigned char *)hbuffer,
								  length, &bytea_sz);
	sql = (char *)malloc(bytea_sz + 2048);
	if (!sql)
	{
		fprintf(stderr, "out of memory\n");
		return 1;
	}
	sprintf(sql, "SELECT lo_get(%u) = E'\\%s'::bytea", loid, bytea_hex);
	res = PQexec(conn, sql);
	if (!res || PQresultStatus(res) != PGRES_TUPLES_OK)
	{
		sql[64] = '\0';
		fprintf(stderr, "SQL failed: %s\n%s\n", PQerrorMessage(conn), sql);
		return 1;
	}
	datum = PQgetvalue(res, 0, 0);

	printf("lo_import_gpu\t%s\n",
		   strcmp(datum, "t") == 0 ? "OK" : "FAIL");
    PQclear(res);

	snprintf(buf, sizeof(buf), "SELECT lo_unlink(%u)", loid);
	res = PQexec(conn, buf);
	if (!res || PQresultStatus(res) != PGRES_TUPLES_OK )
	{
		fprintf(stderr, "SQL failed: %s\n%s\n", PQerrorMessage(conn), buf);
		return 1;
	}
	PQclear(res);

	PQfinish(conn);
}
