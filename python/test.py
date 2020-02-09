import psycopg2
import cupy_strom

conn = psycopg2.connect("host=localhost dbname=postgres")
curr = conn.cursor()
curr.execute("select pgstrom.arrow_fdw_export_cupy('ft',ARRAY['id','id','id','id'])")
row = curr.fetchone()
conn.close()

X = cupy_strom.ipc_import(row[0])
print(X)
