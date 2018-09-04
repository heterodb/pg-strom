import psycopg2
import pystrom

conn = psycopg2.connect("host=localhost dbname=postgres")
curr = conn.cursor()
curr.execute("select gstore_export_ipchandle('ft')::bytea")
row = curr.fetchone()
conn.close()

X = pystrom.ipc_import(row[0],['x','y','z'])
Y = pystrom.ipc_import(row[0],['id'])

print(X)
print(Y)
