from pprint import pprint
import sys
import psycopg2
import pystrom

conn = psycopg2.connect("host=localhost dbname=postgres")
curr = conn.cursor()
curr.execute("select gstore_export_ipchandle('ft')::bytea")
row = curr.fetchone()
conn.close()

X = pystrom.my_test(row[0],['x','y','z'])
Y = pystrom.my_test(row[0],['id'])

print(X)
print(Y)
