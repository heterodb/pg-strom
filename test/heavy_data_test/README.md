# Heavy Data Test

This is to confirm PG-Strom has no issues in case of heavy data processing.


## Load heavy data 

```bash
bash load.sh <target_dir_path> <size(GB)> <psql parameters> ...
```

size = the size to be loaded to PostgreSQL

Example: to generate 10GB data and load it to PostgreSQL.
```bash
bash load.sh /tmp 10 postgres
```

