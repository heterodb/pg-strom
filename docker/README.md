# PG-Strom Experimental Dockerfile Documentation

It is possible to run PG-Strom in a Docker container where CUDA is enabled. This is an **experimental** Dockerfile that you can try to run on your own system.

## System Requirements

PG-Strom requires a Linux host and a recent NVIDIA video card. There is an official hardware validation list located [here](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List), but it is not comprehensive and assumes that you are running PG-Strom on a bare-metal system rather than a VM or Docker container.

This Dockerfile was tested on a computer with the following specs:

* Dell Precision 3240 Compact
* 3.3 GHz Intel Xeon W 6-Core (10th Gen)
* NVIDIA Quadro RTX 3000 (6GB GDDR6)
* 64 GB of DDR4 RAM
* Rocky Linux 9.0 (RHEL9 Compatible)
* Linux NVIDIA Driver 520.61.05
* Linux Docker 20.10.21

**Note:** Running in Windows with Docker on WSL 2 Linux is NOT possible, as the Windows NVIDIA driver is lacking the necessary CUDA APIs for PG-Strom, therefore resulting in an error.

## Run the Published Experimental Image `murphye/pgstrom-postgis33-pg15:v1`

If you choose to not build your own image (see next section), you may choose to use a pre-built image on DockerHub. See the next sections for more information on the command line options used. This command will pull and run the container, provided that there are not any errors due to hardware or system configuration.

```
docker run --name postgis-db \
-d \
--privileged \
--gpus=all \
--shm-size=4096MB \
-e POSTGRES_USER=postgis \
-e POSTGRES_PASSWORD=password \
-e POSTGRES_DB=postgis \
-p 5432:5432 \
-v ${HOME}/postgis-data:/var/lib/postgresql/data \
-d murphye/pgstrom-postgis33-pg15:v1
```

Tail the logs and watch for any errors!
```
docker logs postgis-db -f
```

## Build the PostGIS + PG-Strom Docker Image

For PostgreSQL 15:
```
cd docker/pg15
docker build -t pgstrom-postgis33-pg15 -f Dockerfile.pgstrom-postgis33 .
```

Or, for PostgreSQL 13:
```
cd docker/pg13
docker build -t pgstrom-postgis33-pg13 -f Dockerfile.pgstrom-postgis33 .
```

The generated Docker image will be 2.67GB in size. When compressed, the size is approximately 1.2 GB per DockerHub.

```
docker images | grep pgstrom-postgis33
pgstrom-postgis33-pg15                    latest                    96a26ca3d494   4 minutes ago    2.67GB
```

## Run the PostGIS + PG-Strom Docker Container

This command will run the PostGIS + PG-Strom Docker Image with a volume and as detached. `--gpus=all` enables all virtualized GPUs for the container. `--shm-size` sets the System V shared memory size for the container, which is needs to be larger than default to accomodate PG-Strom. `--shm-size` is not the same as `shared_buffers` in `postgresql.conf` which is POSIX shared memory.

```
docker run --name postgis33-pg15-db \
-d \
--privileged \
--gpus=all \
--shm-size=4096MB \
-e POSTGRES_USER=postgis \
-e POSTGRES_PASSWORD=password \
-e POSTGRES_DB=postgis \
-p 5432:5432 \
-v ${HOME}/postgis33-pg15-data:/var/lib/postgresql/data \
-d pgstrom-postgis33-pg15
```

Tail the logs and watch for any errors!
```
docker logs postgis33-pg15-db -f
```

## Access the PostGIS + PG-Strom Docker Container

### Access the PostGIS + PG-Strom Docker Container's Shell

```
docker exec -it postgis33-pg15-db bash
```

### Access the PostGIS Database (After `docker exec -it postgis33-pg15-db bash`)
```
psql -U postgis
```

## Stop and Restart the PostGIS + PG-Strom Docker Container

Stop the PostGIS + PG-Strom Docker container.
```
docker stop postgis33-pg15-db
```

Start the PostGIS + PG-Strom Docker container.
```
docker start postgis33-pg15-db
```

## Clean Up the PostGIS + PG-Strom Docker Image

Stop the PostGIS + PG-Strom Docker container and remove the image.
```
docker stop postgis33-pg15-db
docker rm postgis33-pg15-db
```
Remove the data directory from the PostGIS + PG-Strom Docker volume.
```
sudo rm -r ${HOME}/postgis33-pg15-data
```

## Run and Verify

### Setup Test Table

First, create a very simple table containing a point.

```
CREATE TABLE testtable(test_id serial primary key, description text, geom geometry(POINT));
INSERT INTO testtable(description, geom) VALUES('some place', ST_MakePoint(15, 15));
SELECT * FROM testtable;
```

Next, run an `EXPLAIN SELECT` to verify that the query is planned to be run on the GPU. Notice the `Custom Scan (GpuScan)`.

```
EXPLAIN SELECT * FROM testtable WHERE st_contains('polygon ((11 10,30 10,30 20,10 20,11 10))', geom);

Custom Scan (GpuScan) on testtable  (cost=5348.13..5348.13 rows=1 width=68)
"  GPU Filter: st_contains('01030000000100000005000000000000000000264000000000000024400000000000003E4000000000000024400000000000003E4000000000000034400000000000002440000000000000344000000000000026400000000000002440'::geometry, geom)"
```

Verify that the query can execute without an error.
```
SELECT * FROM testtable WHERE st_contains('polygon ((11 10,30 10,30 20,10 20,11 10))', geom);

1,some place,01010000000000000000002E400000000000002E40
```

### Run NVIDIA-SMI

You can optionally run NVIDIA-SMI to validate that the query is executing on the GPU.

```
nvidia-smi -l 1
```

Next, run `SELECT` rapidly in succession, and you can just run the example `SELECT` from above.

Every 1 second you will see outputs from `nvidia-smi`, and some of the outputs will show a `SELECT` command running at that exact moment. Not all of them will show this. See the last row here in this output as you can see the `SELECT` in the 6th column.

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2524      G   /usr/libexec/Xorg                 585MiB |
|    0   N/A  N/A      2648      G   /usr/bin/gnome-shell              143MiB |
|    0   N/A  N/A      3198      G   ...AAAAAAAAA= --shared-files       53MiB |
|    0   N/A  N/A      3712      G   /usr/lib64/firefox/firefox        142MiB |
|    0   N/A  N/A      4417      G   ...RendererForSitePerProcess       77MiB |
|    0   N/A  N/A      7001      G   ...RendererForSitePerProcess       22MiB |
|    0   N/A  N/A     13036      C   postgres: GPU0 memory keeper       80MiB |
|    0   N/A  N/A     13714      C   ... 172.17.0.1(38868) SELECT      300MiB |
+-----------------------------------------------------------------------------+
```

## Sample Performance Test Result

By using a large set of geographic data from the US Census and Open Addresses, a benchmark was run with and without GPU. This benchmark was run on the test computer described above in the README.md.

Here is the sample query:

```
select a.id
from fl_statewide_addresses a, census_zcta_geometry g
where st_contains(g.geom, a.geom4269)
limit 1000000
```

### Results:

1. 5s with GPU (using PG-Strom)
2. 56s without GPU (CPU-only)


## Publish Docker Image

If you choose to publish the image to your own repository, you can retag the image and use `docker push`. Here is an example for `murphye`.

```
docker tag pgstrom-postgis33-pg15 murphye/pgstrom-postgis33-pg15:v1
docker push murphye/pgstrom-postgis33-pg15:v1
```