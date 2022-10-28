# PG-Strom Experimental Dockerfile Documentation

It is possible to run PG-Strom in a Docker container where CUDA is enabled. This is an experimental Dockerfile that you can try to run on your own system. There is a good possibility that this Dockerfile will NOT work on your own system as-is. This Dockerfile was tested originally using WSL2 Ubuntu on Windows 11, but should, in theory, work on any CUDA-enabled container runtime.

## System Requirements

PG-Strom requires a recent NVIDIA video card. There is an official hardware validation list located [here](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List), but it is not comprehensive and assumes that you are running PG-Strom on a bare-metal system rather than a VM or Docker container.

This Dockerfile was tested on a computer with the following specs:

* Dell Precision 3240 Compact
* 3.3 GHz Intel Xeon W 6-Core (10th Gen)
* NVIDIA Quadro RTX 3000 (6GB GDDR6)
* 64 GB of 2666 MHz DDR4 RAM
* Windows 11
* Windows NVIDIA Driver 517.37
* Windows Subsystem for Linux 2 (WSL 2) with 31Gi of Memory and 8Gi of Swap (Run `free -mh` in Linux)
* Ubuntu 20.04.5 LTS running in WSL 2
* Windows Docker v20.10.17 (WSL 2 Backend)

### WSL2 Docker Support with CUDA GPU Acceleration

The NVIDIA driver supports the virtualization of the GPU with CUDA support under WSL 2, and it works automatically. You can read more on this capability [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). Docker for WSL 2 also support GPU virtualization to the Containers, and you can read more [here](https://www.docker.com/blog/wsl-2-gpu-support-for-docker-desktop-on-nvidia-gpus/). These integrations make it really easy to get up and running with CUDA support in containers!

## Build the PostGIS + PG-Strom Docker Image

```
cd docker/pg13
docker build -t pgstrom-postgis33-pg13-cuda11.7.0-rockylinux8 -f Dockerfile.pgstrom-postgis33-pg13-cuda11.7.0-rockylinux8 .
```

The generated Docker image will be 2.23GB in size.

```
docker images | grep pgstrom-postgis33-pg13-cuda11.7.0-rockylinux8
pgstrom-postgis33-pg13-cuda11.7.0-rockylinux8   latest    c7f30399ebe4   25 minutes ago   2.23GB
```

## Run the PostGIS + PG-Strom Docker Container

This command will run the PostGIS + PG-Strom Docker Image with a volume and as detached.

```
docker run --name postgis-db \
-d \
--gpus=all \
-e POSTGRES_USER=postgis \
-e POSTGRES_PASSWORD=password \
-e POSTGRES_DB=postgis \
-p 5432:5432 \
--shm-size=2gb \
-v ${PWD}/postgis-data:/var/lib/postgresql/data \
-d pgstrom-postgis33-pg13-cuda11.7.0-rockylinux8
```

Tail the logs and watch for any errors!
```
docker logs postgis-db -f
```

## Access the PostGIS + PG-Strom Docker Container

If you are running on Windows, you can open the Task Manager application to watch for utilization of your NVIDIA GPU to verify that the GPU acceleration is working when running queries that PG-Strom supports.

### Access the PostGIS + PG-Strom Docker Container's Shell

```
docker exec -it postgis-db bash
```

### Access the PostGIS Database (After `docker exec -it postgis-db bash`)
```
psql -U postgis
```

## Stop and Restart the PostGIS + PG-Strom Docker Container

Stop the PostGIS + PG-Strom Docker container.
```
docker stop postgis-db
```

Start the PostGIS + PG-Strom Docker container.
```
docker start postgis-db
```

## Clean Up the PostGIS + PG-Strom Docker Image

Stop the PostGIS + PG-Strom Docker container and remove the image.
```
docker stop postgis-db
docker rm postgis-db
```
Remove the data directory from the PostGIS + PG-Strom Docker volume.
```
sudo rm -r postgis-data
```