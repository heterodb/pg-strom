PG-Storm
========

PG-Strom is a custom-scan provider module for the PostgreSQL database.
It is designed to utilize GPU devices to accelarate sequential scan, hash-
based tables join and aggregate functions, at this moment.
Its basic concept is that the CPU and GPU should focus on the workload where
they have the advantage, and perform concurrently.
A CPU has much more flexibility, thus, it has an advantage on complex stuff
such as Disk-I/O, on the other hand, a GPU has much more parallelism for
numerical calculation, thus, it has advantage on massive but simple stuff
such as checks of qualifiers for rows.
