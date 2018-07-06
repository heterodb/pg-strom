@ja:<h1>トラブルシューティング</h1>
@en:<h1>Trouble Shooting</h1>

@ja:# 問題の切り分け
@en:# Identify the problem

@ja{
特定のワークロードを実行した際に何がしかの問題が発生する場合には、それが何に起因するものであるのかを特定するのはトラブルシューティングの第一歩です。

残念ながら、PostgreSQL開発者コミュニティと比べPG-Stromの開発者コミュニティは非常に少ない数の開発者によって支えられています。そのため、ソフトウェアの品質や実績といった観点から、まずPG-Stromが悪さをしていないか疑うのは妥当な判断です。
}
@en{
In case when a particular workloads produce problems, it is the first step to identify which stuff may cause the problem.

Unfortunately, much smaller number of developer supports the PG-Strom development community than PostgreSQL developer's community, thus, due to the standpoint of software quality and history, it is a reasonable estimation to suspect PG-Strom first.
}
@ja{
PG-Stromの全機能を一度に有効化/無効化するには`pg_strom.enabled`パラメータを使用する事ができます。
以下の設定を行う事でPG-Stromは無効化され、標準のPostgreSQLと全く同一の状態となります。
それでもなお問題が再現するかどうかは一つの判断材料となるでしょう。
}
@en{
The `pg_strom.enabled` parameter allows to turn on/off all the functionality of PG-Strom at once.
The configuration below disables PG-Strom, thus identically performs with the standard PostgreSQL.
}
```
# SET pg_strom.enabled = off;
```

@ja{
この他にも、GpuScan、GpuJoin、GpuPreAggといった特定の実行計画のみを無効化するパラメータも定義されています。

これらの詳細は[リファレンス/GPUパラメータ](ref_params.md)を参照してください。
}
@en{
In addition, we provide parameters to disable particular execution plan like GpuScan, GpuJoin and GpuPreAgg.

See [references/GUC Parameters](ref_params.md) for more details.
}

@ja:# クラッシュダンプの採取
@en:# Collecting crash dump

@ja{
システムのクラッシュを引き起こすような重大なトラブルの解析にはクラッシュダンプの採取が欠かせません。
本節では、PostgreSQLとPG-Stromプロセスのクラッシュダンプ(CPU側)、およびPG-StromのGPUカーネルのクラッシュダンプ(GPU側)を取得し、障害発生時のバックトレースを採取するための手段を説明します。
}
@en{
Crash dump is very helpful for analysis of serious problems which lead system crash for example.
This session introduces the way to collect crash dump of the PostgreSQL and PG-Strom process (CPU side) and PG-Strom's GPU kernel, and show the back trace on the serious problems.
}

@ja:## PostgreSQL起動時設定の追加
@en:## Add configuration on PostgreSQL startup

@ja{
プロセスのクラッシュ時にクラッシュダンプ(CPU側)を生成するには、PostgreSQLサーバプロセスが生成する事のできる core ファイルのサイズを無制限に変更する必要があります。これはPostgreSQLサーバプロセスを起動するシェル上で`ulimit -c`コマンドを実行して変更する事ができます。

GPUカーネルのエラー時にクラッシュダンプ(GPU側)を生成するには、PostgreSQLサーバプロセスが環境変数`CUDA_ENABLE_COREDUMP_ON_EXCEPTION`に`1`が設定されている必要があります。
}
@en{
For generation of crash dump (CPU-side) on process crash, you need to change the resource limitation of the operating system for size of core file  PostgreSQL server process can generate.

For generation of crash dump (GPU-size) on errors of GPU kernel, PostgreSQL server process has `CUDA_ENABLE_COREDUMP_ON_EXCEPTION`environment variable, and its value has `1`.
}
@ja{
systemdからPostgreSQLを起動する場合、`/etc/systemd/system/postgresql-<version>.service.d/`以下に設定ファイルを作成し、これらの設定を追加する事ができます。

RPMインストールの場合は、以下の内容の`pg_strom.conf`というファイルが作成されています。
}
@en{
You can put a configuration file at `/etc/systemd/system/postgresql-<version>.service.d/` when PostgreSQL is kicked by systemd.

In case of RPM installation, a configuration file `pg_strom.conf` is also installed on the directory, and contains the following initial configuration.
}
```
[Service]
LimitNOFILE=65536
LimitCORE=infinity
#Environment=CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
```
@ja{
CUDA9.1においては、通常、GPUカーネルのクラッシュダンプの生成には数分以上の時間を要し、その間、エラーを発生したPostgreSQLセッションの応答は完全に停止してしまします。
そのため、は特定クエリの実行において発生するGPUカーネルに起因するエラーの原因調査を行う場合にだけ、`CUDA_ENABLE_COREDUMP_ON_EXCEPTION`環境変数を設定する事をお勧めします。
RPMインストールにおけるデフォルト設定は、`CUDA_ENABLE_COREDUMP_ON_EXCEPTION`環境変数の行をコメントアウトしています。
}
@en{
In CUDA 9.1, it usually takes more than several minutes to generate crash dump of GPU kernel, and it entirely stops response of the PostgreSQL session which causes an error.
So, we recommend to set `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` environment variable only if you investigate errors of GPU kernels which happen on a certain query.
The default configuration on RPM installation comments out the line of `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` environment variable.
}

@ja{
PostgreSQLサーバプロセスを再起動すると、*Max core file size*がunlimitedに設定されているはずです。

以下のように確認する事ができます。
}
@en{
PostgreSQL server process should have unlimited *Max core file size* configuration, after the next restart.

You can check it as follows.
}

```
# cat /proc/<PID of postmaster>/limits
Limit                     Soft Limit           Hard Limit           Units
    :                         :                    :                  :
Max core file size        unlimited            unlimited            bytes
    :                         :                    :                  :
```

@ja:## debuginfoパッケージのインストール
@en:## Installation of debuginfo package

@ja{
クラッシュダンプから意味のある情報を読み取るにはシンボル情報が必要です。

これらは`-debuginfo`パッケージに格納されており、システムにインストールされているPostgreSQLおよびPG-Stromのパッケージに応じてそれぞれ追加インストールが必要です。
}

```
# yum install postgresql10-debuginfo pg_strom-PG10-debuginfo
            :
================================================================================
 Package                  Arch    Version             Repository           Size
================================================================================
Installing:
 pg_strom-PG10-debuginfo  x86_64  1.9-180301.el7      heterodb-debuginfo  766 k
 postgresql10-debuginfo   x86_64  10.3-1PGDG.rhel7    pgdg10              9.7 M

Transaction Summary
================================================================================
Install  2 Packages
            :
Installed:
  pg_strom-PG10-debuginfo.x86_64 0:1.9-180301.el7
  postgresql10-debuginfo.x86_64 0:10.3-1PGDG.rhel7

Complete!
```

@ja:## CPU側バックトレースの確認
@en:## Checking the back-trace on CPU side

@ja{
クラッシュダンプの作成されるパスは、カーネルパラメータ`kernel.core_pattern`および`kernel.core_uses_pid`の値によって決まります。
通常はプロセスのカレントディレクトリに作成されますので、systemdからPostgreSQLを起動した場合はデータベースクラスタが構築される`/var/lib/pgdata`を確認してください。

`core.<PID>`ファイルが生成されているのを確認したら、`gdb`を用いてクラッシュに至るバックトレースを確認します。

`gdb`の`-c`オプションでコアファイルを、`-f`オプションでクラッシュしたプログラムを指定します。
}
@en{
The kernel parameter `kernel.core_pattern` and `kernel.core_uses_pid` determine the path where crash dump is written out.
It is usually created on the current working directory of the process, check `/var/lib/pgdata` where the database cluster is deployed, if you start PostgreSQL server using systemd.

Once `core.<PID>` file gets generated, you can check its back-trace to reach system crash using `gdb`.

`gdb` speficies the core file by `-c` option, and the crashed program by `-f` option.
}
```
# gdb -c /var/lib/pgdata/core.134680 -f /usr/pgsql-10/bin/postgres
GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-100.el7_4.1
       :
(gdb) bt
#0  0x00007fb942af3903 in __epoll_wait_nocancel () from /lib64/libc.so.6
#1  0x00000000006f71ae in WaitEventSetWaitBlock (nevents=1,
    occurred_events=0x7ffee51e1d70, cur_timeout=-1, set=0x2833298)
    at latch.c:1048
#2  WaitEventSetWait (set=0x2833298, timeout=timeout@entry=-1,
    occurred_events=occurred_events@entry=0x7ffee51e1d70,
    nevents=nevents@entry=1, wait_event_info=wait_event_info@entry=100663296)
    at latch.c:1000
#3  0x00000000006210fb in secure_read (port=0x2876120,
    ptr=0xcaa7e0 <PqRecvBuffer>, len=8192) at be-secure.c:166
#4  0x000000000062b6e8 in pq_recvbuf () at pqcomm.c:963
#5  0x000000000062c345 in pq_getbyte () at pqcomm.c:1006
#6  0x0000000000718682 in SocketBackend (inBuf=0x7ffee51e1ef0)
    at postgres.c:328
#7  ReadCommand (inBuf=0x7ffee51e1ef0) at postgres.c:501
#8  PostgresMain (argc=<optimized out>, argv=argv@entry=0x287bb68,
    dbname=0x28333f8 "postgres", username=<optimized out>) at postgres.c:4030
#9  0x000000000047adbc in BackendRun (port=0x2876120) at postmaster.c:4405
#10 BackendStartup (port=0x2876120) at postmaster.c:4077
#11 ServerLoop () at postmaster.c:1755
#12 0x00000000006afb7f in PostmasterMain (argc=argc@entry=3,
    argv=argv@entry=0x2831280) at postmaster.c:1363
#13 0x000000000047bbef in main (argc=3, argv=0x2831280) at main.c:228
```

@ja{
gdbの`bt`コマンドでバックトレースを確認します。
このケースでは、クライアントからのクエリを待っている状態のPostgreSQLバックエンドに`SIGSEGV`シグナルを送出してクラッシュを引き起こしたため、`WaitEventSetWait`延長上の`__epoll_wait_nocancel`でプロセスがクラッシュしている事がわかります。
}
@en{
`bt` command of `gdb` displays the backtrace.
In this case, I sent `SIGSEGV` signal to the PostgreSQL backend which is waiting for queries from the client for intentional crash, the process got crashed at `__epoll_wait_nocancel` invoked by `WaitEventSetWait`.
}


@ja:## GPU側バックトレースの確認
@en:## Checking the backtrace on GPU

@ja{
GPUカーネルのクラッシュダンプは、（`CUDA_COREDUMP_FILE`環境変数を用いて明示的に指定しなければ）PostgreSQLサーバプロセスのカレントディレクトリに生成されます。
systemdからPostgreSQLを起動した場合はデータベースクラスタが構築される`/var/lib/pgdata`を確認してください。以下の名前でGPUカーネルのクラッシュダンプが生成されています。
}
@en{
Crash dump of GPU kernel is generated on the current working directory of PostgreSQL server process, unless you don't specify the path using `CUDA_COREDUMP_FILE` environment variable explicitly.
Check `/var/lib/pgdata` where the database cluster is deployed, if systemd started PostgreSQL. Dump file will have the following naming convension.
}

`core_<timestamp>_<hostname>_<PID>.nvcudmp`

@ja{
なお、デフォルト設定ではGPUカーネルのクラッシュダンプにはシンボル情報などのデバッグ情報が含まれていません。この状態では障害解析を行う事はほとんど不可能ですので、以下の設定を行ってPG-Stromが生成するGPUプログラムにデバッグ情報を含めるようにしてください。

ただし、この設定は実行時のパフォーマンスを低下させるため、恒常的な使用は非推奨です。
トラブル解析時にだけ使用するようにしてください。
}
@en{
Note that the dump-file of GPU kernel contains no debug information like symbol information in the default configuration.
It is nearly impossible to investigate the problem, so enable inclusion of debug information for the GPU programs generated by PG-Strom, as follows.

Also note than we don't recommend to turn on the configuration for daily usage, because it makes query execution performan slow down.
Turn on only when you investigate the troubles.
}
```
nvme=# set pg_strom.debug_jit_compile_options = on;
SET
```

@ja{
生成されたGPUカーネルのクラッシュダンプを確認するには`cuda-gdb`コマンドを使用します。
}
@en{
You can check crash dump of the GPU kernel using `cuda-gdb` command.
}
```
# /usr/local/cuda/bin/cuda-gdb
NVIDIA (R) CUDA Debugger
9.1 release
Portions Copyright (C) 2007-2017 NVIDIA Corporation
        :
For help, type "help".
Type "apropos word" to search for commands related to "word".
(cuda-gdb)
```

@ja{
引数なしで`cuda-gdb`コマンドを実行し、プロンプト上で`target`コマンドを使用して先ほどのクラッシュダンプを読み込みます。
}
@en{
Run `cuda-gdb` command, then load the crash dump file above using `target` command on the prompt.
}
```
(cuda-gdb) target cudacore /var/lib/pgdata/core_1521131828_magro.heterodb.com_216238.nvcudmp
Opening GPU coredump: /var/lib/pgdata/core_1521131828_magro.heterodb.com_216238.nvcudmp
[New Thread 216240]

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7ff4dc82f930 (cuda_gpujoin.h:1159)
[Current focus set to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]
#0  0x00007ff4dc82f938 in _INTERNAL_8_pg_strom_0124cb94::gpujoin_exec_hashjoin (kcxt=0x7ff4f7fffbf8, kgjoin=0x7fe9f4800078,
    kmrels=0x7fe9f8800000, kds_src=0x7fe9f0800030, depth=3, rd_stack=0x7fe9f4806118, wr_stack=0x7fe9f480c118, l_state=0x7ff4f7fffc48,
    matched=0x7ff4f7fffc7c "") at /usr/pgsql-10/share/extension/cuda_gpujoin.h:1159
1159            while (khitem && khitem->hash != hash_value)
```

@ja{
この状態で`bt`コマンドを使用し、問題発生個所へのバックトレースを採取する事ができます。
}
@en{
You can check backtrace where the error happened on GPU kernel using `bt` command.
}
```
(cuda-gdb) bt
#0  0x00007ff4dc82f938 in _INTERNAL_8_pg_strom_0124cb94::gpujoin_exec_hashjoin (kcxt=0x7ff4f7fffbf8, kgjoin=0x7fe9f4800078,
    kmrels=0x7fe9f8800000, kds_src=0x7fe9f0800030, depth=3, rd_stack=0x7fe9f4806118, wr_stack=0x7fe9f480c118, l_state=0x7ff4f7fffc48,
    matched=0x7ff4f7fffc7c "") at /usr/pgsql-10/share/extension/cuda_gpujoin.h:1159
#1  0x00007ff4dc9428f0 in gpujoin_main<<<(30,1,1),(256,1,1)>>> (kgjoin=0x7fe9f4800078, kmrels=0x7fe9f8800000, kds_src=0x7fe9f0800030,
    kds_dst=0x7fe9e8800030, kparams_gpreagg=0x0) at /usr/pgsql-10/share/extension/cuda_gpujoin.h:1347
```
@ja{
より詳細な`cuda-gdb`コマンドの利用法は[CUDA Toolkit Documentation - CUDA-GDB](http://docs.nvidia.com/cuda/cuda-gdb/)を参照してください。
}
@en{
Please check [CUDA Toolkit Documentation - CUDA-GDB](http://docs.nvidia.com/cuda/cuda-gdb/) for more detailed usage of `cuda-gdb` command.
}
