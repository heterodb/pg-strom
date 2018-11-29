@ja:<h1>GPUメモリストア(gstore_fdw)</h1>
@en:<h1>GPU Memory Store(gstore_fdw)</h1>

@ja:#概要
@en:#Overview

@ja{
通常、PG-StromはGPUデバイスメモリを一時的にだけ利用します。クエリの実行中に必要なだけのデバイスメモリを割り当て、その領域にデータを転送してSQLワークロードを実行するためにGPUカーネルを実行します。GPUカーネルの実行が完了すると、当該領域は速やかに開放され、他のワークロードでまた利用する事が可能となります。

これは複数セッションの並行実行やGPUデバイスメモリよりも巨大なテーブルのスキャンを可能にするための設計ですが、状況によっては必ずしも適切ではない場合もあります。

典型的な例は、それほど巨大ではなくGPUデバイスメモリに載る程度の大きさのデータに対して、繰り返し様々な条件で計算を行うといった利用シーンです。これは機械学習やパターンマッチ、類似度サーチといったワークロードが該当します。
S}
@en{
Usually, PG-Strom uses GPU device memory for temporary purpose only. It allocates a certain amount of device memory needed for query execution, then transfers data blocks and launch GPU kernel to process SQL workloads. Once GPU kernel gets finished, these device memory regison shall be released soon, to re-allocate unused device memory for other workloads.

This design allows concurrent multiple session or scan workloads on the tables larger than GPU device memory. It may not be optimal depending on circumstances.

A typical example is, repeated calculation under various conditions for data with a scale large enough to fit in the GPU device memory, not so large. This applies to workloads such as machine-learning, pattern matching or similarity search.
}

@ja{
現在のGPUにとって、数GB程度のデータをオンメモリで処理する事はそれほど難しい処理ではありませんが、PL/CUDA関数の呼び出しの度にGPUへロードすべきデータをCPUで加工し、これをGPUへ転送するのはコストのかかる処理です。

加えて、PostgreSQLの可変長データには1GBのサイズ上限があるため、これをPL/CUDA関数の引数として与える場合、データサイズ自体は十分にGPUデバイスメモリに載るものであってもデータ形式には一定の制約が存在する事になります。
}
@en{
For modern GPUs, it is not so difficult to process a few gigabytes data on memory at most, but it is a costly process to setup data to be loaded onto GPU device memory and transfer them.

In addition, since variable length data in PostgreSQL has size limitation up to 1GB, it restricts the data format when it is givrn as an argument of PL/CUDA function, even if the data size itself is sufficient in the GPU device memory.
}

@ja{
GPUメモリストア(gstore_fdw)は、あらかじめGPUデバイスメモリを確保しデータをロードしておくための機能です。
これにより、PL/CUDA関数の呼び出しの度に引数をセットアップしたりデータを転送する必要がなくなるほか、GPUデバイスメモリの容量が許す限りデータを確保する事ができますので、可変長データの1GBサイズ制限も無くなります。

gstore_fdwはその名の通り、PostgreSQLの外部データラッパ（Foreign Data Wrapper）を使用して実装されています。
gstore_fdwの制御する外部テーブル（Foreign Table）に対して`INSERT`、`UPDATE`、`DELETE`の各コマンドを実行する事で、GPUデバイスメモリ上のデータ構造を更新する事ができます。また、同様に`SELECT`文を用いてデータを読み出す事ができます。

外部テーブルを通してGPUデバイスメモリに格納されたデータは、PL/CUDA関数から参照する事ができます。
現在のところ、SQLから透過的に生成されたGPUプログラムは当該GPUデバイスメモリ領域を参照する事はできませんが、将来のバージョンにおいて改良が予定されています。
}

@en{
GPU memory store (gstore_fdw) is a feature to preserve GPU device memory and to load data to the memory preliminary.
It makes unnecessary to setup arguments and load for each invocation of PL/CUDA function, and eliminates 1GB limitation of variable length data because it allows GPU device memory allocation up to the capacity.

As literal, gstore_fdw is implemented using foreign-data-wrapper of PostgreSQL.
You can modify the data structure on GPU device memory using `INSERT`, `UPDATE` or `DELETE` commands on the foreign table managed by gstore_fdw. In the similar way, you can also read the data using `SELECT` command.

PL/CUDA function can reference the data stored onto GPU device memory through the foreign table.
Right now, GPU programs which is transparently generated from SQL statement cannot reference this device memory region, however, we plan to enhance the feature in the future release.
}

![GPU memory store](./img/gstore_fdw-overview.png)

@ja:#初期設定
@en:#Setup

@ja{
通常、外部テーブルを作成するには以下の3ステップが必要です。

- `CREATE FOREIGN DATA WRAPPER`コマンドにより外部データラッパを定義する
- `CREATE SERVER`コマンドにより外部サーバを定義する
- `CREATE FOREIGN TABLE`コマンドにより外部テーブルを定義する

このうち、最初の2ステップは`CREATE EXTENSION pg_strom`コマンドの実行に含まれており、個別に実行が必要なのは最後の`CREATE FOREIGN TABLE`のみです。
}

@en{
Usually it takes the 3 steps below to create a foreign table.

- Define a foreign-data-wrapper using `CREATE FOREIGN DATA WRAPPER` command
- Define a foreign server using `CREATE SERVER` command
- Define a foreign table using `CREATE FOREIGN TABLE` command

The first 2 steps above are included in the `CREATE EXTENSION pg_strom` command. All you need to run individually is `CREATE FOREIGN TABLE` command last.
}

```
CREATE FOREIGN TABLE ft (
    id int,
    signature smallint[] OPTIONS (compression 'pglz')
)
SERVER gstore_fdw OPTIONS(pinning '0', format 'pgstrom');
```

@ja{
`CREATE FOREIGN TABLE`コマンドを使用して外部テーブルを作成する際、いくつかのオプションを指定することができます。

`SERVER gstore_fdw`は必須です。外部テーブルがgstore_fdwによって制御されることを指定しています。

`OPTIONS`句では以下のオプションがサポートされています。
}

@en{
You can specify some options on creation of foreign table using `CREATE FOREIGN TABLE` command.

`SERVER gstore_fdw` is a mandatory option. It indicates the new foreign table is managed by gstore_fdw.

The options below are supported in the `OPTIONS` clause.
}

@ja{
|名前|対象  |説明       |
|:--:|:----:|:----------|
|`pinning`|テーブル|デバイスメモリを確保するGPUのデバイス番号を指定します。|
|`format`|テーブル|GPUデバイスメモリ上の内部データ形式を指定します。デフォルトは`pgstrom`です。|
|`compression`|カラム|可変長データを圧縮して保持するかどうかを指定します。デフォストは非圧縮です。|
}
@en{
|name|target|description|
|:--:|:----:|:----------|
|`pinning`|table|Specifies device number of the GPU where device memory is preserved.|
|`format`|table|Specifies the internal data format on GPU device memory. Default is `pgstrom`|
|`compression`|column|Specifies whether variable length data is compressed, or not. Default is uncompressed.|
}

@ja{
`format`オプションで選択可能なパラメータは、現在のところ`pgstrom`のみです。これは、PostgreSQLのデータを列形式で保持するために設計されたPG-Stromの独自形式です。
純粋にSQLを用いてデータの入出力を行うだけであればユーザが内部データ形式を意識する必要はありませんが、PL/CUDA関数をプログラミングしたり、IPCハンドルを用いて外部プログラムとGPUデバイスメモリを共有する場合には考慮が必要です。
}
@en{
Right now, only `pgstrom` is supported for `format` option. It is an original data format of PG-Strom to store structured data of PostgreSQL in columnar format.
In most cases, no need to pay attention to internal data format on writing / reading GPU data store using SQL. On the other hands, you need to consider when you program PL/CUDA function or share the GPU device memory with external applications using IPC handle.
}
@ja{
`compression`オプションで選択可能なパラメータは、現在のところ`plgz`のみです。これは、PostgreSQLが可変長データを圧縮する際に用いているものと同一の形式で、PL/CUDA関数からはGPU内関数`pglz_decompress()`を呼び出す事で展開が可能です。圧縮アルゴリズムの特性上、例えばデータの大半が0であるような疎行列を表現する際に有用です。
}
@en{
Right now, only `pglz` is supported for `compression` option. This compression logic adopts an identical data format and algorithm used by PostgreSQL to compress variable length data larger than its threshold.
It can be decompressed by GPU internal function `pglz_decompress()` from PL/CUDA function. Due to the characteristics of the compression algorithm, it is valuable to represent sparse matrix that is mostly zero.
}

@ja:#運用
@en:#Operations

@ja:##データのロード
@en:##Loading data

@ja{
通常のテーブルと同様にINSERT、UPDATE、DELETEによって外部テーブルの背後に存在するGPUデバイスメモリを更新する事ができます。

ただし、gstore_fdwはこれらコマンドの実行開始時に`SHARE UPDATE EXCLUSIVE`ロックを獲得する事に注意してください。これはある時点において１トランザクションのみがgstore_fdw外部テーブルを更新できることを意味します。
この制約は、PL/CUDA関数からgstore_fdw外部テーブルを参照するときに個々のレコード単位で可視性チェックを行う必要がないという特性を得るためのトレードオフです。
}
@en{
Like normal tables, you can write GPU device memory on behalf of the foreign table using `INSERT`, `UPDATE` and `DELETE` command.

Note that gstore_fdw acquires `SHARE UPDATE EXCLUSIVE` lock on the beginning of these commands. It means only single transaction can update the gstore_fdw foreign table at a certain point.
It is a trade-off. We don't need to check visibility per record when PL/CUDA function references gstore_fdw foreign table.
}

@ja{
また、gstore_fdw外部テーブルに書き込まれた内容は、通常のテーブルと同様にトランザクションがコミットされるまでは他のセッションからは不可視です。
この特性は、トランザクションの原子性を担保するには重要な性質ですが、古いバージョンを参照する可能性のある全てのトランザクションがコミットまたはアボートするまでの間は、古いバージョンのgstore_fdw外部テーブルの内容をGPUデバイスメモリに保持しておかねばならない事を意味します。

そのため、通常のテーブルと同様にINSERT、UPDATE、DELETEが可能であるとはいえ、数行を更新してトランザクションをコミットするという事を繰り返すのは避けるべきです。基本的には大量行のINSERTによるバルクロードを行うべきです。
}
@en{
Any contents written to the gstore_fdw foreign table is not visible to other sessions until transaction getting committed, like regular tables.
This is a significant feature to ensure atomicity of transaction, however, it also means the older revision of gstore_fdw foreign table contents must be kept on the GPU device memory until any concurrent transaction which may reference the older revision gets committed or aborted.

So, even though you can run `INSERT`, `UPDATE` or `DELETE` commands as if it is regular tables, you should avoidto update several rows then commit transaction many times. Basically, `INSERT` of massive rows at once (bulk loading) is recommended.
}

@ja{
通常のテーブルとは異なり、gstore_fdwに記録された内容は揮発性です。つまり、システムの電源断やPostgreSQLの再起動によってgstore_fdw外部テーブルの内容は容易に失われてしまいます。したがって、gstore_fdw外部テーブルにロードするデータは、他のデータソースから容易に復元可能な形にしておくべきです。
}
@en{
Unlike regular tables, contents of the gstore_fdw foreign table is vollatile. So, it is very easy to loose contents of the gstore_fdw foreign table by power-down or PostgreSQL restart. So, what we load onto gstore_fdw foreign table should be reconstructable by other data source.
}

@ja:##デバイスメモリ消費量の確認
@en:##Checking the memory consumption

@ja{
gstore_fdwによって消費されるデバイスメモリのサイズを確認するには`pgstrom.gstore_fdw_chunk_info`システムビューを参照します。
}
@en{
See `pgstrom.gstore_fdw_chunk_info` system view to see amount of the device memory consumed by gstore_fdw.
}

```
postgres=# select * from pgstrom.gstore_fdw_chunk_info ;
 database_oid | table_oid | revision | xmin | xmax | pinning | format  |  rawsize  |  nitems
--------------+-----------+----------+------+------+---------+---------+-----------+----------
        13806 |     26800 |        3 |    2 |    0 |       0 | pgstrom | 660000496 | 15000000
        13806 |     26797 |        2 |    2 |    0 |       0 | pgstrom | 440000496 | 10000000
(2 rows)
```

@ja{
`nvidia-smi`コマンドを用いて、各GPUデバイスが実際にどの程度のデバイスメモリを消費しているかを確認する事ができます。
Gstore_fdwが確保したメモリは、PG-Strom GPU memory keeperプロセスが保持・管理しています。ここでは、上記rawsizeの合計約1100MBに加え、CUDAが内部的に確保する領域を合わせて1211MBが占有されている事が分かります。
}

@en{
By `nvidia-smi` command, you can check how much device memory is consumed for each GPU device.
"PG-Strom GPU memory keeper" process actually keeps and manages the device memory area acquired by Gstore_fdw. In this example, 1211MB is preliminary allocated for total of the above rawsize (about 1100MB) and CUDA internal usage.
}

```
$ nvidia-smi
Wed Apr  4 15:11:50 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.30                 Driver Version: 390.30                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P40           Off  | 00000000:02:00.0 Off |                    0 |
| N/A   39C    P0    52W / 250W |   1221MiB / 22919MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      6885      C   ...bgworker: PG-Strom GPU memory keeper     1211MiB |
+-----------------------------------------------------------------------------+
```


@ja:##内部データ形式
@en:##Internal Data Format

@ja{
gstore_fdwがGPUデバイスメモリ上にデータを保持する際の内部データ形式の詳細はノートを参照してください。

- [`pgstrom`フォーマットの詳細](https://github.com/heterodb/pg-strom/wiki/301:-Gstore_fdw-internal-format-of-%27pgstrom%27)
}

@en{
See the notes for details of the internal data format when gstore_fdw write on GPU device memory.

- [Detail of the `pgstrom` format](https://github.com/heterodb/pg-strom/wiki/301:-Gstore_fdw-internal-format-of-%27pgstrom%27)
}

@ja:#外部プログラムとのデータ連携
@en:#Inter-process Data Collaboration

@ja{
CUDAには`cuIpcGetMemHandle()`および`cuIpcOpenMemHandle()`というAPIが用意されています。前者を用いてアプリケーションプログラムが確保したGPUデバイスメモリのユニークな識別子を取得し、後者を用いて別のアプリケーションプログラムから同一のGPUデバイスメモリを参照する事が可能となります。言い換えれば、ホストシステムにおける共有メモリのような仕組みを備えています。

このユニークな識別子は`CUipcMemHandle`型のオブジェクトで、内部的には単純な64バイトのバイナリデータです。
本節では`CUipcMemHandle`識別子を利用して、PostgreSQLと外部プログラムの間でGPUを介したデータ交換を行うための関数について説明します。
}
@en{
CUDA provides special APIs `cuIpcGetMemHandle()` and `cuIpcOpenMemHandle()`.
The first allows to get a unique identifier of GPU device memory allocated by applications. The other one allows to reference a shared GPU device memory region from other applications. In the other words, it supports something like a shared memory on the host system.

This unique identifier is `CUipcMemHandle` object; which is simple binary data in 64bytes.
This session introduces SQL functions which exchange GPU device memory with other applications using `CUipcMemHandle` identifier.
}

@ja:##SQL関数の一覧
@en:##SQL Functions to 

### gstore_export_ipchandle(reggstore)

@ja{
本関数は、gstore_fdw制御下の外部テーブルがGPU上に確保しているデバイスメモリの`CUipcMemHandle`識別子を取得し、bytea型のバイナリデータとして出力します。
外部テーブルが空でGPU上にデバイスメモリを確保していなければNULLを返します。

- 第1引数(*ftable_oid*): 外部テーブルのOID。`reggstore`型なので、外部テーブル名を文字列で指定する事もできる。
- 戻り値: `CUipcMemHandle`識別子のbytea型表現。

}
@en{
This function gets `CUipcMemHandle` identifier of the GPU device memory which is preserved by gstore_fdw foreign table, then returns as a binary data in `bytea` type.
If foreign table is empty and has no GPU device memory, it returns NULL.

- 1st arg(*ftable_oid*): OID of the foreign table. Because it is `reggstore` type, you can specify the foreign table by name string.
- result: `CUipcMemHandle` identifier in the bytea type.
}

```
# select gstore_export_ipchandle('ft');
                                                      gstore_export_ipchandle

------------------------------------------------------------------------------------------------------------------------------------
 \xe057880100000000de3a000000000000904e7909000000000000800900000000000000000000000000020000000000005c000000000000001200d0c10101005c
(1 row)
```

### lo_import_gpu(int, bytea, bigint, bigint, oid=0)

@ja{
本関数は、外部アプリケーションがGPU上に確保したデバイスメモリ領域をPostgreSQL側で一時的にオープンし、当該領域の内容を読み出してPostgreSQLラージオブジェクトとして書き出します。
第5引数で指定したラージオブジェクトが既に存在する場合、ラージオブジェクトはGPUデバイスメモリから読み出した内容で置き換えられます。ただし所有者・パーミッション設定は保持されます。これ以外の場合は、新たにラージオブジェクトを作成し、GPUデバイスメモリから読み出した内容を書き込みます。
}
@en{
This function temporary opens the GPU device memory region acquired by external applications, then read this region and writes out as a largeobject of PostgreSQL.
If largeobject already exists, its contents is replaced by the data read from the GPU device memory. It keeps owner and permission configuration. Elsewhere, it creates a new largeobject, then write out the data which is read from GPU device memory.
}
@ja{
- 第1引数(*device_nr*): デバイスメモリを確保したGPUデバイス番号
- 第2引数(*ipc_mhandle*): `CUipcMemHandle`識別子のbytea型表現。
- 第3引数(*offset*): 読出し開始位置のデバイスメモリ領域先頭からのオフセット
- 第4引数(*length*): バイト単位での読出しサイズ
- 第5引数(*loid*): 書き込むラージオブジェクトのOID。省略した場合 0 が指定されたものと見なす。
- 戻り値: 書き込んだラージオブジェクトのOID
}
@en{
- 1st arg(*device_nr*): GPU device number where device memory is acquired
- 2nd arg(*ipc_mhandle*): `CUipcMemHandle` identifier in bytea type
- 3rd(*offset*): offset of the head position to read, from the GPU device memory region.
- 4th(*length*): size to read in bytes
- 5th(*loid*): OID of the largeobject to be written. 0 is assumed, if no valid value is supplied.
- result: OID of the written largeobject
}

### lo_export_gpu(oid, int, bytea, bigint, bigint)

@ja{
本関数は、外部アプリケーションがGPU上に確保したデバイスメモリ領域をPostgreSQL側で一時的にオープンし、当該領域へPostgreSQLラージオブジェクトの内容を書き出します。
ラージオブジェクトのサイズが指定された書き込みサイズよりも小さい場合、残りの領域は 0 でクリアされます。
}
@en{}
@ja{
- 第1引数(*loid*): 読み出すラージオブジェクトのOID
- 第2引数(*device_nr*): デバイスメモリを確保したGPUデバイス番号
- 第3引数(*ipc_mhandle*): `CUipcMemHandle`識別子のbytea型表現。
- 第4引数(*offset*): 書き込み開始位置のデバイスメモリ領域先頭からのオフセット
- 第5引数(*length*): バイト単位での書き込みサイズ
- 戻り値: 実際に書き込んだバイト数。指定されたラージオブジェクトの大きさが*length*よりも小さな場合、*length*よりも小さな値を返す事がある。
}
@en{
- 1st arg(*loid*): OID of the largeobject to be read
- 2nd arg(*device_nr*): GPU device number where device memory is acquired
- 3rd arg(*ipc_mhandle*): `CUipcMemHandle` identifier in bytea type
- 4th arg(*offset*): offset of the head position to write, from the GPU device memory region.
- 5th arg(*length*): size to write in bytes
- result: Length of bytes actually written. If length of the largeobject is less then *length*, it may return the value less than *length*.
}
