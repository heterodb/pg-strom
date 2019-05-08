@ja:<h1>列指向データストア (Arrow_Fdw)</h1>
@en:<h1>Columnar data store (Arrow_Fdw)</h1>

@ja:#概要
@en:#Overview

@ja{
PostgreSQLのテーブルは内部的に8KBのブロック[^1]と呼ばれる単位で編成され、ブロックは全ての属性及びメタデータを含むタプルと呼ばれるデータ構造を行単位で格納します。行を構成するデータが近傍に存在するため、これはINSERTやUPDATEの多いワークロードに有効ですが、一方で大量データの集計・解析ワークロードには不向きであるとされています。

[^1]: 正確には、4KB～32KBの範囲でビルド時に指定できます
}
@en{
PostgreSQL tables internally consist of 8KB blocks[^1], and block contains tuples which is a data structure of all the attributes and metadata per row. It collocates date of a row closely, so it works effectively for INSERT/UPDATE-major workloads, but not suitable for summarizing or analytics of mass-data.

[^1]: For correctness, block size is configurable on build from 4KB to 32KB. 
}

@ja{
通常、大量データの集計においてはテーブル内の全ての列を参照する事は珍しく、多くの場合には一部の列だけを参照するといった処理になりがちです。この場合、実際には参照されない列のデータをストレージからロードするために消費されるI/Oの帯域は全く無駄ですが、行単位で編成されたデータに対して特定の列だけを取り出すという操作は困難です。
}
@en{
It is not usual to reference all the columns in a table on mass-data processing, and we tend to reference a part of columns in most cases. In this case, the storage I/O bandwidth consumed by unreferenced columns are waste, however, we have no easy way to fetch only particular columns referenced from the row-oriented data structure.
}

@ja{
逆に列単位でデータを編成した場合、INSERTやUPDATEの多いワークロードに対しては極端に不利ですが、大量データの集計・解析を行う際には被参照列だけをストレージからロードする事が可能になるため、I/Oの帯域を最大限に活用する事が可能です。 またプロセッサの処理効率の観点からも、列単位に編成されたデータは単純な配列であるかのように見えるため、GPUにとってはCoalesced Memory Accessというメモリバスの性能を最大限に引き出すアクセスパターンとなる事が期待できます。
}
@en{
In case of column oriented data structure, in an opposite manner, it has extreme disadvantage on INSERT/UPDATE-major workloads, however, it can pull out maximum performance of storage I/O on mass-data processing workloads because it can loads only referenced columns. From the standpoint of processor efficiency also, column-oriented data structure looks like a flat array that pulls out maximum bandwidth of memory subsystem for GPU, by special memory access pattern called Coalesced Memory Access.
}
![Row/Column data structure](./img/row_column_structure.png)


@ja:##Apache Arrowとは
@en:##What is Apache Arrow?

@ja{
Apache Arrowとは、構造化データを列形式で記録、交換するためのデータフォーマットです。 主にビッグデータ処理のためのアプリケーションソフトウェアが対応しているほか、CやC++、Pythonなどプログラミング言語向けのライブラリが整備されているため、自作のアプリケーションからApache Arrow形式を扱うよう設計する事も容易です。
}
@en{
Apache Arrow is a data format of structured data to save in columnar-form and to exchange other applications. Some applications for big-data processing support the format, and it is easy for self-developed applications to use Apache Arrow format since they provides libraries for major programming languages like C,C++ or Python.
}

![Row/Column data structure](./img/arrow_shared_memory.png)

@ja{
Apache Arrow形式ファイルの内部には、データ構造を定義するスキーマ（Schema）部分と、スキーマに基づいて列データを記録する1個以上のレコードバッチ（RecordBatch）部分が存在します。データ型としては、整数や文字列（可変長）、日付時刻型などに対応しており、個々の列データはこれらデータ型に応じた内部表現を持っています。
}
@en{
Apache Arrow format file internally contains Schema portion to define data structure, and one or more RecordBatch to save columnar-data based on the schema definition. For data types, it supports integers, strint (variable-length), date/time types and so on. Indivisual columnar data has its internal representation according to the data types.
}

@ja{
Apache Arrow形式におけるデータ表現は、必ずしも全ての場合でPostgreSQLのデータ表現と一致している訳ではありません。例えば、Arrow形式ではタイムスタンプ型のエポックは`1970-01-01`で複数の精度を持つ事ができますが、PostgreSQLのエポックは`2001-01-01`でマイクロ秒の精度を持ちます。
}
@en{
Data representation in Apache Arrow is not identical with the representation in PostgreSQL. For example, epoch of timestamp in Arrow is `1970-01-01` and it supports multiple precision. On the other hands, epoch of timestamp in PostgreSQL is `2001-01-01` and it has microseconds accuracy.
}

@ja{
Arrow_Fdwは外部テーブルを用いてApache Arrow形式ファイルをPostgreSQL上で読み出す事を可能にします。例えば、列ごとに100万件の列データが存在するレコードバッチを8個内包するArrow形式ファイルをArrow_Fdwを用いてマップした場合、この外部テーブルを介してArrowファイル上の800万件のデータへアクセスする事ができるようになります。
}
@en{
Arrow_Fdw allows to read Apache Arrow files on PostgreSQL using foreign table mechanism. If an Arrow file contains 8 of record batches that has million items for each column data, for example, we can access 8 million rows on the Arrow files through the foreign table.
}

@ja:#運用
@en:#Operations

@ja:##外部テーブルの定義
@en:##Creation of foreign tables

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
CREATE FOREIGN TABLE flogdata (
    ts        timestamp,
    sensor_id int,
    signal1   smallint,
    signal2   smallint,
    signal3   smallint,
    signal4   smallint,
) SERVER arrow_fdw
  OPTIONS (file '/path/to/logdata.arrow');
```

@ja{
`CREATE FOREIGN TABLE`構文で指定した列のデータ型は、マップするArrow形式ファイルのスキーマ定義と厳密に一致している必要があります。
}
@en{
Data type of columns specified by the `CREATE FOREIGN TABLE` command must be matched to schema definition of the Arrow files to be mapped.
}

@ja{
これ以外にも、Arrow_Fdwは`IMPORT FOREIGN SCHEMA`構文を用いた便利な方法に対応しています。これは、Arrow形式ファイルの持つスキーマ情報を利用して、自動的にテーブル定義を生成するというものです。 以下のように、外部テーブル名とインポート先のスキーマ、およびOPTION句でArrow形式ファイルのパスを指定します。 Arrowファイルのスキーマ定義には、列ごとのデータ型と列名（オプション）が含まれており、これを用いて外部テーブルの定義を行います。
}
@en{
Arrow_Fdw also supports a useful manner using `IMPORT FOREIGN SCHEMA` statement. It automatically generates a foreign table definition using schema definition of the Arrow files. It specifies the foreign table name, schema name to import, and path name of the Arrow files using OPTION-clause. Schema definition of Arrow files contains data types and optional column name for each column. It declares a new foreign table using these information.
}

```
IMPORT FOREIGN SCHEMA flogdata
  FROM SERVER arrow_fdw
  INTO public
OPTIONS (file '/path/to/logdata.arrow');
```

@ja:##外部テーブルオプション
@en:##Foreign table options

@ja{
Arrow_Fdwは以下のオプションに対応しています。現状、全てのオプションは外部テーブルに対して指定するものです。

|対象|オプション|説明|
|:---|:---------|:---|
|外部テーブル|`file`|外部テーブルにマップするArrowファイルを1個指定します。|
|外部テーブル|`files`|外部テーブルにマップするArrowファイルをカンマ(,）区切りで複数指定します。|
|外部テーブル|`dir`|指定したディレクトリに格納されている全てのファイルを外部テーブルにマップします。|
|外部テーブル|`suffix`|`dir`オプションの指定時、例えば`.arrow`など、特定の接尾句を持つファイルだけをマップします。|
}
@en{
Arrow_Fdw supports the options below. Right now, all the options are for foreign tables.

|Target|Option|Description|
|:-----|:-----|:----------|
|foreign table|`file`|It maps an Arrow file specified on the foreign table.
|foreign table|`files`|It maps multiple Arrow files specified by comma (,) separated files list on the foreign table.
|foreign table|`dir`|It maps all the Arrow files in the directory specified on the foreign table.
|foreign table|`suffix`|When `dir` option is given, it maps only files with the specified suffix, like `.arrow` for example.
}

@ja:##データ型の対応
@en:##Data type mapping

@ja{
Arrow形式のデータ型と、PostgreSQLのデータ型は以下のように対応しています。

|Arrowデータ型  |PostgreSQLデータ型|備考|
|:--------------|:-----------------|:---|
|`Int`          |`int2,int4,int8`  |`is_signed`属性は無視。`bitWidth`属性は16、32または64のみ対応。|
|`FloatingPoint`|`float2,float4,float8`|`float2`はPG-Stromによる独自拡張|
|`Binary`       |`bytea`           |    |
|`Utf8`         |`text`            |    |
|`Decimal`      |`numeric          |    |
|`Date`         |`date`            |`unitsz=Day`相当に補正|
|`Time`         |`time`            |`unitsz=MicroSecond`相当に補正|
|`Timestamp`    |`timestamp`       |`unitsz=MicroSecond`相当に補正|
|`Interval`     |`interval`        |    |
|`List`         |配列型            |1次元配列のみ対応（予定）|
|`Struct`       |複合型            |対応する複合型を予め定義しておくこと。|
|`Union`        |--------          ||
|`FixedSizeBinary`|`char(n)`       ||
|`FixedSizeList`|--------          ||
|`Map`          |--------          ||
}
@en{
Arrow data types are mapped on PostgreSQL data types as follows.

|Arrow data types|PostgreSQL data types|Remarks|
|:---------------|:--------------------|:------|
|`Int`           |`int2,int4,int8`     |`is_signed` attribute is ignored. `bitWidth` attribute supports only 16,32 or 64.|
|`FloatingPoint` |`float2,float4,float8`|`float2` is enhanced by PG-Strom.|
|`Binary`        |`bytea`              ||
|`Utf8`          |`text`               ||
|`Decimal`       |`numeric`            ||
|`Date`          |`date`               |Adjusted as if `unitsz=Day`|
|`Time`          |`time`               |Adjusted as if `unitsz=MicroSecond`|
|`Timestamp`     |`timestamp`          |Adjusted as if `unitsz=MicroSecond`|
|`Interval`      |`interval`           ||
|`List`          |array of base type   |It supports only 1-dimensional List(WIP).|
|`Struct`        |composite type       |PG composite type must be preliminary defined.|
|`Union`         |--------             ||
|`FixedSizeBinary`|`char(n)`           ||
|`FixedSizeList` |--------             ||
|`Map`           |--------             ||
}

@ja:##EXPLAIN出力の読み方
@en:##How to read EXPLAIN

@ja{
`EXPLAIN`コマンドを用いて、Arrow形式ファイルの読み出しに関する情報を出力する事ができます。

以下の例は、約309GBの大きさを持つArrow形式ファイルをマップしたflineorder外部テーブルを含むクエリ実行計画の出力です。
}
@en{
`EXPLAIN` command show us information about Arrow files reading.

The example below is an output of query execution plan that includes flineorder foreign table that mapps an Arrow file of 309GB.
}

```
=# EXPLAIN
    SELECT sum(lo_extendedprice*lo_discount) as revenue
      FROM flineorder,date1
     WHERE lo_orderdate = d_datekey
       AND d_year = 1993
       AND lo_discount between 1 and 3
       AND lo_quantity < 25;
                                             QUERY PLAN
-----------------------------------------------------------------------------------------------------
 Aggregate  (cost=12632759.02..12632759.03 rows=1 width=32)
   ->  Custom Scan (GpuPreAgg)  (cost=12632754.43..12632757.49 rows=204 width=8)
         Reduction: NoGroup
         Combined GpuJoin: enabled
         GPU Preference: GPU0 (Tesla V100-PCIE-16GB)
         ->  Custom Scan (GpuJoin) on flineorder  (cost=9952.15..12638126.98 rows=572635 width=12)
               Outer Scan: flineorder  (cost=9877.70..12649677.69 rows=4010017 width=16)
               Outer Scan Filter: ((lo_discount >= 1) AND (lo_discount <= 3) AND (lo_quantity < 25))
               Depth 1: GpuHashJoin  (nrows 4010017...572635)
                        HashKeys: flineorder.lo_orderdate
                        JoinQuals: (flineorder.lo_orderdate = date1.d_datekey)
                        KDS-Hash (size: 66.06KB)
               GPU Preference: GPU0 (Tesla V100-PCIE-16GB)
               NVMe-Strom: enabled
               referenced: lo_orderdate, lo_quantity, lo_extendedprice, lo_discount
               files0: /opt/nvme/lineorder_s401.arrow (size: 309.23GB)
               ->  Seq Scan on date1  (cost=0.00..78.95 rows=365 width=4)
                     Filter: (d_year = 1993)
(18 rows)
```

@ja{
これを見るとCustom Scan (GpuJoin)が`flineorder`外部テーブルをスキャンしている事がわかります。 `file0`には外部テーブルの背後にあるファイル名`/opt/nvme/lineorder_s401.arrow`とそのサイズが表示されます。複数のファイルがマップされている場合には、`file1`、`file2`、... と各ファイル毎に表示されます。 `referenced`には実際に参照されている列の一覧が列挙されており、このクエリにおいては`lo_orderdate`、`lo_quantity`、`lo_extendedprice`および`lo_discount`列が参照されている事がわかります。
}
@en{
According to the `EXPLAIN` output, we can see Custom Scan (GpuJoin) scans `flineorder` foreign table. `file0` item shows the filename (`/opt/nvme/lineorder_s401.arrow`) on behalf of the foreign table and its size. If multiple files are mapped, any files are individually shown, like `file1`, `file2`, ... The `referenced` item shows the list of referenced columns. We can see this query touches `lo_orderdate`, `lo_quantity`, `lo_extendedprice` and `lo_discount` columns.
}

@ja{
また、`GPU Preference: GPU0 (Tesla V100-PCIE-16GB)`および`NVMe-Strom: enabled`の表示がある事から、`flineorder`のスキャンにはSSD-to-GPUダイレクトSQL機構が用いられることが分かります。
}
@en{
In addition, `GPU Preference: GPU0 (Tesla V100-PCIE-16GB)` and `NVMe-Strom: enabled` shows us the scan on `flineorder` uses SSD-to-GPU Direct SQL mechanism.
}

@ja{
VERBOSEオプションを付与する事で、より詳細な情報が出力されます。
}
@en{
VERBOSE option outputs more detailed information.
}

```
=# EXPLAIN VERBOSE
    SELECT sum(lo_extendedprice*lo_discount) as revenue
      FROM flineorder,date1
     WHERE lo_orderdate = d_datekey
       AND d_year = 1993
       AND lo_discount between 1 and 3
       AND lo_quantity < 25;
                              QUERY PLAN
--------------------------------------------------------------------------------
 Aggregate  (cost=12632759.02..12632759.03 rows=1 width=32)
   Output: sum((pgstrom.psum((flineorder.lo_extendedprice * flineorder.lo_discount))))
   ->  Custom Scan (GpuPreAgg)  (cost=12632754.43..12632757.49 rows=204 width=8)
         Output: (pgstrom.psum((flineorder.lo_extendedprice * flineorder.lo_discount)))
         Reduction: NoGroup
         GPU Projection: flineorder.lo_extendedprice, flineorder.lo_discount, pgstrom.psum((flineorder.lo_extendedprice * flineorder.lo_discount))
         Combined GpuJoin: enabled
         GPU Preference: GPU0 (Tesla V100-PCIE-16GB)
         ->  Custom Scan (GpuJoin) on public.flineorder  (cost=9952.15..12638126.98 rows=572635 width=12)
               Output: flineorder.lo_extendedprice, flineorder.lo_discount
               GPU Projection: flineorder.lo_extendedprice::bigint, flineorder.lo_discount::integer
               Outer Scan: public.flineorder  (cost=9877.70..12649677.69 rows=4010017 width=16)
               Outer Scan Filter: ((flineorder.lo_discount >= 1) AND (flineorder.lo_discount <= 3) AND (flineorder.lo_quantity < 25))
               Depth 1: GpuHashJoin  (nrows 4010017...572635)
                        HashKeys: flineorder.lo_orderdate
                        JoinQuals: (flineorder.lo_orderdate = date1.d_datekey)
                        KDS-Hash (size: 66.06KB)
               GPU Preference: GPU0 (Tesla V100-PCIE-16GB)
               NVMe-Strom: enabled
               referenced: lo_orderdate, lo_quantity, lo_extendedprice, lo_discount
               files0: /opt/nvme/lineorder_s401.arrow (size: 309.23GB)
                 lo_orderpriority: 33.61GB
                 lo_extendedprice: 17.93GB
                 lo_ordertotalprice: 17.93GB
                 lo_revenue: 17.93GB
               ->  Seq Scan on public.date1  (cost=0.00..78.95 rows=365 width=4)
                     Output: date1.d_datekey
                     Filter: (date1.d_year = 1993)
(28 rows)
```

@ja{
被参照列をロードする際に読み出すべき列データの大きさを、列ごとに表示しています。 `lo_orderdate`、`lo_quantity`、`lo_extendedprice`および`lo_discount`列のロードには合計で87.4GBの読み出しが必要で、これはファイルサイズ309.2GBの28.3%に相当します。
}
@en{
The verbose output additionally displays amount of column-data to be loaded on reference of columns. The load of `lo_orderdate`, `lo_quantity`, `lo_extendedprice` and `lo_discount` columns needs to read 87.4GB in total. It is 28.3% towards the filesize (309.2GB).
}

@ja:#Arrowファイルの作成方法
@en:#How to make Arrow files

@ja{
本節では、既にPostgreSQLデータベースに格納されているデータをApache Arrow形式に変換する方法を説明します。
}
@en{
This section introduces the way to transform dataset already stored in PostgreSQL database system into Apache Arrow file.
}

@ja:##PyArrow+Pandas
@en:##Using PyArrow+Pandas

@ja{
Arrow開発者コミュニティが開発を行っている PyArrow モジュールとPandasデータフレームの組合せを用いて、PostgreSQLデータベースの内容をArrow形式ファイルへと書き出す事ができます。

以下の例は、テーブルt0に格納されたデータを全て読込み、ファイル/tmp/t0.arrowへと書き出すというものです。
}
@en{
A pair of PyArrow module, developed by Arrow developers community, and Pandas data frame can dump PostgreSQL database into an Arrow file.

The example below reads all the data in table `t0`, then write out them into `/tmp/t0.arrow`.
}
```
import pyarrow as pa
import pandas as pd

X = pd.read_sql(sql="SELECT * FROM t0", con="postgresql://localhost/postgres")
Y = pa.Table.from_pandas(X)
f = pa.RecordBatchFileWriter('/tmp/t0.arrow', Y.schema)
f.write_table(Y,1000000)      # RecordBatch for each million rows
f.close()
```
@ja{
ただし上記の方法は、SQLを介してPostgreSQLから読み出したデータベースの内容を一度メモリに保持するため、大量の行を一度に変換する場合には注意が必要です。
}
@en{
Please note that the above operation once keeps query result of the SQL on memory, so should pay attention on memory consumption if you want to transfer massive rows at once.
}

@ja:##Pg2Arrow
@en:##Using Pg2Arrow

@ja{
一方、PG-Strom Development Teamが開発を行っている `pg2arrow` コマンドを使用して、PostgreSQLデータベースの内容をArrow形式ファイルへと書き出す事ができます。 このツールは比較的大量のデータをNVME-SSDなどストレージに書き出す事を念頭に設計されており、PostgreSQLデータベースから`-s|--segment-size`オプションで指定したサイズのデータを読み出すたびに、Arrow形式のレコードバッチ（Record Batch）としてファイルに書き出します。そのため、メモリ消費量は比較的リーズナブルな値となります。

`pg2arrow`コマンドはPG-Stromに同梱されており、PostgreSQL関連コマンドのインストール先ディレクトリに格納されます。
}
@en{
On the other hand, `pg2arrow` command, developed by PG-Strom Development Team, enables us to write out query result into Arrow file. This tool is designed to write out massive amount of data into storage device like NVME-SSD. It fetch query results from PostgreSQL database system, and write out Record Batches of Arrow format for each data size specified by the `-s|--segment-size` option. Thus, its memory consumption is relatively reasonable.

`pg2arrow` command is distributed with PG-Strom. It shall be installed on the `bin` directory of PostgreSQL related utilities.
}

```
$ pg2arrow --help
Usage:
  pg2arrow [OPTION]... [DBNAME [USERNAME]]

General options:
  -d, --dbname=DBNAME     database name to connect to
  -c, --command=COMMAND   SQL command to run
  -f, --file=FILENAME     SQL command from file
      (-c and -f are exclusive, either of them must be specified)
  -o, --output=FILENAME   result file in Apache Arrow format
      (default creates a temporary file)

Arrow format options:
  -s, --segment-size=SIZE size of record batch for each
      (default: 256MB)

Connection options:
  -h, --host=HOSTNAME     database server host
  -p, --port=PORT         database server port
  -U, --username=USERNAME database user name
  -w, --no-password       never prompt for password
  -W, --password          force password prompt

Debug options:
      --dump=FILENAME     dump information of arrow file
      --progress          shows progress of the job.

Report bugs to <pgstrom@heterodbcom>.
```
@ja{
PostgreSQLへの接続パラメータはpsqlやpg_dumpと同様に、`-h`や`-U`などのオプションで指定します。 基本的なコマンドの使用方法は、`-c|--command`オプションで指定したSQLをPostgreSQL上で実行し、その結果を`-o|--output`で指定したファイルへArrow形式で書き出します。
}
@en{
The `-h` or `-U` option specifies the connection parameters of PostgreSQL, like `psql` or `pg_dump`. The simplest usage of this command is running a SQL command specified by `-c|--command` option on PostgreSQL server, then write out results into the file specified by `-o|--output` option in Arrow format.
}

@ja{
以下の例は、テーブル`t0`に格納されたデータを全て読込み、ファイル`/tmp/t0.arrow`へと書き出すというものです。
}
@en{
The example below reads all the data in table `t0`, then write out them into the file `/tmp/t0.arrow`.
}
```
$ pg2arrow -U kaigai -d postgres -c "SELECT * FROM t0" -o /tmp/t0.arrow
```

@ja{
開発者向けオプションですが、`--dump <filename>`でArrow形式ファイルのスキーマ定義やレコードバッチの位置とサイズを可読な形式で出力する事もできます。
}
@en{
Although it is an option for developers, `--dump <filename>` prints schema definition and record-batch location and size of Arrow file in human readable form.
}

@ja:#先進的な使い方
@en:#Advanced Usage


@ja:##SSDtoGPUダイレクトSQL
@en:##SSDtoGPU Direct SQL

@ja{
Arrow_Fdw外部テーブルにマップされた全てのArrow形式ファイルが以下の条件を満たす場合には、列データの読み出しにSSD-to-GPUダイレクトSQLを使用する事ができます。

- Arrow形式ファイルがNVME-SSD区画上に置かれている。
- NVME-SSD区画はExt4ファイルシステムで構築されている。
- Arrow形式ファイルの総計が`pg_strom.nvme_strom_threshold`設定を上回っている。
}
@en{
In case when all the Arrow files mapped on the Arrow_Fdw foreign table satisfies the terms below, PG-Strom enables SSD-to-GPU Direct SQL to load columnar data.

- Arrow files are on NVME-SSD volume.
- NVME-SSD volume is managed by Ext4 filesystem.
- Total size of Arrow files exceeds the `pg_strom.nvme_strom_threshold` configuration.
}

@ja:##パーティション設定
@en:##Partition configuration

@ja{
Arrow_Fdw外部テーブルを、パーティションの一部として利用する事ができます。 通常のPostgreSQLテーブルと混在する事も可能ですが、Arrow_Fdw外部テーブルは書き込みに対応していない事に注意してください。 また、マップされたArrow形式ファイルに含まれるデータは、パーティションの境界条件と矛盾しないように設定してください。これはデータベース管理者の責任です。
}
@en{
Arrow_Fdw foreign tables can be used as a part of partition leafs. Usual PostgreSQL tables can be mixtured with Arrow_Fdw foreign tables. So, pay attention Arrow_Fdw foreign table does not support any writer operations. And, make boundary condition of the partition consistent to the contents of the mapped Arrow file. It is a responsibility of the database administrators.
}

![Example of partition configuration](./img/partition-logdata.png)

@ja{
典型的な利用シーンは、長期間にわたり蓄積したログデータの処理です。

トランザクションデータと異なり、一般的にログデータは一度記録されたらその後更新削除されることはありません。 したがって、一定期間が経過したログデータは、読み出し専用ではあるものの集計処理が高速なArrow_Fdw外部テーブルに移し替えることで、集計・解析ワークロードの処理効率を引き上げる事が可能となります。また、ログデータにはほぼ間違いなくタイムスタンプが付与されている事から、月単位、週単位など、一定期間ごとにパーティション子テーブルを追加する事が可能です。
}
@en{
A typical usage scenario is processing of long-standing accumulated log-data.

Unlike transactional data, log-data is mostly write-once and will never be updated / deleted. Thus, by migration of the log-data after a lapse of certain period into Arrow_Fdw foreign table that is read-only but rapid processing, we can accelerate summarizing and analytics workloads. In addition, log-data likely have timestamp, so it is quite easy design to add partition leafs periodically, like monthly, weekly or others.
}

@ja{
以下の例は、PostgreSQLテーブルとArrow_Fdw外部テーブルを混在させたパーティションテーブルを定義したものです。
}
@en{
The example below defines a partitioned table that mixes a normal PostgreSQL table and Arrow_Fdw foreign tables.
}

@ja{
書き込みが可能なPostgreSQLテーブルをデフォルトパーティションとして指定しておく[^2]事で、一定期間の経過後、DB運用を継続しながら過去のログデータだけをArrow_Fdw外部テーブルへ移す事が可能です。

[^2]: PostgreSQL v11以降で対応
}
@en{
The normal PostgreSQL table, is read-writable, is specified as default partition[^2], so DBA can migrate only past log-data into Arrow_Fdw foreign table under the database system operations.

[^2]: Supported at PostgreSQL v11 or later. 
}

```
CREATE TABLE lineorder (
    lo_orderkey numeric,
    lo_linenumber integer,
    lo_custkey numeric,
    lo_partkey integer,
    lo_suppkey numeric,
    lo_orderdate integer,
    lo_orderpriority character(15),
    lo_shippriority character(1),
    lo_quantity numeric,
    lo_extendedprice numeric,
    lo_ordertotalprice numeric,
    lo_discount numeric,
    lo_revenue numeric,
    lo_supplycost numeric,
    lo_tax numeric,
    lo_commit_date character(8),
    lo_shipmode character(10)
) PARTITION BY RANGE (lo_orderdate);

CREATE TABLE lineorder__now PARTITION OF lineorder default;

CREATE FOREIGN TABLE lineorder__1993 PARTITION OF lineorder
   FOR VALUES FROM (19930101) TO (19940101)
SERVER arrow_fdw OPTIONS (file '/opt/tmp/lineorder_1993.arrow');

CREATE FOREIGN TABLE lineorder__1994 PARTITION OF lineorder
   FOR VALUES FROM (19940101) TO (19950101)
SERVER arrow_fdw OPTIONS (file '/opt/tmp/lineorder_1994.arrow');

CREATE FOREIGN TABLE lineorder__1995 PARTITION OF lineorder
   FOR VALUES FROM (19950101) TO (19960101)
SERVER arrow_fdw OPTIONS (file '/opt/tmp/lineorder_1995.arrow');

CREATE FOREIGN TABLE lineorder__1996 PARTITION OF lineorder
   FOR VALUES FROM (19960101) TO (19970101)
SERVER arrow_fdw OPTIONS (file '/opt/tmp/lineorder_1996.arrow');
```

@ja{
このテーブルに対する問い合わせの実行計画は以下のようになります。 検索条件`lo_orderdate between 19950701 and 19960630`がパーティションの境界条件を含んでいる事から、子テーブル`lineorder__1993`と`lineorder__1994`は検索対象から排除され、他のテーブルだけを読み出すよう実行計画が作られています。
}
@en{
Below is the query execution plan towards the table. By the query condition `lo_orderdate between 19950701 and 19960630` that touches boundary condition of the partition, the partition leaf `lineorder__1993` and `lineorder__1994` are pruned, so it makes a query execution plan to read other (foreign) tables only.
}

```
=# EXPLAIN
    SELECT sum(lo_extendedprice*lo_discount) as revenue
      FROM lineorder,date1
     WHERE lo_orderdate = d_datekey
       AND lo_orderdate between 19950701 and 19960630
       AND lo_discount between 1 and 3
       ABD lo_quantity < 25;

                                 QUERY PLAN
--------------------------------------------------------------------------------
 Aggregate  (cost=172088.90..172088.91 rows=1 width=32)
   ->  Hash Join  (cost=10548.86..172088.51 rows=77 width=64)
         Hash Cond: (lineorder__1995.lo_orderdate = date1.d_datekey)
         ->  Append  (cost=10444.35..171983.80 rows=77 width=67)
               ->  Custom Scan (GpuScan) on lineorder__1995  (cost=10444.35..33671.87 rows=38 width=68)
                     GPU Filter: ((lo_orderdate >= 19950701) AND (lo_orderdate <= 19960630) AND
                                  (lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND
                                  (lo_quantity < '25'::numeric))
                     referenced: lo_orderdate, lo_quantity, lo_extendedprice, lo_discount
                     files0: /opt/tmp/lineorder_1995.arrow (size: 892.57MB)
               ->  Custom Scan (GpuScan) on lineorder__1996  (cost=10444.62..33849.21 rows=38 width=68)
                     GPU Filter: ((lo_orderdate >= 19950701) AND (lo_orderdate <= 19960630) AND
                                  (lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND
                                  (lo_quantity < '25'::numeric))
                     referenced: lo_orderdate, lo_quantity, lo_extendedprice, lo_discount
                     files0: /opt/tmp/lineorder_1996.arrow (size: 897.87MB)
               ->  Custom Scan (GpuScan) on lineorder__now  (cost=11561.33..104462.33 rows=1 width=18)
                     GPU Filter: ((lo_orderdate >= 19950701) AND (lo_orderdate <= 19960630) AND
                                  (lo_discount >= '1'::numeric) AND (lo_discount <= '3'::numeric) AND
                                  (lo_quantity < '25'::numeric))
         ->  Hash  (cost=72.56..72.56 rows=2556 width=4)
               ->  Seq Scan on date1  (cost=0.00..72.56 rows=2556 width=4)
(16 rows)

```

@ja{
この後、`lineorder__now`テーブルから1997年のデータを抜き出し、これをArrow_Fdw外部テーブル側に移すには以下の操作を行います
}
@en{
The operation below extracts the data in `1997` from `lineorder__now` table, then move to a new Arrow_Fdw foreign table.
}

```
$ pg2arrow -d sample  -o /opt/tmp/lineorder_1997.arrow \
           -c "SELECT * FROM lineorder WHERE lo_orderdate between 19970101 and 19971231"
```

@ja{
`pg2arrow`コマンドにより、`lineorder`テーブルから1997年のデータだけを抜き出して、新しいArrow形式ファイルへ書き出します。
}
@en{
`pg2arrow` command extracts the data in 1997 from the `lineorder` table into a new Arrow file.}

```
BEGIN;
--
-- remove rows in 1997 from the read-writable table
--
DELETE FROM lineorder WHERE lo_orderdate BETWEEN 19970101 AND 19971231;
--
-- define a new partition leaf which maps log-data in 1997
--
CREATE FOREIGN TABLE lineorder__1997 PARTITION OF lineorder
   FOR VALUES FROM (19970101) TO (19980101)
SERVER arrow_fdw OPTIONS (file '/opt/tmp/lineorder_1997.arrow');

COMMIT;
```

@ja{
この操作により、PostgreSQLテーブルである`lineorder__now`から1997年のデータを削除し、代わりに同一内容のArrow形式ファイル`/opt/tmp/lineorder_1997.arrow`を外部テーブル`lineorder__1997`としてマップしました。
}
@en{
A series of operations above delete the data in 1997 from `lineorder__new` that is a PostgreSQL table, then maps an Arrow file (`/opt/tmp/lineorder_1997.arrow`) which contains an identical contents as a foreign table `lineorder__1997`.
}
