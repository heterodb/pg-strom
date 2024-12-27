@ja:#Apache Arrow (列指向データストア)
@en:#Apache Arrow (Columnar Store)

@ja:##概要
@en:##Overview

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
Data representation in Apache Arrow is not identical with the representation in PostgreSQL. For example, epoch of timestamp in Arrow is `1970-01-01` and it supports multiple precision. In contrast, epoch of timestamp in PostgreSQL is `2001-01-01` and it has microseconds accuracy.
}

@ja{
Arrow_Fdwは外部テーブルを用いてApache Arrow形式ファイルをPostgreSQL上で読み出す事を可能にします。例えば、列ごとに100万件の列データが存在するレコードバッチを8個内包するArrow形式ファイルをArrow_Fdwを用いてマップした場合、この外部テーブルを介してArrowファイル上の800万件のデータへアクセスする事ができるようになります。
}
@en{
Arrow_Fdw allows to read Apache Arrow files on PostgreSQL using foreign table mechanism. If an Arrow file contains 8 of record batches that has million items for each column data, for example, we can access 8 million rows on the Arrow files through the foreign table.
}

@ja:##運用
@en:##Operations

@ja:###外部テーブルの定義
@en:###Creation of foreign tables

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

@ja:###外部テーブルオプション
@en:###Foreign table options

@ja{
Arrow_Fdwは以下のオプションに対応しています。

####外部テーブルに対するオプション

`file=PATHNAME`
:   外部テーブルにマップするArrowファイルを1個指定します。

`files=PATHNAME1[,PATHNAME2...]`
:   外部テーブルにマップするArrowファイルをカンマ(,）区切りで複数指定します。

`dir=DIRNAME`
:   指定したディレクトリに格納されている全てのファイルを外部テーブルにマップします。

`suffix=SUFFIX`
:   `dir`オプションの指定時、例えば`.arrow`など、特定の接尾句を持つファイルだけをマップします。

`parallel_workers=N_WORKERS`
:   この外部テーブルの並列スキャンに使用する並列ワーカープロセスの数を指定します。一般的なテーブルにおける`parallel_workers`ストレージパラメータと同等の意味を持ちます。

`pattern=PATTERN`
:   `file`、`files`、または`dir`オプションで指定されたファイルのうち、ワイルドカードを含む`PATTERN`にマッチしたものだけを外部テーブルにマップします。
:   ワイルドカードには以下のものを利用することができます。
:   - `?` ... 任意の1文字にマッチする。
:   - `*` ... 任意の0文字以上の文字列にマッチする。
:   - `${KEY}` ... 任意の0文字以上の文字列にマッチする。
:   - `@\{KEY}` ... 任意の0文字以上の数値列にマッチする。
:   
:   このオプションには面白い使い方があり、ワイルドカードの`${KEY}`や`@\{KEY}`でマッチしたファイル名の一部分を、仮想列として参照することができます。詳しくは、'''Arrow_Fdwの仮想列'''を参照してください。

####カラムに対するオプション

`field=FIELD`
:   そのカラムにマップするArrowファイルのフィールド名を指定します。
:   デフォルトでは、この外部テーブルの列名と同じフィールドのうち、最も最初に出現したフィールドをマップします。

`virtual=KEY`
:   そのカラムが仮想列である事を指定します。`KEY`はテーブルオプションの`pattern`オプションで指定されたパターン中のワイルドカードのキー名を指定します。
:   仮想列はファイル名パターンのうち`KEY`にマッチした部分をクエリで参照することができます。

`virtual_metadata=KEY`
:   そのカラムが仮想列である事を指定します。`KEY`はArrowファイルのCustomMetadataフィールドに埋め込まれたKEY-VALUEペアを指定します。指定したKEY-VALUEペアが見つからない場合、このカラムはNULL値を返します。
:   ArrowファイルのCustomMetadataには、スキーマ（PostgreSQLのテーブルに相当）に埋め込まれるものと、フィールド（PostgreSQLの列に相当）に埋め込まれるものの二種類があります。
:   例えば、`lo_orderdate.max_values`のように、KEY値の前に`.`文字で区切られたフィールド名を記述する事で、フィールドに埋め込まれたCustomMetadataを参照する事が出来ます。フィールド名がない場合は、スキーマに埋め込まれたKEY-VALUEペアであるとして扱われます。

`virtual_metadata_split=KEY`
:   そのカラムが仮想列である事を指定します。`KEY`はArrowファイルのCustomMetadataフィールドに埋め込まれたKEY-VALUEペアを指定します。指定したKEY-VALUEペアが見つからない場合、このカラムはNULL値を返します。
:    `virtual_metadata`との違いは、CustomMetadataフィールドの値をデリミタ（`,`）で区切り、それを個々のRecord Batchに先頭から順に当てはめて行くことです。例えば、指定したCustomMetadataの値が`Tokyo,Osaka,Kyoto,Yokohama`であった場合、RecordBatch-0から読み出した行では`'Tokyo'`が、RecordBatch-1から読み出した行では`'Osaka'`が、RecordBatch-2から読み出した行では`'Osaka'`がこの仮想列の値として表示されます。
}
@en{
Arrow_Fdw supports the options below.

####Foreign Table Options

`file=PATHNAME`
:   It maps an Arrow file specified on the foreign table.

`files=PATHNAME1[,PATHNAME2...]`
:   It maps multiple Arrow files specified by comma (,) separated files list on the foreign table.

`dir=DIRNAME`
:   It maps all the Arrow files in the directory specified on the foreign table.

`suffix=SUFFIX`
:   `When `dir` option is given, it maps only files with the specified suffix, like `.arrow` for example.

`parallel_workers=N_WORKERS`
:   It tells the number of workers that should be used to assist a parallel scan of this foreign table; equivalent to `parallel_workers` storage parameter at normal tables.

`pattern=PATTERN`
:   Maps only files specified by the `file`, `files`, or `dir` option that match the `PATTERN`, including wildcards, to the foreign table.
:   The following wildcards can be used:
:   - `?` ... matches any 1 character.
:   - `*` ... matches any string of 0 or more characters.
:   - `${KEY}` ... matches any string of 0 or more characters.
:   - `@\{KEY}` ... matches any numeric string of 0 or more characters.
:   
:   An interesting use of this option is to refer to a portion of a file name matched by the wildcard `${KEY}` or `@\{KEY}` as a virtual column. For more information, see the '''Arrow_Fdw virtual column''' section below.

####Foreign Column Options

`field=FIELD`
:   It specifies the field name of the Arrow file to map to that column.
:   In the default, Arrow_Fdw maps the first occurrence of a field that has the same column name as this foreign table's column name.

`virtual=KEY`
:   It configures the column is a virtual column. `KEY` specifies the wildcard key name in the pattern specified by the `pattern` option of the foreign table option.
:   A virtual column allows to refer to the part of the file name pattern that matches `KEY` in a query.

`virtual_metadata=KEY`
:   It specifies that the column is a virtual column. `KEY` specifies a KEY-VALUE pair embedded in the CustomMetadata field of the Arrow file. If the specified KEY-VALUE pair is not found, the column returns a NULL value.
:   There are two types of CustomMetadata in Arrow files: embedded in the schema (corresponding to a PostgreSQL table) and embedded in the field (corresponding to a PostgreSQL column).
:   For example, you can reference CustomMetadata embedded in a field by writing the field name separated by the `.` character before the KEY value, such as `lo_orderdate.max_values`. If there is no field name, it will be treated as a KEY-VALUE pair embedded in the schema.

`virtual_metadata_split=KEY`
:   It specifies that the column is a virtual column. `KEY` specifies the KEY-VALUE pair embedded in the CustomMetadata field of the Arrow file. If the specified KEY-VALUE pair is not found, this column returns a NULL value.
:   The difference from `virtual_metadata` is that the values of the CustomMetadata field are separated by a delimiter(`,`) and applied to each Record Batch in order from the beginning. For example, if the specified CustomMetadata value is `Tokyo,Osaka,Kyoto,Yokohama`, the row read from RecordBatch-0 will display `'Tokyo'`, the row read from RecordBatch-1 will display `'Osaka'`, and the row read from RecordBatch-2 will display `'Osaka'` as the value of this virtual column.
}

@ja:###データ型の対応
@en:###Data type mapping

@ja{
Arrow形式のデータ型と、PostgreSQLのデータ型は以下のように対応しています。

`Int`
:   `bitWidth`属性の値に応じて、それぞれ`int1`、`int2`、`int4`、`int8`のいずれかに対応。
:   `is_signed`属性の値は無視されます。
:   `int1`はPG-Stromによる独自拡張

`FloatingPoint`
:   `precision`属性の値に応じて、それぞれ`float2`、`float4`、`float8`のいずれかに対応。
:   `float2`はPG-Stromによる独自拡張

`Utf8`, `LargeUtf8`
:   `text`型に対応

`Binary`, `LargeBinary`
:   `bytea`型に対応

`Decimal`
:   `numeric`型に対応

`Date`
:   `date`型に対応。`unit=Day`相当となるように補正される。

`Time`
:   `time`型に対応。`unit=MicroSecond`相当になるように補正される。

`Timestamp`
:   `timestamp`型に対応。`unit=MicroSecond`相当になるように補正される。

`Interval`
:   `interval`型に対応

`List`, `LargeList`
:   要素型の1次元配列型として表現される。

`Struct`
:   複合型として表現される。対応する複合型は予め定義されていなければならない。

`FixedSizeBinary`
:   `byteWidth`属性の値に応じて `char(n)` として表現される。
:   メタデータ `pg_type=TYPENAME` が指定されている場合、該当するデータ型を割り当てる場合がある。現時点では、`inet`および`macaddr`型。

`Union`、`Map`、`Duration`
:   現時点ではPostgreSQLデータ型への対応はなし。
}
@en{
Arrow data types are mapped on PostgreSQL data types as follows.

`Int`
:   mapped to either of `int1`, `int2`, `int4` or `int8` according to the `bitWidth` attribute.
:   `is_signed` attribute shall be ignored.
:   `int1` is an enhanced data type by PG-Strom.

`FloatingPoint`
:   mapped to either of `float2`, `float4` or `float8` according to the `precision` attribute.
:   `float2` is an enhanced data type by PG-Strom.

`Utf8`, `LargeUtf8`
:   mapped to `text` data type

`Binary`, `LargeBinary`
:   mapped to `bytea` data type

`Decimal`
:   mapped to `numeric` data type

`Date`
:   mapped to `date` data type; to be adjusted as if it has `unit=Day` precision.

`Time`
:   mapped to `time` data type; to be adjusted as if it has `unit=MicroSecond` precision.

`Timestamp`
:   mapped to `timestamp` data type; to be adjusted as if it has `unit=MicroSecond` precision.

`Interval`
:   mapped to `interval` data type.

`List`, `LargeList`
:   mapped to 1-dimensional array of the element data type.

`Struct`
:   mapped to compatible composite data type; that shall be defined preliminary.

`FixedSizeBinary`
:   mapped to `char(n)` data type according to the `byteWidth` attribute.
:   If `pg_type=TYPENAME` is configured, PG-Strom may assign the configured data type. Right now, `inet` and `macaddr` are supported.

`Union`, `Map`, `Duration`
:   Right now, PG-Strom cannot map these Arrow data types onto any of PostgreSQL data types.
}

@ja:###EXPLAIN出力の読み方
@en:###How to read EXPLAIN

@ja{
`EXPLAIN`コマンドを用いて、Arrow形式ファイルの読み出しに関する情報を出力する事ができます。

以下の例は、約503GBの大きさを持つArrow形式ファイルをマップしたf_lineorder外部テーブルを含むクエリ実行計画の出力です。
}
@en{
`EXPLAIN` command show us information about Arrow files reading.

The example below is an output of query execution plan that includes f_lineorder foreign table that mapps an Arrow file of 503GB.
}

```
=# EXPLAIN
    SELECT sum(lo_extendedprice*lo_discount) as revenue
      FROM f_lineorder,date1
     WHERE lo_orderdate = d_datekey
       AND d_year = 1993
       AND lo_discount between 1 and 3
       AND lo_quantity < 25;
                                        QUERY PLAN
--------------------------------------------------------------------------------
 Aggregate  (cost=14535261.08..14535261.09 rows=1 width=8)
   ->  Custom Scan (GpuPreAgg) on f_lineorder  (cost=14535261.06..14535261.07 rows=1 width=32)
         GPU Projection: pgstrom.psum(((f_lineorder.lo_extendedprice * f_lineorder.lo_discount))::bigint)
         GPU Scan Quals: ((f_lineorder.lo_discount >= 1) AND (f_lineorder.lo_discount <= 3) AND (f_lineorder.lo_quantity < 25)) [rows: 5999990000 -> 9999983]
         GPU Join Quals [1]: (f_lineorder.lo_orderdate = date1.d_datekey) ... [nrows: 9999983 -> 1428010]
         GPU Outer Hash [1]: f_lineorder.lo_orderdate
         GPU Inner Hash [1]: date1.d_datekey
         referenced: lo_orderdate, lo_quantity, lo_extendedprice, lo_discount
         file0: /opt/nvme/f_lineorder_s999.arrow (read: 89.41GB, size: 502.92GB)
         GPU-Direct SQL: enabled (GPU-0)
         ->  Seq Scan on date1  (cost=0.00..78.95 rows=365 width=4)
               Filter: (d_year = 1993)
(12 rows)
```

@ja{
これを見るとCustom Scan (GpuPreAgg)が`f_lineorder`外部テーブルをスキャンしている事がわかります。 `file0`には外部テーブルの背後にあるファイル名`/opt/nvme/f_lineorder_s999.arrow`とそのサイズが表示されます。複数のファイルがマップされている場合には、`file1`、`file2`、... と各ファイル毎に表示されます。 `referenced`には実際に参照されている列の一覧が列挙されており、このクエリにおいては`lo_orderdate`、`lo_quantity`、`lo_extendedprice`および`lo_discount`列が参照されている事がわかります。
}
@en{
According to the `EXPLAIN` output, we can see Custom Scan (GpuPreAgg) scans `f_lineorder` foreign table. `file0` item shows the filename (`/opt/nvme/lineorder_s999.arrow`) on behalf of the foreign table and its size. If multiple files are mapped, any files are individually shown, like `file1`, `file2`, ... The `referenced` item shows the list of referenced columns. We can see this query touches `lo_orderdate`, `lo_quantity`, `lo_extendedprice` and `lo_discount` columns.
}

@ja{
また、`GPU-Direct SQL: enabled (GPU-0)`の表示がある事から、`f_lineorder`のスキャンにはGPU-Direct SQL機構が用いられることが分かります。
}
@en{
In addition, `GPU-Direct SQL: enabled (GPU-0)` shows us the scan on `f_lineorder` uses GPU-Direct SQL mechanism.
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
      FROM f_lineorder,date1
     WHERE lo_orderdate = d_datekey
       AND d_year = 1993
       AND lo_discount between 1 and 3
       AND lo_quantity < 25;
                                        QUERY PLAN
--------------------------------------------------------------------------------
 Aggregate  (cost=14535261.08..14535261.09 rows=1 width=8)
   Output: pgstrom.sum_int((pgstrom.psum(((f_lineorder.lo_extendedprice * f_lineorder.lo_discount))::bigint)))
   ->  Custom Scan (GpuPreAgg) on public.f_lineorder  (cost=14535261.06..14535261.07 rows=1 width=32)
         Output: (pgstrom.psum(((f_lineorder.lo_extendedprice * f_lineorder.lo_discount))::bigint))
         GPU Projection: pgstrom.psum(((f_lineorder.lo_extendedprice * f_lineorder.lo_discount))::bigint)
         GPU Scan Quals: ((f_lineorder.lo_discount >= 1) AND (f_lineorder.lo_discount <= 3) AND (f_lineorder.lo_quantity < 25)) [rows: 5999990000 -> 9999983]
         GPU Join Quals [1]: (f_lineorder.lo_orderdate = date1.d_datekey) ... [nrows: 9999983 -> 1428010]
         GPU Outer Hash [1]: f_lineorder.lo_orderdate
         GPU Inner Hash [1]: date1.d_datekey
         referenced: lo_orderdate, lo_quantity, lo_extendedprice, lo_discount
         file0: /opt/nvme/f_lineorder_s999.arrow (read: 89.41GB, size: 502.92GB)
           lo_orderdate: 22.35GB
           lo_quantity: 22.35GB
           lo_extendedprice: 22.35GB
           lo_discount: 22.35GB
         GPU-Direct SQL: enabled (GPU-0)
         KVars-Slot: <slot=0, type='int4', expr='f_lineorder.lo_discount'>, <slot=1, type='int4', expr='f_lineorder.lo_quantity'>, <slot=2, type='int8', expr='(f_lineorder.lo_extendedprice * f_lineorder.lo_discount)'>, <slot=3, type='int4', expr='f_lineorder.lo_extendedprice'>, <slot=4, type='int4', expr='f_lineorder.lo_orderdate'>, <slot=5, type='int4', expr='date1.d_datekey'>
         KVecs-Buffer: nbytes: 51200, ndims: 3, items=[kvec0=<0x0000-27ff, type='int4', expr='lo_discount'>, kvec1=<0x2800-4fff, type='int4', expr='lo_quantity'>, kvec2=<0x5000-77ff, type='int4', expr='lo_extendedprice'>, kvec3=<0x7800-9fff, type='int4', expr='lo_orderdate'>, kvec4=<0xa000-c7ff, type='int4', expr='d_datekey'>]
         LoadVars OpCode: {Packed items[0]={LoadVars(depth=0): kvars=[<slot=4, type='int4' resno=6(lo_orderdate)>, <slot=1, type='int4' resno=9(lo_quantity)>, <slot=3, type='int4' resno=10(lo_extendedprice)>, <slot=0, type='int4' resno=12(lo_discount)>]}, items[1]={LoadVars(depth=1): kvars=[<slot=5, type='int4' resno=1(d_datekey)>]}}
         MoveVars OpCode: {Packed items[0]={MoveVars(depth=0): items=[<slot=0, offset=0x0000-27ff, type='int4', expr='lo_discount'>, <slot=3, offset=0x5000-77ff, type='int4', expr='lo_extendedprice'>, <slot=4, offset=0x7800-9fff, type='int4', expr='lo_orderdate'>]}}, items[1]={MoveVars(depth=1): items=[<offset=0x0000-27ff, type='int4', expr='lo_discount'>, <offset=0x5000-77ff, type='int4', expr='lo_extendedprice'>]}}}
         Scan Quals OpCode: {Bool::AND args=[{Func(bool)::int4ge args=[{Var(int4): slot=0, expr='lo_discount'}, {Const(int4): value='1'}]}, {Func(bool)::int4le args=[{Var(int4): slot=0, expr='lo_discount'}, {Const(int4): value='3'}]}, {Func(bool)::int4lt args=[{Var(int4): slot=1, expr='lo_quantity'}, {Const(int4): value='25'}]}]}
         Join Quals OpCode: {Packed items[1]={JoinQuals:  {Func(bool)::int4eq args=[{Var(int4): kvec=0x7800-a000, expr='lo_orderdate'}, {Var(int4): slot=5, expr='d_datekey'}]}}}
         Join HashValue OpCode: {Packed items[1]={HashValue arg={Var(int4): kvec=0x7800-a000, expr='lo_orderdate'}}}
         Partial Aggregation OpCode: {AggFuncs <psum::int[slot=2, expr='(lo_extendedprice * lo_discount)']> arg={SaveExpr: <slot=2, type='int8'> arg={Func(int8)::int8 arg={Func(int4)::int4mul args=[{Var(int4): kvec=0x5000-7800, expr='lo_extendedprice'}, {Var(int4): kvec=0x0000-2800, expr='lo_discount'}]}}}}
         Partial Function BufSz: 16
         ->  Seq Scan on public.date1  (cost=0.00..78.95 rows=365 width=4)
               Output: date1.d_datekey
               Filter: (date1.d_year = 1993)
(28 rows)
```

@ja{
被参照列をロードする際に読み出すべき列データの大きさを、列ごとに表示しています。 `lo_orderdate`、`lo_quantity`、`lo_extendedprice`および`lo_discount`列のロードには合計で89.41GBの読み出しが必要で、これはファイルサイズ502.93GBの17.8%に相当します。
}
@en{
The verbose output additionally displays amount of column-data to be loaded on reference of columns. The load of `lo_orderdate`, `lo_quantity`, `lo_extendedprice` and `lo_discount` columns needs to read 89.41GB in total. It is 17.8% towards the filesize (502.93GB).
}

@ja:##Arrow_Fdwの仮想列
@en:##Arrow_Fdw Virtual Column

@ja{
Arrow_Fdwはスキーマ構造に互換性のある複数のApache Arrowを一個の外部テーブルにマッピングすることができます。例えば、外部テーブルオプションに`dir '/opt/arrow/mydata'`を指定すると、そのディレクトリ配下に存在する全てのファイルをマッピングするようになります。

トランザクショナルなデータベースの内容をApache Arrowファイルに変換するときに年月や特定のカテゴリ毎に分けてファイル化し、それらを反映したファイル名を付けて保存する事はしばしば行われています。

例えば、以下の例をご覧ください。トランザクショナルなテーブルである`lineorder`を`lo_orderdate`の年単位、および`lo_shipmode`のカテゴリ毎にArrowファイルへと変換しています。
}
@en{
Arrow_Fdw allows to map multiple Apache Arrow files with compatible schema structures to a single foreign table. For example, if `dir '/opt/arrow/mydata' is configured for foreign table option, all files under that directory will be mapped.

When you are converting the contents of a transactional database into an Apache Arrow file, we often dump them to separate files by year and month or specific categories, and its file names reflects these properties.

The example below shows an example to convert the transactional table `lineorder` into Arrow files by year of `lo_orderdate` and by category of `lo_shipmode`.
}

```
$ for s in RAIL AIR TRUCK SHIP FOB MAIL;
  do
    for y in 1993 1994 1995 1996 1997;
    do
      pg2arrow -d ssbm -c "SELECT * FROM lineorder_small \
                            WHERE lo_orderdate between ${y}0101 and ${y}1231 \
                              AND lo_shipmode = '${s}'" \
               -o /opt/arrow/mydata/f_lineorder_${y}_${s}.arrow
    done
  done
$ ls /opt/arrow/mydata/
f_lineorder_1993_AIR.arrow    f_lineorder_1995_RAIL.arrow
f_lineorder_1993_FOB.arrow    f_lineorder_1995_SHIP.arrow
f_lineorder_1993_MAIL.arrow   f_lineorder_1995_TRUCK.arrow
f_lineorder_1993_RAIL.arrow   f_lineorder_1996_AIR.arrow
f_lineorder_1993_SHIP.arrow   f_lineorder_1996_FOB.arrow
f_lineorder_1993_TRUCK.arrow  f_lineorder_1996_MAIL.arrow
f_lineorder_1994_AIR.arrow    f_lineorder_1996_RAIL.arrow
f_lineorder_1994_FOB.arrow    f_lineorder_1996_SHIP.arrow
f_lineorder_1994_MAIL.arrow   f_lineorder_1996_TRUCK.arrow
f_lineorder_1994_RAIL.arrow   f_lineorder_1997_AIR.arrow
f_lineorder_1994_SHIP.arrow   f_lineorder_1997_FOB.arrow
f_lineorder_1994_TRUCK.arrow  f_lineorder_1997_MAIL.arrow
f_lineorder_1995_AIR.arrow    f_lineorder_1997_RAIL.arrow
f_lineorder_1995_FOB.arrow    f_lineorder_1997_SHIP.arrow
f_lineorder_1995_MAIL.arrow   f_lineorder_1997_TRUCK.arrow
```

@ja{
これらのApache Arrowファイルは全て同じスキーマ構造を持っており、`dir`オプションを用いて1個の外部テーブルにマッピングできます。
また、データの生成時に絞り込みを行っているため、ファイル名に1995を含むファイルには`lo_orderdate`が19950101～19951231の範囲のレコードしか含まれておらず、ファイル名に`RAIL`を含むファイルには`lo_shipmode`が`RAIL`のレコードしか含まれていません。

つまり、これら複数のArrowファイルをマップしたArrow_Fdw外部テーブルを定義したとしても、ファイル名に1995を含むファイルからデータを読み出している時には、`lo_orderdate`の値が19950101～19951231の範囲であることが事前に分かっており、それを利用した最適化が可能です。

Arrow_Fdwでは、外部テーブルオプション`pattern`を使用する事でファイル名の一部を列として参照する事ができます。これを仮想列と呼び、以下のように設定します。
}
@en{
All these Apache Arrow files have the same schema structure and can be mapped to a single foreign table using the `dir` option.

Also, the Arrow file that has '1995' token in the file name only contains records with `lo_orderdate` in the range 19950101 to 19951231. The Arrow file that has 'RAIL' token in the file name only contains records with `lo_shipmode` of `RAIL`.

In other words, even if you define the Arrow_Fdw foreign  table that maps these multiple Arrow files, when reading data from a file whose file name includes 1995, it is assumed that the value of `lo_orderdate` is in the range of 19950101 to 19951231. It is possible for the optimizer to utilize this knowledge.

In Arrow_Fdw, you can refer to part of the file name as a column by using the foreign table option `pattern`. This is called a virtual column and is configured as follows.
}

```
=# IMPORT FOREIGN SCHEMA f_lineorder
     FROM SERVER arrow_fdw INTO public
  OPTIONS (dir '/opt/arrow/mydata', pattern 'f_lineorder_@\{year}_${shipping}.arrow');
IMPORT FOREIGN SCHEMA

=# \d f_lineorder
                             Foreign table "public.f_lineorder"
       Column       |     Type      | Collation | Nullable | Default |     FDW options
--------------------+---------------+-----------+----------+---------+----------------------
 lo_orderkey        | numeric       |           |          |         |
 lo_linenumber      | integer       |           |          |         |
 lo_custkey         | numeric       |           |          |         |
 lo_partkey         | integer       |           |          |         |
 lo_suppkey         | numeric       |           |          |         |
 lo_orderdate       | integer       |           |          |         |
 lo_orderpriority   | character(15) |           |          |         |
 lo_shippriority    | character(1)  |           |          |         |
 lo_quantity        | numeric       |           |          |         |
 lo_extendedprice   | numeric       |           |          |         |
 lo_ordertotalprice | numeric       |           |          |         |
 lo_discount        | numeric       |           |          |         |
 lo_revenue         | numeric       |           |          |         |
 lo_supplycost      | numeric       |           |          |         |
 lo_tax             | numeric       |           |          |         |
 lo_commit_date     | character(8)  |           |          |         |
 lo_shipmode        | character(10) |           |          |         |
 year               | bigint        |           |          |         | (virtual 'year')
 shipping           | text          |           |          |         | (virtual 'shipping')
Server: arrow_fdw
FDW options: (dir '/opt/arrow/mydata', pattern 'f_lineorder_@\{year}_${shipping}.arrow')
```

@ja{
この外部テーブルオプション`pattern`には2つのワイルドカードが含まれています。
0文字以上の数字列にマッチする`@\{year}`と、0文字以上の文字列にマッチする`${shipping}`です。
ファイル名のうち、この部分にマッチしたパターンは、それぞれ列オプションの`virtual`で指定した部分で参照することができます。
この場合、`IMPORT FOREIGN SCHEMA`が自動的に列定義を加え、Arrowファイル自体に含まれているフィールドに加えて、ワイルドカード`@\{year}`を参照する仮想列`year`（数値列であるため`bigint`データ型）と、`${shipping}`を参照する仮想列`shipping`を追加しています。

これらの仮想列に対応するフィールドはArrowファイルには存在しませんが、例えば、ファイル`f_lineorder_1994_AIR.arrow`から読みだした行を処理するときには`year`列の値は1994に、`shipping`列の値は'AIR'になるわけです。
}
@en{
This foreign table option `pattern` contains two wildcards.
`@\{year}` matches a numeric string larger than or equal to 0 characters, and `${shipping}` matches a string larger than or equal to 0 characters.
The patterns that match this part of the file name can be referenced in the part specified by the `virtual` column option.

In this case, `IMPORT FOREIGN SCHEMA` automatically adds column definitions, in addition to the fields contained in the Arrow file itself, as well as the virtual column `year` (a `bigint` column) that references the wildcard `@\{year}`, and the virtual column `shipping` that references the wildcard `${shipping}`.
}

```
=# SELECT lo_orderkey, lo_orderdate, lo_shipmode, year, shipping
     FROM f_lineorder
    WHERE year = 1995 AND shipping = 'AIR'
    LIMIT 10;
 lo_orderkey | lo_orderdate | lo_shipmode | year | shipping
-------------+--------------+-------------+------+----------
      637892 |     19950512 | AIR         | 1995 | AIR
      638243 |     19950930 | AIR         | 1995 | AIR
      638273 |     19951214 | AIR         | 1995 | AIR
      637443 |     19950805 | AIR         | 1995 | AIR
      637444 |     19950803 | AIR         | 1995 | AIR
      637510 |     19950831 | AIR         | 1995 | AIR
      637504 |     19950726 | AIR         | 1995 | AIR
      637863 |     19950802 | AIR         | 1995 | AIR
      637892 |     19950512 | AIR         | 1995 | AIR
      637987 |     19950211 | AIR         | 1995 | AIR
(10 rows)
```

@ja{
これは言い換えれば、Arrow_Fdw外部テーブルがマップしたArrowファイルを実際に読む前に、仮想列がどのような値になっているのかを知る事ができるという事です。この特徴を使えば、あるArrowファイルの読み出しの前に、検索条件から1件もマッチしない事が明らかである場合には、ファイルの読み出し自体をスキップする事が可能であるという事になります。

以下のクエリとその`EXPLAIN ANALYZE`出力をご覧ください。

この集計クエリは`f_lineorder`外部テーブルを読み出し、いくつかの条件で絞り込んだ後、`lo_extendedprice * lo_discount`の合計値を集計します。
その時、`WHERE year = 1994`という条件句が付加されています。これは実質的には`WHERE lo_orderdate BETWEEN 19940101 AND 19942131`と同じですが、`year`は仮想列であるため、Arrowファイルを読み出す前にマッチする行が存在するかどうかを判定する事ができます。

実際、`Stats-Hint:`行を見ると、`(year = 1994)`という条件によって12個のRecord-Batchがロードされたものの、48個のRecord-Batchはスキップされています。これは単純ですがI/Oの負荷を軽減する手段として極めて有効です。
}
@en{
In other words, you can know what values the virtual columns have before reading the Arrow file mapped by the Arrow_Fdw foreign table. By this feature, if it is obvious that there is no match at all from the search conditions before reading a certain Arrow file, it is possible to skip reading the file itself.

See the query and its `EXPLAIN ANALYZE` output below.

This aggregation query reads the `f_lineorder` foreign table, filters it by some conditions, and then aggregates the total value of `lo_extendedprice * lo_discount`.
At that time, the conditional clause `WHERE year = 1994` is added. This is effectively the same as `WHERE lo_orderdate BETWEEN 19940101 AND 19942131`, but since `year` is a virtual column, you can determine whether a matching row exists before reading the Arrow files.

In fact, looking at the `Stats-Hint:` line, 12 Record-Batches were loaded due to the condition `(year = 1994)`, but 48 Record-Batches were skipped. This is a simple but extremely effective means of reducing I/O load.
}

```
=# EXPLAIN ANALYZE
   SELECT sum(lo_extendedprice*lo_discount) as revenue
     FROM f_lineorder
    WHERE year = 1994
      AND lo_discount between 1 and 3
      AND lo_quantity < 25;
                                               QUERY PLAN
--------------------------------------------------------------------------------------------------------------
 Aggregate  (cost=421987.07..421987.08 rows=1 width=32) (actual time=82.914..82.915 rows=1 loops=1)
   ->  Custom Scan (GpuPreAgg) on f_lineorder  (cost=421987.05..421987.06 rows=1 width=32)      \
                                               (actual time=82.901..82.903 rows=2 loops=1)
         GPU Projection: pgstrom.psum(((lo_extendedprice * lo_discount))::double precision)
         GPU Scan Quals: ((year = 1994) AND (lo_discount <= '3'::numeric) AND                   \
                          (lo_quantity < '25'::numeric) AND                                     \
                          (lo_discount >= '1'::numeric)) [plan: 65062080 -> 542, exec: 13001908 -> 1701726]
         referenced: lo_quantity, lo_extendedprice, lo_discount, year
         Stats-Hint: (year = 1994)  [loaded: 12, skipped: 48]
         file0: /opt/arrow/mydata/f_lineorder_1996_MAIL.arrow (read: 99.53MB, size: 427.16MB)
         file1: /opt/arrow/mydata/f_lineorder_1996_SHIP.arrow (read: 99.52MB, size: 427.13MB)
         file2: /opt/arrow/mydata/f_lineorder_1994_FOB.arrow (read: 99.18MB, size: 425.67MB)
              :                :                                       :             :
         file27: /opt/arrow/mydata/f_lineorder_1997_MAIL.arrow (read: 99.23MB, size: 425.87MB)
         file28: /opt/arrow/mydata/f_lineorder_1995_MAIL.arrow (read: 99.16MB, size: 425.58MB)
         file29: /opt/arrow/mydata/f_lineorder_1993_TRUCK.arrow (read: 99.24MB, size: 425.91MB)
         GPU-Direct SQL: enabled (N=2,GPU0,1; direct=76195, ntuples=13001908)
 Planning Time: 2.402 ms
 Execution Time: 83.857 ms
(39 rows)
```

@ja:##Arrowファイルの作成方法
@en:##How to make Arrow files

@ja{
本節では、既にPostgreSQLデータベースに格納されているデータをApache Arrow形式に変換する方法を説明します。
}
@en{
This section introduces the way to transform dataset already stored in PostgreSQL database system into Apache Arrow file.
}

@ja:###PyArrow+Pandas
@en:###Using PyArrow+Pandas

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

@ja:###Pg2Arrow
@en:###Using Pg2Arrow

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
  pg2arrow [OPTION] [database] [username]

General options:
  -d, --dbname=DBNAME   Database name to connect to
  -c, --command=COMMAND SQL command to run
  -t, --table=TABLENAME Equivalent to '-c SELECT * FROM TABLENAME'
      (-c and -t are exclusive, either of them must be given)
      --inner-join=SUB_COMMAND
      --outer-join=SUB_COMMAND
  -o, --output=FILENAME result file in Apache Arrow format
      --append=FILENAME result Apache Arrow file to be appended
      (--output and --append are exclusive. If neither of them
       are given, it creates a temporary file.)
  -S, --stat[=COLUMNS] embeds min/max statistics for each record batch
                       COLUMNS is a comma-separated list of the target
                       columns if partially enabled.

Arrow format options:
  -s, --segment-size=SIZE size of record batch for each

Connection options:
  -h, --host=HOSTNAME  database server host
  -p, --port=PORT      database server port
  -u, --user=USERNAME  database user name
  -w, --no-password    never prompt for password
  -W, --password       force password prompt

Other options:
      --dump=FILENAME  dump information of arrow file
      --progress       shows progress of the job
      --set=NAME:VALUE config option to set before SQL execution
      --help           shows this message

Report bugs to <pgstrom@heterodb.com>.
```
@ja{
PostgreSQLへの接続パラメータはpsqlやpg_dumpと同様に、`-h`や`-U`などのオプションで指定します。 基本的なコマンドの使用方法は、`-c|--command`オプションで指定したSQLをPostgreSQL上で実行し、その結果を`-o|--output`で指定したファイルへArrow形式で書き出します。
}
@en{
The `-h` or `-U` option specifies the connection parameters of PostgreSQL, like `psql` or `pg_dump`. The simplest usage of this command is running a SQL command specified by `-c|--command` option on PostgreSQL server, then write out results into the file specified by `-o|--output` option in Arrow format.
}
@ja{
`-o|--output`オプションの代わりに`--append`オプションを使用する事ができ、これは既存のApache Arrowファイルへの追記を意味します。この場合、追記されるApache Arrowファイルは指定したSQLの実行結果と完全に一致するスキーマ構造を持たねばなりません。
}
@en{
`--append` option is available, instead of `-o|--output` option. It means appending data to existing Apache Arrow file. In this case, the target Apache Arrow file must have fully identical schema definition towards the specified SQL command.
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
@ja{
`--progress`オプションを指定すると、処理の途中経過を表示する事が可能です。これは巨大なテーブルをApache Arrow形式に変換する際に有用です。
}
@en{
`--progress` option enables to show progress of the task. It is useful when a huge table is transformed to Apache Arrow format.
}

@ja:##先進的な使い方
@en:##Advanced Usage


@ja:###SSDtoGPUダイレクトSQL
@en:###SSDtoGPU Direct SQL

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

@ja:###パーティション設定
@en:###Partition configuration

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
書き込みが可能なPostgreSQLテーブルをデフォルトパーティションとして指定しておく事で、一定期間の経過後、DB運用を継続しながら過去のログデータだけをArrow_Fdw外部テーブルへ移す事が可能です。
}
@en{
The normal PostgreSQL table, is read-writable, is specified as default partition, so DBA can migrate only past log-data into Arrow_Fdw foreign table under the database system operations.
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
