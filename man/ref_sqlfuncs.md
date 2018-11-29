@ja{
<h1>SQLオブジェクト</h1>

本章ではPG-Stromが独自に提供するSQLオブジェクトについて説明します。
}
@en{
<h1>SQL Objects</h1>

This chapter introduces SQL objects additionally provided by PG-Strom.
}

@ja:#デバイス情報関数
@en:#Device Information

@ja{
|関数|戻り値|説明|
|:---|:----:|:---|
|`gpu_device_name(int = 0)`|`text`|指定したGPUデバイスの名前を返します|
|`gpu_global_memsize(int = 0)`|`bigint`|指定したGPUデバイスのデバイスメモリ容量を返します|
|`gpu_max_blocksize(int = 0)`|`int`|指定したGPUデバイスにおけるブロックサイズの最大値を返します。現在サポート対象のGPUでは1024です。|
|`gpu_warp_size(int = 0)`|`int`|指定したGPUデバイスにおけるワープサイズを返します。現在サポート対象のGPUでは32です。|
|`gpu_max_shared_memory_perblock(int = 0)`|`int`|指定したGPUデバイスにおけるブロックあたり共有メモリの最大値を返します。|
|`gpu_num_registers_perblock(int = 0)`|`int`|指定したGPUデバイスにおけるブロックあたりレジスタ数を返します。|
|`gpu_num_multiptocessors(int = 0)`|`int`|指定したGPUデバイスにおけるSM(Streaming Multiprocessor)ユニットの数を返します。|
|`gpu_num_cuda_cores(int = 0)`|`int`|指定したGPUデバイスにおけるCUDAコア数を返します。|
|`gpu_cc_major(int = 0)`|`int`|指定したGPUデバイスのCC(Compute Capability)メジャーバージョンを返します。|
|`gpu_cc_minor(int = 0)`|`int`|指定したGPUデバイスのCC(Compute Capability)マイナーバージョンを返します。|
|`gpu_pci_id(int = 0)`|`int`|指定したGPUデバイスが接続されているPCIバスIDを返します。|
}

@en{
|Function|Result|Description|
|:-------|:----:|:----------|
|`gpu_device_name(int = 0)`|`text`|It tells name of the specified GPU device.|
|`gpu_global_memsize(int = 0)`|`bigint`|It tells amount of the specified GPU device in bytes.|
|`gpu_max_blocksize(int = 0)`|`int`|It tells maximum block-size on the specified GPU device. 1024, in the currently supported GPU models.|
|`gpu_warp_size(int = 0)`|`int`|It tells warp-size on the specified GPU device. 32, in the currently supported GPU models.|
|`gpu_max_shared_memory_perblock(int = 0)`|`int`|It tells maximum shared memory size per block on the specified GPU device.|
|`gpu_num_registers_perblock(int = 0)`|`int`|It tells total number of registers per block on the specified GPU device.|
|`gpu_num_multiptocessors(int = 0)`|`int`|It tells number of SM(Streaming Multiprocessor) units on the specified GPU device.|
|`gpu_num_cuda_cores(int = 0)`|`int`|It tells number of CUDA cores on the specified GPU device.|
|`gpu_cc_major(int = 0)`|`int`|It tells major CC(Compute Capability) version of the specified GPU device.|
|`gpu_cc_minor(int = 0)`|`int`|It tells minor CC(Compute Capability) version of the specified GPU device.|
|`gpu_pci_id(int = 0)`|`int`|It tells PCI bus-id of the specified GPU device.|
}


@ja:#配列ベース行列サポート
@en:#Array-based matrix support

@ja{
PL/CUDA関数と行列データを受け渡しするために、PostgreSQLの配列型を使用する事ができます。
固定長の論理値/数値型データでNULLを含まない二次元配列は（配列データ型のヘッダ領域を除いて）フラットなデータ構造を持っており、行列のインデックスによって各要素のアドレスを一意に特定する事ができます。
PG-Stromは配列ベースの行列を取り扱うためのSQL関数をいくつか提供しています。
}
@en{
You can use array data type of PostgreSQL to deliver matrix-data for PL/CUDA functions.
A two-dimensional array of fixed-length boolean/numeric values without NULL has flat data structure (expect for the array header). It allows to identify the address of elements by indexes of the matrix uniquely.
PG-Strom provides several SQL functions to handle array-based matrix.
}

@ja:##型キャスト
@en:##Type cast

@ja{
|変換先|変換元|説明|
|:-----|:-----|:---|
|`int[]`|`bit`|ビット列型を32bit整数値の配列に変換します。不足するビットは0で埋められます。|
|`bit`|`int[]`|int配列をビット列型に変換します。|
}

@en{
|destination type|source type|description|
|:---------------|:----------|:----------|
|`int[]`|`bit`|convert bit-string to 32bit integer array. Unaligned bits are filled up by 0.|
|`bit`|`int[]`|convert 32bit integer to bit-string|
}

@ja:##配列ベース行列関数
@en:##Array-based matrix functions

@ja{
|関数/演算子|返り値|説明|
|:----------|:-----|:---|
|`array_matrix_validation(anyarray)`|`bool`|配列が配列ベース行列の条件を満足しているかどうかチェックします。|
|`array_matrix_height(anyarray)`|`int`|配列ベース行列の高さを返します。|
|`array_matrix_width(anyarray)`|`int`|配列ベース行列の幅を返します。|
|`array_vector_rawsize(regtype,int)`|`bigint`|指定のデータ型で長さNのベクトルを作成した場合のサイズを返します。|
|`array_matrix_rawsize(regtype,int,int)`|`bigint`|指定のデータ型で高さH幅Wの行列を作成した場合のサイズを返します。|
|`array_cube_rawsize(regtype,int,int,int)`|`bigint`|指定のデータ型で高さH幅W深さDのキューブを作成した場合のサイズを返します。|
|`type_len(regtype)`|`bigint`|指定のデータ型のサイズを返します。|
|`composite_type_rawsize(LEN,...)`|`bigint`|指定のデータ長の並びで複合型を定義した時に必要なサイズを返します。`type_len()`を組み合わせて使用する事を想定しています。<br>`LEN`は`int,bigint`のいずれか|
|`matrix_unnest(anyarray)`|`record`|集合を返す関数で、配列ベース行列の先頭行から順に1行ずつ取り出します。PostgreSQLはレコードの型情報を持っていませんので、`ROW()`句によって型情報を与える必要があります。|
|`rbind(MATRIX,MATRIX)`|`MATRIX`|配列ベース行列を縦方向に結合します。<br>`MATRIX`は`bool,int2,int4,int8,float4,float8`いずれかの配列型|
|`rbind(TYPE,MATRIX)`|`MATRIX`|配列ベース行列の先頭行にスカラ値を結合します。複数列が存在する場合、先頭行の全ての列に同じスカラ値がセットされます。<br>`MATRIX`は`bool,int2,int4,int8,float4,float8`いずれかの配列型<br>`TYPE`は`MATRIX`の要素型|
|`rbind(MATRIX,TYPE)`|`MATRIX`|配列ベース行列の末尾行にスカラ値を結合します。複数列が存在する場合、末尾行の全ての列に同じスカラ値がセットされます。<br>`MATRIX`は`bool,int2,int4,int8,float4,float8`いずれかの配列型<br>`TYPE`は`MATRIX`の要素型|
|`cbind(MATRIX,MATRIX)`|`MATRIX`|配列ベース行列を横方向に結合します。<br>`MATRIX`は`bool,int2,int4,int8,float4,float8`いずれかの配列型|
|`cbind(TYPE,MATRIX)`|`MATRIX`|配列ベース行列の左側にスカラ値を結合します。複数行が存在する場合、左端の全ての行に同じスカラ値がセットされます。<br>`MATRIX`は`bool,int2,int4,int8,float4,float8`いずれかの配列型<br>`TYPE`は`MATRIX`の要素型|
|`cbind(MATRIX,TYPE)`|`MATRIX`|配列ベース行列の右側にスカラ値を結合します。複数行が存在する場合、右端の全ての行に同じスカラ値がセットされます。<br>`MATRIX`は`bool,int2,int4,int8,float4,float8`いずれかの配列型<br>`TYPE`は`MATRIX`の要素型|
|`transpose(MATRIX)`|`MATRIX`|配列ベース行列を転置します。<br>`MATRIX`は`bool,int2,int4,int8,float4,float8`いずれかの配列型|
}

@en{
|functions/operators|result|description|
|:------------------|:-----|:----------|
|`array_matrix_validation(anyarray)`|`bool`|It checks whether the supplied array satisfies the requirement of array-based matrix.|
|`array_matrix_height(anyarray)`|`int`|It tells height of the array-based matrix.|
|`array_matrix_width(anyarray)`|`int`|It tells width of the array-based matrix.|
|`array_vector_rawsize(regtype,int)`|`bigint`|It tells expected size if N-items vector is created with the specified type.|
|`array_matrix_rawsize(regtype,int,int)`|`bigint`|It tells expected size if HxW matrix is created with the specified type.|
|`array_cube_rawsize(regtype,int,int,int)`|`bigint`|It tells expected size if HxWxD cube is created with the specified type.|
|`type_len(regtype)`|`bigint`|It tells unit length of the specified type.|
|`composite_type_rawsize(LEN,...)`|`bigint`|It tells expected size of the composite type if constructed with the specified data-length order. We expect to use the function with `type_len()`<br>`LEN` is either of `int,bigint`|
|`matrix_unnest(anyarray)`|`record`|It is a function to return set, to fetch rows from top of the supplied array-based matrix. PostgreSQL has no type information of the record, so needs to give type information using `ROW()` clause.|
|`rbind(MATRIX,MATRIX)`|`MATRIX`|It combines two array-based matrix vertically.<br>`MATRIX` is array type of any of `bool,int2,int4,int8,float4,float8`|
|`rbind(TYPE,MATRIX)`|`MATRIX`|It adds a scalar value on head of the array-based matrix. If multiple columns exist, the scalar value shall be set on all the column of the head row.<br>`MATRIX` is array type of any of `bool,int2,int4,int8,float4,float8`.<br>`TYPE` is element of `MATRIX`|
|`rbind(MATRIX,TYPE)`|`MATRIX`|It adds a scalar value on bottom of the array-based matrix. If multiple columns exist, the scalar value shall be set on all the column of the bottom row.`MATRIX` is array type of any of `bool,int2,int4,int8,float4,float8`<br>`TYPE` is element type of `MATRIX`|
|`cbind(MATRIX,MATRIX)`|`MATRIX`|It combines two array-based matrix horizontally.<br>`MATRIX` is array type of any of `bool,int2,int4,int8,float4,float8`|
|`cbind(TYPE,MATRIX)`|`MATRIX`|It adds a scalar value on left of the array-based matrix. If multiple rows exist, the scalar value shall be set on all the rows of the left column.<br>`MATRIX` is array type of any of `bool,int2,int4,int8,float4,float8`<br>`TYPE` is element type of `MATRIX`|
|`cbind(MATRIX,TYPE)`|`MATRIX`|It adds a scalar value on right of the array-based matrix. If multiple rows exist, the scalar value shall be set on all the rows of the right column.<br>`MATRIX` is array type of any of `bool,int2,int4,int8,float4,float8`.<br>`TYPE` is element type of `MATRIX`|
|`transpose(MATRIX)`|`MATRIX`|It transposes the array-based matrix.<br>`MATRIX` is array type of any of `bool,int2,int4,int8,float4,float8`|
}

@ja:##集約関数
@en:##Aggregate functions

@ja{
|関数/演算子|返り値|説明|
|:----------|:-----|:---|
|`array_matrix(TYPE,...)`|`TYPE[]`|可変長引数の集約関数です。M個の引数でN行が入力されると、M列xN行の配列ベース行列を返します。<br>`TYPE`は`bool,int2,int4,int8,float4,float8`のいずれかです。|
|`array_matrix(bit)`|`int[]`|ビット列を32bit整数値の組と見なして、`int4[]`型の配列ベース行列として返す集約関数です。|
|`rbind(MATRIX)`|`MATRIX`|入力された配列ベース行列を縦に連結する集約関数です。<br>`MATRIX`は`bool,int2,int4,int8,float4,float8`いずれかの配列型|
|`cbind(MATRIX)`|`MATRIX`|入力された配列ベース行列を横に連結する集約関数です。<br>`MATRIX`は`bool,int2,int4,int8,float4,float8`いずれかの配列型|
}

@en{
|functions/operators|result|description|
|:------------------|:-----|:----------|
|`array_matrix(TYPE,...)`|`TYPE[]`|An aggregate function with varidic arguments. It produces M-cols x N-rows array-based matrix if N-rows were supplied with M-columns.<br>`TYPE` is any of `bool,int2,int4,int8,float4,float8`|
|`array_matrix(bit)`|`bit[]`|An aggregate function to produce `int4[]` array-based matrix. It considers bit-string as a set of 32bits integer values.|
|`rbind(MATRIX)`|`MATRIX`|An aggregate function to combine the supplied array-based matrix vertically.<br>`MATRIX` is array type of any of `bool,int2,int4,int8,float4,float8`|
|`cbind(MATRIX)`|`MATRIX`|An aggregate function to combine the supplied array-based matrix horizontally.`MATRIX` is array type of any of `bool,int2,int4,int8,float4,float8`|
}

@ja:#その他の関数
@en:#Miscellaneous functions

@ja{
|関数|戻り値|説明|
|:---|:----:|:---|
|`gstore_fdw_format(reggstore)`|`text`|gstore_fdw外部テーブルの内部データ形式を返します。|
|`gstore_fdw_nitems(reggstore)`|`bigint`|gstore_fdw外部テーブルの行数を返します。|
|`gstore_fdw_nattrs(reggstore)`|`bigint`|gstore_fdw外部テーブルの列数を返します。|
|`gstore_fdw_rawsize(reggstore)`|`bigint`|gstore_fdw外部テーブルのバイト単位のサイズを返します。|
}
@en{
|Function|Result|Description|
|:-------|:----:|:----------|
|`gstore_fdw_format(reggstore)`|`text`|It tells internal format of the specified gstore_fdw foreign table.|
|`gstore_fdw_nitems(reggstore)`|`bigint`|It tells number of rows of the specified gstore_fdw foreign table.|
|`gstore_fdw_nattrs(reggstore)`|`bigint`|It tells number of columns of the specified gstore_fdw foreign table.|
|`gstore_fdw_rawsize(reggstore)`|`bigint`|It tells raw size of the specified gstore_fdw foreign table in bytes.|
}

@ja{
|関数|戻り値|説明|
|:---|:----:|:---|
|`gstore_export_ipchandle(reggstore)`|`bytea`|gstore_fdwのGPUデバイスメモリ領域のIPCハンドラを返します。|
|`lo_import_gpu(int, bytea, bigint, bigint, oid=0)`|`oid`|外部アプリケーションの確保したGPUデバイスメモリ領域をマップし、その内容をラージオブジェクトへインポートします。|
|`lo_export_gpu(oid, int, bytea, bigint, bigint)`|`bigint`|外部アプリケーションの確保したGPUデバイスメモリ領域をマップし、ラージオブジェクトの内容を当該領域へエクスポートします。|
}
@en{
|Function|Result|Description|
|:-------|:----:|:----------|
|`gstore_export_ipchandle(reggstore)`|`bytea`|It tells IPC-handle of the GPU device memory region of the specified gstore_fdw foreign table.|
|`lo_import_gpu(int, bytea, bigint, bigint, oid=0)`|`oid`|It maps GPU device memory region acquired by external application, then import its contents into a largeobject.|
|`lo_export_gpu(oid, int, bytea, bigint, bigint)`|`bigint`|It maps GPU device memory region acquired by external application, then export contents of the specified largeobject into the region.|
}

@ja{
|関数定義  |結果型|説明|
|:---------|:----:|:---|
|`attnums_of(regclass,text[])`|`smallint[]`|第一引数で指定したテーブルの第二引数で指定した列名（複数可）の列番号を配列として返します。|
|`attnum_of(regclass,text)`|`smallint`|第一引数で指定したテーブルの第二引数で指定した列名の列番号を返します。|
|`atttypes_of(regclass,text[])`|`regtype[]`|第一引数で指定したテーブルの第二引数で指定した列名（複数可）のデータ型を配列として返します。|
|`atttype_of(regclass,text)`|`regtype`|第一引数で指定したテーブルの第二引数で指定した列名のデータ型を返します。|
|`attrs_types_check(regclass,text[],regtype[])`|`bool`|第一引数で指定したテーブルの、第二引数で指定した列名（複数可）のデータ型が、第三引数で指定したデータ型とそれぞれ一致しているかどうかを調べます。|
|`attrs_type_check(regclass,text[],regtype)`|`bool`|第一引数で指定したテーブルの、第二引数で指定した列名（複数可）のデータ型が、全て第三引数で指定したデータ型と一致しているかどうかを調べます。|
}

@en{
|Definition|Result|Description|
|:---------|:----:|:----------|
|`attnums_of(regclass,text[])`|`smallint[]`|It returns attribute numbers for the column names (may be multiple) of the 2nd argument on the table of the 1st argument.|
|`attnum_of(regclass,text)`|`smallint`|It returns attribute number for the column name of the 2nd argument on the table of the 1st argument.|
|`atttypes_of(regclass,text[])`|`regtype[]`|It returns data types for the column names (may be multiple) of the 2nd argument on the table of the 1st argument.|
|`atttype_of(regclass,text)`|`regtype`|It returns data type for the column name of the 2nd argument on the table of the 1st argument.|
|`attrs_types_check(regclass,text[],regtype[])`|`bool`|It checks whether the data types of the columns (may be multiple) of the 2nd argument on the table of the 1st argument match with the data types of the 3rd argument for each.
|`attrs_type_check(regclass,text[],regtype)`|`bool`|It checks whether all the data types of the columns (may be multiple) of the 2nd argument on the table of the 1st argument match with the data type of the 3rd argument.|
}

@ja{
|関数|戻り値|説明|
|:---|:----:|:---|
|`pgstrom.license_validation()`|`text`|商用サブスクリプションを手動でロードします。|
|`pgstrom.license_query()`|`text`|現在ロードされている商用サブスクリプションを表示します。|
}
@en{
|Function|Result|Description|
|:-------|:----:|:----------|
|`pgstrom.license_validation()`|`text`|It validates commercial subscription.|
|`pgstrom.license_query()`|`text`|It shows the active commercial subscription.|
}



@ja:# システムビュー
@en:# System View

@ja{
PG-Stromは内部状態をユーザやアプリケーションに出力するためのシステムビューをいくつか提供しています。
これらのシステムビューは将来のバージョンで情報が追加される可能性があります。そのため、アプリケーションから`SELECT * FROM ...`によってこれらシステムビューを参照する事は避けてください。
}
@en{
PG-Strom provides several system view to export its internal state for users or applications.
The future version may add extra fields here. So, it is not recommended to reference these information schemas using `SELECT * FROM ...`.
}

**pgstrom.device_info**
@ja{
`pgstrom.device_info`システムビューは、PG-Stromが認識しているGPUのデバイス属性値を出力します。
GPUはモデルごとにコア数やメモリ容量、最大スレッド数などのスペックが異なっており、PL/CUDA関数などで直接GPUのプログラミングを行う場合には、これらの情報を元にソフトウェアを最適化する必要があります。

|名前        |データ型  |説明|
|:-----------|:---------|:---|
|device_nr   |`int`     |GPUデバイス番号 |
|aindex      |`int`     |属性インデックス|
|attribute   |`text`    |デバイス属性名  |
|value       |`text`    |デバイス属性値  |
}
@en{
`pgstrom.device_into` system view exports device attributes of the GPUs recognized by PG-Strom.
GPU has different specification for each model, like number of cores, capacity of global memory, maximum number of threads and etc, user's software should be optimized according to the information if you try raw GPU programming with PL/CUDA functions.

|Name        |Data Type |Description|
|:-----------|:---------|:----------|
|device_nr   |`int`     |GPU device number |
|aindex      |`int`     |Attribute index |
|attribute   |`text`    |Attribute name |
|value       |`text`    |Value of the attribute |
}

**pgstrom.device_preserved_meminfo**
@ja{
`pgstrom.device_preserved_meminfo`システムビューは、複数のPostgreSQLバックエンドプロセスから共有するために予め確保済みのGPUデバイスメモリ領域の情報を出力します。
現在のところ、gstore_fdwのみが本機能を使用しています。

|名前        |データ型  |説明|
|:-----------|:---------|:---|
|device_nr   |`int`     |GPUデバイス番号
|handle      |`bytea`   |確保済みGPUデバイスメモリのIPCハンドラ
|owner       |`regrole` |確保済みGPUデバイスメモリの作成者
|length      |`bigint`  |確保済みGPUデバイスメモリのバイト単位の長さ
|ctime       |`timestamp with time zone`|確保済みGPUデバイスメモリの作成時刻
}
@en{
`pgstrom.device_preserved_meminfo` system view exports information of the preserved device memory; which can be shared multiple PostgreSQL backend.
Right now, only gstore_fdw uses this feature.

|Name        |Data Type |Description|
|:-----------|:---------|:----------|
|device_nr   |`int`     |GPU device number
|handle      |`bytea`   |IPC handle of the preserved device memory
|owner       |`regrole` |Owner of the preserved device memory
|length      |`bigint`  |Length of the preserved device memory in bytes
|ctime       |`timestamp with time zone`|Timestamp when the preserved device memory is created

}
