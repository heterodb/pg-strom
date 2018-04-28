@ja:# データ型
@en:# Supported Data Types

@ja{
PG-Stromは以下のデータ型をGPUで利用する事ができます。
}
@en{
PG-Strom support the following data types for use on GPU device.
}

@ja{
**標準の数値データ型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`smallint`        |`cl_short`        |2 bytes |    |
|`integer`         |`cl_int`          |4 bytes |    |
|`bigint`          |`cl_long`         |8 bytes |    |
|`real`            |`cl_float`        |4 bytes |    |
|`float`           |`cl_double`       |8 bytes |    |
|`numeric`         |`cl_ulong`        |可変長  |64bitの内部形式にマップ|
}
@en{
**Built-in numeric types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`smallint`        |`cl_short`        |2 bytes |    |
|`integer`         |`cl_int`          |4 bytes |    |
|`bigint`          |`cl_long`         |8 bytes |    |
|`real`            |`cl_float`        |4 bytes |    |
|`float`           |`cl_double`       |8 bytes |    |
|`numeric`         |`cl_ulong`        |variable length|mapped to 64bit internal format|
}

@ja{
!!! Note
    GPUが`numeric`型のデータを処理する際、実装上の理由からこれを64bitの内部表現に変換して処理します。
    これら内部表現への/からの変換は透過的に行われますが、例えば、桁数の大きな`numeric`型のデータは表現する事ができないため、PG-StromはCPU側でのフォールバック処理を試みます。したがって、桁数の大きな`numeric`型のデータをGPUに与えると却って実行速度が低下してしまう事になります。
    これを避けるには、GUCパラメータ`pg_strom.enable_numeric_type`を使用して`numeric`データ型を含む演算式をGPUで実行しないように設定します。
}
@en{
!!! Note
    When GPU processes values in `numeric` data type, it is converted to an internal 64bit format because of implementation reason.
    It is transparently converted to/from the internal format, on the other hands, PG-Strom cannot convert `numaric` datum with large number of digits, so tries to fallback operations by CPU. Therefore, it may lead slowdown if `numeric` data with large number of digits are supplied to GPU device.
    To avoid the problem, turn off the GUC option `pg_strom.enable_numeric_type` not to run operational expression including `numeric` data types on GPU devices.
}



@ja{
**標準の日付時刻型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`date`            |`DateADT`         |4 bytes |    |
|`time`            |`TimeADT`         |8 bytes |    |
|`timetz`          |`TimeTzADT`       |12 bytes|    |
|`timestamp`       |`Timestamp`       |8 bytes |    |
|`timestamptz`     |`TimestampTz`     |8 bytes |    |
|`interval`        |`Interval`        |16 bytes|time interval|
}

@en{
**Built-in date and time types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`date`            |`DateADT`         |4 bytes |    |
|`time`            |`TimeADT`         |8 bytes |    |
|`timetz`          |`TimeTzADT`       |12 bytes|    |
|`timestamp`       |`Timestamp`       |8 bytes |    |
|`timestamptz`     |`TimestampTz`     |8 bytes |    |
|`interval`        |`Interval`        |16 bytes|    |
}

@ja{
**標準の可変長データ型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`bpchar`          |`varlena *`       |可変長  |    |
|`varchar`         |`varlena *`       |可変長  |    |
|`bytea`           |`varlena *`       |可変長  |    |
|`text`            |`varlena *`       |可変長  |    |
}

@en{
**Built-in variable length types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`bpchar`          |`varlena *`       |variable length|
|`varchar`         |`varlena *`       |variable length|
|`bytea`           |`varlena *`       |variable length|
|`text`            |`varlena *`       |variable length|
}

@ja{
**標準の雑多なデータ型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`boolean`         |`cl_bool`         |1 byte  |    |
|`money`           |`cl_long`         |8 bytes |    |
|`uuid`            |`pg_uuid`         |16 bytes|    |
|`macaddr`         |`macaddr`         |6 bytes |    |
|`inet`            |`inet_struct`     |7 bytes or 19 bytes||
|`cidr`            |`inet_struct`     |7 bytes or 19 bytes||


}
@en{
**Built-in miscellaneous types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`boolean`         |`cl_bool`         |1 byte  |    |
|`money`           |`cl_long`         |8 bytes |    |
|`uuid`            |`pg_uuid`         |16 bytes|    |
|`macaddr`         |`macaddr`         |6 bytes |    |
|`inet`            |`inet_struct`     |7 bytes or 19 bytes||
|`cidr`            |`inet_struct`     |7 bytes or 19 bytes||
}

@ja{
**標準の範囲型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`int4range`       |`__int4range`     |14 bytes|    |
|`int8range`       |`__int8range`     |22 bytes|    |
|`tsrange`         |`__tsrange`       |22 bytes|    |
|`tstzrange`       |`__tstzrange`     |22 bytes|    |
|`daterange`       |`__daterange`     |14 bytes|    |
}
@en{
**Built-in range data types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`int4range`       |`__int4range`     |14 bytes|    |
|`int8range`       |`__int8range`     |22 bytes|    |
|`tsrange`         |`__tsrange`       |22 bytes|    |
|`tstzrange`       |`__tstzrange`     |22 bytes|    |
|`daterange`       |`__daterange`     |14 bytes|    |
}

@ja{
**PG-Stromが追加で提供するデータ型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`float2`          |`half_t`          |2 bytes |半精度浮動小数点数|
|`reggstore`       |`cl_uint`         |4 bytes |gstore_fdwのregclass型。PL/CUDA関数呼出しで特別な扱い。|
}
@en{
**Extra Types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`float2`          |`half_t`          |2 bytes |Half precision data type|
|`reggstore`       |`cl_uint`         |4 bytes |Specific version of regclass for gstore_fdw. Special handling at PL/CUDA function invocation. |
}



@ja:#デバイス関数と演算子
@en:#Device functions and operators

@ja:##型キャスト
@en:##Type cast

|destination type|source type|description|
|:---------------|:----------|:----------|
|`bool`|`int4`||
|`int2`|`int4,int8,float2,float4,float8,numeric`||
|`int4`|`int2,int8,float2,float4,float8,numeric`||
|`int8`|`int2,int4,float2,float4,float8,numeric`||
|`float2`|`int2,int4,int8,float4,float8,numeric`||
|`float4`|`int2,int4,int8,float2,float8,numeric`||
|`float8`|`int2,int4,int8,float2,float4,numeric`||
|`numeric`|`int2,int4,int8,float2,float4,float8`||
|`money`|`int4,int8,numeric`||
|`inet`|`cidr`|
|`date`|`timestamp,timestamptz`||
|`time`|`timetz,timestamp,timestamptz`||
|`timetz`|`time,timestamptz`||
|`timestamp`|`date,timestamptz`||
|`timestamptz`|`date,timestamp`||

@ja:##関数と演算子
@en:##Functions and operators

@ja:**数値型演算子**
@en:**Numeric functions/operators**

|function/operator|description|
|:----------------|:----------|
|`TYPE = TYPE`    |Comparison of two values<br>`TYPE` is any of `int2,int4,int8`<br>`COMP` is any of `=,<>,<,<=,>=,>`|
|`TYPE = TYPE`    |Comparison of two values<br>`TYPE` is any of `float2,float4,float8`<br>`COMP` is any of `=,<>,<,<=,>=,>`|
|`numeric COMP numeric`|Comparison of two values<br>`COMP` is any of `=,<>,<,<=,>=,>`|
|`TYPE + TYPE`|Arithemetic addition<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`TYPE - TYPE`|Arithemetic substract<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`TYPE * TYPE`|Arithemetic multiplication<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`TYPE / TYPE`|Arithemetic division<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`TYPE % TYPE`|Reminer operator<br>`TYPE` is any of `int2,int4,int8`|
|`TYPE & TYPE`|Bitwise AND<br>`TYPE` is any of `int2,int4,int8`|
|<code>TYPE &#124; TYPE</code>|Bitwise OR<br>`TYPE` is any of `int2,int4,int8`|
|`TYPE # TYPE`|Bitwise XOR<br>`TYPE` is any of `int2,int4,int8`|
|`~ TYPE`     |Bitwise NOT<br>`TYPE` is any if `int2,int4,int8`|
|`TYPE >> int4`|Right shift<br>`TYPE` is any of `int2,int4,int8`|
|`TYPE << int4`|Left shift<br>`TYPE` is any of `int2,int4,int8`|
|`+ TYPE`     |Unary plus<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`- TYPE`     |Unary minus<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|
|`@ TYPE`     |Absolute value<br>`TYPE` is any of `int2,int4,int8,float2,float4,float8,numeric`|


@ja:**数学関数**
@en:**Mathematical functions**

|functions/operators|description|
|:------------------|:----------|
|`cbrt(float8)`     |cube root|
|`dcbrt(float8)`    |cube root|
|`ceil(float8)`     |nearest integer greater than or equal to argument|
|`ceiling(float8)`  |nearest integer greater than or equal to argument|
|`exp(float8)`      |exponential|
|`dexp(float8)`     |exponential|
|`floor(float8)`    |nearest integer less than or equal to argument|
|`ln(float8)`       |natural logarithm|
|`dlog1(float8)`    |natural logarithm|
|`log(float8)`      |base 10 logarithm|
|`dlog10(float8)`   |base 10 logarithm|
|`pi()`             |circumference ratio|
|`power(float8,float8)`|power|
|`pow(float8,float8)`  |power|
|`dpow(float8,float8)` |power|
|`round(float8)`    |round to the nearest integer|
|`dround(float8)`   |round to the nearest integer|
|`sign(float8)`     |sign of the argument|
|`sqrt(float8)`     |square root|
|`dsqrt(float8)`    |square root|
|`trunc(float8)`    |truncate toward zero|
|`dtrunc(float8)`   |truncate toward zero|

@ja:**三角関数**
@en:**Trigonometric functions**

|functions/operators|description|
|:------------------|:----------|
|`degrees(float8)`  |radians to degrees|
|`radians(float8)`  |degrees to radians|
|`acos(float8)`     |inverse cosine|
|`asin(float8)`     |inverse sine|
|`atan(float8)`     |inverse tangent|
|`atan2(float8,float8)`|inverse tangent of `arg1 / arg2`|
|`cos(float8)`      |cosine|
|`cot(float8)`      |cotangent|
|`sin(float8)`      |sine|
|`tan(float8)`      |tangent|

@ja:**日付/時刻型演算子**
@en:**Date and time operators**

|functions/operators|description|
|:------------------|:----------|
|`date COMP date`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`date COMP timestamp`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`date COMP timestamptz`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`time COMP time`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timetz COMP timetz`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamp COMP timestamp`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamp COMP date`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamptz COMP timestamptz`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamptz COMP date`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamp COMP timestamptz`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`timestamptz COMP timestamp`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`interval COMP interval`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`date OP int4`|`OP` is either of `+,-`|
|`int4 + date`||
|`date - date`||
|`date + time`||
|`date + timetz`||
|`time + date`||
|`time - time`||
|`timestamp - timestamp`||
|`timetz OP interval`|`OP` is either of `+,-`|
|`timestamptz OP interval`|`OP` is either of `+,-`|
|`overlaps(TYPE,TYPE,TYPE,TYPE)`|`TYPE` is any of `time,timetz,timestamp,timestamptz`|
|`extract(text FROM TYPE)`|`TYPE` is any of `time,timetz,timestamp,timestamptz,interval`|
|`now()`||
|`- interval`|unary minus operator|
|`interval OP interval`|`OP` is either of `+,-`|


@ja:**文字列関数/演算子**
@en:**Text functions/operators**

|functions/operators|description|
|:------------------|:----------|
|`{text,bpchar} COMP {text,bpchar}`|`COMP` is either of `=,<>`|
|`{text,bpchar} COMP {text,bpchar}`|`COMP` is either of `<,<=,>=,>`<br>Only available on no-locale or UTF-8|
|`length(TYPE)`|length of the string<br>`TYPE` is either of `text,bpchar`|
|`TYPE LIKE text`|`TYPE` is either of `text,bpchar`|
|`TYPE NOT LIKE text`|`TYPE` is either of `text,bpchar`|
|`TYPE ILIKE text`|`TYPE` is either of `text,bpchar`<br>Only available on no-locale or UTF-8|
|`TYPE NOT ILIKE text`|`TYPE` is either of `text,bpchar`<br>Only available on no-locale or UTF-8|

@ja:**ネットワーク関数/演算子**
@en:**Network functions/operators**

|functions/operators|description|
|:------------------|:----------|
|`macaddr COMP macaddr`|`COMP` is any of `=,<>,<,<=,>=,>`|
|`macaddr & macaddr`|Bitwise AND operator|
|<code>macaddr &#124; macaddr</code>|Bitwise OR operator|
|`~ macaddr`        |Bitwise NOT operator|
|`trunc(macaddr)`   |Set last 3 bytes to zero|
|`inet COMP inet`   |`COMP` is any of `=,<>,<,<=,>=,>`|
|`inet INCL inet`   |`INCL` is any of `<<,<<=,>>,>>=,&&`|
|`~ inet`           ||
|`inet & inet`      ||
|<code>inet &#124; inet</code>|
|`inet + int8`      ||
|`inet - int8`      ||
|`inet - inet`      ||
|`broadcast(inet)`  ||
|`family(inet)`     ||
|`hostmask(inet)`   ||
|`masklen(inet)`    ||
|`netmask(inet)`    ||
|`network(inet)`    ||
|`set_masklen(cidr,int)` ||
|`set_masklen(inet,int)` ||
|`inet_same_family(inet, inet)`||
|`inet_merge(inet,inet)`||



@ja:**通貨型演算子**
@en:**Currency operators**

|functions/operators|description|
|:------------------|:----------|
|`money COMP money` |`COMP` is any of `=,<>,<,<=,>=,>`|
|`money OP money`   |`OP` is any of `+,-,/`|
|`money * TYPE`     |`TYPE` is any of `int2,int4,float2,float4,float8`|
|`TYPE * money`     |`TYPE` is any of `int2,int4,float2,float4,float8`|
|`money / TYPE`     |`TYPE` is any of `int2,int4,float2,float4,float8`|

@ja:**uuid型演算子**
@en:**UUID operators**

|functions/operators|description|
|:------------------|:----------|
|`uuid COMP uuid`   |`COMP` is any of `=,<>,<,<=,>=,>`|

@ja:**範囲型演算子**
@en:**Range type functions/operators**

|functions/operators|description|
|:------------------|:----------|
|`RANGE = RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE <> RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE < RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE <= RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE > RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE >= RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE @> RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE @> TYPE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`<br>`TYPE` is element type of `RANGE`.|
|`RANGE <@ RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`TYPE <@ RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`<br>`TYPE` is element type of `RANGE`.|
|`RANGE && RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE << RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE >> RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE &< RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE &> RANGE`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|<code>RANGE -&#124;- RANGE</code>  |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE + RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE * RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`RANGE - RANGE`    |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`lower(RANGE)`     |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`upper(RANGE)`     |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`isempty(RANGE)`   |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`lower_inc(RANGE)` |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`upper_inc(RANGE)` |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`lower_inf(RANGE)` |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`upper_inf(RANGE)` |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|
|`range_merge(RANGE,RANGE)` |`RANGE` is any of `int4range,int8range,tsrange,tstzrange,daterange`|


@ja:##その他のデバイス関数
@en:##Miscellaneous device functions

|functions/operators|result|description|
|:------------------|:-----|:----------|
|`as_int8(float8)`  |`int8`  |Re-interpret double-precision floating point bit-pattern as 64bit integer value|
|`as_int4(float4)`  |`int4`  |Re-interpret single-precision floating point bit-pattern as 32bit integer value|
|`as_int2(float2)`  |`int2`  |Re-interpret half-precision floating point bit-pattern as 16bit integer value|
|`as_float8(int8)`  |`float8`|Re-interpret 64bit integer bit-pattern as double-precision floating point value|
|`as_float4(int4)`  |`float4`|Re-interpret 32bit integer bit-pattern as single-precision floating point value|
|`as_float2(int2)`  |`float2`|Re-interpret 16bit integer bit-pattern as half-precision floating point value|


@ja:#PG-Strom独自のSQL関数群
@en:#PG-Strom Specific SQL functions

@ja{
本節ではPG-Stromが独自に提供するSQL関数群を説明します。
}
@en{
This section introduces SQL functions which are additionally provided by PG-Strom.
}

@ja:##デバイス情報関数
@en:##Device Information

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


@ja:##配列ベース行列サポート
@en:##Array-based matrix support

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

@ja:**型キャスト**
@en:**Type cast**

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

@ja:**配列ベース行列関数**
@en:**Array-based matrix functions**

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

@ja:**集約関数**
@en:**Aggregate functions**

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

@ja:##その他の関数
@en:##Miscellaneous functions

@ja{
|関数|戻り値|説明|
|:---|:----:|:---|
|`pgstrom_ccache_enabled(regclass)`|`text`|指定したテーブルに対するインメモリ列キャッシュを有効にします。|
|`pgstrom_ccache_disabled(regclass)`|`text`|指定したテーブルに対するインメモリ列キャッシュを無効にします。|
|`pgstrom_ccache_prewarm(regclass)`|`int`|指定したテーブルに対するインメモリ列キャッシュを同期的に構築します。キャッシュ使用量の上限に達した時は、その時点で終了します。|
}

@en{
|Function|Result|Description|
|:-------|:----:|:----------|
|`pgstrom_ccache_enabled(regclass)`|`text`|Enables in-memory columnar cache on the specified table.|
|`pgstrom_ccache_disabled(regclass)`|`text`|Disables in-memory columnar cache on the specified table.|
|`pgstrom_ccache_prewarm(regclass)`|`int`|Build in-memory columnar cache on the specified table synchronously, until cache usage is less than the threshold.|
}

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
|関数|戻り値|説明|
|:---|:----:|:---|
|`plcuda_kernel_max_blocksz`    |`int`|PL/CUDA関数のヘルパーとして呼ばれた場合、当該GPUカーネルの最大ブロックサイズを返す。|
|`plcuda_kernel_static_shmsz()` |`int`|PL/CUDA関数のヘルパーとして呼ばれた場合、当該GPUカーネルが静的に確保したブロックあたり共有メモリサイズを返す。|
|`plcuda_kernel_dynamic_shmsz()`|`int`|PL/CUDA関数のヘルパーとして呼ばれた場合、当該GPUカーネルが動的に確保する事のできるブロックあたり共有メモリサイズを返す。|
|`plcuda_kernel_const_memsz()`  |`int`|PL/CUDA関数のヘルパーとして呼ばれた場合、当該GPUカーネルが静的に確保したコンスタントメモリのサイズを返す。|
|`plcuda_kernel_local_memsz()`  |`int`|PL/CUDA関数のヘルパーとして呼ばれた場合、当該GPUカーネルが使用するスレッドあたりローカルメモリのサイズを返す。|
}
@en{
|Function|Result|Description|
|:-------|:----:|:----------|
|`plcuda_kernel_max_blocksz`    |`int`|It tells maximum block size of the GPU kernel of PL/CUDA function when it is called as its helper.|
|`plcuda_kernel_static_shmsz()` |`int`|It tells size of the statically acquired shared memory per block by the GPU kernel of PL/CUDA function when it is called as its helper.|
|`plcuda_kernel_dynamic_shmsz()`|`int`|It tells size of the dynamic shared memory per block, which GPU kernel of the PL/CUDA function can allocate, when it is called as its helper.|
|`plcuda_kernel_const_memsz()`  |`int`|It tells size of the constant memory acquired by the GPU kernel of PL/CUDA function, when it is called as its helper.
|`plcuda_kernel_local_memsz()`  |`int`|It tells size of the local memory per thread acquired by the GPU kernel of PL/CUDA function, when it is called as its helper.
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

**pgstrom.ccache_info**
@ja{
`pgstrom.ccache_info`システムビューは、列指向キャッシュの各チャンク（128MB単位）の情報を出力します。

|名前        |データ型  |説明|
|:-----------|:---------|:---|
|database_id |`oid`     |データベースID |
|table_id    |`regclass`|テーブルID |
|block_nr    |`int`     |チャンクの先頭ブロック番号 |
|nitems      |`bigint`  |チャンクに含まれる行数 |
|length      |`bigint`  |キャッシュされたチャンクのサイズ |
|ctime       |`timestamp with time zone`|チャンク作成時刻|
|atime       |`timestamp with time zone`|チャンク最終アクセス時刻|
}
@en{
`pgstrom.ccache_info` system view exports attribute of the columnar-cache chunks (128MB unit for each).

|Name        |Data Type |Description|
|:-----------|:---------|:---|
|database_id |`oid`     |Database Id |
|table_id    |`regclass`|Table Id |
|block_nr    |`int`     |Head block-number of the chunk |
|nitems      |`bigint`  |Number of rows in the chunk |
|length      |`bigint`  |Raw size of the cached chunk |
|ctime       |`timestamp with time zone`|Timestamp of the chunk creation |
|atime       |`timestamp with time zone`|Timestamp of the least access to the chunk |
}


**pgstrom.ccache_builder_info**
@ja{
`pgstrom.ccache_builder_info`システムビューは、列指向キャッシュの非同期ビルダープロセスの情報を出力します。

|名前        |データ型  |説明|
|:-----------|:---------|:---|
|builder_id  |`int`     |列指向キャッシュ非同期ビルダーのID |
|state       |`text`    |ビルダープロセスの状態。（`shutdown`: 停止中、`startup`: 起動中、`loading`: 列指向キャッシュの構築中、`sleep`: 一時停止中）|
|database_id |`oid`     |ビルダープロセスが割り当てられているデータベースID |
|table_id    |`regclass`|`state`が`loading`の場合、読出し中のテーブルID |
|block_nr    |`int`     |`state`が`loading`の場合、読出し中のブロック番号 |
}
@en{
`pgstrom.ccache_builder_info` system view exports information of asynchronous builder process of columnar cache.

|Name        |Data Type  |Description|
|:-----------|:----------|:---|
|builder_id  |`int`      |Asynchronous builder Id of columnar cache |
|state       |`text`     |State of the builder process (`shutdown`, `startup`, `loading` or `sleep`) |
|database_id |`oid`      |Database Id where builder process is assigned on |
|table_id    |`regclass` |Table Id where the builder process is scanning on, if `state` is `loading`. |
|block_nr    |`int`      |Block number where the builder process is scanning on, if `state` is `loading`. |
}

@ja:# GUCパラメータ
@en:# GUC Parameters

@ja{
本節ではPG-Stromの提供する設定パラメータについて説明します。
}
@en{
This session introduces PG-Strom's configuration parameters.
}

@ja{
**特定機能の有効化/無効化**

|パラメータ名                   |型    |初期値|説明       |
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.enabled`             |`bool`|`on` |PG-Strom機能全体を一括して有効化/無効化する。|
|`pg_strom.enable_gpuscan`      |`bool`|`on` |GpuScanによるスキャンを有効化/無効化する。|
|`pg_strom.enable_gpuhashjoin`  |`bool`|`on` |HashJoinによるGpuJoinを有効化/無効化する。|
|`pg_strom.enable_gpunestloop`  |`bool`|`on` |NestLoopによるGpuJoinを有効化/無効化する。|
|`pg_strom.enable_gpupreagg`    |`bool`|`on` |GpuPreAggによる集約処理を有効化/無効化する。|
|`pg_strom.pullup_outer_scan`   |`bool`|`on` |GpuPreAgg/GpuJoin直下の実行計画が全件スキャンである場合に、上位ノードでスキャン処理も行い、CPU/RAM⇔GPU間のデータ転送を省略するかどうかを制御する。|
|`pg_strom.pullup_outer_join`   |`bool`|`on` |GpuPreAgg直下がGpuJoinである場合に、JOIN処理を上位の実行計画に引き上げ、CPU⇔GPU間のデータ転送を省略するかどうかを制御する。|
|`pg_strom.enable_numeric_type` |`bool`|`on` |GPUで`numeric`データ型を含む演算式を処理するかどうかを制御する。|
|`pg_strom.cpu_fallback`        |`bool`|`off`|GPUプログラムが"CPU再実行"エラーを返したときに、実際にCPUでの再実行を試みるかどうかを制御する。|
|`pg_strom.nvme_strom_enabled`  |`bool`|`on` |SSD-to-GPUダイレクトSQL実行機能を有効化/無効化する。|
|`pg_strom.nvme_strom_threshold`|`int` |自動 |SSD-to-GPUダイレクトSQL実行機能を発動させるテーブルサイズの閾値を設定する。|
}

@en{
Enables/disables a particular feature

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.enabled`             |`bool`|`on` |Enables/disables entire PG-Strom features at once|
|`pg_strom.enable_gpuscan`      |`bool`|`on` |Enables/disables GpuScan|
|`pg_strom.enable_gpuhashjoin`  |`bool`|`on` |Enables/disables GpuJoin by HashJoin|
|`pg_strom.enable_gpunestloop`  |`bool`|`on` |Enables/disables GpuJoin by NestLoop|
|`pg_strom.enable_gpupreagg`    |`bool`|`on` |Enables/disables GpuPreAgg|
|`pg_strom.pullup_outer_scan`   |`bool`|`on` |Enables/disables to pull up full-table scan if it is just below GpuPreAgg/GpuJoin, to reduce data transfer between CPU/RAM and GPU.|
|`pg_strom.pullup_outer_join`   |`bool`|`on` |Enables/disables to pull up tables-join if GpuJoin is just below GpuPreAgg, to reduce data transfer between CPU/RAM and GPU.|
|`pg_strom.enable_numeric_type` |`bool`|`on` |Enables/disables support of `numeric` data type in arithmetic expression on GPU device|
|`pg_strom.cpu_fallback`        |`bool`|`off`|Controls whether it actually run CPU fallback operations, if GPU program returned "CPU ReCheck Error"|
|`pg_strom.nvme_strom_enabled`  |`bool`|`on` |Enables/disables the feature of SSD-to-GPU Direct SQL Execution|
|`pg_strom.nvme_strom_threshold`|`int` |自動 |Controls the table-size threshold to invoke the feature of SSD-to-GPU Direct SQL Execution|
}

@ja{
**オプティマイザに関する設定**

|パラメータ名                   |型    |初期値|説明       |
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.chunk_size`          |`int` |65534kB|PG-Stromが1回のGPUカーネル呼び出しで処理するデータブロックの大きさです。かつては変更可能でしたが、ほとんど意味がないため、現在では約64MBに固定されています。|
|`pg_strom.gpu_setup_cost`      |`real`|4000  |GPUデバイスの初期化に要するコストとして使用する値。|
|`pg_strom.gpu_dma_cost`        |`real`|10    |チャンク(64MB)あたりのDMA転送に要するコストとして使用する値。|
|`pg_strom.gpu_operator_cost`   |`real`|0.00015|GPUの演算式あたりの処理コストとして使用する値。`cpu_operator_cost`よりも大きな値を設定してしまうと、いかなるサイズのテーブルに対してもPG-Stromが選択されることはなくなる。|
}
@en{
**Optimizer Configuration**

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.chunk_size`          |`int` |65534kB|Size of the data blocks processed by a single GPU kernel invocation. It was configurable, but makes less sense, so fixed to about 64MB in the current version.|
|`pg_strom.gpu_setup_cost`      |`real`|4000  |Cost value for initialization of GPU device|
|`pg_strom.gpu_dma_cost`        |`real`|10    |Cost value for DMA transfer over PCIe bus per data-chunk (64MB)|
|`pg_strom.gpu_operator_cost`   |`real`|0.00015|Cost value to process an expression formula on GPU. If larger value than `cpu_operator_cost` is configured, no chance to choose PG-Strom towards any size of tables|
}

@ja{
**エグゼキュータに関する設定**

|パラメータ名                       |型    |初期値|説明       |
|:----------------------------------|:----:|:----:|:----------|
|`pg_strom.global_max_async_tasks`  |`int` |160 |PG-StromがGPU実行キューに投入する事ができる非同期タスクのシステム全体での最大値。
|`pg_strom.local_max_async_tasks`   |`int` |8   |PG-StromがGPU実行キューに投入する事ができる非同期タスクのプロセス毎の最大値。CPUパラレル処理と併用する場合、この上限値は個々のバックグラウンドワーカー毎に適用されます。したがって、バッチジョブ全体では`pg_strom.local_max_async_tasks`よりも多くの非同期タスクが実行されることになります。
|`pg_strom.max_number_of_gpucontext`|`int` |自動|GPUデバイスを抽象化した内部データ構造 GpuContext の数を指定します。通常、初期値を変更する必要はありません。
}
@en{
**Executor Configuration**

|Parameter                         |Type  |Default|Description|
|:---------------------------------|:----:|:----:|:----------|
|`pg_strom.global_max_async_tasks` |`int` |160   |Number of asynchronous taks PG-Strom can throw into GPU's execution queue in the whole system.|
|`pg_strom.local_max_async_tasks`  |`int` |8     |Number of asynchronous taks PG-Strom can throw into GPU's execution queue per process. If CPU parallel is used in combination, this limitation shall be applied for each background worker. So, more than `pg_strom.local_max_async_tasks` asynchronous tasks are executed in parallel on the entire batch job.|
|`pg_strom.max_number_of_gpucontext`|`int`|auto  |Specifies the number of internal data structure `GpuContext` to abstract GPU device. Usually, no need to expand the initial value.|
}

@ja{
**列指向キャッシュ関連の設定**

|パラメータ名                  |型      |初期値    |説明       |
|:-----------------------------|:------:|:---------|:----------|
|`pg_strom.ccache_base_dir`    |`string`|`'/dev/shm'`|列指向キャッシュを保持するファイルシステム上のパスを指定します。通常、`tmpfs`がマウントされている`/dev/shm`を変更する必要はありません。|
|`pg_strom.ccache_databases`   |`string`|`''`        |列指向キャッシュの非同期ビルドを行う対象データベースをカンマ区切りで指定します。`pgstrom_ccache_prewarm()`によるマニュアルでのキャッシュビルドには影響しません。|
|`pg_strom.ccache_num_builders`|`int`   |`2`       |列指向キャッシュの非同期ビルドを行うワーカープロセス数を指定します。少なくとも`pg_strom.ccache_databases`で設定するデータベースの数以上にワーカーが必要です。|
|`pg_strom.ccache_log_output`  |`bool`  |`false`   |列指向キャッシュの非同期ビルダーがログメッセージを出力するかどうかを制御します。|
|`pg_strom.ccache_total_size`  |`int`   |自動      |列指向キャッシュの上限を kB 単位で指定します。区画サイズの75%またはシステムの物理メモリの66%のいずれか小さな方がデフォルト値です。|
}
@en{
**Columnar Cache Configuration**

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.ccache_base_dir`    |`string`|`'/dev/shm'`|Specifies the directory path to store columnar cache data files. Usually, no need to change from `/dev/shm` where `tmpfs` is mounted at.|
|`pg_strom.ccache_databases`   |`string`|`''`    |Specified the target databases for asynchronous columnar cache build, in comma separated list. It does not affect to the manual cache build by `pgstrom_ccache_prewarm()`.|
|`pg_strom.ccache_num_builders`|`int`   |`2`     |Specified the number of worker processes for asynchronous columnar cache build. It needs to be larger than or equeal to the number of databases in `pg_strom.ccache_databases`.|
|`pg_strom.ccache_log_output`  |`bool`  |`false` |Controls whether columnar cache builder prints log messages, or not|
|`pg_strom.ccache_total_size`  |`int`   |auto    |Upper limit of the columnar cache in kB. Default is the smaller in 75% of volume size or 66% of system physical memory.|
}

@ja{
**gstore_fdw関連の設定**

|パラメータ名                   |型      |初期値    |説明       |
|:------------------------------|:------:|:---------|:----------|
|`pg_strom.gstore_max_relations`|`int`   |100       |gstore_fdwを用いた外部表数の上限です。パラメータの更新には再起動が必要です。|
}
@en{
**gstore_fdw Configuration**

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.gstore_max_relations`|`int`   |100       |Upper limit of the number of foreign tables with gstore_fdw. It needs restart to update the parameter.|
}

@ja{
**GPUプログラムの生成とビルドに関連する設定**

|パラメータ名                   |型      |初期値  |説明       |
|:------------------------------|:------:|:-------|:----------|
|`pg_strom.program_cache_size`  |`int`   |`256MB` |ビルド済みのGPUプログラムをキャッシュしておくための共有メモリ領域のサイズです。パラメータの更新には再起動が必要です。|
|`pg_strom.num_program_builders`|`int`|`2`|GPUプログラムを非同期ビルドするためのバックグラウンドプロセスの数を指定します。パラメータの更新には再起動が必要です。|
|`pg_strom.debug_jit_compile_options`|`bool`|`off`|GPUプログラムのJITコンパイル時に、デバッグオプション（行番号とシンボル情報）を含めるかどうかを指定します。GPUコアダンプ等を用いた複雑なバグの解析に有用ですが、性能のデグレードを引き起こすため、通常は使用すべきでありません。||`pg_strom.debug_kernel_source` |`bool`  |`off`    |このオプションが`on`の場合、`EXPLAIN VERBOSE`コマンドで自動生成されたGPUプログラムを書き出したファイルパスを出力します。|
}
@en{
**Configuration of GPU code generation and build**

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.program_cache_size`  |`int` |`256MB` |Amount of the shared memory size to cache GPU programs already built. It needs restart to update the parameter.|
|`pg_strom.num_program_builders`|`int`|`2`|Number of background workers to build GPU programs asynchronously. It needs restart to update the parameter.|
|`pg_strom.debug_jit_compile_options`|`bool`|`off`|Controls to include debug option (line-numbers and symbol information) on JIT compile of GPU programs. It is valuable for complicated bug analysis using GPU core dump, however, should not be enabled on daily use because of performance degradation.|
|`pg_strom.debug_kernel_source` |`bool`  |`off`   |If enables, `EXPLAIN VERBOSE` command also prints out file paths of GPU programs written out.|
}

@ja{
**GPUデバイスに関連する設定**

|パラメータ名                   |型      |初期値 |説明       |
|:------------------------------|:------:|:------|:----------|
|`pg_strom.cuda_visible_devices`|`string`|`''`   |PostgreSQLの起動時に特定のGPUデバイスだけを認識させてい場合は、カンマ区切りでGPUデバイス番号を記述します。これは環境変数`CUDA_VISIBLE_DEVICES`を設定するのと同等です。|
|`pg_strom.gpu_memory_segment_size`|`int`|`512MB`|PG-StromがGPUメモリをアロケーションする際に、1回のCUDA API呼び出しで獲得するGPUデバイスメモリのサイズを指定します。この値が大きいとAPI呼び出しのオーバーヘッドは減らせますが、デバイスメモリのロスは大きくなります。
|`pg_strom.max_num_preserved_gpu_memory`|`int`|2048|確保済みGPUデバイスメモリのセグメント数の上限を指定します。通常は初期値を変更する必要はありません。|
}
@en{
**GPU Device Configuration**

|Parameter                      |Type  |Default|Description|
|:------------------------------|:----:|:----:|:----------|
|`pg_strom.cuda_visible_devices`|`string`|`''`   |List of GPU device numbers in comma separated, if you want to recognize particular GPUs on PostgreSQL startup. It is equivalent to the environment variable `CUDAVISIBLE_DEVICES`|
|`pg_strom.gpu_memory_segment_size`|`int`|`512MB`|Specifies the amount of device memory to be allocated per CUDA API call. Larger configuration will reduce the overhead of API calls, but not efficient usage of device memory.|
|`pg_strom.max_num_preserved_gpu_memory`|`int`|2048|Upper limit of the number of preserved GPU device memory segment. Usually, don't need to change from the default value.|
}





