@ja{
<h1>データ型</h1>
PG-Stromは以下のデータ型をGPUで利用する事ができます。
}
@en{
<h1>Data Types</h1>
PG-Strom support the following data types for use on GPU device.
}

@ja{
# 数値データ型

|SQLデータ型       |内部データ形式 |データ長|備考|
|:-----------------|:--------------|:-------|:---|
|`smallint`        |`short`        |2 bytes |    |
|`integer`         |`int`          |4 bytes |    |
|`bigint`          |`long`         |8 bytes |    |
|`float2`          |`short`        |2 bytes |半精度浮動小数点型。PG-Stromによる独自拡張。|
|`real`            |`float`        |4 bytes |    |
|`float`           |`double`       |8 bytes |    |
|`numeric`         |`int128`       |可変長  |内部形式は128bit固定少数点型|
}
@en{
# Numeric types

|SQL data types    |Internal format|Length  |Memo|
|:-----------------|:--------------|:-------|:---|
|`smallint`        |`short`        |2 bytes |    |
|`integer`         |`int`          |4 bytes |    |
|`bigint`          |`long`         |8 bytes |    |
|`float2`          |`short`        |2 bytes |Half precision data type. An extra data type by PG-Strom|
|`real`            |`float`        |4 bytes |    |
|`float`           |`double`       |8 bytes |    |
|`numeric`         |`int128`       |variable length|mapped to 128bit fixed-point numerical internal format|
}

@ja{
!!! Note
    GPUが`numeric`型のデータを処理する際、実装上の理由からこれを128bit固定少数点の内部表現に変換して処理します。（これは Apache Arrow の`Decimal`型と同一の形式です）
    これら内部表現への/からの変換は透過的に行われますが、例えば、桁数の大きな`numeric`型のデータは表現する事ができないため、PG-StromはCPU側でのフォールバック処理を試みます。したがって、桁数の大きな`numeric`型のデータをGPUに与えると却って実行速度が低下してしまう事になります。
    これを避けるには、GUCパラメータ`pg_strom.enable_numeric_type`を使用して`numeric`データ型を含む演算式をGPUで実行しないように設定します。
}
@en{
!!! Note
    When GPU processes values in `numeric` data type, it is converted to an internal 128bit fixed-point number because of implementation reason. (This layout is identical to `Decimal` type in Apache Arrow.)
    It is transparently converted to/from the internal format, on the other hands, PG-Strom cannot convert `numaric` datum with large number of digits, so tries to fallback operations by CPU. Therefore, it may lead slowdown if `numeric` data with large number of digits are supplied to GPU device.
    To avoid the problem, turn off the GUC option `pg_strom.enable_numeric_type` not to run operational expression including `numeric` data types on GPU devices.
}

@ja{
!!! Note
    GPUでは半精度浮動小数点型がハードウェアでサポートされていますが、CPU(x86_64プロセッサ)では未対応です。そのため、`float2`データ型をCPUで処理する場合には、これを一度`float`や`double`型に変換した上で演算を行います。そのため、GPUのように`float2`の方が演算速度で有利という事はありません。機械学習や統計解析用途にデータ量を抑制するための機能です。
}
@en{
!!! Note
    Even though GPU supports half-precision floating-point numbers by hardware, CPU (x86_64 processor) does not support it yet. So, when CPU processes `float2` data types, it transform them to `float` or `double` on calculations. So, CPU has no advantages for calculation performance of `float2`, unlike GPU. It is a feature to save storage/memory capacity for machine-learning / statistical-analytics. 
}

@ja{
# 標準の日付時刻型

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
# Built-in date and time types

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
# 標準の可変長データ型

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`bpchar`          |`varlena *`       |可変長  |    |
|`varchar`         |`varlena *`       |可変長  |    |
|`bytea`           |`varlena *`       |可変長  |    |
|`text`            |`varlena *`       |可変長  |    |
}

@en{
# Built-in variable length types

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`bpchar`          |`varlena *`       |variable length|
|`varchar`         |`varlena *`       |variable length|
|`bytea`           |`varlena *`       |variable length|
|`text`            |`varlena *`       |variable length|
}

@ja{
# 標準の非構造データ型

|SQLデータ型      |内部データ形式    |データ長|備考|
|:----------------|:-----------------|:-------|:---|
|`jsonb`          |`varlena *`       |可変長  |    |

!!! Note
    `jsonb`データ型をGPUで処理させる場合には、次の2つの点に留意してください。
    実際に参照されない属性もストレージから読み出し、GPUに転送する必要があるため、I/Oバスの利用効率は必ずしも良くないデータ型である事。データ長が[TOAST化](https://www.postgresql.jp/document/current/html/storage-toast.html)の閾値（通常は2kB弱）を越えてしまった場合、`jsonb`データ全体がTOASTテーブルへ書き出されるため、GPU側では処理できず非効率なCPU-fallback処理を呼び出してしまう事。
    後者の問題に対しては、テーブルのストレージオプション`toast_tuple_target`を拡大し、TOAST化の閾値を引き上げる事である程度は回避する事も可能です。
}
@en{
# Built-in unstructured data types

|SQL data types   |Internal format   |Length  |Memo|
|:----------------|:-----------------|:-------|:---|
|`jsonb`          |`varlena *`       |variable length|

!!! Note
    Pay attention for the two points below, when GPU processes `jsonb` data types.
    `jsonb` is not performance efficient data types because it has to load unreferenced attributes onto GPU from the storage, so tend to consume I/O bandwidth by junk data.
    In case when `jsonb` data length exceeds the threshold of [datum TOASTen](https://www.postgresql.org/docs/current/storage-toast.html), entire `jsonb` value is written out to TOAST table, thus, GPU cannot process these values and invokes inefficient CPU-fallback operations.
    Regarding to the 2nd problem, you can extend table's storage option `toast_tuple_target` to enlarge the threshold for datum TOASTen.
}

@ja{
# 標準の雑多なデータ型

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
# Built-in miscellaneous types

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
# 標準の範囲型

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`int4range`       |`__int4range`     |14 bytes|    |
|`int8range`       |`__int8range`     |22 bytes|    |
|`tsrange`         |`__tsrange`       |22 bytes|    |
|`tstzrange`       |`__tstzrange`     |22 bytes|    |
|`daterange`       |`__daterange`     |14 bytes|    |
}
@en{
# Built-in range data types

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`int4range`       |`__int4range`     |14 bytes|    |
|`int8range`       |`__int8range`     |22 bytes|    |
|`tsrange`         |`__tsrange`       |22 bytes|    |
|`tstzrange`       |`__tstzrange`     |22 bytes|    |
|`daterange`       |`__daterange`     |14 bytes|    |
}
