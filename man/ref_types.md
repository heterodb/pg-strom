@ja{
<h1>データ型</h1>
PG-Stromは以下のデータ型をGPUで利用する事ができます。
}
@en{
<h1>Data Types</h1>
PG-Strom support the following data types for use on GPU device.
}

@ja{
# 標準の数値データ型

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
# Built-in numeric types

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

@ja{
# PG-Stromが追加で提供するデータ型

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|`float2`          |`half_t`          |2 bytes |半精度浮動小数点数|
|`reggstore`       |`cl_uint`         |4 bytes |gstore_fdwのregclass型。PL/CUDA関数呼出しで特別な扱い。|
}
@en{
# Extra Types

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|`float2`          |`half_t`          |2 bytes |Half precision data type|
|`reggstore`       |`cl_uint`         |4 bytes |Specific version of regclass for gstore_fdw. Special handling at PL/CUDA function invocation. |
}
