@ja{
#データ型
PG-Stromは以下のデータ型をGPUで利用する事ができます。
}
@en{
#Data Types
PG-Strom support the following data types for use on GPU device.
}

@ja{
## 数値データ型

|SQLデータ型       |データ長|備考|
|:-----------------|:-------|:---|
|`smallint`        |2 bytes |16bit 整数型|
|`integer`         |4 bytes |32bit 整数型|
|`bigint`          |8 bytes |64bit 整数型|
|`float2`          |2 bytes |半精度浮動小数点型。PG-Stromによる独自拡張。|
|`real`            |4 bytes |単精度浮動小数点型|
|`float`           |8 bytes |倍精度浮動小数点型|
|`numeric`         |可変長  |内部形式は128bit固定少数点型|
}
@en{
## Numeric types

|SQL data types    |Length  |Memo                   |
|:-----------------|:-------|:----------------------|
|`smallint`        |2 bytes |16bit integer data type|
|`integer`         |4 bytes |32bit integer data type|
|`bigint`          |8 bytes |64bit integer data type|
|`float2`          |2 bytes |Half precision data type. An extra data type by PG-Strom|
|`real`            |4 bytes |Single precision floating-point data type|
|`float`           |8 bytes |Double precision floating-point data type|
|`numeric`         |variable|mapped to 128bit fixed-point numerical internal format|
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
## 標準の日付時刻型

|SQLデータ型       |データ長|備考        |
|:-----------------|:-------|:-----------|
|`date`            |4 bytes |日付データ型|
|`time`            |8 bytes |時刻データ型|
|`timetz`          |12 bytes|タイムゾーン付き時刻データ型|
|`timestamp`       |8 bytes |タイムスタンプ型|
|`timestamptz`     |8 bytes |タイムゾーン付きタイムスタンプ型|
|`interval`        |16 bytes|時間間隔型  |
}

@en{
## Built-in date and time types

|SQL data types    |Length  |Memo|
|:-----------------|:-------|:---|
|`date`            |4 bytes |Date data type     |
|`time`            |8 bytes |Time data type     |
|`timetz`          |12 bytes|Time with timezone data type|
|`timestamp`       |8 bytes |Timestamp data type|
|`timestamptz`     |8 bytes |Timestamp with timezone data type|
|`interval`        |16 bytes|Interval data type|
}

@ja{
## 標準の可変長データ型

|SQLデータ型       |データ長|備考            |
|:-----------------|:-------|:---------------|
|`bpchar`          |可変長  |可変長テキスト型（空白パディングあり）|
|`varchar`         |可変長  |可変長テキスト型|
|`text`            |可変長  |可変長テキスト型|
|`bytea`           |可変長  |可変長バイナリ型|
}

@en{
## Built-in variable length types

|SQL data types    |Length  |Memo          |
|:-----------------|:-------|:-------------|
|`bpchar`          |variable|variable length text with whitespace paddings|
|`varchar`         |variable|variable length text type|
|`text`            |variable|variable length text type|
|`bytea`           |variable|variable length binary type|
}

@ja{
## 標準の非構造データ型

|SQLデータ型      |データ長|備考|
|:----------------|:-------|:---|
|`jsonb`          |可変長  |    |

!!! Note
    `jsonb`データ型をGPUで処理させる場合には、次の2つの点に留意してください。
    実際に参照されない属性もストレージから読み出し、GPUに転送する必要があるため、I/Oバスの利用効率は必ずしも良くないデータ型である事。データ長が[TOAST化](https://www.postgresql.jp/document/current/html/storage-toast.html)の閾値（通常は2kB弱）を越えてしまった場合、`jsonb`データ全体がTOASTテーブルへ書き出されるため、GPU側では処理できず非効率なCPU-fallback処理を呼び出してしまう事。
    後者の問題に対しては、テーブルのストレージオプション`toast_tuple_target`を拡大し、TOAST化の閾値を引き上げる事である程度は回避する事も可能です。
}
@en{
## Built-in unstructured data types

|SQL data types   |Length  |Memo|
|:----------------|:-------|:---|
|`jsonb`          |variable|    |

!!! Note
    Pay attention for the two points below, when GPU processes `jsonb` data types.
    `jsonb` is not performance efficient data types because it has to load unreferenced attributes onto GPU from the storage, so tend to consume I/O bandwidth by junk data.
    In case when `jsonb` data length exceeds the threshold of [datum TOASTen](https://www.postgresql.org/docs/current/storage-toast.html), entire `jsonb` value is written out to TOAST table, thus, GPU cannot process these values and invokes inefficient CPU-fallback operations.
    Regarding to the 2nd problem, you can extend table's storage option `toast_tuple_target` to enlarge the threshold for datum TOASTen.
}

@ja{
## 標準の雑多なデータ型

|SQLデータ型       |データ長|備考          |
|:-----------------|:-------|:-------------|
|`boolean`         |1 byte  |論理値データ型|
|`money`           |8 bytes |通貨データ型  |
|`uuid`            |16 bytes|UUIDデータ型  |
|`macaddr`         |6 bytes |ネットワークMACアドレス型|
|`inet`            |7 or 19 bytes|ネットワークアドレス型|
|`cidr`            |7 or 19 bytes|ネットワークアドレス型|


}
@en{
## Built-in miscellaneous types

|SQL data types    |Length  |Memo             |
|:-----------------|:-------|:----------------|
|`boolean`         |1 byte  |Boolean data type|
|`money`           |8 bytes |Money data type  |
|`uuid`            |16 bytes|UUID data type   |
|`macaddr`         |6 bytes |Network MAC address data type|
|`inet`            |7 or 19 bytes|Network address data type|
|`cidr`            |7 or 19 bytes|Network address data type|
}

@ja{
## 標準の範囲型

|SQLデータ型       |データ長|備考             |
|:-----------------|:-------|:----------------|
|`int4range`       |14 bytes|32bit整数値範囲型|
|`int8range`       |22 bytes|64bit整数値範囲型|
|`tsrange`         |22 bytes|タイムスタンプ範囲型|
|`tstzrange`       |22 bytes|タイムゾーン付きタイムスタンプ範囲型|
|`daterange`       |14 bytes|日付データ範囲型 |
}
@en{
## Built-in range data types

|SQL data types    |Length  |Memo                       |
|:-----------------|:-------|:--------------------------|
|`int4range`       |14 bytes|Range type of 32bit integer|
|`int8range`       |22 bytes|Range type of 64bit integer|
|`tsrange`         |22 bytes|Range type of timestamp data|
|`tstzrange`       |22 bytes|Range type of timestamp with timezone data|
|`daterange`       |14 bytes|Range type of date type|
}

@ja{
## PostGISデータ型

|SQLデータ型       |データ長|備考                         |
|:-----------------|:-------|:----------------------------|
|`geometry`        |可変長  |PostGISジオメトリオブジェクト|
}
@en{
## PostGIS data types

|SQL data types   |Length  |Memo                      |
|:----------------|:-------|:-------------------------|
|`geometry`       |variable|Geometry object of PostGIS|
}
