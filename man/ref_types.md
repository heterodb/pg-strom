@ja:#データ型
@en:#Data Types
@ja{
PG-Stromは以下のデータ型をGPUで利用する事ができます。
}
@en{
PG-Strom support the following data types for use on GPU device.
}

@ja:## 数値データ型
@en:## Numeric types
@ja{
`int1` [データ長: 1byte]
:   8bit整数型。PG-Stromによる独自拡張
}
@en{
`int1` [length: 1byte]
:   8bit integer data type; enhanced data type by PG-Strom
}

@ja{
`int2` (別名 `smallint`)  [データ長: 2bytes]
:   16bit整数型
}
@en{
`int2` (alias `smallint`)  [length: 2bytes]
:   16bit integer data type
}

@ja{
`int4` (別名 `int`)  [データ長: 4bytes]
:   32bit整数型
}
@en{
`int4` (alias `int`)  [length: 4bytes]
:   32bit integer data type
}

@ja{
`int8` (別名 `bigint`)  [データ長: 8bytes]
:   64bit整数型
}
@en{
`int8` (alias `bigint`)  [length: 8bytes]
:   64bit integer data type
}

@ja{
`float2`  [データ長: 2bytes]
:   半精度浮動小数点型。PG-Stromによる独自拡張

!!! Note
    GPUでは半精度浮動小数点型がハードウェアでサポートされていますが、CPU(x86_64プロセッサ)では未対応です。そのため、`float2`データ型をCPUで処理する場合には、これを一度`float`や`double`型に変換した上で演算を行います。そのため、GPUのように`float2`の方が演算速度で有利という事はありません。機械学習や統計解析用途にデータ量を抑制するための機能です。
}
@en{
`float2`  [length: 2bytes]
:   Half precision data type; enhanced data type by PG-Strom

!!! Note
    Even though GPU supports half-precision floating-point numbers by hardware, CPU (x86_64 processor) does not support it yet. So, when CPU processes `float2` data types, it transform them to `float` or `double` on calculations. So, CPU has no advantages for calculation performance of `float2`, unlike GPU. It is a feature to save storage/memory capacity for machine-learning / statistical-analytics. 
}

@ja{
`float4` (別名 `real`) [データ長: 4bytes]
:   単精度浮動小数点型
}
@en{
`float4` (alias `real`) [length: 4bytes]
:   Single precision floating-point data type
}

@ja{
`float8` (別名 `double precision`) [データ長: 8bytes]
:   倍精度浮動小数点型
}
@en{
`float8` (alias `double precision`) [length: 8bytes]
:   Double precision floating-point data type
}

@ja{
`numeric` [データ長: 可変]
:   実数型。GPU側では128bit固定小数点として扱われる。

!!! Note
    GPUが`numeric`型のデータを処理する際、実装上の理由からこれを128bit固定少数点の内部表現に変換して処理します。（これは Apache Arrow の`Decimal`型と同一の形式です）
    これら内部表現への/からの変換は透過的に行われますが、例えば、桁数の大きな`numeric`型のデータは表現する事ができないため、PG-StromはCPU側でのフォールバック処理を試みます。したがって、桁数の大きな`numeric`型のデータをGPUに与えると却って実行速度が低下してしまう事になります。
    これを避けるには、GUCパラメータ`pg_strom.enable_numeric_type`を使用して`numeric`データ型を含む演算式をGPUで実行しないように設定します。
}
@en{
`numeric` [length: variable]
:   Real number data type; handled as a 128bit fixed-point value in GPU

!!! Note
    When GPU processes values in `numeric` data type, it is converted to an internal 128bit fixed-point number because of implementation reason. (This layout is identical to `Decimal` type in Apache Arrow.)
    It is transparently converted to/from the internal format, on the other hands, PG-Strom cannot convert `numaric` datum with large number of digits, so tries to fallback operations by CPU. Therefore, it may lead slowdown if `numeric` data with large number of digits are supplied to GPU device.
    To avoid the problem, turn off the GUC option `pg_strom.enable_numeric_type` not to run operational expression including `numeric` data types on GPU devices.
}

@ja:## 日付時刻型
@en:## Date and time types
@ja{
`date` [データ長: 4bytes]
:   日付データ型
}
@en{
`date` [length: 4bytes]
:   Date data type
}

@ja{
`time` (別名 `time without time zone`) [データ長: 8bytes]
:   時刻データ型
}
@en{
`time` (alias `time without time zone`) [length: 8bytes]
:   Time data type
}

@ja{
`timetz` (別名 `time with time zone`) [データ長: 12bytes]
:   時刻データ型（タイムゾーン付き）
}
@en{
`timetz` (alias `time with time zone`) [length: 12bytes]
:   Time with timezone data type
}

@ja{
`timestamp` (別名 `timestamp without time zone`) [データ長: 8bytes]
:   タイムスタンプ型
}
@en{
`timestamp` (alias `timestamp without time zone`) [length: 8bytes]
:   Timestamp data type
}

@ja{
`timestamptz` (別名 `timestamp with time zone`) [データ長: 8bytes]
:   タイムスタンプ型（タイムゾーン付き）
}
@en{
`timestamptz` (alias `timestamp with time zone`) [length: 8bytes]
:   Timestamp with timezone data type
}

@ja{
`interval` [データ長: 16bytes]
:   時間間隔型
}
@en{
`interval` [length: 16bytes]
:   Interval data type
}

@ja:## 可変長データ型
@en:## Variable length types
@ja{
`bpchar` [データ長: 可変長]
:   可変長テキスト型（空白パディングあり）
}
@en{
`bpchar` [length: variable]
:   variable length text with whitespace paddings
}

@ja{
`varchar` [データ長: 可変長]
:   可変長テキスト型
}
@en{
`varchar` [length: variable]
:   variable length text type
}

@ja{
`text` [データ長: 可変長]
:   可変長テキスト型
}
@en{
`text` [length: variable]
:   variable length text type

}

@ja{
`bytea` [データ長: 可変長]
:   可変長バイナリ型
}
@en{
`bytea` [length: variable]
:   variable length binary type
}

@ja:## 非構造データ型
@en:## unstructured data types
@ja{
`jsonb` [length: 可変長]
:   バイナリインデックスを内包するJSONデータ型

!!! Note
    `jsonb`データ型をGPUで処理させる場合には、次の2つの点に留意してください。
    実際に参照されない属性もストレージから読み出し、GPUに転送する必要があるため、I/Oバスの利用効率は必ずしも良くないデータ型である事。データ長が[TOAST化](https://www.postgresql.jp/document/current/html/storage-toast.html)の閾値（通常は2kB弱）を越えてしまった場合、`jsonb`データ全体がTOASTテーブルへ書き出されるため、GPU側では処理できず非効率なCPU-fallback処理を呼び出してしまう事。
    後者の問題に対しては、テーブルのストレージオプション`toast_tuple_target`を拡大し、TOAST化の閾値を引き上げる事である程度は回避する事も可能です。
}
@en{
`jsonb` [length: variable]
:   JSON data type with binary indexed keys

!!! Note
    Pay attention for the two points below, when GPU processes `jsonb` data types.
    `jsonb` is not performance efficient data types because it has to load unreferenced attributes onto GPU from the storage, so tend to consume I/O bandwidth by junk data.
    In case when `jsonb` data length exceeds the threshold of [datum TOASTen](https://www.postgresql.org/docs/current/storage-toast.html), entire `jsonb` value is written out to TOAST table, thus, GPU cannot process these values and invokes inefficient CPU-fallback operations.
    Regarding to the 2nd problem, you can extend table's storage option `toast_tuple_target` to enlarge the threshold for datum TOASTen.
}

@ja:## 雑多なデータ型
@en:## Miscellaneous types
@ja{
`boolean` [データ長: 1byte]
:   論理値データ型
}
@en{
`boolean` [length: 1byte]
:   Boolean data type
}

@ja{
`money` [データ長: 8bytes]
:   通貨データ型
}
@en{
`money` [length: 8bytes]
:   Money data type
}

@ja{
`uuid`  [データ長: 16bytes]
:   UUIDデータ型
}
@en{
`uuid`  [length: 16bytes]
:   UUID data type
}

@ja{
`macaddr` [データ長: 6bytes]
:   ネットワークMACアドレス型
}
@en{
`macaddr` [length: 6bytes]
:   Network MAC address data type
}

@ja{
`inet` [データ長: 7 or 19bytes]
:   ネットワークアドレス型
}
@en{
`inet` [length: 7 or 19bytes]
:   Network address data type
}

@ja{
`cidr` [データ長: 7 or 19butes]
:   ネットワークアドレス型
}
@en{
`cidr` [length: 7 or 19butes]
:   Network address data type
}

@ja{
`cube` [データ長: 可変長]
:   `contrib/cube`によって提供される拡張データ型
}
@en{
`cube` [length: variable]
:   Extra data type provided by `contrib/cube`
}

<!--
@ja:## 範囲型
@en:## Range data types
@ja{
`int4range` [データ長: 14bytes]
:   32bit整数値範囲型
}
@en{
`int4range` [length: 14bytes]
:   Range type of 32bit integer
}

@ja{
`int8range` [データ長: 22bytes]
:   64bit整数値範囲型
}
@en{
`int8range` [length: 22bytes]
:   Range type of 64bit integer
}

@ja{
`tsrange` [データ長: 22bytes]
:   タイムスタンプ範囲型
}
@en{
`tsrange` [length: 22bytes]
:   Range type of timestamp data
}

@ja{
`tstzrange` [データ長: 22bytes]
:   タイムゾーン付きタイムスタンプ範囲型
}
@en{
`tstzrange` [length: 22bytes]
:   Range type of timestamp with timezone data
}

@ja{
`daterange` [データ長: 14bytes]
:   日付データ範囲型
}
@en{
`daterange` [length: 14bytes]
:   Range type of date type
}
-->

@ja:## ジオメトリ型
@en:## Geometry data types
@ja{
`geometry` [データ長: 可変]
:   PostGISジオメトリオブジェクト
}
@en{
`geometry` [length: variable]
:   Geometry object of PostGIS
}
@ja{
`box2df` [データ長: 16bytes]
:   2次元バウンディングボックス（GiSTインデックス用）
}
@en{
`box2df` [length: 16bytes]
:   2-dimension bounding box (used to GiST-index)
}


