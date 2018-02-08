@ja:#概要
@en:# Overview

@ja: 本章ではPG-Stromの概要、および開発者コミュニティについて説明します。
@en: This chapter introduces the overview of PG-Strom, and developer's community.

@ja{
## PG-Stromとは?
PG-StromはPostgreSQL v9.6および以降のバージョン向けに設計された拡張モジュールで、GPU(Graphic Processor Unit)デバイスを利用する事で、大規模なデータセットに対するデータ解析やバッチ処理向けのSQLワークロードを高速化する事ができます。本モジュールは PostgreSQL のクエリオプティマイザやエグゼキュータと連携し、ユーザやアプリケーションから透過的に動作します。
PG-Stromの中核を構成するのは、SQL構文から自動的にGPUプログラムを生成するコードジェネレータと、デバイスあたり数千コアを持つGPUを有効に活用する非同期・並列実行エンジン、そして大量のデータを高速にプロセッサへと供給するためのストレージ周辺機能です。
これらの機構は一般的にOLAP(Online Analytical Processing)と呼ばれるワークロードに対して有効に作用しますが、更新系処理や同時多重処理には適していません。PostgreSQLのクエリオプティマイザはクエリの特性に応じて適切な実行計画を組み立てるため、PG-Stromが不得意な処理にはGPUは使用されず、PostgreSQL標準の実装が採用されます。
}
@en{
## What is PG-Strom?
PG-Strom is an extension module of PostgreSQL designed for version 9.6 or later. By utilization of GPU (Graphic Processor Unit) device, it enables to accelerate SQL workloads for data analytics, batch processing and so on. It cooperates with query-optimizer and -executor of PostgreSQL, and works transparently from users or applications.
The core of PG-Strom is consists of the code generator which build a GPU program from the supplied SQL statement automatically, asynchronous parallel execution engine which utilizes GPU devices with thousands cores per unit, and storage comprehensives to provide massive data chunks for the processors.
These mechanism efficiently works to OLAP(Online Analytical Processing) class workloads in general, however, not efficient to transactional or simultaneous multi-processing jobs. The query optimizer of PostgreSQL constructs appropriate query execution plan, therefore, it does not choose GPU for what GPU is not suitable instead of the PostgreSQL's built-in implementation.
}

@ja:### PG-Stromの機能
@en:### Brief feature of PG-Strom

#### 透過的なSQLワークロードの高速化
#### I/Oの高速化
#### In-database計算エンジン




@ja:### ライセンス
@en:### License


### 動作環境



## コミュニティ


### バグや障害の報告

### 新機能の提案


### サポートポリシー


## 関連情報

- PostgreSQLのドキュメントなど


@ja:#インストールと設定
@en:#Install and configuration

本章ではPG-Stromのインストールについて説明します。

@ja:#チェックリスト
@en:#Checklist

@ja:RPMによるインストール
@en:Installation with RPM



@ja:ソースからのインストール
@en:Installation from the source



@ja:インストール後の設定
@en:Post installation configuration




- チェックリスト
- RPMによるインストール
- ソースからのインストール
- OSの設定
- オプティマイザの設定
- MPSの利用について


@ja:#チュートリアル
@en:#User Tutorials

- 基本的な操作
- 無効化/有効化
- 実行計画の確認
- CPU並列クエリ
- GpuScan
- GpuJoin
- GpuPreAgg
- 性能のためのヒント
- バッファ・ストレージ
- リソース設定
- プロファイラの利用
- リグレッションテスト


@ja:# 特別な機能
@en:# Features

@en:## In-memory Columnar Cache
@ja:## In-memory 列指向キャッシュ



@en:## SSD-to-GPU Direct SQL Execution
@ja:## SSD-to-GPU ダイレクトSQL実行



## Gstore_fdw



# PL/CUDA


@ja:# リファレンス
@en:# References

@ja:## データ型
@en:## Supported Data Types

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
|```smallint```    |```cl_short```    |2 bytes |    |
|```integer```     |```cl_int```      |4 bytes |    |
|```bigint```      |```cl_long```     |8 bytes |    |
|```real```        |```cl_float```    |4 bytes |    |
|```float```       |```cl_double```   |8 bytes |    |
|```numeric```     |```cl_ulong```    |可変長  |64bitの内部形式にマップ|
}
@en{
**Built-in numeric types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|```smallint```    |```cl_short```    |2 bytes |    |
|```integer```     |```cl_int```      |4 bytes |    |
|```bigint```      |```cl_long```     |8 bytes |    |
|```real```        |```cl_float```    |4 bytes |    |
|```float```       |```cl_double```   |8 bytes |    |
|```numeric```     |```cl_ulong```    |variable length|mapped to 64bit internal format|
}

@ja{
!!! Note
    GPUが```numeric```型のデータを処理する際、実装上の理由からこれを64bitの内部表現に変換して処理します。
    これら内部表現への/からの変換は透過的に行われますが、例えば、桁数の大きな```numeric```型のデータは表現する事ができないため、PG-StromはCPU側でのフォールバック処理を試みます。したがって、桁数の大きな```numeric```型のデータをGPUに与えると却って実行速度が低下してしまう事になります。
    これを避けるには、GUCパラメータ```xxxxx```を使用して```numeric```データ型を含む演算式をGPUで実行しないように設定します。
}
@en{
!!! Note
    When GPU processes values in ```numeric``` data type, it is converted to an internal 64bit format because of implementation reason.
    It is transparently converted to/from the internal format, on the other hands, PG-Strom cannot convert ```numaric``` datum with large number of digits, so tries to fallback operations by CPU. Therefore, it may lead slowdown if ```numeric``` data with large number of digits are supplied to GPU device.
    To avoid the problem, turn off the GUC option ```xxxx``` not to run operational expression including ```numeric``` data types on GPU devices.
}



@ja{
**標準の日付時刻型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|```date```        |```DateADT```     |4 bytes |    |
|```time```        |```TimeADT```     |8 bytes |    |
|```timetz```      |```TimeTzADT```   |12 bytes|    |
|```timestamp```   |```Timestamp```   |8 bytes |    |
|```timestamptz``` |```TimestampTz``` |8 bytes |    |
|```interval```    |```Interval```    |16 bytes|time interval|
}

@en{
**Built-in date and time types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|```date```        |```DateADT```     |4 bytes |    |
|```time```        |```TimeADT```     |8 bytes |    |
|```timetz```      |```TimeTzADT```   |12 bytes|    |
|```timestamp```   |```Timestamp```   |8 bytes |    |
|```timestamptz``` |```TimestampTz``` |8 bytes |    |
|```interval```    |```Interval```    |16 bytes|    |
}

@ja{
**標準の可変長データ型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|```bpchar```      |```varlena *```   |可変長  |    |
|```varchar```     |```varlena *```   |可変長  |    |
|```bytea```       |```varlena *```   |可変長  |    |
|```text```        |```varlena *```   |可変長  |    |
}

@en{
**Built-in variable length types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|```bpchar```      |```varlena *```   |variable length|
|```varchar```     |```varlena *```   |variable length|
|```bytea```       |```varlena *```   |variable length|
|```text```        |```varlena *```   |variable length|
}

@ja{
**標準の雑多なデータ型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|```boolean```     |```cl_bool```     |1 byte  |    |
|```money```       |```cl_long```     |8 bytes |    |
|```uuid```        |```pg_uuid```     |16 bytes|    |
|```macaddr```     |```macaddr```     |6 bytes |    |
|```inet```        |```inet_struct``` |7 bytes or 19 bytes||
|```cidr```        |```inet_struct``` |7 bytes or 19 bytes||


}
@en{
**Built-in miscellaneous types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|```boolean```     |```cl_bool```     |1 byte  |    |
|```money```       |```cl_long```     |8 bytes |    |
|```uuid```        |```pg_uuid```     |16 bytes|    |
|```macaddr```     |```macaddr```     |6 bytes |    |
|```inet```        |```inet_struct``` |7 bytes or 19 bytes||
|```cidr```        |```inet_struct``` |7 bytes or 19 bytes||
}

@ja{
**標準の範囲型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|```int4range```   |```__int4range``` |14 bytes|    |
|```int8range```   |```__int8range``` |22 bytes|    |
|```tsrange```     |```__tsrange```   |22 bytes|    |
|```tstzrange```   |```__tstzrange``` |22 bytes|    |
|```daterange```   |```__daterange``` |14 bytes|    |
}
@en{
**Built-in range data types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|```int4range```   |```__int4range``` |14 bytes|    |
|```int8range```   |```__int8range``` |22 bytes|    |
|```tsrange```     |```__tsrange```   |22 bytes|    |
|```tstzrange```   |```__tstzrange``` |22 bytes|    |
|```daterange```   |```__daterange``` |14 bytes|    |
}

@ja{
**PG-Stromが追加で提供するデータ型**

|SQLデータ型       |内部データ形式    |データ長|備考|
|:-----------------|:-----------------|:-------|:---|
|```float2```      |```half_t```      |2 bytes |半精度浮動小数点数|

}
@en{
**Extra Types**

|SQL data types    |Internal format   |Length  |Memo|
|:-----------------|:-----------------|:-------|:---|
|```float2```      |```half_t```      |2 bytes |Half precision data type|
}



@ja:##関数と演算子
@en:## Functions and Operators



@ja:### デバイス情報関数
@en:### Device information functions


@ja:## システムビュー
@en:## System View

@ja{
PG-Stromは内部状態をユーザやアプリケーションに出力するためのシステムビューをいくつか提供しています。
これらのシステムビューは将来のバージョンで情報が追加される可能性があります。そのため、アプリケーションから```SELECT * FROM ...```によってこれらシステムビューを参照する事は避けてください。
}
@en{
PG-Strom provides several system view to export its internal state for users or applications.
The future version may add extra fields here. So, it is not recommended to reference these information schemas using ```SELECT * FROM ...```.
}

### pgstrom.device_info
@ja{
```pgstrom.device_info```システムビューは、PG-Stromが認識しているGPUのデバイス属性値を出力します。
GPUはモデルごとにコア数やメモリ容量、最大スレッド数などのスペックが異なっており、PL/CUDA関数などで直接GPUのプログラミングを行う場合には、これらの情報を元にソフトウェアを最適化する必要があります。

|名前        |データ型 |説明|
|:-----------|:--------|:---|
|device_nr   |int      |GPUデバイス番号 |
|aindex      |int      |属性インデックス|
|attribute   |text     |デバイス属性名  |
|value       |text     |デバイス属性値  |
}
@en{
```pgstrom.device_into``` system view exports device attributes of the GPUs recognized by PG-Strom.
GPU has different specification for each model, like number of cores, capacity of global memory, maximum number of threads and etc, user's software should be optimized according to the information if you try raw GPU programming with PL/CUDA functions.

|Name        |Data Type|Description|
|:-----------|:--------|:----------|
|device_nr   |int      |GPU device number |
|aindex      |int      |Attribute index |
|attribute   |text     |Attribute name |
|value       |text     |Value of the attribute |
}

### pgstrom.device_preserved_meminfo
@ja{
```pgstrom.device_preserved_meminfo```システムビューは、複数のPostgreSQLバックエンドプロセスから共有するために予め確保済みのGPUデバイスメモリ領域の情報を出力します。
現在のところ、gstore_fdwのみが本機能を使用しています。

|名前        |データ型 |説明|
|:-----------|:--------|:---|
|device_nr   |int      |GPUデバイス番号
|handle      |bytea    |確保済みGPUデバイスメモリのIPCハンドラ
|owner       |regrole  |確保済みGPUデバイスメモリの作成者
|length      |bigint   |確保済みGPUデバイスメモリのバイト単位の長さ
|ctime       |timestamp with time zone|確保済みGPUデバイスメモリの作成時刻
}
@en{
```pgstrom.device_preserved_meminfo``` system view exports information of the preserved device memory; which can be shared multiple PostgreSQL backend.
Right now, only gstore_fdw uses this feature.

|Name        |Data Type|Description|
|:-----------|:--------|:----------|
|device_nr   |int      |GPU device number
|handle      |bytea    |IPC handle of the preserved device memory
|owner       |regrole  |Owner of the preserved device memory
|length      |bigint   |Length of the preserved device memory in bytes
|ctime       |timestamp with time zone|Timestamp when the preserved device memory is created

}

### pgstrom.ccache_info
@ja{
```pgstrom.ccache_info```システムビューは、列指向キャッシュの各チャンク（128MB単位）の情報を出力します。

|名前        |データ型 |説明|
|:-----------|:--------|:---|
|database_id |oid      |データベースID |
|table_id    |regclass |テーブルID |
|block_nr    |int      |チャンクの先頭ブロック番号 |
|nitems      |bigint   |チャンクに含まれる行数 |
|length      |bigint   |キャッシュされたチャンクのサイズ |
|ctime       |timestamp with time zone|チャンク作成時刻|
|atime       |timestamp with time zone|チャンク最終アクセス時刻|
}
@en{
```pgstrom.ccache_info``` system view exports attribute of the columnar-cache chunks (128MB unit for each).

|Name        |Data Type|Description|
|:-----------|:--------|:---|
|database_id |oid      |Database Id |
|table_id    |regclass |Table Id |
|block_nr    |int      |Head block-number of the chunk |
|nitems      |bigint   |Number of rows in the chunk |
|length      |bigint   |Raw size of the cached chunk |
|ctime       |timestamp with time zone|Timestamp of the chunk creation |
|atime       |timestamp with time zone|Timestamp of the least access to the chunk |
}


### pgstrom.ccache_builder_info
@ja{
```pgstrom.ccache_builder_info```システムビューは、列指向キャッシュの非同期ビルダープロセスの情報を出力します。

|名前        |データ型 |説明|
|:-----------|:--------|:---|
|builder_id  |int      |列指向キャッシュ非同期ビルダーのID |
|state       |text     |ビルダープロセスの状態。（```shutdown```: 停止中、```startup```: 起動中、```loading```: 列指向キャッシュの構築中、```sleep```: 一時停止中）|
|database_id |oid      |ビルダープロセスが割り当てられているデータベースID |
|table_id    |regclass |```state```が```loading```の場合、読出し中のテーブルID |
|block_nr    |int      |```state```が```loading```の場合、読出し中のブロック番号 |
}
@en{
```pgstrom.ccache_builder_info``` system view exports information of asynchronous builder process of columnar cache.

|Name        |Data Type|Description|
|:-----------|:--------|:---|
|builder_id  |int      |Asynchronous builder Id of columnar cache |
|state       |text     |State of the builder process (```shutdown```, ```startup```, ```loading``` or ```sleep```) |
|database_id |oid      |Database Id where builder process is assigned on |
|table_id    |regclass |Table Id where the builder process is scanning on, if ```state``` is ```loading```. |
|block_nr    |int      |Block number where the builder process is scanning on, if ```state``` is ```loading```. |
}

@ja:##GUCパラメータ
@en:## GUC Parameters


### hoge

aaa

### monu


@ja:# リリースノート
@en:# Release Note

@ja:## PG-Strom v2.0リリース
@en:## PG-Strom v2.0 Release


