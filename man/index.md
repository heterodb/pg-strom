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

@ja:PG-Stromは以下のデータ型をGPUで利用する事ができます。
@en:PG-Strom support the following data types for use on GPU.

**Built-in numeric types**

@ja:|SQLデータ型|内部データ型|データ長|備考|
@en:|SQL data types|Internal data types|data length|memo|
|:----------|:-----------|:-------|:---|
|```smallint```|```cl_short```|2 bytes||
|```integer```|```cl_int```|4 bytes||
|```bigint```|```cl_long```|8 bytes||
|```real```|```cl_float```|4 bytes||
|```float```|```cl_double```|8 bytes||
|```numeric```|```cl_ulong```|@en{variable length}@ja{可変長}|@en{maps to 64bit internal format}@jp{64bitの内部形式で処理}|

**Built-in date and time types**

@ja:|SQLデータ型|内部データ型|データ長|備考|
@en:|SQL data types|Internal data types|data length|memo|
|:----------|:-----------|:-------|:---|
|```date```|```DateADT```|4 bytes||
|```time```|```TimeADT```|8 bytes||
|```timetz```|```TimeTzADT```|12 bytes||
|```timestamp```|```Timestamp```|8 bytes||
|```timestamptz```|```TimestampTz```|8 bytes||
|```interval```|```Interval```|16 bytes||

**Built-in miscellaneous types**

@ja:|SQLデータ型|内部データ型|データ長|備考|
@en:|SQL data types|Internal data types|data length|memo|
|:----------|:-----------|:-------|:---|
|```boolean```|```cl_bool```|1 byte||
|```money```|```cl_long```|8 bytes||
|```uuid```|```pg_uuid```|16 bytes||
|```macaddr```|```macaddr```|6 bytes||
|```inet```|```inet_struct```|7 bytes or 19 bytes||
|```cidr```|```inet_struct```|7 bytes or 19 bytes||

**Built-in variable length types**

@ja:|SQLデータ型|内部データ型|データ長|備考|
@en:|SQL data types|Internal data types|data length|memo|
|:----------|:-----------|:-------|:---|
|```bpchar```|```varlena *```|@en{variable length}@ja{可変長}||
|```varchar```|```varlena *```|@en{variable length}@ja{可変長}||
|```bytea```|```varlena *```|@en{variable length}@ja{可変長}||
|```text```|```varlena *```|@en{variable length}@ja{可変長}||

**Built-in range data types**

@ja:|SQLデータ型|内部データ型|データ長|備考|
@en:|SQL data types|Internal data types|data length|memo|
|:----------|:-----------|:-------|:---|
|```int4range```|```__int4range```|19 bytes||
|```int8range```|```__int8range```|||
|```tsrange```|```__tsrange```|||
|```tstzrange```|```__tstzrange```|||
|```daterange```|```__daterange```|||

**Extra Types**
@ja:|SQLデータ型|内部データ型|データ長|備考|
@en:|SQL data types|Internal data types|data length|memo|
|:----------|:-----------|:-------|:---|
|```float2```|```half_t```|2 bytes||


Built-in Types
==============






@ja:##関数と演算子
@en:## Functions and Operators

@ja:##情報スキーマ
@en:## Information Schema


@ja:##GUCパラメータ
@en:## GUC Parameters


### hoge

aaa

### monu


@ja:# リリースノート
@en:# Release Note

@ja:## PG-Strom v2.0リリース
@en:## PG-Strom v2.0 Release


