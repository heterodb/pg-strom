@ja:#PG-Strom v2.3リリース
@en:#PG-Strom v2.3 Release

<div style="text-align: right;">PG-Strom Development Team (1-Apr-2020)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v2.3における主要な機能強化は以下の通りです。

- GpuJoinのInnerバッファの構築がCPU並列に対応しました。
- Arrow_FdwがINSERT/TRUNCATEに対応しました。
- pg2arrowコマンドが追記モードに対応しました。
- mysql2arrowコマンドが追加されました。
}

@en{
Major changes in PG-Strom v2.3 includes:

- GpuJoin supports parallel construction of inner buffer
- Arrow_Fdw now becomes writable; supports INSERT/TRUNCATE.
- pg2arrow command supports 'append' mode.
- mysql2arrow command was added.
}

@ja:##動作環境
@en:##Prerequisites

@ja{
- PostgreSQL v10, v11, v12
- CUDA Toolkit 10.1 以降
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 6.0 以降 (Pascal or Volta)
}
@en{
- PostgreSQL v10, v11, v12
- CUDA Toolkit 10.1 or later
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 6.0 or later (Pascal or Volta)
}

@ja:##新機能
@en:##New Features

@ja{
- GpuJoinのInnerバッファの構築がCPU並列に対応
    - 従来はGpuJoinのInner側バッファの構築はバックエンドプロセスのみが行っていました。この制約により、パーティション化されたテーブルの並列スキャンが極端に遅延するという問題がありました。
    - 本バージョンでの機能強化により、バックエンド、ワーカープロセスのどちらでもInner側バッファを構築する事が可能となりました。パーティション化テーブルをスキャンする場合でも、各パーティション子テーブルに割り当てられたプロセスが直ちにGpuJoin処理を開始する事ができるようになります。
- Partition-wise Asymmetric GpuJoinの再設計
    - 全体的なデザインの再設計を行い、適切な局面において多段GpuJoinが選択されやすくなるよう改良を行いました。
- Arrow_FdwがINSERT/TRUNCATEに対応しました。
    - Arrow_Fdw外部テーブルに対して、`INSERT`によるバルクロードと、`pgstrom.arrow_fdw_truncate`によるTRUNCATE処理を行う事が可能となりました。
- CuPy連携とデータフレームのGPUエクスポート
    - Arrow_Fdw外部テーブルの内容をGPUデバイスメモリ上のデータフレームにロードし、これをCuPyの`cupy.ndarray`オブジェクトとしてPythonスクリプトから参照する事が可能となりました。
- pg2arrowコマンドが追記モードに対応しました。
    - `pg2arrow`コマンドに`--append`オプションが追加され、既存のApache Arrowに対して追記を行う事が可能となりました。
    - また同時に、`SELECT * FROM table`の別名表記として`-t table`オプションが追加されました。
- mysql2arrowコマンドを追加されました。
    - PostgreSQLではなく、MySQLに接続してクエリを実行し、その結果をApache Arrowファイルとして保存する`mysql2arrow`コマンドを追加しました。
    - 列挙型のデータも通常のUtf8型として保存する（DictionaryBatchを使用しない）以外は、`pg2arrow`と同等の機能を持っています。
- リグレッションテストを追加しました
    - PostgreSQLのリグレッションテストフレームワークに合わせて、幾つかの基本的なテストケースを追加しています。
}
@en{
- GpuJoin supports parallel construction of inner buffer
    - The older version construct inner buffer of GpuJoin by the backend process only. This restriction leads a problem; parallel scan of partitioned table delays extremely.
    - This version allows both of the backend and worker processes to construct inner buffer. In case when we scan a partitioned table, any processes that is assigned to a particular child table can start GpuJoin operations immediately.
- Refactoring of the partition-wise asymmetric GpuJoin
    - By the refactoring of the partition-wise asymmetric GpuJoin, optimizer becomes to prefer multi-level GpuJoin in case when it offers cheaper execution cost.
- Arrow_Fdw becomes writable; INSERT/TRUNCATE supported
    - Arrow_Fdw foreign table allows bulk-loading by `INSERT` and data elimination by `pgstrom.arrow_fdw_truncate`.
- pg2arrow command supports 'append' mode.
    - We added `--append` option for `pg2arrow` command. As literal, it appends query results on existing Apache Arrow file.
    - Also, `-t table` option was added as an alias of `SELECT * FROM table`.
- mysql2arrow command was added.
    - We added `mysql2arrow` command that connects to MySQL server, not PostgreSQL, and write out SQL query results as Apache Arrow files.
    - It has equivalent functionality to `pg2arrow` except for enum data type. `mysql2arrow` saves enum values as flat Utf8 values without DictionaryBatch chunks.
- Regression test was added
    - Several test cases were added according to the PostgreSQL regression test framework.
}

@ja:##修正された主な不具合
@en:##Significant bug fixes

@ja{
- GPUデバイス関数/型のキャッシュ無効化のロジックを改善
    - ALTERコマンドの実行時、全てのGPUデバイス関数/型のメタ情報キャッシュを無効化していましたが、実際に無効化の必要のあるエントリのみをクリアするよう修正を行いました。
- GROUP BYで同じ列を偶数回指定した際に極端なパフォーマンスの低下を修正
    - GROUP BYのキー値が複数ある時に、GpuPreAggはハッシュ値をXORで結合していました。そのため、同じ列を偶数回指定した場合には常にハッシュインデックスが0になるという問題がありました。適当なランダム化処理を加える事でハッシュ値が分散するよう修正しています。
- 潜在的なGpuScan無限ループの問題を修正
    - SSD2GPU Direct SQLの使用時、変数の未初期化によりGpuScanが無限ループに陥る可能性がありました。
- 潜在的なGpuJoinのGPUカーネルクラッシュ
    - 3個以上のテーブルを結合するGpuJoinで、変数の未初期化によりGPUカーネルのクラッシュを引き起こす可能性がありました。
}
@en{
- Revised cache invalidation logic for GPU device functions / types
    - The older version had invalidated all the metadata cache entries of GPU device functions / type on execution of ALTER command. It was revised to invalidate the entries that are actually updated.
- Revised extreme performance degradation if GROUP BY has same grouping key twice or even number times.
    - GpuPreAgg combined hash values of grouping key of GROUP BY using XOR. So, if case when same column appeared even number time, it always leads 0 for hash-index problematically. Now we add a randomization for better hash distribution.
- Potential infinite loop on GpuScan
    - By uninitialized values, GpuScan potentially goes to infinite loop when SSD2GPU Direct SQL is available.
- Potential GPU kernel crash on GpuJoin
    - By uninitialized values, GpuJoin potentially makes GPU kernel crash when 3 or more tables are joined.
}


@ja:##廃止された機能
@en:##Deprecated Features

@ja{
- PostgreSQL v9.6サポート
    - PostgreSQL v9.6のCustomScan APIには、動的共有メモリ(DSM)の正しいハンドリングに必要な幾つかのAPIが欠けており、v10以降と共通のコードを保守する上で障害となっていました。これらの問題から、本バージョンでは PostgreSQL v9.6 はサポート外となります。
- PL/CUDA
    - ユースケースを分析した結果、独自のプログラミング環境よりも、Python言語などユーザの使い慣れた言語環境の方が望ましい事が分かりました。
    - 今後は、Arrow_FdwのGPUエクスポート機能とPL/Python経由でのCuPy呼出しを併用する事で、In-database機械学習/統計解析の代替手段となります。
- Gstore_Fdw
    - 本機能は、書き込み可能Arrow_FdwとGPUエクスポート機能により代替されました。
- Largeobject～GPU間エクスポート/インポート
    - ユースケースを分析した結果、本機能は不要と判断しました。
}

@en{
- PostgreSQL v9.6 Support
    - CustomScan API in PostgreSQL v9.6 lacks a few APIs to handle dynamic shared memory (DSM). It has been a problem to handle a common code for v10 or later. To avoid the problem, we dropped PostgreSQL v9.6 support in this version.
- PL/CUDA
    - According to the usecase analytics, users prefer familiar programming language environment like Python, rather than own special environment.
    - A combination of Arrow_Fdw's GPU export functionality and CuPy invocation at PL/Python is a successor of PL/CUDA, for in-database machine-learning / statistical analytics.
- Gstore_Fdw
    - This feature is replaced by the writable Arrow_Fdw and its GPU export functionality.
- Largeobject export to/import from GPU
    - According to the usecase analytics, we determined this feature is not needed.
}

