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

@ja:#Arrow_Fdw関連
@en:#Arrow_Fdw Supports

@ja{
|関数|戻り値|説明|
|:---|:----:|:---|
|`pgstrom.arrow_fdw_truncate(regclass)`|`bool`|指定されたArrow_Fdw外部テーブルの内容を全て消去します。Arrow_Fdw外部テーブルは`writable`である必要があります。|
}
@en{
|Function|Result|Description|
|:-------|:----:|:----------|
|`pgstrom.arrow_fdw_truncate(regclass)`|`bool`|It truncates contents of the specified Arrow_Fdw foreign table. Arrow_Fdw foreign table must be `writable`.|
}

@ja:#GPUデータフレーム関数
@en:#GPU Data Frame Functions

@ja{
|関数|戻り値|説明|
|:---|:----:|:---|
|`pgstrom.arrow_fdw_export_cupy(regclass, text[], int)`       |`text`|指定された列のArrow_Fdw外部テーブルの内容をcuPyのデータフレーム(`cupy.ndarray`)としてエクスポートします。GPUバッファはセッション終了時に自動的に解放されます。|
|`pgstrom.arrow_fdw_export_cupy_pinned(regclass, text[], int)`|`text`|指定された列のArrow_Fdw外部テーブルの内容をcuPyのデータフレーム(`cupy.ndarray`)としてエクスポートします。GPUバッファはピンニングされ、セッション終了後も有効です。|
|`pgstrom.arrow_fdw_put_gpu_buffer(text)`                     |`bool`|上記の関数でエクスポートされたGPUバッファを解放します。|
|`pgstrom.arrow_fdw_unpin_gpu_buffer(text)`                   |`bool`|上記の関数でエクスポートされたGPUバッファのピンニングを解除します。|
}
@en{
|Function|Result|Description|
|:-------|:----:|:----------|
|`pgstrom.arrow_fdw_export_cupy(regclass, text[], int)`       |`text`|It exports the specified columns of Arrow_Fdw foreign table as cuPy's data frame(`cupy.ndarray`). GPU buffer shall be released automatically on session closed.|
|`pgstrom.arrow_fdw_export_cupy_pinned(regclass, text[], int)`|`text`|It exports the specified columns of Arrow_Fdw foreign table as cuPy's data frame(`cupy.ndarray`), as pinned GPU buffer; that is available after the session closed. |
|`pgstrom.arrow_fdw_put_gpu_buffer(text)`                     |`bool`|It unreference the GPU buffer that is exported with the above functions.
|`pgstrom.arrow_fdw_unpin_gpu_buffer(text)`                   |`bool`|It unpin the GPU buffer that is exported with the above functions.
}

@ja:#テストデータ生成関数
@en:#Test Data Generation

@ja{
|関数|戻り値|説明|
|:---|:----:|:---|
|`pgstrom.random_setseed(int)`                                                             |`void`     |乱数の系列を初期化します。|
|`pgstrom.random_int(float=0.0, bigint=0, bigint=INT_MAX)`                                 |`bigint`   |`bigint`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_float(float=0.0, float=0.0, float=1.0)`                                   |`float`    |`float`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_date(float=0.0, date='2015-01-01', date='2025-12-31')`                    |`date`     |`date`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_time(float=0.0, time='00:00:00', time='23:59:59')`                        |`time`     |`time`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_timetz(float=0.0, time='00:00:00', time='23:59:59')`                      |`timetz`   |`timetz`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_timestamp(float=0.0, timestamp='2015-01-01', timestamp='2025-01-01')`     |`timestamp`|`timestamp`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_macaddr(float=0.0, macaddr='ab:cd:00:00:00', macaddr='ab:cd:ff:ff:ff:ff')`|`macaddr`  |`macaddr`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_inet(float=0.0, inet='192.168.0.1/16')`                                   |`inet`     |`inet`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_text(float=0.0, text='test_**')`                                          |`text`     |`text`型のランダムデータを生成します。第二引数の'*'文字をランダムに置き換えます。|
|`pgstrom.random_text_len(float=0.0, int=10)`                                              |`text`     |`text`型のランダムデータを指定文字列長の範囲内で生成します。|
|`pgstrom.random_int4range(float=0.0, bigint=0, bigint=INT_MAX)`                           |`int4range`|`int4range`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_int8range(float=0.0, bigint=0, bigint=LONG_MAX)`                          |`int8range`|`int8range`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_tsrange(float=0.0, timestamp='2015-01-01', timestamp='2025-01-01')`       |`tsrange`  |`tsrange`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_tstzrange(float=0.0, timestamptz='2015-01-01', timestamptz='2025-01-01')` |`tstzrange`|`tstzrange`型のランダムデータを指定の範囲内で生成します。|
|`pgstrom.random_daterange(float=0.0, date='2015-01-01', date='2025-12-31')`               |`daterange`|`daterange`型のランダムデータを指定の範囲内で生成します。|
}
@en{
|Function|Result|Description|
|:-------|:----:|:----------|
|`pgstrom.random_setseed(int)`                                                             |`void`     |It initializes the random seed.|
|`pgstrom.random_int(float=0.0, bigint=0, bigint=INT_MAX)`                                 |`bigint`   |It generates random data in `bigint` type within the range.|
|`pgstrom.random_float(float=0.0, float=0.0, float=1.0)`                                   |`float`    |It generates random data in `float` type within the range.|
|`pgstrom.random_date(float=0.0, date='2015-01-01', date='2025-12-31')`                    |`date`     |It generates random data in `date` type within the range.|
|`pgstrom.random_time(float=0.0, time='00:00:00', time='23:59:59')`                        |`time`     |It generates random data in `time` type within the range.|
|`pgstrom.random_timetz(float=0.0, time='00:00:00', time='23:59:59')`                      |`timetz`   |It generates random data in `timetz` type within the range.|
|`pgstrom.random_timestamp(float=0.0, timestamp='2015-01-01', timestamp='2025-01-01')`     |`timestamp`|It generates random data in `timestamp` type within the range.|
|`pgstrom.random_macaddr(float=0.0, macaddr='ab:cd:00:00:00', macaddr='ab:cd:ff:ff:ff:ff')`|`macaddr`  |It generates random data in `macaddr` type within the range.|
|`pgstrom.random_inet(float=0.0, inet='192.168.0.1/16')`                                   |`inet`     |It generates random data in `inet` type within the range.|
|`pgstrom.random_text(float=0.0, text='test_**')`                                          |`text`     |It generates random data in `text` type. The '*' characters in 2nd argument shall be replaced randomly.|
|`pgstrom.random_text_len(float=0.0, int=10)`                                              |`text`     |It generates random data in `text` type within the specified length.|
|`pgstrom.random_int4range(float=0.0, bigint=0, bigint=INT_MAX)`                           |`int4range`|It generates random data in `int4range` type within the range.|
|`pgstrom.random_int8range(float=0.0, bigint=0, bigint=LONG_MAX)`                          |`int8range`|It generates random data in `int8range` type within the range.|
|`pgstrom.random_tsrange(float=0.0, timestamp='2015-01-01', timestamp='2025-01-01')`       |`tsrange`  |It generates random data in `tsrange` type within the range.|
|`pgstrom.random_tstzrange(float=0.0, timestamptz='2015-01-01', timestamptz='2025-01-01')` |`tstzrange`|It generates random data in `tstzrange` type within the range.|
|`pgstrom.random_daterange(float=0.0, date='2015-01-01', date='2025-12-31')`               |`daterange`|It generates random data in `daterange` type within the range.|
}

@ja:#その他の関数
@en:#Other Functions

@ja{
|関数|戻り値|説明|
|:---|:----:|:---|
|`pgstrom.license_query()`|`text`|現在ロードされている商用サブスクリプションを表示します。|
}
@en{
|Function|Result|Description|
|:-------|:----:|:----------|
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
現在のところ、Arrow_FdwのPython連携機能がこれを使用しています。

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
Right now, only Python collaboration of Arrow_Fdw uses this feature.

|Name        |Data Type |Description|
|:-----------|:---------|:----------|
|device_nr   |`int`     |GPU device number
|handle      |`bytea`   |IPC handle of the preserved device memory
|owner       |`regrole` |Owner of the preserved device memory
|length      |`bigint`  |Length of the preserved device memory in bytes
|ctime       |`timestamp with time zone`|Timestamp when the preserved device memory is created

}
