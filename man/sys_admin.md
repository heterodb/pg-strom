@ja:#<h1>システム管理</h1>
@en:#<h1>System Administration</h1>

<!-- リソース設定などの記述を加えるべき -->

@ja:# MPSデーモンの利用
@en:# Usage of MPS daemon

@ja{
PostgreSQLのようにマルチプロセス環境でGPUを使用する場合、GPU側コンテキストスイッチの低減やデバイス管理に必要なリソースの低減を目的として、MPS(Multi-Process Service)を使用する事が一般的なソリューションです。
}
@en{
In case when multi-process application like PostgreSQL uses GPU device, it is a well known solution to use MPS (Multi-Process Service) to reduce context switch on GPU side and resource consumption for device management.
}

[https://docs.nvidia.com/deploy/mps/index.html](https://docs.nvidia.com/deploy/mps/index.html)

<!--
@ja{
しかし、PG-Stromの利用シーンでは、MPSサービスの既知問題により正常に動作しないCUDA APIが存在し、以下のような限定された条件下を除いては使用すべきではありません。

- GPUを使用するPostgreSQLプロセス（CPU並列クエリにおけるバックグラウンドワーカを含む）の数が常に16個以下である。Volta世代のGPUの場合は48個以下である。
- gstore_fdwを使用しない事。
}
@en{
However, here is a known issue; some APIs don't work correctly user the use case of PG-Strom due to the problem of MPS daemon. So, we don't recomment to use MPS daemon except for the situation below:

- Number of PostgreSQL processes which use GPU device (including the background workers launched by CPU parallel execution) is always less than 16. If Volta generation, it is less than 48.
- gstore_fdw shall not be used.
}

@ja{
これは`CUipcMemHandle`を用いてプロセス間でGPUデバイスメモリを共有する際に、MPSサービス下のプロセスで獲得したGPUデバイスメモリを非MPSサービス下のプロセスでオープンできない事で、GpuJoinが使用するハッシュ表をバックグラウンドワーカー間で共有できなくなるための制限事項です。

この問題は既にNVIDIAへ報告し、新しいバージョンのCUDA Toolkitにおいて修正されるとの回答を得ています。
}
@en{
This known problem is, when we share GPU device memory inter processes using `CUipcMemHandle`, a device memory region acquired by the process under MPS service cannot be opened by the process which does not use MPS. This problem prevents to share the inner hash-table of GpuJoin with background workers on CPU parallel execution.

This problem is already reported to NVIDIA, then we got a consensu to fix it at the next version of CUDA Toolkit.
}
-->

@ja{
一方、現在のMPSサービスにはいくつかの[制限事項](https://docs.nvidia.com/deploy/mps/index.html#topic_3_3_2)があり、これとPG-Stromの利用する一部機能が被っているため、MPSサービスとPG-Stromを併用する事はできません。PG-Stromを利用する際にはMPSサービスを停止してください。

!!!note
    具体的には、GpuPreAggのGPUカーネル関数が内部のハッシュ表を動的に拡大する際に使用する`cudaDeviceSynchronize()`デバイスランタイム関数が、制限事項であるDynamic Parallelism機能を使用しているため、上記の制限に抵触します。
}
@en{
On the other hands, the current version of MPS daemon has [some limitations](https://docs.nvidia.com/deploy/mps/index.html#topic_3_3_2) which overlap with a part of features of PG-Strom, therefore, you cannot use MPS daemon for PG-Strom. Disables MPS daemon when PG-Strom works.

!!!note
    For details, the `cudaDeviceSynchronize()` device runtime function internally uses dynamic parallelism that is restricted under MPS, when GpuPreAgg's GPU kernel function expands internal hash table on the demand.
}

@ja:# ナレッジベース
@en:# Knowledge base

@ja{
PG-Stromプロジェクトのwikiサイトには、ノートと呼ばれる詳細な技術情報が公開されています。
}
@en{
We publish several articles, just called "notes", on the project wiki-site of PG-Strom.
}
[https://github.com/heterodb/pg-strom/wiki](https://github.com/heterodb/pg-strom/wiki)

