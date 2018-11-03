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

@ja{
PG-StromにおいてもMPSの利用を推奨していますが、下記の制限事項に対して注意が必要です。
}
@en{
It is also recommended for PG-Strom to apply MPS, however, you need to pay attention for several limitations below.
}

@ja{
1個のMPSデーモンがサービスを提供可能なクライアントの数は最大48個（Pascal世代以前は16個）に制限されています。
そのため、GPUを使用するPostgreSQLプロセス（CPU並列処理におけるバックグラウンドワーカを含む）の数が常に48個以下（Pascal世代以前は16個以下）である事を、運用上担保する必要があります。
}

@en{
One MPS daemon can provide its service for up to 48 clients (16 clients if Pascal or older).
So, DB administration must ensure number of PostgreSQL processes using GPU (including background workers in CPU parallelism) is less than 48 (or 16 if Pascal).
}

@ja{
MPSはDynamic Parallelismを利用するGPUプログラムに対応していません。
SQLから自動生成されたGPUプログラムが当該機能を利用する事はありませんが、PL/CUDAユーザ定義関数がCUDAデバイスランタイムをリンクし、サブカーネルを呼び出すなどDynamic Parallelismの機能を利用する場合には当該制限に抵触します。
そのため、PL/CUDA関数の呼び出しにおいてはMPSを利用しません。
}
@en{
MPS does not support dynamic parallelism, and load GPU programs using the feature.
GPU programs automatically generated from SQL will never use dynamic parallelism, however, PL/CUDA user defined function may use dynamic parallelism if it links CUDA device runtime to invoke sub-kernels.
So, we don't use MPS for invocation of PL/CUDA functions.
}

@ja{
MPSのドキュメントにはGPUの動作モードを`EXCLUSIVE_PROCESS`に変更する事が推奨されていますが、PG-Stromを実行する場合は`DEFAULT`動作モードで動作させてください。
上記のPL/CUDAを含む、いくつかの処理では明示的にMPSの利用を無効化してCUDA APIを呼び出しているため、MPSデーモン以外のプロセスがGPUデバイスを利用可能である必要があります。
}
@en{
MPS document recommends to set compute-mode `EXCLUSIVE_PROCESS`, however, PG-Strom requires `DEFAULT` mode.
Several operations, including PL/CUDA above, call CUDA APIs with MPS disabled explicitly, so other processes than MPS daemon must be able to use GPU devices.
}

@ja{
MPSデーモンを起動するには以下の手順でコマンドを実行します。`<UID>`はPostgreSQLプロセスのユーザIDに置き換えてください。
}
@en{
The following commands start MPS daemon. Replace `<UID>` with user-id of PostgreSQL process.
}

```
$ nvidia-cuda-mps-control -d
$ echo start_server -uid <UID> | nvidia-cuda-mps-control
```

@ja{
`nvidia-smi`コマンドによって、MPSデーモンがGPUデバイスを使用している事が分かります。
}
@en{
`nvidia-smi` command shows MPS daemon is using GPU device.
}

```
$ nvidia-smi
Sat Nov  3 12:22:26 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.48                 Driver Version: 410.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:02:00.0 Off |                    0 |
| N/A   45C    P0    38W / 250W |     40MiB / 16130MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     11080      C   nvidia-cuda-mps-server                        29MiB |
+-----------------------------------------------------------------------------+
```

@ja:# ナレッジベース
@en:# Knowledge base

@ja{
PG-Stromプロジェクトのwikiサイトには、ノートと呼ばれる詳細な技術情報が公開されています。
}
@en{
We publish several articles, just called "notes", on the project wiki-site of PG-Strom.
}
[https://github.com/heterodb/pg-strom/wiki](https://github.com/heterodb/pg-strom/wiki)

