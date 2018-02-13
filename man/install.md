@ja:# インストール
@en:# Install


@ja:## チェックリスト
@en:## Checklist

@ja{
- **ハードウェア**
    - CUDA ToolkitのサポートするLinuxオペレーティングシステムを動作可能な x86_64 アーキテクチャのハードウェアが必要です。 
    - CPU、ストレージ、およびネットワークデバイスには特別な要件はありませんが、[note002:HW Validation List](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List)はハードウェア選定の上で参考になるかもしれません。
    - SSD-to-GPUダイレクトSQL実行を利用するにはNVMe規格に対応したSSDが必要で、GPUと同一のPCIe Root Complex配下に接続されている必要があります。
- **GPUデバイス**
    - PG-Stromを実行するには少なくとも一個のGPUデバイスがシステム上に必要です。これらはCUDA Toolkitでサポートされており、computing capability が6.0以降のモデル（Pascal世代以降）である必要があります。
    - [note001:GPU Availability Matrix](https://github.com/heterodb/pg-strom/wiki/001:-GPU-Availability-Matrix)により詳細な情報が記載されています。SSD-to-GPUダイレクトSQL実行の対応状況に関してもこちらを参照してください。
- **Operating System**
    - PG-Stromの実行には、CUDA Toolkitによりサポートされているx86_64アーキテクチャ向けのLinux OSが必要です。推奨環境はRed Hat Enterprise LinuxまたはCentOSのバージョン7.xシリーズです。
    - SSD-to-GPUダイレクトSQL実行を利用するには、Red Hat Enterprise Linux または CentOS のバージョン7.3以降が必要です。
- **PostgreSQL**
    - PG-Stromの実行にはPostgreSQLバージョン9.6以降が必要です。これは、Custom ScanインターフェースがCPU並列実行やGROUP BYに対応するため刷新され、拡張モジュールが提供するカスタム実行計画を自然な形で統合できるようになったためです。
- **CUDA Toolkit**
    - PG-Stromの実行にはCUDA Toolkit バージョン9.1以降が必要です。
    - PG-Stromが提供する半精度浮動小数点（`float2`）型は、内部的にCUDA Cのhalf_t型を使用しており、古いCUDA Toolkitではこれをビルドできないためです。
}
@en{
- **Server Hardware**
    - It requires generic x86_64 hardware that can run Linux operating system supported by CUDA Toolkit. We have no special requirement for CPU, storage and network devices.
    - [note002:HW Validation List](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List) may help you to choose the hardware.
    - SSD-to-GPU Direct SQL Execution needs SSD devices which support NVMe specification, and to be installed under the same PCIe Root Complex where GPU is located on.
- **GPU Device**
    - PG-Strom requires at least one GPU device on the system, which is supported by CUDA Toolkit, has computing capability 6.0 (Pascal generation) or later;
    - [note001:GPU Availability Matrix](https://github.com/heterodb/pg-strom/wiki/001:-GPU-Availability-Matrix) shows more detailed information. Check this list for the support status of SSD-to-GPU Direct SQL Execution.
- **Operating System**
    - PG-Strom requires Linux operating system for x86_64 architecture, and its distribution supported by CUDA Toolkit. Our recommendation is Red Hat Enterprise Linux or CentOS version 7.x series.    - SSD-to-GPU Direct SQL Execution needs Red Hat Enterprise Linux or CentOS version 7.3 or later.
- **PostgreSQL**
    - PG-Strom requires PostgreSQL version 9.6 or later. PostgreSQL v9.6 renew the custom-scan interface for CPU-parallel execution or `GROUP BY` planning, thus, it allows cooperation of custom-plans provides by extension modules.
- **CUDA Toolkit**
    - PG-Strom requires CUDA Toolkit version 9.1 or later.
    - PG-Strom provides half-precision floating point type (`float2`), and it internally use `half_t` type of CUDA C, so we cannot build it with older CUDA Toolkit.
}

@ja:## OSのインストール
@en:## OS Installation





@ja:## CUDA Toolkitのインストール
@en:## CUDA Toolkit Installation




@ja:## PostgreSQLのインストール
@en:## PostgreSQL Installation



@ja:## PG-Stromのインストール
@en:## PG-Strom Installation




@ja:## インストール後の設定
@en:## Post Installation Setup



