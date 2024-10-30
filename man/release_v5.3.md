@ja:#PG-Strom v5.3リリース
@en:#PG-Strom v5.3 Release

<div style="text-align: right;">PG-Strom Development Team (xx-xxx-2024)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v5.3における主要な変更は点は以下の通りです。

- GpuJoin Partitioned Pinned Inner Buffer
- Arrow_Fdw Virtual Columns



- GpuJoin Pinned Inner Buffer
- GPU-Direct SQLの性能改善
- GPUバッファの64bit化
- 行単位CPU-Fallback
- SELECT DISTINCT句のサポート
- pg2arrow並列モードの性能改善
- 累積的なバグの修正
}

@en{
Major changes in PG-Strom v5.3 are as follows:

- GpuJoin Pinned Inner Buffer
- Improved GPU-Direct SQL performance
- 64bit GPU Buffer representation
- Per-tuple CPU-Fallback
- SELECT DISTINCT support
- Improced parallel pg2arrow
- Cumulative bug fixes
}

@ja:##動作環境
@en:##Prerequisites

@ja{
- PostgreSQL v15.x, v16.x
- CUDA Toolkit 12.2 以降
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 6.0 以降 (Pascal以降; Turing以降を推奨)
}
@en{
- PostgreSQL v15.x, v16.x
- CUDA Toolkit 12.2 or later
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 6.0 or later (Pascal at least; Turing or newer is recommended)
}




@ja:##Arrow_Fdw仮想列
@en:##Arrow_Fdw Virtual Columns

注意、外部テーブル定義と1:1マップだったけど、フィールド名と対応付けになっている。





@ja:##累積的なバグの修正
@en:##Cumulative bug fixes

- [#812] CPU-fallback at depth>0 didn't set ecxt_scanslot correctly.
- [#817] GPU-Service didn't detach worker thread's exit status.

