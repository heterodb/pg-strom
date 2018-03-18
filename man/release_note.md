@ja:# PG-Strom v2.0リリース
@en:# PG-Strom v2.0 Release

<div style="text-align: right;">PG-Strom Development Team (xx-Apr-2018)</div>

## 概要
## Overview

@ja{
PG-Strom v2.0における主要な機能強化は以下の通りです。

- GPUを管理する内部インフラストラクチャの全体的な再設計と安定化
- CPU+GPUハイブリッド並列実行
- SSD-to-GPUダイレクトSQL実行
- インメモリ列指向キャッシュ
- GPUメモリストア(store_fdw)
- GpuJoinとGpuPreAggの再設計に伴う高速化
- GpuPreAgg+GpuJoin+GpuScan 密結合GPUカーネル
}

@en{
Major enhancement in PG-Strom v2.0 includes:

- Overall redesign of the internal infrastructure to manage GPU and stabilization
- CPU+GPU hybrid parallel execution
- SSD-to-GPU Direct SQL Execution
- In-memory columnar cache
- GPU memory store (gstore_fdw)
- Redesign of GpuJoin and GpuPreAgg and speed-up
- GpuPreAgg + GpuJoin + GpuScan combined GPU kernel
}

## 動作環境
## Prerequisites

@ja{
- PostgreSQL v9.6, v10
- CUDA Toolkit 9.1
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 6.0 以降 (Pascal以降)
}
@en{
- PostgreSQL v9.6, v10
- CUDA Toolkit 9.1
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 6.0 or later (Pascal or Volta)
}

## 新機能
## New Features

@ja{
- SSD-to-GPUダイレクトSQL実行
- インメモリ列指向キャッシュ

}
@en{

}


## 廃止された機能
## Dropped features

@ja{
- PostgreSQL v9.5サポート
    - PostgreSQL v9.6ではCPU並列クエリの提供に伴い、オプティマイザ/エグゼキュータ共に大きな修正が加えられました。これらと密接に連携する拡張モジュールにとって最もインパクトの大きな変更は『upper planner path-ification』と呼ばれるインターフェースの強化で、集約演算やソートなどの実行計画もコストベースで複数の異なる方法を比較して最適なものを選択できるようになりました。
    - これはGpuPreAggを実装するためにフックを利用して実行計画を書き換えていた従来の方法とは根本的に異なり、より合理的かつ信頼できる方法でGPUを用いた集約演算を挿入する事が可能となり、バグの温床であった実行計画の書き換えロジックを捨てる事が可能になりました。
    - 同時に、CustomScanインターフェースにもCPU並列に対応するためのAPIが拡張され、これらに対応するためにPostgreSQL v9.5サポートは廃止されました。
- GpuSort機能
    - GpuSort機能は性能上のメリットが得られないため廃止されました。
    - ソートはGPUの得意とするワークロードの一つです。しかし、GPUデバイスメモリの大きさを越えるサイズのデータをソートする場合、複数のチャンクに分割して部分ソートを行い、後でCPU側でこれを結合して最終結果を出力する必要があります。
    - 結合フェーズの処理を軽くするには、GPUでソートすべきチャンクのサイズを大きく必要がありますが、一方でチャンクサイズが大きくなるとソート処理を開始するためのリードタイムが長くなり、PG-Stromの特長の一つである非同期処理によるデータ転送レイテンシの隠ぺいが効かなくなるというトレードオフがあります。
    - これらの問題に対処するのは困難、少なくとも時期尚早であると判断し、GpuSort機能は廃止されました。
}
@en{
- PostgreSQL v9.5 Support
    - PostgreSQL v9.6 had big changes in both of the optimizer and executor to support CPU parallel query execution. The biggest change for extension modules that interact them is an enhancement of the interface called "upper planner path-ification". It allows to choose an optimal execution-plan from the multiple candidates based on the estimated cost, even if it is aggregation or sorting.
    - It is fundamentally different from the older way where we rewrote query execution plan to inject GpuPreAgg using the hooks. It allows to inject GpuPreAgg node in more reasonable and reliable way, and we could drop complicated (and buggy) logic to rewrite query execution plan once constructed.
    - CustomScan interface is also enhanced to support CPU parallel execution. Due to the reason, we dropped PostgreSQL v9.5 support to follow these new enhancement.
- GpuSort feature
    - We dropped GpuSort because we have little advantages in the performance.
    - Sorting is one of the GPU suitable workloads. However, in case when we try to sort data blocks larger than GPU device memory, we have to split the data blocks into multiple chunks, then partially sort them and merge them by CPU to generate final results.
    - Larger chunk size is better to reduce the load to merge multiple chunks by CPU, on the other hands, larger chunk size takes larger lead time to launch GPU kernel to sort. It means here is a trade-off; which disallows asynchronous processing by PG-Strom to make data transfer latency invisible.
    - It is hard to solve the problem, or too early to solve the problem, we dropped GpuSort feature once.
}

