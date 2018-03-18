@ja:# PG-Strom v2.0リリース
@en:# PG-Strom v2.0 Release

<div style="text-align: right;">PG-Strom Development Team (xx-Apr-2018)</div>

@ja:## 概要
@en:## Overview

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

@ja:## 動作環境
@en:## Prerequisites

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

@ja:## 新機能
@en:## New Features

@ja{
- GPUを管理する内部インフラストラクチャの全体的な再設計と安定化
    - PostgreSQLバックエンドプロセスは同時に１個のGPUだけを利用するようになりました。マルチGPUを利用する場合はPostgreSQLのCPU並列との併用が前提になりますが、CPUスレッドがGPUへデータを供給するスループットはGPUの処理能力よりもずっと低いため、通常、これは問題とはなりません。設計のシンプル化を優先しました。
    - Pascal世代以降のGPUで採用されたGPUデバイスメモリのデマンドページングをほぼ全面的に採用するようになりました。SQLワークロードの多くは実際に実行してみるまで必要な結果バッファの大きさが分からないため、これまでは必要以上にバッファを獲得し、またメモリ不足時には再実行を行っていましたが、これらは同時実行プロセスの利用可能なリソースを制限し、また複雑な例外ロジックはバグの温床でした。GPUデバイスメモリのデマンドページングを利用する事で、設計のシンプル化を行いました。
    - CUDAの非同期インターフェースの利用を止めました。GPUデバイスメモリのデマンドページングを利用すると、DMA転送のための非同期API（`cuMemCpyHtoD`など）は同期的に振舞うようになるため、GPUカーネルの多重度が低下してしまいます。代わりにPG-Strom自身がワーカースレッドを管理し、これらのワーカースレッドがそれぞれ同期APIを呼び出すよう設計変更を行いました。副産物として、非同期コールバック（`cuStreamAddCallback`）を利用する必要がなくなったので、MPSを利用する事が可能となりました。
}
@ja{
- CPU+GPUハイブリッド並列実行
    - PostgreSQL v9.6で新たにサポートされたCPU並列実行に対応しました。
    - PG-Stromの提供するGpuScan、GpuJoinおよびGpuPreAggの各ロジックは複数のPostgreSQLバックグラウンドワーカープロセスにより並列に実行する事が可能です。
    - PostgreSQL v9.6ではCPU並列実行の際に`EXPLAIN ANALYZE`で取得するPG-Strom独自の統計情報が正しくありません。これは、CustomScanインターフェースAPIで`ShutdownCustomScan`が提供されていなかったため、DSM（動的共有メモリ）の解放前にコーディネータプロセスがワーカープロセスの情報を回収する手段が無かったためです。
}
@ja{
- SSD-to-GPUダイレクトSQL実行
    - Linuxカーネルモジュール `nvme_strom` を用いる事で、NVMe規格に対応したSSD上のPostgreSQLデータブロックを、CPUを介さずダイレクトにGPUデバイスメモリへ転送する事が可能となりました。
    - 本機能を用いる事で、システムRAMに載り切らない大きさのデータを処理する場合であっても、GPUへ安定的に高スループットでデータを供給する事が可能になりました。
    - ブロックデバイス層やファイルシステムを経由しないためハードウェア限界に近い高スループットを引き出す事が可能で、かつ、GPUでSQLワークロードを処理するためCPUの処理すべきデータ量を減らす事ができます。このような特性の組み合わせにより、一般的には計算ワークロードのアクセラレータとして認識されているGPUを、I/Oワークロードの高速化に適用する事に成功しました。
}
@ja{
- インメモリ列指向キャッシュ
    - RAMサイズに載る程度の大きさのデータに対しては、よりGPUでの処理に適した列データ形式に変形してキャッシュする事が可能になりました。テーブルのスキャンに際して、列指向キャッシュが存在する場合にはPostgreSQLの共有バッファよりもこちらを優先して参照します。
    - インメモリ列指向キャッシュは同期的、または非同期的にバックグラウンドで構築する事が可能です。
    - 初期のPG-Stromで似たような機能が存在していた事を覚えておられるかもしれません。v2.0で新たに実装された列指向キャッシュは、キャッシュされた行が更新されると、当該行を含むキャッシュブロックを消去（invalidation）します。行ストアの更新に合わせて列キャッシュ側の更新を行うという事は行わないため、更新ワークロードに対するパフォーマンスの低下は限定的です。
}
@ja{
- GPUメモリストア(gstore_fdw)
    - GPU上に確保したデバイスメモリ領域に対して、外部テーブル（Foreign Table）のインターフェースを利用してSQLのSELECT/INSERT/UPDATE/DELETEにより読み書きを行う機能です。
    - 内部データ形式は `pgstrom` 型のみがサポートされています。これは、PG-Stromのバッファ形式`KDS_FORMAT_COLUMN`タイプと同一の形式でデータを保持するものです。可変長データを保存する場合、LZ方式によるデータ圧縮を行う事も可能です。
    - v2.0の時点では、GPUメモリストアはPL/CUDA関数のデータソースとしてだけ利用する事が可能です。
    - `lo_import_gpu`および`lo_export_gpu`関数により、外部アプリケーションの確保したGPUメモリの内容を直接PostgreSQLのラージオブジェクトに記録したり、逆にラージオブジェクトの内容をGPUメモリに書き出す事が可能です。
}
@ja{
- GpuJoinとGpuPreAggの再設計に伴う高速化
    - 従来、GpuJoinとGpuPreAggで内部的に使用していたDynamic Parallelismの利用をやめ、処理ロジック全体の見直しを行いました。これは、GPUカーネルがサブカーネルを起動後、その完了を待っている間もGPUカーネルの実行スロットを占有してしまっているため、GPUの使用率が上がらないという問題があったためです。
    - この再設計に伴う副産物として、GpuJoinのサスペンド/レジューム機能が実装されました。原理上、SQLのJOIN処理は入力した行数よりも出力する行数の方が増えてしまう事がありますが、処理結果を書き込むバッファの残りサイズが不足した時点でGpuJoinをサスペンドし、新しい結果バッファを割り当ててレジュームするように修正されました。これにより、結果バッファのサイズ推定が簡略化されたほか、実行時のバッファ不足による再実行の必要がなくなりました。
}
@ja{
- GpuPreAgg+GpuJoin+GpuScan 密結合GPUカーネル
    - GPUで実行可能なSCAN、JOIN、GROUP BYが連続しているとき、対応するGpuScan、GpuJoin、GpuPreAggを一回のGPUカーネル呼び出しで実行する事が可能になりました。これは、GpuJoinの結果バッファをそのままGpuPreAggの入力バッファとして扱うなど、CPUとGPUの間のデータ交換を最小限に抑えるためのアプローチです。
    - この機能は特に、SSD-to-GPUダイレクトSQL実行と組み合わせて使用すると効果的です。
}
@ja{
- 新しいデータ型の対応
    - `uuid`型に対応しました。
    - ネットワークアドレス型（`inet`、`cidr`、および`macaddr`）に対応しました。
    - 範囲型（`int4range`、`int8range`、`tsrange`、`tstzrange`、`daterange`）に対応しました。
    - 半精度浮動小数点型（`float2`）に対応しました。半精度浮動小数点型に関連するCPU側の実装はPG-Stromの独自実装によるものです。
- 新しい演算子/関数の対応
    - 日付時刻型に対する`EXTRACT(field FROM timestamp)`演算子に対応しました。
}
@ja{
- PL/CUDA関連の強化
    - `#plcuda_include`の拡張により、`text`型を返すSQL関数を指定できるようになりました。引数の値によって挿入するコードを変える事ができるため、単に外部定義関数を読み込むだけでなく、動的にいくつものGPUカーネルのバリエーションを作り出すことも可能です。
    - PL/CUDA関数の引数に`reggstore`型を指定した場合、GPUカーネル関数へは対応するGPUメモリストアのポインタが渡されます。OID値が渡されるわけではない事に留意してください。
}
@ja{
- パッケージング
    - PostgreSQL Global Development Groupの配布するPostgreSQLパッケージに適合するよう、RPMパッケージ化を行いました。
    - 全てのソフトウェア物件はHeteroDB SWDC(Software Distribution Center)よりダウンロードが可能です。
}
@ja{
- ドキュメント
    - PG-Stromドキュメントをmarkdownとmkdocsを用いて全面的に書き直しました。従来のHTMLを用いたアプローチに比べ、よりメンテナンスが容易で新機能の開発に合わせたドキュメントの拡充が可能となります。
}
@ja{
- テスト
    - PostgreSQLのリグレッションテストフレームワークを使用して、PG-Stromのリグレッションテストを作成しました。
}


@ja:## 廃止された機能
@en:## Dropped features

@ja{
- PostgreSQL v9.5サポート
    - PostgreSQL v9.6ではCPU並列クエリの提供に伴い、オプティマイザ/エグゼキュータ共に大きな修正が加えられました。これらと密接に連携する拡張モジュールにとって最もインパクトの大きな変更は『upper planner path-ification』と呼ばれるインターフェースの強化で、集約演算やソートなどの実行計画もコストベースで複数の異なる方法を比較して最適なものを選択できるようになりました。
    - これはGpuPreAggを実装するためにフックを利用して実行計画を書き換えていた従来の方法とは根本的に異なり、より合理的かつ信頼できる方法でGPUを用いた集約演算を挿入する事が可能となり、バグの温床であった実行計画の書き換えロジックを捨てる事が可能になりました。
    - 同時に、CustomScanインターフェースにもCPU並列に対応するためのAPIが拡張され、これらに対応するためにPostgreSQL v9.5サポートは廃止されました。
}
@en{
- PostgreSQL v9.5 Support
    - PostgreSQL v9.6 had big changes in both of the optimizer and executor to support CPU parallel query execution. The biggest change for extension modules that interact them is an enhancement of the interface called "upper planner path-ification". It allows to choose an optimal execution-plan from the multiple candidates based on the estimated cost, even if it is aggregation or sorting.
    - It is fundamentally different from the older way where we rewrote query execution plan to inject GpuPreAgg using the hooks. It allows to inject GpuPreAgg node in more reasonable and reliable way, and we could drop complicated (and buggy) logic to rewrite query execution plan once constructed.
    - CustomScan interface is also enhanced to support CPU parallel execution. Due to the reason, we dropped PostgreSQL v9.5 support to follow these new enhancement.
}
@ja{
- GpuSort機能
    - GpuSort機能は性能上のメリットが得られないため廃止されました。
    - ソートはGPUの得意とするワークロードの一つです。しかし、GPUデバイスメモリの大きさを越えるサイズのデータをソートする場合、複数のチャンクに分割して部分ソートを行い、後でCPU側でこれを結合して最終結果を出力する必要があります。
    - 結合フェーズの処理を軽くするには、GPUでソートすべきチャンクのサイズを大きく必要がありますが、一方でチャンクサイズが大きくなるとソート処理を開始するためのリードタイムが長くなり、PG-Stromの特長の一つである非同期処理によるデータ転送レイテンシの隠ぺいが効かなくなるというトレードオフがあります。
    - これらの問題に対処するのは困難、少なくとも時期尚早であると判断し、GpuSort機能は廃止されました。
}
@en{
- GpuSort feature
    - We dropped GpuSort because we have little advantages in the performance.
    - Sorting is one of the GPU suitable workloads. However, in case when we try to sort data blocks larger than GPU device memory, we have to split the data blocks into multiple chunks, then partially sort them and merge them by CPU to generate final results.
    - Larger chunk size is better to reduce the load to merge multiple chunks by CPU, on the other hands, larger chunk size takes larger lead time to launch GPU kernel to sort. It means here is a trade-off; which disallows asynchronous processing by PG-Strom to make data transfer latency invisible.
    - It is hard to solve the problem, or too early to solve the problem, we dropped GpuSort feature once.
}

