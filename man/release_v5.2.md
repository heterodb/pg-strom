@ja:#PG-Strom v5.2リリース
@en:#PG-Strom v5.2 Release

<div style="text-align: right;">PG-Strom Development Team (14-Jul-2024)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v5.2における主要な変更は点は以下の通りです。

- GpuJoin Pinned Inner Buffer
- GPU-Direct SQLの性能改善
- GPUバッファの64bit化
- 行単位CPU-Fallback
- SELECT DISTINCT句のサポート
- pg2arrow並列モードの性能改善
- 累積的なバグの修正
}

@en{
Major changes in PG-Strom v5.2 are as follows:

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
- NVIDIA GPU CC 6.0 以降 (Pascal以降必須; Turing以降を推奨)
}
@en{
- PostgreSQL v15.x, v16.x
- CUDA Toolkit 12.2 or later
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 6.0 or later (Pascal at least; Turing or newer is recommended)
}

##GpuJoin Pinned Inner Buffer

@ja{
PG-StromのGpuJoinは、Hash-Joinアルゴリズムを基にGPUで並列にJOIN処理を行えるように設計されています。Hash-Joinアルゴリズムの性質上、INNER側のテーブルはJOINの実行前に予めバッファ上に読み出されている必要がありますが、これまでは、GpuJoinはPostgreSQLの内部APIを介して下位の実行プランから一行ずつINNER側テーブルの内容を読み出していました。

この設計は、INNER側が巨大なテーブルを読み出すGpuScanである場合に無駄の多いものでした。

例えば、以下のようなクエリを想定してみる事にします。このクエリは<code>lineitem</code>(882GB)と、<code>orders</code>(205GB)の約1/4をJOINする処理が含まれています。 JOINを含むGpuPreAggの下位ノードにはGpuScanが存在しており、これが<code>orders</code>テーブルをINNERバッファへ読み出すのですが、これには約3.6億回のGpuScan呼び出しが必要となり、また30GBものINNERバッファをGPUに送出しなければいけませんでした。
}
@en{
PG-Strom's GpuJoin is designed to perform tables JOIN based on the Hash-Join algorithm using GPU in parallel. Due to the nature of the Hash-Join algorithm, the table on the INNER side must be read onto the buffer before the JOIN processing. In the older version, the contents of the INNER tables were read line by line using PostgreSQL's internal APIs.

This design was wasteful when the INNER side was a GpuScan that reads a huge table.

For example, consider the following query. This query includes a JOIN of <code>lineitem</code>(882GB) and about 1/4 of <code>orders</code>(205GB). There is a GpuScan in the lower node of GpuPreAgg that includes JOIN, which reads the <code>orders</code> table into the INNER buffer, but this requires about 360 million GpuScan calls and uses 30GB of INNER buffer which should be moved to the GPU.
}

```
=# explain select l_shipmode, o_shippriority, sum(l_extendedprice)
             from lineitem, orders
            where l_orderkey = o_orderkey
              and o_orderdate >= '1997-01-01'
            group by l_shipmode, o_shippriority;
                                                           QUERY PLAN
--------------------------------------------------------------------------------------------------------------------------------
 HashAggregate  (cost=38665701.61..38665701.69 rows=7 width=47)
   Group Key: lineitem.l_shipmode, orders.o_shippriority
   ->  Custom Scan (GpuPreAgg) on lineitem  (cost=38665701.48..38665701.55 rows=7 width=47)
         GPU Projection: pgstrom.psum((lineitem.l_extendedprice)::double precision), lineitem.l_shipmode, orders.o_shippriority
         GPU Join Quals [1]: (lineitem.l_orderkey = orders.o_orderkey) ... [nrows: 6000244000 -> 1454290000]
         GPU Outer Hash [1]: lineitem.l_orderkey
         GPU Inner Hash [1]: orders.o_orderkey
         GPU-Direct SQL: enabled (GPU-0)
         ->  Custom Scan (GpuScan) on orders  (cost=100.00..10580836.56 rows=363551222 width=12)
               GPU Projection: o_shippriority, o_orderkey
               GPU Pinned Buffer: enabled
               GPU Scan Quals: (o_orderdate >= '1997-01-01'::date) [rows: 1499973000 -> 363551200]
               GPU-Direct SQL: enabled (GPU-0)
(13 rows)
```

@ja{
v5.2では、GpuJoinのINNERバッファ初期セットアッププロセスはより無駄の少ないものに再設計されています。
}
@en{
In the v5.2, GpuJoin's INNER buffer initial setup process has been redesigned to be more effectively.
}

![GpuJoin pinned inner buffer](./img/release_5_2a.png)

@ja{
GpuScanはテーブルから読み出した行にWHERE句のチェックを行い、それをホスト側（PostgreSQLバックエンドプロセス）へ書き戻すという役割を担っています。GpuJoinはその結果を読み出してINNERバッファを構築し、これをGPU側へとコピーするわけですが、元々はGpuScanがINNER側テーブルを処理した時点で必要なデータは全てGPUに載っており、これをわざわざホスト側に書き戻し、再びGPU側へコピーする合理性はあまり大きくありません。

GpuJoin Pinned Inner Bufferは、GpuJoinの下位ノードがGpuScanである場合、その処理結果をホスト側に戻すのではなく、次のGpuJoinに備えてGpuScanの処理結果をGPUに留置しておくというものです。これにより、INNER側テーブルのサイズが非常に大きい場合にGpuJoinの初期セットアップ時間を大きく節約する事が可能となります。
}
@en{
GpuScan checks the WHERE clause on the rows read from the table and writes back to the host side (PostgreSQL backend process). GpuJoin reads the results to setup the INNER buffer, and copies this to the GPU side, but originally all the necessary data was on the GPU when GpuScan processed the INNER side table, and it is not very reasonable to get GpuScan's results back to the host side then copy to the GPU side again.

GpuJoin Pinned Inner Buffer allows GpuScan to keep the processing results, when child node of GpuJoin is GpuScan, to use a part of INNER buffer on the next GpuJoin, instead of returning the GpuScan's results to the host side once. Thie mechanism allows to save a lot of initial setup time of GpuJoin when size of the INNER tables are very large.
}

@ja{
一方で、ホスト側でINNERバッファのセットアップ作業を行わないという事は、GpuJoinのINNERバッファが物理的にCPUメモリ上には存在しないという事になり、CPU Fallback処理を必要とする場合には、SQL全体をエラーによってアボートする必要があります。

こういった副作用が考えられるため、GpuJoin Pinned Inner Buffer機能はデフォルトでは無効化されており、以下のようなコマンドを用いて、例えば『推定されるINNERバッファの大きさが100MBを越える場合にはGpuJoin Pinned Inner Bufferを使用する』という設定を明示的に行う必要があります。
}
@en{
On the other hand, setting up of the INNER buffer on GPU side means that the GpuJoin INNER buffer does not physically exist on the CPU memory, therefore, SQL must be aborted by error if CPU fallback processing is required.

Due to the potential side effects, the GpuJoin Pinned Inner Buffer function is disabled by default. You must explicitly enable the feature using command below; that means GpuJoin uses Pinned Inner Buffer if the estimated INNER buffer size exceeds 100MB.
}

```
=# set pg_strom.pinned_inner_buffer_threshold = '100MB';
SET
```

@ja:##行ごとのCPU-Fallback
@en:##Per-tuple CPU-Fallback

@ja{
GPUでSQLワークロードを処理する場合、原理的にGPUでの実行が不可能なパターンのデータが入力される場合があります。 例えば、長い可変長データがPostgreSQLブロックの大きさに収まらず、断片化して外部テーブルに保存される場合（TOAST機構とよばれます）には、外部テーブルのデータを持たないGPUでは処理を継続する事ができません。
}
@en{
When processing SQL workloads on GPUs, input data may have patterns that cannot be executed on GPUs in principle. For example, if long variable-length data does not fit into the PostgreSQL block size and is fragmented into an external table (this is called the TOAST mechanism), GPUs that do not have data in the external table will not continue processing.
}
@ja{
PG-StromにはCPU-Fallbackという仕組みが備わっており、こういったデータに対する処理をCPU側で行う仕組みが備わっています。 通常、GpuJoinやGpuPreAggなどの各種処理ロジックは、64MB分のデータ（チャンクと呼ぶ）をテーブルから読み出し、これに対してSQLワークロードを処理するためにGPUカーネルを起動します。
}
@en{
PG-Strom has a mechanism called CPU-Fallback, which allows processing of such data on the CPU side. Typically, processing logic such as GpuJoin and GpuPreAgg reads 64MB of data (called chunks) from a table and launches a GPU kernel to process the SQL workload.
}
@ja{
以前の実装では、SQLワークロードの処理中にCPU-Fallbackエラーが発生すると、そのチャンク全体のGPU処理をキャンセルしてCPU側に書き戻すという処理を行っていました。しかし、この戦略は２つの点において問題がありました。 一つは、通常、数十万行のデータを含むチャンクにたった1個の不良データが存在しただけでチャンク全体がGPUでの処理をキャンセルされてしまう事。もう一つは、GpuPreAggのように集計バッファを更新し続けるワークロードにおいては「どこまで集計表に反映されたか分からない」という状態が起こり得る事です。（したがって、v5.1以前ではGpuPreAggのCPU-Fallbackはエラー扱いとなっていました）
}
@en{
In the older version, if a CPU-Fallback error occurred while processing a SQL workload, the GPU kernel performing on the entire chunk was canceled and written back to the CPU side. However, this strategy was problematic in two points. First, if there is just one piece of bad data in a chunk containing hundreds of thousands of rows of data, the GPU processing of the entire chunk will be canceled. Another problem is that GpuPreAgg that keep updating the aggregation buffer, a situation may occur where it is difficult to know how much has been reflected in the aggregation table. (Therefore, before v5.1, GpuPreAgg's CPU-Fallback was treated as an error)
}

![Pe-tuple CPU Fallback](./img/release_5_2b.png)

@ja{
PG-Strom v5.2では、これらの問題を解決するためにCPU-Fallbackの実装が改良されています。

CPU-Fallbackエラーが発生した場合、従来のようにチャンク全体の処理をキャンセルするのではなく、通常の処理結果を書き出す「Destination Buffer」の他に「Fallback Buffer」を用意してCPU-Fallbackエラーを起こしたタプルだけを書き込むようにします。 「Fallback Buffer」の内容は後でまとめてCPU側に書き戻され、改めてCPUで評価が行われます。そのため、必要最小限のタプルだけをCPU-Fallback処理すれば良いだけでなく、GpuPreAggの集計バッファが重複して更新されてしまう心配もありません。
}
@en{
PG-Strom v5.2 improves the CPU-Fallback implementation to resolve these issues.

When a CPU-Fallback error occurs, instead of canceling the processing of the entire chunk as in the past, we prepare a "Fallback Buffer" in addition to the "Destination Buffer" that writes the normal processing result to handle the CPU-Fallback error. Only the generated tuples are written. The contents of "Fallback Buffer" are later written back to the CPU side and evaluated again by the CPU. Therefore, not only do you need to perform CPU-Fallback processing on only the minimum number of tuples required, but you also do not have to worry about the GpuPreAgg aggregation buffer being updated redundantly.
}

@ja:##GPUバッファの64bit化
@en:##64bit GPU Buffer representation

@ja{
現在でこそ、48GBや80GBといったメモリを搭載したGPUが販売されていますが、PG-Stromが内部的に使用するGPUバッファのデータ形式を設計したのはv2.0の開発中である2017年頃。つまり、ハイエンドGPUでも16GBや24GB、それ以外では8GB以下というのが一般的でした。 そういった前提では、物理的にあり得ない大容量のデータを過不足なく表現できるよりも、メモリ使用量を削る事のできるデータ形式が優先でした。
}
@en{
GPUs with memory such as 48GB or 80GB are now on sale, but the data format of the GPU buffer used internally by PG-Strom was designed around 2017 during the development of v2.0. In other words, even high-end GPUs were typically 16GB or 24GB, and others were generally 8GB or less. Under such a premise, a data format that can reduce memory usage was prioritized rather than being able to express physically impossible large amounts of data without too much or too little.
}
@ja{
PG-StromのGPUバッファにロードされたタプルは常に8バイト境界にアラインされているため、32bitのオフセット値を3bitだけシフトしてやれば、実質的に35bit (= 32GB)分のアドレス幅を表現する事が可能でした。2020年に40GBのメモリを搭載したNVIDIA A100が発表されましたが、32GBのバッファ長制限というのは実質的には無意味な制限事項ではありました。

なぜなら、PG-Stromは64MBのチャンク単位でストレージからデータを読み出すため、GPUバッファのサイズが32GBを超えるというケースはほとんどあり得ない状況であったわけです。
}
@en{
Tuples loaded into PG-Strom's GPU buffer are always guaranteed to be 8-byte aligned. Therefore, by shifting the 32-bit offset value by 3 bits, it was actually possible to express an address width of 35 bits (= 32 GB). In 2020, the NVIDIA A100 with 40GB of memory was announced, but the 32GB buffer length limit was essentially a meaningless restriction.

This is because PG-Strom reads data from storage in 64MB chunks, so it was almost impossible for the GPU buffer size to exceed 32GB.
}
@ja{
しかしながら、以下の状況において、非常に大きな結果バッファを想定する必要が出てきました。

- GPUキャッシュが非常に巨大なデータを保持している場合。
- GpuJoinのINNERバッファが巨大なサイズに膨れ上がったケース。
- GROUP BY または SELECT DISTINCT の結果として生じる行数が大幅に増加した場合。

PG-Strom v5.2においては、GPUバッファ上の全てのオフセット値は64bit表現に置き換えられました。結果として、32GBより大きなGPUバッファを扱うことができるようになり、これら前述のワークロードもGPUの物理RAMサイズの範囲まで扱えるようになりました。
}
@en{
However, in the following situations it becomes necessary to assume a very large result buffer.

- When the GPU cache holds a very large amount of data.
- A case where the GpuJoin INNER buffer swelled to a huge size.
- When the number of rows resulting from GROUP BY or SELECT DISTINCT increases significantly.

In PG-Strom v5.2, all offset values ​​on GPU buffers have been replaced with 64-bit representation. As a result, it is now possible to handle GPU buffers larger than 32GB, and even with workloads such as those mentioned above, it is now possible to handle up to the physical RAM size range.
}

@ja:##その他の新機能
@en:##Other new features


@ja:###GPU-Direct SQLの性能改善
@en:###Improved GPU-Direct SQL performance
@ja{
NVIDIAのcuFileライブラリは、内部的にDevive Primary Contextを仮定していました。そのため、独自に生成したCUDA Contextを使用していたPG-Strom GPU ServiceからのAPI呼出しに対して、CUDA Contextを切り替えるコストが発生していました。

PG-Strom v5.2では、GPU ServiceもDevice Primary Contextを使用するように設計変更を行い、cuFileライブラリ内のスイッチングコストとそれに付随するCPUのBusy Loopを無くすことで、およそ10%程度の性能改善が行われています。
}
@en{
NVIDIA's cuFile library assumed Devive Primary Context internally. As a result, there was a cost to switch the CUDA Context for API calls from the PG-Strom GPU Service, which was using a uniquely generated CUDA Context.

In PG-Strom v5.2, GPU Service has also been redesigned to use Device Primary Context, which has improved performance by approximately 10% by eliminating switching costs in the cuFile library and the associated CPU Busy Loop. is being carried out.
}

@ja:###SELECT DISTINCT句のサポート
@en:###SELECT DISTINCT support
@ja{
PG-Strom v5.2では<code>SELECT DISTINCT ...</code>句がサポートされました。 以前のバージョンでは、これを<code>GROUP BY</code>句に書き直す必要がありました。
}
@en{
PG-Strom v5.2 supports the <code>SELECT DISTINCT...</code> clause. In previous versions, this had to be rewritten as a <code>GROUP BY</code> clause.
}

@ja:###pg2arrow並列モードの性能改善
@en:###Improced parallel pg2arrow
@ja{
<code>pg2arrow</code>コマンドで<code>-t</code>オプションと<code>-n</code>オプションを併用した場合、読出し対象のテーブルサイズを調べ、各ワーカースレッドが重複してテーブルを読み出さないようにスキャン範囲を調整してクエリを発行します。
}
@en{
If you use the <code>-t</code> option and <code>-n</code> option together with the <code>pg2arrow</code> command, check the table size to be read, adjust the scan range so that each worker thread does not read the table redundantly, and execute the query. Issue.
}

@ja:###IMPORT FOREIGN SCHEMAで重複列に別名
@en:###Aliases for duplicated names on IMPORT FOREIGN SCHEMA
@ja{
<code>IMPORT FOREIGN SCHEMA</code>や<code>pgstrom.arrow_fdw_import_file()</code>関数を用いてArrowファイルをインポートする際、Fieldが重複した名前を持っていた場合、重複した名前を持つ2番目以降の列には別名を付けるようになりました。
}
@en{
When importing an Arrow file using the <code>IMPORT FOREIGN SCHEMA</code> or <code>pgstrom.arrow_fdw_import_file()</code> function, if a Field has a duplicate name, give an alias to the second and subsequent columns with duplicate names. Now it looks like this.
}

@ja:###部分的な処理結果の返却
@en:###Partial concurrent results responding
@ja{
GpuJoinやGpuPreAggなど各種の処理でDestination Bufferを使い尽くした場合、GPUカーネルの実行を一時停止し、部分的な処理結果をバックエンドプロセスに返却してから、再度GPUカーネルの実行を再開するようになりました。 これまではDestination Bufferを拡大してチャンクの処理結果を全て保持するような構造になっていたため、入力に対して結果セットが巨大である場合には、CPUやGPUのメモリを過剰に消費してシステムが不安定になるという問題がありました。
}
@en{
When the Destination Buffer is used up in various processes such as GpuJoin or GpuPreAgg, execution of the GPU kernel will be paused, partial processing results will be returned to the backend process, and then execution of the GPU kernel will be resumed again. became. Until now, the structure was such that the Destination Buffer was expanded to hold all the processing results of the chunk, so if the result set was huge for the input, it would consume excessive CPU and GPU memory. There was a problem with the system becoming unstable.
}

@ja:##累積的なバグの修正
@en:##Cumulative bug fixes

- [#664] Too much CPU consumption ratio with cuFile/GDS on many threads
- [#757] wrong varnullingrels at setrefs.c
- [#762] arrow_fdw: scan-hint needs to resolve expressions using INDEV_VAR
- [#763] Var-nodes on kvdef->expr was not compared correctly
- [#764][#757] Var::varnullingrels come from the prior level join are not consistent
- [#673] lookup_input_varnode_defitem() should not use equal() to compare Var-nodes
- [#729] update RPM build chain
- [#765] Add regression test for PostgreSQL 16
- [#771] Update regression test for PostgreSQL 15
- [#768] fix dead loop in `gpuCacheAutoPreloadConnectDatabase
- [#xxx] Wrong GiST-Index based JOIN results
- [#774] add support of SELECT DISTINCT
- [#778] Disable overuse of kds_dst buffer in projection/gpupreagg
- [#752] add KDS_FORMAT_HASH support in execGpuJoinProjection()
- [#784] CPU-Fallback JOIN didn't handle LEFT/FULL OUTER case if tuple has no matched inner row
- [#777] Fix the bug of dynamically allocating fallback buffer size and nrooms
- [#776] Fix the out of range bug in pgfn_interval_um()
- [#706] gpucache: drop active cache at DROP DATABASE from shmem / gpumem
- [#791] gpujoin: wrong logic to detect unmatched inner tuple
- [#794] assertion failure at cost_memoize_rescan()
- [#xxx] pg2arrow: outer/inner-join subcommand initialization
- [#xxx] IMPORT FOREIGN SCHEMA renames duplicated field name
- [#xxx] arrow_fdw: correct pg_type hint parsing
- [#748] Add support CPU-fallback on GpuPreAgg, and revise fallback implementation
- [#778] Add XpuCommandTag__SuccessHalfWay response tag
