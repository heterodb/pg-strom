#Pinned Inner Buffer

@ja{
本章では、GPU-Joinでサイズの大きなテーブル同士を結合する際の処理効率を向上させるための技術であるPinned Inner Buffer機能について説明します。
}
@en{
This chapter introduces the Pinned Inner Buffer feature, a technology that improves efficiency of large tables join using GPU-Join.
}

@ja:##概要
@en:##Overview

@ja{
以下の実行計画を見てください。
PG-Stromがテーブルを結合する際、通常は最もサイズの大きなテーブル（この場合は`lineorder`で、OUTER表と呼びます）を非同期的に読み込みながら、他のテーブルとの結合処理および集計処理を進めます。
JOINアルゴリズムの制約上、予めそれ以外のテーブル（この場合は`date1`、`part`、`supplier`で、INNER表と呼びます）をメモリ上に読み出し、またJOINキーのハッシュ値を計算する必要があります。これらのテーブルはOUTER表ほど大きなサイズではないものの、数GBを越えるようなINNERバッファの準備は相応に重い処理となります。
}
@en{
Look at the EXPLAIN output below.
When PG-Strom joins tables, it usually reads the largest table (`lineorder` in this case; called the OUTER table) asynchronously, while performing join processing and aggregation processing with other tables. Let's proceed.
Due to the constraints of the JOIN algorithm, it is necessary to read other tables (`date1`, `part`, `supplier` in this case; called the INNER tables) into memory in advance, and also calculate the hash value of the JOIN key. Although these tables are not as large as the OUTER table, preparing an INNER buffer that exceeds several GB is a heavy process.
}
```
=# explain
   select sum(lo_revenue), d_year, p_brand1
     from lineorder, date1, part, supplier
    where lo_orderdate = d_datekey
      and lo_partkey = p_partkey
      and lo_suppkey = s_suppkey
      and p_brand1 between 'MFGR#2221' and 'MFGR#2228'
      and s_region = 'ASIA'
    group by d_year, p_brand1;
                                                  QUERY PLAN
---------------------------------------------------------------------------------------------------------------
 GroupAggregate  (cost=31007186.70..31023043.21 rows=6482 width=46)
   Group Key: date1.d_year, part.p_brand1
   ->  Sort  (cost=31007186.70..31011130.57 rows=1577548 width=20)
         Sort Key: date1.d_year, part.p_brand1
         ->  Custom Scan (GpuJoin) on lineorder  (cost=275086.19..30844784.03 rows=1577548 width=20)
               GPU Projection: date1.d_year, part.p_brand1, lineorder.lo_revenue
               GPU Join Quals [1]: (part.p_partkey = lineorder.lo_partkey) ... [nrows: 5994236000 -> 7804495]
               GPU Outer Hash [1]: lineorder.lo_partkey
               GPU Inner Hash [1]: part.p_partkey
               GPU Join Quals [2]: (supplier.s_suppkey = lineorder.lo_suppkey) ... [nrows: 7804495 -> 1577548]
               GPU Outer Hash [2]: lineorder.lo_suppkey
               GPU Inner Hash [2]: supplier.s_suppkey
               GPU Join Quals [3]: (date1.d_datekey = lineorder.lo_orderdate) ... [nrows: 1577548 -> 1577548]
               GPU Outer Hash [3]: lineorder.lo_orderdate
               GPU Inner Hash [3]: date1.d_datekey
               GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on part  (cost=0.00..59258.00 rows=2604 width=14)
                     Filter: ((p_brand1 >= 'MFGR#2221'::bpchar) AND (p_brand1 <= 'MFGR#2228'::bpchar))
               ->  Custom Scan (GpuScan) on supplier  (cost=100.00..190348.83 rows=2019384 width=6)
                     GPU Projection: s_suppkey
                     GPU Pinned Buffer: enabled
                     GPU Scan Quals: (s_region = 'ASIA'::bpchar) [rows: 9990357 -> 2019384]
                     GPU-Direct SQL: enabled (GPU-0)
               ->  Seq Scan on date1  (cost=0.00..72.56 rows=2556 width=8)
(24 rows)
```
@ja{
GpuJoinは通常、PostgreSQLのAPIを通してINNER表を一行ごとに読み出し、そのハッシュ値を計算するとともに共有メモリ上のINNERバッファに書き込みます。GPU-Serviceプロセスは、このINNERバッファをGPUメモリに転送し、そこではじめてOUTER表を読み出してJOIN処理を開始する事ができるようになります。
INNER表が相応に大きくGPUで実行可能な検索条件を含む場合、以下の実行計画のように、GpuJoinの配下にGpuScanが存在するケースがあり得ます。この場合、INNER表はいったんGpuScanによってGPUで処理された後、その実行結果をCPU側に戻し、さらにINNERバッファに書き込まれた後でもう一度GPUへロードされます。ずいぶんと無駄なデータの流れが存在するように見えます。
}
@en{
GpuJoin usually reads the INNER table through the PostgreSQL API row-by-row, calculates its hash value, and writes them to the INNER buffer on the host shared memory. The GPU-Service process transfers this INNER buffer onto the GPU device memory, then we can start reading the OUTER table and processing the JOIN with inner tables.
If the INNER table is relatively large and contains search conditions that are executable on the GPU, GpuScan may exists under GpuJoin, as in the EXPLAIN output below. In this case, the INNER table is once processed on the GPU by GpuScan, the execution results are returned to the CPU, and then written to the INNER buffer before it is loaded onto the GPU again. It looks like there is quite a bit of wasted data flow.
}
![GPU-Join Pinned-Inner-Buffer](./img/pinned_inner_buffer_00.png)

@ja{
このように、INNER表の読出しやINNERバッファの構築の際にCPUとGPUの間でデータのピンポンが発生する場合、***Pinned Inner Buffer***を使用することで、GpuJoinの実行開始リードタイムの短縮や、メモリ使用量を削減する事ができます。
上の実行計画では、`supplier`表の読出しがGpuScanにより行われる事になっており、統計情報によれば約200万行が読み出されると推定されています。その一方で、`GPU Pinned Buffer: enabled`の出力に注目してください。これは、INNER表の推定サイズが`pg_strom.pinned_inner_buffer_threshold`の設定値を越える場合、GpuScanの処理結果をそのままGPUメモリに残しておき、それを次のGpuJoinでINNERバッファの一部として利用するという機能です（必要であればハッシュ値の計算もGPUで行います）。
そのため、`supplier`表の内容はGPU-Direct SQLによってストレージからGPUへと読み出された後、CPU側に戻されたり、再度GPUへロードされたりすることなく、次のGpuJoinで利用される事になります。
}
@en{
In this way, if data ping-pong occurs between the CPU and GPU when reading the INNER table or building the INNER buffer, you can configure GPUJoin to use ***Pinned Inner Buffer***. It is possible to shorten the execution start lead time and reduce memory usage.
In the above EXPLAIN output, reading of the `supplier` table will be performed by GpuScan, and according to the statistical information, it is estimated that about 2 million rows will be read from the table. Meanwhile, notice the output of `GPU Pinned Buffer: enabled`. This is a function that if the estimated size of the INNER table exceeds the configuration value of `pg_strom.pinned_inner_buffer_threshold`, the processing result of GpuScan is retained in the GPU memory and used as part of the INNER buffer at the next GpuJoin. (If necessary, hash value calculation is also performed on the GPU).
Therefore, after the contents of the `supplier` table are read from storage to the GPU using GPU-Direct SQL, they can be used in the next GPUJoin without being returned to the CPU or loaded to the GPU again. 
}
@ja{
一方で注意すべき点もあります。
Pinned Inner Bufferを使用するには、CPU-Fallbackを無効化する必要があります。

CPU-Fallbackとは、GPUでは処理できなかったデータをCPUに書き戻して再実行するための機能で、例えばTOAST化された可変長データを参照する条件式は原理上GPUで実行できないため、CPUに書き戻して再実行するために用いている機能です。しかしGpuScanを実行中にCPU-Fallbackが発生すると、GPUメモリ上の結果バッファ（これはGpuJoinのINNERバッファとして使用される）が完全な結果セットである事を保証できません。
また、Pinned Inner Bufferを使用するGpuJoinの実行にCPU-Fallbackが発生した場合、そもそもCPUはJOINに必要なINNERバッファを持っていないためにフォールバック処理を実行する事ができません。

そのため、Pinned Inner Bufferを使用するには`SET pg_strom.cpu_fallback = off`を指定してCPU-Fallbackを無効化する必要があります。
これは[GPU-Sort](gpusort.md)でもCPU-Fallback処理の無効化を要求している理由と同じです。
}
@en{
However, there are some points to be aware of.

To use Pinned Inner Buffer, CPU-Fallback must be disabled.

CPU-Fallback is a function that writes back to the CPU data that could not be processed by the GPU and re-executes it. For example, a conditional expression that references TOASTed variable-length data cannot be executed by the GPU in principle, so this function is used to write it back to the CPU and re-execute it. However, if a CPU-Fallback occurs while executing GpuScan, it cannot be guaranteed that the result buffer in the GPU memory (which is used as the INNER buffer for GpuJoin) is a complete result set.

In addition, if a CPU-Fallback occurs when executing GpuJoin that uses Pinned Inner Buffer, the CPU cannot execute the fallback process because it does not have the INNER buffer required for JOIN in the first place.

Therefore, to use Pinned Inner Buffer, it is necessary to disable CPU-Fallback by specifying `SET pg_strom.cpu_fallback = off`.
This is the same reason why [GPU-Sort](gpusort.md) also requires disabling CPU-Fallback processing.
}

@ja:##マルチGPUの場合
@en:##in case multi-GPUs

@ja{
多くのシステムではサーバ本体のRAMと比較してGPU搭載RAMの容量は限定的で、ハッシュ表のサイズにも制約があります。
複数のGPUにハッシュ表を分割配置する事でこの制限を緩和する事ができますが、あるGPU上でJOINの実行中に別のGPU上に配置されているINNER行を参照してしまうと、GPUメモリのスラッシングと呼ばれる現象が発生し強烈な速度低下を招いてしまうため、GPU-Joinの実行中にはメモリアクセスの局所性を確保できる仕組みが必要です。
}
@en{
In many systems, the capacity of GPU RAM is limited compared to the host system RAM, and there are also constraints on the size of the hash table.
This limitation can be alleviated by splitting the hash table across multiple GPUs, but if an INNER row located on one GPU is referenced while a JOIN is being executed on another GPU, a phenomenon known as GPU memory thrashing occurs, resulting in a significant slowdown in speed. Therefore, a mechanism is needed to ensure locality of memory access while GPU-Join is being executed.
}

@ja{
マルチGPUシステムにおいて、Pinned Inner Bufferは次のように動作します。

GPU-Joinに先立ってINNER側テーブルのスキャン処理を複数のGPUで実行し、その処理結果をGPUメモリ上に留置してハッシュ表を構築した場合、それぞれのGPUにどのような行が載っているかは完全にランダムです。
次ステップのHash-Join処理でOUTER側から読み出した行が、最初にGPU1上のINNER行と結合し、次にGPU2上のINNER行と、最後にGPU0上のINNER行と結合するといった形になってしまうと、極端なスラッシングが発生し強烈な性能低下を引き起こします。

そのため、マルチGPUでのPinned-Inner-Buffer利用時には再構築（reconstruction）処理を挟み、ハッシュ表を適切なGPU上に再配置します。

例えば3台のGPUを搭載しているシステムで、ほぼハッシュ表の大きさが3台のGPU搭載RAMに収まる場合、INNER側テーブルのGPU-Scan終了後、次のGPU-Joinで利用する結合キーのハッシュ値を計算し、それを3で割った剰余が0の場合はGPU0に、1の場合はGPU1に、2の場合はGPU2にという再配置を行います。

この処理を挟む事で、GPU-JoinをGPU0上で実行した場合にハッシュ表にはハッシュ値を3で割った剰余が0であるINNER行しか存在せず、同様にGPU1にはハッシュ値を3で割った剰余が1であるINNER行しか存在しないという状態を作ることができます。
}
@en{
In a multi-GPU system, the Pinned Inner Buffer works as follows:

If the INNER table scan process is executed on multiple GPUs prior to GPU-Join and the results of that process are stored in GPU memory to build a hash table, it is completely random which rows are on each GPU.

If the rows read from the OUTER side in the next step, the Hash-Join process, are first joined with the INNER row on GPU1, then with the INNER row on GPU2, and finally with the INNER row on GPU0, extreme thrashing will occur, causing a severe drop in performance.

For this reason, when using the Pinned-Inner-Buffer in a multi-GPU system, a reconstruction process is inserted and the hash table is reallocated to the appropriate GPU.

For example, in a system equipped with three GPUs, if the size of the hash table fits roughly into the RAM of the three GPUs, after the GPU-Scan of the INNER table is completed, the hash value of the join key to be used in the next GPU-Join is calculated, and if the remainder when dividing this by 3 is 0, it is reallocated to GPU0, if it is 1 then it is reallocated to GPU1, and if it is 2 then it is reallocated to GPU2.

By inserting this process, it is possible to create a state in which when GPU-Join is executed on GPU0, the hash table will only contain INNER rows whose remainder when the hash value is divided by 3 is 0, and similarly, GPU1 will only contain INNER rows whose remainder when the hash value is divided by 3 is 1.
}
![Multi-GPUs-Join Pinned-Inner-Buffer](./img/pinned_inner_buffer_01.png)

@ja{
次にこの分割されたハッシュ表を用いてGPU-Joinを実行する場合、最初にOUTER側のテーブルからデータをロードしたGPU（ここではGPU2としましょう）がハッシュ表を参照する際、OUTER側の行から計算したハッシュ値を3で割った剰余が2以外であると、そのGPU上でマッチするINNER側の行は明らかに存在しません。
そのため、GPU2ではハッシュ値を3で割った剰余が2であるものだけから成る結合結果が生成されます。次に、このOUTER側のデータはGPU-to-GPU CopyによってGPU1へと転送され、そこではハッシュ値を3で割った剰余が1であるものだけから成る結合結果が生成されます。

これを繰り返すと、各GPU上で「部分的なHash-Joinの結果」が生成されますが、これらを統合したものは完全なHash-Joinの結果と等しくなり、結果としてGPU搭載RAMよりも大きなサイズのINNER側ハッシュ表であってもGPU-Joinを実行する事ができるようになりました。
}
@en{
Next, when GPU-Join is executed using this divided hash table, when the GPU that first loaded data from the OUTER table (let's call it GPU2 here) references the hash table, if the remainder when dividing the hash value calculated from the OUTER row by 3 is other than 2, then there will obviously be no matching INNER row on that GPU.

Therefore, GPU2 will generate a join result consisting of only hash values ​​whose remainder when divided by 3 is 2. Next, this OUTER data is transferred to GPU1 by GPU-to-GPU Copy, which generates a join result consisting of only hash values ​​whose remainder when divided by 3 is 1.

By repeating this process, "partial Hash-Join results" are generated on each GPU, but the combination of these is equal to the complete Hash-Join result, and as a result, it is now possible to execute GPU-Join even if the INNER hash table is larger in size than the GPU's on-board RAM.
}
![Multi-GPUs-Join Pinned-Inner-Buffer](./img/pinned_inner_buffer_02.png)
@ja{
本機能に関連して、`pg_strom.pinned_inner_buffer_partition_size`パラメータが追加されました。
これはPinned-Inner-Bufferを複数のGPUに分割する際の閾値となるサイズを指定するもので、初期値としてGPU搭載メモリの80～90%程度の値が設定されていますので、通常は管理者がこれを変更する必要はありません。
}
@en{
In relation to this feature, the `pg_strom.pinned_inner_buffer_partition_size` parameter has been added.
This specifies the threshold size for dividing the Pinned-Inner-Buffer among multiple GPUs. The initial value is set to about 80-90% of the GPU's installed memory, so administrators usually do not need to change this.
}
