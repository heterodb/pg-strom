@ja:<h1>GPUメモリストア (Gstore_Fdw)</h1>
@en:<h1>GPU Memory Store (Gstore_Fdw)</h1>

@ja:#概要
@en:#Overview

<!--
@ja{
GPUにはホストシステムのRAMとは独立なデバイスメモリが搭載されており、GPUで計算するにはPCI-Eバスなどを通じて、一旦、ホストシステムやストレージデバイスからデータをGPUデバイスメモリ側へ転送する必要があります。
PG-StromがSQLクエリをGPUで処理する場合も例外ではなく、内部的には、PostgreSQLテーブルから読み出したレコードをGPUへと転送し、その上でGPUでSQLの各種処理を実行します。
しかしこれらの処理には、多かれ少なかれテーブルの読み出しやデータの転送に時間を要します（多くの場合、GPUでの処理よりも遥かに長い時間の！）。
}
@en{}

@ja{
GPUメモリストア（Gstore_Fdw）とは、GPUデバイスメモリ上に予め確保した領域を外部テーブル（Foreign Table）を介して読み書きする機能です。
主に検索・分析を目的としたSQLクエリを実行する際には、既にGPU上にデータがロードされているため、改めてテーブルからレコードを読み出したり、PCI-Eバスを介してデータを転送したりする必要がありません。


}
@en{}

-->

under construction


@ja:#運用
@en:#Operations

<!--
テーブル定義、オプション、チューニングのヒント（PMEMとかreuse
-->

under construction


@ja:#保守
@en:#Maintenance

under construction

<!--

![Architecture of Gstore_Fdw](./img/gstore_fdw-overview.png)

![Replication/Backup of Gstore_Fdw](./img/gstore_fdw-replication.png)

![Internal Layout of Gstore_Fdw](./img/gstore_fdw-layout.png)

-->





