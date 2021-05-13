@ja:#GPU版PostGIS
@en:#GPU-PostGIS

@ja{
本章ではGPU版PostGISについて説明します。
}
@en{}

@ja:##概要
@en:##Overview

@ja{
PostGISとは、地理空間情報を取り扱うためのPostgreSQL向け拡張モジュールです。
点、線分、ポリゴンなど地理データを取り扱うためのデータ型（<code>Geometry</code>）を提供しているほか、距離の計算や包含、交差関係の判定など、地理データ要素を評価するための関数や演算子を数多く取り揃えています。
また、一部の演算子についてはPostgreSQLのGiST(Generalized Search Tree)の仕組みを用いてR木による高速な検索も可能になっています。
2001年に最初のバージョンが公開されて以降、20年以上にわたり開発者コミュニティによって機能強化やメンテナンスが行われています。
}

@en{}

@ja{
これらPostGISの提供する関数や演算子は、総数で500を超える非常に大規模なものです。
そのため、PG-Stromでは比較的利用頻度の高いいくつかのPostGIS関数だけをGPU用に移植しています。

例えば、以下のようなPostGIS関数がそれに該当します。

- <code>geometry st_point(float8 lon,float8 lat)</code>
    - 経度緯度から、点（Point）であるジオメトリ型を生成する。
- <code>bool st_contains(geometry a,geometry b)</code>
    - ジオメトリaがジオメトリbが包含するかどうかを判定する。
- <code>bool st_crosses(geometry,geometry)</code>
    - ジオメトリ同士が交差するかどうかを判定する。
- <code>text st_relate(geometry,geometry)</code>
    - ジオメトリ同士の関係を[DE-9IM(Dimensionally Extended 9-Intersection Model)](https://en.wikipedia.org/wiki/DE-9IM)の行列表現として返します。

また、テーブル同士の結合条件がGiSTインデックス（R木）の利用に適する場合、GpuJoinはGiSTインデックス（R木）をGPU側にロードし、結合すべき行の絞り込みを高速化するために使用する事ができます。
これは例えば、GPSから取得したモバイル機器の位置（点）とエリア定義データ（ポリゴン）を突き合わせるといった処理の高速化に寄与します。
}
@en{}

@ja:##PostGISの利用
@en:##PostGIS Usage

@ja{
GPU版PostGISを利用するために特別な設定は必要ありません。

PostGISをパッケージ又はソースコードからインストールし、CREATE EXTENSION構文を用いてジオメトリデータ型やPostGIS関数が定義されていれば、PG-Stromはクエリに出現したPostGIS関数がGPUで実行可能かどうかを自動的に判定します。
}

@ja{
PostGIS自体のインストールについては、[PostGISのドキュメント](http://postgis.net/docs/postgis-ja.html)を参照してください。
}
@en{
http://postgis.net/docs/
}

@ja{
例えば、以下のクエリにはGPU実行可能なPostGIS関数である<code>st_contains()</code>と<code>st_makepoint()</code>を使用しており、ジオメトリ型の定数<code>'polygon ((10 10,30 10,30 20,10 20,10 10))'</code>の範囲内にテーブルから読み出した二次元の点が含まれるかどうかを判定します。

これらの関数が GPU Filter: の一部として表示されている事からも分かるように、PG-Stromは対応済みのPostGIS関数を自動的に検出し、可能な限りGPUで実行しようと試みます。
}
@en{

}

```
=# explain select * from dpoints where st_contains('polygon ((10 10,30 10,30 20,10 20,10 10))', st_makepoint(x,y));

                              QUERY PLAN
------------------------------------------------------------------------------------------
 Custom Scan (GpuScan) on dpoints  (cost=1397205.10..12627630.76 rows=800 width=28)
   GPU Filter: st_contains('01030000000100000005000000000000000000244000000000000024400000000000003E4000000000000024400000000000003E4000000000000034400000000000002440000000000000344000000000000024400000000000002440'::geometry, st_makepoint(x, y))
   GPU Preference: GPU0 (NVIDIA Tesla V100-PCIE-16GB)
(3 rows)
```

@ja:##GiSTインデックス
@en:##GiST Index

@ja{
<code>st_contains()</code>や<code>st_crosses()</code>などジオメトリ同士の関係性を評価するPostGIS関数の一部は、GiSTインデックス（R木）に対応しており、CPUだけを用いて検索を行う場合にも高速な絞り込みを行う事が可能です。
PG-StromのGpuJoinでは、テーブル同士の結合条件がGiSTインデックス（R木）で高速化可能な場合、結合対象テーブルの中身だけでなく、GiSTインデックスも同時にGPU側へ転送し、結合対象の行を高速に絞り込むために使用する事があります。
この処理は通常、CPUよりも遥かに高い並列度で実行されるため、かなりの高速化を期待する事ができます。

一方、GpuScanはテーブル単体のスキャンにGiSTインデックスを使用しません。これは、CPUによるIndexScanでの絞り込みの方が高速である事が多いからです。
}

@en{

}

@ja{
ジオメトリデータにGiSTインデックスを設定するには、<code>CREATE INDEX</code>構文を使用します。

以下の例は、市町村の境界線データ（giscityテーブルのgeom列）に対してGiSTインデックスを設定するものです。
}
@en{}

```
=# CREATE INDEX on giscity USING gist (geom);
CREATE INDEX
```

@ja{
以下の実行計画は、市町村の境界線データ（giscityテーブル）と緯度経度データ（dpointsテーブル）を突き合わせ、ポリゴンとして表現された市町村の領域内に含まれる緯度経度データ（点）の数を市町村ごとに出力するものです。

オプティマイザによりGpuJoinが選択され、giscityテーブルとdpointsテーブルの結合にはGpuGiSTJoinが選択されている。
IndexFilter:の行には、GiSTインデックスによる絞り込み条件が<code>(g.geom ~ st_makepoint(d.x, d.y))</code>であり、使用するインデックスが<code>giscity_geom_idx</code>である事が示されています。

GiSTインデックスの使用により、GPUであっても比較的「重い」処理であるPostGIS関数を実行する前に、明らかに条件にマッチしない組み合わせを排除する事ができるため、大幅な検索処理の高速化が期待できます。
}
@en{}

```
=# EXPLAIN
   SELECT pref, city, count(*)
     FROM giscity g, dpoints d
    WHERE pref = 'Tokyo' AND st_contains(g.geom,st_makepoint(d.x, d.y))
    GROUP BY pref, city;
                                                QUERY PLAN
-----------------------------------------------------------------------------------------------------------
 GroupAggregate  (cost=5700646.35..5700759.39 rows=5024 width=29)
   Group Key: g.n03_001, g.n03_004
   ->  Sort  (cost=5700646.35..5700658.91 rows=5024 width=29)
         Sort Key: g.n03_004
         ->  Custom Scan (GpuPreAgg)  (cost=5700274.71..5700337.51 rows=5024 width=29)
               Reduction: Local
               Combined GpuJoin: enabled
               GPU Preference: GPU0 (NVIDIA Tesla V100-PCIE-16GB)
               ->  Custom Scan (GpuJoin) on dpoints d  (cost=638671.58..5668511.23 rows=50821573 width=21)
                     Outer Scan: dpoints d  (cost=0.00..141628.18 rows=7999618 width=16)
                     Depth 1: GpuGiSTJoin(nrows 7999618...50821573)
                              HeapSize: 3251.36KB
                              IndexFilter: (g.geom ~ st_makepoint(d.x, d.y)) on giscity_geom_idx
                              JoinQuals: st_contains(g.geom, st_makepoint(d.x, d.y))
                     GPU Preference: GPU0 (NVIDIA Tesla V100-PCIE-16GB)
                     ->  Seq Scan on giscity g  (cost=0.00..8929.24 rows=6353 width=1883)
                           Filter: ((pref)::text = 'Tokyo'::text)
(17 rows)

```

