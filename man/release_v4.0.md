@ja:#PG-Strom v4.0リリース
@en:#PG-Strom v4.0 Release

<div style="text-align: right;">PG-Strom Development Team (xx-xxx-2021)</div>

@ja:##概要
@en:##Overview

@ja{
PG-Strom v4.0における主要な変更は点は以下の通りです。

@ja{
- GpuPreAggのGPU側処理を一新し、特にグループ数が多い場合の処理速度が向上しました。
- Arrow_Fdwがmin/max統計値を用いた範囲インデックススキャンに対応しました。
- Pg2Arrowがmin/max統計値付きのArrowファイル出力に対応しました。
- ネットワークパケットをキャプチャするPcap2Arrowツールを追加しました。
- HyperLogLogアルゴリズムを用いたカーディナリティの推定に対応しました。

- PostgreSQL v14に対応しました。

- todo: GPU-Direct SQLで、削除済みタプルを含むブロックの直接読み出しに対応しました。
- todo: GPU版PostGISでSt_intersect()に対応しました。
- todo: heterodb-extraパッケージがUbuntu 20.04に対応しました。
}
@en{}

@ja:##動作環境
@en:##Prerequisites

@ja{
- PostgreSQL v11, v12, v13, v14
- CUDA Toolkit 11.4 以降
- CUDA ToolkitのサポートするLinuxディストリビューション
- Intel x86 64bit アーキテクチャ(x86_64)
- NVIDIA GPU CC 6.0 以降 (Pascal以降; Volta以降を推奨)
}
@en{
- PostgreSQL v11, v12, v13, v14
- CUDA Toolkit 11.4 or later
- Linux distributions supported by CUDA Toolkit
- Intel x86 64bit architecture (x86_64)
- NVIDIA GPU CC 6.0 or later (Pascal at least; Volta or newer is recommended)
}

@ja:##Arrow_Fdwの統計情報サポート
@en:##Arrow_Fdw supports min/max statistics

@ja{
Pg2Arrowでmin/max統計情報付きのApache Arrowファイルを生成する事ができるようになりました。

Apache Arrowファイルは、内部的にRecord Batchと呼ばれる副ブロック単位でデータを管理しており、
例えるなら、全体で1億件のデータを有するApache Arrowファイルの内部に、50万件のデータを保持する
Record Batchを200個保持している、といったファイル形式を有しています。

Pg2Arrowの新たなオプション`--stat=COLUMN_NAME`は、RecordBatch単位で指定した列の最大値/最小値を
記録しておき、それをApache ArrowのCustom-Metadataメカニズムを利用してフッタに埋め込みます。

min/max統計情報に対応していないアプリケーションからは、単純にCustom-Metadataフィールドに
埋め込まれた未知のKey-Value属性として扱われるため、相互運用性に問題が生じる事はありません。
}

@ja{
Arrow_Fdwを介してApache Arrowファイルを読み出す際、上記のmin/max統計情報を利用した
範囲インデックススキャンに対応しました。

例えば、Arrow_Fdw外部テーブルに対する検索条件が以下のようなものであった場合、

`WHERE ymd BETERRN '2020-01-01'::date AND '2021-12-31'::date`

ymdフィールドの最大値が`'2020-01-01'::date`未満であるRecord Batchや、
ymdフィールドの最小値が`'2021-12-31`::date`より大きなRecord Batchは、
検索条件にマッチしない事が明らかであるため、Arrow_FdwはこのRecord Batchを読み飛ばします。

これにより、例えばログデータのタイムスタンプなど、近しい値を持つレコードが近傍に集まっている
パターンのデータセットにおいては、範囲インデックスを用いた絞込みと同等の性能が得られます。
}

@ja:##集約処理（GpuPreAgg）の改良とHyperLogLog対応
@en:##Aggregation (GpuPreAgg) renewal and HyperLogLog support

@ja{
集約処理（GpuPreAgg）のGPU側の実装を一新しました。

`GROUP BY`のない集約演算や、グループ数の少ない集約処理の場合は、L1キャッシュと同等の
アクセス速度を持つ「GPU共有メモリ」上での集約処理を行い、速度低下の原因となりやすい、
グローバルメモリへのアトミック演算の回数を削減しています。

また、GPU上のハッシュテーブルのデータ構造を改良する事で、ハッシュスロットの競合が発生した
際にもパフォーマンスの低下が起こりにくくなっています。
}

@ja{
HyperLogLogによるカーディナリティの推定に対応しました。

`SELECT COUNT(distinct KEY)`は非常によく使われる処理ですが、大規模なデータに対して
重複のないキー値の数（カーディナリティ）を集計する場合、並列分散処理と相性が悪いため、
長い処理時間を要するという問題がありました。

HyperLogLogとは、ハッシュ化したキー値の統計的性質を利用して、線形時間かつ予測可能な
メモリ消費量でカーディナリティを推定するアルゴリズムです。

PG-Strom v4.0では、HyperLogLogを使用してカーディナリティを推定するための集約関数
`hll_count(KEY)`が追加されました。
}







@ja:##その他の変更
@en:##Other updates









@ja:##廃止された機能
@en:##Deprecated features
















