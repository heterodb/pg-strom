@ja:<h1>インデックス対応</h1>
@en:<h1>Index Support</h1>

@ja:#概要
@en:#Overview

@ja{
PostgreSQLは何種類かのインデックス形式に対応しており、デフォルトで選択されるB-treeインデックスは特定の値を持つレコードを高速に検索する事が可能です。これ以外にも、Hash、BRIN、GiST、GINなど特性の異なるインデックス形式が提供されており、現在のところPG-StromはBRINインデックスにのみ対応しています。
}
@en{
PostgreSQL supports several index strategies. The default is B-tree that can rapidly fetch records with a particular value. Elsewhere, it also supports Hash, BRIN, GiST, GIN and others that have own characteristics for each.
PG-Strom supports only BRIN-index right now.
}

@ja{
BRINインデックスは、時系列データにおけるタイムスタンプ値など、物理的に近傍に位置するレコード同士が近しい値を持っている事が期待できるデータセットに対して有効に作用します。本来は全件スキャンが必要な操作であっても、明らかに条件句にマッチしない領域を読み飛ばし、全件スキャンに伴うI/O量を削減する事が可能です。
}
@en{
BRIN-index works efficiently on the dataset we can expect physically neighbor records have similar key values, like timestamp values of time-series data.
It allows to skip blocks-range if any records in the range obviously don't match to the scan qualifiers, then, also enables to reduce the amount of I/O due to full table scan.
}

@ja{
PG-StromにおいてもBRINインデックスの特性を活用し、GPUにロードすべきデータブロックのうち明らかに不要であるものを読み飛ばす事が可能になっています。
}
@en{
PG-Strom also utilizes the feature of BRIN-index, to skip obviously unnecessary blocks from the ones to be loaded to GPU.
}

![BRIN-index Ovewview](./img/brin-index-overview.png)

@ja:#設定
@en:#Configuration

@ja{
BRINインデックスを利用するために特別な設定は必要ありません。

CREATE INDEX構文を用いて対象列にインデックスが設定されており、かつ、検索条件がBRINインデックスに適合するものであれば自動的に適用されます。

BRINインデックス自体の説明は、[PostgreSQLのドキュメント](https://www.postgresql.jp/document/current/html/brin.html)を参照してください。
}

@en{
No special configurations are needed to use BRIN-index.

PG-Strom automatically applies BRIN-index based scan if BRIN-index is configured on the referenced columns and scan qualifiers are suitable to the index.

Also see the [PostgreSQL Documentation](https://www.postgresql.org/docs/current/static/brin.html) for the BRIN-index feature.
}


@ja{
以下のGUCパラメータにより、PG-StromがBRINインデックスを使用するかどうかを制御する事ができます。デバッグやトラブルシューティングの場合を除き、通常は初期設定のままで構いません。

|パラメータ名          |型    |初期値|説明       |
|:---------------------|:----:|:----:|:----------|
|`pg_strom.enable_brin`|`bool`|`on`  |BRINインデックスを使用するかどうかを制御する。|

}

@en{
By the GUC parameters below, PG-Strom enables/disables usage of BRIN-index. It usually don't need to change from the default configuration, except for debugging or trouble shooting.

|Parameter             |Type  |Default|Description|
|:---------------------|:----:|:-----:|:----------|
|`pg_strom.enable_brin`|`bool`|`on`   |enables/disables usage of BRIN-index|
}

@ja:#操作
@en:#Operations

@ja{
`EXPLAIN`構文によりBRINインデックスが使用されているかどうかを確認する事ができます。
}

@en{
By `EXPLAIN`, we can check whether BRIN-index is in use.
}

```
postgres=# EXPLAIN ANALYZE
           SELECT * FROM dt
            WHERE ymd BETWEEN '2018-01-01' AND '2018-12-31'
              AND cat LIKE '%aaa%';
                                   QUERY PLAN
--------------------------------------------------------------------------------
 Custom Scan (GpuScan) on dt  (cost=94810.93..176275.00 rows=169992 width=44)
                              (actual time=1777.819..1901.537 rows=175277 loops=1)
   GPU Filter: ((ymd >= '2018-01-01'::date) AND (ymd <= '2018-12-31'::date) AND (cat ~~ '%aaa%'::text))
   Rows Removed by GPU Filter: 4385491
   BRIN cond: ((ymd >= '2018-01-01'::date) AND (ymd <= '2018-12-31'::date))
   BRIN skipped: 424704
 Planning time: 0.529 ms
 Execution time: 2323.063 ms
(7 rows)
```

@ja{
上記の例では`ymd`列にBRINインデックスが設定されており、`BRIN cond`の表示はBRINインデックスによる絞り込み条件を、`BRIN skipped`の表示はBRINインデックスにより実際に読み飛ばされたブロックの数を示しています。

この例では424704ブロックが読み飛ばされ、さらに、読み込んだブロックに含まれているレコードのうち4385491行が条件句によってフィルタされた事が分かります。

}

@en{
In the example above, BRIN-index is configured on the `ymd` column. `BRIN cond` shows the qualifier of BRIN-index for concentration. `BRIN skipped` shows the number of skipped blocks actually.

In this case, 424704 blocks are skipped, then, it filters out 4385491 rows in the loaded blocks by the scan qualifiers.
}

@ja{
データ転送のロスを減らすため、GpuJoinやGpuPreAggが直下のテーブルスキャンを引き上げ、自らテーブルのスキャン処理を行う事があります。この場合でも、BRINインデックスが利用可能であればインデックスによる絞り込みを行います。
以下の例は、GROUP BYを含む処理でBRINインデックスが使用されているケースです。
}

@en{
GpuJoin and GpuPreAgg often pulls up its underlying table scan and runs the scan by itself, to reduce inefficient data transfer. In this case, it also uses the BRIN-index to concentrate the scan.

The example below shows a usage of BRIN-index in a query which includes GROUP BY.
}

```
postgres=# EXPLAIN ANALYZE
           SELECT cat,count(*)
             FROM dt WHERE ymd BETWEEN '2018-01-01' AND '2018-12-31'
         GROUP BY cat;
                                   QUERY PLAN
--------------------------------------------------------------------------------
 GroupAggregate  (cost=6149.78..6151.86 rows=26 width=12)
                 (actual time=427.482..427.499 rows=26 loops=1)
   Group Key: cat
   ->  Sort  (cost=6149.78..6150.24 rows=182 width=12)
             (actual time=427.465..427.467 rows=26 loops=1)
         Sort Key: cat
         Sort Method: quicksort  Memory: 26kB
         ->  Custom Scan (GpuPreAgg) on dt  (cost=6140.68..6142.95 rows=182 width=12)
                                            (actual time=427.331..427.339 rows=26 loops=1)
               Reduction: Local
               Outer Scan: dt  (cost=4000.00..4011.99 rows=4541187 width=4)
                               (actual time=78.573..415.961 rows=4560768 loops=1)
               Outer Scan Filter: ((ymd >= '2018-01-01'::date) AND (ymd <= '2018-12-31'::date))
               Rows Removed by Outer Scan Filter: 15564
               BRIN cond: ((ymd >= '2018-01-01'::date) AND (ymd <= '2018-12-31'::date))
               BRIN skipped: 424704
 Planning time: 30.992 ms
 Execution time: 818.994 ms
(14 rows)
```
