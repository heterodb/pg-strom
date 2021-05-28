
@ja:#はじめに
@en:#Home

@ja: 本章ではPG-Stromの概要、および開発者コミュニティについて説明します。
@en: This chapter introduces the overview of PG-Strom, and developer's community.

@ja:##PG-Stromとは?
@en:##What is PG-Strom?

@ja{
PG-StromはPostgreSQL v11および以降のバージョン向けに設計された拡張モジュールで、チップあたり数千個のコアを持つGPU(Graphic Processor Unit)デバイスを利用する事で、大規模なデータセットに対する集計・解析処理やバッチ処理向けのSQLワークロードを高速化するために設計されています。
}
@en{
PG-Strom is an extension module of PostgreSQL designed for version 11 or later. By utilization of GPU (Graphic Processor Unit) device which has thousands cores per chip, it enables to accelerate SQL workloads for data analytics or batch processing to big data set.
}
@ja{
PG-Stromの中核となる機能は、SQL命令から自動的にGPUプログラムを生成するコードジェネレータと、SQLワークロードをGPU上で非同期かつ並列に実行する実行エンジンです。現バージョンではSCAN（WHERE句の評価）、JOINおよびGROUP BYのワークロードに対応しており、GPU処理にアドバンテージがある場合にはPostgreSQL標準の実装を置き換える事で、ユーザやアプリケーションからは透過的に動作します。
}
@en{
Its core features are GPU code generator that automatically generates GPU program according to the SQL commands and asynchronous parallel execution engine to run SQL workloads on GPU device. The latest version supports SCAN (evaluation of WHERE-clause), JOIN and GROUP BY workloads. In the case when GPU-processing has advantage, PG-Strom replaces the vanilla implementation of PostgreSQL and transparentlly works from users and applications.
}
@ja{
PG-Stromは２つのストレージオプションを持っています。一つは行形式でデータを保存するPostgreSQLのheapストレージシステムで、これは必ずしも集計・解析系ワークロードに最適ではありませんが、一方で、トランザクション系データベースからデータを移動する事なく集計処理を実行できるというアドバンテージがあります。もう一つは、列形式の構造化データ形式である Apache Arrow ファイルで、行単位のデータ更新には不向きであるものの、効率的に大量データをインポートする事ができ、外部データラッパ(FDW)を通して効率的なデータの検索・集計が可能です。
}
@en{
PG-Strom has two storage options. The first one is the heap storage system of PostgreSQL. It is not always optimal for aggregation / analysis workloads because of its row data format, on the other hands, it has an advantage to run aggregation workloads without data transfer from the transactional database. The other one is Apache Arrow files, that have structured columnar format. Even though it is not suitable for update per row basis, it enables to import large amount of data efficiently, and efficiently search / aggregate the data through foreign data wrapper (FDW).
}

@ja{
PG-Stromの特徴的な機能の一つが、NVME/NVME-oFデバイスからCPU/RAMをバイパスしてGPUに直接データを読み出し、GPUでSQL処理を実行する事でデバイスの帯域を最大限に引き出すGPUダイレクトSQL機能です。v3.0では新たにNVIDIA GPUDirect Storageにも対応し、ローカルNVME-SSDだけでなく、NVME-oFを介したSDS(Software Defined Storage)デバイスや、共有ファイルシステムからの読み出しにも対応します。
}
@en{
One of the characteristic feature of PG-Strom is GPUDirect SQL that bypasses the CPU/RAM to read the data from NVME / NVME-oF to the GPU directly. SQL processing on the GPU maximizes the bandwidth of these devices. PG-Strom v3.0 newly supports NVIDIA GPUDirect Storage, it allows to support SDS (Software Defined Storage) over the NVME-oF protocol and shared filesystems.
}

@ja{
また、v3.0では一部のPostGIS関数と、ジオメトリデータのGiSTインデックス探索をGPU側で実行する事が可能になりました。更新の多いテーブルの内容を予めGPUに複製しておくGPUキャッシュ機能と併せて、リアルタイムな位置情報に基づく検索、分析処理が可能となります。
}
@en{
Also, the v3.0 newly supports execution of some PostGIS function and GiST index search on the GPU side. Along with the GPU cache, that duplicates the table contents often updated very frequently, it enables search / analysis processing based on the real-time locational information.
}

@ja:## ライセンスと著作権
@en:## License and Copyright

@ja{
PG-StromはPostgreSQLライセンスに基づいて公開・配布されているオープンソースソフトウェアです。
ライセンスの詳細は[LICENSE](https://raw.githubusercontent.com/heterodb/pg-strom/master/LICENSE)を参照してください。
}
@en{
PG-Strom is an open source software distributed under the PostgreSQL License.
See [LICENSE](https://raw.githubusercontent.com/heterodb/pg-strom/master/LICENSE) for the license details.
}

@ja:##コミュニティ
@en:##Community

@ja{
PG-Stromに関する質問や要望、障害報告などは、[GitHubのDiscussion](https://github.com/heterodb/pg-strom/discussions)ページに投稿するようお願いします。

本掲示板は、世界中に公開されたパブリックの掲示板である事に留意してください。つまり、自己責任の下、秘密情報が誤って投稿されないように注意してください。

本掲示板の優先言語は英語です。ただ一方で、歴史的経緯によりPG-Stromユーザの多くの割合が日本人である事は承知しており、Discussion上で日本語を利用した議論が行われることも可能とします。その場合、Subject(件名)に`(JP)`という接頭句を付ける事を忘れないようにしてください。これは非日本語話者が不要なメッセージを読み飛ばすために有用です。
}
@en{
Please post your questions, requests and trouble reports to the [Discussion of GitHubの](https://github.com/heterodb/pg-strom/discussions).

Please pay attention it is a public board for world wide. So, it is your own responsibility not to disclose confidential information.

The primary language of the discussion board is English. On the other hands, we know major portion of PG-Strom users are Japanese because of its development history, so we admit to have a discussion on the list in Japanese language. In this case, please don't forget to attach `(JP)` prefix on the subject like, for non-Japanese speakers to skip messages.
}

@ja:###バグや障害の報告
@en:###Bug or troubles report
@ja{
結果不正やシステムクラッシュ/ロックアップ、その他の疑わしい動作を発見した場合は、[PG-Strom Issue Tracker](https://github.com/heterodb/pg-strom/issues)で新しいイシューをオープンし **bug** タグを付けてください。
}
@en{
If you got troubles like incorrect results, system crash / lockup, or something strange behavior, please open a new issue with **bug** tag at the [PG-Strom Issue Tracker](https://github.com/heterodb/pg-strom/issues).
}


@ja{
バグレポートの作成に際しては、下記の点に留意してください。
- 同じ問題を最新版で再現する事ができるかどうか?
    - PG-Stromの最新版だけでなく、OS、CUDA、PostgreSQLおよび関連ソフトウェアの最新版でテストする事をお勧めします。
- PG-Stromが無効化された状態でも同じ問題を再現できるかどうか?
    - GUCパラメータ `pg_strom.enabled` によってPG-Stromの有効/無効を切り替える事ができます。
- 同じ既知問題が既にGitHubのイシュートラッカーに存在するかどうか？
    - _close_ 状態のイシューを検索するのを忘れないようにしてください。
}

@en{
Please ensure the items below on bug reports.

- Whether you can reproduce the same problem on the latest revision?
    - Hopefully, we recommend to test on the latest OS, CUDA, PostgreSQL and related software.
- Whether you can reproduce the same problem if PG-Strom is disabled?
    - GUC option pg_strom.enabled can turn on/off PG-Strom.
- Is there any known issues on the issue tracker of GitHub?
    - Please don't forget to search _closed_ issues
}
@ja{
以下のような情報はバグ報告において有用です。

- 問題クエリの`EXPLAIN VERBOSE`出力
- 関連するテーブルのデータ構造（`psql`上で`\d+ <table name>`を実行して得られる）
- 出力されたログメッセージ（verbose出力が望ましい）
- デフォルト値から変更しているGUCオプションの値
- ハードウェア設定（特にGPUの型番とRAM容量）
}
@en{
The information below are helpful for bug-reports.

- Output of `EXPLAIN VERBOSE` for the queries in trouble.
- Data structure of the tables involved with `\d+ <table name>` on psql command.
- Log messages (verbose messages are more helpful)
- Status of GUC options you modified from the default configurations.
- Hardware configuration - GPU model and host RAM size especially.
}
@ja{
あなたの環境で発生した疑わしい動作がバグかどうか定かではない場合、新しいイシューのチケットをオープンする前にDiscussion掲示板へ報告してください。追加的な情報採取の依頼など、開発者は次に取るべきアクションを提案してくれるでしょう。
}
@en{
If you are not certain whether the strange behavior on your site is bug or not, please report it to the discussion board prior to the open a new issue ticket. Developers may be able to suggest you next action - like a request for extra information.
}

@ja:### 新機能の提案
@en:### New features proposition
@ja{
何か新機能のアイデアがある場合、[PG-Strom Issue Tracker](https://github.com/heterodb/pg-strom/issues)で新しいイシューをオープンし **feature** タグを付けてください。続いて、他の開発者と議論を行いましょう。
}
@en{
If you have any ideas of new features, please open a new issue with **feature** tag at the [PG-Strom Issue Tracker](https://github.com/heterodb/pg-strom/issues), then have a discussion with other developers.
}
@ja{
望ましい新機能提案は以下のような要素を含んでいます。

- あなたはどのような問題を解決/改善したいのか？
- あなたのワークロード/ユースケースにとってどの程度深刻なのか？
- どのようにそれを実装するのか？
- （もしあれば）予想される欠点・トレードオフ
}
@en{
A preferable design proposal will contain the items below.

- What is your problem to solve / improve?
- How much serious is it on your workloads / user case?
- Way to implement your idea?
- Expected downside, if any.
}
@ja{
開発者の間でその必要性に関してコンセンサスが得られると、コーディネーターはイシューチケットに**accepted**タグを付け、そのチケットはその後の開発作業のトラッキングのために利用されます。それ以外の場合、イシューチケットには**rejected**タグを付けてクローズされます。

一度プロポーザルが却下されたとしても、将来においてまた異なった決定があるかもしれません。周辺状況が変わった場合、新機能の再提案を躊躇する必要はありません。
}
@en{
Once we could make a consensus about its necessity, coordinator will attach accepted tag and the issue ticket is used to track rest of the development. Elsewhere, the issue ticket got rejected tag and closed.

Once a proposal got rejected, we may have different decision in the future. If comprehensive circumstance would be changed, you don't need to hesitate revised proposition again.
}
@ja{
開発段階では、パッチファイルをイシューチケットに添付するようにしてください。pull-requestは使用しません。
}
@en{
On the development stage, please attach patch file on the issue ticket. We don't use pull request.
}

@ja:## サポートポリシー
@en:## Support Policy
@ja{
PG-Strom development teamはHeteroDB Software Distribution Centerから配布された最新版のみをサポートします。
トラブルが発生した場合、まずその問題は最新版のリリースで再現するかどうかを確かめてください。

また、これはボランティアベースのコミュニティサポートのポリシーである事に留意してください。つまり、サポートはベストエフォートでかつ、SLAの定義もありません。

もし商用のサポートが必要である場合、HeteroDB社（contact@heterodb.com）にコンタクトしてください。
}
@en{
The PG-Strom development team will support the latest release which are distributed from the HeteroDB Software Distribution Center only. So, people who met troubles needs to ensure the problems can be reproduced with the latest release.

Please note that it is volunteer based community support policy, so our support is best effort and no SLA definition.

If you need commercial support, contact to HeteroDB,Inc (contact@heterodb.com).
}
