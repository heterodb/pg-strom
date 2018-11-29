@ja: 本章ではPG-Stromの概要、および開発者コミュニティについて説明します。
@en: This chapter introduces the overview of PG-Strom, and developer's community.

@ja:# PG-Stromとは?
@en:# What is PG-Strom?

@ja{
PG-StromはPostgreSQL v9.6および以降のバージョン向けに設計された拡張モジュールで、チップあたり数千個のコアを持つGPU(Graphic Processor Unit)デバイスを利用する事で、大規模なデータセットに対する集計・解析処理やバッチ処理向けのSQLワークロードを高速化するために設計されています。
}
@en{
PG-Strom is an extension module of PostgreSQL designed for version 9.6 or later. By utilization of GPU (Graphic Processor Unit) device which has thousands cores per chip, it enables to accelerate SQL workloads for data analytics or batch processing to big data set.
}
@ja{
PG-Stromの中核となる機能は、SQL命令から自動的にGPUプログラムを生成するコードジェネレータと、SQLワークロードをGPU上で非同期かつ並列に実行する実行エンジンです。現バージョンではSCAN（WHERE句の評価）、JOINおよびGROUP BYのワークロードに対応しており、GPU処理にアドバンテージがある場合にはPostgreSQL標準の実装を置き換える事で、ユーザやアプリケーションからは透過的に動作します。
}
@en{
Its core features are GPU code generator that automatically generates GPU program according to the SQL commands and asynchronous parallel execution engine to run SQL workloads on GPU device. The latest version supports SCAN (evaluation of WHERE-clause), JOIN and GROUP BY workloads. In the case when GPU-processing has advantage, PG-Strom replaces the vanilla implementation of PostgreSQL and transparentlly works from users and applications.
}
@ja{
また、PG-StromはいくつかのDWH専用システムとは異なり、行形式でデータを保存するPostgreSQLとストレージシステムを共有しています。これは必ずしも集計・解析系ワークロードに最適ではありませんが、一方で、トランザクション系データベースからデータを移動することなく集計処理を実行できるというアドバンテージでもあります。
}
@en{
Unlike some DWH systems, PG-Strom shares the storage system of PostgreSQL which saves data in row-format. It is not always best choice for summary or analytics workloads, however, it is also an advantage as well. Users don't need to export and transform the data from transactional database for processing.
}
@ja{
PG-Strom v2.0ではストレージ読出し能力が強化されました。SSD-to-GPUダイレクトSQL機構はストレージ（NVME-SSD）からGPUへ直接データをロードし、クエリを処理するGPUへ高速にデータを供給する事を可能にします。
}
@en{
PG-Strom v2.0 enhanced the capability of reading from storage. SSD-to-GPU Direct SQL mechanism allows to load from storage (NVME-SSD) to GPU directly, and supply data to GPU that runs SQL workloads.
}
@ja{
一方、高度な統計解析や機械学習といった極めて計算集約度の高い問題に対しても、PL/CUDAやgstore_fdwといった機能を使用する事で、データベース管理システム上で計算処理を行い、結果だけをユーザへ返すといった使い方をする事が可能です。
}
@en{
On the other hands, the feature of PL/CUDA and gstore_fdw allows to run highly computing density problems, like advanced statistical analytics or machine learning, on the database management system, and to return only results to users.
}

@ja:## ライセンスと著作権
@en:## License and Copyright

@ja{
PG-StromはGPL(GNU Public License)v2に基づいて公開・配布されているオープンソースソフトウェアです。
ライセンスの詳細は[LICENSE](https://raw.githubusercontent.com/heterodb/pg-strom/master/LICENSE)を参照してください。
}
@en{
PG-Strom is an open source software distributed under the GPL(GNU Public License) v2.
See [LICENSE](https://raw.githubusercontent.com/heterodb/pg-strom/master/LICENSE) for the license details.
}
@ja{
PG-Stromの著作権はPG-Strom Development Teamが有しています。PG-Strom Development Teamは法的な主体ではなく、国籍を問わず、PG-Stromプロジェクトに貢献した個々の開発者や企業の総称です。
}
@en{
PG-Strom Development Team reserves the copyright of the software.
PG-Strom Development Team is an international, unincorporated association of individuals and companies who have contributed to the PG-Strom project, but not a legal entity.
}

@ja:# コミュニティ
@en:# Community

@ja{
開発者コミュニティのMLが[PG-Strom community ML](https://groups.google.com/a/heterodb.com/forum/#!forum/pgstrom)に準備されています。
PG-Stromに関連した質問、要望、障害報告などはこちらにポストしてください。

本MLは世界中に公開されたパブリックのMLである事に留意してください。つまり、自己責任の下、秘密情報が誤って投稿されないように注意してください。

本MLの優先言語は英語です。ただ一方で、歴史的経緯によりPG-Stromユーザの多くの割合が日本人である事は承知しており、ML上で日本語を利用した議論が行われることも可能とします。その場合、Subject(件名)に`(JP)`という接頭句を付ける事を忘れないようにしてください。これは非日本語話者が不要なメッセージを読み飛ばすために有用です。

}
@en{
We have a community mailing-list at: [PG-Strom community ML](https://groups.google.com/a/heterodb.com/forum/#!forum/pgstrom) It is a right place to post questions, requests, troubles and etc, related to PG-Strom project.

Please pay attention it is a public list for world wide. So, it is your own responsibility not to disclose confidential information.

The primary language of the mailing-list is English. On the other hands, we know major portion of PG-Strom users are Japanese because of its development history, so we admit to have a discussion on the list in Japanese language. In this case, please don't forget to attach `(JP)` prefix on the subject like, for non-Japanese speakers to skip messages.
}

@ja:## バグや障害の報告
@en:## Bug or troubles report
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
あなたの環境で発生した疑わしい動作がバグかどうか定かではない場合、新しいイシューのチケットをオープンする前にメーリングリストへ報告してください。追加的な情報採取の依頼など、開発者は次に取るべきアクションを提案してくれるでしょう。
}
@en{
If you are not certain whether the strange behavior on your site is bug or not, please report it to the mailing-list prior to the open a new issue ticket. Developers may be able to suggest you next action - like a request for extra information.
}

@ja:## 新機能の提案
@en:## New features proposition
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

@ja:##バージョンポリシー
@en:##Versioning Policy

@ja{
PG-Stromのバージョン番号は`<major>.<minor>`という2つの要素から成ります。

マイナーバージョン番号は各リリース毎に増加し、バグ修正と新機能の追加を含みます。

メジャーバージョン番号は以下のような場合に増加します。

- 対応しているPostgreSQLバージョンのうち、いくつかがサポート対象外となった場合。
- 対応しているGPUデバイスのうち、いくつかがサポート対象外となった場合。
- エポックメイキングな新機能が追加となった場合。
}
@en{
PG-Strom's version number is consists of two portion; major and minor version. `<major>.<minor>`

Its minor version shall be incremented for each release; including bug fixes and new features.
Its major version shall be incremented in the following situation.

- Some of supported PostgreSQL version gets deprecated.
- Some of supported GPU devices gets deprecated.
- New version adds epoch making features.
}
