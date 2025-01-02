PG-Strom
========
PG-Strom is an extension for PostgreSQL database.
It is designed to accelerate mostly batch and analytics workloads with
utilization of GPU and NVME-SSD, and Apache Arrow columnar.

For more details, see the documentation below.
http://heterodb.github.io/pg-strom/

Software is fully open source, distributed under PostgreSQL License.

---

PG-Strom + GPU Direct Storage Quickstart
========================================
このドキュメントはPG-Stromの導入から、GPU Direct Storageまでの設定を説明するものです。
本書はLocal NVMe SSDをGPU Direct Storage向けに利用する例を説明しています。

RHEL9向けに書かれていますが、[RHEL]と見出しに書かれている部分以外はRocky LinuxなどのRHELクローンOSと概ね共通なので参考になると思います。

それではこのガイドに従って環境構築して、PG-Stromの世界に足を踏み入れてみてください。


## [RHEL]バージョン固定

RHELではリリースバージョンを変更できます。インストールするCUDAに合わせて、リリースバージョンを固定化すると良いでしょう。CUDAとMOFED、Linuxのバージョンを適切かつ自由に選択できる場合はこの設定は不要です。

(rhel9)

```
$ sudo subscription-manager release --set=9.4
```

## ソフトウェアアップデートの実施
ソフトウェアアップデートを実行します。

```
$ sudo dnf update -y
```

## [RHEL]EUSリポジトリーの有効化
RHELではシステムへの延長更新サポート (EUS)が利用できます。EUSは通常のサポート期間より長い期間、マイナーバージョンのメンテナンスアップデートを受けられます。長期間の安定した利用を必要とする場合は適用してください。CUDAとMOFED、Linuxのバージョンを適切かつ自由に選択できる場合はこの設定は不要です。

(rhel9)

```
$ sudo subscription-manager repos --enable rhel-9-for-x86_64-appstream-eus-rpms --enable rhel-9-for-x86_64-baseos-eus-rpms
```

EPELとCodeReady Linux Builder（PowerTools）の有効化
EPELの利用には開発で使うパッケージセットリポジトリー（CodeReady Linux Builder）の有効化が必要です。ディストリビューションによって、リポジトリーの名称が異なることがあります。

詳細はアップストリームのドキュメントを確認してください。

- https://docs.fedoraproject.org/en-US/epel/getting-started/

(rhel9)

```
$ sudo subscription-manager repos --enable codeready-builder-for-rhel-9-$(arch)-rpms && sudo dnf install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
```

## 開発ツールなどのインストール

```
$ sudo dnf install wget git-core -y
$ sudo dnf groupinstall 'Development Tools' -y
```

## IOMMUの無効化

(rhel9)

```
$ sudo vi /etc/default/grub
  GRUB_CMDLINE_LINUX="crashkernel=auto iommu=off intel_iommu=off"
  GRUB_CMDLINE_LINUX_DEFAULT="rd.auto=1 iommu=off intel_iommu=off"

$ sudo grub2-mkconfig -o /boot/grub2/grub.cfg
```

## Nouveauドライバーの無効化

```
# cat > /etc/modprobe.d/disable-nouveau.conf <<EOF
blacklist nouveau
options nouveau modeset=0
EOF
```

## システム再起動

```
# shutdown -r now
```

## カーネルヘッダーなどのインストール
再起動後に、現在利用しているLinuxカーネルバージョン用のヘッダーなどをインストールします。

```
$ sudo dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
```

## MOFEDのインストール
ダウンロードURLから、利用予定のCUDAバージョンとLinuxディストリビューション、Linuxカーネルのバージョンに対応したMOFEDをダウンロードします。

- https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/

事前に必要なパッケージをインストールする。次はRHEL9の例。バージョンによって必要なパッケージは異なります。ビルド時に不足しているというメッセージが出たら、パッケージを追加します。

(rhel9)

```
$ sudo dnf install kernel-modules-extra kernel-rpm-macros lsof pciutils createrepo tk tcl gcc-gfortran perl-sigtrap
```

展開してビルド&インストールします。

(rhel9)

```
$ sudo mount MLNX_OFED_LINUX-23.10-3.2.2.0-rhel9.4-x86_64.iso /mnt
$ cd  /mnt
$ sudo ./mlnxofedinstall --with-nvmf --with-nfsrdma --enable-gds --add-kernel-support && sudo dracut -f
```

## DNF設定変更
MOFEDパッケージをシステムのパッケージで上書きしないように、excludepkgsで指定します。

```
$ sudo vi /etc/dnf/dnf.conf

[main]
...
excludepkgs=mstflint,openmpi,perftest,mpitests_openmpi,mlnx-*,ofed-*
```

`ofed_rpm_info`コマンドで出てくるパッケージを置き換えないように設定すること！
MOFEDのバージョンアップや削除時には取り除くこと。

一旦、再起動します。

```
$ sudo reboot
```


## Local NVMe周りの設定
MOFEDでインストールしたモジュールを確認します。`extra`以下のパスにあれば、Linuxカーネル外部のモジュールが使える状態です。

```
$ modinfo nvme
filename:       /lib/modules/5.14.0-427.31.1.el9_4.x86_64/extra/mlnx-nvme/host/nvme.ko
version:        1.0
license:        GPL
author:         Matthew Wilcox <willy@linux.intel.com>
rhelversion:    9.4
...
```

モジュールを読み込みます。

```
$ sudo su -
# modprobe nvme
# echo nvme > /etc/modules-load.d/nvme.conf
```

デバイスの数を確認します。

```
$ ls /dev |grep nvme
```

デバイスの数に合わせて、ソフトウェアRAIDの設定(-nでNVMe SSDデバイス数を指定)を行います。
 
```
# mdadm -C /dev/md0 -c 128 -l 0 -n 4 /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1
# mdadm --detail --scan > /etc/mdadm.conf
```

RAID ボリュームの udev ルールの`--no-devices`が指定された行の内容を修正します。

```
# vi /lib/udev/rules.d/63-md-raid-arrays.rules
...
#IMPORT{program}="/usr/sbin/mdadm --detail --no-devices --export $devnode"
IMPORT{program}="/usr/sbin/mdadm --detail --export $devnode"
```

partedコマンドを使って、パーティション作成を行います。ファイルシステムはext4を指定します。

```
# parted /dev/md0
(parted) p                                                                
Error: /dev/md0: unrecognised disk label
Model: Linux Software RAID Array (md)                                     
Disk /dev/md0: 4001GB
...
(parted) mklabel gpt                                                      
(parted) mkpart                                                           
Partition name?  []? "PG-Strom Storages"                                  
File system type?  [ext2]? ext4                                           
Start? 0%
End? 100%                                                                 
(parted) p                                                                                                             
Model: Linux Software RAID Array (md)
Disk /dev/md0: 4001GB
Sector size (logical/physical): 512B/512B
Partition Table: gpt
Disk Flags: 

Number  Start   End     Size    File system  Name               Flags
 1      1049kB  4001GB  4001GB  ext4         PG-Strom Storages

(parted) quit                                                             
Information: You may need to update /etc/fstab.
```

ファイルシステムを作成します。

```
# mkfs.ext4 /dev/md0p1
```

マウントします。ストレージを永続化するために、`/etc/fstab`に記述します。

```
# mkdir -p /opt/nvme
# mount /dev/md0p1 /opt/nvme   
# vi /etc/fstab
...
/dev/md0p1  /opt/nvme ext4 data=ordered 0 0
```

## CUDAのインストール
PG-Strom 5.xはCUDA 12.2以降が現時点の最低要件になります。適切なOSやカーネルバージョンを用意した上で対応するCUDAをインストールします。

- https://developer.nvidia.com/cuda-toolkit-archive


事前にdkmsパッケージを入れておくと便利です。このパッケージはEPELリポジトリーにあります。

```
$ sudo dnf install -y dkms
```

リポジトリーを追加します。

(rhel9)

```
$ sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
```

パッケージでCUDAをインストールするときは、バージョンを指定します。

```
sudo dnf install -y cuda-toolkit-12-3
```

GPUドライバーをインストールします。CUDAバージョンによってレガシードライバーとOpenドライバーが存在しますが、CUDA 12.3以降はOpenドライバーがインストールされていないとGDSが利用できません。また、OpenドライバーはTuring世代以降のGPUにしか対応していませんので、対応するGPUを用意するか、CUDA 12.2GAもしくは12.2 Update1までのバージョンが利用可能なGPUと環境を用意してください。

```
sudo dnf module install -y nvidia-driver:open-dkms
```

Note:
CUDA 12.2 Update2以降はGDSの仕様変更でGPUとGPUドライバーの組み合わせが重要になる。Turing未満（P,V）の世代では12.2 Update1を利用すること。Turing以降ではOpen GPUドライバーが必要。

MOFEDを入れたあと、CUDAと同じバージョンのnvidia-gdsパッケージをインストールします。

```
sudo dnf install -y nvidia-gds-12-3
```

## PostgreSQLのインストール
RHELおよびRHELクローンOSはPostgreSQLパッケージを提供していますが、提供されるパッケージやビルドポリシーが異なるために、若干動作に違いが発生することがあります。本書ではPostgreSQLコミュニティが提供するパッケージを利用する前提で解説しています。

- https://www.postgresql.org/download/

(rhel9)

```
$ sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-9-x86_64/pgdg-redhat-repo-latest.noarch.rpm
```

PostgreSQLパッケージのインストールをします。本書を書いた時点ではPostgreSQL 15位以降のバージョンに対応しています。PostgreSQL 16をインストールする場合は次のように実行します。

```
$ sudo dnf -qy module disable postgresql && sudo dnf install -y postgresql16-server postgresql16-devel
```

PostgreSQLの初期化とサービスの起動を設定します。

```
$ sudo /usr/pgsql-16/bin/postgresql-16-setup initdb && sudo systemctl enable --now postgresql-16
```

SELinux周りを実行します。実際利用するパスに置き換えて実行してください。

```
$ sudo chown postgres:postgres -R /opt/nvme && sudo chcon -R system_u:object_r:postgresql_db_t:s0 /opt/nvme/
```

## HereroDBリポジトリーの追加
HeteroDB Software Distribution CenterからリポジトリーRPMパッケージをダウンロードして、`heterodb-extra`パッケージをインストールします。

- [HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/)

(rhel9)

```
$ sudo dnf install -y https://heterodb.github.io/swdc/yum/rhel9-noarch/heterodb-swdc-1.3-1.el9.noarch.rpm
$ sudo dnf install -y heterodb-extra
```

入手したライセンスを割り当てます。

```
$ sudo sh -c "cat heterodb.license > /etc/heterodb.license"
$ sudo systemctl restart postgresql-16
```


## PG-Strom のインストール
ここまで準備ができたら、[インストールガイド](https://heterodb.github.io/pg-strom/ja/install/)に従って、PG-Stromのインストール、設定を行います。


## その他の設定
筆者がよく行う設定をまとめました。必要に応じて設定します。

### PostgreSQL + PG-Strom環境を外部クライアントから利用する
外部クライアントから利用するPG-Strom環境へアクセスしたい場合は、PostgreSQLの設定の変更が必要です。
次のような方法で対応してください。

リッスンアドレスの設定を変更します。

```
sudo su - postgres
vi /var/lib/pgsql/16/data/postgresql.conf
...
listen_addresses = '*'
```

リモートアクセスを許可するユーザーを作成します。ユーザーにはアクセスに適切なロールを設定します。以下はかなり緩い設定です。設定内容の詳細については設定ファイルのコメントを確認してください。

```
$ createuser -d -r -s -P pguser01
Enter password for new role:  <パスワードを設定>
＄vi /var/lib/pgsql/16/data/pg_hba.conf
（追記）
host    all    pguser01    127.0.0.1/32     scram-sha-256
host    all    pguser01    172.16.0.0/16    scram-sha-256
host    all    pguser01    172.17.0.0/16    scram-sha-256
$ exit
```

サービスを再起動します。

```
$ sudo systemctl restart postgresql-16.service
$ journalctl -u postgresql-16
```

ファイアウォールポート開放
リモートアクセスを許可するにはポート開放が必要です。Firewalldが利用されている環境では以下のいずれかの方法で必要なポートを解放します。設定しているポートが5432ではない場合は、適切なポートを指定してください。


```
$ sudo firewall-cmd --add-service=postgresql --permanent 

or

$ sudo firewall-cmd --add-port=5432/tcp --permanent 

then

$ sudo firewall-cmd --reload
```


### GDS領域にデータベースを作成する
PG-Stromにとって、GPUDirect Storageは重要なコンポーネントの一つです。
GPUDirect Storageの環境が整っても、PostgreSQLのデータがGPUDirect Storageが有効なストレージにないと、十分な性能を出すことができません。次のような方法でデータベースとテーブルを作成した上で、データを取り扱ってください。

```
CREATE TABLESPACE nvme LOCATION '/opt/nvme';
CREATE DATABASE testdb TABLESPACE nvme;
```

テーブル空間についての詳細は、アップストリームのドキュメントを確認してください。

- https://www.postgresql.jp/document/16/html/manage-ag-tablespaces.html

