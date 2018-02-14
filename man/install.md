@ja:# インストール
@en:# Install


@ja:## チェックリスト
@en:## Checklist

@ja{
- **ハードウェア**
    - CUDA ToolkitのサポートするLinuxオペレーティングシステムを動作可能な x86_64 アーキテクチャのハードウェアが必要です。 
    - CPU、ストレージ、およびネットワークデバイスには特別な要件はありませんが、[note002:HW Validation List](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List)はハードウェア選定の上で参考になるかもしれません。
    - SSD-to-GPUダイレクトSQL実行を利用するにはNVMe規格に対応したSSDが必要で、GPUと同一のPCIe Root Complex配下に接続されている必要があります。
- **GPUデバイス**
    - PG-Stromを実行するには少なくとも一個のGPUデバイスがシステム上に必要です。これらはCUDA Toolkitでサポートされており、computing capability が6.0以降のモデル（Pascal世代以降）である必要があります。
    - [note001:GPU Availability Matrix](https://github.com/heterodb/pg-strom/wiki/001:-GPU-Availability-Matrix)により詳細な情報が記載されています。SSD-to-GPUダイレクトSQL実行の対応状況に関してもこちらを参照してください。
- **Operating System**
    - PG-Stromの実行には、CUDA Toolkitによりサポートされているx86_64アーキテクチャ向けのLinux OSが必要です。推奨環境はRed Hat Enterprise LinuxまたはCentOSのバージョン7.xシリーズです。
    - SSD-to-GPUダイレクトSQL実行を利用するには、Red Hat Enterprise Linux または CentOS のバージョン7.3以降が必要です。
- **PostgreSQL**
    - PG-Stromの実行にはPostgreSQLバージョン9.6以降が必要です。これは、Custom ScanインターフェースがCPU並列実行やGROUP BYに対応するため刷新され、拡張モジュールが提供するカスタム実行計画を自然な形で統合できるようになったためです。
- **CUDA Toolkit**
    - PG-Stromの実行にはCUDA Toolkit バージョン9.1以降が必要です。
    - PG-Stromが提供する半精度浮動小数点（`float2`）型は、内部的にCUDA Cのhalf_t型を使用しており、古いCUDA Toolkitではこれをビルドできないためです。
}
@en{
- **Server Hardware**
    - It requires generic x86_64 hardware that can run Linux operating system supported by CUDA Toolkit. We have no special requirement for CPU, storage and network devices.
    - [note002:HW Validation List](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List) may help you to choose the hardware.
    - SSD-to-GPU Direct SQL Execution needs SSD devices which support NVMe specification, and to be installed under the same PCIe Root Complex where GPU is located on.
- **GPU Device**
    - PG-Strom requires at least one GPU device on the system, which is supported by CUDA Toolkit, has computing capability 6.0 (Pascal generation) or later;
    - [note001:GPU Availability Matrix](https://github.com/heterodb/pg-strom/wiki/001:-GPU-Availability-Matrix) shows more detailed information. Check this list for the support status of SSD-to-GPU Direct SQL Execution.
- **Operating System**
    - PG-Strom requires Linux operating system for x86_64 architecture, and its distribution supported by CUDA Toolkit. Our recommendation is Red Hat Enterprise Linux or CentOS version 7.x series.    - SSD-to-GPU Direct SQL Execution needs Red Hat Enterprise Linux or CentOS version 7.3 or later.
- **PostgreSQL**
    - PG-Strom requires PostgreSQL version 9.6 or later. PostgreSQL v9.6 renew the custom-scan interface for CPU-parallel execution or `GROUP BY` planning, thus, it allows cooperation of custom-plans provides by extension modules.
- **CUDA Toolkit**
    - PG-Strom requires CUDA Toolkit version 9.1 or later.
    - PG-Strom provides half-precision floating point type (`float2`), and it internally use `half_t` type of CUDA C, so we cannot build it with older CUDA Toolkit.
}

@ja:## OSのインストール
@en:## OS Installation

@ja{
CUDA ToolkitのサポートするLinuxディストリビューションを選択し、個々のディストリビューションのインストールプロセスに従ってインストール作業を行ってください。 CUDA ToolkitのサポートするLinuxディストリビューションは、NVIDIA DEVELOPER ZONEにおいて紹介されています。
}
@ja{
Red Hat Enterprise Linux 7.x系列、またはCentOS 7.x系列の場合、「...」「....」を選択し、以下のパッケージを追加してください。

- あ
- と
- で
- 確
- 認
}

@ja:### OSインストール後の設定
@en:### Post OS Installation Configuration

@ja{
システムへのOSのインストール後、後のステップでGPUドライバとNVMe-Stromドライバをインストールするために、いくつかの追加設定が必要です。
}

@ja:#### DKMSのインストール
@en:#### DKMS Installation

@ja{
DKMS (Dynamic Kernel Module Support) は、動作中のLinuxカーネル向けのLinuxカーネルモジュールを必要に応じてビルドするためのフレームワークで、NVIDIAのドライバも対応しています。Linuxカーネルのバージョンアップに追従してカーネルモジュールも更新されるため、DKMSのセットアップは推奨です。

DKMSパッケージはEPEL (Extra Packages for Enterprise Linux) の一部として配布されています。ですので、CentOSのパブリックFTPサイトから `epel-release-<distribution version>.noarch.rpm` をダウンロードし、これをインストールしてください。 いったん epel-release パッケージがインストールされると、EPELリポジトリから非標準のパッケージを入手するためのyumシステムへの設定が追加されます。

- Fedora Project Public FTP Site
    - [https://dl.fedoraproject.org/pub/epel/7/x86_64/](https://dl.fedoraproject.org/pub/epel/7/x86_64/)

DKMSに加えて、nvidiaカーネルモジュールをビルドするには以下のパッケージが必要です。以降のステップに進む前に、これらをインストールしてください。

- kernel-devel
- kernel-headers
- kernel-tools
- kernel-tools-libs
}

@ja:#### nouveauドライバの無効化
@en:#### Disables nouveau driver

@ja{
CUDA Toolkitが動作するためにはNVIDIAの提供する`nvidia`ドライバが必要です。しかし、オープンソースの互換ドライバである`nouveau`と競合してしまうため、OSインストーラが`nouveau`ドライバをインストールしている場合には、CUDA Toolkitのインストールに先立って`nouveau`ドライバを無効化する必要があります。

`nouveau`ドライバがロードされないよう、以下の設定を`/etc/modprobe.d/blacklist-nouveau.conf`に追加します。
}

```
blacklist nouveau
options nouveau modeset=0
```

@ja{
その後、カーネルブートイメージを更新するために `dracut -f /boot/initramfs-$(uname -r).img $(uname -r)` を実行してください。

最後に、この設定を反映させるために`shutdown -r now`を実行してシステムを再起動します。`lsmod`の出力に`nouveau`が含まれていなければ正しく設定されています。
}
```
$ lsmod | grep nouveau
$
```

@ja:## CUDA Toolkitのインストール
@en:## CUDA Toolkit Installation

@ja{
本節ではCUDA Toolkitのインストールについて説明します。 既に対応バージョンのCUDA Toolkitをインストール済みであれば、本節の内容は読み飛ばして構いません

NVIDIAはCUDA Toolkitのインストールに２通りの方法を提供しています。一つは自己実行型アーカイブ（runfileと呼ばれる）によるもの。もう一つはRPMパッケージによるものです。
ソフトウェアの更新が容易である事から、後者のRPMパッケージによるインストールが推奨です。

CUDA Toolkitのインストール用パッケージはNVIDIA DEVELOPER ZONEからダウンロードする事ができます。 適切なOS、アーキテクチャ、ディストリビューション、バージョンを指定し、『rpm(network)』版を選択してください。
}

![CUDA Toolkit download](./img/cuda-download.png)

@ja{
『rpm(network)』パッケージにはCUDA Toolkitを配布するyumリポジトリの定義情報が含まれているだけです。これは OSのインストール においてシステムにEPELリポジトリの定義を追加したのと同様の方法です。 したがって、cudaリポジトリを登録した後、関連したRPMパッケージをネットワークインストールする必要があります。 下記のコマンドを実行してください。
}

```
$ sudo rpm -i cuda-repo-<distribution>-<version>.x86_64.rpm
$ sudo yum clean all
$ sudo yum install cuda
```

@ja{
正常にインストールが完了すると、`/usr/local/cuda`配下にCUDA Toolkitが導入されています。
}

```
$ ls /usr/local/cuda
bin     include  libnsight         nvml       samples  tools
doc     jre      libnvvp           nvvm       share    version.txt
extras  lib64    nsightee_plugins  pkgconfig  src
```

!!! hint
   `nvidia`ドライバのビルドやインストールに失敗する場合は、 OSのインストール において必要なパッケージが導入されている事、および`nouveau`ドライバが無効化されている事を確認してください。特に`dracut`スクリプトを実行するまでは`nouveau`ドライバの無効化設定が起動イメージに反映されない事に留意してください。

@ja:### インストール後の設定
@en:### Post Installation Configuration

@ja{
少なくともCUDA-9.1において、PG-Stromの使用するNVRTCライブラリのリンケージに関して問題がある事が分かっており、これを回避するため、ワークアラウンドとして以下の設定を行ってください。
}

```
# echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda-lib64.conf
# ldconfig
```

@ja{
CUDAおよび`nvidia`ドライバのインストールが完了したら、GPUが正しく認識されている事を確認してください。`nvidia-smi`コマンドを実行すると、以下の出力例のように、システムに搭載されているGPUの情報が表示されます。
}

```
$ nvidia-smi
Wed Feb 14 09:43:48 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 387.26                 Driver Version: 387.26                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:02:00.0 Off |                    0 |
| N/A   41C    P0    37W / 250W |      0MiB / 16152MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

@ja:## PostgreSQLのインストール
@en:## PostgreSQL Installation

@ja{
本節ではRPMによるPostgreSQLのインストールについて紹介します。
ソースからのインストールに関しては既にドキュメントが数多く存在し、`./configure`スクリプトのオプションが多岐にわたる事から、ここでは紹介しません。
}

@ja{
Linuxディストリビューションの配布するパッケージにもPostgreSQLは含まれていますが、必ずしも最新ではなく、PG-Stromの対応バージョンよりも古いものである事が多々あります。例えば、Red Hat Enterprise Linux 7.xやCentOS 7.xで配布されているPostgreSQLはv9.2.xですが、これはPostgreSQLコミュニティとして既にEOLとなっているバージョンです。

PostgreSQL Global Development Groupは、最新のPostgreSQLおよび関連ソフトウェアの配布のためにyumリポジトリを提供しています。
EPELの設定のように、yumリポジトリの設定を行うだけの小さなパッケージをインストールし、その後、PostgreSQLやその他のソフトウェアをインストールします。
}

@ja{
yumリポジトリ定義の一覧は [http://yum.postgresql.org/repopackages.php](http://yum.postgresql.org/repopackages.php) です。

PostgreSQLメジャーバージョンとLinuxディストリビューションごとに多くのリポジトリ定義がありますが、あなたのLinuxディストリビューション向けのPostgreSQL 9.6以降のものを選択する必要があります。

以下のように、yumリポジトリの定義をインストールし、次いで、PostgreSQLパッケージをインストールすれば完了です。 PostgreSQL v10を使用する場合、PG-Stromのインストールには以下のパッケージが必要です。

- postgresql10-devel
- postgresql10-server
}

```
$ sudo rpm -ivh pgdg-redhat10-10-2.noarch.rpm
$ sudo yum install -y postgresql10-server postgresql10-devel
            :
            :
================================================================================
 Package                  Arch        Version                 Repository   Size
================================================================================
Installing:
 postgresql10-devel       x86_64      10.2-1PGDG.rhel7        pgdg10      2.0 M
 postgresql10-server      x86_64      10.2-1PGDG.rhel7        pgdg10      4.4 M
Installing for dependencies:
 postgresql10             x86_64      10.2-1PGDG.rhel7        pgdg10      1.5 M
 postgresql10-libs        x86_64      10.2-1PGDG.rhel7        pgdg10      354 k

Transaction Summary
================================================================================
Install  2 Packages (+2 Dependent packages)

Total download size: 8.3 M
Installed size: 35 M
            :
            :
Installed:
  postgresql10-devel.x86_64 0:10.2-1PGDG.rhel7
  postgresql10-server.x86_64 0:10.2-1PGDG.rhel7

Dependency Installed:
  postgresql10.x86_64 0:10.2-1PGDG.rhel7
  postgresql10-libs.x86_64 0:10.2-1PGDG.rhel7

Complete!
```

@ja:## PG-Stromのインストール
@en:## PG-Strom Installation

@ja{
本節ではPG-Stromのインストール方法について説明します。
推奨はRPMによるインストールですが、開発者向けにソースコードからのビルド方法についても紹介します。
}
@{
This section introduces the steps to install PG-Strom.
We recommend RPM installation, however, also mention about the steps to build PG-Strom from the source code.
}


@ja:### RPMによるインストール
@en:### RPM Installation

@ja{
PG-Stromおよび関連パッケージは[HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/)より配布されています。このサイトはyumリポジトリとしても機能するよう作成されており、`heterodb-swdc`パッケージをインストールする事でyumリポジトリのエントリが追加されます。

}

```
$ wget https://heterodb.github.io/swdc/yum/rhel7-noarch/heterodb-swdc-1.0-1.el7.noarch.rpm
$ sudo rpm -ivh heterodb-swdc-1.0-1.el7.noarch.rpm
Preparing...                          ################################# [100%]
Updating / installing...
   1:heterodb-swdc-1.0-1.el7          ################################# [100%]

```

@ja{
続いて、PG-StromのRPMパッケージをインストールします。
対象となるPostgreSQLバージョン毎に別個のRPMパッケージが準備されており、PostgreSQL v9.6用であれば`pg-strom-PG96`パッケージを、PostgreSQL v10用であれば`pg-strom-PG10`パッケージをインストールします。
}

```

```

@ja{
以上でパッケージのインストールは完了です。
}


@ja:### ソースからのビルド
@en:### Build from the source

@ja{
開発者向けに、ソースコードからPG-Stromをビルドする方法についても紹介します。
}

@ja:### ソースコードの入手
@en:### Getting the source code
@ja{
RPMパッケージと同様に、ソースコードのtarballを[HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/)から入手する事ができます。

ただ、tarballのリリースにはある程度のタイムラグが生じてしまうため、最新の開発版を使いたい場合には[PG-StromのGitHubリポジトリ](https://github.com/heterodb/pg-strom)の`master`ブランチをチェックアウトする方法の方が好まれるかもしれません。
}

```
$ git clone https://github.com/heterodb/pg-strom.git
Cloning into 'pg-strom'...
remote: Counting objects: 13797, done.
remote: Compressing objects: 100% (215/215), done.
remote: Total 13797 (delta 208), reused 339 (delta 167), pack-reused 13400
Receiving objects: 100% (13797/13797), 11.81 MiB | 1.76 MiB/s, done.
Resolving deltas: 100% (10504/10504), done.
```

@ja:### PG-Stromのビルド
@en:### Building the PG-Strom
@ja{
PG-Stromをビルドする時のコンフィグは、インストール先のPostgreSQLと厳密に一致していなければいけません。例えば、同じ構造体がビルド時のコンフィグによりPostgreSQLとPG-Stromで異なったレイアウトを持ってしまったとすれば、非常に発見の難しいバグを生み出してしまうかもしれません。 したがって、（一貫性のない状態を避けるため）PG-Stromは独自にconfigureスクリプトを走らせたりはせず、`pg_config`を使ってPostgreSQLのビルド時設定を参照します。

`pg_config`にパスが通っており、それがインストール先のPostgreSQLのものであれば、そのまま`make`、`make install`を実行します。
直接パスが通っていない場合は、`make`コマンドに`PG_CONFIG=...`パラメータを与え、`pg_config`のフルパスを渡します。
}

```
$ cd pg-strom
$ make PG_CONFIG=/usr/pgsql-10/bin/pg_config
$ sudo make install PG_CONFIG=/usr/pgsql-10/bin/pg_config
```


@ja:## インストール後の設定
@en:## Post Installation Setup



