@ja{
本章ではPG-Stromのインストール手順について説明します。
}
@en{
This chapter introduces the steps to install PG-Strom.
}

@ja:# チェックリスト
@en:# Checklist

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
    - PG-Stromの実行にはCUDA Toolkit バージョン9.2以降が必要です。
    - PG-Stromが内部的に利用しているAPIの中には、これ以前のバージョンでは提供されていないものが含まれています。
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
    - PG-Strom requires CUDA Toolkit version 9.2 or later.
    - Some of CUDA Driver APIs used by PG-Strom internally are not included in the former versions.
}

@ja:# OSのインストール
@en:# OS Installation

@ja{
CUDA ToolkitのサポートするLinuxディストリビューションを選択し、個々のディストリビューションのインストールプロセスに従ってインストール作業を行ってください。 CUDA ToolkitのサポートするLinuxディストリビューションは、[NVIDIA DEVELOPER ZONE](https://developer.nvidia.com/)において紹介されています。
}
@en{
Choose a Linux distribution which is supported by CUDA Toolkit, then install the system according to the installation process of the distribution. [NVIDIA DEVELOPER ZONE](https://developer.nvidia.com/) introduces the list of Linux distributions which are supported by CUDA Toolkit.
}
@ja{
Red Hat Enterprise Linux 7.x系列、またはCentOS 7.x系列の場合、ベース環境として「最小限のインストール」を選択し、さらに以下のアドオンを選択してください。

- デバッグツール
- 開発ツール
}
@en{
In case of Red Hat Enterprise Linux 7.x or CentOS 7.x series, choose "Minimal installation" as base environment, and also check the following add-ons.

- Debugging Tools
- Development Tools
}


@ja:## OSインストール後の設定
@en:## Post OS Installation Configuration

@ja{
システムへのOSのインストール後、後のステップでGPUドライバとNVMe-Stromドライバをインストールするために、いくつかの追加設定が必要です。
}
@en{
Next to the OS installation, a few additionsl configurations are required to install GPU-drivers and NVMe-Strom driver on the later steps.
}

@ja:### EPELリポジトリの設定
@en:### Setup EPEL Repository

@ja{
PG-Stromの実行に必要なソフトウェアモジュールのいくつかは、EPEL(Extra Packages for Enterprise Linux)の一部として配布されています。
これらのソフトウェアを入手するためにEPELパッケージ群のリポジトリ定義をyumシステムに追加する必要があります。
}
@en{
Several software modules required by PG-Strom are distributed as a part of EPEL (Extra Packages for Enterprise Linux).
You need to add a repository definition of EPEL packages for yum system to obtain these software.
}
@ja{
EPELリポジトリから入手するパッケージの一つがDKMS(Dynamic Kernel Module Support)です。これは動作中のLinuxカーネルに適合したLinuxカーネルモジュールをオンデマンドでビルドするためのフレームワークで、NVIDIAのGPUデバイスドライバや、SSD-to-GPUダイレクトSQL実行をサポートするカーネルモジュール(nvme_strom)が使用しています。
Linuxカーネルモジュールは、Linuxカーネルのバージョンアップに追従して再ビルドが必要であるため、DKMSなしでのシステム運用は現実的ではありません。
}
@en{
One of the package we will get from EPEL repository is DKMS (Dynamic Kernel Module Support). It is a framework to build Linux kernel module for the running Linux kernel on demand; used for NVIDIA's GPU driver or NVMe-Strom which is a kernel module to support SSD-to-GPU Direct SQL Execution.
}
@ja{
EPELリポジトリの定義は`epel-release`パッケージにより提供されます。
これはFedora ProjectのパブリックFTPサイトから入手する事が可能で、`epel-release-<distribution version>.noarch.rpm`をダウンロードし、これをインストールしてください。 
`epel-release`パッケージがインストールされると、EPELリポジトリからソフトウェアを入手するための設定がyumシステムへ追加されます。
}
@en{
`epel-release` package provides the repository definition of EPEL.
You can obtain this package from the public FTP site of Fedora Project. Downloads the `epel-release-<distribution version>.noarch.rpm`, and install the package.
Once `epel-release` package gets installed, yum system configuration is updated to get software from the EPEL repository.
}
- Fedora Project Public FTP Site
    - [https://dl.fedoraproject.org/pub/epel/7/x86_64/](https://dl.fedoraproject.org/pub/epel/7/x86_64/)
@ja{
!!! Tip
    上記URLから`Packages`→`e`へとディレクトリ階層を下ります。
}
@en{
!!! Tip
    Walk down the directory: `Packages` --> `e`, from the above URL.
}

@ja{
以下のようにepel-releaseパッケージをインストールします。
}
@en{
Install the `epel-release` package as follows.
}
```
$ sudo yum install https://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-7-11.noarch.rpm
          :
================================================================================
 Package           Arch        Version     Repository                      Size
================================================================================
Installing:
 epel-release      noarch      7-11        /epel-release-7-11.noarch       24 k

Transaction Summary
================================================================================
Install  1 Package
          :
Installed:
  epel-release.noarch 0:7-11

Complete!
```

@ja:### HeteroDB-SWDCのインストール
@en:### HeteroDB-SWDC Installation

@ja{
PG-Stromほか関連パッケージは[HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/)から配布されています。
これらのソフトウェアを入手するために、HeteroDB-SWDCのリポジトリ定義をyumシステムに追加する必要があります。
}
@en{
PG-Strom and related packages are distributed from [HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/).
You need to add a repository definition of HeteroDB-SWDC for you system to obtain these software.
}

@ja{
HeteroDB-SWDCリポジトリの定義はheterodb-swdcパッケージにより提供されます。
Webブラウザなどで[HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/)へアクセスし、ページの先頭にリンクの記載されている`heterodb-swdc-1.0-1.el7.noarch.rpm`をダウンロードしてインストールしてください。
heterodb-swdcパッケージがインストールされると、HeteroDB-SWDCからソフトウェアを入手するためのyumシステムへの設定が追加されます。
}
@en{
`heterodb-swdc` package provides the repository definition of HeteroDB-SWDC.
Access to the [HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/) using Web browser, download the `heterodb-swdc-1.0-1.el7.noarch.rpm` on top of the file list, then install this package.
Once heterodb-swdc package gets installed, yum system configuration is updated to get software from the HeteroDB-SWDC repository.
}
@ja{
以下のようにheterodb-swdcパッケージをインストールします。
}
@en{
Install the `heterodb-swdc` package as follows.
}

```
$ sudo yum install https://heterodb.github.io/swdc/yum/rhel7-x86_64/heterodb-swdc-1.0-1.el7.noarch.rpm
          :
================================================================================
 Package         Arch     Version       Repository                         Size
================================================================================
Installing:
 heterodb-swdc   noarch   1.0-1.el7     /heterodb-swdc-1.0-1.el7.noarch   2.4 k

Transaction Summary
================================================================================
Install  1 Package
          :
Installed:
  heterodb-swdc.noarch 0:1.0-1.el7

Complete!
```


@ja:# CUDA Toolkitのインストール
@en:# CUDA Toolkit Installation

@ja{
本節ではCUDA Toolkitのインストールについて説明します。 既に最新のCUDA Toolkitをインストール済みであれば、本節の内容は読み飛ばして構いません
}
@en{
This section introduces the installation of CUDA Toolkit. If you already installed the latest CUDA Toolkit, you can skip this section.
}
@ja{
NVIDIAはCUDA Toolkitのインストールに２通りの方法を提供しています。一つは自己実行型アーカイブ（runfileと呼ばれる）によるもの。もう一つはRPMパッケージによるものです。
ソフトウェアの更新が容易である事から、後者のRPMパッケージによるインストールが推奨です。
}
@en{
NVIDIA offers two approach to install CUDA Toolkit; one is by self-extracting archive (called runfile), and the other is by RPM packages.
We recommend RPM installation because it allows simple software updates.
}
@ja{
CUDA Toolkitのインストール用パッケージはNVIDIA DEVELOPER ZONEからダウンロードする事ができます。 適切なOS、アーキテクチャ、ディストリビューション、バージョンを指定し、『rpm(network)』版を選択してください。
}
@en{
You can download the installation package for CUDA Toolkit from NVIDIA DEVELOPER ZONE. Choose your OS, architecture, distribution and version, then choose "rpm(network)" edition.
}

![CUDA Toolkit download](./img/cuda-download.png)

@ja{
『rpm(network)』パッケージにはCUDA Toolkitを配布するyumリポジトリの定義情報が含まれているだけです。これは OSのインストール においてシステムにEPELリポジトリの定義を追加したのと同様の方法です。 したがって、cudaリポジトリを登録した後、関連したRPMパッケージをネットワークインストールする必要があります。 下記のコマンドを実行してください。
}
@en{
The "rpm(network)" edition contains only yum repositoty definition to distribute CUDA Toolkit. It is similar to the EPEL repository definition at the OS installation.
So, you needs to installa the related RPM packages over network after the resistoration of CUDA repository. Run the following command.
}

```
$ sudo rpm -i cuda-repo-<distribution>-<version>.x86_64.rpm
$ sudo yum clean all
$ sudo yum install cuda --enablerepo=rhel-7-server-e4s-optional-rpms
 or
$ sudo yum install cuda 
```

@ja{
正常にインストールが完了すると、`/usr/local/cuda`配下にCUDA Toolkitが導入されています。
}

@en{
Once installation completed successfully, CUDA Toolkit is deployed at `/usr/local/cuda`.
}

@ja{
!!! Tip
    RHEL7の場合、CUDA Toolkitのインストールに必要な`vulkan-filesystem`パッケージを配布する`rhel-7-server-e4s-optional-rpms`リポジトリは、デフォルトで有効化されていません。CUDA Toolkitをインストールする際には、`/etc/yum.repos.d/redhat.repo`を編集して当該リポジトリを有効化するか、yumコマンドの`--enablerepo`オプションを用いて当該リポジトリを一時的に有効化してください。
}
@en{
!!! Tip
    RHEL7 does not enable `rhel-7-server-e4s-optional-rpms` repository in the default. It distributes `vulkan-filesystem` packaged required by CUDA Toolkit installation. When you kick installation of CUDA Toolkit, edit `/etc/yum.repos.d/redhat.repo` to enable the repository, or use `--enablerepo` option of yum command to resolve dependency.
}

```
$ ls /usr/local/cuda
bin     include  libnsight         nvml       samples  tools
doc     jre      libnvvp           nvvm       share    version.txt
extras  lib64    nsightee_plugins  pkgconfig  src
```

@ja{
インストールが完了したら、GPUが正しく認識されている事を確認してください。`nvidia-smi`コマンドを実行すると、以下の出力例のように、システムに搭載されているGPUの情報が表示されます。
}
@en{
Once installation gets completed, ensure the system recognizes the GPU devices correctly.
`nvidia-smi` command shows GPU information installed on your system, as follows.
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

@ja{
!!! Tip
    nvidiaドライバと競合するnouveauドライバがロードされている場合、直ちにnvidiaドライバをロードする事ができません。
    この場合は、nouveauドライバの無効化設定を行った上でシステムを一度再起動してください。
    runfileによるインストールの場合、CUDA Toolkitのインストーラがnouveauドライバの無効化設定も行います。RPMによるインストールの場合は、以下の設定を行ってください。
}
@en{
!!! Tip
    If nouveau driver which conflicts to nvidia driver is loaded, system cannot load the nvidia driver immediately.
    In this case, reboot the operating system after a configuration to disable the nouveau driver.
    If CUDA Toolkit is installed by the runfile installer, it also disables the nouveau driver. Elsewhere, in case of RPM installation, do the following configuration.
}

@ja{
nouveauドライバを無効化するには、以下の設定を`/etc/modprobe.d/disable-nouveau.conf`という名前で保存し、`dracut`コマンドを実行してLinux kernelのブートイメージに反映します。
}
@en{
To disable the nouveau driver, put the following configuration onto `/etc/modprobe.d/disable-nouveau.conf`, then run `dracut` command to apply them on the boot image of Linux kernel.
}
```
# cat > /etc/modprobe.d/disable-nouveau.conf <<EOF
blacklist nouveau
options nouveau modeset=0
EOF
# dracut -f
```

@ja:# PostgreSQLのインストール
@en:# PostgreSQL Installation

@ja{
本節ではRPMによるPostgreSQLのインストールについて紹介します。
ソースからのインストールに関しては既にドキュメントが数多く存在し、`./configure`スクリプトのオプションが多岐にわたる事から、ここでは紹介しません。
}
@en{
This section introduces PostgreSQL installation with RPM.
We don't introduce the installation steps from the source because there are many documents for this approach, and there are also various options for the `./configure` script.
}

@ja{
Linuxディストリビューションの配布するパッケージにもPostgreSQLは含まれていますが、必ずしも最新ではなく、PG-Stromの対応バージョンよりも古いものである事が多々あります。例えば、Red Hat Enterprise Linux 7.xやCentOS 7.xで配布されているPostgreSQLはv9.2.xですが、これはPostgreSQLコミュニティとして既にEOLとなっているバージョンです。
}
@en{
PostgreSQL is also distributed in the packages of Linux distributions, however, it is not the latest one, and often older than the version which supports PG-Strom. For example, Red Hat Enterprise Linux 7.x or CentOS 7.x distributes PostgreSQL v9.2.x series. This version had been EOL by the PostgreSQL community.
}
@ja{
PostgreSQL Global Development Groupは、最新のPostgreSQLおよび関連ソフトウェアの配布のためにyumリポジトリを提供しています。
EPELの設定のように、yumリポジトリの設定を行うだけの小さなパッケージをインストールし、その後、PostgreSQLやその他のソフトウェアをインストールします。
}
@en{
PostgreSQL Global Development Group provides yum repository to distribute the latest PostgreSQL and related packages.
Like the configuration of EPEL, you can install a small package to set up yum repository, then install PostgreSQL and related software.
}

@ja{
yumリポジトリ定義の一覧は [http://yum.postgresql.org/repopackages.php](http://yum.postgresql.org/repopackages.php) です。

PostgreSQLメジャーバージョンとLinuxディストリビューションごとに多くのリポジトリ定義がありますが、あなたのLinuxディストリビューション向けのPostgreSQL 9.6以降のものを選択する必要があります。
}
@en{
Here is the list of yum repository definition: [http://yum.postgresql.org/repopackages.php](http://yum.postgresql.org/repopackages.php).

Repository definitions are per PostgreSQL major version and Linux distribution. You need to choose the one for your Linux distribution, and for PostgreSQL v9.6 or later.
}

@ja{
以下のように、yumリポジトリの定義をインストールし、次いで、PostgreSQLパッケージをインストールすれば完了です。 PostgreSQL v10を使用する場合、PG-Stromのインストールには以下のパッケージが必要です。
}
@en{
All you need to install are yum repository definition, and PostgreSQL packages. If you choose PostgreSQL v10, the pakages below are required to install PG-Strom.
}
- postgresql10-devel
- postgresql10-server
```
$ sudo yum install -y https://download.postgresql.org/pub/repos/yum/10/redhat/rhel-7-x86_64/pgdg-redhat10-10-2.noarch.rpm
$ sudo yum install -y postgresql10-server postgresql10-devel
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
          :
Installed:
  postgresql10-devel.x86_64 0:10.2-1PGDG.rhel7
  postgresql10-server.x86_64 0:10.2-1PGDG.rhel7

Dependency Installed:
  postgresql10.x86_64 0:10.2-1PGDG.rhel7
  postgresql10-libs.x86_64 0:10.2-1PGDG.rhel7

Complete!
```

@ja{
PostgreSQL Global Development Groupの提供するRPMパッケージは`/usr/pgsql-<version>`という少々変則的なディレクトリにソフトウェアをインストールするため、`psql`等の各種コマンドを実行する際にはパスが通っているかどうか注意する必要があります。

`postgresql-alternatives`パッケージをインストールしておくと、各種コマンドへのシンボリックリンクを`/usr/local/bin`以下に作成するため各種オペレーションが便利です。また、複数バージョンのPostgreSQLをインストールした場合でも、`alternatives`コマンドによってターゲットとなるPostgreSQLバージョンを切り替える事が可能です。
}
@en{
The RPM packages provided by PostgreSQL Global Development Group installs software under the `/usr/pgsql-<version>` directory, so you may pay attention whether the PATH environment variable is configured appropriately.

`postgresql-alternative` package set up symbolic links to the related commands under `/usr/local/bin`, so allows to simplify the operations. Also, it enables to switch target version using `alternatives` command even if multiple version of PostgreSQL.
}

```
$ sudo yum install postgresql-alternatives
          :
Resolving Dependencies
--> Running transaction check
---> Package postgresql-alternatives.noarch 0:1.0-1.el7 will be installed
--> Finished Dependency Resolution

Dependencies Resolved
          :
================================================================================
 Package                      Arch        Version           Repository     Size
================================================================================
Installing:
 postgresql-alternatives      noarch      1.0-1.el7         heterodb      9.2 k

Transaction Summary
================================================================================
          :
Installed:
  postgresql-alternatives.noarch 0:1.0-1.el7

Complete!
```

@ja:# PG-Stromのインストール
@en:# PG-Strom Installation

@ja{
本節ではPG-Stromのインストール方法について説明します。
推奨はRPMによるインストールですが、開発者向けにソースコードからのビルド方法についても紹介します。
}
@{
This section introduces the steps to install PG-Strom.
We recommend RPM installation, however, also mention about the steps to build PG-Strom from the source code.
}

@ja:## RPMによるインストール
@en:## RPM Installation

@ja{
PG-Stromおよび関連パッケージは[HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/)より配布されています。
既にyumシステムへリポジトリを追加済みであれば、それほど作業は多くありません。
}
@en{
PG-Strom and related packages are distributed from [HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/).
If you repository definition has been added, not many tasks are needed.
}
@ja{
基盤となるPostgreSQLのバージョンごとに別個のPG-StromのRPMパッケージが準備されており、PostgreSQL v9.6用であれば`pg_strom-PG96`パッケージを、PostgreSQL v10用であれば`pg_strom-PG10`パッケージをインストールします。
}
@en{
We provide individual RPM packages of PG-Strom for each base PostgreSQL version. `pg_strom-PG96` package is built for PostgreSQL 9.6, and `pg_strom-PG10` is also built for PostgreSQL v10.

}

```
$ sudo yum install pg_strom-PG10
          :
================================================================================
 Package              Arch          Version               Repository       Size
================================================================================
Installing:
 pg_strom-PG10        x86_64        1.9-180301.el7        heterodb        320 k

Transaction Summary
================================================================================
          :
Installed:
  pg_strom-PG10.x86_64 0:1.9-180301.el7

Complete!
```

@ja{
以上でパッケージのインストールは完了です。
}
@en{
That's all for package installation.
}

@ja:## ソースからのインストール
@en:## Installation from the source

@ja{
開発者向けに、ソースコードからPG-Stromをビルドする方法についても紹介します。
}
@en{
For developers, we also introduces the steps to build and install PG-Strom from the source code.
}

@ja:### ソースコードの入手
@en:### Getting the source code
@ja{
RPMパッケージと同様に、ソースコードのtarballを[HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/)から入手する事ができます。

ただ、tarballのリリースにはある程度のタイムラグが生じてしまうため、最新の開発版を使いたい場合には[PG-StromのGitHubリポジトリ](https://github.com/heterodb/pg-strom)の`master`ブランチをチェックアウトする方法の方が好まれるかもしれません。
}
@en{
Like RPM packages, you can download tarball of the source code from [HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/).
On the other hands, here is a certain time-lags to release the tarball, it may be preferable to checkout the master branch of [PG-Strom on GitHub](https://github.com/heterodb/pg-strom) to use the latest development branch.
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
@en{
Configuration to build PG-Strom must match to the target PostgreSQL strictly. For example, if a particular `strcut` has inconsistent layout by the configuration at build, it may lead problematic bugs; not easy to find out.
Thus, not to have inconsistency, PG-Strom does not have own configure script, but references the build configuration of PostgreSQL using `pg_config` command.

If PATH environment variable is set to the `pg_config` command of the target PostgreSQL, run `make` and `make install`.
Elsewhere, give `PG_CONFIG=...` parameter on `make` command to tell the full path of the `pg_config` command.
}

```
$ cd pg-strom
$ make PG_CONFIG=/usr/pgsql-10/bin/pg_config
$ sudo make install PG_CONFIG=/usr/pgsql-10/bin/pg_config
```

@ja:## インストール後の設定
@en:## Post Installation Setup

@ja:### データベースクラスタの作成
@en:### Creation of database cluster

@ja{
データベースクラスタの作成が済んでいない場合は、`initdb`コマンドを実行してPostgreSQLの初期データベースを作成します。

RPMインストールにおけるデフォルトのデータベースクラスタのパスは`/var/lib/pgsql/<version number>/data`です。
`postgresql-alternatives`パッケージをインストールしている場合は、PostgreSQLのバージョンに拠らず`/var/lib/pgdata`で参照する事ができます。
}
@en{
Database cluster is not constructed yet, run `initdb` command to set up initial database of PostgreSQL.

The default path of the database cluster on RPM installation is `/var/lib/pgsql/<version number>/data`.
If you install `postgresql-alternatives` package, this default path can be referenced by `/var/lib/pgdata` regardless of the PostgreSQL version.
}
```
$ sudo su - postgres
$ initdb -D /var/lib/pgdata/
The files belonging to this database system will be owned by user "postgres".
This user must also own the server process.

The database cluster will be initialized with locale "en_US.UTF-8".
The default database encoding has accordingly been set to "UTF8".
The default text search configuration will be set to "english".

Data page checksums are disabled.

fixing permissions on existing directory /var/lib/pgdata ... ok
creating subdirectories ... ok
selecting default max_connections ... 100
selecting default shared_buffers ... 128MB
selecting dynamic shared memory implementation ... posix
creating configuration files ... ok
running bootstrap script ... ok
performing post-bootstrap initialization ... ok
syncing data to disk ... ok

WARNING: enabling "trust" authentication for local connections
You can change this by editing pg_hba.conf or using the option -A, or
--auth-local and --auth-host, the next time you run initdb.

Success. You can now start the database server using:

    pg_ctl -D /var/lib/pgdata/ -l logfile start
```

@ja:### postgresql.confの編集
@en:### Setup postgresql.conf

@ja{
続いて、PostgreSQLの設定ファイルである `postgresql.conf` を編集します。

PG-Stromを動作させるためには、最低限、以下のパラメータの設定が必要です。
これ以外のパラメータについても、システムの用途や想定ワークロードを踏まえて検討してください。
}
@en{
Next, edit `postgresql.conf` which is a configuration file of PostgreSQL.
The parameters below should be edited at least to work PG-Strom.
Investigate other parameters according to usage of the system and expected workloads.
}

@ja{
- **shared_preload_libraries**
    - PG-Stromモジュールは`shared_preload_libraries`パラメータによりpostmasterプロセスの起動時にロードされる必要があります。オンデマンドでの拡張モジュールのロードはサポート対象外です。したがって、以下の設定項目は必須です。
    - ```shared_preload_libraries = '$libdir/pg_strom'```
- **max_worker_processes**
    - PG-Stromは数個のバックグラウンドワーカーを内部的に使用します。そのため、デフォルト値である 8 では、それ以外の処理に利用できるバックグラウンドワーカープロセス数があまりにも少なすぎてしまいます。
    - 以下のように、ある程度の余裕を持った値を設定すべきです。
    - ```max_worker_processes = 100```
- **shared_buffers**
    - ワークロードによりますが、`shared_buffers`の初期設定は非常に小さいため、PG-Stromが有効に機能する水準のデータサイズに対しては、ストレージへの読み書きが律速要因となってしまい、GPUの並列計算機能を有効に利用できない可能性があります。
    - 以下のように、ある程度の余裕を持った値を設定すべきです。
    - ```shared_buffers = 10GB```
    - 明らかにメモリサイズよりも大きなデータを処理する必要がある場合は、SSD-to-GPUダイレクトSQL実行の利用を検討してください。
- **work_mem**
    - ワークロードによりますが、`work_mem`の初期設定は非常に小さいため、解析系クエリで最適なクエリ実行計画が選択されない可能性があります。
    - 典型的な例は、ソート処理にオンメモリのクイックソートではなく、ディスクベースのマージソートを選択するといったものです。
    - 以下のように、ある程度の余裕を持った値を設定すべきです。
    - ```work_mem = 1GB```
}
@en{
- **shared_preload_libraries**
    - PG-Strom module must be loaded on startup of the postmaster process by the `shared_preload_libraries`. Unable to load it on demand. Therefore, you must add the configuration below.
    - ```shared_preload_libraries = '$libdir/pg_strom'```
- **max_worker_processes**
    - PG-Strom internally uses several background workers, so the default configuration (= 8) is too small for other usage. So, we recommand to expand the variable for a certain margin.
    - ```max_worker_processes = 100```
- **shared_buffers**
    - Although it depends on the workloads, the initial configuration of `shared_buffers` is too small for the data size where PG-Strom tries to work, thus storage workloads restricts the entire performance, and may be unable to work GPU efficiently.
    - So, we recommend to expand the variable for a certain margin.
    - ```shared_buffers = 10GB```
    - Please consider to apply **SSD-to-GPU Direct SQL Execution** to process larger than system's physical RAM size.
- **work_mem**
    - Although it depends on the workloads, the initial configuration of `work_mem` is too small to choose the optimal query execution plan on analytic queries.
    - An typical example is, disk-based merge sort may be chosen instead of the in-memory quick-sorting.
    - So, we recommend to expand the variable for a certain margin.
    - ```work_mem = 1GB```
}

@ja:### OSのリソース制限の拡張
@en:### Expand OS resource limits

@ja{
SSD-to-GPUダイレクトSQLを使用する場合は特に、同時に大量のファイルをオープンする事があるため、プロセスあたりファイルディスクリプタ数の上限を拡大しておく必要があります。

また、PostgreSQLのクラッシュ時に確実にコアダンプを生成できるよう、コアファイルのサイズ上限を制限しないことを推奨します。
}
@en{
SSD-to-GPU Direct SQL especially tries to open many files simultaneously, so resource limit for number of file descriptors per process should be expanded.

Also, we recommend not to limit core file size to generate core dump of PostgreSQL certainly on system crash.
}
@ja{
PostgreSQLをsystemd経由で起動する場合、リソース制限に関する設定は`/etc/systemd/system/postgresql-XX.service.d/pg_strom.conf`に記述します。

RPMによるインストールの場合、デフォルトで以下の内容が設定されます。

環境変数 `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` に関する設定がコメントアウトされています。これは開発者向けのオプションで、これを有効にして起動すると、GPU側でエラーが発生した場合にGPUのコアダンプを生成させる事ができます。詳しくは[CUDA-GDB:GPU core dump support](https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-coredump)をご覧ください。
}
@en{
If PostgreSQL service is launched by systemd, you can put the configurations of resource limit at `/etc/systemd/system/postgresql-XX.service.d/pg_strom.conf`.

RPM installation setups the configuration below by the default.

It comments out configuration to the environment variable `CUDA_ENABLE_COREDUMP_ON_EXCEPTION`. This is a developer option that enables to generate GPU's core dump on any CUDA/GPU level errors, if enabled. See [CUDA-GDB:GPU core dump support](https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-coredump) for more details.
}
```
[Service]
LimitNOFILE=65536
LimitCORE=infinity
#Environment=CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
```

@ja:### PostgreSQLの起動
@en:### Start PostgreSQL

@ja{
PostgreSQLを起動します。

正常にセットアップが完了していれば、ログにPG-StromがGPUを認識した事を示すメッセージが記録されているはずです。
以下の例では、Tesla V100(PCIe; 16GB版)を認識しています。
}
@en{
Start PostgreSQL service.

If PG-Strom is set up appropriately, it writes out log message which shows PG-Strom recognized GPU devices.
The example below recognized the Tesla V100(PCIe; 16GB edition) device.
}

```
# systemctl start postgresql-10
# systemctl status -l postgresql-10
* postgresql-10.service - PostgreSQL 10 database server
   Loaded: loaded (/usr/lib/systemd/system/postgresql-10.service; disabled; vendor preset: disabled)
   Active: active (running) since Sat 2018-03-03 15:45:23 JST; 2min 21s ago
     Docs: https://www.postgresql.org/docs/10/static/
  Process: 24851 ExecStartPre=/usr/pgsql-10/bin/postgresql-10-check-db-dir ${PGDATA} (code=exited, status=0/SUCCESS)
 Main PID: 24858 (postmaster)
   CGroup: /system.slice/postgresql-10.service
           |-24858 /usr/pgsql-10/bin/postmaster -D /var/lib/pgsql/10/data/
           |-24890 postgres: logger process
           |-24892 postgres: bgworker: PG-Strom GPU memory keeper
           |-24896 postgres: checkpointer process
           |-24897 postgres: writer process
           |-24898 postgres: wal writer process
           |-24899 postgres: autovacuum launcher process
           |-24900 postgres: stats collector process
           |-24901 postgres: bgworker: PG-Strom ccache-builder2
           |-24902 postgres: bgworker: PG-Strom ccache-builder1
           `-24903 postgres: bgworker: logical replication launcher

Mar 03 15:45:19 saba.heterodb.com postmaster[24858]: 2018-03-03 15:45:19.195 JST [24858] HINT:  Run 'nvidia-cuda-mps-control -d', then start server process. Check 'man nvidia-cuda-mps-control' for more details.
Mar 03 15:45:20 saba.heterodb.com postmaster[24858]: 2018-03-03 15:45:20.509 JST [24858] LOG:  PG-Strom: GPU0 Tesla V100-PCIE-16GB (5120 CUDA cores; 1380MHz, L2 6144kB), RAM 15.78GB (4096bits, 856MHz), CC 7.0
Mar 03 15:45:20 saba.heterodb.com postmaster[24858]: 2018-03-03 15:45:20.510 JST [24858] LOG:  NVRTC - CUDA Runtime Compilation vertion 9.1
Mar 03 15:45:23 saba.heterodb.com postmaster[24858]: 2018-03-03 15:45:23.378 JST [24858] LOG:  listening on IPv6 address "::1", port 5432
Mar 03 15:45:23 saba.heterodb.com postmaster[24858]: 2018-03-03 15:45:23.378 JST [24858] LOG:  listening on IPv4 address "127.0.0.1", port 5432
Mar 03 15:45:23 saba.heterodb.com postmaster[24858]: 2018-03-03 15:45:23.442 JST [24858] LOG:  listening on Unix socket "/var/run/postgresql/.s.PGSQL.5432"
Mar 03 15:45:23 saba.heterodb.com postmaster[24858]: 2018-03-03 15:45:23.492 JST [24858] LOG:  listening on Unix socket "/tmp/.s.PGSQL.5432"
Mar 03 15:45:23 saba.heterodb.com postmaster[24858]: 2018-03-03 15:45:23.527 JST [24858] LOG:  redirecting log output to logging collector process
Mar 03 15:45:23 saba.heterodb.com postmaster[24858]: 2018-03-03 15:45:23.527 JST [24858] HINT:  Future log output will appear in directory "log".
Mar 03 15:45:23 saba.heterodb.com systemd[1]: Started PostgreSQL 10 database server.
```

@ja:### PG-Strom関連オブジェクトの作成
@en:### Creation of PG-Strom related objects

@ja{
最後に、PG-Stromに関連するSQL関数などのDBオブジェクトを作成します。
この手順はPostgreSQLのEXTENSION機能を用いてパッケージ化されており、SQLコマンドラインで`CREATE EXTENSION`コマンドを実行するだけです。
}
@en{
At the last, create database objects related to PG-Strom, like SQL functions.
This steps are packaged using EXTENSION feature of PostgreSQL. So, all you needs to run is `CREATE EXTENSION` on the SQL command line.
}
@ja{
なお、この手順は新しいデータベースを作成するたびに必要になる事に注意してください。
新しいデータベースを作成した時点で既にPG-Strom関連オブジェクトが作成されていてほしい場合は、予め`template1`データベースでPG-Stromエクステンションを作成しておけば、`CREATE DATABASE`コマンドの実行時に新しいデータベースへ設定がコピーされます。
}
@en{
Please note that this step is needed for each new database.
If you want PG-Strom is pre-configured on new database creation, you can create PG-Strom extension on the `template1` database, its configuration will be copied to the new database on `CREATE DATABASE` command.
}

```
$ psql postgres -U postgres
psql (10.2)
Type "help" for help.

postgres=# CREATE EXTENSION pg_strom ;
CREATE EXTENSION
```

@ja{
以上でインストール作業は完了です。
}
@en{
That's all for the installation.
}


@ja:# NVME-Stromモジュール
@en:# NVME-Strom module

@ja{
PG-Stromとは独立した別個のソフトウェアモジュールではありますが、SSD-to-GPUダイレクトSQL実行など、PG-Stromの中核機能と密接に関係しているNVME-Stromカーネルモジュールについても本節で説明します。
}
@en{
This section also introduces NVME-Strom Linux kernel module which is closely cooperating with core features of PG-Strom like SSD-to-GPU Direct SQL Execution, even if it is an independent software module.
}

@ja:## モジュールの入手とインストール
@en:## Getting the module and installation

@ja{
他のPG-Strom関連モジュールと同様、NVME-Stromは(https://heterodb.github.io/swdc/)[HeteroDB Software Distribution Center]からフリーソフトウェアとして配布されています。すなわち、オープンソースソフトウェアではありません。

`heterodb-swdc`パッケージを導入済みであれば、`yum install`コマンドを用いてRPMパッケージをダウンロード、インストールする事ができます。
}
@en{
Like other PG-Strom related modules, NVME-Strom is distributed at the (https://heterodb.github.io/swdc/)[HeteroDB Software Distribution Center] as a free software. In other words, it is not an open source software.

If your system already setup `heterodb-swdc` package, `yum install` command downloads the RPM file and install the `nvme_strom` package.
}

```
$ sudo yum install nvme_strom
Loaded plugins: fastestmirror
Loading mirror speeds from cached hostfile
 * base: mirrors.cat.net
 * epel: ftp.iij.ad.jp
 * extras: mirrors.cat.net
 * ius: mirrors.kernel.org
 * updates: mirrors.cat.net
Resolving Dependencies
--> Running transaction check
---> Package nvme_strom.x86_64 0:1.3-1.el7 will be installed
--> Finished Dependency Resolution

Dependencies Resolved

================================================================================
 Package             Arch            Version            Repository         Size
================================================================================
Installing:
 nvme_strom          x86_64          1.3-1.el7          heterodb          273 k

Transaction Summary
================================================================================
Install  1 Package

Total download size: 273 k
Installed size: 1.5 M
Is this ok [y/d/N]: y
Downloading packages:
No Presto metadata available for heterodb
nvme_strom-1.3-1.el7.x86_64.rpm                            | 273 kB   00:00
Running transaction check
Running transaction test
Transaction test succeeded
Running transaction
  Installing : nvme_strom-1.3-1.el7.x86_64                                  1/1
  :
<snip>
  :
DKMS: install completed.
  Verifying  : nvme_strom-1.3-1.el7.x86_64                                  1/1

Installed:
  nvme_strom.x86_64 0:1.3-1.el7

Complete!
```

@ja:## ライセンスの有効化
@en:## License activation

@ja{
NVME-Stromモジュールの全ての機能を利用するには、HeteroDB社が提供するライセンスの有効化が必要です。ライセンスなしで運用する事も可能ですが、その場合、下記の機能が制限を受けます。

- 複数個のGPUの利用
- SSD-to-GPUダイレクトSQL実行におけるストライピング(md-raid0)対応
}

@en{
License activation is needed to use all the features of NVME-Strom module, provided by HeteroDB,Inc. You can operate the system without license, but features below are restricted.
- Multiple GPUs support
- Striping support (md-raid0) at SSD-to-GPU Direct SQL
}

@ja{
ライセンスファイルは以下のような形式でHeteroDB社から入手する事ができます。
}
@en{
You can obtain a license file, like as a plain text below, from HeteroDB,Inc.
}
```
IAgIVdKxhe+BSer3Y67jQW0+uTzYh00K6WOSH7xQ26Qcw8aeUNYqJB9YcKJTJb+QQhjmUeQpUnboNxVwLCd3HFuLXeBWMKp11/BgG0FSrkUWu/ZCtDtw0F1hEIUY7m767zAGV8y+i7BuNXGJFvRlAkxdVO3/K47ocIgoVkuzBfLvN/h9LffOydUnHPzrFHfLc0r3nNNgtyTrfvoZiXegkGM9GBTAKyq8uWu/OGonh9ybzVKOgofhDLk0rVbLohOXDhMlwDl2oMGIr83tIpCWG+BGE+TDwsJ4n71Sv6n4bi/ZBXBS498qShNHDGrbz6cNcDVBa+EuZc6HzZoF6UrljEcl=
----
VERSION:2
SERIAL_NR:HDB-TRIAL
ISSUED_AT:2019-05-09
EXPIRED_AT:2019-06-08
GPU_UUID:GPU-a137b1df-53c9-197f-2801-f2dccaf9d42f
```

@ja{
これを `/etc/heterodb.license` にコピーし、PostgreSQLを再起動します。

以下のようにPostgreSQLの起動ログにライセンス情報が出力され、ライセンスの有効化が行われた事が分かります。
}
@en{
Copy the license file to `/etc/heterodb.license`, then restart PostgreSQL.

The startup log messages of PostgreSQL dumps the license information, and it tells us the license activation is successfully done.
}

```
$ pg_ctl restart
   :
LOG:  PG-Strom version 2.2 built for PostgreSQL 11
LOG:  PG-Strom: GPU0 Tesla P40 (3840 CUDA cores; 1531MHz, L2 3072kB), RAM 22.38GB (384bits, 3.45GHz), CC 6.1
   :
LOG:  HeteroDB License: { "version" : 2, "serial_nr" : "HDB-TRIAL", "issued_at" : "9-May-2019", "expired_at" : "8-Jun-2019", "gpus" : [ { "uuid" : "GPU-a137b1df-53c9-197f-2801-f2dccaf9d42f", "pci_id" : "0000:02:00.0" } ] }
LOG:  listening on IPv6 address "::1", port 5432
LOG:  listening on IPv4 address "127.0.0.1", port 5432
    :
```


@ja:## カーネルモジュールパラメータ
@en:## Kernel module parameters

@ja{
NVME-Stromカーネルモジュールにはパラメータがあります。

|パラメータ名        |型   |初期値|説明|
|:------------------:|:---:|:----:|:-----:|
|`verbose`           |`int`|`0`   |詳細なデバッグ出力を行います。|
|`fast_ssd_mode`     |`int`|`0`   |高速なNVME-SSDに適した動作モードです。|
|`p2p_dma_max_depth` |`int`|`1024`|NVMEデバイスのI/Oキューに同時に送出する事のできる非同期DMA要求の最大数です。|
|`p2p_dma_max_unitsz`|`int`|`256` |P2P DMA要求で一度に読み出すデータブロックの最大長（kB単位）です。|
}
@en{
NVME-Strom Linux kernel module has some parameters.

|Parameter           |Type |Default|Description|
|:------------------:|:---:|:----:|:-----:|
|`verbose`           |`int`|`0`   |Enables detailed debug output|
|`fast_ssd_mode`     |`int`|`0`   |Operating mode for fast NVME-SSD|
|`p2p_dma_max_depth` |`int`|`1024`|Maximum number of asynchronous P2P DMA request can be enqueued on the I/O-queue of NVME device|
|`p2p_dma_max_unitsz`|`int`|`256` |Maximum length of data blocks, in kB, to be read by a single P2P DMA request at once|
}

@ja{
`fast_ssd_mode`パラメータについての補足説明を付記します。

NVME-StromモジュールがSSD-to-GPU間のダイレクトデータ転送の要求を受け取ると、まず該当するデータブロックがOSのページキャッシュに載っているかどうかを調べます。
`fast_ssd_mode`が`0`の場合、データブロックが既にページキャッシュに載っていれば、その内容を呼び出し元のユーザ空間バッファに書き戻し、アプリケーションにCUDA APIを用いたHost->Device間のデータ転送を行うよう促します。これはPCIe x4接続のNVME-SSDなど比較的低速なデバイス向きの動作です。

一方、PCIe x8接続の高速SSDを使用したり、複数のSSDをストライピング構成で使用する場合は、バッファ間コピーの後で改めてHost->Device間のデータ転送を行うよりも、SSD-to-GPUのダイレクトデータ転送を行った方が効率的である事もあります。`fast_ssd_mode`が`0`以外の場合、NVME-StromドライバはOSのページキャッシュの状態に関わらず、SSD-to-GPUダイレクトのデータ転送を行います。

ただし、いかなる場合においてもOSのページキャッシュが dirty である場合にはSSD-to-GPUダイレクトでのデータ転送は行われません。
}

@en{
Here is an extra explanation for `fast_ssd_mode` parameter.

When NVME-Strom Linux kernel module get a request for SSD-to-GPU direct data transfer, first of all, it checks whether the required data blocks are caches on page-caches of operating system.
If `fast_ssd_mode` is `0`, NVME-Strom once writes back page caches of the required data blocks to the userspace buffer of the caller, then indicates application to invoke normal host-->device data transfer by CUDA API. It is suitable for non-fast NVME-SSDs such as PCIe x4 grade.

On the other hands, SSD-to-GPU direct data transfer may be faster, if you use PCIe x8 grade fast NVME-SSD or use multiple SSDs in striping mode, than normal host-->device data transfer after the buffer copy. If `fast_ssd_mode` is not `0`, NVME-Strom kicks SSD-to-GPU direct data transfer regardless of the page cache state.

However, it shall never kicks SSD-to-GPU direct data transfer if page cache is dirty.
}

@ja{
`p2p_dma_max_depth`パラメータに関する補足説明を付記します。

NVME-Stromモジュールは、SSD-to-GPU間のダイレクトデータ転送のDMA要求を作成し、それをNVMEデバイスのI/Oキューに送出します。
NVMEデバイスの能力を越えるペースで非同期DMA要求が投入されると、NVME-SSDコントローラはDMA要求を順に処理する事になるため、DMA要求のレイテンシは極めて悪化します。（一方、NVME-SSDコントローラには切れ目なくDMA要求が来るため、スループットは最大になります）
DMA要求の発行から処理結果が返ってくるまでの時間があまりにも長いと、場合によっては、これが何らかのエラーと誤認され、I/O要求のタイムアウトとエラーを引き起こす可能性があります。そのため、NVMEデバイスが遊ばない程度にDMA要求をI/Oキューに詰めておけば、それ以上のDMA要求を一度にキューに投入するのは有害無益という事になります。

`p2p_dma_max_depth`パラメータは、NVMEデバイス毎に、一度にI/Oキューに投入する事のできる非同期P2P DMA要求の数を制御します。設定値以上のDMA要求を投入しようとすると、スレッドは現在実行中のDMAが完了するまでブロックされ、それによってNVMEデバイスの高負荷を避ける事が可能となります。

}
@en{
Here is an extra explanation for `p2p_dma_max_depth` parameter.

NVME-Strom Linux kernel module makes DMA requests for SSD-to-GPU direct data transfer, then enqueues them to I/O-queue of the source NVME devices.
When asynchronous DMA requests are enqueued more than the capacity of NVME devices, latency of individual DMA requests become terrible because NVME-SSD controler processes the DMA requests in order of arrival. (On the other hands, it maximizes the throughput because NVME-SSD controler receives DMA requests continuously.)
If turn-around time of the DMA requests are too large, it may be wrongly considered as errors, then can lead timeout of I/O request and return an error status. Thus, it makes no sense to enqueue more DMA requests to the I/O-queue more than the reasonable amount of pending requests for full usage of NVME devices.

`p2p_dma_max_depth` parameter controls number of asynchronous P2P DMA requests that can be enqueued at once per NVME device. If application tries to enqueue DMA requests more than the configuration, the caller thread will block until completion of the running DMA. So, it enables to avoid unintentional high-load of NVME devices.
}
