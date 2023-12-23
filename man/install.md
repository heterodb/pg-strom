@ja:#インストール
@en:#Installation

@ja{
本章ではPG-Stromのインストール手順について説明します。
}
@en{
This chapter introduces the steps to install PG-Strom.
}

@ja:##チェックリスト
@en:##Checklist

@ja{
- **ハードウェア**
    - CUDA ToolkitのサポートするLinuxオペレーティングシステムを動作可能な x86_64 アーキテクチャのハードウェアが必要です。 
    - CPU、ストレージ、およびネットワークデバイスには特別な要件はありませんが、[note002:HW Validation List](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List)はハードウェア選定の上で参考になるかもしれません。
    - GPUダイレクトSQL実行を利用するにはNVME規格に対応したSSD、またはRoCEに対応した高速NICが必要で、GPUと同一のPCIe Root Complex配下に接続されている必要があります。
- **GPUデバイス**
    - PG-Stromを実行するには少なくとも一個のGPUデバイスがシステム上に必要です。これらはCUDA Toolkitでサポートされており、computing capability が6.0以降のモデル（Pascal世代以降）である必要があります。
    - [002: HW Validation List - List of supported GPU models](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List#list-of-supported-gpu-models)を参考にGPUを選定してください。
- **Operating System**
    - PG-Stromの実行には、CUDA Toolkitによりサポートされているx86_64アーキテクチャ向けのLinux OSが必要です。推奨環境はRed Hat Enterprise LinuxまたはCentOSのバージョン8.xシリーズです。
    - GPUダイレクトSQL（cuFileドライバ）を利用するには、CUDA Toolkitに含まれるnvidia-fsドライバと、Mellanox OFED (OpenFabrics Enterprise Distribution) ドライバのインストールが必要です。
- **PostgreSQL**
    - PG-Strom v5.0の実行にはPostgreSQLバージョン15以降が必要です。
    - PG-Stromが内部的に利用しているAPIの中には、これ以前のバージョンでは提供されていないものが含まれています。
- **CUDA Toolkit**
    - PG-Stromの実行にはCUDA Toolkit バージョン12.2以降が必要です。
    - PG-Stromが内部的に利用しているAPIの中には、これ以前のバージョンでは提供されていないものが含まれています。
}
@en{
- **Server Hardware**
    - It requires generic x86_64 hardware that can run Linux operating system supported by CUDA Toolkit. We have no special requirement for CPU, storage and network devices.
    - [note002:HW Validation List](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List) may help you to choose the hardware.
    - GPU Direct SQL Execution needs NVME-SSD devices, or fast network card with RoCE support, and to be installed under the same PCIe Root Complex where GPU is located on.
- **GPU Device**
    - PG-Strom requires at least one GPU device on the system, which is supported by CUDA Toolkit, has computing capability 6.0 (Pascal generation) or later;
    - Please check at [002: HW Validation List - List of supported GPU models](https://github.com/heterodb/pg-strom/wiki/002:-HW-Validation-List#list-of-supported-gpu-models) for GPU selection.
- **Operating System**
    - PG-Strom requires Linux operating system for x86_64 architecture, and its distribution supported by CUDA Toolkit. Our recommendation is Red Hat Enterprise Linux or CentOS version 8.x series.
    - GPU Direct SQL (with cuFile driver) needs the `nvidia-fs` driver distributed with CUDA Toolkit, and Mellanox OFED (OpenFabrics Enterprise Distribution) driver.
- **PostgreSQL**
    - PG-Strom v5.0 requires PostgreSQL v15 or later.
    - Some of PostgreSQL APIs used by PG-Strom internally are not included in the former versions.
- **CUDA Toolkit**
    - PG-Strom requires CUDA Toolkit version 12.2 or later.
    - Some of CUDA Driver APIs used by PG-Strom internally are not included in the former versions.
}

<!--
もうこの記述は必要ないはず

@ja:### GPUダイレクトSQL実行ドライバの選択
@en:### Selection of GPU Direct SQL Execiton drivers

@ja{
インストール作業の前に、GPUダイレクトSQLのソフトウェアスタックを検討してください。

[GPUダイレクトSQL](ssd2gpu.md)を実行するために必要なLinux kernelドライバには以下の２種類があります。

- HeteroDB NVME-Strom
    - 2018年にリリースされ、PG-Strom v2.0以降でサポートされているHeteroDB社製の専用ドライバ。
    - RHEL7.x/RHEL8.xに対応し、GPUDirect RDMA機構を用いてローカルのNVME-SSDからGPUへの直接データ読み出しが可能です。
- NVIDIA GPUDirect Storage
    - NVIDIA社が開発している、NVME/NVME-oFデバイスからGPUへ直接データ読み出しを可能にするドライバで、2021年5月現在、パブリックベータ版が提供されています。
    - PG-Strom v3.0で実験的に対応しており、RHEL8.3/8.4、およびUbuntu 18.04/20.04に対応しています。
    - HeteroDB社を含む複数のパートナー企業が対応を表明しており、共有ファイルシステムやNVME-oFを通じたSDS(Software Defined Storage)デバイスからの直接読み出しも可能です。

どちらのドライバを使用しても、性能面での差異はほとんどありません。
しかし、GPUDirect Storageドライバの方が、対応するストレージやファイルシステムの種類といった周辺エコシステムや、ソフトウェア品質管理体制において優位性があると考えられるため、RHEL7/CentOS7でPG-Stromを利用する場合を除き、今後はGPUDirect Storageドライバの利用を推奨します。
}
@en{
Please consider the software stack for GPUDirect SQL, prior to the installation.

There are two individual Linux kernel driver for [GPUDirect SQL](ssd2gpu.md) execution, as follows:

- HeteroDB NVME-Strom
    - The dedicated Linux kernel module, released at 2018, supported since PG-Strom v2.0.
    - It supports RHEL7.x/RHEL8.x, enables direct read from local NVME-SSDs to GPU using GPUDirect RDMA.
- NVIDIA GPUDirect Storage
    - The general purpose driver stack, has been developed by NVIDIA, to support direct read from NVME/NVME-oF devices to GPU. At May-2021, its public beta revision has been published.
    - PG-Strom v3.0 experimentally supports the GPUDirect Storage, that supports RHEL8.3/8.4 and Ubuntu 18.04/20.04.
    - Some partners, including HeteroDB, expressed to support this feature. It also allows direct read from shared-filesystems or SDS(Software Defined Storage) devices over NVME-oF protocols.

Here is little performance differences on the above two drivers.
On the other hands, GPUDirect Storage has more variations of the supported storages and filesystems, and more mature software QA process, expect for the case of PG-Strom on RHEL7/CentOS7, we will recommend to use GPUDirect Storage driver.
}
-->

@ja:##インストール手順
@en:##Steps to Install

@ja{
一連のインストール手順は以下の通りとなります。

1. H/Wの初期設定
1. OSのインストール
1. CUDA Toolkit のインストール
1. MOFEDドライバのインストール
1. HeteroDB拡張モジュールのインストール
1. PostgreSQLのインストール
1. PG-Stromのインストール
1. PostgreSQL拡張モジュールのインストール（必要に応じて）
    - PostGIS
    - contrib/cube
}
@en{
The overall steps to install are below:

1. Hardware Configuration
1. OS Installation
1. CUDA Toolkit installation
1. MOFED Driver installation
1. HeteroDB Extra Module installation
1. PostgreSQL installation
1. PG-Strom installation
1. PostgreSQL Extensions installation
    - PostGIS
    - contrib/cube
}

@ja:## OSのインストール
@en:## OS Installation

@ja{
CUDA ToolkitのサポートするLinuxディストリビューションを選択し、個々のディストリビューションのインストールプロセスに従ってインストール作業を行ってください。 CUDA ToolkitのサポートするLinuxディストリビューションは、[NVIDIA DEVELOPER ZONE](https://developer.nvidia.com/)において紹介されています。
}
@en{
Choose a Linux distribution which is supported by CUDA Toolkit, then install the system according to the installation process of the distribution. [NVIDIA DEVELOPER ZONE](https://developer.nvidia.com/) introduces the list of Linux distributions which are supported by CUDA Toolkit.
}
@ja{
例えば、Red Hat Enterprise Linux 8.x系列の場合、ベース環境として「最小限のインストール」を選択し、さらに以下のアドオンを選択してください。

- 開発ツール
}
@en{
In case of Red Hat Enterprise Linux 8.x series, choose "Minimal installation" as base environment, and also check the following add-ons.

- Development Tools
}

![RHEL8/CentOS8 Package Selection](./img/centos8_package_selection.png)

@ja{
サーバーへのOSインストール後、サードパーティーのパッケージをインストールするために、パッケージリポジトリの設定を行います。

なお、インストーラで「開発ツール」を選択しなかった場合、以下のコマンドでOSインストール後に追加インストールする事が可能です。
}
@en{
Next to the OS installation on the server, go on the package repository configuration to install the third-party packages.

If you didn't check the "Development Tools" at the installer, we can additionally install the software using the command below after the operating system installation.
}

```
# dnf groupinstall 'Development Tools'
```

@ja{
!!! Tip
    サーバに搭載されているGPUが新しすぎる場合、OS起動中にクラッシュ等の問題が発生する場合があります。
    その場合、カーネル起動オプションに`nouveau.modeset=0`を追加して標準のグラフィックスドライバを無効化する事で
    問題を回避できるかもしれません。
}
@en{
!!! Tip
    If GPU devices installed on the server are too new, it may cause system crash during system boot.
    In this case, you may avoid the problem by adding `nouveau.modeset=0` onto the kernel boot option, to disable
    the inbox graphic driver.
}

@ja:### nouveauドライバの無効化
@en:### Disables nouveau driver

@ja{
NVIDIA製GPU向けオープンソースの互換ドライバであるnouveauドライバがロードされている場合、nvidiaドライバをロードする事ができません。
この場合は、nouveauドライバの無効化設定を行った上でシステムを一度再起動してください。

nouveauドライバを無効化するには、以下の設定を`/etc/modprobe.d/disable-nouveau.conf`という名前で保存し、`dracut`コマンドを実行してLinux kernelのブートイメージに反映します。
その後、システムを一度再起動してください。
}
@en{
When the nouveau driver, that is an open source compatible driver for NVIDIA GPUs, is loaded, it prevent to load the nvidia driver.
In this case, reboot the operating system after a configuration to disable the nouveau driver.

To disable the nouveau driver, put the following configuration onto `/etc/modprobe.d/disable-nouveau.conf`, and run `dracut` command to apply them on the boot image of Linux kernel.
Then, restart the system once.
}
```
# cat > /etc/modprobe.d/disable-nouveau.conf <<EOF
blacklist nouveau
options nouveau modeset=0
EOF
# dracut -f
# shutdown -r now
```

@ja:### IOMMUの無効化
@en:### Disables IOMMU

@ja{
GPU-Direct SQLはCUDAのGPUDirect Storage (cuFile)というAPIを使用しています。

GPUDirect Storageの利用に先立って、OS側でIOMMUの設定を無効化する必要があります。

[NVIDIA GPUDirect Storage Installation and Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#install-prereqs)の記述を参考に、カーネル起動オプションの設定を行ってください。
}
@en{
GPU-Direct SQL uses GPUDirect Storage (cuFile) API of CUDA.

Prior to using GPUDirect Storage, it needs to disable the IOMMU configuration on the OS side.

Configure the kernel boot option according to the [NVIDIA GPUDirect Storage Installation and Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#install-prereqs) description.
}

@ja{
IOMMUを無効化するには、カーネル起動オプションに`amd_iommu=off` (AMD製CPUの場合)または、`intel_iommu=off` (Intel製CPUの場合)の設定を付加します。

エディタで`/etc/default/grub`を編集し、`GRUB_CMDLINE_LINUX_DEFAULT=`行に上記のオプションを付加してください。

例えば、以下のような設定となるはずです。
}
@en{
To disable IOMMU, add `amd_iommu=off` (for AMD CPU) or `intel_iommu=off` (for Intel CPU) to the kernel boot options.

Open `/etc/default/grub` with an editor and add the above option to the `GRUB_CMDLINE_LINUX_DEFAULT=` line.

For example, the settings should look like this:
}
```
  :
GRUB_CMDLINE_LINUX="rhgb quiet amd_iommu=off"
  :
```
@ja{
以下のコマンドを実行し、この設定をカーネル起動オプションに反映します。
}
@en{
Run the following commands to apply the configuration to the kernel bool options.
}
```
$ sudo update-grub
$ sudo shutdown -r now
```

@ja:### 追加リポジトリの有効化
@en:### Enables extra repositories

#### EPEL(Extra Packages for Enterprise Linux)
@ja{
PG-Stromの実行に必要なソフトウェアモジュールのいくつかは、EPEL(Extra Packages for Enterprise Linux)の一部として配布されています。
これらのソフトウェアを入手するためにEPELパッケージ群のリポジトリ定義をyumシステムに追加する必要があります。
}
@en{
Several software modules required by PG-Strom are distributed as a part of EPEL (Extra Packages for Enterprise Linux).
You need to add a repository definition of EPEL packages for yum system to obtain these software.
}
@ja{
EPELリポジトリから入手するパッケージの一つがDKMS(Dynamic Kernel Module Support)です。これは動作中のLinuxカーネルに適合したLinuxカーネルモジュールをオンデマンドでビルドするためのフレームワークで、NVIDIAのGPUデバイスドライバなど関連するカーネルモジュールが使用しています。
Linuxカーネルモジュールは、Linuxカーネルのバージョンアップに追従して再ビルドが必要であるため、DKMSなしでのシステム運用は現実的ではありません。
}
@en{
One of the package we will get from EPEL repository is DKMS (Dynamic Kernel Module Support). It is a framework to build Linux kernel module for the running Linux kernel on demand; used for NVIDIA's GPU driver and related.
Linux kernel module must be rebuilt according to version-up of Linux kernel, so we don't recommend to operate the system without DKMS.
}
@ja{
EPELリポジトリの定義は `epel-release` パッケージにより提供され、[Fedora Project](https://docs.fedoraproject.org/en-US/epel/#_quickstart)のサイトから入手する事ができます。
CentOS8では`dnf`コマンドを用いてのインストールも可能です。
}
@en{
`epel-release` package provides the repository definition of EPEL. You can obtain the package from the [Fedora Project](https://docs.fedoraproject.org/en-US/epel/#_quickstart) website.
For CentOS8, it can be installed using `dnf` command.
}

```
-- For RHEL8
# dnf install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm

-- For CentOS8
# dnf install epel-release
```
#### Red Hat CodeReady Linux Builder
@ja{
MOFED（Mellanox OpenFabrics Enterprise Distribution）ドライバのインストールには、Red Hat Enterprise Linux 8.xの標準インストール構成では無効化されている Red Hat CodeReady Linux Builder リポジトリを有効化する必要があります。CentOSにおいては、このリポジトリは PowerTools と呼ばれています。

このリポジトリを有効化するには、以下のコマンドを実行します。
}
@en{
Installation of MOFED (Mellanox OpenFabrics Enterprise Distribution) driver requires the ***Red Hat CodeReady Linux Builder*** repository which is disabled in the default configuration of Red Hat Enterprise Linux 8.x installation. In CentOS, it is called ***PowerTools***

To enable this repository, run the command below:
}

```
-- For RHEL8
# subscription-manager repos --enable codeready-builder-for-rhel-8-x86_64-rpms

-- For CentOS8
# dnf config-manager --set-enabled powertools
```

@ja:### heterodb-swdcのインストール
@en:### heterodb-swdc Installation

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
Webブラウザなどで[HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/)へアクセスし、ページの先頭にリンクの記載されている`heterodb-swdc-1.2-1.el8.noarch.rpm`をダウンロードしてインストールしてください。
heterodb-swdcパッケージがインストールされると、HeteroDB-SWDCからソフトウェアを入手するためのyumシステムへの設定が追加されます。
}
@en{
`heterodb-swdc` package provides the repository definition of HeteroDB-SWDC.
Access to the [HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/) using Web browser, download the `heterodb-swdc-1.2-1.el8.noarch.rpm` on top of the file list, then install this package.
Once heterodb-swdc package gets installed, yum system configuration is updated to get software from the HeteroDB-SWDC repository.
}
@ja{
以下のようにheterodb-swdcパッケージをインストールします。
}
@en{
Install the `heterodb-swdc` package as follows.
}

```
# dnf install https://heterodb.github.io/swdc/yum/rhel8-noarch/heterodb-swdc-1.2-1.el8.noarch.rpm
```

@ja:## CUDA Toolkitのインストール
@en:## CUDA Toolkit Installation

@ja{
本節ではCUDA Toolkitのインストールについて説明します。 既に最新のCUDA Toolkitをインストール済みである場合、本節の初期設定と合致しているかどうか確認してください。
}
@en{
This section introduces the installation of CUDA Toolkit. If you already installed the latest CUDA Toolkit, you can check whether your installation is identical with the configuration described in this section.
}
@ja{
NVIDIAはCUDA Toolkitのインストールに２通りの方法を提供しています。一つは自己実行型アーカイブ（runfile）によるもの。もう一つはRPMパッケージによるものです。
インストールオプションを指定するため、ここではrunfileを用いる方法を推奨します。
}
@en{
NVIDIA offers two approach to install CUDA Toolkit; one is by self-extracting archive (runfile), and the other is by RPM packages.
We recommend the runfile installation to specify the install-options individually.
}
@ja{
CUDA Toolkitのインストール用パッケージはNVIDIA DEVELOPER ZONEからダウンロードする事ができます。 適切なOS、アーキテクチャ、ディストリビューション、バージョンを指定し、『runfile(local)』版を選択してください。
}
@en{
You can download the installation package for CUDA Toolkit from NVIDIA DEVELOPER ZONE. Choose your OS, architecture, distribution and version, then choose "runfile(local)" edition.
}

![CUDA Toolkit download](./img/cuda-download.png)

@ja{
『runfile(local)』を選択すると、自己実行型アーカイブをダウンロードするURLと、それを起動するためのシェルコマンドが表示されます。
ガイダンス通りにインストールを進めてください。
}
@en{
Once you choose the "runfile(local)" option, it shows the URL to download the self-extracting archive and a few step-by-step shell commands to launch.
Run the installation according to the guidance.
}

```
$ wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda_12.3.1_545.23.08_linux.run
$ sudo sh cuda_12.3.1_545.23.08_linux.run
```

@ja{
インストーラが起動すると、いくつかのインストールオプションを指定する事ができます。

まず、デフォルトではチェックの外れている`[X] Kernel Objects` --> `[X] nvidia-fs` にチェックを入れてください。これはGPU-Direct SQLを使用するために必須のオプションです。
}
@en{
You can specify several installation options on the installer prompt.

First, check `[X] Kernel Objects` --> `[X] nvidia-fs`, which is unchecked by default. This is a required option to use GPU-Direct SQL.
}
![CUDA Runfile Installer1](./img/cuda-runfile-install-01.png)

@ja{
次に、`Options` --> `Driver Options` を下ってゆき、`[X] Install the kernel open module flavor` にチェックを入れてください。将来的に、GPUDirect Storage機能はオープンソース版のドライバでのみサポートされるという事が予告されています。
}
@en{
Next, go down to `Options` --> `Driver Options`, then check `[X] Install the kernel open module flavor`. It has been announced that GPUDirect Storage feature will only be supported on the open source version of the driver in the future.
}
![CUDA Runfile Installer1](./img/cuda-runfile-install-02.png)

@ja{
正常にインストールが完了すると、`/usr/local/cuda`配下にCUDA Toolkitが導入されています。
}

@en{
Once installation completed successfully, CUDA Toolkit is deployed at `/usr/local/cuda`.
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
Mon Dec 18 18:02:14 2023
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA L40S                    Off | 00000000:41:00.0 Off |                    0 |
| N/A   40C    P0              90W / 350W |      4MiB / 46068MiB |      3%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```


@ja:### PCI Bar1メモリの設定
@en:### PCI Bar1 Memory Configuration

@ja{
GPU-Direct SQLは、GPUデバイスメモリをホストシステム上のPCI BAR1領域（物理アドレス空間）にマップし、そこをDestinationとするP2P-RDMA要求をNVME機器に対して行う事で、ロスのない高速なデータ読出しを実現します。

十分な多重度を持ったP2P-RDMAを行うには、GPUがバッファをマップするのに十分なPCI BAR1領域を有している必要があります。大半のGPUではPCI BAR1領域の大きさは固定で、PG-Stromにおいては、それがGPUデバイスメモリのサイズを上回っている製品を推奨しています。

しかし、一部のGPU製品においては『動作モード』を切り替える事でPCI BAR1領域のサイズを切り替える事ができるものが存在します。お使いのGPUがそれに該当する場合は、[NVIDIA Display Mode Selector Tool](https://developer.nvidia.com/displaymodeselector)を参照の上、PCI BAR1領域のサイズを最大化するモードへと切り替えてください。

2023年12月時点では、以下のGPUの場合にNVIDIA Display Mode Selector Toolを利用して***Display Off**モードへと切り替える必要があります。
}
@en{
GPU-Direct SQL maps GPU device memory to the PCI BAR1 region (physical address space) on the host system, and sends P2P-RDMA requests to NVME devices with that as the destination for the shortest data transfer.

To perform P2P-RDMA with sufficient multiplicity, the GPU must have enough PCI BAR1 space to map the device buffer. The size of the PCI BAR1 area is fixed for most GPUs, and PG-Strom recommends products whose size exceeds the GPU device memory size.

However, some GPU products allow to change the size of the PCI BAR1 area by switching the operation mode. If your GPU is either of the following, refer to the [NVIDIA Display Mode Selector Tool](https://developer.nvidia.com/displaymodeselector) and switch to the mode that maximizes the PCI BAR1 area size.
}

- NVIDIA L40S
- NVIDIA L40
- NVIDIA A40
- NVIDIA RTX 6000 Ada
- NVIDIA RTX A6000
- NVIDIA RTX A5500
- NVIDIA RTX A5000

@ja{
システムに搭載されているGPUのメモリサイズやPCI BAR1サイズを確認するには、`nvidia-smi -q`コマンドを利用します。以下のように、メモリ関連の状態が表示されます。
}
@en{
To check the GPU memory size and PCI BAR1 size installed in the system, use the `nvidia-smi -q` command. Memory-related status is displayed as shown below.
}
```
$ nvidia-smi -q
        :
    FB Memory Usage
        Total                             : 46068 MiB
        Reserved                          : 685 MiB
        Used                              : 4 MiB
        Free                              : 45377 MiB
    BAR1 Memory Usage
        Total                             : 65536 MiB
        Used                              : 1 MiB
        Free                              : 65535 MiB
        :
```

@ja:##HeteroDB 拡張モジュール
@en:##HeteroDB extra modules

@ja{
`heterodb-extra`モジュールは、PG-Stromに以下の機能を追加します。

- マルチGPUの対応
- GPUダイレクトSQL
- GiSTインデックス対応
- ライセンス管理機能

これらの機能を使用せず、オープンソース版の機能のみを使用する場合は `heterodb-extra` モジュールのインストールは不要です。
本節の内容は読み飛ばして構いません。
}
@en{
`heterodb-extra` module enhances PG-Strom the following features.

- multi-GPUs support
- GPUDirect SQL
- GiST index support on GPU
- License management

If you don't use the above features, only open source modules, you don't need to install the `heterodb-extra` module here.
Please skip this section.
}

@ja{
以下のように、SWDCから`heterodb-extra`パッケージをインストールしてください。
}
@en{
Install the `heterodb-extra` package, downloaded from the SWDC, as follows.
}
```
# dnf install heterodb-extra
```

@ja:##MOFEDドライバのインストール
@en:##MOFED Driver Installation

@ja{
MOFEDドライバは、[こちら](https://mellanox.com/products/infiniband-drivers/linux/mlnx_ofed)からダウンロードする事ができます。

本節ではtgzアーカイブからのインストール例を紹介します。
}
@en{
You can download the latest MOFED driver from [here](https://mellanox.com/products/infiniband-drivers/linux/mlnx_ofed).

This section introduces the example of installation from the tgz archive.
}

![MOFED Driver Selection](./img/mofed-download.png)

@ja{
tgzアーカイブを展開し、`mlnxofedinstall`スクリプトを実行します。この時、GPUDirect Storageのサポートを有効化するオプションを付加するのを忘れないでください。
}
@en{
Extract the tgz archive, then kick `mlnxofedinstall` script. Please don't forget the options to enable GPUDirect Storage features.
}

```
# tar xvf MLNX_OFED_LINUX-5.8-3.0.7.0-rhel8.8-x86_64.tgz
# cd MLNX_OFED_LINUX-5.8-3.0.7.0-rhel8.8-x86_64
# ./mlnxofedinstall --with-nvmf --with-nfsrdma --add-kernel-support
# dracut -f
```

@ja{
MOFEDドライバのビルドおよびインストール中、不足パッケージのインストールを要求される事があります。
その場合、エラーメッセージを確認して要求されたパッケージの追加インストールを行ってください。
}
@en{
During the build and installation of MOFED drivers, the installer may require additional packages.
In this case, error message shall guide you the missing packages. So, please install them using `dnf` command.
}

```
Error: One or more required packages for installing MLNX_OFED_LINUX are missing.
Please install the missing packages using your Linux distribution Package Management tool.
Run:
yum install tcsh tcl tk kernel-modules-extra gcc-gfortran python36
```

@ja{
MOFEDドライバのインストールが完了すると、nvmeドライバなど、OS標準のものが置き換えられているはずです。

例えば以下の例では、OS標準の`nvme-rdma`ドライバ（`/lib/modules/<KERNEL_VERSION>/kernel/drivers/nvme/host/nvme-rdma.ko.xz`）ではなく、追加インストールされた`/lib/modules/<KERNEL_VERSION>/extra/mlnx-nvme/host/nvme-rdma.ko`が優先して使用されています。
}
@en{
Once MOFED drivers got installed, it should replace several INBOX drivers like nvme driver.

For example, the command below shows the `/lib/modules/<KERNEL_VERSION>/extra/mlnx-nvme/host/nvme-rdma.ko` that is additionally installed, instead of the INBOX `nvme-rdma` (`/lib/modules/<KERNEL_VERSION>/kernel/drivers/nvme/host/nvme-rdma.ko.xz`).
}

```
# modinfo nvme-rdma
filename:       /lib/modules/4.18.0-477.10.1.el8_8.x86_64/extra/mlnx-nvme/host/nvme-rdma.ko
license:        GPL v2
rhelversion:    8.8
srcversion:     533BB7E5866E52F63B9ACCB
depends:        nvme-core,mlx_compat,rdma_cm,ib_core,nvme-fabrics
name:           nvme_rdma
vermagic:       4.18.0-477.10.1.el8_8.x86_64 SMP mod_unload modversions
parm:           register_always:Use memory registration even for contiguous memory regions (bool)
```

@ja{
既にロードされているカーネルモジュール（例: `nvme`）を置き換えるため、ここで一度システムのシャットダウンと再起動を行います。

`mlnxofedinstall`スクリプトの完了後に、`dracut -f`を実行するのを忘れないでください。
}
@en{
Then, shutdown the system and restart, to replace the kernel modules already loaded (like `nvme`).

Please don't forget to run `dracut -f` after completion of the `mlnxofedinstall` script.
}

@ja{
再起動が完了すると、GPUDirect Storageが利用可能な状態となっているはずです。
以下の通り、`gdscheck`ツールを用いてストレージデバイス毎のコンフィグを確認してください。
}
@en{
After the reboot, your system will become the GPUDirect Storage ready state.
Run `gdscheck` tool to confirm the configuration for each storage devices, as follows.
}

```
# /usr/local/cuda/gds/tools/gdscheck -p
 GDS release version: 1.0.0.82
 nvidia_fs version:  2.7 libcufile version: 2.4
 ============
 ENVIRONMENT:
 ============
 =====================
 DRIVER CONFIGURATION:
 =====================
 NVMe               : Supported
 NVMeOF             : Supported
 SCSI               : Unsupported
 ScaleFlux CSD      : Unsupported
 NVMesh             : Unsupported
 DDN EXAScaler      : Unsupported
 IBM Spectrum Scale : Unsupported
 NFS                : Unsupported
 WekaFS             : Unsupported
 Userspace RDMA     : Unsupported
 --Mellanox PeerDirect : Enabled
 --rdma library        : Not Loaded (libcufile_rdma.so)
 --rdma devices        : Not configured
 --rdma_device_status  : Up: 0 Down: 0
 =====================
 CUFILE CONFIGURATION:
 =====================
 properties.use_compat_mode : true
 properties.gds_rdma_write_support : true
 properties.use_poll_mode : false
 properties.poll_mode_max_size_kb : 4
 properties.max_batch_io_timeout_msecs : 5
 properties.max_direct_io_size_kb : 16384
 properties.max_device_cache_size_kb : 131072
 properties.max_device_pinned_mem_size_kb : 33554432
 properties.posix_pool_slab_size_kb : 4 1024 16384
 properties.posix_pool_slab_count : 128 64 32
 properties.rdma_peer_affinity_policy : RoundRobin
 properties.rdma_dynamic_routing : 0
 fs.generic.posix_unaligned_writes : false
 fs.lustre.posix_gds_min_kb: 0
 fs.weka.rdma_write_support: false
 profile.nvtx : false
 profile.cufile_stats : 0
 miscellaneous.api_check_aggressive : false
 =========
 GPU INFO:
 =========
 GPU index 0 NVIDIA A100-PCIE-40GB bar:1 bar size (MiB):65536 supports GDS
 GPU index 1 NVIDIA A100-PCIE-40GB bar:1 bar size (MiB):65536 supports GDS
 ==============
 PLATFORM INFO:
 ==============
 IOMMU: disabled
 Platform verification succeeded

```

@ja{
!!! Tips
    **RAIDを使用する場合の追加設定**
    
    GPUDirect Storageを利用してSoftware RAID (md-raid0) 区画からデータを読み出す場合、
    以下の一行を`/lib/udev/rules.d/63-md-raid-arrays.rules` 設定に追加する必要があります。
    
    ```
    IMPORT{​program}="/usr/sbin/mdadm --detail --export $devnode"
    ```
    
    その後、設定を反映させるためにシステムを再起動してください。
    詳しくは[NVIDIA GPUDirect Storage Installation and Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#adding-udev-rules)を参照してください。
}

@en{
!!! Tips
    **Additional configuration for RAID volume**
    
    For data reading from software RAID (md-raid0) volumes by GPUDirect Storage,
    the following line must be added to the `/lib/udev/rules.d/63-md-raid-arrays.rules` configuration file.
    
    ```
    IMPORT{​program}="/usr/sbin/mdadm --detail --export $devnode"
    ```
    
    Then reboot the system to ensure the new configuration.
    See [NVIDIA GPUDirect Storage Installation and Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#adding-udev-rules) for the details.
}

@ja:### ライセンスの有効化
@en:### License activation

@ja{
`heterodb-extra`モジュールの全ての機能を利用するには、HeteroDB社が提供するライセンスの有効化が必要です。ライセンスなしで運用する事も可能ですが、その場合、下記の機能が制限を受けます。

- マルチGPUの利用
- GPUダイレクトSQLにおける複数NVME-SSDによるストライピング(md-raid0)
- GPUダイレクトSQLにおけるNVME-oFデバイスの利用
- GPU版PostGISにおけるGiSTインデックスの利用
}

@en{
License activation is needed to use all the features of `heterodb-extra`, provided by HeteroDB,Inc. You can operate the system without license, but features below are restricted.

- Multiple GPUs support
- Striping of NVME-SSD drives (md-raid0) on GPUDirect SQL
- Support of NVME-oF device on GPUDirect SQL
- Support of GiST index on GPU-version of PostGIS workloads
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
    :
 LOG:  HeteroDB Extra module loaded [api_version=20231105,cufile=on,nvme_strom=off,githash=9ca2fe4d2fbb795ad2d741dcfcb9f2fe499a5bdf]
 LOG:  HeteroDB License: { "version" : 2, "serial_nr" : "HDB-TRIAL", "issued_at" : "2022-11-19", "expired_at" : "2099-12-31", "nr_gpus" : 1, "gpus" : [ { "uuid" : "GPU-13943bfd-5b30-38f5-0473-78979c134606" } ]}
 LOG:  PG-Strom version 5.0.1 built for PostgreSQL 15 (githash: 972441dbafed6679af86af40bc8613be2d73c4fd)
    :
```

@ja:## PostgreSQLのインストール
@en:## PostgreSQL Installation

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

PostgreSQLメジャーバージョンとLinuxディストリビューションごとに多くのリポジトリ定義がありますが、あなたのLinuxディストリビューション向けのPostgreSQL 11以降のものを選択する必要があります。
}
@en{
Here is the list of yum repository definition: [http://yum.postgresql.org/repopackages.php](http://yum.postgresql.org/repopackages.php).

Repository definitions are per PostgreSQL major version and Linux distribution. You need to choose the one for your Linux distribution, and for PostgreSQL v11 or later.
}

@ja{
以下のステップで PostgreSQL のインストールを行います。

- yumリポジトリの定義をインストール
- OS標準のPostgreSQLモジュールの無効化
- PostgreSQLパッケージのインストール

例えばPostgreSQL v15を使用する場合、PG-Stromのインストールには `postgresql15-server`および`postgresql15-devel`パッケージが必要です。

以下は、RHEL8/CentOS8においてPostgreSQL v15をインストールする手順の例です。
}
@en{
You can install PostgreSQL as following steps:

- Installation of yum repository definition.
- Disables the distribution's default PostgreSQL module
- Installation of PostgreSQL packages.
}

```
# dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-8-x86_64/pgdg-redhat-repo-latest.noarch.rpm
# dnf -y module disable postgresql
# dnf install -y postgresql15-devel postgresql15-server
```

@ja{
!!! Note
    Red Hat Enterprise Linux 8 および CentOS 8の場合、パッケージ名`postgresql`がディストリビューション標準のものと競合してしまい、PGDG提供のパッケージをインストールする事ができません。そのため、`dnf -y module disable postgresql` コマンドを用いてディストリビューション標準の`postgresql`モジュールを無効化します。
}
@en{
!!! Note
    On the Red Hat Enterprise Linux 8 and CentOS 8, the package name `postgresql` conflicts to the default one at the distribution, thus, unable to install the packages from PGDG. So, disable the `postgresql` module by the distribution, using `dnf -y module disable postgresql`.
}

@ja{
PostgreSQL Global Development Groupの提供するRPMパッケージは`/usr/pgsql-<version>`という少々変則的なディレクトリにソフトウェアをインストールするため、`psql`等の各種コマンドを実行する際にはパスが通っているかどうか注意する必要があります。

`postgresql-alternatives`パッケージをインストールしておくと、各種コマンドへのシンボリックリンクを`/usr/local/bin`以下に作成するため各種オペレーションが便利です。また、複数バージョンのPostgreSQLをインストールした場合でも、`alternatives`コマンドによってターゲットとなるPostgreSQLバージョンを切り替える事が可能です。
}
@en{
The RPM packages provided by PostgreSQL Global Development Group installs software under the `/usr/pgsql-<version>` directory, so you may pay attention whether the PATH environment variable is configured appropriately.

`postgresql-alternative` package set up symbolic links to the related commands under `/usr/local/bin`, so allows to simplify the operations. Also, it enables to switch target version using `alternatives` command even if multiple version of PostgreSQL.
}

```
# dnf install postgresql-alternatives
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
PG-Stromおよび関連パッケージは[HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/)より配布されています。
既にyumシステムへリポジトリを追加済みであれば、それほど作業は多くありません。
}
@en{
PG-Strom and related packages are distributed from [HeteroDB Software Distribution Center](https://heterodb.github.io/swdc/).
If you repository definition has been added, not many tasks are needed.
}
@ja{
基盤となるPostgreSQLのバージョンごとに別個のPG-StromのRPMパッケージが準備されており、PostgreSQL v15用であれば`pg_strom-PG15`パッケージを、PostgreSQL v16用であれば`pg_strom-PG16`パッケージをインストールします。

これは、PostgreSQL拡張モジュールのバイナリ互換性に伴う制約です。
}
@en{
We provide individual RPM packages of PG-Strom for each PostgreSQL major version. `pg_strom-PG15` package is built for PostgreSQL v15, and `pg_strom-PG16` is also built for PostgreSQL v16.

It is a restriction due to binary compatibility of extension modules for PostgreSQL.
}

```
# dnf install -y pg_strom-PG15
```

@ja{
以上でパッケージのインストールは完了です。
}
@en{
That's all for package installation.
}

@ja:### ソースからのインストール
@en:### Installation from the source

@ja{
開発者向けに、ソースコードからPG-Stromをビルドする方法についても紹介します。
}
@en{
For developers, we also introduces the steps to build and install PG-Strom from the source code.
}

@ja:#### ソースコードの入手
@en:#### Getting the source code
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

@ja:#### PG-Stromのビルド
@en:#### Building the PG-Strom
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
$ make PG_CONFIG=/usr/pgsql-15/bin/pg_config
$ sudo make install PG_CONFIG=/usr/pgsql-15/bin/pg_config
```

@ja:### インストール後の設定
@en:### Post Installation Setup

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
# su - postgres
$ initdb -D /var/lib/pgdata/
The files belonging to this database system will be owned by user "postgres".
This user must also own the server process.

The database cluster will be initialized with locale "en_US.UTF-8".
The default database encoding has accordingly been set to "UTF8".
The default text search configuration will be set to "english".

Data page checksums are disabled.

fixing permissions on existing directory /var/lib/pgdata ... ok
creating subdirectories ... ok
selecting dynamic shared memory implementation ... posix
selecting default max_connections ... 100
selecting default shared_buffers ... 128MB
selecting default time zone ... Asia/Tokyo
creating configuration files ... ok
running bootstrap script ... ok
performing post-bootstrap initialization ... ok
syncing data to disk ... ok

initdb: warning: enabling "trust" authentication for local connections
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
GPUダイレクトSQLを使用する場合は特に、同時に大量のファイルをオープンする事があるため、プロセスあたりファイルディスクリプタ数の上限を拡大しておく必要があります。

また、PostgreSQLのクラッシュ時に確実にコアダンプを生成できるよう、コアファイルのサイズ上限を制限しないことを推奨します。
}
@en{
GPU Direct SQL especially tries to open many files simultaneously, so resource limit for number of file descriptors per process should be expanded.

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
以下の例では、NVIDIA A100 (PCIE版; 40GB) を認識しており、また、NVME-SSDごとに近傍のGPUがどちらであるのか出力されています。
}
@en{
Start PostgreSQL service.

If PG-Strom is set up appropriately, it writes out log message which shows PG-Strom recognized GPU devices.
The example below recognized two NVIDIA A100 (PCIE; 40GB), and displays the closest GPU identifier foe each NVME-SSD drive.
}

```
# systemctl start postgresql-13
# journalctl -u postgresql-13
-- Logs begin at Thu 2021-05-27 17:02:03 JST, end at Fri 2021-05-28 13:26:35 JST. --
May 28 13:09:33 kujira.heterodb.in systemd[1]: Starting PostgreSQL 13 database server...
May 28 13:09:33 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:33.500 JST [6336] LOG:  NVRTC 11.3 is successfully loaded.
May 28 13:09:33 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:33.510 JST [6336] LOG:  failed on open('/proc/nvme-strom'): No such file or directory - likely nvme_strom.ko is not loaded
May 28 13:09:33 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:33.510 JST [6336] LOG:  HeteroDB Extra module loaded (API=20210525; NVIDIA cuFile)
May 28 13:09:33 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:33.553 JST [6336] LOG:  HeteroDB License: { "version" : 2, "serial_nr" : "HDB-TRIAL", "issued_at" : "2021-05-27", "expired_at" : "2021-06-26", "gpus" : [ { "uuid" : "GPU-cca38cf1-ddcc-6230-57fe-d42ad0dc3315" }, { "uuid" : "GPU-13943bfd-5b30-38f5-0473-78979c134606" } ]}
May 28 13:09:33 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:33.553 JST [6336] LOG:  PG-Strom version 2.9 built for PostgreSQL 13
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.748 JST [6336] LOG:  PG-Strom: GPU0 NVIDIA A100-PCIE-40GB (108 SMs; 1410MHz, L2 40960kB), RAM 39.59GB (5120bits, 1.16GHz), PCI-E Bar1 64GB, CC 8.0
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.748 JST [6336] LOG:  PG-Strom: GPU1 NVIDIA A100-PCIE-40GB (108 SMs; 1410MHz, L2 40960kB), RAM 39.59GB (5120bits, 1.16GHz), PCI-E Bar1 64GB, CC 8.0
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.755 JST [6336] LOG:  - nvme0n1 (INTEL SSDPEDKE020T7; 0000:5e:00.0)
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.755 JST [6336] LOG:  - nvme1n1 (INTEL SSDPE2KX010T8; 0000:8a:00.0 --> GPU0)
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.755 JST [6336] LOG:  - nvme2n1 (INTEL SSDPE2KX010T8; 0000:8b:00.0 --> GPU0)
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.755 JST [6336] LOG:  - nvme4n1 (INTEL SSDPE2KX010T8; 0000:8d:00.0 --> GPU0)
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.755 JST [6336] LOG:  - nvme3n1 (INTEL SSDPE2KX010T8; 0000:8c:00.0 --> GPU0)
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.755 JST [6336] LOG:  - nvme6n1 (INTEL SSDPE2KX010T8; 0000:b5:00.0 --> GPU1)
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.755 JST [6336] LOG:  - nvme7n1 (INTEL SSDPE2KX010T8; 0000:b6:00.0 --> GPU1)
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.755 JST [6336] LOG:  - nvme5n1 (INTEL SSDPE2KX010T8; 0000:b4:00.0 --> GPU1)
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.755 JST [6336] LOG:  - nvme8n1 (INTEL SSDPE2KX010T8; 0000:b7:00.0 --> GPU1)
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.909 JST [6336] LOG:  redirecting log output to logging collector process
May 28 13:09:43 kujira.heterodb.in postmaster[6336]: 2021-05-28 13:09:43.909 JST [6336] HINT:  Future log output will appear in directory "log".
May 28 13:09:44 kujira.heterodb.in systemd[1]: Started PostgreSQL 13 database server.
```

@ja:### PG-Stromエクステンションの作成
@en:### Creation of PG-Strom Extension

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
$ psql -U postgres
psql (15.1)
Type "help" for help.

postgres=# create extension pg_strom ;
CREATE EXTENSION
```

@ja{
以上でインストール作業は完了です。
}
@en{
That's all for the installation.
}



@ja:##PostGISのインストール
@en:##PostGIS Installation

@ja{
PG-Stromは一部のPostGIS関数のGPU処理をサポートしています。
本節ではPostGISのインストール手順について説明を行いますが、必要に応じて読み飛ばしてください。
}
@en{
PG-Strom supports execution of a part of PostGIS functions on GPU devices.
This section introduces the steps to install PostGIS module. Skip it on your demand.
}

@ja{
PostgreSQLと同様に、PostgreSQL Global Development GroupのyumリポジトリからPostGISモジュールをインストールする事ができます。
以下の例は、PostgreSQL v15向けにビルドされたPostGIS v3.2をインストールするものです。
}
@en{
PostGIS module can be installed from the yum repository by PostgreSQL Global Development Group, like PostgreSQL itself.
The example below shows the command to install PostGIS v3.2 built for PostgreSQL v15.
}

```
-- For RHEL8
# dnf install -y postgis32_15 --enablerepo=codeready-builder-for-rhel-8-x86_64-rpms

-- For CentOS8
# dnf install -y postgis32_15 --enablerepo=powertools
```

@ja{
データベースクラスタを作成してPostgreSQLサーバを起動し、SQLクライアントから`CREATE EXTENSION`コマンドを実行してGeometryデータ型や地理情報分析のためのSQL関数を作成します。
これでPostGISのインストールは完了です。
}
@en{
Start PostgreSQL server after the initial setup of database cluster, then run `CREATE EXTENSION` command from SQL client to define geometry data type and SQL functions for geoanalytics.
}

```
postgres=# CREATE EXTENSION postgis;
CREATE EXTENSION
```

