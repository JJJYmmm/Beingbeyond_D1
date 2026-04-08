# Lima Usage

这份文档记录了在本机上为 `Beingbeyond_D1` 项目搭建 Lima Linux 虚拟机、配置 Python 环境、以及尝试 USB 串口转发的完整经验。

适用环境：
- 宿主机：macOS Apple Silicon
- 虚拟机：Lima
- 项目目录：`/Users/jjjymmm/Code/Beingbeyond_D1`

## 1. 为什么要用 x86_64 虚拟机

这个仓库自带的 SDK wheel 是：

```text
lib/beingbeyond_d1_sdk-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

关键点：
- 这是 `Linux x86_64` wheel
- 不是 macOS wheel
- 也不是 Linux `aarch64` wheel

所以：
- 宿主机 macOS 不能直接安装
- Lima 里的 Linux ARM 虚拟机也不能直接安装
- 需要一台 `Linux x86_64` 虚拟机

## 2. 安装 Lima

宿主机先安装 Lima：

```bash
brew install lima
```

在 Apple Silicon 上，如果要跑 `x86_64` guest，还需要额外安装 guest agent：

```bash
brew install lima-additional-guestagents
```

这是因为跨架构的 `x86_64` Lima 实例依赖额外 guest agent 支持。

## 3. 创建 x86_64 Ubuntu 22.04 虚拟机

推荐直接创建一台 `qemu + x86_64` 的 Ubuntu 22.04：

```bash
limactl start \
  --name=ubuntu-22.04-x86_64 \
  --vm-type=qemu \
  --arch=x86_64 \
  --mount-writable \
  --set='.propagateProxyEnv = false' \
  template:ubuntu-22.04
```

说明：
- `--vm-type=qemu`：Apple Silicon 上跑 `x86_64` guest 需要 QEMU
- `--arch=x86_64`：和项目 wheel 架构匹配
- `--mount-writable`：让宿主机挂载目录在 guest 里可写
- `--set='.propagateProxyEnv = false'`：避免宿主机代理环境变量影响 guest

当前实例名：

```text
ubuntu-22.04-x86_64
```

查看实例：

```bash
limactl list
```

## 4. 挂载与文件位置

Lima 实例的数据目录在：

```text
/Users/jjjymmm/.lima/ubuntu-22.04-x86_64
```

重要文件：
- 磁盘镜像：`/Users/jjjymmm/.lima/ubuntu-22.04-x86_64/disk`
- 实例配置：`/Users/jjjymmm/.lima/ubuntu-22.04-x86_64/lima.yaml`
- SSH 配置：`/Users/jjjymmm/.lima/ubuntu-22.04-x86_64/ssh.config`

目录关系：
- 虚拟机自己的 Linux 系统文件在 `disk` 里
- 宿主机的 `/Users/jjjymmm` 会挂载进虚拟机
- 所以项目目录在虚拟机里仍然是：

```text
/Users/jjjymmm/Code/Beingbeyond_D1
```

这意味着：
- 你在虚拟机里修改项目文件，实际改的是宿主机文件
- 这正适合“在 Linux 环境运行，但直接操作本机项目目录”的工作流

## 5. SSH 连接虚拟机

Lima 会生成 SSH 配置：

```text
/Users/jjjymmm/.lima/ubuntu-22.04-x86_64/ssh.config
```

推荐在 `~/.ssh/config` 顶部加入：

```sshconfig
Include /Users/jjjymmm/.lima/ubuntu-22.04-x86_64/ssh.config
```

这样就可以直接连接：

```bash
ssh lima-ubuntu-22-04-x86-64
```

优点：
- 不需要手动关心 SSH 端口
- 如果 Lima 重启后端口变化，复用生成的配置仍然有效

## 6. 在虚拟机里安装项目环境

### 6.1 安装 uv

在虚拟机里执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

然后把 `uv` 加进 PATH：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 6.2 创建虚拟环境

进入项目目录：

```bash
cd /Users/jjjymmm/Code/Beingbeyond_D1
uv venv --python 3.10 .venv
```

### 6.3 安装依赖

按 README 的思路安装：

```bash
uv pip install --python .venv/bin/python -U pip
uv pip install --python .venv/bin/python numpy pyrealsense2 opencv-python
uv pip install --python .venv/bin/python lib/beingbeyond_d1_sdk-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

额外系统依赖：

```bash
sudo env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy apt-get update
sudo env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy apt-get install -y libgl1 libglib2.0-0 socat
```

说明：
- `libgl1` / `libglib2.0-0` 用于 `opencv-python`
- `socat` 后面串口转发会用到
- `apt` 前面显式去掉代理环境变量，是因为之前遇到过 `jammy-security` 走代理返回 `502`

### 6.4 环境验证

```bash
source .venv/bin/activate
python - <<'PY'
import numpy
import pyrealsense2
import cv2
import beingbeyond_d1_sdk
print("all imports ok")
PY
```

## 7. 日常使用流程

常用流程：

```bash
ssh lima-ubuntu-22-04-x86-64
cd /Users/jjjymmm/Code/Beingbeyond_D1
source .venv/bin/activate
```

如果只是想进入 Lima，也可以：

```bash
limactl shell ubuntu-22.04-x86_64
```

## 8. USB 串口问题总结

### 8.1 为什么 guest 里没有 `/dev/ttyUSB*`

这次定位的结论是：
- 宿主机已经识别到 USB 串口设备
- 例如宿主机上存在 `/dev/cu.usbserial-31310`
- 但 Lima guest 并没有直接看到这个 USB 设备

也就是说，这不是 Ubuntu 里 `brltty` 或 `ModemManager` 抢占串口导致的第一层问题，而是：

```text
USB 设备没有真正直通到 Lima guest
```

所以 guest 里自然不会有：
- `/dev/ttyUSB*`
- `/dev/ttyACM*`

### 8.2 这次 USB 设备的实际情况

宿主机能看到：
- `/dev/cu.usbserial-31310`
- `/dev/tty.usbserial-31310`

从设备信息看，它很像是 CH340/CH341 一类 USB 转串口。

### 8.3 `brltty` 不是当前主因

README 里建议：

```bash
sudo apt remove brltty
```

但这次 guest 里实际上并没有安装 `brltty`。

同时：
- `ModemManager` 也没有安装
- `usb-modeswitch` 在 guest 里存在，但不是当前阻塞点

因此当前问题不是“某个 Ubuntu 包占用了串口”，而是“guest 没有物理 USB 设备”。

## 9. 串口转发脚本

为了绕过 Lima guest 无法直接看到宿主机 USB 的问题，项目里增加了一个宿主机侧脚本：

[`scripts/lima_usb_forward.sh`](/Users/jjjymmm/Code/Beingbeyond_D1/scripts/lima_usb_forward.sh)

### 9.1 作用

它的目标是：
- 在宿主机打开真实串口
- 宿主机通过 TCP 暴露串口数据流
- 在 guest 里创建一个伪串口
- 把它链接到 `/dev/ttyUSB0`

### 9.2 使用方式

```bash
cd /Users/jjjymmm/Code/Beingbeyond_D1
./scripts/lima_usb_forward.sh start
./scripts/lima_usb_forward.sh stop
./scripts/lima_usb_forward.sh restart
./scripts/lima_usb_forward.sh status
```

### 9.3 可选环境变量

```bash
USB_SERIAL_DEVICE=/dev/cu.usbserial-31310 ./scripts/lima_usb_forward.sh start
BAUD=115200 ./scripts/lima_usb_forward.sh start
HOST_PORT=5555 ./scripts/lima_usb_forward.sh start
GUEST_TTY_LINK=/tmp/ttyUSB0 ./scripts/lima_usb_forward.sh start
GUEST_DEV_LINK=/dev/ttyUSB0 ./scripts/lima_usb_forward.sh start
```

### 9.4 当前脚本状态

这份脚本已经把大量基础工作自动化了：
- 检测宿主机 USB 串口
- 启动宿主机 `socat`
- 在 guest 中创建 `/dev/ttyUSB0`
- 尝试用 `systemd-run` 托管 guest 侧 `socat`

但要如实说明：

```text
串口转发脚本目前还没有验证到“稳定可长期复用”的程度。
```

具体表现是：
- `start` 命令现在可以成功启动
- `/dev/ttyUSB0` 的符号链接能创建出来
- 但 guest 侧伪串口在后续被程序实际打开时，仍然存在稳定性问题

所以这份脚本目前更像是一个“继续调试的基础版本”，而不是完全定稿的生产脚本。

## 10. 已经踩过的坑

### 10.1 不能用 ARM guest

`ubuntu-22.04` 的默认 Lima 实例如果是 `aarch64`，会装不上项目 wheel。

### 10.2 Apple Silicon 上跑 x86_64 guest 要用 QEMU

并且需要：

```bash
brew install lima-additional-guestagents
```

### 10.3 Lima 挂载默认不一定可写

创建实例时最好显式加：

```bash
--mount-writable
```

### 10.4 代理会影响 guest 的 apt

之前出现过：
- `jammy-security` 返回 `502`

因此在 guest 里运行 `apt` 时，建议显式去掉代理环境变量。

### 10.5 SSH 里直接跑后台进程容易不稳定

这次实践里，多次遇到：
- `ControlMaster` 干扰
- SSH 会话结束后子进程被带走
- 多层引号导致远端命令解析不稳定

所以凡是要在 guest 里启动常驻进程，最好优先考虑：
- systemd
- 或者更明确的守护化方式

## 11. 当前建议

如果你的目标只是：

```text
在 Linux x86_64 环境里安装 SDK、运行普通 Python 代码
```

那么当前 Lima 方案已经足够好用。

如果你的目标是：

```text
在 Lima guest 里稳定使用真实 USB 串口设备
```

那目前更建议：
- 继续完善 `scripts/lima_usb_forward.sh`
- 或者改用 USB 直通支持更直接的虚拟机方案，例如 UTM / VMware / VirtualBox

## 12. DexHand / can0 最终结论

这次已经针对 `DexHand` 在 Lima 中的可行性做了完整验证，结论可以直接写死：

```text
当前这台 Apple Silicon + Lima + Ubuntu x86_64 方案，不适合继续尝试 DexHand 的 can0 连接。
```

### 12.1 实际验证过的事实

#### A. SDK 确实要求现成的 CAN 接口

`DexHand` 初始化时会直接尝试：

```bash
ip link set can0 type can bitrate ...
ip link set can0 up
```

如果系统里没有 `can0`，就会报：

```text
Cannot find device "can0"
```

#### B. Lima guest 里没有真实 CAN 设备

实际检查到 guest 只有：
- `lo`
- `eth0`

没有：
- `can0`
- 任何 USB 直通形成的 CAN 网络接口

#### C. 走串口转 CAN 的备选方案也失败了

宿主机识别到的设备是：

```text
/dev/cu.usbserial-31310
```

这说明它更像是一个“USB 串口设备”，而不是原生 SocketCAN 网卡。

因此尝试了：
- 把宿主机串口转发到 Lima guest
- 在 guest 里通过 `slcand` 创建 `slcan0`

但最终失败，原因是 guest 内核没有 `slcan` 模块：

```text
modprobe: FATAL: Module slcan not found in directory /lib/modules/5.15.0-173-generic
```

随后的 `slcand` 也失败：

```text
ioctl TIOCSETD: Invalid argument
Cannot find device "slcan0"
```

### 12.2 是否还值得继续在 Lima 上尝试

结论是不值得。

原因不是“理论上绝对不可能”，而是：
- 当前 guest 没有原生 `can0`
- 当前 guest 内核没有 `slcan`
- USB 串口转发本身也还有稳定性问题

也就是说，DexHand 所需的通信链路在当前 Lima 方案下并没有成型。

### 12.3 建议直接放弃当前 Lima + DexHand 路线

建议把结论记成一句话：

```text
Lima 可以用来搭 Python / SDK 运行环境，但不要再继续用它尝试连接 DexHand。
```

更合适的下一步是：
- 换一台原生 `Linux x86_64` 机器
- 或者换一个更容易做 USB/CAN 直通的虚拟机平台，并确保 guest 内核支持对应驱动

如果目标是尽快驱动真机，优先推荐：

```text
直接在原生 Linux x86_64 机器上跑
```

## 13. 相关路径速查

项目目录：

```text
/Users/jjjymmm/Code/Beingbeyond_D1
```

Lima 实例目录：

```text
/Users/jjjymmm/.lima/ubuntu-22.04-x86_64
```

Lima SSH 配置：

```text
/Users/jjjymmm/.lima/ubuntu-22.04-x86_64/ssh.config
```

USB 转发脚本：

```text
/Users/jjjymmm/Code/Beingbeyond_D1/scripts/lima_usb_forward.sh
```

当前宿主机串口设备：

```text
/dev/cu.usbserial-31310
```
