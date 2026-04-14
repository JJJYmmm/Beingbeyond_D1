# D1 外骨骼遥操使用说明

本文说明如何在 Linux 下完成以下流程：

1. 安装 CH343 串口驱动
2. 识别遥操臂与遥操手套设备
3. 使用 `udev` 规则为设备创建稳定名字
4. 配置并运行：

* `teleop_sim.py`
* `teleop_real.py`

---

## 1. 环境说明

本项目中涉及两个外设：

* **arm_exo**：遥操臂
* **hand_exo**：遥操手套

---

## 2. 安装 CH343 驱动

先安装 WCH 官方 CH343 Linux 驱动。

```bash
git clone https://github.com/WCHSoftGroup/ch343ser_linux
cd ch343ser_linux/driver
make
sudo make load
sudo make install
sudo rmmod cdc-acm
ls -l /dev/ttyCH343USB*
```

说明：

* `make`：编译驱动
* `sudo make load`：加载驱动
* `sudo make install`：安装驱动
* `sudo rmmod cdc-acm`：避免默认驱动占用

---

## 3. 检查设备节点

先插入遥操臂：

```bash
ls -l /dev/ttyCH343USB*
```

再插入遥操手套：

```bash
ls -l /dev/ttyCH343USB*
```

注意 `/dev/ttyCH343USB0` 和 `/dev/ttyCH343USB1` 不保证永久固定。

---

## 4. 获取设备唯一序列号

### 4.1 遥操臂

```bash
udevadm info --query=property --name=/dev/ttyCH343USB0 | grep ID_SERIAL_SHORT
```

示例输出：

```
ID_SERIAL_SHORT=5B62022511
```

---

### 4.2 遥操手套

```bash
udevadm info --query=property --name=/dev/ttyCH343USB1 | grep ID_SERIAL_SHORT
```

示例输出：

```
ID_SERIAL_SHORT=5ABA130703
```

---

## 5. 创建稳定命名规则

创建规则文件：

```bash
sudo vim /etc/udev/rules.d/99-ch343-dexhand.rules
```

写入：

```udev
SUBSYSTEM=="tty", ENV{ID_SERIAL_SHORT}=="5B62022511", SYMLINK+="arm_exo", MODE:="0666"
SUBSYSTEM=="tty", ENV{ID_SERIAL_SHORT}=="5ABA130703", SYMLINK+="hand_exo", MODE:="0666"
```

---

## 6. 重载规则

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

检查：

```bash
ls -l /dev/arm_exo /dev/hand_exo
```

---

## 7. 权限检查

```bash
groups
```

如果没有 `dialout`：

```bash
sudo usermod -aG dialout $USER
```

重新登录。

---

## 8. teleop_sim

运行：

```bash
python teleop_sim.py
```

---

## 9. teleop_real

运行：

```bash
python teleop_real.py
```

---


## 10. 安全提醒

真机运行前请确认：

* 急停按钮可随时按下
* 初始姿态安全