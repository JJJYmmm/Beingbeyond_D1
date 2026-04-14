# D1 Exoskeleton Teleoperation User Guide

This document explains how to complete the following workflow on Linux:

1. Install the CH343 serial driver
2. Identify the teleoperation arm and glove devices
3. Create stable device names using `udev` rules
4. Configure and run:

* `teleop_sim.py`
* `teleop_real.py`

---

## 1. Environment Overview

This project involves two external devices:

* **arm_exo**: teleoperation arm
* **hand_exo**: teleoperation glove

---

## 2. Install the CH343 Driver

First, install the official WCH CH343 Linux driver.

```bash
git clone https://github.com/WCHSoftGroup/ch343ser_linux
cd ch343ser_linux/driver
make
sudo make load
sudo make install
sudo rmmod cdc-acm
ls -l /dev/ttyCH343USB*
```

Explanation:

* `make`: compile the driver
* `sudo make load`: load the driver
* `sudo make install`: install the driver
* `sudo rmmod cdc-acm`: prevent the default driver from taking over the device

---

## 3. Check Device Nodes

First plug in the teleoperation arm:

```bash
ls -l /dev/ttyCH343USB*
```

Then plug in the teleoperation glove:

```bash
ls -l /dev/ttyCH343USB*
```

Note that `/dev/ttyCH343USB0` and `/dev/ttyCH343USB1` are **not guaranteed to remain fixed permanently**.

---

## 4. Get the Unique Device Serial Number

### 4.1 Teleoperation Arm

```bash
udevadm info --query=property --name=/dev/ttyCH343USB0 | grep ID_SERIAL_SHORT
```

Example output:

```text
ID_SERIAL_SHORT=5B62022511
```

---

### 4.2 Teleoperation Glove

```bash
udevadm info --query=property --name=/dev/ttyCH343USB1 | grep ID_SERIAL_SHORT
```

Example output:

```text
ID_SERIAL_SHORT=5ABA130703
```

---

## 5. Create Stable Device Naming Rules

Create the rule file:

```bash
sudo vim /etc/udev/rules.d/99-ch343-dexhand.rules
```

Add the following:

```udev
SUBSYSTEM=="tty", ENV{ID_SERIAL_SHORT}=="5B62022511", SYMLINK+="arm_exo", MODE:="0666"
SUBSYSTEM=="tty", ENV{ID_SERIAL_SHORT}=="5ABA130703", SYMLINK+="hand_exo", MODE:="0666"
```

---

## 6. Reload Rules

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Check:

```bash
ls -l /dev/arm_exo /dev/hand_exo
```

---

## 7. Permission Check

```bash
groups
```

If `dialout` is not included:

```bash
sudo usermod -aG dialout $USER
```

Then log in again.

---

## 8. teleop_sim

Run:

```bash
python teleop_sim.py
```

---

## 9. teleop_real

Run:

```bash
python teleop_real.py
```

---

## 10. Safety Notice

Before running on the real robot, please make sure:

* The emergency stop button is always within reach
* The initial posture is safe
* The surrounding area is clear
