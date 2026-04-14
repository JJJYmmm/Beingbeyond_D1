from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass

import numpy as np

from arm_exo_driver import ArmExoCfg, ArmExoDriver
from hand_exo_driver import HandExoCfg, HandExoDriver

from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
from D1_robot import D1Robot
import math

def deg_list_to_rad(xs):
    return [math.radians(v) for v in xs]

def rad_list_to_deg(xs):
    return [math.degrees(v) for v in xs]


ARM_INIT_RAD = np.array(
    [0.0, -90.0 * math.pi / 180.0, 90.0 * math.pi / 180.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TeleopRealCfg:
    arm_port: str = "/dev/arm_exo"
    hand_port: str = "/dev/hand_exo"

    urdf_path: str = get_default_urdf_path()
    robot_arm_dev: str = "/dev/ttyUSB0"
    robot_arm_baud: int = 1000000
    robot_hand_type: str = "right"
    robot_hand_can: str = "can0"
    robot_hand_baud: int = 1000000

    startup_wait_s: float = 3.0
    loop_hz: float = 60.0

    arm_min_valid: int = 4
    arm_stale_s: float = 0.20
    hand_stale_s: float = 0.20

    enable_vision: bool = True
    vision_filtered: bool = False

    head_deg: tuple[float, float] = (-30.0, 15.0)

    arm_pos_tol_deg: float = 5.0
    arm_vel_tol_deg_s: float = 5.0

    hand_speed: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    dbg: bool = False


class TeleopReal:
    def __init__(self, cfg: TeleopRealCfg = TeleopRealCfg()):
        self.cfg = cfg

        self.arm = ArmExoDriver(
            ArmExoCfg(
                p=self.cfg.arm_port,
            )
        )
        self.hand = HandExoDriver(
            HandExoCfg(
                p=self.cfg.hand_port,
                ac=True,
            )
        )

        self.robot: D1Robot | None = None
        self.teleop = np.zeros(14, dtype=np.float64)

    def close(self) -> None:
        try:
            if self.robot is not None:
                self.robot.close()
        except Exception:
            pass

        try:
            self.arm.stop()
        except Exception:
            pass

        try:
            self.hand.stop()
        except Exception:
            pass

    def _wait_ready(self) -> None:
        arm_ok = self.arm.wait_ready(
            timeout=self.cfg.startup_wait_s,
            min_valid=self.cfg.arm_min_valid,
        )
        hand_ok = self.hand.wait_ready(timeout=self.cfg.startup_wait_s)

        if not arm_ok:
            raise RuntimeError("arm exo not ready")
        if not hand_ok:
            raise RuntimeError("hand exo not ready")

        self.arm.zero()

    def _open_robot(self) -> None:
        self.robot = D1Robot(
            urdf_path=self.cfg.urdf_path,
            arm_dev=self.cfg.robot_arm_dev,
            arm_baud=self.cfg.robot_arm_baud,
            hand_type=self.cfg.robot_hand_type,
            hand_can=self.cfg.robot_hand_can,
            hand_baud=self.cfg.robot_hand_baud,
        )
        self.robot.__enter__()

        self.robot.hand.set_speed(speed=list(self.cfg.hand_speed))

        if self.cfg.enable_vision:
            self.robot.start_vision_thread(filtered=self.cfg.vision_filtered)

    def _move_to_initial_pose(self) -> None:
        assert self.robot is not None

        head_q = np.deg2rad(np.asarray(self.cfg.head_deg, dtype=np.float64))
        arm_q = ARM_INIT_RAD.copy()
        hand_q = np.zeros(6, dtype=np.float64)

        q = np.concatenate([head_q, arm_q, hand_q], axis=0).tolist()
        self.robot.set_q(q)

        self.robot.head_arm.wait_until_reached(
            list(head_q) + list(arm_q),
            pos_tol_deg=self.cfg.arm_pos_tol_deg,
            vel_tol_deg_s=self.cfg.arm_vel_tol_deg_s,
        )

    def _read_head(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.cfg.head_deg, dtype=np.float64))

    def _read_arm(self) -> np.ndarray:
        s = self.arm.get_state()

        now = time.time()
        ts = float(s["ts"])
        valid = np.asarray(s["valid"], dtype=bool)
        rel = np.asarray(s["rad"], dtype=np.float64).reshape(6)

        if ts <= 0.0 or (now - ts) > self.cfg.arm_stale_s:
            return self.teleop[2:8].copy()

        if int(valid.sum()) < self.cfg.arm_min_valid:
            return self.teleop[2:8].copy()

        q = ARM_INIT_RAD + rel
        prev = self.teleop[2:8].copy()

        for i in range(6):
            if valid[i]:
                prev[i] = q[i]
        return prev

    def _read_hand(self) -> np.ndarray:
        s = self.hand.get_state()

        now = time.time()
        ts = float(s["ts"])
        x = np.asarray(s["norm"], dtype=np.float64).reshape(6)

        if ts <= 0.0 or (now - ts) > self.cfg.hand_stale_s:
            return self.teleop[8:14].copy()

        return np.clip(x, 0.0, 1.0)

    def _build_teleop(self) -> np.ndarray:
        x = np.zeros(14, dtype=np.float64)
        x[0:2] = self._read_head()
        x[2:8] = self._read_arm()
        x[8:14] = self._read_hand()
        return x

    def _apply(self, x: np.ndarray) -> None:
        assert self.robot is not None
        self.robot.set_q(x.tolist())

    def _safe_exit(self) -> None:
        if self.robot is None:
            return

        try:
            head_q = np.deg2rad(np.asarray(self.cfg.head_deg, dtype=np.float64))
            arm_q = ARM_INIT_RAD.copy()
            hand_q = np.zeros(6, dtype=np.float64)

            q = np.concatenate([head_q, arm_q, hand_q], axis=0).tolist()
            self.robot.set_q(q)

            self.robot.head_arm.wait_until_reached(
                list(head_q) + list(arm_q),
                pos_tol_deg=self.cfg.arm_pos_tol_deg,
                vel_tol_deg_s=self.cfg.arm_vel_tol_deg_s,
            )
            time.sleep(0.2)
        except Exception as e:
            print(f"[teleop_real] safe exit failed: {e}")

    def run(self) -> None:
        print("=== Teleop Real ===")
        print("\033[91mWARNING: Always keep the physical emergency stop button within reach.\033[0m")
        print("\033[91m         Press it immediately if the robot motion looks unsafe.\033[0m\n")

        self._wait_ready()
        self._open_robot()
        self._move_to_initial_pose()

        print("========================")
        print("Start teleop real: head reserved. Ctrl+C to exit.")

        dt = 1.0 / self.cfg.loop_hz

        try:
            while True:
                t0 = time.perf_counter()

                self.teleop = self._build_teleop()
                self._apply(self.teleop)

                if self.cfg.dbg:
                    print(
                        "head =", np.array2string(self.teleop[0:2], precision=3, suppress_small=True),
                        "arm =", np.array2string(self.teleop[2:8], precision=3, suppress_small=True),
                        "hand =", np.array2string(self.teleop[8:14], precision=3, suppress_small=True),
                    )

                used = time.perf_counter() - t0
                rest = dt - used
                if rest > 0:
                    time.sleep(rest)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
        finally:
            self._safe_exit()
            self.close()


def main() -> None:
    teleop = TeleopReal(TeleopRealCfg())
    teleop.run()


if __name__ == "__main__":
    main()