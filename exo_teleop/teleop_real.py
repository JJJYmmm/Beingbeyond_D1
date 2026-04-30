from __future__ import annotations

import json
import math
import select
import sys
import termios
import time
import traceback
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from arm_exo_driver import ArmExoCfg, ArmExoDriver
from hand_exo_driver import HandExoCfg, HandExoDriver

from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path
from D1_robot import D1Robot

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

    head_deg: tuple[float, float] = (-20.0, 25.0)

    arm_pos_tol_deg: float = 5.0
    arm_vel_tol_deg_s: float = 5.0

    hand_speed: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    dbg: bool = False
    record_data: bool = True
    record_root: str = "teleop_records"
    record_rgb_jpg_quality: int = 95
    record_start_key: str = "r"
    record_stop_key: str = "s"


class TeleopEpisodeRecorder:
    def __init__(self, cfg: TeleopRealCfg):
        self.cfg = cfg
        self.enabled = bool(cfg.record_data)
        self.record_root = Path(cfg.record_root)
        self.episode_dir: Path | None = None
        self.rgb_dir: Path | None = None
        self.depth_dir: Path | None = None

        self.samples: list[dict[str, object]] = []
        self.camera_t: list[float] = []
        self.camera_frame_idx: list[int] = []
        self._frame_idx = 0
        self._last_camera_ts = 0.0
        self._episode_idx = 0
        self._episode_start_perf = 0.0
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self.enabled and self._recording

    def start_episode(self) -> Path | None:
        if not self.enabled or self._recording:
            return

        self.record_root.mkdir(parents=True, exist_ok=True)
        self._episode_idx += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_dir = self.record_root / f"episode_{stamp}_{self._episode_idx:03d}"
        self.rgb_dir = self.episode_dir / "rgb"
        self.depth_dir = self.episode_dir / "depth"

        self.rgb_dir.mkdir(parents=True, exist_ok=False)
        self.depth_dir.mkdir(parents=True, exist_ok=False)
        self.samples = []
        self.camera_t = []
        self.camera_frame_idx = []
        self._frame_idx = 0
        self._last_camera_ts = 0.0
        self._episode_start_perf = time.perf_counter()
        self._recording = True

        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "loop_hz": self.cfg.loop_hz,
            "head_deg": list(self.cfg.head_deg),
            "vision_filtered": bool(self.cfg.vision_filtered),
            "robot_arm_dev": self.cfg.robot_arm_dev,
            "robot_hand_can": self.cfg.robot_hand_can,
        }
        (self.episode_dir / "meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        print(f"[record] episode dir: {self.episode_dir}")
        return self.episode_dir

    def record_sample(
        self,
        *,
        t: float,
        teleop: np.ndarray,
        arm_state: dict,
        hand_state: dict,
        robot_q_rad: np.ndarray,
        robot_dq_rad: np.ndarray,
        robot_hand_q: np.ndarray,
        color_rgb: np.ndarray | None,
        depth_m: np.ndarray | None,
        camera_ts: float,
    ) -> None:
        if not self.is_recording:
            return

        sample = {
            "t": float(t),
            "teleop": np.asarray(teleop, dtype=np.float64).copy(),
            "arm_exo_rad": np.asarray(arm_state["rad"], dtype=np.float64).copy(),
            "arm_exo_valid": np.asarray(arm_state["valid"], dtype=bool).copy(),
            "arm_exo_ts": float(arm_state["ts"]),
            "hand_exo_norm": np.asarray(hand_state["norm"], dtype=np.float64).copy(),
            "hand_exo_raw": np.asarray(hand_state["raw"], dtype=np.float64).copy(),
            "hand_exo_ts": float(hand_state["ts"]),
            "robot_q_rad": np.asarray(robot_q_rad, dtype=np.float64).copy(),
            "robot_dq_rad": np.asarray(robot_dq_rad, dtype=np.float64).copy(),
            "robot_hand_q": np.asarray(robot_hand_q, dtype=np.float64).copy(),
        }
        self.samples.append(sample)

        if (
            color_rgb is None
            or depth_m is None
            or camera_ts <= 0.0
            or camera_ts <= self._last_camera_ts
        ):
            return

        assert self.rgb_dir is not None
        assert self.depth_dir is not None

        rgb_path = self.rgb_dir / f"{self._frame_idx:06d}.jpg"
        depth_path = self.depth_dir / f"{self._frame_idx:06d}.png"

        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(rgb_path),
            color_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.record_rgb_jpg_quality)],
        )

        depth_mm = np.clip(np.rint(depth_m * 1000.0), 0.0, 65535.0).astype(np.uint16)
        cv2.imwrite(str(depth_path), depth_mm)

        self.camera_t.append(float(camera_ts))
        self.camera_frame_idx.append(int(self._frame_idx))
        self._frame_idx += 1
        self._last_camera_ts = float(camera_ts)

    def stop_episode(self) -> Path | None:
        if not self.enabled or self.episode_dir is None or not self._recording:
            return

        n = len(self.samples)
        if n == 0:
            print("[record] no samples captured in episode")
            self._recording = False
            return

        np.savez_compressed(
            self.episode_dir / "episode_data.npz",
            t=np.asarray([s["t"] for s in self.samples], dtype=np.float64),
            teleop=np.stack([s["teleop"] for s in self.samples], axis=0),
            arm_exo_rad=np.stack([s["arm_exo_rad"] for s in self.samples], axis=0),
            arm_exo_valid=np.stack([s["arm_exo_valid"] for s in self.samples], axis=0),
            arm_exo_ts=np.asarray([s["arm_exo_ts"] for s in self.samples], dtype=np.float64),
            hand_exo_norm=np.stack([s["hand_exo_norm"] for s in self.samples], axis=0),
            hand_exo_raw=np.stack([s["hand_exo_raw"] for s in self.samples], axis=0),
            hand_exo_ts=np.asarray([s["hand_exo_ts"] for s in self.samples], dtype=np.float64),
            robot_q_rad=np.stack([s["robot_q_rad"] for s in self.samples], axis=0),
            robot_dq_rad=np.stack([s["robot_dq_rad"] for s in self.samples], axis=0),
            robot_hand_q=np.stack([s["robot_hand_q"] for s in self.samples], axis=0),
            camera_t=np.asarray(self.camera_t, dtype=np.float64),
            camera_frame_idx=np.asarray(self.camera_frame_idx, dtype=np.int32),
        )

        summary = {
            "num_control_samples": n,
            "num_camera_frames": len(self.camera_t),
            "duration_s": float(self.samples[-1]["t"]) if self.samples else 0.0,
            "episode_dir": str(self.episode_dir),
        }
        (self.episode_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(
            f"[record] saved {n} control samples and {len(self.camera_t)} camera frames to {self.episode_dir}"
        )
        out = self.episode_dir
        self._recording = False
        return out

    def current_episode_time(self) -> float:
        if not self.is_recording:
            return 0.0
        return max(0.0, time.perf_counter() - self._episode_start_perf)


class TerminalKeyReader:
    def __init__(self):
        self._fd: int | None = None
        self._old_attrs = None
        self.enabled = False

    def start(self) -> bool:
        if not sys.stdin.isatty():
            return False
        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.enabled = True
        return True

    def poll_key(self) -> str | None:
        if not self.enabled or self._fd is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.read(1)

    def close(self) -> None:
        if not self.enabled or self._fd is None or self._old_attrs is None:
            return
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
        self.enabled = False


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
        self.recorder = TeleopEpisodeRecorder(self.cfg)
        self.key_reader = TerminalKeyReader()

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

    def _record_step(self, t: float, teleop: np.ndarray) -> None:
        if not self.recorder.is_recording or self.robot is None:
            return

        arm_state = self.arm.get_state()
        hand_state = self.hand.get_state()
        robot_q_rad, robot_dq_rad = self.robot.head_arm.get_positions_and_velocities()
        robot_hand_q = self.robot.hand.read_joint_pos()
        color_rgb, depth_m, camera_ts = self.robot.get_latest_vision_frames()

        self.recorder.record_sample(
            t=t,
            teleop=teleop,
            arm_state=arm_state,
            hand_state=hand_state,
            robot_q_rad=np.asarray(robot_q_rad, dtype=np.float64),
            robot_dq_rad=np.asarray(robot_dq_rad, dtype=np.float64),
            robot_hand_q=np.asarray(robot_hand_q, dtype=np.float64),
            color_rgb=color_rgb,
            depth_m=depth_m,
            camera_ts=camera_ts,
        )

    def _handle_record_keys(self) -> None:
        if not self.cfg.record_data:
            return

        key = self.key_reader.poll_key()
        if key is None:
            return

        key = key.lower()
        if key == self.cfg.record_start_key.lower():
            if self.recorder.is_recording:
                print("[record] already recording")
                return
            self.recorder.start_episode()
            return

        if key == self.cfg.record_stop_key.lower():
            if not self.recorder.is_recording:
                print("[record] not currently recording")
                return
            self.recorder.stop_episode()

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
        if self.cfg.record_data:
            print(f"Recording enabled: episodes will be saved under {self.cfg.record_root}")
            if self.key_reader.start():
                print(
                    f"Press '{self.cfg.record_start_key}' to start a new episode, "
                    f"'{self.cfg.record_stop_key}' to stop and save it."
                )
            else:
                print("[record] stdin is not a TTY, manual record hotkeys disabled")

        dt = 1.0 / self.cfg.loop_hz

        try:
            while True:
                t0 = time.perf_counter()
                self._handle_record_keys()

                self.teleop = self._build_teleop()
                self._apply(self.teleop)
                self._record_step(self.recorder.current_episode_time(), self.teleop)

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
            if self.recorder.is_recording:
                self.recorder.stop_episode()
            self.key_reader.close()
            self._safe_exit()
            self.close()


def main() -> None:
    teleop = TeleopReal(TeleopRealCfg())
    teleop.run()


if __name__ == "__main__":
    main()
