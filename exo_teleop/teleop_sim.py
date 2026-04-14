from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass

import numpy as np

from arm_exo_driver import ArmExoCfg, ArmExoDriver
from hand_exo_driver import HandExoCfg, HandExoDriver
from d1_sim import D1IsaacSimulator


ARM_INIT_RAD = np.array(
    [0.0, -90.0 * math.pi / 180.0, 90.0 * math.pi / 180.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TeleopSimCfg:
    arm_port: str = "/dev/arm_exo"
    hand_port: str = "/dev/hand_exo"

    startup_wait_s: float = 3.0
    loop_hz: float = 60.0

    arm_min_valid: int = 4
    arm_stale_s: float = 0.20
    hand_stale_s: float = 0.20

    dbg: bool = False

class TeleopSimulator:
    def __init__(self, cfg: TeleopSimCfg = TeleopSimCfg()):
        self.cfg = cfg

        self.sim = D1IsaacSimulator()

        self.arm = ArmExoDriver(ArmExoCfg(p=self.cfg.arm_port,))
        self.hand = HandExoDriver(HandExoCfg(p=self.cfg.hand_port,))

        self.teleop = np.zeros(14, dtype=np.float64)

    def close(self) -> None:
        try:
            self.arm.stop()
        except Exception:
            pass
        try:
            self.hand.stop()
        except Exception:
            pass
        try:
            self.sim.close()
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
        self.sim.reset()

    def _read_head(self) -> np.ndarray:
        head_action = np.zeros(2, dtype=np.float64)
        head_action[0] = np.deg2rad(-30)
        head_action[1] = np.deg2rad(15)
        return head_action

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

        y = np.clip(x, 0.0, 1.0)
        return y

    def _build_teleop(self) -> np.ndarray:
        x = np.zeros(14, dtype=np.float64)
        x[0:2] = self._read_head()
        x[2:8] = self._read_arm()
        x[8:14] = self._read_hand()
        return x

    def _safe_exit(self) -> None:
        try:
            x = self.teleop.copy()
            x[8:14] = 0.0
            self.sim.apply_teleop_vector(x)
            for _ in range(3):
                if self.sim.step(sync_time=False) is False:
                    break
                time.sleep(0.01)
        except Exception:
            pass

        try:
            self.sim.reset()
            for _ in range(2):
                if self.sim.step(sync_time=False) is False:
                    break
                time.sleep(0.01)
        except Exception:
            pass

    def run(self) -> None:
        self._wait_ready()

        print("========================")
        print("Start teleop sim: head reserved. Ctrl+C to exit.")

        dt = 1.0 / self.cfg.loop_hz

        try:
            while True:
                t0 = time.perf_counter()

                self.teleop = self._build_teleop()
                self.sim.apply_teleop_vector(self.teleop)

                ok = self.sim.step(sync_time=False)
                if ok is False:
                    break

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
    teleop = TeleopSimulator(TeleopSimCfg())
    teleop.run()


if __name__ == "__main__":
    main()