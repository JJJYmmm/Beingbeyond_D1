import os
import sys
import time
import traceback
import numpy as np

from beingbeyond_d1_sdk.head_arm import HeadArmRobot
from beingbeyond_d1_sdk.dex_hand import DexHand
from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path

'''
python replay_demo_data.py demo_data.npz

python replay_demo_data.py demo_data.npz 2.0 200

SPEED=1.5 SEND_HZ=150 python replay_demo_data.py demo_data.npz
'''

DEV = "/dev/ttyUSB1"
CAN_IFACE = "can0"
HAND_TYPE = "right"

# Optional: go to first pose before replay
GO_FIRST = True


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def get_env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def get_arg_float(i: int, default: float) -> float:
    # sys.argv: [script, path, speed?, send_hz?]
    if len(sys.argv) > i:
        try:
            return float(sys.argv[i])
        except ValueError:
            return default
    return default


def interp_linear_vec(t_play: float, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    if t_play <= t[0]:
        return x[0]
    if t_play >= t[-1]:
        return x[-1]

    idx = int(np.searchsorted(t, t_play, side="right") - 1)
    idx = max(0, min(idx, len(t) - 2))

    t0 = float(t[idx])
    t1 = float(t[idx + 1])
    x0 = x[idx]
    x1 = x[idx + 1]

    dt = t1 - t0
    if dt <= 1e-12:
        return x1
    u = (t_play - t0) / dt
    return x0 * (1.0 - u) + x1 * u


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python replay_demo_data.py <demo_npz_path> [speed] [send_hz]")
        print("Also support env: SPEED, SEND_HZ")
        sys.exit(1)

    path = sys.argv[1]
    data = np.load(path, allow_pickle=False)

    t = np.asarray(data["t"], dtype=np.float64)
    q = np.asarray(data["q_rad"], dtype=np.float64)       # recorded q (N,8)
    hand_q = np.asarray(data["hand_q"], dtype=np.float64) # recorded hand (N,6), normalized

    if len(t) < 2:
        raise RuntimeError("Not enough samples in file.")
    if q.ndim != 2 or q.shape[1] != 8:
        raise RuntimeError(f"q_rad should be (N,8), got {q.shape}")
    if hand_q.ndim != 2 or hand_q.shape[1] != 6:
        raise RuntimeError(f"hand_q should be (N,6), got {hand_q.shape}")

    # speed / send_hz interface:
    speed = get_arg_float(2, get_env_float("SPEED", 1.0))
    send_hz = get_arg_float(3, get_env_float("SEND_HZ", 100.0))

    if speed <= 1e-6:
        raise RuntimeError("speed must be > 0")
    if send_hz <= 1e-6:
        raise RuntimeError("send_hz must be > 0")

    dt_wall = 1.0 / float(send_hz)

    urdf = get_default_urdf_path()

    print("\033[91mWARNING: Always keep the physical emergency stop button within reach,\033[0m")
    print("\033[91m         and press it immediately if the robot motion looks unsafe.\033[0m\n")
    print(f"[REPLAY] file: {path}")
    print(f"[REPLAY] speed={speed}x, send_hz={send_hz}")

    # precompute relative time base
    t0 = float(t[0])
    t_end = float(t[-1])

    # Hand init
    hand = DexHand(hand_type=HAND_TYPE, can_iface=CAN_IFACE)
    hand.set_speed(speed=[1.0] * 6)
    hand.set_torque(torque=[0.6] * 6)

    try:
        with HeadArmRobot(urdf_path=urdf, dev=DEV) as robot:
            joint_names = robot.joint_names
            n_joints = len(joint_names)
            print("Joint order:", joint_names)
            if n_joints != 8:
                print(f"[WARN] robot reports n_joints={n_joints}, but data is 8. Continuing anyway.")

            # optional profile
            try:
                v_deg = 120.0
                a_deg = 120.0
                vels_rad = [math.radians(v_deg)] * 8
                accs_rad = [math.radians(a_deg)] * 8
                robot.set_profile(vels_rad, accs_rad)
            except Exception:
                pass

            if GO_FIRST:
                q0_cmd = [float(x) for x in q[0].tolist()]
                h0_cmd = [clamp01(float(x)) for x in hand_q[0].tolist()]
                robot.set_positions(q0_cmd)
                hand.set_joint_pos(h0_cmd)
                time.sleep(0.2)

            wall_start = time.monotonic()
            next_wall = wall_start
            last_print = wall_start

            while True:
                now = time.monotonic()
                if now < next_wall:
                    time.sleep(next_wall - now)
                    now = next_wall

                elapsed = now - wall_start
                t_play = t0 + elapsed * float(speed)
                if t_play >= t_end:
                    break

                q_cmd = interp_linear_vec(t_play, t, q)
                h_cmd = interp_linear_vec(t_play, t, hand_q)

                # clamp hand to [0,1]
                h_cmd = np.clip(h_cmd, 0.0, 1.0)

                robot.set_positions([float(x) for x in q_cmd.tolist()])
                hand.set_joint_pos([clamp01(float(x)) for x in h_cmd.tolist()])

                next_wall += dt_wall

            # send final once
            robot.set_positions([float(x) for x in q[-1].tolist()])
            hand.set_joint_pos([clamp01(float(x)) for x in np.clip(hand_q[-1], 0.0, 1.0).tolist()])
            time.sleep(0.1)

            print("[REPLAY] Done.")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[REPLAY] Error: {e}")
        traceback.print_exc()
    finally:
        try:
            hand.set_joint_pos([0.0] * 6)
            time.sleep(0.15)
        except Exception:
            pass
        try:
            hand.close_can()
        except Exception:
            pass


if __name__ == "__main__":
    main()
