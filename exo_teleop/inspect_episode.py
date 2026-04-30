from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def resolve_episode_dir(path_arg: str | None, record_root: Path) -> Path:
    if path_arg:
        path = Path(path_arg).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Episode path not found: {path}")
        return path

    candidates = sorted(
        [p for p in record_root.glob("episode_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No episode directories found under {record_root}")
    return candidates[-1]


def load_episode(episode_dir: Path) -> dict[str, object]:
    data_path = episode_dir / "episode_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing episode data: {data_path}")

    data = np.load(data_path, allow_pickle=False)
    meta = {}
    summary = {}

    meta_path = episode_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    summary_path = episode_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "dir": episode_dir,
        "data": data,
        "meta": meta,
        "summary": summary,
    }


def print_summary(episode: dict[str, object]) -> None:
    episode_dir = episode["dir"]
    data = episode["data"]
    meta = episode["meta"]
    summary = episode["summary"]

    t = np.asarray(data["t"], dtype=np.float64)
    teleop = np.asarray(data["teleop"], dtype=np.float64)
    robot_q = np.asarray(data["robot_q_rad"], dtype=np.float64)
    robot_hand = np.asarray(data["robot_hand_q"], dtype=np.float64)

    print(f"Episode: {episode_dir}")
    print(f"Samples: {len(t)}")
    print(f"Duration: {float(t[-1]) if len(t) else 0.0:.3f} s")
    print(f"Teleop shape: {teleop.shape}")
    print(f"Robot q shape: {robot_q.shape}")
    print(f"Robot hand shape: {robot_hand.shape}")
    print(f"Camera frames: {len(np.asarray(data['camera_t']))}")

    if meta:
        print("Meta:")
        for key in ["created_at", "loop_hz", "head_deg", "vision_filtered", "robot_arm_dev", "robot_hand_can"]:
            if key in meta:
                print(f"  {key}: {meta[key]}")

    if summary:
        print("Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")


def make_depth_vis(depth_mm: np.ndarray) -> np.ndarray:
    depth_valid = np.nan_to_num(depth_mm.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth_norm = cv2.normalize(depth_valid, None, 0, 255, cv2.NORM_MINMAX)
    depth_u8 = depth_norm.astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def export_video(episode_dir: Path, output_path: Path, fps: float) -> None:
    rgb_dir = episode_dir / "rgb"
    depth_dir = episode_dir / "depth"
    rgb_paths = sorted(rgb_dir.glob("*.jpg"))
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")

    first_rgb = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise RuntimeError(f"Failed to read first RGB frame: {rgb_paths[0]}")

    h, w = first_rgb.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w * 2, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for rgb_path in rgb_paths:
            depth_path = depth_dir / f"{rgb_path.stem}.png"
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                continue

            depth_vis = make_depth_vis(depth)
            frame = np.hstack([rgb, depth_vis])
            cv2.putText(
                frame,
                f"frame {rgb_path.stem}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    print(f"Exported video: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a teleop_real recorded episode.")
    parser.add_argument("episode_dir", nargs="?", help="Episode directory. Defaults to the latest one.")
    parser.add_argument("--record-root", default="teleop_records", help="Root directory containing episodes.")
    parser.add_argument("--export-video", help="Optional MP4 path for exporting RGB+Depth video.")
    parser.add_argument("--fps", type=float, default=15.0, help="FPS for exported video.")
    args = parser.parse_args()

    record_root = Path(args.record_root).expanduser()
    episode_dir = resolve_episode_dir(args.episode_dir, record_root)
    episode = load_episode(episode_dir)
    print_summary(episode)

    if args.export_video:
        export_video(episode_dir, Path(args.export_video).expanduser(), args.fps)


if __name__ == "__main__":
    main()
