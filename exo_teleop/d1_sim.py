import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from isaacgym import gymapi
from isaacgym import gymutil

from beingbeyond_d1_sdk.urdf_path import get_default_urdf_path


@dataclass(frozen=True)
class AssetCfg:
    asset_path: str = get_default_urdf_path()
    fix_base_link: bool = True
    flip_visual_attachments: bool = False
    armature: float = 0.01


@dataclass(frozen=True)
class SimCfg:
    dt: float = 1.0 / 60.0
    substeps: int = 2
    up_axis: int = gymapi.UP_AXIS_Z
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.8)
    use_gpu_pipeline: bool = False


@dataclass(frozen=True)
class DofDriveCfg:
    stiffness: float = 400.0
    damping: float = 40.0
    drive_mode: int = gymapi.DOF_MODE_POS


@dataclass(frozen=True)
class ViewerCfg:
    cam_pos: Tuple[float, float, float] = (1.4, 1.2, 1.0)
    cam_target: Tuple[float, float, float] = (0.0, 0.0, 0.45)


@dataclass(frozen=True)
class TeleopCfg:
    head_dim: int = 2
    arm_dim: int = 6
    hand_dim: int = 6

    head_joint_names: Tuple[str, ...] = ()
    arm_joint_names: Tuple[str, ...] = ()
    hand_active_joint_names: Tuple[str, ...] = (
        "thumb_cmc_pitch",
        "thumb_cmc_yaw",
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    )

    default_head_rad: Tuple[float, float] = (0.0, 0.0)
    default_arm_rad: Tuple[float, float, float, float, float, float] = (
        0.0, -1.57079632679, 1.57079632679, 0.0, 0.0, 0.0
    )
    default_hand_norm: Tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )


def _findall_ns_agnostic(root: ET.Element, tag: str) -> List[ET.Element]:
    return root.findall(f".//{tag}") + root.findall(f".//{{*}}{tag}")


def _clip(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


class D1IsaacSimulator:
    def __init__(
        self,
        asset_cfg: Optional[AssetCfg] = None,
        sim_cfg: Optional[SimCfg] = None,
        dof_cfg: Optional[DofDriveCfg] = None,
        viewer_cfg: Optional[ViewerCfg] = None,
        teleop_cfg: Optional[TeleopCfg] = None,
        args=None,
    ) -> None:
        self.asset_cfg = asset_cfg or AssetCfg()
        self.sim_cfg = sim_cfg or SimCfg()
        self.dof_cfg = dof_cfg or DofDriveCfg()
        self.viewer_cfg = viewer_cfg or ViewerCfg()
        self.teleop_cfg = teleop_cfg or TeleopCfg()

        if not os.path.isfile(self.asset_cfg.asset_path):
            raise FileNotFoundError(f"URDF not found: {self.asset_cfg.asset_path}")

        self.gym = gymapi.acquire_gym()
        self.args = args if args is not None else gymutil.parse_arguments(
            description="D1 Isaac Gym Teleop Simulator"
        )

        self.sim = self._create_sim()
        self.viewer = self._create_viewer()
        self._add_ground()

        self.asset, self.urdf_path = self._load_asset()
        self.env, self.actor = self._create_env_and_actor()
        self._configure_dofs()
        self._set_viewer_camera()

        self.joint_names: List[str] = list(self.gym.get_actor_dof_names(self.env, self.actor))
        self.name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(self.joint_names)}
        self.num_dofs: int = self.gym.get_actor_dof_count(self.env, self.actor)

        self.dof_state_buf = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_ALL)
        self.joint_lower = self.dof_props["lower"].astype(np.float64).copy()
        self.joint_upper = self.dof_props["upper"].astype(np.float64).copy()

        self.mimic_child_to_parent_idx: Dict[int, Tuple[int, float, float]] = {}
        self.mimic_parent_to_children_idx: Dict[int, List[int]] = {}

        self._parse_urdf_mimic()
        self._resolve_teleop_slots()

        self.current_targets = np.zeros(self.num_dofs, dtype=np.float32)
        self.reset()

    def _create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = self.sim_cfg.dt
        sim_params.substeps = self.sim_cfg.substeps
        sim_params.up_axis = self.sim_cfg.up_axis
        sim_params.gravity = gymapi.Vec3(*self.sim_cfg.gravity)

        if self.args.physics_engine == gymapi.SIM_FLEX:
            sim_params.flex.solver_type = 5
            sim_params.flex.num_outer_iterations = 4
            sim_params.flex.num_inner_iterations = 15
            sim_params.flex.relaxation = 0.75
            sim_params.flex.warm_start = 0.8
        elif self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu

        sim_params.use_gpu_pipeline = self.sim_cfg.use_gpu_pipeline
        sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            self.args.physics_engine,
            sim_params,
        )
        if sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")
        return sim

    def _create_viewer(self):
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer")
        return viewer

    def _add_ground(self) -> None:
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _load_asset(self):
        asset_root = os.path.dirname(self.asset_cfg.asset_path)
        asset_file = os.path.basename(self.asset_cfg.asset_path)

        opt = gymapi.AssetOptions()
        opt.fix_base_link = self.asset_cfg.fix_base_link
        opt.flip_visual_attachments = self.asset_cfg.flip_visual_attachments
        opt.armature = self.asset_cfg.armature

        asset = self.gym.load_asset(self.sim, asset_root, asset_file, opt)
        return asset, self.asset_cfg.asset_path

    def _create_env_and_actor(self):
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        env = self.gym.create_env(self.sim, env_lower, env_upper, 1)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        actor = self.gym.create_actor(env, self.asset, pose, "d1", 0, 1)
        return env, actor

    def _configure_dofs(self) -> None:
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)
        for i in range(len(self.dof_props)):
            self.dof_props["driveMode"][i] = self.dof_cfg.drive_mode
            self.dof_props["stiffness"][i] = self.dof_cfg.stiffness
            self.dof_props["damping"][i] = self.dof_cfg.damping
        self.gym.set_actor_dof_properties(self.env, self.actor, self.dof_props)

    def _set_viewer_camera(self) -> None:
        cam_pos = gymapi.Vec3(*self.viewer_cfg.cam_pos)
        cam_target = gymapi.Vec3(*self.viewer_cfg.cam_target)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _parse_urdf_mimic(self) -> None:
        try:
            root = ET.parse(self.urdf_path).getroot()
            joints = _findall_ns_agnostic(root, "joint")
            for j in joints:
                child_name = j.get("name")
                if not child_name:
                    continue

                mimics = _findall_ns_agnostic(j, "mimic")
                if not mimics:
                    continue

                m = mimics[0]
                parent_name = m.get("joint")
                if not parent_name:
                    continue

                if child_name not in self.name_to_idx or parent_name not in self.name_to_idx:
                    continue

                mul = float(m.get("multiplier", "1.0"))
                off = float(m.get("offset", "0.0"))

                ci = self.name_to_idx[child_name]
                pi = self.name_to_idx[parent_name]
                self.mimic_child_to_parent_idx[ci] = (pi, mul, off)
                self.mimic_parent_to_children_idx.setdefault(pi, []).append(ci)
        except Exception as e:
            print(f"[warn] URDF mimic parse failed: {e}")

    def _resolve_name_list(self, names: Sequence[str]) -> List[int]:
        out = []
        for n in names:
            if n not in self.name_to_idx:
                raise KeyError(f"Joint name not found in actor DOFs: {n}")
            out.append(self.name_to_idx[n])
        return out

    def _is_mimic_child(self, idx: int) -> bool:
        return idx in self.mimic_child_to_parent_idx

    def _is_hand_like_joint_name(self, name: str) -> bool:
        name_l = name.lower()
        hand_keywords = [
            "thumb", "index", "middle", "ring", "little", "pinky",
            "finger", "hand", "gripper"
        ]
        return any(k in name_l for k in hand_keywords)

    def _resolve_teleop_slots(self) -> None:
        cfg = self.teleop_cfg
        non_mimic_indices = [i for i in range(len(self.joint_names)) if not self._is_mimic_child(i)]

        if len(cfg.head_joint_names) > 0:
            self.head_indices = self._resolve_name_list(cfg.head_joint_names)
        else:
            self.head_indices = non_mimic_indices[: cfg.head_dim]

        if len(cfg.arm_joint_names) > 0:
            self.arm_indices = self._resolve_name_list(cfg.arm_joint_names)
        else:
            start = len(self.head_indices)
            end = start + cfg.arm_dim
            self.arm_indices = non_mimic_indices[start:end]

        if len(cfg.hand_active_joint_names) > 0:
            self.hand_active_indices = self._resolve_name_list(cfg.hand_active_joint_names)
        else:
            used = set(self.head_indices + self.arm_indices)
            remaining = [i for i in non_mimic_indices if i not in used]
            hand_candidates = [i for i in remaining if self._is_hand_like_joint_name(self.joint_names[i])]
            if len(hand_candidates) < cfg.hand_dim:
                for i in remaining:
                    if i not in hand_candidates:
                        hand_candidates.append(i)
                    if len(hand_candidates) >= cfg.hand_dim:
                        break
            self.hand_active_indices = hand_candidates[: cfg.hand_dim]

        if len(self.head_indices) != cfg.head_dim:
            raise RuntimeError(f"Head slot dim mismatch: expected {cfg.head_dim}, got {len(self.head_indices)}")
        if len(self.arm_indices) != cfg.arm_dim:
            raise RuntimeError(f"Arm slot dim mismatch: expected {cfg.arm_dim}, got {len(self.arm_indices)}")
        if len(self.hand_active_indices) != cfg.hand_dim:
            raise RuntimeError(f"Hand slot dim mismatch: expected {cfg.hand_dim}, got {len(self.hand_active_indices)}")

        self.teleop_dim = cfg.head_dim + cfg.arm_dim + cfg.hand_dim

    def _apply_all_mimic(self, q: np.ndarray) -> np.ndarray:
        out = q.copy()
        for _ in range(2):
            for child_idx, (parent_idx, mul, off) in self.mimic_child_to_parent_idx.items():
                v = out[parent_idx] * mul + off
                out[child_idx] = np.clip(v, self.joint_lower[child_idx], self.joint_upper[child_idx])
        return out

    def _clip_full_targets(self, q: np.ndarray) -> np.ndarray:
        return _clip(q, self.joint_lower, self.joint_upper)

    def _map_hand_norm_to_targets(self, hand: Sequence[float]) -> np.ndarray:
        hand = np.asarray(hand, dtype=np.float64).reshape(self.teleop_cfg.hand_dim)
        out = np.zeros(self.teleop_cfg.hand_dim, dtype=np.float64)
        for k, idx in enumerate(self.hand_active_indices):
            lo = self.joint_lower[idx]
            hi = self.joint_upper[idx]
            u = float(np.clip(hand[k], 0.0, 1.0))
            out[k] = lo + u * (hi - lo)
        return out

    def build_default_targets(self) -> np.ndarray:
        cfg = self.teleop_cfg
        q = np.zeros(self.num_dofs, dtype=np.float64)
        q = self._clip_full_targets(q)

        q[self.head_indices] = np.asarray(cfg.default_head_rad, dtype=np.float64)
        q[self.arm_indices] = np.asarray(cfg.default_arm_rad, dtype=np.float64)
        q[self.hand_active_indices] = self._map_hand_norm_to_targets(cfg.default_hand_norm)

        q = self._apply_all_mimic(q)
        q = self._clip_full_targets(q)
        return q

    def reset(self) -> None:
        q = self.build_default_targets()
        self.current_targets[:] = q.astype(np.float32)

        self.dof_state_buf["pos"][:] = self.current_targets
        self.dof_state_buf["vel"][:] = 0.0

        self.gym.set_actor_dof_states(self.env, self.actor, self.dof_state_buf, gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, self.actor, self.current_targets)

    def set_teleop_targets(
        self,
        head: Sequence[float],
        arm: Sequence[float],
        hand: Sequence[float],
    ) -> None:
        q = self.current_targets.astype(np.float64).copy()

        head = np.asarray(head, dtype=np.float64).reshape(self.teleop_cfg.head_dim)
        arm = np.asarray(arm, dtype=np.float64).reshape(self.teleop_cfg.arm_dim)
        hand = np.asarray(hand, dtype=np.float64).reshape(self.teleop_cfg.hand_dim)

        for k, idx in enumerate(self.head_indices):
            q[idx] = np.clip(head[k], self.joint_lower[idx], self.joint_upper[idx])

        for k, idx in enumerate(self.arm_indices):
            q[idx] = np.clip(arm[k], self.joint_lower[idx], self.joint_upper[idx])

        hand_targets = self._map_hand_norm_to_targets(hand)
        for k, idx in enumerate(self.hand_active_indices):
            q[idx] = hand_targets[k]

        q = self._apply_all_mimic(q)
        q = self._clip_full_targets(q)

        self.current_targets[:] = q.astype(np.float32)
        self.gym.set_actor_dof_position_targets(self.env, self.actor, self.current_targets)

    def apply_teleop_vector(self, x: Sequence[float]) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(self.teleop_dim)
        h = self.teleop_cfg.head_dim
        a = self.teleop_cfg.arm_dim
        head = x[:h]
        arm = x[h:h + a]
        hand = x[h + a:h + a + self.teleop_cfg.hand_dim]
        self.set_teleop_targets(head=head, arm=arm, hand=hand)

    def step(self, sync_time: bool = True) -> bool:
        if self.gym.query_viewer_has_closed(self.viewer):
            return False

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        if sync_time:
            self.gym.sync_frame_time(self.sim)
        return True

    def get_joint_positions(self) -> np.ndarray:
        dof_states = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_POS)
        return dof_states["pos"].copy()

    def get_joint_velocities(self) -> np.ndarray:
        dof_states = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_VEL)
        return dof_states["vel"].copy()

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "q": self.get_joint_positions(),
            "dq": self.get_joint_velocities(),
            "targets": self.current_targets.copy(),
            "head_targets": self.current_targets[self.head_indices].copy(),
            "arm_targets": self.current_targets[self.arm_indices].copy(),
            "hand_targets": self.current_targets[self.hand_active_indices].copy(),
        }

    def close(self) -> None:
        try:
            if self.viewer is not None:
                self.gym.destroy_viewer(self.viewer)
        finally:
            if self.sim is not None:
                self.gym.destroy_sim(self.sim)