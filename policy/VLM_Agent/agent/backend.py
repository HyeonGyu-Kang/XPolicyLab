"""RoboDojoBackend: the `robot.*` / `sim.*` layer on top of TASK_ENV.

Everything the tools need from the simulator goes through here, so tools.py
never touches TASK_ENV directly. Each robot-moving call costs one or more of
the episode's `step_lim` (800) steps.

Action contract (ee mode, dual arm) — EVERY take_action must carry all four keys:
    left_ee_pose  [x y z qw qx qy qz]  world frame (sim solves IK via cuRobo)
    left_ee_joint_state  [g]           gripper 0=closed .. 1=open (normalised)
    right_ee_pose, right_ee_joint_state
so the backend keeps the last commanded pose/gripper per arm and resends it.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

ARMS = ("left", "right")


class RoboDojoBackend:
    def __init__(self, task_env, model_client, cameras: list[str]):
        self.env = task_env
        self.client = model_client
        self.cameras = cameras
        self.obs: dict[str, Any] | None = None
        self._cmd_pose: dict[str, np.ndarray] = {}
        self._cmd_grip: dict[str, float] = {}
        self.refresh_obs()
        # seed commanded state from the real state so the first action is a hold
        for arm in ARMS:
            self._cmd_pose[arm] = np.asarray(self.obs["state"][f"{arm}_ee_pose"], dtype=np.float32)
            self._cmd_grip[arm] = float(np.asarray(self.obs["state"][f"{arm}_ee_joint_state"]).reshape(-1)[0])

    # ---- observation ------------------------------------------------------
    def refresh_obs(self) -> dict[str, Any]:
        self.obs = self.env.get_obs()
        return self.obs

    def rgb(self, camera: str) -> np.ndarray:
        """uint8 [H, W, 3] RGB from the latest obs."""
        return np.asarray(self.obs["vision"][camera]["color"])

    def depth(self, camera: str) -> np.ndarray | None:
        """float32 [H, W] metres, or None if depth capture is disabled in env_cfg.

        Enable in env_cfg/arx_x5.yml (observation.vision.depth: true) and the
        per-camera distance_to_image_plane_capture in env_cfg/camera/camera_config.yml.
        """
        cam = self.obs["vision"][camera]
        return None if "depth" not in cam else np.asarray(cam["depth"], dtype=np.float32)

    def camera_calib(self, camera: str) -> dict[str, np.ndarray] | None:
        cam = self.obs["vision"][camera]
        if "intrinsic_matrix" not in cam:
            return None  # enable intrinsic_matrix/extrinsic_matrix in env_cfg
        return {"K": np.asarray(cam["intrinsic_matrix"]), "extrinsic": np.asarray(cam["extrinsic_matrix"])}

    def instruction(self) -> str:
        return str(self.obs.get("instruction", ""))

    def ee_pose(self, arm: str) -> np.ndarray:
        return np.asarray(self.obs["state"][f"{arm}_ee_pose"], dtype=np.float32)

    def gripper(self, arm: str) -> float:
        return float(np.asarray(self.obs["state"][f"{arm}_ee_joint_state"]).reshape(-1)[0])

    def state_summary(self) -> dict[str, Any]:
        out = {}
        for arm in ARMS:
            p = self.ee_pose(arm)
            out[arm] = {"ee_xyz": [round(float(v), 4) for v in p[:3]],
                        "ee_quat_wxyz": [round(float(v), 4) for v in p[3:]],
                        "gripper": round(self.gripper(arm), 3)}
        out["step"] = self.step_count
        out["step_limit"] = self.step_limit
        return out

    # ---- episode bookkeeping ---------------------------------------------
    @property
    def step_count(self) -> int:
        return int(self.env.take_action_cnt[0])

    @property
    def step_limit(self) -> int:
        return int(self.env.step_lim)

    def episode_over(self) -> bool:
        return bool(self.env.is_episode_end())

    def success(self) -> bool:
        return bool(self.env.success[0])

    # ---- actuation ---------------------------------------------------------
    def _action(self) -> dict[str, np.ndarray]:
        a = {}
        for arm in ARMS:
            a[f"{arm}_ee_pose"] = self._cmd_pose[arm].astype(np.float32)
            a[f"{arm}_ee_joint_state"] = np.asarray([self._cmd_grip[arm]], dtype=np.float32)
        return a

    def _step(self, n: int = 1) -> None:
        for _ in range(n):
            if self.episode_over():
                return
            self.env.take_action(self._action())
        self.refresh_obs()

    def go_to_pose(self, arm: str, pose_wxyz: list[float], settle_steps: int = 1) -> dict[str, Any]:
        """Command one arm's EE to a world-frame pose; the other arm holds."""
        self._cmd_pose[arm] = np.asarray(pose_wxyz, dtype=np.float32).reshape(7)
        self._step(settle_steps)
        reached = self.ee_pose(arm)
        err = float(np.linalg.norm(reached[:3] - self._cmd_pose[arm][:3]))
        return {"arm": arm, "target_xyz": [float(v) for v in self._cmd_pose[arm][:3]],
                "reached_xyz": [float(v) for v in reached[:3]], "position_error_m": round(err, 4)}

    def set_gripper(self, arm: str, value: float, settle_steps: int = 10) -> dict[str, Any]:
        self._cmd_grip[arm] = float(np.clip(value, 0.0, 1.0))
        self._step(settle_steps)
        return {"arm": arm, "commanded": self._cmd_grip[arm], "measured": self.gripper(arm)}

    def wait_steps(self, n: int) -> None:
        self._step(n)

    # ---- sim ground truth (NOT benchmark-legal; debugging only) -----------
    def object_names(self) -> list[str]:
        objs = self.env.scene_manager.get_objects(env_ids=[0])
        return sorted(objs.keys())

    # ---- remote tools hosted in the policy server ---------------------------
    def remote(self, func_name: str, **kwargs) -> Any:
        return self.client.call(func_name=func_name, **kwargs)
