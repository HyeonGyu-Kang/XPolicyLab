"""LangChain tools the VLM can call.

Three groups:
  robot.*   -> RoboDojoBackend (moves the sim, costs steps)
  percept.* -> open-robot-skills tool bundles (sam3 via policy server,
               geometry in-process) wrapped so arrays go to the Blackboard
  control   -> done

Every tool returns a short string/JSON; arrays never reach the LLM.
Add a new open-robot-skills tool by following `segment_text` / `mask_to_obb`.
"""
from __future__ import annotations

import json
from typing import Literal

import numpy as np
from langchain_core.tools import tool

from .backend import ARMS, RoboDojoBackend
from .blackboard import Blackboard

Arm = Literal["left", "right"]
Camera = Literal["cam_head", "cam_left_wrist", "cam_right_wrist"]


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def build_tools(backend: RoboDojoBackend, bb: Blackboard) -> list:
    # ------------------------------------------------------------------ robot
    @tool
    def get_robot_state() -> str:
        """Return both arms' end-effector world pose, gripper opening, and the step budget used."""
        return _j(backend.state_summary())

    @tool
    def move_ee(arm: Arm, x: float, y: float, z: float,
                qw: float = 0.0, qx: float = 1.0, qy: float = 0.0, qz: float = 0.0,
                settle_steps: int = 1) -> str:
        """Move one arm's end-effector to a world-frame pose [x,y,z] with quaternion (qw,qx,qy,qz).
        Default quaternion (0,1,0,0) points the gripper straight down. The other arm holds still.
        IK is solved by the simulator; if unreachable the arm does not move (check position_error_m)."""
        return _j(backend.go_to_pose(arm, [x, y, z, qw, qx, qy, qz], settle_steps=settle_steps))

    @tool
    def move_ee_relative(arm: Arm, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                         settle_steps: int = 1) -> str:
        """Translate one arm's end-effector by (dx,dy,dz) metres in world frame, keeping orientation."""
        p = backend.ee_pose(arm).copy()
        p[:3] += np.asarray([dx, dy, dz], dtype=np.float32)
        return _j(backend.go_to_pose(arm, p.tolist(), settle_steps=settle_steps))

    @tool
    def open_gripper(arm: Arm) -> str:
        """Fully open one arm's gripper (takes ~10 steps)."""
        return _j(backend.set_gripper(arm, 1.0))

    @tool
    def close_gripper(arm: Arm) -> str:
        """Fully close one arm's gripper to grasp (takes ~10 steps)."""
        return _j(backend.set_gripper(arm, 0.0))

    @tool
    def wait(steps: int = 5) -> str:
        """Hold all commands and let the simulation settle for `steps` steps."""
        backend.wait_steps(int(steps))
        return _j({"waited": int(steps), "step": backend.step_count})

    # -------------------------------------------------------------- percept
    @tool
    def segment_text(camera: Camera, prompt: str) -> str:
        """Find object instances matching a text prompt in a camera image (SAM3).
        Returns mask ids, pixel bounding boxes [x0,y0,x1,y1] and scores; use a mask id with mask_to_obb."""
        img = backend.rgb(camera)
        res = backend.remote("segment_text", image=img, prompt=prompt)  # hosted in model.py
        out = []
        for i in range(len(res["scores"])):
            mid = bb.put("mask", np.asarray(res["masks"][i]), meta={"camera": camera, "prompt": prompt})
            out.append({"mask_id": mid, "box": [int(v) for v in res["boxes"][i]],
                        "score": round(float(res["scores"][i]), 3)})
        return _j({"camera": camera, "instances": out})

    @tool
    def mask_to_obb(mask_id: str) -> str:
        """Back-project a mask through depth into a world-frame oriented bounding box
        (center xyz, extent xyz, rotation quaternion). Requires depth + calibration enabled in env_cfg."""
        meta = bb.meta(mask_id)
        cam = meta["camera"]
        depth, calib = backend.depth(cam), backend.camera_calib(cam)
        if depth is None or calib is None:
            return _j({"error": "depth/calibration not in obs; enable in env_cfg/arx_x5.yml"})
        # open-robot-skills geometry bundle (CPU). Install graph-as-policy or vendor the bundle.
        from gap_skills.tools.geometry.tools import filter_and_compute_obb, mask_to_world_points  # noqa
        cloud = mask_to_world_points(mask=bb.get(mask_id), depth=depth,
                                     intrinsics=calib["K"], camera_pose=_extrinsic_to_pose(calib["extrinsic"]))
        obb = filter_and_compute_obb(points=cloud["points"])
        cid = bb.put("cloud", cloud["points"], meta={"mask_id": mask_id})
        return _j({"cloud_id": cid, "obb": obb})

    @tool
    def list_blackboard() -> str:
        """List ids of masks/clouds stored so far."""
        return _j(bb.keys())

    # -------------------------------------------------------------- control
    @tool
    def done(reason: str) -> str:
        """Declare the episode finished (task satisfied, or impossible). Give the reason."""
        return _j({"done": True, "reason": reason})

    return [get_robot_state, move_ee, move_ee_relative, open_gripper, close_gripper, wait,
            segment_text, mask_to_obb, list_blackboard, done]


def _extrinsic_to_pose(ext: np.ndarray) -> dict:
    """4x4 camera->world matrix -> gap Se3Pose dict. TODO: verify RoboDojo's extrinsic convention."""
    from scipy.spatial.transform import Rotation as R
    ext = np.asarray(ext, dtype=np.float64).reshape(4, 4)
    q = R.from_matrix(ext[:3, :3]).as_quat()  # xyzw
    return {"position": {"x": ext[0, 3], "y": ext[1, 3], "z": ext[2, 3]},
            "rotation": {"w": q[3], "x": q[0], "y": q[1], "z": q[2]}}
