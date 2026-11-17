"""Policy-server side of VLM_Agent.

The ws server dispatches `model_client.call(func_name=X, **kw)` to `Model.X(**kw)`,
so any method defined here becomes a remote tool the client-side agent can call.
Use it to host GPU-heavy perception (SAM3, Grounding DINO) in the server venv
while the agent + Isaac Sim stay in the RoboDojo env.

The standard policy interface (update_obs / get_action) is a no-op: the agent
never asks this server for actions.
"""
from __future__ import annotations

import numpy as np

from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self._sam3 = None  # lazy
        print("[VLM_Agent/server] ready (tool host only)")

    # ---- standard interface (unused) -------------------------------------
    def update_obs(self, obs):
        pass

    def update_obs_batch(self, obs_list):
        pass

    def get_action(self, **kwargs):
        raise RuntimeError("VLM_Agent decides client-side; get_action is not used")

    def get_action_batch(self, env_idx_list=None, **kwargs):
        raise RuntimeError("VLM_Agent decides client-side; get_action_batch is not used")

    def reset(self):
        pass

    # ---- remote tools -----------------------------------------------------
    def ping(self, **kwargs):
        return {"ok": True}

    def segment_text(self, image: np.ndarray, prompt: str, **kwargs):
        """Text-prompted instance segmentation (open-robot-skills `sam3.segment_text`).

        TODO: load the sam3 bundle here (server venv must have it installed):
            from gap_skills.tools.sam3.tools import segment_text
            return segment_text(image=image, prompt=prompt)
        Expected return: {"masks": uint8[N,H,W], "scores": [...], "boxes": [[x0,y0,x1,y1], ...]}
        """
        raise NotImplementedError("sam3 not wired yet (see model.py:segment_text)")
