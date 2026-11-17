"""The agent loop: LLM call -> tool execution -> feedback, repeated until done.

Hand-written (no AgentExecutor/LangGraph) because each round must
  * inject fresh camera images after the world changed,
  * stop on TASK_ENV.is_episode_end() / step budget,
  * run tools sequentially (each one advances the simulator).
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from PIL import Image

from .backend import RoboDojoBackend
from .blackboard import Blackboard
from .llm import make_llm
from .prompts import SYSTEM_PROMPT
from .tools import build_tools

_CFG_PATH = Path(__file__).resolve().parent.parent / "deploy.yml"


def _load_cfg() -> dict:
    with open(_CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _png_b64(img: np.ndarray, max_side: int) -> str:
    im = Image.fromarray(np.asarray(img, dtype=np.uint8))
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _obs_message(backend: RoboDojoBackend, cameras: list[str], max_side: int, header: str) -> HumanMessage:
    content = [{"type": "text", "text": header + "\nrobot_state=" + json.dumps(backend.state_summary())}]
    for cam in cameras:
        content.append({"type": "text", "text": f"[{cam}]"})
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + _png_b64(backend.rgb(cam), max_side)}})
    return HumanMessage(content=content)


class _Logger:
    def __init__(self, log_dir: str | None, save_dir: str | None):
        root = log_dir or (os.path.join(save_dir, "agent_log") if save_dir else None)
        self.path = None
        if root:
            os.makedirs(root, exist_ok=True)
            self.path = os.path.join(root, f"trace_{int(time.time())}.jsonl")

    def write(self, rec: dict) -> None:
        print(f"[VLM_Agent] {json.dumps(rec, ensure_ascii=False)[:400]}")
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def run_episode(task_env, model_client) -> dict:
    cfg = _load_cfg()
    acfg, lcfg = cfg.get("agent", {}), cfg.get("llm", {})
    cameras = acfg.get("cameras", ["cam_head", "cam_left_wrist", "cam_right_wrist"])
    max_side = int(acfg.get("image_max_side", 480))
    max_rounds = int(acfg.get("max_rounds", 60))

    backend = RoboDojoBackend(task_env, model_client, cameras)
    bb = Blackboard()
    tools = build_tools(backend, bb)
    tool_map = {t.name: t for t in tools}
    llm = make_llm(lcfg).bind_tools(tools)
    log = _Logger(acfg.get("log_dir"), getattr(task_env, "save_dir", None))

    instruction = backend.instruction()
    messages = [SystemMessage(SYSTEM_PROMPT),
                _obs_message(backend, cameras, max_side, f"Instruction: {instruction}")]
    log.write({"event": "start", "instruction": instruction, "tools": list(tool_map)})

    finished_reason = None
    for rnd in range(max_rounds):
        if backend.episode_over():
            finished_reason = "episode_end"
            break

        ai: AIMessage = llm.invoke(messages)          # 1. LLM call
        messages.append(ai)
        log.write({"event": "llm", "round": rnd, "text": ai.content,
                   "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in ai.tool_calls]})

        if not ai.tool_calls:                         # model stopped calling tools -> nudge once
            messages.append(HumanMessage("Continue by calling a tool, or call `done`."))
            continue

        for tc in ai.tool_calls:                      # 2. execute tools (sequential)
            t = tool_map.get(tc["name"])
            if t is None:
                tool_msg = ToolMessage(content=json.dumps({"error": f"unknown tool {tc['name']}"}),
                                       tool_call_id=tc["id"])
            else:
                try:
                    tool_msg = t.invoke(tc)
                except Exception as e:  # keep the loop alive; let the model react
                    tool_msg = ToolMessage(content=json.dumps({"error": f"{type(e).__name__}: {e}"}),
                                           tool_call_id=tc["id"])
            messages.append(tool_msg)                 # 3. feed result back
            log.write({"event": "tool", "round": rnd, "name": tc["name"], "result": str(tool_msg.content)[:300]})
            if tc["name"] == "done":
                finished_reason = "done"
            if backend.episode_over():
                finished_reason = finished_reason or "episode_end"
        if finished_reason:
            break

        backend.refresh_obs()                         # 4. new images after the world changed
        messages.append(_obs_message(backend, cameras, max_side, f"Round {rnd + 1} observation"))
    else:
        finished_reason = "max_rounds"

    result = {"reason": finished_reason, "steps": backend.step_count, "success": backend.success()}
    log.write({"event": "end", **result})
    return result
