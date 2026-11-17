# VLM_Agent — LangChain tool-calling VLM policy for RoboDojo

The decision loop runs **client-side** in `deploy.py` (same process as Isaac Sim).
The ws policy server (`model.py`) is only a host for heavy perception tools.

```
deploy.py            eval_one_episode -> agent/loop.py::run_episode
model.py             server: no-op policy + remote tools (segment_text, ...)
agent/backend.py     robot.*/sim.* on TASK_ENV (ee-mode actions, obs access, step budget)
agent/blackboard.py  id-keyed store for masks/clouds the LLM must not see raw
agent/tools.py       LangChain @tool wrappers (robot, open-robot-skills, done)
agent/loop.py        LLM call -> tool exec -> feedback loop, image injection each round
agent/llm.py         ChatOpenAI against any OpenAI-compatible endpoint (vLLM)
agent/prompts.py     system prompt
```

## Setup
```bash
conda activate RoboDojo
# Isaac Sim 5.1 pins typing_extensions==4.12.2 / websockets==12.0 / packaging==23.0;
# LangChain 1.x needs typing_extensions>=4.13, so stay on the 0.3 line under constraints:
printf 'typing_extensions==4.12.2\nwebsockets==12.0\npackaging==23.0\n' > /tmp/isaac-constraints.txt
pip install -c /tmp/isaac-constraints.txt "langchain-core<1" "langchain-openai<1"
# (installed 2026-11-16: langchain-core 0.3.86, langchain-openai 0.3.35, openai 2.28)
# VLM server (separate venv / docker, vLLM >= 0.17):
vllm serve Qwen/Qwen3.8-27B --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 --mm-encoder-tp-mode data --max-model-len 32768 --port 8001
```
Optional: `VLM_AGENT_MODEL`, `VLM_AGENT_BASE_URL`, `VLM_AGENT_API_KEY` override `deploy.yml:llm`.

## Run
```bash
source ~/miniforge3/etc/profile.d/conda.sh
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/VLM_Agent --task stack_bowls \
    --ckpt none --action-type ee --seed 0 --policy-env RoboDojo --policy-gpu 1 --env-gpu 0 --eval-num 1
```
Trace: `<save_dir>/agent_log/trace_*.jsonl`.

## Not wired yet (TODO)
- `model.py::segment_text` — load the open-robot-skills `sam3` bundle in the server env.
- `tools.py::mask_to_obb` — needs depth + intrinsic/extrinsic enabled in `env_cfg/arx_x5.yml`
  and `env_cfg/camera/camera_config.yml`, plus the `geometry` bundle importable
  (`pip install graph-as-policy` or vendor `tools/geometry`). Verify extrinsic convention.
- Grasp candidate / planner tools (`geometry.top_down_grasp_candidates`, cuRobo) — add
  following the same wrapper pattern.
