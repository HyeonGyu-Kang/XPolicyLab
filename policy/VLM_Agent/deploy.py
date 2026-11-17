"""RoboDojo entry point for the VLM tool-calling agent.

`eval_one_episode` is what `src/eval_client/eval_env.py::run_eval` imports and
calls. Unlike learned policies, the decision loop lives HERE (client side), so
tools can interleave observe/act inside one decision. `model_client` is kept
only as a channel to heavy perception tools hosted in the policy server.
"""
from XPolicyLab.policy.VLM_Agent.agent.loop import run_episode


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    run_episode(TASK_ENV, model_client)


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("VLM_Agent is single-env only; set eval_batch: false in deploy.yml")
