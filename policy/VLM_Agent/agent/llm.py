"""LLM factory. Any OpenAI-compatible endpoint works (vLLM, OpenRouter, OpenAI).

Precedence: env vars VLM_AGENT_MODEL / VLM_AGENT_BASE_URL / VLM_AGENT_API_KEY
override deploy.yml `llm:` block.
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def make_llm(cfg: dict):
    model = os.getenv("VLM_AGENT_MODEL", cfg.get("model"))
    base_url = os.getenv("VLM_AGENT_BASE_URL", cfg.get("base_url"))
    api_key = os.getenv("VLM_AGENT_API_KEY", cfg.get("api_key", "none"))
    extra_body = {}
    if cfg.get("enable_thinking") is not None:
        # Qwen3.x: thinking toggled per request via chat_template_kwargs
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(cfg["enable_thinking"])}
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=float(cfg.get("temperature", 0.0)),
        extra_body=extra_body or None,
        max_retries=3,
    )
