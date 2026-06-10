"""Configuration management - environment variables with defaults."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QAWikiConfig:
    # ---- LLM ----
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_endpoint: str = ""
    llm_api_key_2: str = ""
    llm_endpoint_2: str = ""

    # ---- Embedding ----
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_endpoint: str = ""

    # ---- Knowledge Base ----
    kb_dir: str = "./knowledge_bank"

    # ---- Phase I: Distillation ----
    rollouts_per_sample: int = 4
    insight_max_ops: int = 3
    insight_max_items: int = 500
    insight_similarity_threshold: float = 0.70
    large_batch_size: int = 8

    # ---- Phase II: Retrieval ----
    retrieval_top_k: int = 5
    retrieval_min_similarity: float = 0.0
    retrieval_enable_decomposition: bool = True
    retrieval_enable_rewrite: bool = True

    # ---- Framework ----
    framework_word_threshold: int = 3000

    # ---- Runtime ----
    num_workers: int = 4
    max_turns: int = 20
    temperature: float = 0.6
    top_p: float = 1.0

    # ---- Tools (optional) ----
    serpapi_key: str = ""
    jina_api_key: str = ""


def load_config(env_prefix: str = "QAWIKI_") -> QAWikiConfig:
    """Load configuration from environment variables.

    Environment variable mapping:
      QAWIKI_LLM_MODEL          -> llm_model
      QAWIKI_LLM_API_KEY        -> llm_api_key
      QAWIKI_LLM_ENDPOINT       -> llm_endpoint
      QAWIKI_KB_DIR             -> kb_dir
      ...and so on for all fields.
    """
    cfg = QAWikiConfig()

    field_map = {
        "llm_model": "LLM_MODEL",
        "llm_api_key": "LLM_API_KEY",
        "llm_endpoint": "LLM_ENDPOINT",
        "llm_api_key_2": "LLM_API_KEY_2",
        "llm_endpoint_2": "LLM_ENDPOINT_2",
        "embedding_model": "EMBEDDING_MODEL",
        "embedding_api_key": "EMBEDDING_API_KEY",
        "embedding_endpoint": "EMBEDDING_ENDPOINT",
        "kb_dir": "KB_DIR",
        "rollouts_per_sample": "ROLLOUTS_PER_SAMPLE",
        "insight_max_ops": "INSIGHT_MAX_OPS",
        "insight_max_items": "INSIGHT_MAX_ITEMS",
        "large_batch_size": "LARGE_BATCH_SIZE",
        "retrieval_top_k": "RETRIEVAL_TOP_K",
        "num_workers": "NUM_WORKERS",
        "max_turns": "MAX_TURNS",
        "temperature": "TEMPERATURE",
        "top_p": "TOP_P",
        "serpapi_key": "SERPAPI_KEY",
        "jina_api_key": "JINA_API_KEY",
    }

    for attr, env_key in field_map.items():
        env_val = os.environ.get(f"{env_prefix}{env_key}")
        if env_val is not None:
            field_type = type(getattr(cfg, attr))
            if field_type == bool:
                setattr(cfg, attr, env_val.lower() in ("true", "1", "yes"))
            elif field_type == int:
                setattr(cfg, attr, int(env_val))
            elif field_type == float:
                setattr(cfg, attr, float(env_val))
            else:
                setattr(cfg, attr, env_val)

    # Fallback: embedding api key from llm api key
    if not cfg.embedding_api_key:
        cfg.embedding_api_key = cfg.llm_api_key
    if not cfg.embedding_endpoint:
        cfg.embedding_endpoint = cfg.llm_endpoint

    # Also try OPENAI_* as fallback
    if not cfg.llm_api_key:
        cfg.llm_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not cfg.llm_endpoint and os.environ.get("OPENAI_BASE_URL"):
        cfg.llm_endpoint = os.environ["OPENAI_BASE_URL"]

    return cfg


def print_config(cfg: QAWikiConfig) -> None:
    """Pretty-print configuration (masking secrets)."""
    mask = lambda s: s[:8] + "..." if len(s) > 10 else "***" if s else "(unset)"
    print(f"LLM Model:      {cfg.llm_model}")
    print(f"LLM API Key:    {mask(cfg.llm_api_key)}")
    print(f"LLM Endpoint:   {cfg.llm_endpoint or '(default)'}")
    print(f"Embedding:      {cfg.embedding_model}")
    print(f"KB Directory:   {cfg.kb_dir}")
    print(f"Rollouts:       {cfg.rollouts_per_sample}")
    print(f"Workers:        {cfg.num_workers}")
    print(f"Insight Limit:  {cfg.insight_max_items}")
    print(f"Retrieval K:    {cfg.retrieval_top_k}")
