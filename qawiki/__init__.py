"""QAWiki - Knowledge distillation and retrieval framework for QA agents.

Inspired by XSkill (Jiang et al., ICML 2026, MIT License)
https://github.com/XSkill-Agent/XSkill
"""

from .core import (
    KnowledgeLLM,
    KnowledgeRetriever,
    batch_merge_insights,
    refine_insight_library,
    load_insight_library,
    save_insight_library,
    generate_framework,
    merge_frameworks,
    adapt_framework_for_query,
    summarize_rollouts,
    cross_source_insights,
)
