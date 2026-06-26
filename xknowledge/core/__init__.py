"""
QAWiki core - knowledge distillation and retrieval.

Adapted from XSkill eval/exskill/
Original work by Jiang et al. (ICML 2026, MIT License)
"""

from .llm import ExperienceLLM, KnowledgeLLM
from .retriever import KnowledgeRetriever, ExperienceRetriever, rewrite_insights_for_task, rewrite_experiences_for_task
from .insight import batch_merge_insights, batch_merge, refine_insight_library, refine_experience_library, InsightMemoryProvider
from .utils import (
    load_insight_library, load_existing,
    save_insight_library, save_library,
    load_insights, load_experiences,
    format_insights_for_prompt, format_for_prompt,
    image_to_base64,
)
from .framework import (
    generate_framework, generate_skill_for_sample,
    merge_frameworks, merge_skills,
    adapt_framework_for_query, adapt_skill_for_task,
    refine_framework_document, refine_skill_document,
)
from .summarizer import summarize_rollouts
from .critic import intra_sample_experiences as cross_source_insights
from .source import (
    SourceRegistry,
    SectionDiff,
    hash_text,
    hash_file,
    section_key_for,
)

__all__ = [
    "ExperienceLLM", "KnowledgeLLM",
    "KnowledgeRetriever", "ExperienceRetriever",
    "rewrite_insights_for_task", "rewrite_experiences_for_task",
    "batch_merge_insights", "batch_merge",
    "refine_insight_library", "refine_experience_library",
    "InsightMemoryProvider",
    "load_insight_library", "load_existing",
    "load_insight_library_meta",
    "save_insight_library", "save_library",
    "load_insights", "load_experiences",
    "format_insights_for_prompt", "format_for_prompt",
    "image_to_base64",
    "generate_framework", "generate_skill_for_sample",
    "merge_frameworks", "merge_skills",
    "adapt_framework_for_query", "adapt_skill_for_task",
    "refine_framework_document", "refine_skill_document",
    "summarize_rollouts",
    "cross_source_insights",
    "SourceRegistry", "SectionDiff",
    "hash_text", "hash_file", "section_key_for",
]
