"""
QAKnowledge - unified knowledge base facade.

Wraps insight management, framework management, and retrieval
into a single API for Q&A workflows.
"""

import os
import json
import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .config import QAWikiConfig, load_config
from .core.llm import ExperienceLLM, KnowledgeLLM
from .core.retriever import (
    KnowledgeRetriever,
    rewrite_insights_for_task,
)
from .core.insight import batch_merge_insights, refine_insight_library
from .core.utils import (
    load_insight_library, save_insight_library,
    format_insights_for_prompt,
)
from .core.framework import (
    generate_framework, merge_frameworks,
    adapt_framework_for_query, refine_framework_document,
)
from .core.summarizer import summarize_rollouts
from .core.critic import intra_sample_experiences as cross_source_critique
from .prompts.query import FRAMEWORK_INJECTION_HEADER


class QAKnowledge:
    """Knowledge base facade for Q&A with continual learning."""

    def __init__(self, config: Optional[QAWikiConfig] = None):
        self.config = config or load_config()
        self.kb_dir = Path(self.config.kb_dir)
        self._insight_path = str(self.kb_dir / "insights.json")
        self._framework_dir = self.kb_dir / "frameworks"
        self._framework_path = str(self._framework_dir / "global.md")
        self._embeddings_dir = str(self.kb_dir / "embeddings")

        self._llm: Optional[ExperienceLLM] = None
        self._retriever: Optional[KnowledgeRetriever] = None
        self._insights: Dict[str, str] = {}
        self._framework: str = ""

        self._ensure_dirs()
        self._load()

    def _ensure_dirs(self):
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self._framework_dir.mkdir(parents=True, exist_ok=True)

    def _load(self):
        self._insights = load_insight_library(self._insight_path)
        if self._framework_path and os.path.exists(self._framework_path):
            with open(self._framework_path, "r", encoding="utf-8") as f:
                self._framework = f.read()

    def _save(self):
        save_insight_library(self._insight_path, self._insights)
        if self._framework:
            with open(self._framework_path, "w", encoding="utf-8") as f:
                f.write(self._framework)

    @property
    def llm(self) -> ExperienceLLM:
        if self._llm is None:
            os.environ.setdefault("EXPERIENCE_MODEL_NAME", self.config.llm_model)
            os.environ.setdefault("EXPERIENCE_API_KEY", self.config.llm_api_key)
            os.environ.setdefault("EXPERIENCE_END_POINT", self.config.llm_endpoint)
            self._llm = ExperienceLLM(model_name=self.config.llm_model)
        return self._llm

    @property
    def retriever(self) -> KnowledgeRetriever:
        if self._retriever is None:
            self._retriever = KnowledgeRetriever(
                experiences=self._insights,
                embedding_model=self.config.embedding_model,
                embedding_api_key=self.config.embedding_api_key,
                embedding_endpoint=self.config.embedding_endpoint,
                cache_dir=self._embeddings_dir,
                llm_client=self.llm,
                insight_library_path=self._insight_path,
            )
        return self._retriever

    # ---- Phase II: Retrieval ----

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question using the knowledge base.
        
        Returns dict with: insight, framework, adapted_guide, retrieval_info
        """
        result = {"question": question, "insights": {}, "framework": "", "adapted_guide": ""}

        # Retrieve insights
        if self.config.retrieval_enable_decomposition and self._insights:
            insights, retrieval_info = self.retriever.retrieve_with_decomposition(
                task_description=question,
                top_k=self.config.retrieval_top_k,
                min_similarity=self.config.retrieval_min_similarity,
            )
        elif self._insights:
            insights, retrieval_info = self.retriever.retrieve(
                query=question,
                top_k=self.config.retrieval_top_k,
                min_similarity=self.config.retrieval_min_similarity,
            )
        else:
            insights, retrieval_info = {}, {}

        # Rewrite insights if enabled
        if insights and self.config.retrieval_enable_rewrite:
            original = copy.deepcopy(insights)
            insights = rewrite_insights_for_task(insights, question, self.llm)
            result["original_insights"] = original

        result["insights"] = insights
        result["retrieval_info"] = retrieval_info

        # Adapt framework
        if self._framework:
            exp_text = "\n\n".join(f"[{k}] {v}" for k, v in insights.items())
            adapted = adapt_framework_for_query(self._framework, exp_text, question, self.llm)
            result["framework"] = self._framework
            result["adapted_guide"] = adapted
        elif insights:
            result["adapted_guide"] = format_insights_for_prompt(insights)

        return result

    def inject_context(self, question: str) -> str:
        """Build context string for injection into Agent's system prompt."""
        response = self.ask(question)

        parts = []
        if response["adapted_guide"]:
            parts.append(response["adapted_guide"])
        if response["insights"]:
            parts.append(format_insights_for_prompt(response["insights"]))

        return "\n\n".join(parts) if parts else ""

    # ---- Phase I: Distillation ----

    def learn_from_trajectories(
        self, traj_paths: List[str], question: str,
        ground_truth: str = "", sample_dir: Optional[str] = None
    ) -> int:
        """Learn from interaction trajectories.

        Returns number of new insights added.
        """
        # Summarize
        summary = summarize_rollouts(traj_paths, self.llm, sample_dir=sample_dir)
        if not summary:
            return 0

        summaries_only = {k: v for k, v in summary.items()
                          if k not in ("question", "ground_truth", "system_prompt")}

        # Cross critique
        ops = cross_source_critique(
            question, ground_truth, summaries_only, self.llm,
            max_ops=self.config.insight_max_ops,
            debug_dir=sample_dir,
        )

        if not ops:
            return 0

        # Normalize
        norm_ops = []
        for o in ops:
            if isinstance(o, dict):
                exp_txt = o.get("experience") or ""
                if exp_txt.strip():
                    norm_ops.append({"experience": exp_txt.strip()})

        if not norm_ops:
            return 0

        # Merge
        self._insights = batch_merge_insights(
            self._insights, norm_ops, self.llm,
            experience_limit=self.config.insight_max_items,
            similarity_threshold=self.config.insight_similarity_threshold,
        )

        self._save()
        self.retriever.update_experiences(self._insights)
        return len(norm_ops)

    def learn_from_document(self, file_path: str) -> int:
        """Distill knowledge from a document.

        Returns number of new insights added.
        """
        from .parsers import BaseParser
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".md" or ext == ".markdown":
            from .parsers.markdown import MarkdownParser
            parser = MarkdownParser()
        elif ext == ".docx":
            from .parsers.word import WordParser
            parser = WordParser()
        else:
            from .parsers.markdown import MarkdownParser
            parser = MarkdownParser()

        segments = parser.parse(file_path)
        if not segments:
            return 0

        # Summarize each segment
        summaries = {}
        for i, seg in enumerate(segments):
            try:
                resp = self.llm.chat(
                    f"Summarize the key knowledge in this text segment:\n\n{seg.content}\n\n"
                    "Extract actionable insights (≤64 words each).",
                    max_tokens=2048,
                )
                summaries[f"seg_{i}"] = resp
            except Exception:
                pass

        if len(summaries) < 2:
            return 0

        # Cross critique the summaries
        question = f"Document: {os.path.basename(file_path)}"
        ops = cross_source_critique(question, "", summaries, self.llm, max_ops=self.config.insight_max_ops)

        norm_ops = []
        for o in ops:
            if isinstance(o, dict):
                exp_txt = o.get("experience") or ""
                if exp_txt.strip():
                    norm_ops.append({"experience": exp_txt.strip()})

        if not norm_ops:
            return 0

        self._insights = batch_merge_insights(
            self._insights, norm_ops, self.llm,
            experience_limit=self.config.insight_max_items,
            similarity_threshold=self.config.insight_similarity_threshold,
        )

        # Try to generate framework
        try:
            all_text = "\n\n".join(s.content for s in segments)
            new_fw = generate_framework({"question_id": os.path.basename(file_path), "sample_dir": str(self.kb_dir)}, self.llm, None, ground_truth="")
            if new_fw.get("success") and new_fw.get("skill_content"):
                self._framework = merge_frameworks(self._framework, [new_fw["skill_content"]], self.llm, None)
                if len(self._framework.split()) > self.config.framework_word_threshold:
                    self._framework = refine_framework_document(
                        self._framework, self.llm,
                        word_threshold=self.config.framework_word_threshold,
                        force_refine=True,
                    )
        except Exception:
            pass

        self._save()
        self.retriever.update_experiences(self._insights)
        return len(norm_ops)

    # ---- Status ----

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "insight_count": len(self._insights),
            "framework_words": len(self._framework.split()) if self._framework else 0,
            "kb_dir": str(self.kb_dir),
            "insight_path": self._insight_path,
            "framework_path": self._framework_path,
        }
