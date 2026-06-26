"""
QAKnowledge - unified knowledge base facade.

Wraps insight management, framework management, and retrieval
into a single API for Q&A workflows.

Insight storage model
---------------------
Insights are kept in memory as two parallel views over the same data:

  * ``self._insights``      - ``Dict[str, str]`` mapping id -> text.
    This is the text-only view that the distillation/retrieval core
    (``insight.py``, ``retriever.py``, ``critic.py``) operates on, so
    those modules are untouched by the source-provenance upgrade.
  * ``self._insight_meta``  - ``Dict[str, dict]`` mapping id -> rich
    metadata (source_doc, section, added_at, content_hash). Populated
    on distillation and persisted alongside the text.

Both are loaded/saved together so the on-disk schema is
``{id: {text, ...meta}}`` while the in-memory core still sees strings.
"""

import os
import json
import copy
import time
from dataclasses import asdict
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
    load_insight_library,
    load_insight_library_meta,
    save_insight_library,
    format_insights_for_prompt,
)
from .core.source import (
    SourceRegistry,
    hash_text,
    hash_file,
    section_key_for,
    SectionDiff,
    SOURCE_DISTILLABLE,
    SOURCE_REFERENCE,
    STATE_DRAFT,
    STATE_PENDING,
    STATE_PUBLISHED,
    STATE_DEPRECATED,
    SERVED_STATES,
)
from .core.framework import (
    generate_framework, merge_frameworks,
    adapt_framework_for_query, refine_framework_document,
)
from .core.summarizer import summarize_rollouts
from .core.critic import intra_sample_experiences as cross_source_critique
from .prompts.query import FRAMEWORK_INJECTION_HEADER
from .governance import (
    State, transition, IllegalTransition,
    Role, Action, Actor, PermissionDenied,
    can, require, resolve_role,
    ReviewQueue, TARGET_INSIGHT, TARGET_REFERENCE, TARGET_FRAMEWORK,
)
from .routing import classify_source, Classification, DISTILLABLE, REFERENCE
from .references import ReferenceStore


# Number of new ops from a section before we renumber them down.
_OPS_PER_SECTION_BUDGET = 5


class QAKnowledge:
    """Knowledge base facade for Q&A with continual learning."""

    def __init__(self, config: Optional[QAWikiConfig] = None):
        self.config = config or load_config()
        self.kb_dir = Path(self.config.kb_dir)
        self._insight_path = str(self.kb_dir / "insights.json")
        self._framework_dir = self.kb_dir / "frameworks"
        self._framework_path = str(self._framework_dir / "global.md")
        self._embeddings_dir = str(self.kb_dir / "embeddings")
        self._sources_dir = self.kb_dir / "sources"
        self._registry_path = str(self._sources_dir / "registry.json")
        self._work_dir = self._sources_dir / "_work"
        self._references_dir = self.kb_dir / "references"
        self._review_queue_path = str(self.kb_dir / "review_queue.json")

        self._llm: Optional[ExperienceLLM] = None
        self._retriever: Optional[KnowledgeRetriever] = None
        self._insights: Dict[str, str] = {}
        self._insight_meta: Dict[str, Dict[str, Any]] = {}
        self._framework: str = ""
        self._registry: Optional[SourceRegistry] = None
        self._review_queue: Optional[ReviewQueue] = None
        self._reference_store: Optional[ReferenceStore] = None
        # The actor governing write operations. CLI sets this via --as.
        # Defaults to an automated "system" editor for backward-compat with
        # the existing ingest pipeline; transitions still enforce policy.
        self.actor: Actor = Actor(user_id="system", role=Role.EDITOR)

        self._ensure_dirs()
        self._load()

    def _ensure_dirs(self):
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self._framework_dir.mkdir(parents=True, exist_ok=True)
        self._sources_dir.mkdir(parents=True, exist_ok=True)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._references_dir.mkdir(parents=True, exist_ok=True)

    def _load(self):
        # Load rich metadata, then derive the text-only view.
        self._insight_meta = load_insight_library_meta(self._insight_path)
        self._insights = {
            eid: entry.get("text", "")
            for eid, entry in self._insight_meta.items()
        }
        # Prune meta entries that no longer have a matching insight, and
        # backfill state on legacy insights (default published = old behavior).
        self._insight_meta = {
            eid: m for eid, m in self._insight_meta.items() if eid in self._insights
        }
        for m in self._insight_meta.values():
            m.setdefault("state", STATE_PUBLISHED)
        if self._framework_path and os.path.exists(self._framework_path):
            with open(self._framework_path, "r", encoding="utf-8") as f:
                self._framework = f.read()

    def _save(self):
        save_insight_library(
            self._insight_path, self._insights, meta=self._insight_meta
        )
        if self._framework:
            with open(self._framework_path, "w", encoding="utf-8") as f:
                f.write(self._framework)

    @property
    def registry(self) -> SourceRegistry:
        if self._registry is None:
            self._registry = SourceRegistry(self._registry_path)
        return self._registry

    @property
    def review_queue(self) -> ReviewQueue:
        if self._review_queue is None:
            self._review_queue = ReviewQueue(self._review_queue_path)
        return self._review_queue

    @property
    def reference_store(self) -> ReferenceStore:
        if self._reference_store is None:
            self._reference_store = ReferenceStore(
                str(self._references_dir), registry=self.registry
            )
        return self._reference_store

    def set_actor(self, actor: Actor):
        """Set the actor governing subsequent write operations."""
        self.actor = actor

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

    def _served_insights(self) -> Dict[str, str]:
        """The insight subset served by default retrieval.

        Viewers (and automated askers) only see ``published`` insights.
        Contributors with ``view_pending`` see everything. The retriever is
        keyed on the full library, so we post-filter here to keep the
        embedding cache complete while enforcing governance on results.
        """
        can_see_pending = can(self.actor.role, Action.VIEW_PENDING)
        out: Dict[str, str] = {}
        for eid, text in self._insights.items():
            state = self._insight_meta.get(eid, {}).get("state", STATE_PUBLISHED)
            if state in SERVED_STATES or can_see_pending:
                out[eid] = text
        return out

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question using the knowledge base.

        Returns dict with: insight, framework, adapted_guide, retrieval_info.

        Only ``published`` insights are returned to viewers; a contributor
        with ``view_pending`` additionally sees pending ones.
        """
        result = {"question": question, "insights": {}, "framework": "", "adapted_guide": ""}

        served = self._served_insights()

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

        # Post-filter to governance-served set: the retriever indexes the
        # full library (keeps the embedding cache complete), so a top-K hit
        # may be a pending insight that viewers must not see.
        insights = {k: v for k, v in insights.items() if k in served}

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

    def learn_from_document(self, file_path: str) -> Dict[str, Any]:
        """Distill knowledge from a document with section-level incremental updates.

        On first ingestion every section is distilled. On re-ingestion:

          * If the whole-file hash is unchanged -> no-op short-circuit.
          * Otherwise the per-section registry diff is computed and only
            added/changed sections are re-distilled; removed sections'
            insights are dropped; unchanged sections keep their insights.

        New insights are tagged with provenance metadata (source_doc,
        section, added_at, content_hash) and a per-document SOP/framework
        is generated and merged into the global framework.

        Args:
            file_path: Path to a Markdown or Word document.

        Returns:
            A result dict describing the update:
              {
                "doc_path", "action": "new"|"updated"|"unchanged",
                "diff": {"added","changed","removed","unchanged"} counts,
                "added_insights", "removed_insights",
                "framework_generated": bool,
              }
        """
        doc_path = os.path.abspath(file_path)
        result = {
            "doc_path": doc_path,
            "action": "new",
            "diff": {"added": 0, "changed": 0, "removed": 0, "unchanged": 0},
            "added_insights": 0,
            "removed_insights": 0,
            "framework_generated": False,
        }

        if not os.path.exists(doc_path):
            print(f"  Warning: document not found: {doc_path}")
            return result

        # ---- Parse + section ----
        segments = self._parse_document(doc_path)
        if not segments:
            print(f"  Warning: no content parsed from {doc_path}")
            return result

        # section_key -> section content text
        new_sections: Dict[str, str] = {}
        section_meta: Dict[str, Dict[str, Any]] = {}
        for idx, seg in enumerate(segments):
            heading = seg.metadata.get("heading", "")
            key = section_key_for(heading, seg.metadata.get("index", idx + 1))
            # Merge duplicate keys (shouldn't normally happen) by appending.
            if key in new_sections:
                new_sections[key] = new_sections[key] + "\n\n" + seg.content
            else:
                new_sections[key] = seg.content
            section_meta[key] = {
                "heading": heading,
                "index": seg.metadata.get("index", idx + 1),
            }

        # ---- Whole-document short-circuit ----
        whole_hash = hash_file(doc_path)
        reg = self.registry
        if reg.whole_doc_unchanged(doc_path, whole_hash):
            result["action"] = "unchanged"
            print(f"  Document unchanged (hash match), skipping: {doc_path}")
            return result

        # ---- Section diff ----
        diff: SectionDiff = reg.diff_sections(doc_path, new_sections)
        result["action"] = "updated" if reg.has_doc(doc_path) else "new"
        result["diff"] = {
            "added": len(diff.added),
            "changed": len(diff.changed),
            "removed": len(diff.removed),
            "unchanged": len(diff.unchanged),
        }
        print(f"  [{result['action']}] {os.path.basename(doc_path)}: {diff.summary()}")

        # ---- 1. Remove insights for removed + changed sections ----
        to_drop_ids: List[str] = []
        for key in list(diff.removed) + list(diff.changed):
            to_drop_ids.extend(reg.section_insight_ids(doc_path, key))
        if to_drop_ids:
            self._drop_insights(to_drop_ids)
            result["removed_insights"] = len(to_drop_ids)

        # ---- 2. Distill added + changed sections ----
        keys_to_distill = diff.added + diff.changed
        new_ops: List[Dict[str, str]] = []
        # Track which op text came from which section to bind insights later.
        op_to_section: List[Tuple[str, str]] = []  # (op_text, section_key)

        for key in keys_to_distill:
            content = new_sections[key]
            ops = self._distill_section(
                doc_path, section_meta.get(key, {}), content
            )
            for op_text in ops:
                new_ops.append({"experience": op_text})
                op_to_section.append((op_text, key))

        # ---- 3. Merge new ops into the insight library ----
        added_insight_texts: List[Tuple[str, str]] = []  # (op_text, section_key)
        if new_ops:
            before_ids = set(self._insights.keys())
            before_texts = set(self._insights.values())
            self._insights = batch_merge_insights(
                self._insights, new_ops, self.llm,
                experience_limit=self.config.insight_max_items,
                similarity_threshold=self.config.insight_similarity_threshold,
            )
            after_ids = set(self._insights.keys())
            after_texts = set(self._insights.values())

            # Newly added ids are those not present before.
            new_ids = [i for i in sorted(after_ids - before_ids)]
            result["added_insights"] = len(new_ids)

            # Bind provenance metadata to each new insight. We can't map
            # op_text -> insight id perfectly (merge may renumber), so we
            # match new insight texts that came from a known section op.
            unmatched_ops = list(op_to_section)
            for nid in new_ids:
                text = self._insights[nid]
                # Exact match first.
                match = next(
                    (t for t, _ in unmatched_ops if t == text), None
                )
                if match is None:
                    # Substring match (merged insights contain op text).
                    match = next(
                        (t for t, _ in unmatched_ops if t in text), None
                    )
                if match is not None:
                    _, section_key = next(
                        (t, k) for t, k in unmatched_ops if t == match
                    )
                    self._insight_meta[nid] = {
                        "source_doc": doc_path,
                        "section": section_key,
                        "section_heading": section_meta.get(section_key, {}).get("heading", ""),
                        "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "content_hash": hash_text(text),
                        # Governance: auto-distilled insights default to
                        # pending unless the actor may bypass review.
                        "state": STATE_PUBLISHED if can(self.actor.role, Action.BYPASS_REVIEW) else STATE_PENDING,
                        "origin": "distillable",
                    }
                    added_insight_texts.append((text, section_key))
                    unmatched_ops = [x for x in unmatched_ops if x[0] != match]
                else:
                    # New insight whose provenance can't be pinned (likely
                    # a merge product). Attribute to the document itself.
                    self._insight_meta[nid] = {
                        "source_doc": doc_path,
                        "section": "",
                        "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "content_hash": hash_text(text),
                        "state": STATE_PUBLISHED if can(self.actor.role, Action.BYPASS_REVIEW) else STATE_PENDING,
                        "origin": "distillable",
                    }
        else:
            result["added_insights"] = 0

        # ---- 4. Update registry: rebuild section map ----
        sections_record: Dict[str, Dict[str, Any]] = {}
        for key, content in new_sections.items():
            existing_ids: List[str] = []
            if key in diff.unchanged:
                existing_ids = reg.section_insight_ids(doc_path, key)
            elif key in diff.added or key in diff.changed:
                # Collect new insight ids attributed to this section.
                existing_ids = [
                    nid for nid, m in self._insight_meta.items()
                    if m.get("source_doc") == doc_path and m.get("section") == key
                ]
            sections_record[key] = {
                "hash": hash_text(content),
                "insight_ids": existing_ids,
            }
        reg.record_sections(doc_path, whole_hash, sections_record)

        # ---- 5. Framework generation (fixes the silent-failure bug) ----
        framework_generated = self._maybe_generate_framework(
            doc_path, new_sections, section_meta, diff
        )
        result["framework_generated"] = framework_generated

        # ---- 6. Submit new insights to the review queue ----
        # Insights produced automatically land in ``pending`` (see step 3);
        # enqueue them for review unless the actor bypassed review.
        if result["added_insights"] and not can(self.actor.role, Action.BYPASS_REVIEW):
            for nid, m in self._insight_meta.items():
                if m.get("state") == STATE_PENDING and m.get("source_doc") == doc_path:
                    self.review_queue.submit(
                        target_type=TARGET_INSIGHT,
                        target_id=nid,
                        title=self._insights[nid][:60],
                        reason=f"auto-distilled from {os.path.basename(doc_path)}",
                        proposed_by=self.actor.user_id,
                        payload_preview=self._insights[nid][:200],
                    )
            self.review_queue.save()

        # ---- 7. Persist everything ----
        self._save()
        reg.save()
        if self._retriever is not None:
            self.retriever.update_experiences(self._insights)

        return result

    # ---- Unified source intake (P1 routing) ----

    def add_source(
        self,
        file_path: str,
        declared_type: Optional[str] = None,
        title: Optional[str] = None,
        state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route a new document into the KB by source type.

        The single entry point the CLI / API calls for any new document.
        Runs the router (declaration-first, LLM-fallback) and dispatches:

          * ``distillable`` -> ``learn_from_document`` (existing distill
            pipeline; new insights default to ``pending`` and enter the
            review queue unless the actor bypasses review).
          * ``reference``   -> ``ReferenceStore.register`` (preserve the
            file verbatim, version it, record it; never distilled).

        Args:
            file_path: Path to the document.
            declared_type: Optional ``--type`` override (``distillable`` /
                ``reference``). Wins over all heuristics.
            title: Optional title (used for reference docs).
            state: Optional governance state override for the resulting
                artifact (default: ``pending`` for reference, and
                pending/bypass per actor for distillable).

        Returns:
            A result dict describing what happened.
        """
        from .routing import classify_source, DISTILLABLE, REFERENCE

        doc_path = os.path.abspath(file_path)
        if not os.path.exists(doc_path):
            raise FileNotFoundError(doc_path)

        classification = classify_source(
            doc_path,
            declared_type=declared_type,
            llm=self.llm if self._llm is not None else None,
        )
        result: Dict[str, Any] = {
            "doc_path": doc_path,
            "source_type": classification.source_type,
            "classification": classification.as_dict(),
        }

        if classification.source_type == DISTILLABLE:
            # Reuse the existing distill pipeline.
            distill_result = self.learn_from_document(doc_path)
            result.update(distill_result)
            result["route"] = "distilled"
            return result

        # ---- reference path: preserve verbatim ----
        # Only contributors may submit reference docs.
        from .governance.permissions import require, Action
        require(self.actor.role, Action.SUBMIT)

        ref_state = state or STATE_PENDING
        manifest = self.reference_store.register(
            file_path=doc_path,
            title=title,
            origin=classification.origin,
            state=ref_state,
        )
        # Submit to the review queue so reviewers can approve it.
        if ref_state == STATE_PENDING:
            self.review_queue.submit(
                target_type=TARGET_REFERENCE,
                target_id=manifest.doc_id,
                title=manifest.title,
                reason=f"reference doc: {os.path.basename(doc_path)} ({classification.method})",
                proposed_by=self.actor.user_id,
                payload_preview=f"{manifest.origin} · {len(manifest.versions)} version(s)",
            )
            self.review_queue.save()

        result.update({
            "route": "reference",
            "doc_id": manifest.doc_id,
            "version": manifest.current_version,
            "state": manifest.state,
            "registered": True,
        })
        return result

    # ---- Governance: approve / reject ----

    def _apply_review_decision(
        self,
        target_type: str,
        target_id: str,
        decision: str,
    ) -> Dict[str, Any]:
        """Apply an approve/reject decision to a pending item.

        Checks the actor can perform the transition, mutates the target's
        state (insight meta or reference manifest), and records the
        decision in the review queue. Returns a summary dict.
        """
        from .governance.states import transition as make_transition

        action_name = "approve" if decision == "approved" else "reject"
        # Permission gate.
        require(self.actor.role, Action(action_name))

        queue = self.review_queue
        item = queue.find_open(target_type, target_id)
        if item is None:
            return {"target_type": target_type, "target_id": target_id,
                    "error": "no open review item found"}

        new_state = STATE_PUBLISHED if decision == "approved" else STATE_DRAFT

        if target_type == TARGET_INSIGHT:
            if target_id not in self._insights:
                return {"target_type": target_type, "target_id": target_id,
                        "error": "insight not found"}
            meta = self._insight_meta.setdefault(target_id, {})
            old = meta.get("state", STATE_PENDING)
            make_transition(old, new_state)  # validates the transition
            meta["state"] = new_state
            meta["reviewed_by"] = self.actor.user_id
            self._save()
        elif target_type == TARGET_REFERENCE:
            manifest = self.reference_store.get(target_id)
            if manifest is None:
                return {"target_type": target_type, "target_id": target_id,
                        "error": "reference doc not found"}
            make_transition(manifest.state, new_state)
            self.reference_store.set_state(target_id, new_state)
        else:
            return {"target_type": target_type, "target_id": target_id,
                    "error": f"unsupported target type {target_type}"}

        # Record the decision in the queue.
        queue.decide(item.id, decision, decided_by=self.actor.user_id)
        queue.save()

        return {
            "target_type": target_type,
            "target_id": target_id,
            "decision": decision,
            "new_state": new_state,
            "decided_by": self.actor.user_id,
        }

    def approve_insight(self, insight_id: str) -> Dict[str, Any]:
        """Approve a pending insight (requires reviewer+)."""
        return self._apply_review_decision(
            TARGET_INSIGHT, insight_id, "approved"
        )

    def reject_insight(self, insight_id: str) -> Dict[str, Any]:
        """Reject a pending insight (requires reviewer+)."""
        return self._apply_review_decision(
            TARGET_INSIGHT, insight_id, "rejected"
        )

    def approve_reference(self, doc_id: str) -> Dict[str, Any]:
        """Approve a pending reference document (requires reviewer+)."""
        return self._apply_review_decision(
            TARGET_REFERENCE, doc_id, "approved"
        )

    def reject_reference(self, doc_id: str) -> Dict[str, Any]:
        """Reject a pending reference document (requires reviewer+)."""
        return self._apply_review_decision(
            TARGET_REFERENCE, doc_id, "rejected"
        )

    def list_pending_reviews(self) -> List[Dict[str, Any]]:
        """Return the open review queue (for the review CLI/UI)."""
        from dataclasses import asdict
        return [asdict(i) for i in self.review_queue.list_open()]

    # ---- Document distillation internals ----

    def _parse_document(self, file_path: str):
        """Pick and run the right parser based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".docx",):
            from .parsers.word import WordParser
            return WordParser().parse(file_path)
        # Default to Markdown (.md/.markdown/anything else).
        from .parsers.markdown import MarkdownParser
        return MarkdownParser().parse(file_path)

    def _distill_section(
        self, doc_path: str, meta: Dict[str, Any], content: str
    ) -> List[str]:
        """Summarize one section and critique it into insight texts.

        A single section is treated as one "source" for the cross-source
        critique; to extract contrasting lessons we split the section
        into paragraph-level summaries so the critic has multiple
        trajectories to compare (mirroring how XSkill critiques multiple
        rollouts).
        """
        # Paragraph-level sub-summaries give the critic multiple sources.
        chunks = self._split_into_chunks(content, max_chars=1200)
        if not chunks:
            return []

        summaries: Dict[str, str] = {}
        for i, chunk in enumerate(chunks):
            try:
                resp = self.llm.chat(
                    "Summarize the key knowledge in this text segment. "
                    "Focus on actionable, reusable guidance:\n\n"
                    f"{chunk}",
                    max_tokens=2048,
                )
                if resp and resp.strip():
                    summaries[f"seg_{i}"] = resp.strip()
            except Exception as e:
                print(f"    Warning: section summary failed (chunk {i}): {e}")

        if not summaries:
            return []

        # Need >=2 summaries for cross-critique to be meaningful; if only
        # one chunk, synthesize insights directly from its summary.
        if len(summaries) < 2:
            return self._insights_from_single_summary(
                doc_path, meta, next(iter(summaries.values()))
            )

        question = f"Document: {os.path.basename(doc_path)}"
        if meta.get("heading"):
            question += f" - Section: {meta['heading']}"

        try:
            ops = cross_source_critique(
                question, "", summaries, self.llm,
                max_ops=self.config.insight_max_ops,
            )
        except Exception as e:
            print(f"    Warning: cross-source critique failed: {e}")
            return []

        texts: List[str] = []
        for o in (ops if isinstance(ops, list) else []):
            if isinstance(o, dict):
                t = (o.get("experience") or "").strip()
                if t:
                    texts.append(t)
        return texts

    def _insights_from_single_summary(
        self, doc_path: str, meta: Dict[str, Any], summary: str
    ) -> List[str]:
        """Extract 1-2 insights when a section yields only one summary."""
        heading = meta.get("heading", "")
        ctx = f"Document: {os.path.basename(doc_path)}"
        if heading:
            ctx += f" - Section: {heading}"
        prompt = (
            "From the following section summary, extract 1-2 concise, "
            "generalizable insights (each under 64 words). Return ONLY a "
            "JSON array of objects with an \"experience\" key.\n\n"
            f"Context: {ctx}\n\nSummary:\n{summary}\n\n"
            "```json\n[\n  {\"experience\": \"...\"}\n]\n```"
        )
        try:
            resp = self.llm.chat(prompt, max_tokens=1024)
            import re
            m = re.search(r"\[.*\]", resp, re.DOTALL)
            if not m:
                return []
            arr = json.loads(m.group(0))
            return [
                (o.get("experience") or "").strip()
                for o in arr
                if isinstance(o, dict) and (o.get("experience") or "").strip()
            ]
        except Exception as e:
            print(f"    Warning: single-summary insight extraction failed: {e}")
            return []

    @staticmethod
    def _split_into_chunks(text: str, max_chars: int = 1200) -> List[str]:
        """Split text into roughly paragraph-sized chunks under max_chars."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []
        chunks: List[str] = []
        buf = ""
        for p in paragraphs:
            if not buf:
                buf = p
            elif len(buf) + len(p) + 2 <= max_chars:
                buf = buf + "\n\n" + p
            else:
                chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        # If a single paragraph exceeds max_chars, hard-split it.
        final: List[str] = []
        for c in chunks:
            if len(c) <= max_chars:
                final.append(c)
            else:
                for i in range(0, len(c), max_chars):
                    final.append(c[i:i + max_chars])
        return final

    def _drop_insights(self, insight_ids: List[str]):
        """Remove insights by id and drop their metadata + embeddings."""
        ids = [i for i in insight_ids if i in self._insights]
        if not ids:
            return
        for i in ids:
            self._insights.pop(i, None)
            self._insight_meta.pop(i, None)
        print(f"    Dropped {len(ids)} insights (source sections changed/removed)")

    def _maybe_generate_framework(
        self,
        doc_path: str,
        sections: Dict[str, str],
        section_meta: Dict[str, Dict[str, Any]],
        diff: SectionDiff,
    ) -> bool:
        """Generate a per-document SOP from changed sections and merge it.

        Fixes the prior bug where ``learn_from_document`` passed
        ``sample_dir=kb_dir`` to ``generate_framework``: the framework
        generator looks for ``<sample_dir>/exp_summary_prompt.txt``, which
        was never written, so framework distillation silently no-op'd.
        Here we materialize that prompt file into a per-document work
        directory so the framework path actually runs.
        """
        # Only generate from sections that actually changed.
        keys = diff.added + diff.changed
        if not keys:
            return False
        try:
            doc_stem = Path(doc_path).stem
            work_dir = str(self._work_dir / doc_stem)
            os.makedirs(work_dir, exist_ok=True)

            # Build a <trajectory> body from the distilled sections.
            traj_parts = []
            for key in keys:
                meta = section_meta.get(key, {})
                heading = meta.get("heading", "")
                body = sections.get(key, "")
                label = heading if heading else f"(section {meta.get('index', '?')})"
                traj_parts.append(f"==== Section: {label} ====\n{body}")
            trajectory = "\n\n".join(traj_parts)
            prompt_body = (
                "<trajectory>\n" + trajectory + "\n</trajectory>"
            )
            prompt_path = os.path.join(work_dir, "exp_summary_prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_body)

            sample_info = {
                "question_id": doc_stem,
                "sample_dir": work_dir,
            }
            fw = generate_framework(sample_info, self.llm, None, ground_truth="")
            if not (fw.get("success") and fw.get("skill_content")):
                return False

            self._framework = merge_frameworks(
                self._framework, [fw["skill_content"]], self.llm, None
            )
            if len(self._framework.split()) > self.config.framework_word_threshold:
                self._framework = refine_framework_document(
                    self._framework, self.llm,
                    word_threshold=self.config.framework_word_threshold,
                    force_refine=True,
                )
            return True
        except Exception as e:
            print(f"    Warning: framework generation failed: {e}")
            return False

    # ---- Status ----

    @property
    def status(self) -> Dict[str, Any]:
        published = sum(
            1 for m in self._insight_meta.values()
            if m.get("state", STATE_PUBLISHED) == STATE_PUBLISHED
        )
        pending = sum(
            1 for m in self._insight_meta.values()
            if m.get("state") == STATE_PENDING
        )
        return {
            "insight_count": len(self._insights),
            "published_insights": published,
            "pending_insights": pending,
            "pending_reviews": self.review_queue.open_count,
            "framework_words": len(self._framework.split()) if self._framework else 0,
            "kb_dir": str(self.kb_dir),
            "insight_path": self._insight_path,
            "framework_path": self._framework_path,
            "registry_path": self._registry_path,
            "doc_count": self.registry.doc_count,
            "reference_count": self.reference_store.status["reference_count"],
        }
