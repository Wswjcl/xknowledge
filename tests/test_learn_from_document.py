"""End-to-end test of learn_from_document with a STUBBED LLM.

This exercises the full update-mechanism orchestration without hitting a
real API:
  1. First ingest  -> distills, registers doc, writes insights.json + registry
  2. Re-ingest     -> whole-file hash short-circuits (no LLM calls)
  3. Edit + ingest -> only the changed section re-distills
  4. Remove + ingest -> orphaned insights dropped

The stub LLM mimics the contract the real distillation path expects:
  - section chunk summaries
  - cross-source critique JSON [{"experience": ...}, ...]
  - framework generation (markdown SOP)
  - framework merge (pass-through)
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A dummy embedding key lets KnowledgeRetriever construct (the merge path
# spins one up internally); actual embedding HTTP calls are expected to
# fail in tests, and the code already degrades to "just add" on empty
# embeddings, so no real network is needed.
os.environ.setdefault("INSIGHT_EMBEDDING_API_KEY", "stub-key-for-tests")
os.environ.setdefault("INSIGHT_EMBEDDING_ENDPOINT", "http://localhost:0/v1")

from xknowledge.knowledge import QAKnowledge
from xknowledge.config import QAWikiConfig
import xknowledge.core.retriever as _retriever_mod


def _install_noop_embeddings():
    """Make embedding generation return empty so the merge path skips
    similarity-based merging (its existing fallback) without any network."""
    def _noop_batch(self, texts, batch_size=30, max_retries=3):
        return [None] * len(texts)

    def _noop_single(self, text, max_retries=3):
        return None

    _retriever_mod.KnowledgeRetriever._generate_embeddings_batch = _noop_batch
    _retriever_mod.KnowledgeRetriever._generate_embedding = _noop_single


_install_noop_embeddings()


DOC_V1 = """# Runbook

Intro line.

## Deploy

Deploy the service with the standard pipeline.

## Rollback

Roll back by reverting the image tag.
"""

DOC_V2 = """# Runbook

Intro line.

## Deploy

Deploy the service with the standard pipeline, then verify health checks.

## Rollback

Roll back by reverting the image tag.
"""

DOC_V3 = """# Runbook

Intro line.

## Deploy

Deploy the service with the standard pipeline, then verify health checks.
"""


class StubLLM:
    """Mimics ExperienceLLM.chat returning canned responses."""

    def __init__(self, *a, **kw):
        self.calls = 0

    def chat(self, prompt, max_tokens=None, temperature=None, top_p=1.0):
        self.calls += 1
        # Section-summary requests -> return a short summary.
        if "Summarize the key knowledge" in prompt:
            return "STUB SUMMARY: standard deployment guidance."
        # Single-summary insight extraction.
        if "JSON array" in prompt and "experience" in prompt:
            return '[{"experience": "Stub insight from single summary."}]'
        # Cross-source critique -> returns ops JSON.
        if "experience" in prompt and "add" in prompt:
            return (
                '```json\n'
                '[{"option": "add", "experience": "Verify health after deploy."}, '
                '{"option": "add", "experience": "Revert image tag for rollback."}]'
                '\n```'
            )
        # Framework generation / merge / refine -> markdown SOP.
        if "<trajectory>" in prompt or "SKILL.md" in prompt or "skill" in prompt.lower():
            return "---\nname: RunbookSOP\ndescription: stub\n---\n# Runbook SOP\n## Workflow\n1. Deploy\n2. Verify\n"
        # Fallback.
        return "[]"


class StubRetriever:
    """No-op retriever that only records update calls."""

    def __init__(self, *a, **kw):
        self.updated = []

    def update_experiences(self, insights):
        self.updated.append(dict(insights))


def _make_kb(tmpdir):
    cfg = QAWikiConfig(kb_dir=tmpdir)
    kb = QAKnowledge(cfg)
    # Inject stubs so no API is called. Both properties return the
    # private slot, so setting the slot is enough.
    kb._llm = StubLLM()
    kb._retriever = StubRetriever()
    return kb


# Override the lazy property accessors once at import so they return the
# pre-set stub slots instead of constructing real clients.
QAKnowledge.llm = property(lambda self: self._llm)
QAKnowledge.retriever = property(lambda self: self._retriever)


def main():
    with tempfile.TemporaryDirectory() as d:
        # --- 1. First ingest ---
        doc = os.path.join(d, "runbook.md")
        with open(doc, "w", encoding="utf-8") as f:
            f.write(DOC_V1)
        kb = _make_kb(d)
        res1 = kb.learn_from_document(doc)
        assert res1["action"] == "new", res1
        assert res1["added_insights"] >= 1, res1
        assert res1["diff"]["added"] >= 1
        print("[ok] 1. first ingest:", res1["action"], "added=", res1["added_insights"],
              "diff=", res1["diff"])

        # Insights persisted with provenance meta.
        meta = kb._insight_meta
        assert any(m.get("source_doc") == os.path.abspath(doc) for m in meta.values()), \
            f"no insight has source_doc provenance: {meta}"
        print("[ok]    provenance attached to", len(meta), "insights")

        # Registry persisted.
        reg_path = os.path.join(d, "sources", "registry.json")
        assert os.path.exists(reg_path)
        with open(reg_path, encoding="utf-8") as f:
            reg = json.load(f)
        assert os.path.abspath(doc) in reg, reg.keys()
        print("[ok]    registry persisted:", len(reg), "doc(s)")

        # --- 2. Re-ingest unchanged -> short-circuit ---
        kb2 = _make_kb(d)
        before_calls = kb2._llm.calls
        res2 = kb2.learn_from_document(doc)
        assert res2["action"] == "unchanged", res2
        assert kb2._llm.calls == before_calls, "LLM should NOT be called when unchanged"
        print("[ok] 2. re-ingest unchanged -> short-circuit, 0 LLM calls")

        # --- 3. Edit one section -> only that section re-distills ---
        with open(doc, "w", encoding="utf-8") as f:
            f.write(DOC_V2)  # Deploy section edited
        kb3 = _make_kb(d)
        res3 = kb3.learn_from_document(doc)
        assert res3["action"] == "updated", res3
        assert res3["diff"]["changed"] >= 1, res3["diff"]
        print("[ok] 3. edited section ->", res3["action"], "diff=", res3["diff"],
              "insights +=", res3["added_insights"], "- =", res3["removed_insights"])

        # --- 4. Remove a section -> orphaned insights cleaned ---
        with open(doc, "w", encoding="utf-8") as f:
            f.write(DOC_V3)  # Rollback section removed
        kb4 = _make_kb(d)
        res4 = kb4.learn_from_document(doc)
        assert res4["diff"]["removed"] >= 1, res4["diff"]
        print("[ok] 4. removed section ->", res4["action"], "diff=", res4["diff"],
              "removed_insights=", res4["removed_insights"])

        # --- 5. Framework work file materialized (bug-fix check) ---
        work_prompt = os.path.join(d, "sources", "_work", "runbook", "exp_summary_prompt.txt")
        assert os.path.exists(work_prompt), f"framework prompt file missing: {work_prompt}"
        with open(work_prompt, encoding="utf-8") as f:
            assert "<trajectory>" in f.read()
        # Global framework should have content now.
        assert kb4._framework.strip(), "global framework empty after distillation"
        print("[ok] 5. framework bug fixed: prompt file materialized, global.md has content")

        # --- 6. Backward-compat: legacy insights.json loads cleanly ---
        legacy_path = os.path.join(d, "legacy_kb")
        os.makedirs(legacy_path)
        lp = os.path.join(legacy_path, "insights.json")
        with open(lp, "w", encoding="utf-8") as f:
            json.dump({"insights": {"E0": "legacy plain string"}}, f)
        cfg2 = QAWikiConfig(kb_dir=legacy_path)
        kb5 = QAKnowledge(cfg2)
        assert kb5._insights == {"E0": "legacy plain string"}, kb5._insights
        # Legacy string entries are wrapped with {"text": ...} and backfilled
        # with a default governance state (published = pre-governance behavior).
        meta = kb5._insight_meta["E0"]
        assert meta["text"] == "legacy plain string", meta
        assert meta.get("state") == "published", meta
        print("[ok] 6. legacy insights.json loaded with backward-compat wrapping + state backfill")

    print("\nALL E2E TESTS PASSED")


if __name__ == "__main__":
    main()
