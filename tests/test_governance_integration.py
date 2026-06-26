"""End-to-end governance + routing integration test.

Verifies the P1+P3 wiring:
  1. add_source() routes a distillable doc -> insights land in `pending`
     and enter the review queue (auto-distilled = controlled).
  2. add_source() routes a reference doc -> preserved verbatim, versioned,
     and also enters the review queue.
  3. A viewer's ask() does NOT return pending insights; a reviewer's does.
  4. approve_insight() publishes a pending insight so viewers see it.
  5. A viewer cannot approve (permission gate).

Uses the same stubbed-LLM harness as test_learn_from_document.py.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("INSIGHT_EMBEDDING_API_KEY", "stub-key-for-tests")
os.environ.setdefault("INSIGHT_EMBEDDING_ENDPOINT", "http://localhost:0/v1")

from xknowledge.knowledge import QAKnowledge
from xknowledge.config import QAWikiConfig
from xknowledge.governance import Actor, Role, PermissionDenied
from xknowledge.core.source import STATE_PENDING, STATE_PUBLISHED
import xknowledge.core.retriever as _retriever_mod


# --- stub LLM + retriever (reuse the pattern from test_learn_from_document) ---

class StubLLM:
    def __init__(self, *a, **kw):
        self.calls = 0

    def chat(self, prompt, max_tokens=None, temperature=None, top_p=1.0):
        self.calls += 1
        if "Summarize the key knowledge" in prompt:
            return "STUB SUMMARY."
        if "JSON array" in prompt and "experience" in prompt:
            return '[{"experience": "Stub insight from single summary."}]'
        if "experience" in prompt and "add" in prompt:
            return '```json\n[{"option": "add", "experience": "Verify health after deploy."}]```'
        if "<trajectory>" in prompt or "SKILL.md" in prompt or "skill" in prompt.lower():
            return "---\nname: SOP\ndescription: stub\n---\n# SOP\n## Workflow\n1. Deploy\n"
        # Router LLM classifier.
        if "source_type" in prompt and "distillable" in prompt:
            return '{"source_type": "distillable", "confidence": 0.7, "reason": "prose"}'
        return "[]"


class StubRetriever:
    def __init__(self, *a, **kw):
        self.updated = []

    def update_experiences(self, insights):
        self.updated.append(dict(insights))

    def retrieve(self, query, top_k=5, min_similarity=0.0):
        # Return all known insights so the governance filter is the only filter.
        return dict(self._latest), {"retrieved_experiences": list(self._latest.keys())}

    def retrieve_with_decomposition(self, task_description, top_k=5, min_similarity=0.0):
        return self.retrieve(task_description, top_k, min_similarity)

    @property
    def _latest(self):
        return self.updated[-1] if self.updated else {}


def _install_noop_embeddings():
    _retriever_mod.KnowledgeRetriever._generate_embeddings_batch = lambda self, texts, batch_size=30, max_retries=3: [None] * len(texts)
    _retriever_mod.KnowledgeRetriever._generate_embedding = lambda self, text, max_retries=3: None


_install_noop_embeddings()

# Override lazy properties so stubs are returned.
QAKnowledge.llm = property(lambda self: self._llm)
QAKnowledge.retriever = property(lambda self: self._retriever)


DOC = """# Runbook

## Deploy

Deploy the service with the standard pipeline.
"""


def _make_kb(tmpdir, actor=None):
    cfg = QAWikiConfig(kb_dir=tmpdir)
    kb = QAKnowledge(cfg)
    kb._llm = StubLLM()
    kb._retriever = StubRetriever()
    if actor is not None:
        kb.set_actor(actor)
    return kb


def main():
    with tempfile.TemporaryDirectory() as d:
        # --- 1. Editor adds a distillable doc -> insights go pending ---
        doc = os.path.join(d, "runbook.md")
        with open(doc, "w", encoding="utf-8") as f:
            f.write(DOC)
        editor = Actor(user_id="alice", role=Role.EDITOR)
        kb = _make_kb(d, actor=editor)
        res = kb.add_source(doc)
        assert res["route"] == "distilled", res
        assert res["added_insights"] >= 1, res

        # All new insights must be pending (editor cannot bypass review).
        new_pending = [eid for eid, m in kb._insight_meta.items()
                       if m.get("state") == STATE_PENDING]
        assert len(new_pending) >= 1, "no pending insights after distill"
        print(f"[ok] 1. distillable routed -> {len(new_pending)} pending insights")

        # Review queue must contain them.
        open_items = kb.review_queue.list_open()
        assert any(i.target_type == "insight" for i in open_items), \
            "insights not in review queue"
        print(f"[ok]    {len(open_items)} items in review queue")

        # --- 2. Viewer ask() should NOT see pending insights ---
        viewer_kb = _make_kb(d, actor=Actor(user_id="bob", role=Role.VIEWER))
        # Force retriever to know about all insights.
        viewer_kb.retriever.update_experiences(viewer_kb._insights)
        resp = viewer_kb.ask("how to deploy")
        served = resp["insights"]
        # None of the pending insights should be served to a viewer.
        for eid in served:
            assert viewer_kb._insight_meta[eid].get("state") == STATE_PUBLISHED, \
                f"viewer got pending insight {eid}"
        print(f"[ok] 2. viewer sees 0 pending insights (served={len(served)})")

        # --- 3. Reviewer ask() SHOULD see pending insights ---
        reviewer_kb = _make_kb(d, actor=Actor(user_id="carol", role=Role.REVIEWER))
        reviewer_kb.retriever.update_experiences(reviewer_kb._insights)
        resp_r = reviewer_kb.ask("how to deploy")
        served_r = resp_r["insights"]
        # Reviewer should see at least the pending ones.
        assert len(served_r) >= len(new_pending), \
            f"reviewer should see pending; got {len(served_r)} < {len(new_pending)}"
        print(f"[ok] 3. reviewer sees {len(served_r)} insights (incl. pending)")

        # --- 4. Approve a pending insight -> published, viewer sees it ---
        reviewer_kb2 = _make_kb(d, actor=Actor(user_id="carol", role=Role.REVIEWER))
        target_id = new_pending[0]
        result = reviewer_kb2.approve_insight(target_id)
        assert result["new_state"] == STATE_PUBLISHED, result
        assert reviewer_kb2._insight_meta[target_id]["state"] == STATE_PUBLISHED
        # Review queue no longer has it open.
        assert not reviewer_kb2.review_queue.has_open("insight", target_id)
        print(f"[ok] 4. approved insight {target_id} -> published")

        # Now a viewer should see it.
        viewer_kb2 = _make_kb(d, actor=Actor(user_id="bob", role=Role.VIEWER))
        viewer_kb2.retriever.update_experiences(viewer_kb2._insights)
        resp2 = viewer_kb2.ask("how to deploy")
        assert target_id in resp2["insights"], \
            "approved insight not visible to viewer"
        print(f"[ok]    viewer now sees approved insight {target_id}")

        # --- 5. Viewer cannot approve (permission gate) ---
        viewer_kb3 = _make_kb(d, actor=Actor(user_id="bob", role=Role.VIEWER))
        another_pending = [eid for eid, m in viewer_kb3._insight_meta.items()
                           if m.get("state") == STATE_PENDING]
        if another_pending:
            try:
                viewer_kb3.approve_insight(another_pending[0])
                assert False, "viewer should not be able to approve"
            except PermissionDenied:
                print(f"[ok] 5. viewer denied approval (PermissionDenied)")

        # --- 6. Reference doc: preserved verbatim, pending, in queue ---
        xlsx = os.path.join(d, "schedule.xlsx")
        with open(xlsx, "wb") as f:
            f.write(b"fake-xlsx-bytes")
        editor2 = _make_kb(d, actor=Actor(user_id="alice", role=Role.EDITOR))
        ref_res = editor2.add_source(xlsx, title="On-call Schedule")
        assert ref_res["route"] == "reference", ref_res
        assert ref_res["state"] == STATE_PENDING, ref_res
        assert ref_res["registered"] is True
        # File archived.
        assert os.path.exists(os.path.join(
            str(editor2._references_dir), ref_res["doc_id"], "manifest.json"))
        # In review queue.
        assert editor2.review_queue.has_open("reference", ref_res["doc_id"])
        print(f"[ok] 6. reference routed -> doc_id={ref_res['doc_id']}, "
              f"pending, archived, in queue")

        # Reference content must NOT be in the insight pool.
        assert len(editor2._insights) == len(kb._insights), \
            "reference doc leaked into insight pool"
        print(f"[ok]    reference not distilled (insight pool unchanged)")

        # --- 7. Approve the reference doc ---
        reviewer_kb4 = _make_kb(d, actor=Actor(user_id="carol", role=Role.REVIEWER))
        ap = reviewer_kb4.approve_reference(ref_res["doc_id"])
        assert ap["new_state"] == STATE_PUBLISHED, ap
        manifest = reviewer_kb4.reference_store.get(ref_res["doc_id"])
        assert manifest.state == STATE_PUBLISHED
        print(f"[ok] 7. reference approved -> published")

    print("\nALL GOVERNANCE INTEGRATION TESTS PASSED")


if __name__ == "__main__":
    main()
