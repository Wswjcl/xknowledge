"""Unit tests for P1 (routing) + P3 (governance) primitives.

Exercises the pure logic with no LLM and no filesystem state where
possible. Knowledge.py integration is covered separately.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xknowledge.governance import (
    State, transition, can_transition, IllegalTransition,
    Role, Action, Actor, can, require, resolve_role, PermissionDenied,
    ReviewQueue, TARGET_INSIGHT, TARGET_REFERENCE,
)
from xknowledge.routing import classify_source, DISTILLABLE, REFERENCE
from xknowledge.core.source import (
    SourceRegistry, SOURCE_DISTILLABLE, SOURCE_REFERENCE,
    STATE_PENDING, STATE_PUBLISHED, STATE_DEPRECATED, SERVED_STATES,
)
from xknowledge.references import ReferenceStore


# ---------------- governance/states ----------------

def test_state_transitions_allowed():
    assert transition(State.DRAFT, State.PENDING).action == "submit"
    assert transition(State.PENDING, State.PUBLISHED).action == "approve"
    assert transition(State.PENDING, State.DRAFT).action == "reject"
    assert transition(State.PUBLISHED, State.DEPRECATED).action == "deprecate"
    assert transition(State.DEPRECATED, State.PUBLISHED).action == "rollback"
    print("[ok] states: allowed transitions + action names")


def test_state_transitions_illegal():
    bad = [
        (State.DRAFT, State.PUBLISHED),   # cannot skip review
        (State.DRAFT, State.DEPRECATED),
        (State.PUBLISHED, State.DRAFT),
    ]
    for frm, to in bad:
        assert not can_transition(frm, to), f"{frm}->{to} should be illegal"
        try:
            transition(frm, to)
            assert False, f"{frm}->{to} should raise"
        except IllegalTransition:
            pass
    print("[ok] states: illegal transitions rejected (no skipping review)")


# ---------------- governance/permissions ----------------

def test_permission_matrix():
    assert can(Role.VIEWER, Action.VIEW_PUBLISHED)
    assert not can(Role.VIEWER, Action.VIEW_PENDING)
    assert not can(Role.VIEWER, Action.APPROVE)
    assert not can(Role.EDITOR, Action.APPROVE)          # editor can't approve
    assert can(Role.REVIEWER, Action.APPROVE)
    assert can(Role.REVIEWER, Action.BYPASS_REVIEW)
    assert not can(Role.REVIEWER, Action.DELETE)         # reviewer can't delete
    assert can(Role.ADMIN, Action.DELETE)
    print("[ok] permissions: matrix enforced (viewer<editor<reviewer<admin)")


def test_require_raises():
    try:
        require(Role.VIEWER, Action.APPROVE)
        assert False, "viewer approving must raise"
    except PermissionDenied:
        pass
    print("[ok] permissions: require() raises PermissionDenied")


def test_resolve_role_from_manifest():
    actor = Actor(user_id="alice", role=Role.VIEWER)
    perms = {"editors": ["bob"], "reviewers": ["alice"]}
    assert resolve_role(actor, "doc1", manifest_perms=perms) == Role.REVIEWER
    # bob is editor, alice override to admin via doc_roles wins.
    bob = Actor(user_id="bob", role=Role.VIEWER)
    assert resolve_role(bob, "doc1", manifest_perms=perms) == Role.EDITOR
    carol = Actor(user_id="carol", role=Role.VIEWER)
    assert resolve_role(carol, "doc1", manifest_perms=perms) == Role.VIEWER
    print("[ok] permissions: role resolved from manifest (admin>reviewer>editor)")


# ---------------- governance/review ----------------

def test_review_queue_lifecycle():
    with tempfile.TemporaryDirectory() as d:
        q = ReviewQueue(os.path.join(d, "review_queue.json"))
        item = q.submit(TARGET_INSIGHT, "E7", title="tip", reason="auto", proposed_by="system")
        assert item.state == "pending"
        assert q.open_count == 1
        assert q.has_open(TARGET_INSIGHT, "E7")

        # Idempotent submit.
        again = q.submit(TARGET_INSIGHT, "E7")
        assert again.id == item.id
        assert q.open_count == 1

        # Approve.
        decided = q.decide(item.id, "approved", "carol")
        assert decided.state == "approved"
        assert q.open_count == 0
        q.save()

        # Reload persists.
        q2 = ReviewQueue(os.path.join(d, "review_queue.json"))
        assert q2.open_count == 0
        assert q2.list_decided()[0].decision == "approved"
        print("[ok] review: submit (idempotent) -> approve -> persisted")


# ---------------- routing/classifier ----------------

def test_routing_declared_overrides_everything():
    # Declared type wins even if extension/directory say otherwise.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "schedule.xlsx")
        with open(p, "w") as f:
            f.write("a\tb\n1\t2\n")
        c = classify_source(p, declared_type=DISTILLABLE)
        assert c.source_type == DISTILLABLE
        assert c.method == "declared"
        print("[ok] routing: declared type overrides extension")


def test_routing_ext_heuristic():
    with tempfile.TemporaryDirectory() as d:
        xlsx = os.path.join(d, "data.xlsx")
        open(xlsx, "w").close()
        assert classify_source(xlsx).source_type == REFERENCE
        md = os.path.join(d, "note.md")
        with open(md, "w") as f:
            f.write("# Title\n\nsome prose here\n")
        assert classify_source(md).source_type == DISTILLABLE
        print("[ok] routing: extension heuristic (xlsx->ref, md->distill)")


def test_routing_manifest_and_dir():
    with tempfile.TemporaryDirectory() as d:
        # Sibling manifest.json.
        refdir = os.path.join(d, "docs")
        os.makedirs(refdir)
        md = os.path.join(refdir, "shared.md")
        with open(md, "w") as f:
            f.write("prose\n")
        with open(os.path.join(refdir, "manifest.json"), "w") as f:
            json.dump({"source_type": "reference"}, f)
        assert classify_source(md).source_type == REFERENCE

        # Directory convention: /references/ => reference.
        rdir = os.path.join(d, "references", "sub")
        os.makedirs(rdir)
        md2 = os.path.join(rdir, "roster.md")
        with open(md2, "w") as f:
            f.write("names\n")
        assert classify_source(md2).source_type == REFERENCE
        print("[ok] routing: manifest + directory convention")


def test_routing_llm_fallback():
    class FakeLLM:
        def __init__(self): self.calls = 0
        def chat(self, prompt, max_tokens=None):
            self.calls += 1
            return '{"source_type": "reference", "confidence": 0.8, "reason": "schedule"}'

    with tempfile.TemporaryDirectory() as d:
        # Unknown extension + no declaration => LLM path.
        p = os.path.join(d, "mystery.dat")
        with open(p, "w") as f:
            f.write("Some ambiguous content\nbut not a known extension.\n")
        llm = FakeLLM()
        c = classify_source(p, llm=llm)
        assert llm.calls == 1, "LLM should have been called once"
        assert c.source_type == REFERENCE
        assert c.method == "llm"
        print("[ok] routing: LLM fallback invoked for ambiguous file")


# ---------------- source registry governance fields ----------------

def test_registry_governance_fields_and_compat():
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "registry.json")
        # Legacy record without governance fields.
        with open(rp, "w") as f:
            json.dump({
                "/old/doc.md": {"content_hash": "x", "ingested_at": "t", "sections": {}}
            }, f)
        reg = SourceRegistry(rp)
        # Backfilled defaults.
        assert reg.get_source_type("/old/doc.md") == SOURCE_DISTILLABLE
        assert reg.get_state("/old/doc.md") == STATE_PUBLISHED
        print("[ok] registry: legacy record backfilled (distillable/published)")

        # Setters persist.
        reg.set_source_type("/old/doc.md", SOURCE_REFERENCE)
        reg.set_state("/old/doc.md", STATE_PENDING)
        reg.save()
        reg2 = SourceRegistry(rp)
        assert reg2.get_source_type("/old/doc.md") == SOURCE_REFERENCE
        assert reg2.get_state("/old/doc.md") == STATE_PENDING
        print("[ok] registry: setters persist source_type/state")

        # Filtered listing.
        reg2.record_sections("/new.md", "h1", {}, source_type=DISTILLABLE, state=STATE_PUBLISHED)
        refs = reg2.list_docs(source_type=SOURCE_REFERENCE)
        assert len(refs) == 1 and refs[0]["doc_path"] == "/old/doc.md"
        pub = reg2.list_docs(state=STATE_PUBLISHED)
        assert any(r["doc_path"] == "/new.md" for r in pub)
        assert not any(r["doc_path"] == "/old/doc.md" for r in pub)
        print("[ok] registry: filtered list_docs by source_type/state")


# ---------------- references/store ----------------

def test_reference_store_register_and_update():
    with tempfile.TemporaryDirectory() as d:
        reg = SourceRegistry(os.path.join(d, "registry.json"))
        store = ReferenceStore(os.path.join(d, "references"), registry=reg)

        f1 = os.path.join(d, "roster.xlsx")
        with open(f1, "wb") as f:
            f.write(b"v1-content")
        m = store.register(f1, title="On-call Roster", state=STATE_PENDING)
        assert m.doc_id and m.current_version == 1
        assert m.state == STATE_PENDING
        assert store.exists(m.doc_id)
        # Registry synced.
        assert reg.get_source_type(m.doc_id) == SOURCE_REFERENCE
        assert reg.get_state(m.doc_id) == STATE_PENDING

        # Update with new content => new version.
        f2 = os.path.join(d, "roster2.xlsx")
        with open(f2, "wb") as f:
            f.write(b"v2-content-very-different")
        m2 = store.update(m.doc_id, f2, new_state=STATE_PUBLISHED)
        assert m2.current_version == 2
        assert m2.state == STATE_PUBLISHED
        assert reg.get_state(m.doc_id) == STATE_PUBLISHED

        # Same content => no-op.
        m3 = store.update(m.doc_id, f2)
        assert m3.current_version == 2

        # set_state only.
        store.set_state(m.doc_id, STATE_DEPRECATED)
        assert store.get(m.doc_id).state == STATE_DEPRECATED
        print("[ok] references: register -> update (version bump) -> set_state")


def main():
    tests = [
        test_state_transitions_allowed,
        test_state_transitions_illegal,
        test_permission_matrix,
        test_require_raises,
        test_resolve_role_from_manifest,
        test_review_queue_lifecycle,
        test_routing_declared_overrides_everything,
        test_routing_ext_heuristic,
        test_routing_manifest_and_dir,
        test_routing_llm_fallback,
        test_registry_governance_fields_and_compat,
        test_reference_store_register_and_update,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"{failed}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
