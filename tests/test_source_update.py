"""Smoke tests for the article-source update mechanism.

Covers the non-LLM logic that the refactor added:
  * Markdown parser sectioning with stable (heading,index) keys
  * SourceRegistry diff (added/changed/removed/unchanged)
  * whole-document short-circuit
  * backward-compatible insight schema load/save (legacy strings + new meta)
  * round-trip of rich metadata through save_insight_library/load_insight_library_meta

LLM-dependent paths (actual distillation, framework generation) are
intentionally not exercised here.
"""

import json
import os
import sys
import tempfile

# Allow running from repo root without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xknowledge.parsers.markdown import MarkdownParser
from xknowledge.core.source import (
    SourceRegistry,
    SectionDiff,
    hash_text,
    hash_file,
    section_key_for,
)
from xknowledge.core.utils import (
    load_insight_library,
    load_insight_library_meta,
    save_insight_library,
)


MD_DOC = """# Title

Intro paragraph here.

## First Section

Content of first section.

## Second Section

Content of second section.

### Subsection

Sub content.
"""


def _write(tmpdir, name, text):
    p = os.path.join(tmpdir, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def test_markdown_parser_sectioning():
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "doc.md", MD_DOC)
        segs = MarkdownParser().parse(path)

        # Preamble + 4 heading sections.
        assert len(segs) >= 3, f"expected >=3 segments, got {len(segs)}"

        headings = [s.metadata.get("heading") for s in segs]
        indices = [s.metadata.get("index") for s in segs]
        levels = [s.metadata.get("level") for s in segs]

        # `# Title` is a real heading at the very start, so the first
        # segment is the H1 title section (no preamble in this doc).
        assert segs[0].metadata["heading"] == "Title", f"first heading should be 'Title': {segs[0].metadata}"
        assert segs[0].metadata["level"] == 1
        # Headings present.
        assert "First Section" in headings
        assert "Second Section" in headings
        assert "Subsection" in headings
        # Levels captured, including a level-3.
        assert max(levels) >= 3, f"expected a level-3 heading, levels={levels}"
        # Indices unique.
        assert len(set(indices)) == len(indices), f"non-unique indices: {indices}"
        print("[ok] markdown parser sectioning:", headings)

    # Separate case: real preamble before the first heading.
    with tempfile.TemporaryDirectory() as d:
        with_preamble = "Some intro with no heading.\n\n# Title\n\nbody.\n"
        path = _write(d, "preamble.md", with_preamble)
        segs = MarkdownParser().parse(path)
        assert segs[0].metadata["heading"] == "", f"preamble heading not empty: {segs[0].metadata}"
        assert segs[0].segment_type == "preamble"
        assert segs[1].metadata["heading"] == "Title"
        print("[ok] markdown parser preamble handling")


def test_registry_diff_first_ingest():
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "doc.md", MD_DOC)
        reg = SourceRegistry(os.path.join(d, "reg.json"))
        segs = MarkdownParser().parse(path)
        keys = {section_key_for(s.metadata["heading"], s.metadata["index"]): s.content for s in segs}

        diff = reg.diff_sections(path, keys)
        assert isinstance(diff, SectionDiff)
        assert len(diff.added) == len(keys)
        assert not diff.changed and not diff.removed and not diff.unchanged
        assert diff.has_changes
        print("[ok] first ingest: all added ->", diff.summary())


def test_registry_unchanged_shortcircuit():
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "doc.md", MD_DOC)
        reg = SourceRegistry(os.path.join(d, "reg.json"))
        segs = MarkdownParser().parse(path)
        keys = {section_key_for(s.metadata["heading"], s.metadata["index"]): s.content for s in segs}

        sections_rec = {
            k: {"hash": hash_text(v), "insight_ids": [f"E{i}"]}
            for i, (k, v) in enumerate(keys.items())
        }
        reg.record_sections(path, hash_file(path), sections_rec)
        reg.save()

        # Reload from disk and verify persistence + short-circuit.
        reg2 = SourceRegistry(os.path.join(d, "reg.json"))
        assert reg2.doc_count == 1
        assert reg2.whole_doc_unchanged(path, hash_file(path))
        diff = reg2.diff_sections(path, keys)
        assert not diff.added and not diff.changed and not diff.removed
        assert len(diff.unchanged) == len(keys)
        print("[ok] unchanged short-circuit ->", diff.summary())


def test_registry_section_change_detection():
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "doc.md", MD_DOC)
        reg_path = os.path.join(d, "reg.json")
        reg = SourceRegistry(reg_path)
        segs = MarkdownParser().parse(path)
        keys = {section_key_for(s.metadata["heading"], s.metadata["index"]): s.content for s in segs}
        sections_rec = {k: {"hash": hash_text(v), "insight_ids": []} for k, v in keys.items()}
        reg.record_sections(path, hash_file(path), sections_rec)
        reg.save()

        # Edit ONE section's body (keep heading the same -> stable key, new hash).
        edited = MD_DOC.replace("Content of first section.", "Content of first section EDITED.")
        reg = SourceRegistry(reg_path)
        segs2 = MarkdownParser().parse(_write(d, "doc2.md", edited))
        keys2 = {section_key_for(s.metadata["heading"], s.metadata["index"]): s.content for s in segs2}
        diff = reg.diff_sections(path, keys2)
        assert len(diff.changed) == 1, f"expected 1 changed, got {diff.changed}"
        assert not diff.added and not diff.removed
        # The changed key must be the First Section one.
        first_keys = [k for k in keys if "First Section" in k or k.endswith("#section-0")]
        # Confirm at least one changed key corresponds to first section by content.
        changed_content = keys2[diff.changed[0]]
        assert "EDITED" in changed_content
        print("[ok] section change detected ->", diff.summary(), "changed=", diff.changed[0])


def test_registry_section_removal():
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "doc.md", MD_DOC)
        reg_path = os.path.join(d, "reg.json")
        reg = SourceRegistry(reg_path)
        segs = MarkdownParser().parse(path)
        keys = {section_key_for(s.metadata["heading"], s.metadata["index"]): s.content for s in segs}
        sections_rec = {k: {"hash": hash_text(v), "insight_ids": []} for k, v in keys.items()}
        reg.record_sections(path, hash_file(path), sections_rec)
        reg.save()

        # Remove the Second Section + its subsection entirely.
        removed_block = "## Second Section\n\nContent of second section.\n\n### Subsection\n\nSub content.\n"
        trimmed = MD_DOC.replace(removed_block, "")
        reg = SourceRegistry(reg_path)
        segs2 = MarkdownParser().parse(_write(d, "doc3.md", trimmed))
        keys2 = {section_key_for(s.metadata["heading"], s.metadata["index"]): s.content for s in segs2}
        diff = reg.diff_sections(path, keys2)
        assert len(diff.removed) >= 2, f"expected >=2 removed, got {diff.removed}"
        print("[ok] removal detected ->", diff.summary())


def test_insight_schema_backward_compat():
    with tempfile.TemporaryDirectory() as d:
        legacy = os.path.join(d, "legacy.json")
        # Legacy schema: plain strings.
        with open(legacy, "w", encoding="utf-8") as f:
            json.dump({"insights": {"E0": "old text", "E1": "another"}}, f)

        # Text view.
        texts = load_insight_library(legacy)
        assert texts == {"E0": "old text", "E1": "another"}, texts
        # Meta view wraps legacy strings into {"text": ...}.
        meta = load_insight_library_meta(legacy)
        assert meta["E0"] == {"text": "old text"}, meta["E0"]
        assert meta["E1"] == {"text": "another"}, meta["E1"]
        print("[ok] legacy schema backward-compat:", list(meta))


def test_insight_schema_roundtrip_with_meta():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "kb.json")
        texts = {"E0": "tip one", "E1": "tip two"}
        meta = {
            "E0": {
                "text": "tip one",
                "source_doc": "/abs/doc.md",
                "section": "1#First",
                "added_at": "2026-01-01T00:00:00Z",
                "content_hash": "abc123",
            },
            "E1": {"text": "tip two", "source_doc": "/abs/doc.md", "section": "2#Second"},
        }
        save_insight_library(p, texts, meta=meta)

        # Text view preserves order/values.
        loaded_texts = load_insight_library(p)
        assert loaded_texts == texts, loaded_texts
        # Meta view preserves provenance.
        loaded_meta = load_insight_library_meta(p)
        assert loaded_meta["E0"]["source_doc"] == "/abs/doc.md"
        assert loaded_meta["E0"]["content_hash"] == "abc123"
        assert loaded_meta["E1"]["section"] == "2#Second"
        # Text always wins over stale text in meta.
        assert loaded_meta["E0"]["text"] == "tip one"
        print("[ok] rich metadata round-trip:", len(loaded_meta), "entries")


def test_insight_save_without_meta_is_legacy():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "kb.json")
        save_insight_library(p, {"E0": "x"}, meta=None)
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Without meta, entries are plain strings (legacy-compatible).
        assert raw["insights"]["E0"] == "x", raw
        print("[ok] save without meta writes legacy strings")


def main():
    tests = [
        test_markdown_parser_sectioning,
        test_registry_diff_first_ingest,
        test_registry_unchanged_shortcircuit,
        test_registry_section_change_detection,
        test_registry_section_removal,
        test_insight_schema_backward_compat,
        test_insight_schema_roundtrip_with_meta,
        test_insight_save_without_meta_is_legacy,
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
        print(f"{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
