"""Document source registry for incremental knowledge distillation.

Tracks which documents have been distilled into the knowledge base and,
crucially, the per-section provenance of every insight. This enables:

  * **Fast short-circuit** - re-ingesting an unchanged document is a
    single hash comparison, no LLM calls.
  * **Section-level diff** - when a document is edited, only the
    added/changed sections are re-distilled; unchanged sections keep
    their insights (saves LLM cost and avoids churn).
  * **Insight cleanup** - when a section is removed, the insights it
    contributed can be located and dropped.

Persisted as ``<kb_dir>/sources/registry.json``::

    {
      "<doc_path>": {
        "content_hash": "<sha256 of whole file>",
        "ingested_at": "<iso8601>",
        "sections": {
          "<section_key>": {"hash": "<sha256 of section content>",
                            "insight_ids": ["E3", "E7"]}
        }
      },
      ...
    }

The registry is the foundation for the upcoming "full knowledge base
management" layer (CRUD, versioning, review); for now it only powers
``QAKnowledge.learn_from_document``'s incremental update path.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Canonical source types (mirror routing.classifier for convenience).
SOURCE_DISTILLABLE = "distillable"
SOURCE_REFERENCE = "reference"

# Governance states (mirror governance.states.State values).
STATE_DRAFT = "draft"
STATE_PENDING = "pending"
STATE_PUBLISHED = "published"
STATE_DEPRECATED = "deprecated"

# States that default retrieval serves to viewers.
SERVED_STATES = frozenset({STATE_PUBLISHED})


# Type alias: section_key -> {"hash": str, "insight_ids": [str, ...]}
SectionMap = Dict[str, Dict[str, Any]]


@dataclass
class SectionDiff:
    """Result of comparing a document's new sections against the registry."""

    added: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.removed)

    def summary(self) -> str:
        return (
            f"+{len(self.added)} added, ~{len(self.changed)} changed, "
            f"-{len(self.removed)} removed, ={len(self.unchanged)} unchanged"
        )


def hash_text(text: str) -> str:
    """SHA-256 of a text blob (whitespace-normalized for stability)."""
    norm = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def hash_file(path: str) -> str:
    """SHA-256 of a file's raw bytes for whole-document change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def section_key_for(heading: Optional[str], index: int) -> str:
    """Build a stable section key from heading text and ordinal index.

    The index disambiguates sections that share a heading (e.g. multiple
    "## Notes") and keeps keys stable when content under a heading changes.
    """
    h = (heading or "").strip().replace("\n", " ") or f"section-{index}"
    return f"{index}#{h[:80]}"


class SourceRegistry:
    """Persistent registry of distilled document sources.

    The registry is loaded lazily and flushed explicitly via ``save()``.
    Callers mutate state through ``record_sections`` / ``remove_doc`` /
    ``rebind_insights`` and then persist once at the end of an ingest.
    """

    def __init__(self, path: str):
        """Initialize the registry, pointing at a JSON file on disk.

        Args:
            path: Path to ``registry.json``. The parent directory must
                exist (created by the caller, e.g. QAKnowledge).
        """
        self.path = path
        # doc_path -> {"content_hash", "ingested_at", "sections": SectionMap}
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ---- persistence ----

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._docs = {
                    k: self._normalize(v)
                    for k, v in data.items() if isinstance(v, dict)
                }
        except Exception:
            self._docs = {}

    @staticmethod
    def _normalize(rec: Dict[str, Any]) -> Dict[str, Any]:
        """Backfill new governance fields on legacy registry records.

        Old records have no source_type/state/origin; treat them as
        published distillable docs (the pre-governance behavior).
        """
        rec.setdefault("source_type", SOURCE_DISTILLABLE)
        rec.setdefault("state", STATE_PUBLISHED)
        rec.setdefault("origin", SOURCE_DISTILLABLE)
        rec.setdefault("sections", {})
        return rec

    def save(self):
        """Flush the registry to disk."""
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, ensure_ascii=False, indent=2)

    # ---- queries ----

    def get_doc(self, doc_path: str) -> Optional[Dict[str, Any]]:
        """Return the stored record for a document, or None if unseen."""
        return self._docs.get(doc_path)

    def has_doc(self, doc_path: str) -> bool:
        return doc_path in self._docs

    def section_insight_ids(self, doc_path: str, section_key: str) -> List[str]:
        """Return insight IDs previously produced by a section."""
        rec = self._docs.get(doc_path)
        if not rec:
            return []
        return list(rec.get("sections", {}).get(section_key, {}).get("insight_ids", []))

    # ---- governance field accessors / setters ----

    def get_source_type(self, doc_path: str) -> str:
        rec = self._docs.get(doc_path)
        return rec.get("source_type", SOURCE_DISTILLABLE) if rec else SOURCE_DISTILLABLE

    def get_state(self, doc_path: str) -> str:
        rec = self._docs.get(doc_path)
        return rec.get("state", STATE_PUBLISHED) if rec else STATE_PUBLISHED

    def get_origin(self, doc_path: str) -> str:
        rec = self._docs.get(doc_path)
        return rec.get("origin", SOURCE_DISTILLABLE) if rec else SOURCE_DISTILLABLE

    def set_source_type(self, doc_path: str, source_type: str):
        """Set a doc's source_type (distillable | reference)."""
        self._ensure_doc(doc_path)["source_type"] = source_type

    def set_state(self, doc_path: str, state: str):
        """Set a doc's governance state."""
        self._ensure_doc(doc_path)["state"] = state

    def set_origin(self, doc_path: str, origin: str):
        """Set a doc's origin (provenance label)."""
        self._ensure_doc(doc_path)["origin"] = origin

    def _ensure_doc(self, doc_path: str) -> Dict[str, Any]:
        rec = self._docs.get(doc_path)
        if rec is None:
            rec = {
                "content_hash": "",
                "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sections": {},
                "source_type": SOURCE_DISTILLABLE,
                "state": STATE_PUBLISHED,
                "origin": SOURCE_DISTILLABLE,
            }
            self._docs[doc_path] = rec
        else:
            self._normalize(rec)
        return rec

    def list_docs(
        self,
        source_type: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return registered documents, optionally filtered.

        Args:
            source_type: Filter to one source type (e.g. ``"reference"``).
            state: Filter to one governance state (e.g. ``"published"``).
        """
        out = []
        for path, rec in self._docs.items():
            if source_type and rec.get("source_type") != source_type:
                continue
            if state and rec.get("state") != state:
                continue
            entry = {"doc_path": path}
            entry.update(rec)
            out.append(entry)
        return out

    def doc_insight_ids(self, doc_path: str) -> List[str]:
        """Return all insight IDs associated with any section of a doc."""
        rec = self._docs.get(doc_path)
        if not rec:
            return []
        ids = []
        for sec in rec.get("sections", {}).values():
            ids.extend(sec.get("insight_ids", []))
        return ids

    # ---- diff ----

    def diff_sections(
        self,
        doc_path: str,
        new_sections: Dict[str, str],
    ) -> SectionDiff:
        """Compare new section hashes against what is registered.

        Args:
            doc_path: Document whose sections are being re-evaluated.
            new_sections: Mapping of section_key -> section content text.

        Returns:
            A SectionDiff partitioning section keys into added / changed /
            removed / unchanged.
        """
        rec = self.get_doc(doc_path)
        old: SectionMap = rec.get("sections", {}) if rec else {}
        new_hashes = {k: hash_text(v) for k, v in new_sections.items()}

        diff = SectionDiff()
        for key, content in new_sections.items():
            new_hash = new_hashes[key]
            if key not in old:
                diff.added.append(key)
            elif old[key].get("hash") != new_hash:
                diff.changed.append(key)
            else:
                diff.unchanged.append(key)
        for key in old:
            if key not in new_sections:
                diff.removed.append(key)
        return diff

    def whole_doc_unchanged(self, doc_path: str, content_hash: str) -> bool:
        """True iff this exact document (by whole-file hash) was ingested."""
        rec = self.get_doc(doc_path)
        return bool(rec and rec.get("content_hash") == content_hash)

    # ---- mutations ----

    def record_sections(
        self,
        doc_path: str,
        content_hash: str,
        sections: SectionMap,
        source_type: Optional[str] = None,
        state: Optional[str] = None,
        origin: Optional[str] = None,
    ):
        """Create or overwrite a document's registration.

        Args:
            doc_path: Source document path (acts as the identity key).
            content_hash: Whole-document hash for short-circuit checks.
            sections: section_key -> {"hash": str, "insight_ids": [ids]}.
                ``insight_ids`` may be empty for newly distilled sections.
            source_type: Optional override for the doc's source type.
            state: Optional override for the doc's governance state.
            origin: Optional override for the doc's origin label.
        """
        existing = self._docs.get(doc_path, {})
        rec = {
            "content_hash": content_hash,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sections": dict(sections),
            # Preserve governance fields, then apply explicit overrides.
            "source_type": source_type or existing.get("source_type", SOURCE_DISTILLABLE),
            "state": state or existing.get("state", STATE_PUBLISHED),
            "origin": origin or existing.get("origin", SOURCE_DISTILLABLE),
        }
        self._docs[doc_path] = rec

    def rebind_insights(
        self,
        doc_path: str,
        section_insight_ids: Dict[str, List[str]],
    ):
        """Attach distilled insight IDs back to their source sections.

        ``section_insight_ids`` maps section_key -> list of insight IDs
        produced from that section in the latest distillation pass. Only
        the sections present in the map are touched; others are left as-is.
        """
        rec = self._docs.setdefault(
            doc_path,
            {
                "content_hash": "",
                "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sections": {},
            },
        )
        secs: SectionMap = rec.setdefault("sections", {})
        for section_key, ids in section_insight_ids.items():
            entry = secs.setdefault(section_key, {"hash": "", "insight_ids": []})
            entry["insight_ids"] = list(ids)

    def clear_section_insights(self, doc_path: str, section_keys: List[str]):
        """Drop insight bindings for sections (e.g. removed/changed sections)."""
        rec = self._docs.get(doc_path)
        if not rec:
            return
        secs: SectionMap = rec.get("sections", {})
        for key in section_keys:
            if key in secs:
                secs[key]["insight_ids"] = []

    def remove_doc(self, doc_path: str):
        """Unregister a document entirely (does not touch insights.json)."""
        self._docs.pop(doc_path, None)

    # ---- diagnostics ----

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    @property
    def status(self) -> Dict[str, Any]:
        total_sections = sum(
            len(rec.get("sections", {})) for rec in self._docs.values()
        )
        total_insights = sum(
            len(sec.get("insight_ids", []))
            for rec in self._docs.values()
            for sec in rec.get("sections", {}).values()
        )
        return {
            "doc_count": self.doc_count,
            "section_count": total_sections,
            "bound_insight_refs": total_insights,
            "registry_path": self.path,
        }
